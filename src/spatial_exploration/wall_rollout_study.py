"""Matched greedy and capped rollout policies with the same wall ridge mapper."""

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .barrier_deepkriging_study import dump
from .barrier_gp import next_measurement
from .barrier_study import BUDGETS, DOORS, NOISE, REGIMES, STARTS, render_example
from .baseline import GPBelief, write_csv
from .rollout import plan_rollout
from .wall_closed_loop_study import TEST_DOORS, load_settings, metrics, ridge_models, scenes
from .wall_topology_study import topology_design


PLANNERS = ("greedy", "rollout")
CALIBRATION_SEED = 1310000
TEST_SEED = 1410000


@dataclass(frozen=True)
class Route:
    sites: tuple
    distances: tuple
    final_variance: float
    planning_ms: float
    candidates: int
    simulated_measurements: int


def plan_route(design, ridge, start, cap, distance_cap, planner):
    """Replan after each covariance update; routes are independent of readings."""
    if planner not in PLANNERS:
        raise ValueError(f"Unknown planner {planner}")
    layout, allowed = design.layout, ~design.heldout
    belief = GPBelief(ridge.prior_covariance, ridge.noise_std)
    sites, travel = [start], [0.0]
    visited = np.zeros_like(allowed)
    visited[start] = True
    belief.observe(start, 0.0)
    candidates = simulations = 0
    elapsed = 0.0
    while len(sites) < cap:
        current, remaining = sites[-1], distance_cap - travel[-1]
        if not np.any(allowed & ~visited & (layout.distances[current] <= remaining + 1e-12)):
            break
        before = time.perf_counter()
        if planner == "rollout":
            plan = plan_rollout(belief.covariance, layout.distances, current, visited, allowed,
                                remaining, belief.noise_variance, remaining_observations=cap - len(sites))
            action = plan.first
            candidates += plan.candidates_evaluated
            simulations += plan.simulated_measurements
        else:
            action = next_measurement(belief, layout, sites, allowed, remaining, cap - len(sites))
        elapsed += time.perf_counter() - before
        sites.append(action)
        travel.append(travel[-1] + float(layout.distances[current, action]))
        visited[action] = True
        belief.observe(action, 0.0)
    return Route(tuple(sites), tuple(travel), float(belief.covariance.diagonal().mean()),
                 elapsed * 1000, candidates, simulations)


def route_table(design, settings):
    models = {cap: ridge_models(design, settings[cap])["wall_graph"] for cap, _ in BUDGETS}
    routes = {(start, cap, method): plan_route(design, models[cap], start, cap, dist, method)
              for cap, dist in BUDGETS for start in STARTS for method in PLANNERS}
    for start in STARTS:
        for cap, _ in BUDGETS:
            if routes[start, cap, "rollout"].final_variance > routes[start, cap, "greedy"].final_variance + 1e-10:
                raise RuntimeError("Rollout regressed on its own terminal variance objective")
    return models, routes


def calibrate(settings, fields):
    ratios = {(cap, method): [] for cap, _ in BUDGETS for method in PLANNERS}
    for door in DOORS:
        design = topology_design(door)
        models, routes = route_table(design, settings)
        held = design.heldout
        for meta, truth, sensor in scenes(design, fields, CALIBRATION_SEED):
            for (cap, method), values in ratios.items():
                sites = routes[meta["start"], cap, method].sites
                posterior = models[cap].condition(sites, sensor[list(sites)])
                values.extend(np.abs(truth[held] - posterior.mean[held]) /
                              np.maximum(posterior.std[held], 1e-8))
    return {key: float(np.quantile(values, .95) / 1.96) for key, values in ratios.items()}


def summarize(rows, route_rows):
    rng = np.random.default_rng(141001)
    result = []
    metrics_to_average = ("rmse", "near_wall_rmse", "distance", "measurements",
                          "modeled_grid_variance", "coverage", "raw_coverage", "width", "interval_score")
    for regime in REGIMES:
        for cap, _ in BUDGETS:
            groups = {p: {} for p in PLANNERS}
            for row in rows:
                if (row["regime"], row["cap"]) == (regime, cap):
                    key = row["door"], row["seed"]
                    group = groups[row["planner"]]
                    if key in group:
                        raise ValueError("Duplicate test scene")
                    group[key] = row
            greedy, rollout = (groups[p] for p in PLANNERS)
            keys = sorted(greedy)
            if set(keys) != set(rollout):
                raise ValueError("Test scenes do not match")
            strata = {}
            for i, k in enumerate(keys):
                g, r = greedy[k], rollout[k]
                if (g["start"], g["kappa"]) != (r["start"], r["kappa"]):
                    raise ValueError("Start or field scale differs within pair")
                strata.setdefault((g["door"], g["start"], g["kappa"]), []).append(i)

            def paired(metric):
                delta = np.array([rollout[k][metric] - greedy[k][metric] for k in keys])
                boot = np.concatenate([rng.choice(delta[idx], (6000, len(idx)), replace=True)
                                       for idx in strata.values()], axis=1).mean(axis=1)
                return dict(difference=float(delta.mean()), interval=np.quantile(boot, [.025, .975]).tolist(),
                            rollout_lower_fields=int(np.sum(delta < -1e-12)))

            configurations = {(r["door"], r["start"], r["planner"]): r for r in route_rows if r["cap"] == cap}
            pairs = [(r, configurations[d, s, "rollout"]) for (d, s, p), r in configurations.items()
                     if p == "greedy"]
            result.append(dict(regime=regime, cap=cap, fields=len(keys), configurations=len(pairs),
                changed_routes=sum(g["sites"] != r["sites"] for g, r in pairs),
                changed_first_moves=sum(g["sites"][1:2] != r["sites"][1:2] for g, r in pairs),
                means={p: {m: float(np.mean([row[m] for row in groups[p].values()]))
                           for m in metrics_to_average} for p in PLANNERS},
                mean_planning_ms={p: float(np.mean([r["planning_ms"] for r in configurations.values()
                                                  if r["planner"] == p])) for p in PLANNERS},
                crosses_rooms_configurations={p: sum(r["crosses_rooms"] for r in configurations.values()
                                                    if r["planner"] == p) for p in PLANNERS},
                observation_cap_configurations={p: sum(len(r["sites"]) == cap for r in configurations.values()
                                                       if r["planner"] == p) for p in PLANNERS},
                modeled_variance_lower_configurations=sum(r["final_variance"] < g["final_variance"] - 1e-12
                                                          for g, r in pairs),
                paired_rmse=paired("rmse"), paired_near_wall_rmse=paired("near_wall_rmse")))
    return result


def render_comparison(output, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.3), sharey=True, layout="constrained")
    labels = [f"{'Barrier' if r['regime'] == 'barrier' else 'Transparent wall'} · {r['cap']} readings"
              for r in summary]
    for axis, metric, label in zip(axes, ("rmse", "distance", "modeled_grid_variance"),
                                  ("Held-out RMSE", "Actual walking distance", "Modeled final variance")):
        axis.axvline(0, color="0.5", linewidth=1)
        for i, row in enumerate(summary):
            g, r = (row["means"][p][metric] for p in PLANNERS)
            color = "#207282" if row["regime"] == "barrier" else "#ad5339"
            if metric == "rmse":
                axis.plot(row["paired_rmse"]["interval"], [i, i], color=color, linewidth=2)
            axis.plot(r - g, i, "o", color=color)
        axis.set_yticks(range(len(labels)), labels)
        axis.set_xlabel(label)
        axis.grid(axis="x", alpha=.22)
    axes[0].invert_yaxis()
    fig.suptitle("Rollout minus greedy · same wall-aware ridge and budget caps")
    fig.savefig(output / "comparison.png", dpi=175)
    plt.close(fig)


def run(reference, output, fields=128, calibration_fields=32):
    if not all(8 <= n < 1000 and n % 8 == 0 for n in (fields, calibration_fields)):
        raise ValueError("Use multiples of eight between 8 and 999")
    settings, digest = load_settings(reference)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Use a fresh output directory")
    protocol = dict(reference_sha256=digest, settings=settings, budgets=BUDGETS, starts=STARTS,
                    test_doors=TEST_DOORS, test_fields_per_layout=fields, test_seed=TEST_SEED,
                    calibration_doors=DOORS, calibration_fields_per_layout=calibration_fields,
                    calibration_seed=CALIBRATION_SEED, regimes=REGIMES, noise_std=NOISE,
                    planner="Every feasible first acquisition; greedy completion to either cap; replan each step",
                    baseline="Greedy whole-grid variance reduction per legal walking distance",
                    primary="Mean per-field final RMSE at the same 28 reserved sites and actual walking distance",
                    planning_inputs="Fixed mapper covariance, acquired indices, layout and remaining budgets only",
                    calibration="Separate layouts/fields; scales for reporting intervals only")
    dump(output / "protocol.json", protocol)
    rows, acquisitions, route_rows, archive, example = [], [], [], {}, None
    with threadpool_limits(limits=1):
        scales = calibrate(settings, calibration_fields)
        dump(output / "frozen.json", {**protocol, "std_scales": {f"{p}-n{n}": v for (n, p), v in scales.items()}})
        for door in TEST_DOORS:
            design = topology_design(door)
            models, routes = route_table(design, settings)
            for (start, cap, planner), route in routes.items():
                route_rows.append(dict(door=door, start=start, cap=cap, planner=planner,
                    crosses_rooms=bool(np.any(design.layout.room[list(route.sites)] != design.layout.room[start])),
                    **asdict(route)))
            records = list(scenes(design, fields, TEST_SEED))
            archive.update({f"d{door}_truth": np.asarray([truth for _, truth, _ in records]),
                            f"d{door}_heldout": design.heldout,
                            f"d{door}_seeds": np.array([meta["seed"] for meta, _, _ in records]),
                            f"d{door}_regimes": np.array([meta["regime"] for meta, _, _ in records])})
            for meta, truth, sensor in records:
                for cap, distance in BUDGETS:
                    display = []
                    for planner in PLANNERS:
                        route = routes[meta["start"], cap, planner]
                        # The route depends only on covariance; replay its acquisitions one site at a time.
                        readings = [float(sensor[site]) for site in route.sites]
                        posterior = models[cap].condition(route.sites, readings)
                        scored = metrics(truth, posterior, design, scales[cap, planner])
                        if abs(scored["modeled_grid_variance"] - route.final_variance) > 1e-12:
                            raise RuntimeError("Scoring and planning used different posterior covariances")
                        rows.append(dict(**meta, planner=planner, cap=cap, distance_cap=distance,
                                         distance=route.distances[-1], measurements=len(route.sites), **scored))
                        acquisitions.extend(dict(**meta, planner=planner, cap=cap, step=i, site=site,
                                                 distance=walked, value=value)
                                            for i, (site, walked, value) in enumerate(
                                                zip(route.sites, route.distances, readings)))
                        display.append((planner.capitalize(), posterior.mean, list(route.sites)))
                    if (example is None and cap == 16 and meta["regime"] == "barrier"
                            and display[0][2] != display[1][2]):
                        example = (design.layout, truth, design.heldout, display, meta["field"])
    summary = summarize(rows, route_rows)
    dump(output / "routes.json", route_rows)
    dump(output / "summary.json", summary)
    write_csv(output / "metrics.csv", rows)
    write_csv(output / "acquisitions.csv", acquisitions)
    np.savez_compressed(output / "evaluation-only.npz", **archive)
    render_comparison(output, summary)
    if example is not None:
        render_example(output, *example)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("outputs/wall-topology-study/frozen.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/wall-rollout-study"))
    parser.add_argument("--fields-per-layout", type=int, default=128)
    args = parser.parse_args()
    for row in run(args.reference, args.output, args.fields_per_layout):
        print(row["regime"], row["cap"], {p: {k: round(v[k], 6) for k in
                                             ("rmse", "distance", "measurements", "modeled_grid_variance")}
                                         for p, v in row["means"].items()})


if __name__ == "__main__":
    main()
