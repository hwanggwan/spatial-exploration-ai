"""Compare terminal-only and distance-aware rollout on the same refined fields."""

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from audit_residential_grid_resolution import expand_walk, holdouts, predictions, refine_geometry, refined_mapper, route_resources
from audit_residential_refined_rollout import ARM_ROLLOUT, SCENARIOS, field_batches, field_key, full_budget_route, remaining_variance
from audit_residential_holdout_distance import CAP, FAMILY
from audit_residential_routes import LayoutCase, STARTS, floor_plan
from spatial_exploration.baseline import GPBelief, write_csv
from spatial_exploration.sensing_world_study import NOISE


WEIGHTS = {"terminal": 0.0, "early": 0.5}
QUARTER_WEIGHT = 0.25
BOOT_SEED = 314787


def plan_routes(variant, layout, anchors, held, prior, previous, edge_cap, weights, cached_routes):
    allowed = ~np.isin(np.arange(len(layout.room)), held)
    distance_cap = edge_cap / (layout.size - 1)
    steps = np.arange(edge_cap + 1)
    routes = {}
    for start in STARTS:
        origin = int(anchors[start])
        for arm, weight in weights.items():
            cached = cached_routes.get((variant, start, arm))
            if cached is None:
                before = perf_counter()
                sites, terminal, branches = full_budget_route(layout, allowed, prior, origin, distance_cap,
                                                              early_weight=weight)
                plan_ms = (perf_counter() - before) * 1000
            else:
                sites, terminal = cached["sites"], cached["modeled_variance"][-1]
                plan_ms, branches = cached["plan_ms"], cached["branches"]
            walk = expand_walk(layout, sites)
            if cached is not None:
                np.testing.assert_array_equal(walk, cached["walk"])
            resources = route_resources(layout, walk, sites, anchors, edge_cap)
            if sites[0] != origin or not allowed[sites].all():
                raise ValueError("Invalid start or held-out visit")
            costs = layout.distances[sites[:-1], sites[1:]]
            arrivals = np.r_[0, np.rint(costs * (layout.size - 1)).astype(int).cumsum()]
            if arrivals[-1] != resources["edges"]:
                raise ValueError("Walking distance mismatch")
            if arm == "terminal":
                old = previous[variant, start]
                np.testing.assert_array_equal(sites, old["sites"])
                np.testing.assert_array_equal(walk, old["walk"])
            prefix_ticks = np.searchsorted(arrivals, steps, side="right")
            fixed = GPBelief(prior, NOISE)
            variances = []
            for site in sites:
                fixed.observe(site, 0)
                variances.append(float(np.diag(fixed.covariance).mean()))
            trajectory = np.asarray(variances)[prefix_ticks-1]
            np.testing.assert_allclose(terminal, trajectory[-1], rtol=0, atol=1e-12)
            first = GPBelief(prior, NOISE)
            first.observe(origin, 0)
            final, integral = remaining_variance(first.covariance, layout.distances, origin, sites[1:],
                                                  first.noise_variance, distance_cap, distance_cap)
            np.testing.assert_allclose((final, integral), (trajectory[-1], trajectory[:-1].mean()),
                                       rtol=0, atol=1e-12)
            late_count = int(np.count_nonzero(arrivals > .75 * arrivals[-1]))
            budget_late_count = int(np.count_nonzero(arrivals > .75 * edge_cap))
            routes[start, arm] = dict(variant=variant, start=start, arm=arm, sites=sites, walk=walk.tolist(),
                                     arrivals=arrivals.tolist(), counts_at_distance=prefix_ticks.tolist(),
                                     plan_ms=plan_ms, branches=branches, **resources,
                                     late_count=late_count, late_fraction=late_count/(len(sites)-1),
                                     budget_late_fraction=budget_late_count/(len(sites)-1),
                                     modeled_variance=trajectory.tolist())
        print(f"Planned layout {variant}, start {start}", flush=True)
    return routes


def evaluate(variant, model, previous_routes, previous_fields, edge_cap, weights, cached_routes, cached_fields):
    coarse = floor_plan(LayoutCase(FAMILY, variant))
    layout, anchors = refine_geometry(coarse)
    held, _ = holdouts(coarse, layout, anchors, variant)
    profiles, kernels, prior, curve = refined_mapper(coarse, layout, anchors, model)
    log_prior = np.log(prior)
    routes = plan_routes(variant, layout, anchors, held, np.einsum("m,mij->ij", prior, kernels),
                         previous_routes, edge_cap, weights, cached_routes)
    rows, field_curves = [], []
    for regime, start, profile, scenario, truth, noisy in field_batches(variant, layout, profiles, curve):
        held_truth = truth[:, held]
        metrics, distance_curves = {}, []
        for arm in weights:
            route = routes[start, arm]
            sites = np.asarray(route["sites"])
            errors = []
            for n in range(1, len(sites)+1):
                selected = sites[:n]
                mean, sd = predictions(kernels, selected, noisy[:, selected], held, log_prior)
                residual = mean - held_truth
                errors.append(np.sqrt(np.mean(residual**2, axis=1)))
            distance_curves.append(np.asarray(errors).T[:, np.asarray(route["counts_at_distance"])-1])
            metrics[f"{arm}_coverage"] = np.mean(abs(residual) <= 1.96 * sd, axis=1)
            metrics[f"{arm}_width"] = np.mean(3.92 * sd, axis=1)
        distances = np.stack(distance_curves, axis=1)
        first = distances[:, :, 0]
        np.testing.assert_allclose(first, np.broadcast_to(first[:, :1], first.shape), rtol=0, atol=1e-12)
        for draw in range(len(truth)):
            row = dict(variant=variant, regime=regime, start=start, profile=profile, draw=draw, scenario=scenario)
            expected = previous_fields[field_key(row)]
            np.testing.assert_allclose(distances[draw, 0, -1], float(expected[f"{ARM_ROLLOUT}_all"]),
                                       rtol=0, atol=1e-10)
            if cached_fields:
                previous = cached_fields[field_key(row)]
                for i, arm in enumerate(WEIGHTS):
                    series = distances[draw, i]
                    np.testing.assert_allclose((series[:-1].mean(), series[-1]),
                        (float(previous[f"{arm}_distance_average"]), float(previous[f"{arm}_final"])),
                        rtol=0, atol=1e-10)
            rows.append(dict(**row, **{name: data[draw] for name, data in metrics.items()}))
        field_curves.append(distances)
    return rows, np.concatenate(field_curves), list(routes.values())


def summarize(rows, curves, routes, edge_cap, weights):
    rng = np.random.default_rng(BOOT_SEED)
    steps = np.arange(edge_cap + 1)
    checkpoints = [0, edge_cap//4, edge_cap//2, edge_cap*3//4, edge_cap]
    has_quarter = "quarter" in weights
    summary, curve_rows = {}, []
    for scenario in SCENARIOS:
        ids = [i for i, row in enumerate(rows) if row["scenario"] == scenario]
        values = curves[ids]
        mean = values.mean(axis=0)
        strata = {}
        for index, i in enumerate(ids):
            row = rows[i]
            key = tuple(row[k] for k in ("variant", "regime", "start", "profile"))
            strata.setdefault(key, []).append(index)
        bootstrap = np.concatenate([rng.choice(indices, (3000, len(indices))) for indices in strata.values()], axis=1)
        paired = values[:, 1] - values[:, 0]

        def contrast(deltas):
            return dict(delta=float(deltas.mean()),
                        ci95=np.quantile(deltas[bootstrap].mean(axis=1), [.025, .975]).tolist())

        comparisons = {}
        for step in steps:
            diff = contrast(paired[:, step])
            row = dict(scenario=scenario, distance=int(step), terminal=float(mean[0, step]), early=float(mean[1, step]),
                       delta=diff["delta"], ci95_low=diff["ci95"][0], ci95_high=diff["ci95"][1])
            if has_quarter:
                row["quarter"] = float(mean[2, step])
                for i, arm in enumerate(WEIGHTS):
                    difference = contrast(values[:, 2, step] - values[:, i, step])
                    key = f"quarter_minus_{arm}"
                    row.update({key: difference["delta"], f"{key}_ci95_low": difference["ci95"][0],
                                f"{key}_ci95_high": difference["ci95"][1]})
            curve_rows.append(row)
            if step in checkpoints:
                comparisons[int(step)] = row
        within_route = values[:, :, :-1].mean(axis=2)
        summary[scenario] = dict(fields=len(ids), checkpoints=comparisons,
                                 distance_average=dict(zip(weights, within_route.mean(axis=0).tolist())),
                                 distance_average_contrast=contrast(within_route[:, 1] - within_route[:, 0]),
                                 coverage={a: float(np.mean([rows[i][f"{a}_coverage"] for i in ids])) for a in weights},
                                 width={a: float(np.mean([rows[i][f"{a}_width"] for i in ids])) for a in weights})
        if has_quarter:
            summary[scenario]["quarter_distance_average_contrasts"] = {
                arm: contrast(within_route[:, 2] - within_route[:, i]) for i, arm in enumerate(WEIGHTS)
            }
    resources = {}
    counts_by_distance = {}
    for arm in weights:
        group = [route for route in routes if route["arm"] == arm]
        keys = ("edges", "readings", "late_count", "late_fraction", "budget_late_fraction", "plan_ms")
        resources[arm] = {k: dict(mean=float(np.mean([route[k] for route in group])),
                                  range=[min(route[k] for route in group), max(route[k] for route in group)]) for k in keys}
        resources[arm]["modeled_variance_average"] = float(np.mean([np.mean(r["modeled_variance"][:-1]) for r in group]))
        resources[arm]["modeled_variance_final"] = float(np.mean([r["modeled_variance"][-1] for r in group]))
        counts_by_distance[arm] = np.mean([r["counts_at_distance"] for r in group], axis=0).tolist()
    for row in curve_rows:
        row.update({f"{arm}_readings": counts_by_distance[arm][row["distance"]] for arm in weights})
    return dict(scenarios=summary, routes=resources), curve_rows


def render(path, curve_rows, weights):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"terminal": "#7774a4", "early": "#09776b", "quarter": "#b95b2b"}
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4), facecolor="#fbfbf7", sharey=True, layout="constrained")
    for ax, scenario in zip(axes, SCENARIOS):
        rows = [row for row in curve_rows if row["scenario"] == scenario]
        x = [row["distance"] for row in rows]
        for arm in sorted(weights, key=weights.get):
            ax.step(x, [row[arm] for row in rows], where="post", label=f"{weights[arm]:.0%} distance weight",
                    color=colors[arm], linewidth=2)
        ax.axvspan(39, 52, color="#e5ebea", alpha=.75, zorder=-1)
        ax.set(title=scenario.capitalize() + " range", xlabel="Distance walked (grid steps)", xticks=[0, 13, 26, 39, 52])
        ax.grid(axis="y", color="#d7dedd")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Hold-out RMSE")
    axes[2].legend(loc="upper right", frameon=False, fontsize=8.5)
    fig.suptitle("Map accuracy over a shared 52-step and 24-reading budget", fontsize=15)
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("outputs/sensing-world-residential-refined-rollout"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-quarter", action="store_true")
    parser.add_argument("--reference-run", type=Path, default=Path("outputs/sensing-world-residential-early-rollout"))
    args = parser.parse_args()
    weights = WEIGHTS | {"quarter": QUARTER_WEIGHT} if args.include_quarter else WEIGHTS
    if args.output is None:
        suffix = "quarter-rollout" if args.include_quarter else "early-rollout"
        args.output = Path(f"outputs/sensing-world-residential-{suffix}")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use an empty output directory.")
    protocol = json.loads((args.source / "protocol.json").read_text())
    model = json.loads(Path(protocol["model"]).read_text())
    previous_routes = {(r["variant"], r["start"]): r for r in json.loads((args.source / "routes.json").read_text())
                       if r["arm"] == ARM_ROLLOUT}
    with (args.source / "fields.csv").open() as stream:
        previous_fields = {field_key(row): row for row in csv.DictReader(stream)}
    edge_cap = protocol["fine_edge_cap"]
    cached_routes, cached_fields = {}, {}
    if args.include_quarter:
        config = json.loads((args.reference_run / "protocol.json").read_text())
        if (Path(config["source"]).resolve() != args.source.resolve() or config["cap"] != CAP
                or config["edge_cap"] != edge_cap or config["early_weight"] != WEIGHTS["early"]):
            parser.error("Expected the paired 0% and 50% reference run with the same budgets.")
        cached_routes = {(r["variant"], r["start"], r["arm"]): r
                         for r in json.loads((args.reference_run / "routes.json").read_text())}
        with (args.reference_run / "fields.csv").open() as stream:
            cached_fields = {field_key(row): row for row in csv.DictReader(stream)}
        if set(cached_routes) != {(*key, arm) for key in previous_routes for arm in WEIGHTS}:
            raise ValueError("Incomplete reference routes")
        if set(cached_fields) != set(previous_fields):
            raise ValueError("Reference fields differ from the refined benchmark")
    rows, field_curves, routes = [], [], []
    with threadpool_limits(limits=1):
        for variant in sorted({key[0] for key in previous_routes}):
            fields, curves, plans = evaluate(variant, model, previous_routes, previous_fields, edge_cap,
                                             weights, cached_routes, cached_fields)
            rows.extend(fields)
            field_curves.append(curves)
            routes.extend(plans)
    if len(rows) != len(previous_fields) or {field_key(row) for row in rows} != set(previous_fields):
        raise ValueError("Replay differs from the archived fields")
    curves = np.concatenate(field_curves)
    summary, curve_rows = summarize(rows, curves, routes, edge_cap, weights)
    args.output.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(rows):
        for arm, series in zip(weights, curves[i]):
            row[f"{arm}_distance_average"] = float(series[:-1].mean())
            row[f"{arm}_final"] = float(series[-1])
    write_csv(args.output / "fields.csv", rows)
    write_csv(args.output / "distance_curve.csv", curve_rows)
    config = dict(source=str(args.source), cap=CAP, edge_cap=edge_cap, early_weight=WEIGHTS["early"],
                  objective="Blend terminal whole-grid variance with its integral over the original walking budget")
    if args.include_quarter:
        config.update(weights=weights, reference_run=str(args.reference_run))
    for name, payload in (("protocol", config), ("routes", routes), ("summary", summary)):
        (args.output / f"{name}.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    render(args.output / "distance-rmse.png", curve_rows, weights)
    for scenario, result in summary["scenarios"].items():
        print(scenario, result["distance_average_contrast"], result["checkpoints"][edge_cap])
    print(summary["routes"])


if __name__ == "__main__":
    main()
