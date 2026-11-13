"""Compare room allocation and full rollout on three unequal, known rooms."""

import argparse
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .barrier_deepkriging_study import NeuralMapperConfig, dump, graph_heat_features, standardized_features
from .barrier_gp import transport_covariance
from .barrier_study import KAPPAS, NOISE, REGIMES
from .baseline import GPBelief, write_csv
from .basis_bayes import BayesianBasisModel
from .three_room_quota import SIZE, STARTS, choose_allocation, quota_next, three_rooms
from .wall_rollout_study import Route, load_settings, metrics, plan_route


CAP = 16
BUDGETS = {"tight": 18 / 11, "loose": 26 / 11}
CAL_LAYOUTS, TEST_LAYOUTS = ((3, 8), (9, 4)), ((2, 9), (8, 3))
CAL_SEED, TEST_SEED = 1810000, 1910000
PRIMARY = ("equal", "priority", "rollout")
MATCHED = ("rollout_at_equal_distance", "rollout_at_priority_distance")
METHODS = (*PRIMARY, *MATCHED)


def design_for(doors):
    layout = three_rooms(doors)
    rng = np.random.default_rng(18001 + 12 * doors[0] + doors[1])
    held = np.zeros(SIZE**2, dtype=bool)
    col = np.arange(SIZE**2) % SIZE
    near = np.isin(col, [2, 3, 8, 9])
    eligible = ~np.isin(np.arange(SIZE**2), STARTS)
    for room, per_band in enumerate(((2, 5), (4, 10), (2, 5))):
        for band, count in zip((near, ~near), per_band):
            indices = np.flatnonzero(eligible & band & (layout.room == room))
            held[rng.choice(indices, count, replace=False)] = True
    base = standardized_features(layout.points, "deepkriging", NeuralMapperConfig())[0]
    graph = np.column_stack((base, graph_heat_features(layout))).astype(np.float32)
    return SimpleNamespace(layout=layout, heldout=held, near_wall=near,
                           features=graph)


def quota_route(design, model, start, budget, method):
    layout, allowed = design.layout, ~design.heldout
    belief = GPBelief(model.prior_covariance, model.noise_std)
    belief.observe(start, 0.0)
    before = time.perf_counter()
    order, pending = choose_allocation(layout, allowed, start, CAP, budget, belief.covariance, method)
    elapsed = time.perf_counter() - before
    visited = np.zeros_like(allowed)
    visited[start] = True
    sites, distances = [start], [0.0]
    while pending.any():
        before = time.perf_counter()
        nxt = quota_next(layout, allowed, visited, sites[-1], budget - distances[-1], order, pending, belief)
        elapsed += time.perf_counter() - before
        distances.append(distances[-1] + float(layout.distances[sites[-1], nxt]))
        sites.append(nxt)
        visited[nxt] = True
        pending[layout.room[nxt]] -= 1
        belief.observe(nxt, 0.0)
    return Route(tuple(sites), tuple(distances), float(np.mean(belief.covariance.diagonal())),
                 elapsed * 1000, 0, 0)


def make_routes(design, model, label):
    routes = {}
    for start in STARTS:
        for scenario, budget in BUDGETS.items():
            by_method = {m: quota_route(design, model, start, budget, m) for m in PRIMARY[:2]}
            by_method["rollout"] = plan_route(design, model, start, CAP, budget, "rollout")
            for arm in PRIMARY[:2]:
                by_method[f"rollout_at_{arm}_distance"] = plan_route(
                    design, model, start, CAP, by_method[arm].distances[-1], "rollout")
            for method, route in by_method.items():
                distance_cap = budget if method in PRIMARY else by_method[method.split("_")[2]].distances[-1]
                sites = list(route.sites)
                if (sites[0] != start or len(set(sites)) != len(sites) or len(sites) > CAP
                        or design.heldout[sites].any() or route.distances[-1] > distance_cap + 1e-12):
                    raise RuntimeError("Route violates sensing or travel limits")
                for a, b, step in zip(sites, sites[1:], np.diff(route.distances)):
                    walk = design.layout.walk(a, b)
                    if (not all(design.layout.movement[u, v] for u, v in zip(walk, walk[1:]))
                            or abs(step - (len(walk) - 1) / (SIZE - 1)) > 1e-12):
                        raise RuntimeError("Route uses an illegal walking leg")
                counts = np.bincount(design.layout.room[sites], minlength=3).tolist()
                routes[start, scenario, method] = dict(layout=label, start=start, budget=scenario,
                    method=method, distance_cap=distance_cap, counts=counts, **asdict(route))
    return routes


def fields(design, label, count, seed_base):
    factors = {(r, k): np.linalg.cholesky(transport_covariance(design.layout, p, k))
               for r, p in REGIMES.items() for k in KAPPAS}
    for i in range(count):
        seed = seed_base + 1000 * label + i
        start = STARTS[i % len(STARTS)]
        scale = KAPPAS[(i // len(STARTS)) % len(KAPPAS)]
        z, noise = np.random.default_rng(seed).standard_normal((2, SIZE**2))
        for regime in REGIMES:
            truth = factors[regime, scale] @ z
            yield dict(layout=label, start=start, kappa=scale, seed=seed, regime=regime), truth, truth + NOISE * noise


def evaluate(door_pairs, spec, count, seed, scales=None):
    ratios = {(b, m): [] for b in BUDGETS for m in METHODS}
    rows, route_rows, designs, archive, measurements = [], [], {}, {}, []
    for label, doors in enumerate(door_pairs):
        design = designs[label] = design_for(doors)
        features = design.features
        model = BayesianBasisModel(features, NOISE, spec["prior"],
                                   intercept_scale=np.sqrt((features.shape[1] + 1) * spec["intercept"]))
        routes = make_routes(design, model, label)
        route_rows.extend(routes.values())
        mask = design.heldout
        samples = list(fields(design, label, count, seed))
        if scales is not None:
            archive.update({f"layout{label}_truth": np.stack([t for _, t, _ in samples]),
                            f"layout{label}_seeds": np.array([r["seed"] for r, _, _ in samples]),
                            f"layout{label}_regimes": np.array([r["regime"] for r, _, _ in samples]),
                            f"layout{label}_heldout": mask})
        for meta, truth, sensor in samples:
            for b in BUDGETS:
                for m in METHODS:
                    route = routes[meta["start"], b, m]
                    sites = list(route["sites"])
                    observed = sensor[sites]
                    belief = model.condition(sites, observed)
                    if abs(belief.covariance.diagonal().mean() - route["final_variance"]) > 1e-12:
                        raise RuntimeError("Replay covariance does not match planning")
                    if scales is None:
                        ratios[b, m].extend(np.abs(truth[mask] - belief.mean[mask]) /
                                            np.maximum(belief.std[mask], 1e-8))
                    else:
                        result = metrics(truth, belief, design, scales[b, m])
                        error = (truth - belief.mean)**2
                        rows.append(dict(**meta, budget=b, method=m, distance=route["distances"][-1],
                            measurements=len(sites), rooms=sum(n > 0 for n in route["counts"]),
                            **{f"room{j}_count": n for j, n in enumerate(route["counts"])},
                            **{f"room{j}_rmse": float(np.sqrt(np.mean(error[mask & (design.layout.room == j)])))
                               for j in range(3)}, **result))
                        measurements.extend(dict(**meta, budget=b, method=m, step=i, site=s,
                            distance=d, value=float(v))
                            for i, (s, d, v) in enumerate(zip(sites, route["distances"], observed)))
    calibrated = ({k: float(np.quantile(r, .95) / 1.96) for k, r in ratios.items()} if scales is None else {})
    return rows, route_rows, designs, calibrated, archive, measurements


def summarize(rows, routes):
    rng = np.random.default_rng(191001)
    result = []
    cols = ("rmse", "near_wall_rmse", "distance", "measurements", "rooms", "coverage", "raw_coverage",
            "width", "modeled_grid_variance", *(f"room{i}_count" for i in range(3)),
            *(f"room{i}_rmse" for i in range(3)))
    for r in REGIMES:
        for b in BUDGETS:
            groups = {m: {(x["layout"], x["seed"]): x for x in rows
                          if (x["regime"], x["budget"], x["method"]) == (r, b, m)} for m in METHODS}
            keys = sorted(groups["equal"])
            if not all(set(g) == set(keys) for g in groups.values()):
                raise RuntimeError("Unmatched fields")
            strata = {}
            for i, k in enumerate(keys):
                x = groups["equal"][k]
                strata.setdefault((x["layout"], x["start"], x["kappa"]), []).append(i)
            contrasts = {}
            for a, ref in (("priority", "equal"), ("rollout", "priority"), ("rollout", "equal"),
                           *zip(MATCHED, PRIMARY[:2])):
                delta = np.array([groups[a][k]["rmse"] - groups[ref][k]["rmse"] for k in keys])
                boot = np.concatenate([rng.choice(delta[idx], (6000, len(idx))) for idx in strata.values()],
                                      axis=1).mean(axis=1)
                contrasts[f"{a}_minus_{ref}"] = dict(difference=float(delta.mean()),
                    interval=np.quantile(boot, [.025, .975]).tolist())
            stats = {}
            for m in METHODS:
                selected = [x for x in routes if (x["budget"], x["method"]) == (b, m)]
                stats[m] = dict(planning_ms=float(np.mean([x["planning_ms"] for x in selected])),
                    all_rooms=sum(all(x["counts"]) for x in selected),
                    full_count=sum(len(x["sites"]) == CAP for x in selected),
                    configurations=len(selected))
            result.append(dict(regime=r, budget=b, fields=len(keys), contrasts=contrasts, routes=stats,
                means={m: {c: float(np.mean([x[c] for x in g.values()])) for c in cols}
                       for m, g in groups.items()}))
    return result


def render(output, routes, designs, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ("Equal room quota", "Priority room quota", "Rollout")
    colors = ("#eaf3f5", "#f5efde", "#eceaf5")
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.6), layout="constrained")
    layout = designs[0].layout
    pad = .5 / (SIZE - 1)
    boundaries = (2.5 / 11, 8.5 / 11)
    for i, b in enumerate(BUDGETS):
        for ax, m, label in zip(axes[i], PRIMARY, labels):
            record = next(x for x in routes if (x["layout"], x["start"], x["budget"], x["method"])
                          == (0, STARTS[0], b, m))
            sites = list(record["sites"])
            path = [sites[0]]
            for a, t in zip(sites, sites[1:]):
                path.extend(layout.walk(a, t)[1:])
            bounds = (-pad, *boundaries, 1 + pad)
            for left, right, color in zip(bounds, bounds[1:], colors):
                ax.axvspan(left, right, color=color)
            for x, door in zip(boundaries, layout.door_row):
                y = door / 11
                ax.vlines([x, x], [-pad, y + pad], [y - pad, 1 + pad], color="#202020", lw=2)
            ax.scatter(*layout.points[designs[0].heldout].T, s=11, c="#a0a0a0")
            ax.plot(*layout.points[path].T, color="#22687a", lw=1.7)
            ax.scatter(*layout.points[sites].T, s=23, facecolor="white", edgecolor="#22687a", zorder=3)
            ax.scatter(*layout.points[sites[0]], s=54, marker="s", color="#ffc751", edgecolor="#202020", zorder=4)
            ax.set(xlim=(-pad, 1 + pad), ylim=(-pad, 1 + pad), xticks=[], yticks=[], aspect="equal")
            ax.set_title(f"{label} · {b}\n{' / '.join(map(str, record['counts']))} readings · "
                         f"{record['distances'][-1]:.3f} distance", fontsize=10)
    fig.suptitle("Three-room sensing · same start · counts ordered left to right")
    fig.savefig(output / "routes.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), sharey=True, layout="constrained")
    for ax, regime in zip(axes, REGIMES):
        for idx, (m, label, color) in enumerate(zip(PRIMARY, labels, ("#637c8e", "#b55f36", "#287878"))):
            series = [s for s in summary if s["regime"] == regime]
            ax.bar(np.arange(2) + (idx - 1) * .24, [s["means"][m]["rmse"] for s in series],
                   width=.23, label=label, color=color)
        ax.set_title("Weak across walls" if regime == "barrier" else "Dependence crosses walls")
        ax.set_xticks(range(2), ["Tight travel cap", "Loose travel cap"])
        ax.grid(axis="y", alpha=.22)
    axes[0].set_ylabel("Mean held-out latent RMSE")
    axes[1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Same 16-reading cap · 256 fresh fields per condition")
    fig.savefig(output / "comparison.png", dpi=170)
    plt.close(fig)


def run(reference, output):
    settings, digest = load_settings(reference)
    spec = settings[CAP]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Use a fresh output directory")
    protocol = dict(reference_sha256=digest, ridge=spec, widths=[3, 6, 3], starts=STARTS,
        cal_layouts=CAL_LAYOUTS, test_layouts=TEST_LAYOUTS, cal_seed=CAL_SEED, test_seed=TEST_SEED,
        calibration_fields_per_layout=32, test_fields_per_layout=128, regime_permeability=REGIMES,
        kappa=KAPPAS, sensor_noise=NOISE, cap=CAP, travel_caps=BUDGETS,
        equal="Most feasible readings, then most rooms, then shortest completion witness; uniform allocation",
        priority="Mass after start / (1 + entry distance); apportion by weight; choose max sum of weight*(1-exp(-n/4))",
        witness="Nearest-neighbor completion in each enumerated room order; preserve feasibility on every greedy choice",
        rollout="All feasible first moves, greedy completion to both caps, replan after each move",
        matched="Rollout capped separately at each quota route's actual distance; counts can differ",
        metrics="Mean field RMSE on 28 reserved sites; 7, 14, 7 per room")
    dump(output / "protocol.json", protocol)
    with threadpool_limits(limits=1):
        _, _, _, scales, _, _ = evaluate(CAL_LAYOUTS, spec, 32, CAL_SEED)
        dump(output / "frozen.json", {**protocol, "std_scales": {f"{b}-{m}": v for (b, m), v in scales.items()}})
        rows, routes, designs, _, truth, measured = evaluate(TEST_LAYOUTS, spec, 128, TEST_SEED, scales)
    summary = summarize(rows, routes)
    write_csv(output / "metrics.csv", rows)
    write_csv(output / "acquisitions.csv", measured)
    dump(output / "routes.json", routes)
    dump(output / "summary.json", summary)
    np.savez_compressed(output / "evaluation-only.npz", **truth)
    render(output, routes, designs, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("outputs/wall-topology-study/frozen.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/three-room-study"))
    args = parser.parse_args()
    for row in run(args.reference, args.output):
        print(row["regime"], row["budget"],
              {m: {k: round(v[k], 5) for k in ("rmse", "distance", "measurements", "rooms")}
               for m, v in row["means"].items()})


if __name__ == "__main__":
    main()
