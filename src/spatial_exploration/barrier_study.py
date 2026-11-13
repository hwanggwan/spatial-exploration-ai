"""Compare stationary and floor-plan-aware GPs at matched sensing/travel caps."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .barrier_gp import (GRAPH_WEIGHTS, LENGTHS, kernel_banks, next_measurement,
                         room_layout, transport_covariance)
from .baseline import GPBelief, write_csv

SIZE, NOISE = 12, 0.12
DOORS = (3, 8)
STARTS = (13, 22, 121, 130)
BUDGETS = ((8, 1.0), (16, 2.0))
REGIMES = {"barrier": 0.02, "transparent": 1.0}
KAPPAS = (0.42, 0.68)
MODELS = ("stationary", "geometry")


def evaluation_sites(layout):
    rng = np.random.default_rng(55000 + layout.door_row)
    col = np.arange(SIZE**2) % SIZE
    near = np.isin(col, (SIZE // 2 - 1, SIZE // 2))
    eligible = ~np.isin(np.arange(SIZE**2), STARTS)
    heldout = np.zeros(SIZE**2, dtype=bool)
    for room in (layout.room, ~layout.room):
        for band, count in ((near, 4), (~near, 10)):
            heldout[rng.choice(np.flatnonzero(room & band & eligible), count, replace=False)] = True
    return heldout, near


def collect(layout, start, allowed, budget, sense, fit):
    """The only way to obtain a value is to commit to a legal sensing site."""
    cap, distance_cap = budget
    route, distances, readings = [start], [0.0], [sense(start)]
    fit_seconds, plan_seconds = 0.0, 0.0
    while True:
        before = time.perf_counter()
        belief, parameters = fit(route, readings)
        fit_seconds += time.perf_counter() - before
        before = time.perf_counter()
        action = next_measurement(belief, layout, route, allowed,
                                  distance_cap - distances[-1], cap - len(route))
        plan_seconds += time.perf_counter() - before
        if action is None:
            return route, distances, readings, belief, parameters, fit_seconds, plan_seconds
        distances.append(distances[-1] + layout.distances[route[-1], action])
        route.append(action)
        readings.append(sense(action))


def fixed_design(layout, start, allowed, budget, bank):
    # Only covariance drives this precomputed design; zero is a dummy update.
    prior = bank.covariances[bank.initial]
    belief = GPBelief(prior, NOISE)

    def advance(route, readings):
        belief.observe(route[-1], readings[-1])
        return belief, bank.parameters[bank.initial]

    route, distance, *_ = collect(layout, start, allowed, budget, lambda _: 0.0, advance)
    return route, distance


def scores(truth, belief, heldout, near):
    error = np.abs(truth - belief.mean)
    half_width = 1.96 * belief.std[heldout]
    residual = error[heldout]
    # Lower interval score rewards precision and penalizes misses.
    return dict(rmse=float(np.sqrt(np.mean(residual**2))),
                near_wall_rmse=float(np.sqrt(np.mean(error[heldout & near]**2))),
                coverage=float(np.mean(residual <= half_width)),
                interval_width=float(np.mean(2 * half_width)),
                interval_score=float(np.mean(2 * half_width + 40 * np.maximum(residual - half_width, 0))))


def route_record(layout, route, distances):
    return dict(measurements=len(route), distance=float(distances[-1]),
                crosses_rooms=bool(np.any(layout.room[route] != layout.room[route[0]])))


def paired_interval(deltas, doors, rng):
    # Resample independent field realizations within each fixed floor plan.
    boot = np.concatenate([
        rng.choice(deltas[doors == door], (4000, int((doors == door).sum())), replace=True)
        for door in DOORS
    ], axis=1).mean(axis=1)
    return np.quantile(boot, [0.025, 0.975]).tolist()


def summarize(rows):
    metrics = ("rmse", "near_wall_rmse", "coverage", "interval_width", "interval_score",
               "measurements", "distance", "crosses_rooms", "fit_ms", "plan_ms")
    summary = []
    rng = np.random.default_rng(55001)
    for protocol in ("shared", "closed"):
        for regime in REGIMES:
            for cap, _ in BUDGETS:
                records = [r for r in rows if (r["protocol"], r["regime"], r["cap"])
                           == (protocol, regime, cap)]
                groups = {m: sorted((r for r in records if r["model"] == m),
                                    key=lambda r: (r["door"], r["field"])) for m in MODELS}
                base, graph = (groups[m] for m in MODELS)
                keys = lambda group: [(r["door"], r["field"]) for r in group]
                assert keys(base) == keys(graph)
                delta = np.array([b["rmse"] - a["rmse"] for a, b in zip(base, graph)])
                doors = np.array([r["door"] for r in base])
                summary.append(dict(protocol=protocol, regime=regime, cap=cap, fields=len(delta),
                                    means={m: {k: float(np.mean([r[k] for r in groups[m]]))
                                               for k in metrics} for m in MODELS},
                                    geometry_minus_stationary_rmse=float(delta.mean()),
                                    paired_rmse_interval=paired_interval(delta, doors, rng),
                                    geometry_wins=int((delta < -1e-12).sum()),
                                    same_score=int((np.abs(delta) <= 1e-12).sum())))
    return summary


def summarize_crossover(rows):
    """Change the mapper or the route separately, using existing acquisitions."""
    rng = np.random.default_rng(55002)
    contrasts = []
    for regime in REGIMES:
        for cap, _ in BUDGETS:
            grouped = {(model, route): {} for model in MODELS for route in MODELS}
            for r in rows:
                if r["regime"] == regime and r["cap"] == cap and r["route_model"] in MODELS:
                    grouped[r["model"], r["route_model"]][r["door"], r["field"]] = r["rmse"]
            keys = sorted(grouped[MODELS[0], MODELS[0]])
            doors = np.array([door for door, _ in keys])
            for changed in ("mapper", "route"):
                for fixed in MODELS:
                    base, geometry = ((m, fixed) if changed == "mapper" else (fixed, m) for m in MODELS)
                    delta = np.array([grouped[geometry][k] - grouped[base][k] for k in keys])
                    contrasts.append(dict(regime=regime, cap=cap, changed=changed, fixed=fixed,
                                          geometry_minus_stationary_rmse=float(delta.mean()),
                                          paired_rmse_interval=paired_interval(delta, doors, rng)))
    return contrasts


def render_summary(output, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), sharex=True, sharey=True, layout="constrained")
    for ax, regime in zip(axes, REGIMES):
        records = [r for r in summary if r["regime"] == regime]
        ax.axvline(0, color="0.5", linewidth=1)
        for i, r in enumerate(records):
            low, high = r["paired_rmse_interval"]
            val = r["geometry_minus_stationary_rmse"]
            color = "#146b84" if r["protocol"] == "shared" else "#b34e2c"
            ax.plot([low, high], [i, i], color=color, linewidth=2)
            ax.plot(val, i, "o", color=color)
        ax.set_yticks(range(len(records)), [f"{'Same observations' if r['protocol'] == 'shared' else 'Closed loop'} · {r['cap']} cap"
                                            for r in records])
        ax.set_title("Wall weakens dependence" if regime == "barrier" else "Dependence crosses wall")
        ax.set_xlabel("Geometry-aware minus stationary RMSE")
        ax.grid(axis="x", alpha=.22)
    axes[0].invert_yaxis()
    fig.suptitle("Spatial prediction on unseen sites (left of zero favors geometry)")
    fig.savefig(output / "paired-rmse.png", dpi=170)
    plt.close(fig)


def render_example(output, layout, truth, heldout, example, field):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.1), layout="constrained")
    limit = float(np.max(np.abs(truth)))
    margin = 0.5 / (SIZE - 1)
    y_door = layout.door_row / (SIZE - 1)
    for ax, (label, values, route) in zip(axes, [("Hidden field", truth, None), *example]):
        img = ax.imshow(values.reshape(SIZE, SIZE), origin="lower", cmap="RdBu_r",
                        extent=(-margin, 1 + margin, -margin, 1 + margin), vmin=-limit, vmax=limit)
        ax.plot([.5, .5], [-margin, y_door - margin], color="#202020", lw=2.5)
        ax.plot([.5, .5], [y_door + margin, 1 + margin], color="#202020", lw=2.5)
        ax.scatter(*layout.points[heldout].T, s=7, c="0.25", alpha=.45)
        if route is not None:
            walk = [route[0]]
            for a, b in zip(route, route[1:]):
                walk.extend(layout.walk(a, b)[1:])
            ax.plot(*layout.points[walk].T, color="#216044", lw=1.6)
            ax.scatter(*layout.points[route].T, s=14, color="#fff8e7", edgecolors="#183b2b",
                       linewidths=.7, zorder=3)
            ax.scatter(*layout.points[route[0]], s=45, marker="s", color="#ffcb52",
                       edgecolors="#183b2b", linewidths=1, zorder=4)
        ax.set_title(label)
        ax.set(xticks=[], yticks=[])
    fig.colorbar(img, ax=axes, shrink=.74, label="Latent sensor field")
    fig.suptitle(f"Barrier field {field}, door row {layout.door_row} · 16-reading cap · square marks start")
    fig.savefig(output / "routes.png", dpi=170)
    plt.close(fig)


def run(output, fields_per_layout):
    if fields_per_layout < 1:
        raise ValueError("At least one field is required per layout")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Use a fresh output directory")
    protocol = dict(grid=SIZE, doors=DOORS, starts=STARTS, budgets=BUDGETS, regimes=REGIMES,
                    kappa=KAPPAS, fields_per_layout=fields_per_layout, noise_std=NOISE,
                    lengths=LENGTHS, geometry_weights=GRAPH_WEIGHTS, min_fit_observations=4,
                    field_seeds="56000 + 10000 * door_row + field_index",
                    mask="four near-wall and ten farther test sites per room; cannot sense, can transit",
                    movement="4-neighbor shortest paths through one door; start counted; no required return",
                    truth="unit-diagonal resolvent-squared graph covariance with hidden permeability/kappa",
                    model="zero-mean unit-diagonal RBF or RBF/graph-heat blend; acquired-label grid MLE",
                    planner="shared greedy integrated posterior variance reduction per graph distance",
                    primary_metric="latent RMSE on reserved common test sites")
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    rows, acquisitions, samples, fixed_rows = [], [], [], []
    example, example_crosses = None, False
    for door in DOORS:
        layout = room_layout(SIZE, door)
        heldout, near = evaluation_sites(layout)
        banks = kernel_banks(layout)
        designs = {(start, cap): fixed_design(layout, start, ~heldout, (cap, distance), banks["stationary"])
                   for start in STARTS for cap, distance in BUDGETS}
        factors = {(regime, kappa): np.linalg.cholesky(transport_covariance(layout, permeability, kappa))
                   for regime, permeability in REGIMES.items() for kappa in KAPPAS}
        for field in range(fields_per_layout):
            seed = 56000 + 10000 * door + field
            start, kappa = STARTS[field % len(STARTS)], KAPPAS[(field // len(STARTS)) % len(KAPPAS)]
            z, noise = np.random.default_rng(seed).standard_normal((2, SIZE**2))
            for regime in REGIMES:
                truth = factors[regime, kappa] @ z
                sensor = truth + NOISE * noise
                samples.append((door, field, regime, truth, heldout))
                for cap, distance in BUDGETS:
                    meta = dict(door=door, field=field, seed=seed, regime=regime, cap=cap,
                                distance_cap=distance, start=start)
                    fixed_route, fixed_distance = designs[start, cap]
                    fixed_values = sensor[fixed_route].tolist()
                    fixed_rows.extend(dict(**meta, step=step, site=site, distance=d, value=v)
                                      for step, (site, d, v) in enumerate(zip(fixed_route, fixed_distance, fixed_values)))
                    closed = {}
                    for model, bank in banks.items():
                        def fit(indices, values):
                            return bank.posterior(indices, values, NOISE)

                        before = time.perf_counter()
                        belief, params = fit(fixed_route, fixed_values)
                        elapsed = time.perf_counter() - before
                        rows.append(dict(**meta, protocol="shared", model=model, route_model="fixed",
                                         length=params[0], graph_weight=params[1],
                                         **scores(truth, belief, heldout, near),
                                         **route_record(layout, fixed_route, fixed_distance),
                                         fit_ms=elapsed * 1000, plan_ms=0.0))

                        result = collect(layout, start, ~heldout, (cap, distance), sensor.__getitem__, fit)
                        route, distances, values, belief, params, fit_time, plan_time = result
                        closed[model] = result
                        rows.append(dict(**meta, protocol="closed", model=model, route_model=model,
                                         length=params[0], graph_weight=params[1],
                                         **scores(truth, belief, heldout, near), **route_record(layout, route, distances),
                                         fit_ms=fit_time * 1000, plan_ms=plan_time * 1000))
                        acquisitions.extend(dict(**meta, model=model, step=step, site=site,
                                                 distance=d, value=v)
                                            for step, (site, d, v) in enumerate(zip(route, distances, values)))
                    for model, other in (MODELS, MODELS[::-1]):
                        route, distances, values, *_ = closed[other]
                        before = time.perf_counter()
                        belief, params = banks[model].posterior(route, values, NOISE)
                        elapsed = time.perf_counter() - before
                        rows.append(dict(**meta, protocol="cross", model=model, route_model=other,
                                         length=params[0], graph_weight=params[1],
                                         **scores(truth, belief, heldout, near), **route_record(layout, route, distances),
                                         fit_ms=elapsed * 1000, plan_ms=0.0))
                    if regime == "barrier" and cap == BUDGETS[-1][0] and not example_crosses:
                        crossed = [route_record(layout, *closed[m][:2])["crosses_rooms"] for m in MODELS]
                        # Illustrate the first different room-crossing decision, not a score winner.
                        if example is None or (not crossed[0] and crossed[1]):
                            example = (layout, truth, heldout, [
                                (label, closed[model][3].mean, closed[model][0])
                                for model, label in zip(MODELS, ("Stationary GP", "Geometry-aware GP"))], field)
                            example_crosses = not crossed[0] and crossed[1]
    summary = summarize(rows)
    write_csv(output / "metrics.csv", rows)
    write_csv(output / "closed-observations.csv", acquisitions)
    write_csv(output / "shared-observations.csv", fixed_rows)
    door, field, regime, truth, mask = zip(*samples)
    np.savez_compressed(output / "evaluation-only.npz", door=door, field=field, regime=regime,
                        truth=truth, heldout=mask)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "crossover.json").write_text(json.dumps(summarize_crossover(rows), indent=2) + "\n")
    render_example(output, *example)
    render_summary(output, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fields-per-layout", type=int, default=48)
    parser.add_argument("--output", type=Path, default=Path("outputs/barrier-study"))
    args = parser.parse_args()
    for row in run(args.output, args.fields_per_layout):
        a, b = (row["means"][m]["rmse"] for m in MODELS)
        print(f"{row['protocol']:6s} {row['regime']:11s} cap {row['cap']:2}: "
              f"RMSE {a:.4f} vs {b:.4f}; 95% paired interval {row['paired_rmse_interval']}")


if __name__ == "__main__":
    main()
