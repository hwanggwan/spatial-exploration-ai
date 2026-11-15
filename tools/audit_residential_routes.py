"""Compare frozen sensing models on corridor and loop floor plans."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.sparse.csgraph import shortest_path
from threadpoolctl import threadpool_limits

from audit_sensing_world_shifts import correlation_area, match_scale, within_room_correlogram
from spatial_exploration.barrier_deepkriging_study import (
    NeuralMapperConfig, graph_heat_features, standardized_features,
)
from spatial_exploration.barrier_gp import RoomLayout, graph_spectrum, unit_covariance
from spatial_exploration.baseline import make_grid, write_csv
from spatial_exploration.basis_bayes import BayesianBasisModel
from spatial_exploration.sensing_world import Belief, choose_rollout_site, kernel_basis, rollout_samples
from spatial_exploration.sensing_world_generalization import EDGES, wall_layout
from spatial_exploration.sensing_world_study import CAP, NOISE, fixed_route_rmse, run_episode
from spatial_exploration.wall_rollout_study import plan_route


SIZE = 12
STARTS = (14, 21, 64, 128)
AREAS = (2.5, 4.5)
CONDUCTANCES = {"weak": .02, "persistent": 1.}
SEED = 302001
FAMILIES = ("row", "corridor", "loop")


@dataclass(frozen=True)
class LayoutCase:
    family: str
    variant: int


def door_layout(zones, doors):
    """Construct a connected movement graph with explicit doorway edges."""
    room = zones.ravel()
    grid = np.arange(len(room))
    r, c = np.divmod(grid, SIZE)
    adjacent = (abs(r[:, None] - r) + abs(c[:, None] - c)) == 1
    walls = adjacent & (room[:, None] != room[None, :])
    for left, right in doors:
        a = int(np.ravel_multi_index(left, zones.shape))
        b = int(np.ravel_multi_index(right, zones.shape))
        if not walls[a, b]:
            raise ValueError("Door must connect adjacent sites across a wall")
        walls[a, b] = walls[b, a] = False
    movement = adjacent & ~walls
    distances, paths = shortest_path(movement.astype(float), directed=False, unweighted=True, return_predecessors=True)
    if not np.isfinite(distances).all():
        raise ValueError("Every zone must be reachable")
    return RoomLayout(SIZE, doors, make_grid(SIZE), room, walls, movement, distances/(SIZE-1), paths)


def floor_plan(case):
    r, c = np.indices((SIZE, SIZE))
    v = case.variant
    if case.family == "row":
        return wall_layout(((3, 9), (0, 0), (((2, 9)[v],), ((9, 2)[v],)), False))
    if case.family == "corridor":
        left_wall = 6 + v
        zones = np.where(r < 5, (c >= left_wall).astype(int), np.where(r <= 6, 2, 3))
        doors = (((4, 2 + 2*v), (5, 2 + 2*v)), ((4, 9), (5, 9)), ((6, 5 + 3*v), (7, 5 + 3*v)))
    elif case.family == "loop":
        split_row, split_col = 6 - v, 6 + v
        zones = 2*(r >= split_row).astype(int) + (c >= split_col).astype(int)
        doors = (((1 + v, split_col-1), (1 + v, split_col)), ((9, split_col-1), (9, split_col)),
                 ((split_row-1, 2), (split_row, 2)), ((split_row-1, 9 - v), (split_row, 9 - v)))
    else:
        raise ValueError(f"Unknown floor plan family {case.family}")
    return door_layout(zones, doors)


def design_for(layout, seed):
    rng = np.random.default_rng(seed)
    coords = standardized_features(layout.points, "deepkriging", NeuralMapperConfig())[0]
    features = np.column_stack((coords, graph_heat_features(layout))).astype(np.float32)
    # Preserve 28 reserved sites, with coverage of every zone.
    held = np.zeros(len(layout.points), dtype=bool)
    eligible = ~np.isin(np.arange(len(held)), STARTS)
    counts = np.bincount(layout.room)
    allocation = 28 * counts / counts.sum()
    take = allocation.astype(int)
    take[np.argsort(-(allocation - take), kind="stable")[:28-take.sum()]] += 1
    for zone, n in enumerate(take):
        held[rng.choice(np.flatnonzero(eligible & (layout.room == zone)), n, replace=False)] = True
    return SimpleNamespace(layout=layout, features=features, heldout=held)


def field_covariances(layout, permeability):
    """Return the simulator covariances at the two target correlogram areas."""
    correlogram = within_room_correlogram(layout)
    eigenvalues, vectors = graph_spectrum(layout, permeability)

    def covariance(scale):
        return unit_covariance(vectors, (scale**2 + eigenvalues)**-2)

    return {target: match_scale(covariance, correlogram, correlation_area, target) for target in AREAS}


def field_cases(layout, case, draws, diagnostics, *, covariances=None):
    """Yield paired range conditions from common field and sensor innovations."""
    correlogram = within_room_correlogram(layout)
    rng = np.random.default_rng(SEED + 100 + case.variant)
    for regime, p in CONDUCTANCES.items():
        models = field_covariances(layout, p) if covariances is None else covariances[regime]
        factors = {}
        for target, (scale, matrix) in models.items():
            factors[target] = np.linalg.cholesky(matrix)
            diagnostics.append(dict(family=case.family, variant=case.variant, regime=regime, target=target,
                area=correlation_area(correlogram(matrix)), scale=scale))
        for start in STARTS:
            for draw in range(draws):
                z, noise = rng.standard_normal((2, len(layout.points)))
                for target, factor in factors.items():
                    truth = factor @ z
                    sensor = truth + NOISE * noise
                    key = dict(family=case.family, variant=case.variant, regime=regime, area=target, start=start, draw=draw)
                    yield key, truth, sensor


def evaluate(case, spec, weights, draws):
    with threadpool_limits(limits=1):
        return evaluate_layout(case, spec, weights, draws)


def evaluate_layout(case, spec, weights, draws):
    layout = floor_plan(case)
    design = design_for(layout, SEED + case.variant)
    intercept = np.sqrt((design.features.shape[1]+1) * spec["intercept"])
    ridge = BayesianBasisModel(design.features, NOISE, spec["prior"], intercept_scale=intercept)
    gp = Belief.prior(ridge.prior_covariance[None], NOISE)
    world = Belief.prior(np.einsum("mb,bij->mij", weights, kernel_basis(layout)), NOISE)
    routes = {s: plan_route(design, ridge, s, CAP, EDGES/(SIZE-1), "rollout") for s in STARTS}
    planner = partial(choose_rollout_site, innovations=rollout_samples(CAP))
    rows, diagnostics, examples = [], [], []
    for record, truth, sensor in field_cases(layout, case, draws, diagnostics):
        start = record["start"]
        route = routes[start]
        episodes = (run_episode(gp, design, start, EDGES, sensor.__getitem__, fixed=route.sites),
                    run_episode(world, design, start, EDGES, sensor.__getitem__, planner=planner))
        for name, episode in zip(("gp", "world"), episodes):
            sites, walk, _, readings, states, forecasts, seconds = episode
            obs_mean, obs_sd = np.asarray(forecasts).T
            record.update({f"{name}_rmse": float(np.sqrt(np.mean((states[-1][0][design.heldout] - truth[design.heldout])**2))),
                f"{name}_edges": len(walk)-1, f"{name}_readings": len(sites),
                f"{name}_rooms": len(set(layout.room[sites]) - ({2} if case.family == "corridor" else set())),
                f"{name}_coverage": float(np.mean(abs(readings[1:] - obs_mean) <= 1.96 * obs_sd)),
                f"{name}_width": float(np.mean(3.92 * obs_sd)),
                f"{name}_plan_ms": route.planning_ms if name == "gp" else seconds*1000})
        record["world_on_gp"] = fixed_route_rmse(world, episodes[0][0], sensor.__getitem__, truth, design.heldout)
        record["gp_on_world"] = fixed_route_rmse(gp, episodes[1][0], sensor.__getitem__, truth, design.heldout)
        rows.append(record)
        if (record["regime"], record["area"], start, record["draw"]) == ("weak", AREAS[-1], STARTS[0], 0):
            examples.append(dict(family=case.family, variant=case.variant, start=start,
                held=np.flatnonzero(design.heldout).tolist(),
                routes={name: dict(sites=episode[0], walk=episode[1])
                        for name, episode in zip(("gp", "world"), episodes)}))
    return rows, diagnostics, examples


def summarize(rows):
    rng = np.random.default_rng(SEED + 200)
    result = []
    for family in FAMILIES:
        for area in AREAS:
            group = [r for r in rows if r["family"] == family and r["area"] == area]
            strata = {}
            for i, r in enumerate(group):
                strata.setdefault((r["variant"], r["regime"], r["start"]), []).append(i)
            boot = np.concatenate([rng.choice(i, (6000, len(i))) for i in strata.values()], axis=1)
            metrics = {k: float(np.mean([r[k] for r in group])) for k in group[0] if k.startswith(("gp_", "world_"))}
            scores = {k: np.array([r[k] for r in group]) for k in ("gp_rmse", "world_rmse", "world_on_gp", "gp_on_world")}
            contrasts = {}
            for label, delta in (("closed_loop_world_minus_gp", scores["world_rmse"]-scores["gp_rmse"]),
                                 ("same_gp_route_world_minus_gp", scores["world_on_gp"]-scores["gp_rmse"]),
                                 ("world_routes_minus_gp_routes", scores["world_rmse"]-scores["world_on_gp"])):
                contrasts[label] = dict(mean=float(delta.mean()),
                    ci95=np.quantile(delta[boot].mean(axis=1), [.025, .975]).tolist())
            result.append(dict(family=family, area=area, fields=len(group), **metrics, contrasts=contrasts))
    return result


def render(examples, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    fig, axes = plt.subplots(3, 4, figsize=(11.5, 8.4), layout="constrained")
    colors = ListedColormap(["#edf1df", "#e4eee9", "#e5eaf3", "#f6e8db"])
    names = {"row": "Linear rooms", "corridor": "Apartment corridor", "loop": "Connected rooms"}
    for example in examples:
        family, variant = example["family"], example["variant"]
        layout = floor_plan(LayoutCase(family, variant))
        for j, (arm, route) in enumerate(example["routes"].items()):
            ax = axes[FAMILIES.index(family), variant*2+j]
            ax.imshow(layout.room.reshape(SIZE, SIZE), cmap=colors, vmin=0, vmax=3, interpolation="nearest")
            for a, b in zip(*np.nonzero(np.triu(layout.wall_edges, 1))):
                r1, c1 = divmod(a, SIZE)
                r2, c2 = divmod(b, SIZE)
                if r1 == r2:
                    ax.plot([c1+.5]*2, [r1-.5, r1+.5], color="#243934", lw=1.8)
                else:
                    ax.plot([c1-.5, c1+.5], [r1+.5]*2, color="#243934", lw=1.8)
            c = "#137b78" if arm == "gp" else "#7445a3"
            y, x = np.divmod(route["walk"], SIZE)
            ax.plot(x, y, color=c, alpha=.8, lw=1.25)
            y, x = np.divmod(route["sites"], SIZE)
            ax.scatter(x, y, s=15, c="white", edgecolor=c, linewidth=.9, zorder=3)
            ax.scatter(x[0], y[0], s=34, c="#f6c34f", edgecolor="#273733", marker="s", zorder=4)
            label = "GP" if arm == "gp" else "World model"
            ax.set_title(f"{names[family]} {variant+1} · {label}", fontsize=9, color="#223731")
            ax.set(xticks=[], yticks=[], xlim=(-.5, 11.5), ylim=(11.5, -.5))
            for spine in ax.spines.values():
                spine.set_color("#aab8ae")
    fig.suptitle("Sampled routes on residential floor plans\nSame start and budget · long correlation · weak cross-wall dependence",
                 fontsize=15, color="#223731")
    fig.savefig(output / "floor-plans.png", dpi=165, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("outputs/sensing-world-modes/frozen.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/sensing-world-residential"))
    parser.add_argument("--draws", type=int, default=6, help="Paired draws per layout, wall conductance, and start.")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.draws < 2 or args.workers < 1:
        parser.error("Use at least two draws and one worker.")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use an empty output directory.")
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.model.read_text())
    cases = [LayoutCase(family, variant) for family in FAMILIES for variant in range(2)]
    protocol = dict(model=str(args.model), areas=AREAS, conductances=CONDUCTANCES, starts=STARTS,
        draws_per_start=args.draws, edges=EDGES, cap=CAP, seed=SEED)
    (args.output/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    local = partial(evaluate, spec=config["gp"], weights=np.asarray(config["world_weights"]["world_4"]), draws=args.draws)
    rows, diagnostics, examples = [], [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for case, (cases_rows, range_rows, scenes) in zip(cases, pool.map(local, cases)):
            rows.extend(cases_rows)
            diagnostics.extend(range_rows)
            examples.extend(scenes)
            print(f"Completed {case.family} plan {case.variant+1}: {len(cases_rows)} paired fields", flush=True)
    write_csv(args.output/"metrics.csv", rows)
    write_csv(args.output/"range.csv", diagnostics)
    summary = summarize(rows)
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    (args.output/"routes.json").write_text(json.dumps(examples, indent=2)+"\n")
    render(examples, args.output)


if __name__ == "__main__":
    main()
