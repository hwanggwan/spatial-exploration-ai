"""Test learned sensing on unseen wall geometry and shifted fields."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.sparse.csgraph import shortest_path
from threadpoolctl import threadpool_limits

from .barrier_gp import RoomLayout, graph_spectrum, unit_covariance
from .baseline import make_grid, write_csv
from .basis_bayes import BayesianBasisModel
from .sensing_world import Belief, choose_rollout_site, fit_kernel_weights, kernel_basis, rollout_samples
from .sensing_world_study import CAP, MODES, NOISE, learn, run_episode
from .three_room_study import design_for
from .wall_closed_loop_study import load_settings
from .wall_rollout_study import plan_route


SIZE, EDGES = 12, 26
TRAIN_SEED, TEST_SEED = 294001, 295001
DRAWS_PER_MODE = 192
ARMS = ("gp_rollout", "original_world", "diverse_world")
# Each wall can bend or have two openings. Shift moves its upper half right or left.
# Training and test geometry differ in wall positions, bends, and door rows.
TRAIN_GEOMETRIES = (
    ((2, 8), (0, 0), ((2, 7), (5,)), False),
    ((4, 9), (0, 0), ((4,), (1, 9)), True),
    ((3, 8), (1, 0), ((3,), (2, 9)), False),
    ((2, 7), (0, 1), ((1, 6), (8,)), True),
    ((4, 8), (-1, 0), ((2, 10), (4,)), False),
    ((3, 9), (1, 0), ((4,), (2, 8)), True),
    ((2, 9), (0, -1), ((3, 9), (5,)), False),
    ((3, 8), (-1, 1), ((5,), (3, 8)), True),
)
TEST_GEOMETRIES = (
    ((2, 7), (1, 0), ((2, 9), (4, 10)), False),
    ((4, 9), (-1, -1), ((1, 8), (4, 10)), True),
    ((3, 8), (0, -1), ((1, 8), (3,)), False),
)
STANDARD_DOORS = ((2, 9), (8, 3), (1, 8))
STARTS = (13, 52, 88, 118)


def wall_layout(geometry):
    walls, shifts, doors, transpose = geometry
    row, col = np.indices((SIZE, SIZE))
    boundaries = [wall + shift * (row >= SIZE // 2) for wall, shift in zip(walls, shifts)]
    rooms = (col >= boundaries[0]).astype(int) + (col >= boundaries[1]).astype(int)
    door_edges = []
    for boundary, openings in zip(boundaries, doors):
        for r in openings:
            c = boundary[r, 0]
            left, right = (r, c - 1), (r, c)
            if transpose:
                left, right = left[::-1], right[::-1]
            door_edges.append((np.ravel_multi_index(left, rooms.shape), np.ravel_multi_index(right, rooms.shape)))
    if transpose:
        rooms = rooms.T
    rooms, row, col = rooms.ravel(), row.ravel(), col.ravel()
    adjacent = (abs(row[:, None] - row) + abs(col[:, None] - col)) == 1
    wall = adjacent & (rooms[:, None] != rooms[None, :])
    for a, b in door_edges:
        if not wall[a, b]:
            raise ValueError("Door must cross a wall")
        wall[a, b] = wall[b, a] = False
    movement = adjacent & ~wall
    hops, predecessors = shortest_path(movement, directed=False, unweighted=True, return_predecessors=True)
    if not np.isfinite(hops).all():
        raise ValueError("The floor plan must be connected")
    return RoomLayout(SIZE, doors, make_grid(SIZE), rooms, wall, movement, hops/(SIZE-1), predecessors)


def make_design(layout, seed):
    # The same coordinate-based and wall-graph inputs used by three_room_study.
    from .barrier_deepkriging_study import NeuralMapperConfig, graph_heat_features, standardized_features

    coordinate = standardized_features(layout.points, "deepkriging", NeuralMapperConfig())[0]
    features = np.column_stack((coordinate, graph_heat_features(layout))).astype(np.float32)
    rng = np.random.default_rng(seed)
    held = np.zeros(len(layout.points), dtype=bool)
    eligible = ~np.isin(np.arange(len(held)), STARTS)
    for room, quota in enumerate((7, 14, 7)):
        sites = np.flatnonzero(eligible & (layout.room == room))
        held[rng.choice(sites, quota, replace=False)] = True
    return SimpleNamespace(layout=layout, heldout=held, features=features)


def field_covariance(layout, permeability, kappa, *, power=2., source=None):
    values, vectors = graph_spectrum(layout, permeability)
    covariance = unit_covariance(vectors, (kappa**2 + values)**-power)
    if source is not None:
        center, width = source
        bump = np.exp(-np.sum((layout.points - center)**2, axis=1) / (2 * width**2))
        covariance = .76 * covariance + .40 * np.outer(bump, bump)
    return covariance


def train_diverse():
    rng = np.random.default_rng(TRAIN_SEED)
    training = []
    for geometry in TRAIN_GEOMETRIES:
        layout = wall_layout(geometry)
        groups = []
        for regime, scale in MODES:
            draws = []
            for i in range(4):
                # Preserve four modes, but train each on a range of spatial and sensor patterns.
                p = (.01, .04, .09, .16)[i] if regime == "barrier" else (.58, .73, .88, 1.)[i]
                kappa = scale * (.82, .94, 1.06, 1.18)[i]
                source = (rng.uniform(.2, .8, size=2), rng.uniform(.12, .24)) if i % 2 else None
                factor = np.linalg.cholesky(field_covariance(layout, p, kappa, power=(1.8, 2., 2.2, 2.4)[i],
                                                            source=source))
                innovations = rng.standard_normal((DRAWS_PER_MODE//4, len(factor)))
                noise = rng.normal(0, NOISE, size=innovations.shape)
                draws.append(innovations @ factor.T + noise)
            groups.append(np.concatenate(draws))
        training.append((kernel_basis(layout), groups))
    return fit_kernel_weights(training, NOISE)


@dataclass(frozen=True)
class Job:
    geometry: str
    layout_id: int
    field_count: int


def simulate(design, job):
    """Yield paired sensor realizations for benchmark fields and shifted fields."""
    rng = np.random.default_rng(TEST_SEED + job.layout_id)
    factors = {}
    for i, (regime, kappa) in enumerate(MODES):
        standard = .02 if regime == "barrier" else 1.
        # The shifted generator uses intermediate wall leakage, a different
        # covariance spectrum, and local source centers outside the training range.
        shifted = .25 if regime == "barrier" else .45
        edge_x = .10 if i % 2 else .90
        center = (edge_x, .22 if i < 2 else .78)
        factors["benchmark", i] = np.linalg.cholesky(field_covariance(design.layout, standard, kappa))
        factors["shifted", i] = np.linalg.cholesky(field_covariance(design.layout, shifted, kappa,
            power=1.45, source=(center, .14)))
    for index in range(job.field_count):
        mode, start = (index // len(STARTS)) % len(MODES), STARTS[index % len(STARTS)]
        z, noise = rng.standard_normal((2, len(design.heldout)))
        for shift in ("benchmark", "shifted"):
            truth = factors[shift, mode] @ z
            yield index, start, mode, shift, truth, truth + NOISE * noise


def evaluate_layout(job, gp_config, weights):
    with threadpool_limits(limits=1):
        if job.geometry == "benchmark":
            layout = design_for(STANDARD_DOORS[job.layout_id]).layout
        else:
            layout = wall_layout(TEST_GEOMETRIES[job.layout_id])
        design = make_design(layout, TEST_SEED + 100 * job.layout_id)
        ridge = BayesianBasisModel(design.features, NOISE, gp_config["prior"],
            intercept_scale=np.sqrt((design.features.shape[1] + 1) * gp_config["intercept"]))
        gp = Belief.prior(ridge.prior_covariance[None], NOISE)
        basis = kernel_basis(layout)
        priors = [gp] + [Belief.prior(np.einsum("mb,bij->mij", w, basis), NOISE) for w in weights]
        allowed = ~design.heldout
        fixed = {start: plan_route(design, ridge, start, CAP, EDGES/(SIZE-1), "rollout") for start in STARTS}
        innovations = rollout_samples(CAP)

        def planner(belief, hops, allowed, sites, remaining, readings):
            return choose_rollout_site(belief, hops, allowed, sites, remaining, readings, innovations=innovations)

        rows = []
        for index, start, mode, shift, truth, sensor in simulate(design, job):
            for arm, prior in zip(ARMS, priors):
                route = fixed[start].sites if arm == "gp_rollout" else None
                sites, walk, _, readings, states, forecasts, seconds = run_episode(
                    prior, design, start, EDGES, sensor.__getitem__, fixed=route, planner=planner)
                actual = states[-1][0][design.heldout] - truth[design.heldout]
                mu, sd = np.asarray(forecasts).T
                rows.append(dict(geometry=job.geometry, layout=job.layout_id, field=shift, case=index,
                    arm=arm, start=start, mode=mode, rmse=float(np.sqrt(np.mean(actual**2))), readings=len(sites),
                    edges=len(walk)-1, rooms=len(np.unique(layout.room[sites])),
                    coverage=float(np.mean(np.abs(np.asarray(readings[1:])-mu) <= 1.96*sd)),
                    interval_width=float(np.mean(3.92*sd)),
                    plan_ms=fixed[start].planning_ms if arm == "gp_rollout" else seconds*1000))
        return rows


def summarize(rows):
    rng = np.random.default_rng(295111)
    groups = defaultdict(dict)
    for r in rows:
        condition = r["geometry"], r["field"]
        groups[condition][r["layout"], r["case"], r["arm"]] = r
    result = []
    for (geometry, field), group in sorted(groups.items()):
        keys = sorted((layout, case) for layout, case, arm in group if arm == ARMS[0])
        metrics = ("rmse", "readings", "edges", "rooms", "coverage", "interval_width", "plan_ms")
        means = {arm: {metric: float(np.mean([group[(*key, arm)][metric] for key in keys]))
                       for metric in metrics} for arm in ARMS}
        strata = defaultdict(list)
        for i, key in enumerate(keys):
            r = group[(*key, ARMS[0])]
            strata[key[0], r["start"]].append(i)
        draws = np.concatenate([rng.choice(idx, (6000, len(idx))) for idx in strata.values()], axis=1)
        scores = {arm: np.asarray([group[(*key, arm)]["rmse"] for key in keys]) for arm in ARMS}
        effects = {}
        for a, b in ((ARMS[1], ARMS[0]), (ARMS[2], ARMS[0]), (ARMS[2], ARMS[1])):
            delta = scores[a]-scores[b]
            effects[f"{a}_minus_{b}"] = dict(mean=float(delta.mean()),
                ci95=np.quantile(delta[draws].mean(axis=1), [.025, .975]).tolist())
        result.append(dict(geometry=geometry, field=field, n=len(keys), means=means, contrasts=effects))
    return result


def render(output, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 3.9), layout="constrained")
    for arm, label, color, offset in (("original_world", "Original world model", "#8c627a", -.13),
                                      ("diverse_world", "Diverse training", "#226e65", .13)):
        for i, row in enumerate(rows):
            effect = row["contrasts"][f"{arm}_minus_gp_rollout"]
            y = i + offset
            ax.plot(effect["ci95"], [y, y], color=color, lw=2)
            ax.plot(effect["mean"], y, "o", color=color, label=label if i == 0 else None)
    labels = {
        ("benchmark", "benchmark"): "Standard walls · benchmark fields",
        ("benchmark", "shifted"): "Standard walls · shifted fields",
        ("unseen", "benchmark"): "New walls · benchmark fields",
        ("unseen", "shifted"): "New walls · shifted fields",
    }
    ax.set(yticks=range(len(rows)), yticklabels=[labels[r["geometry"], r["field"]] for r in rows],
           xlabel="World model minus GP rollout holdout RMSE", title="Generalization at 26 walking steps and 16 readings")
    ax.axvline(0, color=".55", lw=1)
    ax.grid(axis="x", alpha=.2)
    ax.invert_yaxis()
    ax.legend(frameon=False, loc="best")
    fig.savefig(output / "generalization.png", dpi=170)
    plt.close(fig)


def run(reference, output, count, workers):
    if count < 16 or count % 16:
        raise ValueError("Use a multiple of 16 fields per cell and layout to balance the four modes and starts")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Use a fresh output directory")
    settings, digest = load_settings(reference)
    gp_config = settings[CAP]
    with threadpool_limits(limits=1):
        weights = [learn(), train_diverse()]
    protocol = dict(reference=digest, gp=gp_config, train_geometry=TRAIN_GEOMETRIES, test_geometry=TEST_GEOMETRIES,
        standard_doors=STANDARD_DOORS, starts=STARTS, train_seed=TRAIN_SEED, test_seed=TEST_SEED, modes=MODES,
        fields_per_layout=count, training_draws_per_mode=DRAWS_PER_MODE, caps=[CAP, EDGES],
        weights={arm: w.tolist() for arm, w in zip(ARMS[1:], weights)})
    (output/"frozen.json").write_text(json.dumps(protocol, indent=2)+"\n")
    jobs = [Job(g, i, count) for g in ("benchmark", "unseen") for i in range(len(TEST_GEOMETRIES))]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(evaluate_layout, job, gp_config, weights): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            rows.extend(future.result())
            print(f"Verified {job.geometry} layout {job.layout_id + 1}", flush=True)
    rows.sort(key=lambda r: (r["geometry"], r["layout"], r["field"], r["case"], r["arm"]))
    write_csv(output/"metrics.csv", rows)
    result = summarize(rows)
    (output/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    render(output, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, default=Path("outputs/wall-topology-study/frozen.json"))
    p.add_argument("--output", type=Path, default=Path("outputs/sensing-world-generalization"))
    p.add_argument("--fields-per-layout", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    for r in run(args.reference, args.output, args.fields_per_layout, args.workers):
        print(r["geometry"], r["field"], {a: round(m["rmse"], 4) for a, m in r["means"].items()})


if __name__ == "__main__":
    main()
