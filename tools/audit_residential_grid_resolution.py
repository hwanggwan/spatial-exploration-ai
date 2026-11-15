"""Separate reconstruction and measurement placement at finer corridor resolution."""

import argparse
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from scipy.sparse.csgraph import shortest_path
from threadpoolctl import threadpool_limits

from audit_residential_holdout_distance import CAP, FAMILY
from audit_residential_routes import AREAS, CONDUCTANCES, LayoutCase, SEED, STARTS, design_for, floor_plan
from audit_residential_zone_ranges import (
    GLOBAL_WEIGHTS, load_archive, local_mode_prior, map_predictions, model_covariances, range_profiles, weighted_covariance,
)
from audit_sensing_world_modes import posterior
from audit_sensing_world_shifts import correlation_area, match_scale, within_room_correlogram
from spatial_exploration.barrier_gp import RoomLayout, graph_spectrum, unit_covariance
from spatial_exploration.baseline import make_grid, write_csv
from spatial_exploration.sensing_world_study import NOISE
from spatial_exploration.wall_rollout_study import plan_route


SEED_FIELDS = 312181
DRAWS = 12
OFFGRID_HOLDOUT_PER_ZONE = 18
PLANNER_ARMS = ("coarse_candidates", "fine_candidates")
PREFIX_ARMS = ("coarse_prefix", "fine_prefix")
ARMS = ("coarse_fixed", "fine_fixed", *PLANNER_ARMS, *PREFIX_ARMS)
GROUPS = ("shared", "offgrid")


def refine_geometry(coarse):
    size = 2 * coarse.size - 1
    r, c = np.indices((size, size)).reshape(2, -1)
    containing_cell = ((r + 1) // 2) * coarse.size + (c + 1) // 2
    shared = (r % 2 == 0) & (c % 2 == 0)
    anchors = np.flatnonzero(shared)
    adjacent = (abs(r[:, None] - r) + abs(c[:, None] - c)) == 1
    walls = adjacent & coarse.wall_edges[containing_cell[:, None], containing_cell]
    movement = adjacent & ~walls
    edges, predecessors = shortest_path(movement, directed=False, unweighted=True, return_predecessors=True)
    if not np.isfinite(edges).all():
        raise ValueError("Refined floor plan must be connected")
    fine = RoomLayout(size, coarse.door_row, make_grid(size), coarse.room[containing_cell],
                      walls, movement, edges / (size-1), predecessors)
    np.testing.assert_allclose(fine.points[anchors], coarse.points, atol=1e-15, rtol=0)
    np.testing.assert_array_equal(fine.room[anchors], coarse.room)
    for a, b in zip(*np.where(np.triu(coarse.movement, 1))):
        ia, ib = anchors[[a, b]]
        center = (ia + ib) // 2
        if not (fine.movement[ia, center] and fine.movement[center, ib]):
            raise ValueError("An original walking edge was lost")
    return fine, anchors


def interpolation_matrix(coarse, fine, points):
    r, c = np.divmod(points, fine.size)
    x, y = c / 2, r / 2
    left = np.floor(x).astype(int)
    bottom = np.floor(y).astype(int)
    right = np.minimum(left + 1, coarse.size - 1)
    top = np.minimum(bottom + 1, coarse.size - 1)
    dx, dy = np.clip(x-left, 0, 1), np.clip(y-bottom, 0, 1)
    ids = np.column_stack(tuple(row * coarse.size + col
                                for row, col in ((bottom, left), (bottom, right), (top, left), (top, right))))
    weights = np.column_stack(((1-dx)*(1-dy), dx*(1-dy), (1-dx)*dy, dx*dy))
    weights *= coarse.room[ids] == fine.room[points, None]
    result = np.zeros((len(points), len(coarse.points)))
    np.add.at(result, (np.arange(len(points))[:, None], ids), weights)
    return result / result.sum(axis=1, keepdims=True)


def holdouts(coarse, fine, anchors, variant):
    shared = anchors[np.flatnonzero(design_for(coarse, SEED + variant).heldout)]
    rng = np.random.default_rng(SEED_FIELDS + variant)
    offgrid = ~np.isin(np.arange(len(fine.room)), anchors)
    extra = np.concatenate([rng.choice(np.flatnonzero(offgrid & (fine.room == zone)),
                                       OFFGRID_HOLDOUT_PER_ZONE, replace=False)
                            for zone in np.unique(fine.room)])
    return np.concatenate((shared, extra)), len(shared)


def field_factors(fine, profiles, conductance, correlogram):
    eigenvalues, eigenvectors = graph_spectrum(fine, conductance)

    def resolvent(scale):
        return unit_covariance(eigenvectors, (scale**2 + eigenvalues)**-2)

    short, long = [match_scale(resolvent, correlogram, correlation_area, area)[1] for area in AREAS]
    return [np.linalg.cholesky(weighted_covariance(short, long, p[fine.room])) for p in profiles]


def predictions(kernels, sites, readings, targets, log_prior):
    component_means, likelihoods, variances = posterior(kernels, sites, readings, targets, with_variance=True)
    mean, sd, _ = map_predictions(component_means, likelihoods, variances, log_prior)
    return mean, sd


def expand_walk(layout, sites):
    walk = [int(sites[0])]
    for a, b in zip(sites, sites[1:]):
        walk.extend(layout.walk(a, b)[1:])
    return np.asarray(walk)


def route_resources(layout, walk, sites, anchors, limit):
    walked = len(walk) - 1
    if (walked > limit or len(sites) > CAP or len(sites) != len(set(sites))
            or not layout.movement[walk[:-1], walk[1:]].all()):
        raise ValueError("Invalid route resources")
    return dict(edges=walked, readings=len(sites), new_sites=int(np.count_nonzero(~np.isin(sites, anchors))))


def refined_mapper(coarse, fine, anchors, model):
    coarse_curve = within_room_correlogram(coarse)

    def anchored_curve(covariance):
        return coarse_curve(covariance[np.ix_(anchors, anchors)])

    profiles = range_profiles(int(coarse.room.max()) + 1)
    bank = model_covariances(fine, model, profiles, correlogram=anchored_curve)
    kernels = bank.reshape(-1, len(fine.room), len(fine.room))
    return profiles, kernels, local_mode_prior(len(profiles), bank.shape[1]), anchored_curve


def evaluate_layout(variant, config, model, archive):
    rng = np.random.default_rng(SEED_FIELDS + 10 + variant)
    coarse = floor_plan(LayoutCase(FAMILY, variant))
    fine, anchors = refine_geometry(coarse)
    held, n_shared = holdouts(coarse, fine, anchors, variant)
    groups = (slice(0, n_shared), slice(n_shared, None))
    interp = interpolation_matrix(coarse, fine, held)
    coarse_hold = np.searchsorted(anchors, held[:n_shared])
    np.testing.assert_array_equal(interp[:n_shared], np.eye(len(anchors))[coarse_hold])

    profiles, kernels_fine, prior, anchored_curve = refined_mapper(coarse, fine, anchors, model)
    coarse_bank = model_covariances(coarse, model, profiles)
    kernels_coarse = coarse_bank.reshape(-1, len(anchors), len(anchors))
    log_prior = np.log(prior)
    planning_prior = np.einsum("m,mij->ij", prior, kernels_fine)
    planner = SimpleNamespace(prior_covariance=planning_prior, noise_std=NOISE)
    edge_limit = 2 * config["edges"]
    distance = edge_limit / (fine.size - 1)
    base_allowed = ~np.isin(np.arange(len(fine.room)), held)
    mixed_profiles = profiles[len(GLOBAL_WEIGHTS):]
    rows, saved_routes = [], []
    sample = None

    for start in STARTS:
        archived = np.asarray(archive[variant, start]["sites"])
        fixed = anchors[archived]
        replanned = {}
        for name, allowed in (("coarse_candidates", base_allowed & np.isin(np.arange(len(fine.room)), anchors)),
                              ("fine_candidates", base_allowed)):
            design = SimpleNamespace(layout=fine, heldout=~allowed)
            route = plan_route(design, planner, anchors[start], CAP, distance, "greedy")
            replanned[name] = np.asarray(route.sites)
        site_sets = dict(fixed=fixed, **replanned)
        for arm, sites in site_sets.items():
            if not base_allowed[sites].all() or sites[0] != anchors[start]:
                raise ValueError("Wrong start or hold-out contamination")
            if arm == "fixed":
                coarse_walk = anchors[expand_walk(coarse, archived)]
                walk = np.empty(coarse_walk.size * 2 - 1, dtype=int)
                walk[::2] = coarse_walk
                walk[1::2] = (coarse_walk[:-1] + coarse_walk[1:]) // 2
            else:
                walk = expand_walk(fine, sites)
            resources = route_resources(fine, walk, sites, anchors, edge_limit)
            if arm == "fixed" and resources["edges"] != edge_limit:
                raise ValueError("Archived route changed its travel distance")
            nearest = fine.distances[np.ix_(held, sites)].min(axis=1)
            saved_routes.append(dict(variant=variant, start=start, arm=arm,
                                     sites=sites.tolist(), walk=walk.tolist(),
                                     zones=int(len(set(fine.room[sites]))),
                                     nearest_holdout_coarse_steps=float(nearest.mean() * (coarse.size - 1)),
                                     **resources))
        planner_routes = {r["arm"]: r for r in saved_routes if r["start"] == start and r["arm"] in PLANNER_ARMS}
        horizon = min(r["edges"] for r in planner_routes.values())
        for original, prefix in zip(PLANNER_ARMS, PREFIX_ARMS):
            route = planner_routes[original]
            sites = np.asarray(route["sites"])
            arrival = np.r_[0, np.rint(fine.distances[sites[:-1], sites[1:]] * (fine.size - 1)).cumsum()]
            np.testing.assert_allclose(arrival[-1], route["edges"], rtol=0, atol=1e-10)
            sites = sites[arrival <= horizon]
            walk = np.asarray(route["walk"][:horizon+1])
            nearest = fine.distances[np.ix_(held, sites)].min(axis=1)
            resources = route_resources(fine, walk, sites, anchors, horizon)
            saved_routes.append(dict(variant=variant, start=start, arm=prefix,
                                     sites=sites.tolist(), walk=walk.tolist(),
                                     zones=int(len(set(fine.room[sites]))),
                                     nearest_holdout_coarse_steps=float(nearest.mean() * (coarse.size - 1)),
                                     **resources))

    for regime, conductance in CONDUCTANCES.items():
        factors = field_factors(fine, mixed_profiles, conductance, anchored_curve)
        for start in STARTS:
            records = {r["arm"]: np.asarray(r["sites"]) for r in saved_routes if r["start"] == start}
            fixed = records["fixed"]
            coarse_sites = np.searchsorted(anchors, fixed)
            np.testing.assert_array_equal(anchors[coarse_sites], fixed)
            for profile, factor in enumerate(factors):
                truth = rng.standard_normal((DRAWS, len(fine.room))) @ factor.T
                noisy = truth + NOISE * rng.standard_normal(truth.shape)
                before = perf_counter()
                low_mean, low_sd = predictions(kernels_coarse, coarse_sites, noisy[:, fixed],
                                               np.arange(len(anchors)), log_prior)
                predicted = {"coarse_fixed": (low_mean @ interp.T, low_sd[:, coarse_hold])}
                times = {"coarse_fixed": (perf_counter()-before) * 1000 / DRAWS}
                fine_observations = dict(fine_fixed=fixed, **{arm: records[arm] for arm in ARMS[2:]})
                for arm, sites in fine_observations.items():
                    before = perf_counter()
                    predicted[arm] = predictions(kernels_fine, sites, noisy[:, sites], held, log_prior)
                    times[arm] = (perf_counter()-before) * 1000 / DRAWS

                metrics = {}
                for arm, (mean, sd) in predicted.items():
                    error = mean - truth[:, held]
                    for name, subset in zip(GROUPS, groups):
                        metrics[f"{arm}_{name}"] = np.sqrt(np.mean(error[:, subset]**2, axis=1))
                    metrics[f"{arm}_shared_coverage"] = np.mean(abs(error[:, :n_shared]) <= 1.96*sd[:, :n_shared], axis=1)
                for draw in range(DRAWS):
                    rows.append(dict(variant=variant, regime=regime, start=start, profile=profile, draw=draw,
                                     **{name: values[draw] for name, values in metrics.items()},
                                     **{f"{name}_ms": value for name, value in times.items()}))
                if (variant, regime, start, profile) == (1, "weak", STARTS[0], 0):
                    fine_nodes = np.arange(len(fine.room))
                    def full(arm):
                        sites = records[arm]
                        return predictions(kernels_fine, sites, noisy[:1, sites], fine_nodes, log_prior)[0][0]

                    walks = {r["arm"]: r["walk"] for r in saved_routes if r["start"] == start}
                    sample = dict(layout=fine, held=held, truth=truth[0],
                                  coarse_fixed=interpolation_matrix(coarse, fine, fine_nodes) @ low_mean[0],
                                  fine_fixed=full("fixed"), coarse_candidates=full("coarse_candidates"),
                                  fine_candidates=full("fine_candidates"), routes=records, walks=walks)
                    for arm in ("coarse_fixed", "fine_fixed", *PLANNER_ARMS):
                        np.testing.assert_allclose(sample[arm][held], predicted[arm][0][0], atol=1e-12, rtol=0)
    return rows, saved_routes, sample


def summarize(rows, routes):
    rng = np.random.default_rng(SEED_FIELDS + 30)
    strata = {}
    for i, row in enumerate(rows):
        strata.setdefault(tuple(row[k] for k in ("variant", "regime", "start", "profile")), []).append(i)
    boot = np.concatenate([rng.choice(ids, (3000, len(ids))) for ids in strata.values()], axis=1)
    means, differences = {}, {}
    for group in GROUPS:
        mags = {arm: np.array([r[f"{arm}_{group}"] for r in rows]) for arm in ARMS}
        means[group] = {arm: float(scores.mean()) for arm, scores in mags.items()}
        differences[group] = {}
        for name, first, second in (("finer_mapping", "coarse_fixed", "fine_fixed"),
                                    ("more_candidate_sites", *PLANNER_ARMS),
                                    ("equal_realized_travel", *PREFIX_ARMS)):
            delta = mags[second] - mags[first]
            differences[group][name] = dict(mean=float(delta.mean()),
                                             ci95=np.quantile(delta[boot].mean(axis=1), [.025, .975]).tolist())
    resources = {}
    for arm in ("fixed", *PLANNER_ARMS, *PREFIX_ARMS):
        selected = [row for row in routes if row["arm"] == arm]
        resources[arm] = {key: dict(mean=float(np.mean([row[key] for row in selected])),
                                    range=[min(row[key] for row in selected), max(row[key] for row in selected)])
                          for key in ("readings", "edges", "new_sites", "zones", "nearest_holdout_coarse_steps")}
    coverage = {arm: float(np.mean([r[f"{arm}_shared_coverage"] for r in rows])) for arm in ARMS}
    timings = {arm: float(np.mean([r[f"{arm}_ms"] for r in rows])) for arm in ARMS}
    return dict(fields=len(rows), means=means, paired_differences=differences, resources=resources,
                shared_coverage=coverage, map_ms=timings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("outputs/sensing-world-residential-oracle-budget"))
    parser.add_argument("--output", type=Path, default=Path("outputs/sensing-world-residential-grid-resolution"))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use an empty output directory.")
    config, model, archive = load_archive(args.source)
    rows, routes = [], []
    sample = None
    with threadpool_limits(limits=1):
        for variant in sorted({variant for variant, _ in archive}):
            fields, layouts, example = evaluate_layout(variant, config, model, archive)
            rows.extend(fields)
            routes.extend(layouts)
            if example is not None:
                sample = example
    result = summarize(rows, routes)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "fields.csv", rows)
    protocol = dict(source=str(args.source), model=config["model"], coarse_size=12, fine_size=23, cap=CAP,
                    fine_edge_cap=2*config["edges"], seed=SEED_FIELDS, draws=DRAWS,
                    planner="Fixed-prior covariance-reduction-per-distance greedy on the refined movement graph")
    for name, payload in (("protocol", protocol), ("summary", result), ("routes", routes)):
        (args.output / f"{name}.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    if sample is not None:
        from plot_residential_grid_resolution import render
        render(args.output / "grid-resolution-maps.png", sample)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
