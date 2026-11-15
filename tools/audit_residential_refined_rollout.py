"""Compare greedy and remaining-budget rollout on one shared refined mapper."""

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits

from audit_residential_grid_resolution import (
    DRAWS, GROUPS, PLANNER_ARMS, SEED_FIELDS, expand_walk, field_factors, holdouts, predictions,
    refine_geometry, refined_mapper, route_resources,
)
from audit_residential_holdout_distance import CAP, FAMILY
from audit_residential_routes import CONDUCTANCES, LayoutCase, STARTS, floor_plan
from audit_residential_zone_ranges import GLOBAL_WEIGHTS
from spatial_exploration.baseline import GPBelief, write_csv
from spatial_exploration.lookahead import mean_variance_reduction
from spatial_exploration.rollout import plan_rollout
from spatial_exploration.sensing_world_study import NOISE
from spatial_exploration.wall_rollout_study import plan_route


ARM_ROLLOUT = "fine_rollout"
ARMS = (*PLANNER_ARMS, ARM_ROLLOUT)
MATCHED_ARMS = dict(zip(PLANNER_ARMS, ("rollout_at_coarse_distance", "rollout_at_fine_distance")))
SCORED_ARMS = (*ARMS, *MATCHED_ARMS.values())
SCENARIOS = ("short", "mixed", "long")
SHORTLIST_PER_SCORE = 8


def full_budget_route(layout, allowed, prior, start, distance_cap):
    belief = GPBelief(prior, NOISE)
    belief.observe(start, 0.0)
    sites = [start]
    visited = np.zeros_like(allowed)
    visited[start] = True
    travel = 0.0
    branches = 0
    while len(sites) < CAP:
        current = sites[-1]
        remaining = distance_cap - travel
        costs = layout.distances[current]
        feasible = np.flatnonzero(allowed & ~visited & (costs <= remaining + 1e-12))
        if not len(feasible):
            break
        gains = mean_variance_reduction(belief.covariance, belief.noise_variance, feasible)
        per_distance = np.argsort(-(gains / costs[feasible]), kind="stable")[:SHORTLIST_PER_SCORE]
        absolute = np.argsort(-gains, kind="stable")[:SHORTLIST_PER_SCORE]
        shortlist = feasible[np.union1d(per_distance, absolute)]
        plan = plan_rollout(belief.covariance, layout.distances, current, visited, allowed,
                            remaining, belief.noise_variance, remaining_observations=CAP-len(sites),
                            first_move_candidates=shortlist)
        action = plan.first
        branches += plan.candidates_evaluated
        travel += costs[action]
        sites.append(action)
        visited[action] = True
        belief.observe(action, 0.0)
    return sites, float(np.trace(belief.covariance) / len(allowed)), branches


def plan_routes(variant, layout, anchors, held, prior, archive, edge_cap):
    base_allowed = ~np.isin(np.arange(len(layout.room)), held)
    coarse_allowed = base_allowed & np.isin(np.arange(len(layout.room)), anchors)
    distance_cap = edge_cap / (layout.size - 1)
    planner = SimpleNamespace(prior_covariance=prior, noise_std=NOISE)
    rows = []
    for start in STARTS:
        start_node = int(anchors[start])
        terminal = {}
        for arm in ARMS:
            allowed = coarse_allowed if arm == PLANNER_ARMS[0] else base_allowed
            begin = perf_counter()
            if arm == ARM_ROLLOUT:
                sites, variance, branches = full_budget_route(layout, allowed, prior, start_node, distance_cap)
            else:
                design = SimpleNamespace(layout=layout, heldout=~allowed)
                route = plan_route(design, planner, start_node, CAP, distance_cap, "greedy")
                sites, variance, branches = route.sites, route.final_variance, None
            elapsed = (perf_counter() - begin) * 1000
            sites = np.asarray(sites)
            walk = expand_walk(layout, sites)
            costs = np.rint(layout.distances[sites[:-1], sites[1:]] * (layout.size-1)).astype(int)
            resources = route_resources(layout, walk, sites, anchors, edge_cap)
            if not allowed[sites].all() or sites[0] != start_node or costs.sum() != resources["edges"]:
                raise ValueError("Invalid start, measurement mask, or route resources")
            if arm in PLANNER_ARMS:
                old = archive[variant, start, arm]
                np.testing.assert_array_equal(sites, old["sites"])
                np.testing.assert_array_equal(walk, old["walk"])
            nearest = layout.distances[np.ix_(held, sites)].min(axis=1) * (layout.size-1)
            terminal[arm] = variance
            rows.append(dict(variant=variant, start=start, arm=arm, sites=sites.tolist(), walk=walk.tolist(),
                             plan_ms=elapsed, branches=branches, terminal_variance=variance, **resources,
                             adjacent_stop_fraction=float(np.mean(costs == 1)),
                             zones=int(len(set(layout.room[sites]))), mean_holdout_distance=float(nearest.mean())))
        if terminal[ARM_ROLLOUT] > terminal[PLANNER_ARMS[1]] + 1e-10:
            raise ValueError("Rollout regressed on its shared modeled objective")
        by_arm = {r["arm"]: r for r in rows if r["start"] == start}
        rollout = by_arm[ARM_ROLLOUT]
        chosen = np.asarray(rollout["sites"])
        distances = layout.distances[chosen[:-1], chosen[1:]]
        arrival = np.r_[0, np.rint(distances * (layout.size-1)).cumsum()]
        for baseline, arm in MATCHED_ARMS.items():
            horizon = by_arm[baseline]["edges"]
            if horizon > rollout["edges"]:
                raise ValueError("Rollout ended before the matched walking distance")
            retained = chosen[arrival <= horizon]
            prefix = np.asarray(rollout["walk"][:horizon+1])
            resources = route_resources(layout, prefix, retained, anchors, edge_cap)
            if resources["edges"] != horizon:
                raise ValueError("Wrong matched walking distance")
            rows.append(dict(variant=variant, start=start, arm=arm, sites=retained.tolist(),
                             walk=prefix.tolist(), **resources))
        print(f"Planned layout {variant}, start {start}", flush=True)
    return rows


def field_key(row):
    return tuple(str(row[k]) for k in ("variant", "regime", "start", "profile", "draw"))


def field_batches(variant, layout, profiles, anchored_curve):
    rng = np.random.default_rng(SEED_FIELDS + 10 + variant)
    uniform_rng = np.random.default_rng(SEED_FIELDS + 100 + variant)
    mixed_profiles = profiles[len(GLOBAL_WEIGHTS):]
    true_profiles = np.vstack((mixed_profiles, np.ones_like(profiles[0]), np.zeros_like(profiles[0])))
    cases = ["mixed"] * len(mixed_profiles) + ["short", "long"]
    for regime, conductance in CONDUCTANCES.items():
        factors = field_factors(layout, true_profiles, conductance, anchored_curve)
        for start in STARTS:
            for profile, (scenario, factor) in enumerate(zip(cases, factors)):
                random = rng if scenario == "mixed" else uniform_rng
                truth = random.standard_normal((DRAWS, len(layout.room))) @ factor.T
                noisy = truth + NOISE * random.standard_normal(truth.shape)
                yield regime, start, profile, scenario, truth, noisy


def evaluate(variant, model, archive, previous, edge_cap):
    coarse = floor_plan(LayoutCase(FAMILY, variant))
    layout, anchors = refine_geometry(coarse)
    held, n_shared = holdouts(coarse, layout, anchors, variant)
    profiles, kernels, prior, anchored_curve = refined_mapper(coarse, layout, anchors, model)
    log_prior = np.log(prior)
    plan_prior = np.einsum("m,mij->ij", prior, kernels)
    routes = plan_routes(variant, layout, anchors, held, plan_prior, archive, edge_cap)
    site_sets = {(r["start"], r["arm"]): np.asarray(r["sites"]) for r in routes}
    rows = []
    for regime, start, profile, scenario, truth, noisy in field_batches(variant, layout, profiles, anchored_curve):
        metrics = {}
        for arm in SCORED_ARMS:
            sites = site_sets[start, arm]
            mean, sd = predictions(kernels, sites, noisy[:, sites], held, log_prior)
            error = mean - truth[:, held]
            metrics[f"{arm}_all"] = np.sqrt(np.mean(error**2, axis=1))
            for name, subset in zip(GROUPS, (slice(0, n_shared), slice(n_shared, None))):
                metrics[f"{arm}_{name}"] = np.sqrt(np.mean(error[:, subset]**2, axis=1))
            metrics[f"{arm}_coverage"] = np.mean(abs(error) <= 1.96*sd, axis=1)
            metrics[f"{arm}_width"] = np.mean(3.92*sd, axis=1)
        for draw in range(DRAWS):
            row = dict(variant=variant, regime=regime, start=start, profile=profile, draw=draw,
                       scenario=scenario, **{name: values[draw] for name, values in metrics.items()})
            if scenario == "mixed":
                reference = previous[field_key(row)]
                keys = [f"{arm}_{group}" for arm in PLANNER_ARMS for group in GROUPS]
                np.testing.assert_allclose([row[k] for k in keys], [float(reference[k]) for k in keys],
                                           atol=1e-10, rtol=0)
            rows.append(row)
    return rows, routes


def summarize(rows, routes):
    rng = np.random.default_rng(SEED_FIELDS + 140)
    result = {}
    for scenario in SCENARIOS:
        paired = [row for row in rows if row["scenario"] == scenario]
        strata = {}
        for i, row in enumerate(paired):
            key = tuple(row[k] for k in ("variant", "regime", "start", "profile"))
            strata.setdefault(key, []).append(i)
        boot = np.concatenate([rng.choice(ids, (3000, len(ids))) for ids in strata.values()], axis=1)
        metrics = ("all", *GROUPS, "coverage", "width")
        values = {arm: {m: np.asarray([r[f"{arm}_{m}"] for r in paired]) for m in metrics} for arm in SCORED_ARMS}
        means = {arm: {m: float(v.mean()) for m, v in d.items()} for arm, d in values.items()}
        contrasts, matched = {}, {}
        for baseline in PLANNER_ARMS:
            contrasts[baseline] = {}
            matched[baseline] = {}
            for metric in ("all", *GROUPS):
                for arm, report in ((ARM_ROLLOUT, contrasts), (MATCHED_ARMS[baseline], matched)):
                    delta = values[arm][metric] - values[baseline][metric]
                    report[baseline][metric] = dict(mean=float(delta.mean()),
                                                    ci95=np.quantile(delta[boot].mean(axis=1), [.025, .975]).tolist())
        result[scenario] = dict(fields=len(paired), means=means, rollout_minus=contrasts,
                                matched_travel_rollout_minus=matched)
    resources = {}
    for arm in ARMS:
        selected = [row for row in routes if row["arm"] == arm]
        keys = ("edges", "readings", "new_sites", "plan_ms", "terminal_variance",
                "adjacent_stop_fraction", "zones", "mean_holdout_distance")
        resources[arm] = {k: dict(mean=float(np.mean([r[k] for r in selected])),
                                  median=float(np.median([r[k] for r in selected])),
                                  range=[min(r[k] for r in selected), max(r[k] for r in selected)]) for k in keys}
    for arm in MATCHED_ARMS.values():
        selected = [row for row in routes if row["arm"] == arm]
        resources[arm] = {k: dict(mean=float(np.mean([r[k] for r in selected])),
                                  range=[min(r[k] for r in selected), max(r[k] for r in selected)])
                          for k in ("edges", "readings")}
    return dict(scenarios=result, routes=resources)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("outputs/sensing-world-residential-grid-resolution"))
    parser.add_argument("--output", type=Path, default=Path("outputs/sensing-world-residential-refined-rollout"))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use an empty output directory.")
    config = json.loads((args.source / "protocol.json").read_text())
    model = json.loads(Path(config["model"]).read_text())
    archive = {(row["variant"], row["start"], row["arm"]): row
               for row in json.loads((args.source / "routes.json").read_text()) if row["arm"] in PLANNER_ARMS}
    with (args.source / "fields.csv").open() as stream:
        baseline = {field_key(row): row for row in csv.DictReader(stream)}
    rows, routes = [], []
    with threadpool_limits(limits=1):
        for variant in sorted({v for v, _, _ in archive}):
            fields, plans = evaluate(variant, model, archive, baseline, config["fine_edge_cap"])
            rows.extend(fields)
            routes.extend(plans)
    if {field_key(row) for row in rows if row["scenario"] == "mixed"} != set(baseline):
        raise ValueError("Replay differs from the archived mixed fields")
    summary = summarize(rows, routes)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "fields.csv", rows)
    protocol = dict(source=str(args.source), model=config["model"], fine_size=config["fine_size"],
                    fine_edge_cap=config["fine_edge_cap"], cap=CAP, shortlist_per_score=SHORTLIST_PER_SCORE,
                    score="Final whole-grid variance under a shared fixed prior after greedy completions")
    for name, payload in (("protocol", protocol), ("routes", routes), ("summary", summary)):
        (args.output / f"{name}.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    for scenario, result in summary["scenarios"].items():
        print(scenario, result)


if __name__ == "__main__":
    main()
