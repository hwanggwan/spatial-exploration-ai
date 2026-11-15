"""Test observation-driven local range weights in refined-grid rollout."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from functools import partial
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.linalg import solve_triangular
from scipy.linalg.blas import dger
from scipy.special import softmax
from threadpoolctl import threadpool_limits

from audit_residential_grid_resolution import DRAWS, expand_walk, holdouts, predictions, refine_geometry, refined_mapper, route_resources
from audit_residential_holdout_distance import CAP, FAMILY
from audit_residential_refined_rollout import SCENARIOS, SHORTLIST_PER_SCORE, field_batches, field_key, remaining_variance
from audit_residential_room_rollout import EARLY_WEIGHT, ROOMS
from audit_residential_routes import LayoutCase, STARTS, floor_plan
from audit_residential_zone_ranges import GLOBAL_WEIGHTS
from spatial_exploration.baseline import GPBelief, write_csv
from spatial_exploration.sensing_world_study import NOISE


EDGES = 52
ARMS = ("quarter", "room", "adaptive")
SHORT_BONUS = 4.0
PRIORITY_ARM = "short_priority"
ORACLE_ARM = "short_oracle"
MATCHED_ARM = "short_oracle_matched"
ADAPTIVE_OUTPUT = Path("outputs/sensing-world-residential-adaptive-rollout")
PRIORITY_OUTPUT = Path("outputs/sensing-world-residential-short-priority-rollout")
ORACLE_OUTPUT = Path("outputs/sensing-world-residential-short-oracle-rollout")
MATCHED_OUTPUT = Path("outputs/sensing-world-residential-matched-oracle-rollout")
METRICS = ("rmse", "worst_room_rmse", "edges", "readings", "short_readings", "short_share", "short_rooms",
           "short_rmse", "range_gap", "range_mae", "coverage", "interval_width")


def profile_weights(kernels, sites, values, log_prior, n_profiles):
    """Marginalize wall modes after scoring the observed values in each joint mode."""
    observed = kernels[:, sites[:, None], sites] + NOISE**2 * np.eye(len(sites))
    factor = np.linalg.cholesky(observed)
    whitened = np.linalg.solve(factor, values[None, :, None])[..., 0]
    likelihood = -0.5 * np.sum(whitened**2, axis=1) - np.log(factor.diagonal(axis1=1, axis2=2)).sum(axis=1)
    return softmax(log_prior + likelihood).reshape(n_profiles, -1).sum(axis=1)


def conditional_covariance(prior, sites):
    factor = np.linalg.cholesky(prior[np.ix_(sites, sites)] + NOISE**2 * np.eye(len(sites)))
    cross = solve_triangular(factor, prior[sites], lower=True, check_finite=False)
    return prior - cross.T @ cross


def rollout_move(covariance, distances, current, candidates, remaining, remaining_readings, distance_cap,
                 *, target_weights=None, can_complete=None):
    """Run the refined rollout with row updates restricted to still-affordable sites."""
    costs = distances[current, candidates]
    rows = covariance[candidates]
    target_scale = np.ones(len(covariance))
    if target_weights is not None:
        # These columns contribute to weighted squared covariance in each gain.
        target_scale = np.sqrt(len(covariance) * target_weights)
        rows *= target_scale
    gains = np.mean(rows**2, axis=1) / (covariance.diagonal()[candidates] + NOISE**2)
    per_distance = np.argsort(-(gains / costs), kind="stable")
    absolute = np.argsort(-gains, kind="stable")
    if can_complete is not None:
        legal = can_complete(current, candidates, remaining, remaining_readings, ())
        per_distance = per_distance[legal[per_distance]]
        absolute = absolute[legal[absolute]]
        if not len(per_distance):
            raise RuntimeError("No next measurement can complete the required observations")
    shortlist = np.union1d(per_distance[:SHORTLIST_PER_SCORE], absolute[:SHORTLIST_PER_SCORE])
    greedy = per_distance[0]
    best_score, best_first = np.inf, None
    for first in np.r_[greedy, shortlist[shortlist != greedy]]:
        active, active_rows = candidates, rows
        action_index, depart, travel = first, current, 0.0
        suffix = []
        while True:
            action = active[action_index]
            travel += distances[depart, action]
            suffix.append(action)
            if len(suffix) == remaining_readings:
                break
            costs = distances[action, active]
            keep = (costs > 0) & (costs <= remaining - travel + 1e-12)
            if not keep.any():
                break
            # On a shortest-path metric, an unaffordable site cannot become affordable later.
            chosen = active_rows[action_index]
            active_rows = active_rows[keep]
            active = active[keep]
            # Undo the observed column's scaling in the covariance update.
            scale = -1 / (chosen[action] + NOISE**2 * target_scale[action])
            active_rows = dger(scale, chosen, active_rows[:, action], a=active_rows.T, overwrite_a=1).T
            variances = active_rows[np.arange(len(active)), active] / target_scale[active]
            gain = np.einsum("ij,ij->i", active_rows, active_rows) / (variances + NOISE**2)
            density = gain / costs[keep]
            if can_complete is not None:
                legal = can_complete(action, active, remaining - travel, remaining_readings - len(suffix), suffix)
                if not legal.any():
                    break
                density[~legal] = -np.inf
            action_index = np.argmax(density)
            depart = action
        terminal, integral = remaining_variance(covariance, distances, current, suffix, NOISE**2, remaining, distance_cap,
                                                target_weights=target_weights)
        score = (1 - EARLY_WEIGHT) * terminal + EARLY_WEIGHT * integral
        if score < best_score - 1e-12:
            best_score, best_first = score, int(suffix[0])
    return best_first


def priority_weights(zone_weights, zones, *, short_zones=None):
    zone_scores = 1 + SHORT_BONUS * np.maximum(zone_weights - 0.5, 0)
    if short_zones is not None:
        # Keep the inferred scores and give the largest ones to the true short zones.
        order = np.lexsort((zone_weights, short_zones))
        zone_scores[order] = np.sort(zone_scores)
    score_weights = zone_scores[zones]
    return score_weights / score_weights.sum()


def adaptive_route(layout, origin, allowed, sensor, kernels, profiles, profile_priors, log_prior, *,
                   prioritize_short=False, oracle_profile=None, match_bonus=False):
    sites = [origin]
    available = allowed.copy()
    travel = 0.0
    distance_cap = EDGES / (layout.size - 1)
    trace = []
    values = [sensor(origin)]
    # A uniform oracle profile gives every cell the same weight.
    target_weights = None
    short_zones = None
    if oracle_profile is not None and np.ptp(oracle_profile) > 0:
        if match_bonus:
            short_zones = oracle_profile > 0.5
        else:
            target_weights = priority_weights(oracle_profile, layout.room)
    while True:
        observed = np.asarray(sites)
        weights = profile_weights(kernels, observed, np.asarray(values), log_prior, len(profiles))
        zone_weights = weights @ profiles
        trace.append(zone_weights.tolist())
        if len(sites) == CAP:
            break
        current = sites[-1]
        available[current] = False
        remaining = distance_cap - travel
        candidates = np.flatnonzero(available & (layout.distances[current] <= remaining + 1e-12))
        if not len(candidates):
            break
        prior = (weights @ profile_priors).reshape(len(allowed), -1)
        covariance = conditional_covariance(prior, observed)
        if prioritize_short or short_zones is not None:
            target_weights = priority_weights(zone_weights, layout.room, short_zones=short_zones)
        action = rollout_move(covariance, layout.distances, current, candidates, remaining, CAP-len(sites), distance_cap,
                              target_weights=target_weights)
        travel += layout.distances[current, action]
        sites.append(action)
        values.append(sensor(action))
    return np.asarray(sites), trace


def check_baseline(layout, allowed, prior, sites):
    """Check that the compact rollout reproduces every first move from the archive."""
    belief = GPBelief(prior, NOISE)
    available = allowed.copy()
    distance_cap = EDGES / (layout.size - 1)
    travel = 0.0
    for count, (current, expected) in enumerate(zip(sites[:-1], sites[1:]), start=1):
        belief.observe(current, 0.0)
        available[current] = False
        remaining = distance_cap - travel
        candidates = np.flatnonzero(available & (layout.distances[current] <= remaining + 1e-12))
        actual = rollout_move(belief.covariance, layout.distances, current, candidates, remaining, CAP-count, distance_cap)
        if actual != expected:
            raise ValueError(f"Compact rollout changed baseline move {count}: {actual} vs {expected}")
        travel += layout.distances[current, actual]


def score_route(layout, anchors, held, kernels, log_prior, target, sites, noisy, truth, estimated, plan_ms):
    walk = expand_walk(layout, sites)
    resources = route_resources(layout, walk, sites, anchors, EDGES)
    if np.isin(sites, held).any():
        raise ValueError("A rollout measured a hold-out site")
    mean, sd = predictions(kernels, sites, noisy[None, sites], held, log_prior)
    error = mean[0] - truth[held]
    short = target > 0.5
    measured_zones = layout.room[sites]
    held_zones = layout.room[held]
    short_cells = short[held_zones]
    counts = np.bincount(measured_zones, minlength=len(target))
    worst = max(np.sqrt(np.mean(error[held_zones == room]**2)) for room in ROOMS)
    metrics = dict(rmse=np.sqrt(np.mean(error**2)), worst_room_rmse=worst,
                   edges=resources["edges"], readings=resources["readings"],
                   short_readings=int(counts[short].sum()), short_share=float(short[measured_zones].mean()),
                   short_rooms=int(np.count_nonzero(counts[short])),
                   short_rmse=np.sqrt(np.mean(error[short_cells]**2)) if short_cells.any() else None,
                   range_gap=float(estimated[short].mean() - estimated[~short].mean()) if short.any() and not short.all() else None,
                   range_mae=float(abs(estimated - target).mean()), coverage=float(np.mean(abs(error) <= 1.96 * sd[0])),
                   interval_width=float(np.mean(3.92 * sd[0])), plan_ms=plan_ms)
    return metrics


def evaluate_case(case, model, archive, previous_fields, draws, active_arm, reference_fields):
    variant, selected_start = case
    with threadpool_limits(limits=1):
        return evaluate_layout(variant, selected_start, model, archive, previous_fields, draws, active_arm, reference_fields)


def evaluate_layout(variant, selected_start, model, archive, previous_fields, draws, active_arm, reference_fields):
    coarse = floor_plan(LayoutCase(FAMILY, variant))
    layout, anchors = refine_geometry(coarse)
    held, _ = holdouts(coarse, layout, anchors, variant)
    profiles, kernels, prior_weights, curve = refined_mapper(coarse, layout, anchors, model)
    log_prior = np.log(prior_weights)
    # Average over the original wall-mode prior for each range profile in the planner.
    priors_by_profile = kernels.reshape(len(profiles), -1, len(layout.room)**2).mean(axis=1)
    prior = np.einsum("m,mij->ij", prior_weights, kernels)
    allowed = ~np.isin(np.arange(len(layout.room)), held)
    origin = int(anchors[selected_start])
    reference = {arm: np.asarray(archive[variant, selected_start, arm]["sites"]) for arm in ARMS[:2]}
    check_baseline(layout, allowed, prior, reference["quarter"])
    targets = np.vstack((profiles[len(GLOBAL_WEIGHTS):], np.ones_like(profiles[0]), np.zeros_like(profiles[0])))
    rows, routes = [], []
    for regime, start, profile, scenario, truth, noisy in field_batches(variant, layout, profiles, curve):
        if start != selected_start:
            continue
        target = targets[profile]
        for draw in range(draws):
            key = dict(variant=variant, regime=regime, start=start, profile=profile, draw=draw, scenario=scenario)
            before = perf_counter()
            sites, trace = adaptive_route(layout, origin, allowed, noisy[draw].__getitem__,
                                          kernels, profiles, priors_by_profile, log_prior,
                                          prioritize_short=active_arm == PRIORITY_ARM,
                                          oracle_profile=target if active_arm in (ORACLE_ARM, MATCHED_ARM) else None,
                                          match_bonus=active_arm == MATCHED_ARM)
            plan_ms = (perf_counter() - before) * 1000
            if sites[0] != origin:
                raise ValueError("The adaptive policy changed the starting location")
            key_tuple = field_key(key)
            row = dict(key)
            score_sites = {active_arm: sites}
            if reference_fields is not None:
                cached = reference_fields[key_tuple]
                if cached["scenario"] != scenario:
                    raise ValueError("The reference field has a different range label")
                row.update({name: float(value) if value else None for name, value in cached.items() if name not in key})
                for arm in ARMS[:2]:
                    np.testing.assert_allclose(row[f"{arm}_rmse"], float(previous_fields[key_tuple][f"{arm}_map_final"]),
                                               atol=1e-10, rtol=0)
            else:
                score_sites = reference | score_sites
            for arm, observed in score_sites.items():
                estimated = np.asarray(trace[-1]) if arm == active_arm else (
                    profile_weights(kernels, observed, noisy[draw, observed], log_prior, len(profiles)) @ profiles)
                metrics = score_route(layout, anchors, held, kernels, log_prior, target, observed,
                                      noisy[draw], truth[draw], estimated, plan_ms if arm == active_arm else None)
                if arm != active_arm:
                    expected = float(previous_fields[key_tuple][f"{arm}_map_final"])
                    np.testing.assert_allclose(metrics["rmse"], expected, atol=1e-10, rtol=0)
                row.update({f"{arm}_{metric}": value for metric, value in metrics.items()})
            rows.append(row)
            routes.append(dict(key, sites=sites.tolist(), zone_range_weights=trace, true_zone_range_weights=target.tolist()))
    return rows, routes


def summarize(rows, arms=ARMS):
    active_arm = arms[-1]
    rng = np.random.default_rng(314793)
    summary = {}
    for scenario in SCENARIOS:
        selected = [r for r in rows if r["scenario"] == scenario]
        strata = {}
        for index, row in enumerate(selected):
            key = tuple(row[k] for k in ("variant", "regime", "start", "profile"))
            strata.setdefault(key, []).append(index)
        boot = np.concatenate([rng.choice(ids, (3000, len(ids))) for ids in strata.values()], axis=1)
        timing = f"{active_arm}_plan_ms"
        report = {"fields": len(selected), timing: float(np.mean([r[timing] for r in selected]))}
        for metric in METRICS:
            if selected[0][f"{active_arm}_{metric}"] is None:
                continue
            values = {arm: np.array([r[f"{arm}_{metric}"] for r in selected]) for arm in arms}
            report[metric] = {arm: float(values[arm].mean()) for arm in arms}
            for arm in arms[:-1]:
                change = values[active_arm] - values[arm]
                report[metric][f"{active_arm}_minus_{arm}"] = dict(delta=float(change.mean()),
                    ci95=np.quantile(change[boot].mean(axis=1), [.025, .975]).tolist())
        summary[scenario] = report
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("outputs/sensing-world-residential-room-rollout"))
    parser.add_argument("--output", type=Path)
    policy = parser.add_mutually_exclusive_group()
    policy.add_argument("--prioritize-short", action="store_true", help="Score estimated short-zone variance reductions more heavily.")
    policy.add_argument("--oracle-short", action="store_true", help="Use true range coefficients for the short-zone score only.")
    policy.add_argument("--match-oracle-bonus", action="store_true",
                        help="Assign the largest inferred score weights to true short zones.")
    parser.add_argument("--adaptive-reference", type=Path, default=ADAPTIVE_OUTPUT,
                        help="Archived covariance-adaptive run for the short-priority comparison.")
    parser.add_argument("--priority-reference", type=Path, default=PRIORITY_OUTPUT,
                        help="Archived estimated short-priority run for the oracle comparison.")
    parser.add_argument("--oracle-reference", type=Path, default=ORACLE_OUTPUT,
                        help="Archived full-strength oracle run for the matched-strength comparison.")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--draws", type=int, default=4)
    args = parser.parse_args()
    active_arm, default_output = "adaptive", ADAPTIVE_OUTPUT
    reference_path, reference_bonus, reference_priority_source = None, 0, None
    if args.match_oracle_bonus:
        active_arm, default_output = MATCHED_ARM, MATCHED_OUTPUT
        reference_path, reference_bonus, reference_priority_source = args.oracle_reference, SHORT_BONUS, "oracle"
    elif args.oracle_short:
        active_arm, default_output = ORACLE_ARM, ORACLE_OUTPUT
        reference_path, reference_bonus = args.priority_reference, SHORT_BONUS
    elif args.prioritize_short:
        active_arm, default_output = PRIORITY_ARM, PRIORITY_OUTPUT
        reference_path = args.adaptive_reference
    args.output = args.output or default_output
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use an empty output directory.")
    if not 1 <= args.draws <= DRAWS or args.workers < 1:
        parser.error(f"Choose 1–{DRAWS} draws and at least one worker.")
    source = json.loads((args.source / "protocol.json").read_text())
    if (source["cap"], source["edge_cap"], source["early_weight"]) != (CAP, EDGES, EARLY_WEIGHT):
        parser.error("The archive must match the 52-edge, 24-reading, 25% protocol.")
    model = json.loads(Path(source["model"]).read_text())
    archive = {(r["variant"], r["start"], r["arm"]): r for r in json.loads((args.source / "routes.json").read_text())}
    with (args.source / "fields.csv").open() as stream:
        previous_fields = {field_key(r): r for r in csv.DictReader(stream) if int(r["draw"]) < args.draws}
    reference_fields = None
    if reference_path is not None:
        reference_protocol = json.loads((reference_path / "protocol.json").read_text())
        protocol_keys = ("model", "cap", "edge_cap", "early_weight")
        if (any(reference_protocol[k] != source[k] for k in protocol_keys)
                or reference_protocol.get("short_bonus", 0) != reference_bonus
                or reference_protocol.get("short_priority_source") != reference_priority_source):
            parser.error("The reference archive must match the model, budget, and expected scoring policy.")
        with (reference_path / "fields.csv").open() as stream:
            reference_fields = {field_key(r): r for r in csv.DictReader(stream) if int(r["draw"]) < args.draws}
        if reference_fields.keys() != previous_fields.keys():
            parser.error("The reference archive must cover the selected fields.")
    cases = [(variant, start) for variant in (0, 1) for start in STARTS]
    evaluate = partial(evaluate_case, model=model, archive=archive, previous_fields=previous_fields, draws=args.draws,
                       active_arm=active_arm, reference_fields=reference_fields)
    args.output.mkdir(parents=True, exist_ok=True)
    rows, routes = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for case, (scored, paths) in zip(cases, pool.map(evaluate, cases)):
            rows.extend(scored)
            routes.extend(paths)
            print(f"Completed layout {case[0]}, start {case[1]} with {len(scored)} paired fields", flush=True)
    if {field_key(row) for row in rows} != previous_fields.keys():
        raise ValueError("Unexpected field subset")
    write_csv(args.output / "fields.csv", rows)
    protocol = dict(source=str(args.source), model=source["model"], edge_cap=EDGES, cap=CAP, early_weight=EARLY_WEIGHT,
                    draws=args.draws, workers=args.workers, short_threshold=0.5)
    arms = ARMS
    if args.prioritize_short:
        protocol.update(adaptive_reference=str(args.adaptive_reference), short_bonus=SHORT_BONUS)
        arms = (*ARMS, PRIORITY_ARM)
    elif args.oracle_short:
        protocol.update(priority_reference=str(args.priority_reference), short_bonus=SHORT_BONUS,
                        short_priority_source="oracle")
        arms = (*ARMS, PRIORITY_ARM, ORACLE_ARM)
    elif args.match_oracle_bonus:
        protocol.update(oracle_reference=str(args.oracle_reference), short_bonus=SHORT_BONUS,
                        short_priority_source="rank_matched_oracle")
        arms = (*ARMS, PRIORITY_ARM, ORACLE_ARM, MATCHED_ARM)
    for name, data in (("routes", routes), ("protocol", protocol), ("summary", summarize(rows, arms))):
        (args.output / f"{name}.json").write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
