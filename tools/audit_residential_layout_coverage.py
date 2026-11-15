"""Benchmark the two-reading-per-room requirement on row and loop layouts."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from audit_residential_grid_resolution import field_factors, holdouts, predictions, refine_geometry, refined_mapper
from audit_residential_holdout_distance import CAP
from audit_residential_room_quota import EARLY_WEIGHT, TWO_ARM, TwoReadingCompletion, plan_route
from audit_residential_routes import CONDUCTANCES, STARTS, LayoutCase, floor_plan
from audit_residential_zone_ranges import GLOBAL_WEIGHTS
from spatial_exploration.baseline import write_csv
from spatial_exploration.sensing_world_study import NOISE


FAMILIES = ("row", "loop")
VARIANTS = (0, 1)
TRAVEL = 156
ARMS = ("quarter", TWO_ARM)
SEED = 389127
TIE = 1e-10
METRICS = ("whole_rmse", "short_rmse", "reserved_rmse", "reserved_short_rmse")


def plan_pairs(layout, anchors, held, weights, kernels, case):
    allowed = ~np.isin(np.arange(len(layout.room)), held)
    rooms = tuple(map(int, np.unique(layout.room)))
    completion = TwoReadingCompletion(layout, allowed, rooms)
    prior = np.einsum("m,mij->ij", weights, kernels)
    routes = {}
    for start in STARTS:
        routes[start] = []
        for arm, constraint, minimum in zip(ARMS, (None, completion), (0, 2)):
            begin = perf_counter()
            route = plan_route(layout, anchors, allowed, prior, start, constraint,
                               minimum=minimum, edge_cap=TRAVEL, rooms=rooms)
            if route["sites"][0] != anchors[start] or not allowed[route["sites"]].all():
                raise ValueError("A route violated the shared start or reserved hold-outs")
            routes[start].append(dict(family=case.family, variant=case.variant, start=start,
                                      arm=arm, plan_ms=(perf_counter() - begin) * 1000, **route))
    return routes


def score_pair(layout, held, kernels, log_prior, routes, profile, truth, noisy):
    sites = [np.asarray(route["sites"]) for route in routes]
    common = np.setdiff1d(np.arange(len(layout.room)), np.union1d(*sites))
    short = profile[layout.room[common]] > .5
    reserved = np.isin(common, held)
    if reserved.sum() != len(held) or not np.any(reserved & short):
        raise ValueError("The evaluation requires common and reserved hold-outs in the short region")
    groups = (slice(None), short, reserved, reserved & short)
    stats = {}
    for arm, measured in zip(ARMS, sites):
        mean, sd = predictions(kernels, measured, noisy[:, measured], common, log_prior)
        residual = mean - truth[:, common]
        for name, mask in zip(METRICS, groups):
            stats[f"{arm}_{name}"] = np.sqrt(np.mean(residual[:, mask]**2, axis=1))
        stats[f"{arm}_coverage_95"] = np.mean(np.abs(residual) <= 1.96 * sd, axis=1)
    return stats, len(common), int(short.sum())


def mixed_field_batches(case, layout, mixed, curve, draws, regimes=CONDUCTANCES):
    for regime_id, (regime, conductance) in enumerate(CONDUCTANCES.items()):
        if regime not in regimes:
            continue
        rng = np.random.default_rng([SEED, FAMILIES.index(case.family), case.variant, regime_id])
        factors = field_factors(layout, mixed, conductance, curve)
        for profile, (values, factor) in enumerate(zip(mixed, factors)):
            truth = rng.standard_normal((draws, len(layout.room))) @ factor.T
            noisy = truth + NOISE * rng.standard_normal(truth.shape)
            yield regime, profile, values, truth, noisy


def evaluate(case, model, draws):
    with threadpool_limits(limits=1):
        coarse = floor_plan(case)
        layout, anchors = refine_geometry(coarse)
        held, _ = holdouts(coarse, layout, anchors, case.variant)
        profiles, kernels, weights, curve = refined_mapper(coarse, layout, anchors, model)
        routes = plan_pairs(layout, anchors, held, weights, kernels, case)
        log_prior = np.log(weights)
        mixed = profiles[len(GLOBAL_WEIGHTS):]
        rows = []
        for regime, profile, values, truth, noisy in mixed_field_batches(case, layout, mixed, curve, draws):
            for start, pair in routes.items():
                stats, cells, short_cells = score_pair(layout, held, kernels, log_prior, pair, values, truth, noisy)
                metadata = dict(family=case.family, variant=case.variant, regime=regime, start=start, profile=profile,
                                short_rooms=",".join(map(str, np.flatnonzero(values > .5))),
                                cells=cells, short_cells=short_cells, reserved_cells=len(held),
                                same_sites=set(pair[0]["sites"]) == set(pair[1]["sites"]),
                                full_resources=all((route["edges"], route["readings"]) == (TRAVEL, CAP)
                                                   for route in pair))
                rows.extend(dict(metadata, draw=draw, **{key: val[draw] for key, val in stats.items()})
                            for draw in range(draws))
        print(f"{case.family} layout {case.variant}: {len(rows)} paired cases", flush=True)
    return rows, [route for pair in routes.values() for route in pair]


def paired_intervals(rows, delta):
    """Resample fields within layout and range profile, keeping their starts together."""
    strata = defaultdict(lambda: defaultdict(list))
    for row, change in zip(rows, delta):
        strata[row["variant"], row["profile"]][row["draw"]].append(change)
    rng = np.random.default_rng(SEED + 1)
    samples = np.zeros((3000, 2))
    for fields in strata.values():
        totals = np.array([(sum(changes), np.count_nonzero(np.asarray(changes) < -TIE))
                           for changes in fields.values()])
        indices = rng.integers(len(totals), size=(len(samples), len(totals)))
        samples += totals[indices].sum(axis=1)
    return np.quantile(samples / len(rows), [.025, .975], axis=0).T.tolist()


def summarize(rows, *, intervals=False):
    result = dict(cases=len(rows), distinct_fields=len({(row["variant"], row["profile"], row["draw"]) for row in rows}))
    if not rows:
        return result
    for metric in METRICS:
        existing, quota = (np.array([row[f"{arm}_{metric}"] for row in rows]) for arm in ARMS)
        delta = quota - existing
        result[metric] = dict(existing=float(existing.mean()), minimum_two=float(quota.mean()),
                              delta=float(delta.mean()), wins=int(np.count_nonzero(delta < -TIE)),
                              ties=int(np.count_nonzero(np.abs(delta) <= TIE)),
                              losses=int(np.count_nonzero(delta > TIE)))
        if intervals:
            delta_ci, win_ci = paired_intervals(rows, delta)
            result[metric].update(delta_ci95=delta_ci, win_fraction_ci95=win_ci)
    result["coverage_95"] = {arm: float(np.mean([row[f"{arm}_coverage_95"] for row in rows])) for arm in ARMS}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=32)
    parser.add_argument("--model", type=Path, default=Path("outputs/sensing-world-modes/frozen.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/sensing-world-residential-layout-coverage"))
    args = parser.parse_args()
    if args.draws < 1 or (args.output.exists() and any(args.output.iterdir())):
        parser.error("Use a positive field count and an empty output directory.")
    cases = [LayoutCase(family, variant) for family in FAMILIES for variant in VARIANTS]
    run = partial(evaluate, model=json.loads(args.model.read_text()), draws=args.draws)
    fields, routes = [], []
    with ProcessPoolExecutor(max_workers=len(cases)) as pool:
        for observations, paths in pool.map(run, cases):
            fields.extend(observations)
            routes.extend(paths)
    summary = {}
    for family in FAMILIES:
        summary[family] = {}
        for regime in CONDUCTANCES:
            rows = [row for row in fields if (row["family"], row["regime"]) == (family, regime)]
            summary[family][regime] = dict(
                all=summarize(rows, intervals=True),
                moved_sites=summarize([row for row in rows if not row["same_sites"]]),
                full_resources=summarize([row for row in rows if row["full_resources"]]),
                by_layout_start={f"{variant}-{start}": summarize(
                    [r for r in rows if (r["variant"], r["start"]) == (variant, start)])
                    for variant in VARIANTS for start in STARTS},
            )
    protocol = dict(model=str(args.model), seed=SEED, families=FAMILIES, variants=VARIANTS, starts=STARTS,
                    conductances=CONDUCTANCES, draws_per_profile=args.draws, travel_cap=TRAVEL, reading_cap=CAP,
                    intermediate_weight=EARLY_WEIGHT, minimum_distinct_readings_per_room=2, tie_tolerance=TIE,
                    evaluation="same cells left unmeasured by both routes for each layout/start; reserved cells also scored",
                    field_design="all mixed profiles; independent fields per layout/wall regime/profile, shared across starts")
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "fields.csv", fields)
    for name, data in (("summary", summary), ("routes", routes), ("protocol", protocol)):
        (args.output / f"{name}.json").write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
