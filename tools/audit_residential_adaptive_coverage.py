"""Compare adaptive rollout with and without two distinct readings per room."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits

from audit_residential_adaptive_rollout import adaptive_route, rollout_move_fast
from audit_residential_grid_resolution import expand_walk, holdouts, refine_geometry, refined_mapper, route_resources
from audit_residential_layout_coverage import ARMS, FAMILIES, SEED, TRAVEL, VARIANTS, mixed_field_batches, score_pair, summarize
from audit_residential_room_quota import CAP, TwoReadingCompletion
from audit_residential_routes import CONDUCTANCES, STARTS, LayoutCase, floor_plan
from audit_residential_zone_ranges import GLOBAL_WEIGHTS
from spatial_exploration.baseline import write_csv


def initialize(model, draws, regimes):
    global spec, archived_draws, wall_conditions
    spec, archived_draws, wall_conditions = model, draws, regimes
    threadpool_limits(limits=1)


@lru_cache(maxsize=4)
def setup(case):
    coarse = floor_plan(case)
    layout, anchors = refine_geometry(coarse)
    held, _ = holdouts(coarse, layout, anchors, case.variant)
    profiles, kernels, weights, curve = refined_mapper(coarse, layout, anchors, spec)
    allowed = ~np.isin(np.arange(len(layout.room)), held)
    rooms = tuple(map(int, np.unique(layout.room)))
    fields = {(regime, profile): (values, truth, noisy) for regime, profile, values, truth, noisy in
              mixed_field_batches(case, layout, profiles[len(GLOBAL_WEIGHTS):], curve, archived_draws, wall_conditions)}
    return SimpleNamespace(layout=layout, anchors=anchors, held=held, allowed=allowed, rooms=rooms,
                           profiles=profiles, kernels=kernels, log_prior=np.log(weights), fields=fields,
                           profile_priors=kernels.reshape(len(profiles), -1, len(allowed)**2).mean(axis=1),
                           completion=TwoReadingCompletion(layout, allowed, rooms))


def plan(design, start, noisy, completion):
    origin = int(design.anchors[start])
    received = []

    def read(site):
        received.append(site)
        return noisy[site]

    begin = perf_counter()
    sites, _ = adaptive_route(design.layout, origin, design.allowed, read, design.kernels,
                              design.profiles, design.profile_priors, design.log_prior,
                              edge_cap=TRAVEL, completion=completion, choose_move=rollout_move_fast)
    planning_ms = (perf_counter() - begin) * 1000
    np.testing.assert_array_equal(received, sites)
    if sites[0] != origin or not design.allowed[sites].all():
        raise ValueError("A route violated the shared start or measured a reserved cell")
    walk = expand_walk(design.layout, sites)
    counts = np.bincount(design.layout.room[sites], minlength=len(design.rooms))
    if completion is not None and (counts < 2).any():
        raise ValueError("A constrained route missed the two-reading requirement")
    return dict(sites=sites.tolist(), room_readings=counts.tolist(), plan_ms=planning_ms,
                **route_resources(design.layout, walk, sites, design.anchors, TRAVEL))


def evaluate(task):
    family, variant, regime, profile_id, draw = task
    design = setup(LayoutCase(family, variant))
    profile, truth, noisy = design.fields[regime, profile_id]
    observation = noisy[draw]
    rows, routes = [], []
    for start in STARTS:
        pair = [plan(design, start, observation, completion) for completion in (None, design.completion)]
        scores, cells, short_cells = score_pair(design.layout, design.held, design.kernels, design.log_prior,
                                               pair, profile, truth[draw:draw+1], noisy[draw:draw+1])
        metadata = dict(family=family, variant=variant, regime=regime, profile=profile_id, draw=draw, start=start)
        rows.append(dict(metadata, cells=cells, short_cells=short_cells,
                         same_sites=set(pair[0]["sites"]) == set(pair[1]["sites"]),
                         full_resources=all((route["edges"], route["readings"]) == (TRAVEL, CAP) for route in pair),
                         **{name: float(values[0]) for name, values in scores.items()}))
        routes.extend(dict(metadata, arm=arm, **route) for arm, route in zip(ARMS, pair))
    return rows, routes


def save_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    temporary.replace(path)


def summarize_run(output, tasks, regimes):
    fields, routes = [], []
    for task in tasks:
        rows, paths = json.loads((output / "cases" / ("_".join(map(str, task)) + ".json")).read_text())
        fields.extend(rows)
        routes.extend(paths)
    summary = {}
    for family in FAMILIES:
        summary[family] = {}
        for regime in regimes:
            rows = [row for row in fields if (row["family"], row["regime"]) == (family, regime)]
            paths = [r for r in routes if (r["family"], r["regime"]) == (family, regime)]
            resources = {}
            for arm in ARMS:
                selected = [r for r in paths if r["arm"] == arm]
                resources[arm] = dict(
                    **{metric: float(np.mean([r[metric] for r in selected])) for metric in ("edges", "readings", "plan_ms")},
                    met_quota=sum(min(r["room_readings"]) >= 2 for r in selected),
                )
            summary[family][regime] = dict(
                all=summarize(rows, intervals=True),
                moved_sites=summarize([r for r in rows if not r["same_sites"]]),
                full_resources=summarize([r for r in rows if r["full_resources"]]),
                by_layout_start={f"{variant}-{start}": summarize(
                    [r for r in rows if (r["variant"], r["start"]) == (variant, start)])
                    for variant in VARIANTS for start in STARTS},
                resources=resources,
            )
    suffix = f"-{max(t[-1] for t in tasks) + 1}"
    write_csv(output / f"fields{suffix}.csv", fields)
    save_json(output / f"routes{suffix}.json", routes)
    save_json(output / f"summary{suffix}.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("outputs/sensing-world-residential-layout-coverage"))
    parser.add_argument("--output", type=Path, default=Path("outputs/sensing-world-residential-adaptive-coverage"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    source = json.loads((args.source / "protocol.json").read_text())
    draws = source["draws_per_profile"]
    regimes = tuple(source["conductances"])
    if (tuple(source["families"]), tuple(source["variants"]), tuple(source["starts"]), source["seed"],
        source["conductances"], source["travel_cap"], source["reading_cap"]) != (
            FAMILIES, VARIANTS, STARTS, SEED, CONDUCTANCES, TRAVEL, CAP):
        parser.error("The source protocol must match the fixed-planner floor-plan benchmark.")
    model = Path(source["model"]).read_bytes()
    protocol = dict(source=source, model_sha256=hashlib.sha256(model).hexdigest(), method="adaptive_rollout")
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_path = args.output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        parser.error("This output directory contains a different protocol.")
    save_json(protocol_path, protocol)
    checkpoint = args.output / "cases"
    checkpoint.mkdir(exist_ok=True)
    tasks = [(family, variant, regime, profile, draw)
             for draw in range(draws) for regime in regimes for family, rooms in zip(FAMILIES, (3, 4))
             for variant in VARIANTS for profile in range(2**rooms - 2)]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize,
                             initargs=(json.loads(model), draws, regimes)) as pool:
        for draw in range(draws):
            batch = [task for task in tasks if task[-1] == draw]
            paths = [checkpoint / ("_".join(map(str, task)) + ".json") for task in batch]
            missing = [(task, path) for task, path in zip(batch, paths) if not path.is_file()]
            for (_, path), result in zip(missing, pool.map(evaluate, [task for task, _ in missing])):
                save_json(path, result)
            print(f"Completed archived draw {draw + 1}/{draws} across all starts, layouts, and wall conditions", flush=True)
            if draw + 1 in (1, 2, 4, 8, 16, draws):
                summarize_run(args.output, [task for task in tasks if task[-1] <= draw], regimes)


if __name__ == "__main__":
    main()
