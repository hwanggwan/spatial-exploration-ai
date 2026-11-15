"""Add a terminal worst-room variance penalty to the 25% distance-aware rollout."""

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from audit_residential_grid_resolution import (
    expand_walk, holdouts, predictions, refine_geometry, refined_mapper, route_resources,
)
from audit_residential_holdout_distance import CAP, FAMILY
from audit_residential_refined_rollout import SCENARIOS, field_batches, field_key, full_budget_route, remaining_variance
from audit_residential_routes import LayoutCase, STARTS, floor_plan
from spatial_exploration.baseline import GPBelief, write_csv
from spatial_exploration.sensing_world_study import NOISE


EARLY_WEIGHT = 0.25
ROOM_WEIGHTS = {"quarter": 0.0, "room": 0.25}
ROOMS = (0, 1, 3)  # Zone 2 is the hallway.
METRICS = ("map_39", "worst_room_39", "map_final", "worst_room_final")


def describe_route(layout, anchors, allowed, prior, sites, start, edge_cap, room_weight):
    walk = expand_walk(layout, sites)
    costs = layout.distances[sites[:-1], sites[1:]] * (layout.size - 1)
    arrivals = np.r_[0, np.rint(costs).astype(int).cumsum()]
    np.testing.assert_array_equal(walk[arrivals], sites)
    resources = route_resources(layout, walk, sites, anchors, edge_cap)
    if sites[0] != start or len(np.unique(sites)) != len(sites) or not allowed[sites].all():
        raise ValueError("Invalid start or measurement sites")
    if arrivals[-1] != resources["edges"] or len(sites) > CAP:
        raise ValueError("Invalid realized resources")

    room_masks = tuple(layout.room == room for room in ROOMS)
    belief = GPBelief(prior, NOISE)
    belief.observe(start, 0.0)
    distance_cap = edge_cap / (layout.size - 1)
    terminal, integral = remaining_variance(
        belief.covariance, layout.distances, start, sites[1:], belief.noise_variance,
        distance_cap, distance_cap, room_masks=room_masks, room_weight=room_weight,
    )
    variances = [belief.covariance.diagonal().mean()]
    for site in sites[1:]:
        belief.observe(site, 0.0)
        variances.append(belief.covariance.diagonal().mean())
    counts = np.searchsorted(arrivals, np.arange(edge_cap + 1), side="right")
    trajectory = np.asarray(variances)[counts - 1]
    worst_room = float(max(belief.covariance.diagonal()[mask].mean() for mask in room_masks))
    expected_terminal = (1 - room_weight) * trajectory[-1] + room_weight * worst_room
    np.testing.assert_allclose([terminal, integral], [expected_terminal, trajectory[:-1].mean()],
                               atol=1e-12, rtol=0)

    counts_39 = np.bincount(layout.room[sites[:counts[39]]], minlength=4)
    return dict(**resources, sites=sites.tolist(), walk=walk.tolist(), arrivals=arrivals.tolist(),
                counts_39=counts_39.tolist(), rooms_39=int(np.count_nonzero(counts_39[list(ROOMS)])),
                readings_39=int(counts[39]), map_variance_final=float(trajectory[-1]),
                worst_room_variance_final=worst_room, mean_walking_variance=float(integral))


def evaluate(variant, model, archive, previous_fields, edge_cap):
    coarse = floor_plan(LayoutCase(FAMILY, variant))
    layout, anchors = refine_geometry(coarse)
    held, _ = holdouts(coarse, layout, anchors, variant)
    profiles, kernels, mode_weights, curve = refined_mapper(coarse, layout, anchors, model)
    prior = np.einsum("m,mij->ij", mode_weights, kernels)
    log_prior = np.log(mode_weights)
    allowed = ~np.isin(np.arange(len(layout.room)), held)
    room_masks = tuple(layout.room == room for room in ROOMS)
    held_rooms = tuple(mask[held] for mask in room_masks)
    distance_cap = edge_cap / (layout.size - 1)
    routes = {}
    for start in STARTS:
        origin = int(anchors[start])
        previous = archive[variant, start]
        for arm, room_weight in ROOM_WEIGHTS.items():
            before = perf_counter()
            sites, terminal, branches = full_budget_route(
                layout, allowed, prior, origin, distance_cap, early_weight=EARLY_WEIGHT,
                room_masks=room_masks, room_weight=room_weight,
            )
            plan_ms = 1000 * (perf_counter() - before)
            route = describe_route(layout, anchors, allowed, prior, np.asarray(sites), origin, edge_cap, room_weight)
            np.testing.assert_allclose(route["map_variance_final"], terminal, rtol=0, atol=1e-12)
            if arm == "quarter":
                for key in ("sites", "walk", "arrivals"):
                    np.testing.assert_array_equal(route[key], previous[key])
            routes[start, arm] = dict(variant=variant, start=start, arm=arm, plan_ms=plan_ms, branches=branches, **route)
        print(f"Verified both routes for layout {variant}, start {start}", flush=True)

    rows = []
    for regime, start, profile, scenario, truth, noisy in field_batches(variant, layout, profiles, curve):
        metrics = {}
        for arm in ROOM_WEIGHTS:
            route = routes[start, arm]
            for label, count in (("39", route["readings_39"]), ("final", route["readings"])):
                sites = route["sites"][:count]
                mean, sd = predictions(kernels, sites, noisy[:, sites], held, log_prior)
                error = mean - truth[:, held]
                squared_error = error**2
                metrics[f"{arm}_map_{label}"] = np.sqrt(squared_error.mean(axis=1))
                room_rmse = np.stack([np.sqrt(squared_error[:, mask].mean(axis=1)) for mask in held_rooms])
                metrics[f"{arm}_worst_room_{label}"] = room_rmse.max(axis=0)
            metrics[f"{arm}_coverage"] = np.mean(abs(error) <= 1.96 * sd, axis=1)
            metrics[f"{arm}_width"] = np.mean(3.92 * sd, axis=1)
        for draw in range(len(truth)):
            row = dict(variant=variant, regime=regime, start=start, profile=profile, draw=draw, scenario=scenario,
                       **{key: values[draw] for key, values in metrics.items()})
            np.testing.assert_allclose(row["quarter_map_final"], float(previous_fields[field_key(row)]["quarter_final"]),
                                       atol=1e-10, rtol=0)
            rows.append(row)
    return rows, list(routes.values())


def summarize(rows, routes):
    rng = np.random.default_rng(314790)
    scenarios = {}
    for scenario in SCENARIOS:
        selected = [row for row in rows if row["scenario"] == scenario]
        strata = {}
        for index, row in enumerate(selected):
            key = tuple(row[k] for k in ("variant", "regime", "start", "profile"))
            strata.setdefault(key, []).append(index)
        resamples = np.concatenate([rng.choice(ids, (3000, len(ids))) for ids in strata.values()], axis=1)
        scores = {}
        for metric in (*METRICS, "coverage", "width"):
            base, room = (np.asarray([row[f"{arm}_{metric}"] for row in selected]) for arm in ROOM_WEIGHTS)
            delta = room - base
            scores[metric] = dict(quarter=float(base.mean()), room=float(room.mean()), delta=float(delta.mean()),
                                  ci95=np.quantile(delta[resamples].mean(axis=1), [.025, .975]).tolist())
        scenarios[scenario] = dict(fields=len(selected), **scores)
    resources = {}
    for arm in ROOM_WEIGHTS:
        group = [route for route in routes if route["arm"] == arm]
        keys = ("edges", "readings", "readings_39", "rooms_39", "plan_ms", "branches",
                "map_variance_final", "worst_room_variance_final", "mean_walking_variance")
        resources[arm] = {k: dict(mean=float(np.mean([r[k] for r in group])),
                                  range=[min(r[k] for r in group), max(r[k] for r in group)]) for k in keys}
        resources[arm]["counts_39"] = np.mean([r["counts_39"] for r in group], axis=0).tolist()
    return dict(scenarios=scenarios, routes=resources)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("outputs/sensing-world-residential-quarter-rollout"))
    parser.add_argument("--output", type=Path, default=Path("outputs/sensing-world-residential-room-rollout"))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use an empty output directory.")
    metadata = json.loads((args.source / "protocol.json").read_text())
    if metadata["cap"] != CAP or metadata["edge_cap"] != 52 or metadata["weights"]["quarter"] != EARLY_WEIGHT:
        parser.error("The baseline archive must match the 52-edge, 24-reading, 25% protocol.")
    model_path = json.loads((Path(metadata["source"]) / "protocol.json").read_text())["model"]
    model = json.loads(Path(model_path).read_text())
    archive = {(r["variant"], r["start"]): r for r in json.loads((args.source / "routes.json").read_text())
               if r["arm"] == "quarter"}
    with (args.source / "fields.csv").open() as stream:
        previous_fields = {field_key(row): row for row in csv.DictReader(stream)}
    rows, routes = [], []
    with threadpool_limits(limits=1):
        for variant in sorted({variant for variant, _ in archive}):
            scored, planned = evaluate(variant, model, archive, previous_fields, metadata["edge_cap"])
            rows.extend(scored)
            routes.extend(planned)
    if {field_key(row) for row in rows} != previous_fields.keys():
        raise ValueError("Field replay must cover the entire 25% archive")
    summary = summarize(rows, routes)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "fields.csv", rows)
    protocol = dict(source=str(args.source), model=model_path, edge_cap=metadata["edge_cap"], cap=CAP,
                    early_weight=EARLY_WEIGHT, terminal_room_weights=ROOM_WEIGHTS, rooms=ROOMS,
                    checkpoint=39, bootstrap_resamples=3000)
    for name, payload in (("protocol", protocol), ("routes", routes), ("summary", summary)):
        (args.output / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {len(rows)} paired field results to {args.output}")


if __name__ == "__main__":
    main()
