"""Export verified 23 × 23 corridor replays at each measurement prefix."""

import csv
import hashlib
import json

import numpy as np
from threadpoolctl import threadpool_limits

from audit_residential_adaptive_coverage import coverage_field_batches
from audit_residential_grid_resolution import expand_walk, holdouts, predictions, refine_geometry, refined_mapper
from audit_residential_layout_coverage import ARMS as COVERAGE_POLICIES
from audit_residential_refined_rollout import ARM_ROLLOUT, MATCHED_ARMS, PLANNER_ARMS, field_batches, field_key
from audit_residential_room_quota import ROOMS, TwoReadingCompletion, plan_route
from audit_residential_routes import LayoutCase, floor_plan
from audit_residential_zone_ranges import GLOBAL_WEIGHTS


GREEDY = PLANNER_ARMS[1]
ARMS = (GREEDY, ARM_ROLLOUT)
RANGES = ("short", "mixed", "long")
COVERAGE_ARMS = ("quarter_156", "room_quota_two_156")
CORRIDOR_ZONES = ["Lower-left room", "Lower-right room", "Corridor", "Upper room"]
ADDITIONAL_ROOM_CASES = (
    ("row", "Three rooms in a row", 64, 1, 1, ("Left room", "Middle room", "Right room")),
    ("loop", "Four-room loop", 128, 0, 1, ("Lower-left room", "Lower-right room", "Upper-left room", "Upper-right room")),
)


def pack_layout(layout, held):
    size = layout.size
    low, high = np.where(np.triu(layout.wall_edges, k=1))
    walls = []
    for a, b in zip(low, high):
        ar, ac = divmod(int(a), size)
        br, bc = divmod(int(b), size)
        if ar == br:
            z = size - ar - 1
            walls.append([bc, z, bc, z + 1])
        else:
            z = size - br
            walls.append([ac, z, ac + 1, z])
    return dict(size=size, walls=walls, held=held.tolist(), zones=layout.room.tolist())


def choose_examples(metrics, variant, start):
    selected = {}
    for scenario in RANGES:
        rows = [row for row in metrics if (row["variant"], row["start"], row["scenario"])
                == (str(variant), str(start), scenario)]
        # Use the same short-range example as the matched-distance figure.
        if scenario == "short":
            delta = [float(r[f"{MATCHED_ARMS[GREEDY]}_all"]) - float(r[f"{GREEDY}_all"]) for r in rows]
            selected[scenario] = rows[int(np.argmin(abs(np.asarray(delta) - np.mean(delta))))]
        else:
            selected[scenario] = rows[0]
    return selected


def pack_plan(layout, held, kernels, log_prior, route, truth, readings, *, score_groups=None):
    sites = np.asarray(route["sites"])
    walk = np.asarray(route["walk"])
    distance = np.rint(layout.distances[sites[:-1], sites[1:]] * (layout.size - 1)).astype(int)
    arrivals = np.r_[0, distance.cumsum()]
    if (not np.array_equal(walk[arrivals], sites) or arrivals[-1] != route["edges"]
            or not layout.movement[walk[:-1], walk[1:]].all() or np.isin(sites, held).any()):
        raise ValueError("Invalid corridor replay route")
    targets = np.arange(len(truth))
    mean, std, errors = [], [], []
    groups = {name: [] for name in score_groups} if score_groups else {}
    for count in range(1, len(sites) + 1):
        measured = sites[:count]
        pred, sd = predictions(kernels, measured, readings[measured][None], targets, log_prior)
        mean.append(np.rint(pred[0] * 1000).astype(int).tolist())
        std.append(np.rint(sd[0] * 1000).astype(int).tolist())
        errors.append(float(np.sqrt(np.mean((pred[0, held] - truth[held])**2))))
        for name in groups:
            group = score_groups[name]
            groups[name].append(float(np.sqrt(np.mean((pred[0, group] - truth[group])**2))))
    return dict(sites=sites.tolist(), walk=walk.tolist(), endpoints=(arrivals + 1).tolist(),
                mean=mean, std=std, rmse=errors, **groups)


def build_residential_replay(study, root):
    config = json.loads((study / "protocol.json").read_text())
    weights = json.loads((root / config["model"]).read_text())
    combined = json.loads((study / "summary.json").read_text())
    routes = {(r["variant"], r["start"], r["arm"]): r for r in json.loads((study / "routes.json").read_text())}
    with (study / "fields.csv").open() as stream:
        metrics = list(csv.DictReader(stream))
    layouts, cases = [], []
    start = 14
    with threadpool_limits(limits=1):
        for variant in sorted({r[0] for r in routes}):
            coarse = floor_plan(LayoutCase("corridor", variant))
            fine, anchors = refine_geometry(coarse)
            held, _ = holdouts(coarse, fine, anchors, variant)
            layouts.append(pack_layout(fine, held))
            profiles, kernels, prior, curve = refined_mapper(coarse, fine, anchors, weights)
            examples = {(r["regime"], int(r["profile"])): r
                        for r in choose_examples(metrics, variant, start).values()}
            for wall, origin, profile, scenario, truth, noisy in field_batches(variant, fine, profiles, curve):
                if origin != start or (wall, profile) not in examples:
                    continue
                row = examples.pop((wall, profile))
                if scenario != row["scenario"]:
                    raise ValueError("Incorrect replay scenario")
                draw = int(row["draw"])
                hidden, readings = truth[draw], noisy[draw]
                plans = {}
                for arm in ARMS:
                    route = routes[variant, start, arm]
                    plan = pack_plan(fine, held, kernels, np.log(prior), route, hidden, readings)
                    if len(plan["sites"]) > config["cap"] or route["edges"] > config["fine_edge_cap"]:
                        raise ValueError("Corridor replay exceeds a resource cap")
                    np.testing.assert_allclose(plan["rmse"][-1], float(row[f"{arm}_all"]), atol=1e-10, rtol=0)
                    plans[arm] = plan
                matched = routes[variant, start, MATCHED_ARMS[GREEDY]]
                cutoff = matched["edges"]
                prefix = plans[ARM_ROLLOUT]
                arrived = sum(i <= cutoff + 1 for i in prefix["endpoints"])
                if (prefix["walk"][:cutoff+1] != matched["walk"]
                        or prefix["sites"][:arrived] != matched["sites"]
                        or cutoff != len(plans[GREEDY]["walk"]) - 1):
                    raise ValueError("Incorrect equal-travel prefix")
                np.testing.assert_allclose(prefix["rmse"][arrived-1],
                    float(row[f"{MATCHED_ARMS[GREEDY]}_all"]), atol=1e-10, rtol=0)
                cases.append(dict(layout=variant, scenario=scenario, cutoff=cutoff,
                                  truth=np.rint(hidden * 1000).astype(int).tolist(), plans=plans))
                if not examples:
                    break
            if examples:
                raise ValueError("Missing residential replay scenes")
    results = {}
    for name in RANGES:
        stats = combined["scenarios"][name]
        results[name] = dict(fields=stats["fields"],
            caps=[stats["means"][key]["all"] for key in (*PLANNER_ARMS, ARM_ROLLOUT)],
            matched=[stats["means"][GREEDY]["all"], stats["means"][MATCHED_ARMS[GREEDY]]["all"]],
            interval=stats["matched_travel_rollout_minus"][GREEDY]["all"]["ci95"])
    return dict(layouts=layouts, cases=cases, results=results, cap=config["cap"], travel=config["fine_edge_cap"])


def build_room_coverage_replay(study, root, adaptive_study=None, adaptive_corridor=None):
    """Export a paired 156-step field that illustrates a room-coverage gain."""
    config = json.loads((study / "protocol.json").read_text())
    source = json.loads((study / "summary.json").read_text())
    weights = json.loads((root / config["model"]).read_text())
    variant, start = (config["example"][k] for k in ("variant", "start"))
    regime, profile, draw, travel = "weak", 10, 2, 156
    example = dict(variant=variant, regime=regime, start=start, profile=profile, draw=draw)
    arms = COVERAGE_ARMS
    routes = {row["arm"]: row for row in json.loads((study / "routes.json").read_text())
              if (row["variant"], row["start"]) == (variant, start)}
    with (study / "fields.csv").open() as stream:
        metric = next(row for row in csv.DictReader(stream) if field_key(row) == field_key(example))
    coarse = floor_plan(LayoutCase("corridor", variant))
    fine, anchors = refine_geometry(coarse)
    held, _ = holdouts(coarse, fine, anchors, variant)
    grid = np.arange(len(fine.room))
    common = np.setdiff1d(grid, np.concatenate([routes[arm]["sites"] for arm in arms]))
    benchmark = np.setdiff1d(grid, np.concatenate([row["sites"] for row in routes.values()]))
    if not np.isin(held, common).all():
        raise ValueError("Room comparison measured a reserved site")
    with threadpool_limits(limits=1):
        profiles, kernels, prior, curve = refined_mapper(coarse, fine, anchors, weights)
        batch = next(batch for batch in field_batches(variant, fine, profiles, curve)
                     if batch[:4] == (regime, start, profile, "mixed"))
        truth, readings = (values[draw] for values in batch[4:])
        short_profile = profiles[len(GLOBAL_WEIGHTS) + profile]
        short = benchmark[short_profile[fine.room[benchmark]] > 0.5]
        example_short = common[short_profile[fine.room[common]] > 0.5]
        plans = {}
        short_scores = []
        for arm in arms:
            route = routes[arm]
            if (route["sites"][0], len(route["sites"]), route["edges"]) != (anchors[start], config["cap"], travel):
                raise ValueError("Room replay must match the start and both realized budgets")
            plans[arm] = pack_plan(fine, common, kernels, np.log(prior), route, truth, readings,
                                  score_groups=dict(original_rmse=held, benchmark_short_rmse=short,
                                                    example_short_rmse=example_short))
            plan = plans[arm]
            np.testing.assert_allclose((plan.pop("original_rmse")[-1], plan.pop("benchmark_short_rmse")[-1]),
                                       (float(metric[f"{arm}_original_rmse"]), float(metric[f"{arm}_short_rmse"])),
                                       atol=1e-10, rtol=0)
            short_scores.append(plan.pop("example_short_rmse")[-1])
        if not (plans[arms[1]]["rmse"][-1] < plans[arms[0]]["rmse"][-1] and short_scores[1] < short_scores[0]):
            raise ValueError("The displayed scene should illustrate lower whole-map and short-zone RMSE")
    scores = []
    for budget in config["budgets"]:
        pair = [source["scores"][f"{method}_{budget}"] for method in ("quarter", "room_quota_two")]
        scores.append(dict(budget=budget, overall=[score["common_rmse"] for score in pair],
                           short=[score["short_rmse"] for score in pair]))
    alternatives = (build_adaptive_room_replays(adaptive_study, root, travel, config["cap"])
                    if adaptive_study else build_additional_room_replays(weights, travel, config["cap"]))
    if adaptive_corridor:
        alternatives.extend(build_adaptive_room_replays(adaptive_corridor, root, travel, config["cap"]))
    return dict(id="corridor-fixed" if adaptive_corridor else "corridor", label="Three rooms off a corridor",
                zoneNames=CORRIDOR_ZONES,
                shortZones=(short_profile > 0.5).tolist(), alternatives=alternatives,
                layout=pack_layout(fine, np.empty(0, dtype=int)), travel=travel, cap=config["cap"], plans=plans,
                truth=np.rint(truth * 1000).astype(int).tolist(), scores=scores, fields=source["fields"],
                evaluationCells=len(common))


def build_additional_room_replays(model, travel, cap):
    """Plan and score the selected row and loop examples on common unmeasured cells."""
    variants = []
    with threadpool_limits(limits=1):
        for family, label, start, profile, draw, zone_names in ADDITIONAL_ROOM_CASES:
            coarse = floor_plan(LayoutCase(family, 0))
            fine, anchors = refine_geometry(coarse)
            held, _ = holdouts(coarse, fine, anchors, 0)
            grid = np.arange(len(fine.room))
            allowed = ~np.isin(grid, held)
            rooms = tuple(map(int, np.unique(fine.room)))
            if rooms != tuple(range(len(zone_names))):
                raise ValueError("Room names must match the movement graph")
            profiles, kernels, weights, curve = refined_mapper(coarse, fine, anchors, model)
            prior = np.einsum("m,mij->ij", weights, kernels)
            completion = TwoReadingCompletion(fine, allowed, rooms)
            routes = [plan_route(fine, anchors, allowed, prior, start, constraint,
                                 minimum=minimum, edge_cap=travel, rooms=rooms)
                      for constraint, minimum in ((None, 0), (completion, 2))]
            for route in routes:
                if (route["sites"][0], len(route["sites"]), route["edges"]) != (anchors[start], cap, travel):
                    raise ValueError(f"{family} replay must use the same start and both budgets")
            common = np.setdiff1d(grid, np.concatenate([route["sites"] for route in routes]))
            if not np.isin(held, common).all():
                raise ValueError(f"{family} replay measured a reserved cell")
            batch = next(batch for batch in field_batches(0, fine, profiles, curve)
                         if batch[:4] == ("weak", start, profile, "mixed"))
            truth, readings = (values[draw] for values in batch[4:])
            short_zones = profiles[len(GLOBAL_WEIGHTS) + profile] > 0.5
            short = common[short_zones[fine.room[common]]]
            plans = {arm: pack_plan(fine, common, kernels, np.log(weights), route, truth, readings,
                                   score_groups=dict(example_short_rmse=short))
                     for arm, route in zip(COVERAGE_ARMS, routes)}
            short_scores = [plan.pop("example_short_rmse")[-1] for plan in plans.values()]
            baseline, constrained = (plans[arm]["rmse"][-1] for arm in COVERAGE_ARMS)
            if not (constrained < baseline and short_scores[1] < short_scores[0]):
                raise ValueError(f"{family} replay must illustrate gains on both evaluation measures")
            variants.append(dict(id=family, label=label, zoneNames=zone_names, shortZones=short_zones.tolist(),
                                 layout=pack_layout(fine, np.empty(0, dtype=int)), travel=travel, cap=cap, plans=plans,
                                 truth=np.rint(truth * 1000).astype(int).tolist(), shortScores=short_scores,
                                 evaluationCells=len(common)))
    return variants


def build_adaptive_room_replays(study, root, travel, cap):
    """Export a resource-matched adaptive example and benchmark per floor-plan family."""
    archive = json.loads((study / "protocol.json").read_text())
    source = archive["source"]
    draws = archive.get("draws_per_profile", source.get("draws_per_profile"))
    model = (root / source["model"]).read_bytes()
    source_travel = source.get("travel_cap", max(source.get("budgets", (0,))))
    source_cap = source.get("reading_cap", source.get("cap"))
    if (archive["method"] != "adaptive_rollout" or hashlib.sha256(model).hexdigest() != archive["model_sha256"]
            or (source_travel, source_cap) != (travel, cap)):
        raise ValueError("Adaptive replay protocol and frozen mapper must match the benchmark")
    mapper = json.loads(model)
    summary = json.loads((study / f"summary-{draws}.json").read_text())
    saved_routes = json.loads((study / f"routes-{draws}.json").read_text())
    with (study / f"fields-{draws}.csv").open() as stream:
        metrics = list(csv.DictReader(stream))

    def key(record):
        return (record["family"], *field_key(record))

    by_key = {(key(row), row["arm"]): row for row in saved_routes}
    replays = []
    layouts = [(family, label, zones) for family, label, _, _, _, zones in ADDITIONAL_ROOM_CASES]
    layouts.append(("corridor", "Three rooms off a corridor", CORRIDOR_ZONES))
    with threadpool_limits(limits=1):
        for family, label, zone_names in layouts:
            if family not in summary:
                continue
            rooms = ROOMS if family == "corridor" else range(len(zone_names))
            winners = []
            for row in metrics:
                if (row["family"], row["regime"], row["full_resources"]) != (family, "weak", "True"):
                    continue
                whole_gain, short_gain = (float(row[f"{COVERAGE_POLICIES[0]}_{metric}_rmse"])
                                          - float(row[f"{COVERAGE_POLICIES[1]}_{metric}_rmse"])
                                          for metric in ("whole", "short"))
                if whole_gain > 0 and short_gain > 0:
                    baseline = by_key[key(row), COVERAGE_POLICIES[0]]
                    missed_room = min(baseline["room_readings"][room] for room in rooms) < 2
                    winners.append((missed_room, whole_gain, short_gain, row))
            missed_room = any(item[0] for item in winners)
            winners = sorted((item for item in winners if item[0] == missed_room), key=lambda item: item[1])
            if not winners:
                raise ValueError(f"No adaptive {family} example meets both budgets and improves both scores")
            if family == "corridor":
                # Keep both displayed gains near the median among qualifying scenes.
                typical = np.median([item[1:3] for item in winners], axis=0)
                example = min(winners, key=lambda item: np.abs(np.asarray(item[1:3]) / typical - 1).sum())[-1]
            else:
                example = winners[(len(winners) - 1) // 2][-1]
            variant, start, profile_id, draw = (int(example[k]) for k in ("variant", "start", "profile", "draw"))
            case = LayoutCase(family, variant)
            coarse = floor_plan(case)
            fine, anchors = refine_geometry(coarse)
            held, _ = holdouts(coarse, fine, anchors, variant)
            profiles, kernels, weights, curve = refined_mapper(coarse, fine, anchors, mapper)
            field_key_at_start = ("weak", start if family == "corridor" else None, profile_id)
            values, fields, sensor = next((values, fields, sensor) for batch_key, values, fields, sensor in
                                          coverage_field_batches(case, fine, profiles, curve, draws, ("weak",))
                                          if batch_key == field_key_at_start)
            truth, readings = fields[draw], sensor[draw]
            routes = [by_key[key(example), arm].copy() for arm in COVERAGE_POLICIES]
            for route in routes:
                if (route["sites"][0], len(route["sites"]), route["edges"]) != (anchors[start], cap, travel):
                    raise ValueError("Selected adaptive routes must reach both budgets from the same start")
                route["walk"] = expand_walk(fine, route["sites"]).tolist()
            common = np.setdiff1d(np.arange(len(fine.room)), np.concatenate([route["sites"] for route in routes]))
            if not np.isin(held, common).all() or len(common) != int(example["cells"]):
                raise ValueError("Adaptive replay evaluation sites must match the benchmark")
            short_zones = values > 0.5
            short = common[short_zones[fine.room[common]]]
            plans, short_scores = {}, []
            for arm, policy, route in zip(COVERAGE_ARMS, COVERAGE_POLICIES, routes):
                packed = pack_plan(fine, common, kernels, np.log(weights), route, truth, readings,
                                   score_groups=dict(short_rmse=short))
                short_scores.append(packed.pop("short_rmse")[-1])
                expected = [float(example[f"{policy}_{region}_rmse"]) for region in ("whole", "short")]
                np.testing.assert_allclose([packed["rmse"][-1], short_scores[-1]], expected, rtol=0, atol=1e-10)
                plans[arm] = packed
            benchmark = []
            for regime, report in summary[family].items():
                scopes = ("all", "full_resources") if family == "corridor" else ("all",)
                for scope in scopes:
                    stats = report[scope]
                    if stats["cases"]:
                        benchmark.append(dict(regime=regime, cases=stats["cases"],
                                              whole=stats["whole_rmse"]["wins"], short=stats["short_rmse"]["wins"],
                                              wholeDelta=stats["whole_rmse"]["delta"], shortDelta=stats["short_rmse"]["delta"],
                                              **({"fullBudget": True} if scope == "full_resources" else {})))
            replays.append(dict(id=family, label=label, adaptive=True, zoneNames=zone_names, shortZones=short_zones.tolist(),
                                layout=pack_layout(fine, np.empty(0, dtype=int)), travel=travel, cap=cap, plans=plans,
                                truth=np.rint(truth * 1000).astype(int).tolist(), shortScores=short_scores,
                                evaluationCells=len(common), benchmark=benchmark))
    return replays
