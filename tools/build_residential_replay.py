"""Export verified 23 × 23 corridor replays at each measurement prefix."""

import csv
import json

import numpy as np
from threadpoolctl import threadpool_limits

from audit_residential_grid_resolution import holdouts, predictions, refine_geometry, refined_mapper
from audit_residential_refined_rollout import ARM_ROLLOUT, MATCHED_ARMS, PLANNER_ARMS, field_batches, field_key
from audit_residential_routes import LayoutCase, floor_plan
from audit_residential_zone_ranges import GLOBAL_WEIGHTS


GREEDY = PLANNER_ARMS[1]
ARMS = (GREEDY, ARM_ROLLOUT)
RANGES = ("short", "mixed", "long")


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


def build_room_coverage_replay(study, root):
    """Export the paired 156-step field used in the room-coverage report."""
    config = json.loads((study / "protocol.json").read_text())
    source = json.loads((study / "summary.json").read_text())
    weights = json.loads((root / config["model"]).read_text())
    variant, start = (config["example"][k] for k in ("variant", "start"))
    regime, profile, draw, travel = "weak", 10, 3, 156
    example = dict(variant=variant, regime=regime, start=start, profile=profile, draw=draw)
    arms = ("quarter_156", "room_quota_two_156")
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
    left = fine.room == 0
    doors = np.flatnonzero(left & fine.movement[:, ~left].any(axis=1))
    depth = np.rint(fine.distances[:, doors].min(axis=1) * (fine.size - 1)).astype(int)
    interior = common[left[common] & (depth[common] >= 5)]
    with threadpool_limits(limits=1):
        profiles, kernels, prior, curve = refined_mapper(coarse, fine, anchors, weights)
        batch = next(batch for batch in field_batches(variant, fine, profiles, curve)
                     if batch[:4] == (regime, start, profile, "mixed"))
        truth, readings = (values[draw] for values in batch[4:])
        short_profile = profiles[len(GLOBAL_WEIGHTS) + profile]
        short = benchmark[short_profile[fine.room[benchmark]] > 0.5]
        plans = {}
        for arm in arms:
            route = routes[arm]
            if (route["sites"][0], len(route["sites"]), route["edges"]) != (anchors[start], config["cap"], travel):
                raise ValueError("Room replay must match the start and both realized budgets")
            plans[arm] = pack_plan(fine, common, kernels, np.log(prior), route, truth, readings,
                                  score_groups=dict(interior_rmse=interior, original_rmse=held, benchmark_short_rmse=short))
            plan = plans[arm]
            np.testing.assert_allclose((plan.pop("original_rmse")[-1], plan.pop("benchmark_short_rmse")[-1]),
                                       (float(metric[f"{arm}_original_rmse"]), float(metric[f"{arm}_short_rmse"])),
                                       atol=1e-10, rtol=0)
    scores = []
    for budget in config["budgets"]:
        pair = [source["scores"][f"{method}_{budget}"] for method in ("quarter", "room_quota_two")]
        scores.append(dict(budget=budget, overall=[score["common_rmse"] for score in pair],
                           short=[score["short_rmse"] for score in pair]))
    return dict(layout=pack_layout(fine, np.empty(0, dtype=int)), travel=travel, cap=config["cap"], plans=plans,
                truth=np.rint(truth * 1000).astype(int).tolist(), scores=scores, fields=source["fields"],
                interiorCells=len(interior), evaluationCells=len(common))
