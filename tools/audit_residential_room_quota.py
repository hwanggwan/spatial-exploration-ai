"""Compare refined rollout with minimum numbers of readings in each room."""

import argparse
import csv
from itertools import permutations
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from audit_residential_adaptive_rollout import EDGES, MATCHED_OUTPUT, rollout_move
from audit_residential_grid_resolution import expand_walk, holdouts, refine_geometry, refined_mapper, route_resources
from audit_residential_holdout_distance import CAP, FAMILY
from audit_residential_refined_rollout import field_batches, field_key
from audit_residential_room_rollout import EARLY_WEIGHT, ROOMS
from audit_residential_routes import CONDUCTANCES, LayoutCase, STARTS, floor_plan
from audit_residential_zone_ranges import GLOBAL_WEIGHTS, map_predictions
from audit_sensing_world_modes import posterior
from spatial_exploration.baseline import GPBelief, write_csv
from spatial_exploration.sensing_world_study import NOISE


ARMS = ("quarter", "room_quota")
TWO_ARM = "room_quota_two"
ROOM_NAMES = {0: "Lower left", 1: "Lower right", 3: "Upper"}
EXAMPLE = (1, 64)


class RoomCompletion:
    """Exact minimum travel from each cell to obtain a reading in each remaining room."""

    def __init__(self, zones, distances, allowed, rooms):
        self.distances = distances
        self.bits = np.zeros(len(zones), dtype=int)
        room_sites = {}
        for index, room in enumerate(rooms):
            bit = 1 << index
            self.bits[zones == room] = bit
            room_sites[bit] = np.flatnonzero(allowed & (zones == room))
        self.full_mask = (1 << len(rooms)) - 1
        self.costs = np.zeros((self.full_mask + 1, len(zones)))
        self.counts = np.zeros(self.full_mask + 1, dtype=int)
        for mask in range(1, self.full_mask + 1):
            best = np.full(len(zones), np.inf)
            for bit, sites in room_sites.items():
                if mask & bit:
                    rest = mask ^ bit
                    costs = distances[:, sites] + self.costs[rest, sites]
                    best = np.minimum(best, costs.min(axis=1, initial=np.inf))
                    self.counts[mask] = self.counts[rest] + 1
            self.costs[mask] = best

    def candidates(self, missing, current, sites, remaining, readings_left, suffix):
        measured = np.bitwise_or.reduce(self.bits[np.asarray(suffix, dtype=int)], initial=0)
        after = (missing & ~measured) & ~self.bits[sites]
        return ((self.counts[after] <= readings_left - 1)
                & (self.distances[current, sites] + self.costs[after, sites] <= remaining + 1e-12))

    def reserve(self, observed):
        missing = self.full_mask & ~np.bitwise_or.reduce(self.bits[observed])
        if not missing:
            return None

        def can_complete(current, candidates, remaining, readings_left, suffix):
            return self.candidates(missing, current, candidates, remaining, readings_left, suffix)

        return can_complete


class TwoReadingCompletion:
    """Certify a completion by grouping a room's pending readings into one stop sequence."""

    def __init__(self, layout, allowed):
        self.zones = layout.room
        self.distances = layout.distances
        self.portals = {}
        self.single = {}
        self.double = {}
        self.local_next = {}
        for room in ROOMS:
            inside = layout.room == room
            sites = np.flatnonzero(inside & allowed)
            ports = np.flatnonzero(inside & layout.movement[:, ~inside].any(axis=1))
            if len(sites) < 2 or not len(ports):
                raise ValueError("Each room needs two eligible cells and a doorway")
            self.portals[room] = ports
            exits = np.column_stack((self.distances[np.ix_(sites, ports)], np.zeros(len(sites))))
            entries = self.distances[np.ix_(ports, sites)]
            via_site = entries[:, :, None] + exits[None, :, :]
            best_two = np.argsort(via_site, axis=1, kind="stable")[:, :2, :]
            best_costs = np.take_along_axis(via_site, best_two, axis=1)
            exclude = np.arange(len(inside))[:, None, None] == sites[best_two[:, 0, :]][None, :, :]
            self.single[room] = np.where(exclude, best_costs[:, 1, :], best_costs[:, 0, :])

            between = self.distances[np.ix_(sites, sites)].copy()
            np.fill_diagonal(between, np.inf)
            after_first = (between[:, :, None] + exits[None, :, :]).min(axis=1)
            self.double[room] = (entries[:, :, None] + after_first[None, :, :]).min(axis=1)

            local = self.distances[:, sites].copy()
            local[sites, np.arange(len(sites))] = np.inf
            self.local_next[room] = (local[:, :, None] + exits[None, :, :]).min(axis=1)

    def ordered_cost(self, sites, new_zone, order, pending, excluded):
        downstream = None
        for i in range(len(order)-1, -1, -1):
            room = order[i]
            ports = self.portals[room]
            if i == 0 and room == new_zone:
                transfer = self.local_next[room][sites]
            elif pending[room] == 2:
                transfer = self.double[room]
            else:
                blocked = sites if room == new_zone else excluded[room]
                transfer = self.single[room][blocked]

            if downstream is None:
                cost = transfer[..., -1]
            elif i == 0 and room == new_zone:
                cost = (transfer[..., :-1] + downstream).min(axis=-1)
            else:
                cost = (transfer[..., :-1] + downstream[..., None, :]).min(axis=-1)

            if i == 0:
                if room == new_zone:
                    return cost
                entry = self.distances[np.ix_(sites, ports)]
                return (entry + cost).min(axis=-1)
            inter_room = self.distances[np.ix_(self.portals[order[i-1]], ports)]
            downstream = (inter_room + cost[..., None, :]).min(axis=-1)

    def candidates(self, observed, current, sites, remaining, readings_left):
        seen_zones = self.zones[observed]
        counts = np.bincount(seen_zones, minlength=int(self.zones.max())+1)
        excluded = {room: observed[seen_zones == room][0] for room in ROOMS if counts[room] == 1}
        allowed = np.zeros(len(sites), dtype=bool)
        for new_zone in np.unique(self.zones[sites]):
            mask = self.zones[sites] == new_zone
            group = sites[mask]
            pending = {room: max(2-counts[room]-int(room == new_zone), 0) for room in ROOMS}
            needed = tuple(room for room in ROOMS if pending[room])
            if sum(pending.values()) > readings_left - 1:
                continue
            cost = self.distances[current, group]
            if needed:
                best = np.full(len(group), np.inf)
                for order in permutations(needed):
                    best = np.minimum(best, self.ordered_cost(group, new_zone, order, pending, excluded))
                cost = cost + best
            allowed[mask] = cost <= remaining + 1e-12
        return allowed

    def reserve(self, observed):
        prefix = np.asarray(observed, dtype=int)
        counts = np.bincount(self.zones[prefix], minlength=int(self.zones.max())+1)
        if all(counts[room] >= 2 for room in ROOMS):
            return None

        def can_complete(current, candidates, remaining, readings_left, suffix):
            taken = np.concatenate((prefix, np.asarray(suffix, dtype=int)))
            return self.candidates(taken, current, candidates, remaining, readings_left)

        return can_complete


def plan_route(layout, anchors, allowed, prior, start, completion=None, *, minimum=0, edge_cap=EDGES):
    origin = int(anchors[start])
    available = allowed.copy()
    sites = [origin]
    travel = 0.0
    cap = edge_cap / (layout.size - 1)
    belief = GPBelief(prior, NOISE)
    belief.observe(origin, 0.0)
    while len(sites) < CAP:
        current = sites[-1]
        available[current] = False
        remaining = cap - travel
        candidates = np.flatnonzero(available & (layout.distances[current] <= remaining + 1e-12))
        if not len(candidates):
            break
        can_complete = completion.reserve(sites) if completion is not None else None
        action = rollout_move(belief.covariance, layout.distances, current, candidates, remaining,
                              CAP-len(sites), cap, can_complete=can_complete)
        travel += layout.distances[current, action]
        sites.append(action)
        belief.observe(action, 0.0)
    sites = np.asarray(sites)
    walk = expand_walk(layout, sites)
    arrivals = np.r_[0, np.rint(layout.distances[sites[:-1], sites[1:]] * (layout.size-1)).astype(int).cumsum()]
    np.testing.assert_array_equal(walk[arrivals], sites)
    resources = route_resources(layout, walk, sites, anchors, edge_cap)
    counts = np.bincount(layout.room[sites], minlength=4)
    if any(counts[room] < minimum for room in ROOMS):
        raise RuntimeError("The route stopped before completing its room quota")
    first_reading = {room: int(arrivals[layout.room[sites] == room][0]) if counts[room] else None for room in ROOMS}
    first_entry = {room: int(np.flatnonzero(layout.room[walk] == room)[0]) if room in layout.room[walk] else None for room in ROOMS}
    reading_steps = {room: arrivals[layout.room[sites] == room][:2].tolist() for room in ROOMS}
    pair_distance = {room: round(layout.distances[tuple(sites[layout.room[sites] == room][:2])] * (layout.size-1))
                     if counts[room] >= 2 else None for room in ROOMS}
    return dict(**resources, sites=sites.tolist(), walk=walk.tolist(), arrivals=arrivals.tolist(),
                room_readings=counts.tolist(), first_room_reading=first_reading, first_room_entry=first_entry,
                first_two_room_steps=reading_steps, first_pair_distance=pair_distance,
                terminal_variance=float(belief.covariance.diagonal().mean()))


def evaluate(variant, model, archive, previous, draws, minimum):
    coarse = floor_plan(LayoutCase(FAMILY, variant))
    layout, anchors = refine_geometry(coarse)
    held, _ = holdouts(coarse, layout, anchors, variant)
    profiles, kernels, weights, curve = refined_mapper(coarse, layout, anchors, model)
    prior = np.einsum("m,mij->ij", weights, kernels)
    grid = np.arange(len(layout.room))
    allowed = ~np.isin(grid, held)
    completion = RoomCompletion(layout.room, layout.distances, allowed, ROOMS)
    policies = {"quarter": (None, 0), "room_quota": (completion, 1)}
    if minimum == 2:
        policies[TWO_ARM] = (TwoReadingCompletion(layout, allowed), 2)
    routes = {}
    for start in STARTS:
        for arm, (constraint, readings) in policies.items():
            before = perf_counter()
            route = plan_route(layout, anchors, allowed, prior, start, constraint, minimum=readings)
            route["plan_ms"] = (perf_counter() - before) * 1000
            if arm == "quarter":
                for name in ("sites", "walk", "arrivals"):
                    np.testing.assert_array_equal(route[name], archive[variant, start][name])
            routes[start, arm] = dict(variant=variant, start=start, arm=arm, **route)
        print(f"Verified {len(policies)} routes for layout {variant}, start {start}", flush=True)
    rows = score_mixed_fields(variant, layout, held, profiles, kernels, weights, curve, routes, previous, draws)
    return rows, list(routes.values())


def score_mixed_fields(variant, layout, held, profiles, kernels, weights, curve, routes, previous, draws,
                       *, baseline_arm="quarter"):
    arms = dict.fromkeys(arm for _, arm in routes)
    grid = np.arange(len(layout.room))
    common = {start: np.setdiff1d(grid, np.concatenate([routes[start, arm]["sites"] for arm in arms]))
              for start in STARTS}
    log_prior = np.log(weights)
    mode_profiles = np.repeat(profiles, len(kernels)//len(profiles), axis=0)
    mixed = profiles[len(GLOBAL_WEIGHTS):]
    rows = []
    for regime, start, profile, scenario, truth, noisy in field_batches(variant, layout, profiles, curve):
        if regime != "weak" or scenario != "mixed":
            continue
        cells = common[start]
        short = mixed[profile, layout.room[cells]] > .5
        short_rooms = mixed[profile, list(ROOMS)] > .5
        registered = np.isin(cells, held)
        metrics = {}
        for arm in arms:
            sites = routes[start, arm]["sites"]
            components, likelihood, variance = posterior(kernels, sites, noisy[:draws, sites], cells, with_variance=True)
            mean, sd, probabilities = map_predictions(components, likelihood, variance, log_prior)
            error = mean - truth[:draws, cells]
            groups = dict(common=np.ones(len(cells), dtype=bool), short=short, long=~short,
                          original=registered, original_short=registered & short)
            for group, mask in groups.items():
                metrics[f"{arm}_{group}_rmse"] = np.sqrt(np.mean(error[:, mask]**2, axis=1))
            metrics[f"{arm}_coverage"] = np.mean(abs(error) <= 1.96 * sd, axis=1)
            estimated = probabilities @ mode_profiles[:, list(ROOMS)]
            metrics[f"{arm}_range_accuracy"] = np.mean((estimated > .5) == short_rooms, axis=1)
            metrics[f"{arm}_range_mae"] = np.mean(abs(estimated - mixed[profile, list(ROOMS)]), axis=1)
        for draw in range(draws):
            row = dict(variant=variant, regime=regime, start=start, profile=profile, draw=draw, scenario=scenario,
                       cells=len(cells), short_cells=int(short.sum()), **{k: result[draw] for k, result in metrics.items()})
            expected = previous[field_key(row)]
            np.testing.assert_allclose((row[f"{baseline_arm}_original_rmse"], row[f"{baseline_arm}_original_short_rmse"]),
                                       (float(expected["quarter_rmse"]), float(expected["quarter_short_rmse"])),
                                       atol=1e-10, rtol=0)
            rows.append(row)
    return rows


def paired_resamples(rows):
    rng = np.random.default_rng(314809)
    strata = {}
    for i, row in enumerate(rows):
        key = tuple(row[k] for k in ("variant", "start", "profile"))
        strata.setdefault(key, []).append(i)
    return np.concatenate([rng.choice(ids, (3000, len(ids))) for ids in strata.values()], axis=1)


def summarize(rows, routes, arms):
    boot = paired_resamples(rows)
    summary = dict(fields=len(rows), mean_cells=float(np.mean([row["cells"] for row in rows])))
    for name in ("common_rmse", "short_rmse", "long_rmse", "original_rmse", "original_short_rmse", "coverage",
                 "range_accuracy", "range_mae"):
        values = {arm: np.array([r[f"{arm}_{name}"] for r in rows]) for arm in arms}
        summary[name] = {arm: float(metric.mean()) for arm, metric in values.items()}
        for base in arms[:-1]:
            delta = values[arms[-1]] - values[base]
            contrast = dict(delta=float(delta.mean()), ci95=np.quantile(delta[boot].mean(axis=1), [.025, .975]).tolist())
            if len(arms) == 2:
                summary[name].update(contrast)
            else:
                summary[name][f"{arms[-1]}_minus_{base}"] = contrast
    summary["routes"] = {}
    for arm in arms:
        selected = [r for r in routes if r["arm"] == arm]
        summary["routes"][arm] = dict(
            **{metric: float(np.mean([row[metric] for row in selected]))
               for metric in ("edges", "readings", "plan_ms", "terminal_variance")},
            all_rooms=sum(all(r["room_readings"][room] for room in ROOMS) for r in selected),
            twice_all_rooms=sum(all(r["room_readings"][room] >= 2 for room in ROOMS) for r in selected),
        )
    return summary


def render(output, routes, arms):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    from plot_residential_early_rollout_routes import draw_route
    from plot_residential_zone_ranges import GOLD, INK, MUTED, PAPER

    variant, start = EXAMPLE
    selected = {r["arm"]: r for r in routes if (r["variant"], r["start"]) == EXAMPLE}
    layout, anchors = refine_geometry(floor_plan(LayoutCase(FAMILY, variant)))
    fig = plt.figure(figsize=(12.8 if len(arms) == 3 else 10.2, 5.9 if len(arms) == 3 else 6.35), facecolor=PAPER)
    fig.text(.07, .95, "How the room requirement changes the route", color=INK,
             fontsize=21, weight="bold", va="top")
    fig.text(.071, .889, "Same hallway start  /  At most 52 walking steps and 24 readings", color=MUTED,
             fontsize=10, va="top")
    norm = Normalize(0, EDGES)
    points = layout.points * (layout.size - 1)
    positions = (.067, .373, .679) if len(arms) == 3 else (.071, .54)
    width = .252 if len(arms) == 3 else .385
    titles = ("Existing rollout", "Minimum one per room", "Minimum two per room")
    for arm, left, title in zip(arms, positions, titles):
        route = selected[arm]
        ax = fig.add_axes([left, .224, width, .54])
        image = draw_route(ax, layout, anchors[start], route, norm)
        fig.text(left, .802, title, fontsize=13, weight="bold", color=INK)
        fig.text(left, .773, f"{route['edges']} steps  ·  {route['readings']} readings", fontsize=9.4, color=MUTED)
        for room, step in route["first_room_reading"].items():
            if step is None:
                point = points[layout.room == room].mean(axis=0)
                ax.text(*point, "unmeasured", ha="center", va="center", fontsize=8, color=MUTED)
                continue
            point = points[route["walk"][step]]
            ax.scatter(*point, s=120, facecolors="none", edgecolors="white", linewidths=2.5, zorder=9)
            ax.scatter(*point, s=116, facecolors="none", edgecolors=INK, linewidths=.9, zorder=10)
            offset = (7, -11) if point[1] >= 10 else (8, 7)
            ax.annotate(str(step), point, xytext=offset, textcoords="offset points", fontsize=8, weight="bold",
                        color=INK, bbox=dict(boxstyle="round,pad=.19", facecolor=PAPER, edgecolor="none"), zorder=11)
        if arm == TWO_ARM and len(route["first_two_room_steps"][0]) == 2:
            step = route["first_two_room_steps"][0][1]
            point = points[route["walk"][step]]
            ax.annotate(str(step), point, xytext=(-13, 6), textcoords="offset points", fontsize=8, weight="bold",
                        color=INK, bbox=dict(boxstyle="round,pad=.19", facecolor=PAPER, edgecolor="none"), zorder=11)
        by_room = " / ".join(f"{ROOM_NAMES[r]} {route['room_readings'][r]}" for r in ROOMS)
        fig.text(left+.006, .179, by_room, fontsize=8.6, color=MUTED)

    colorbar = fig.colorbar(image, cax=fig.add_axes([.076,.104,.39,.018]), orientation="horizontal", ticks=[0,13,26,39,52])
    colorbar.set_label("Walking step at each reading", color=MUTED, fontsize=8.7, labelpad=2)
    colorbar.ax.tick_params(labelsize=8, colors=MUTED, pad=2)
    colorbar.outline.set_visible(False)
    handles = [Line2D([], [], marker="D", markerfacecolor=GOLD, markeredgecolor=INK, linestyle="none", label="Start"),
               Line2D([], [], marker="o", markerfacecolor="none", markeredgecolor=INK, linestyle="none", label="First room reading")]
    fig.legend(handles=handles, loc="center right", bbox_to_anchor=(.937,.105), ncol=2, frameon=False,
               labelcolor=MUTED, fontsize=9, columnspacing=1.1)
    fig.savefig(output, dpi=190, facecolor=PAPER)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=MATCHED_OUTPUT)
    parser.add_argument("--minimum", type=int, choices=(1, 2), default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    suffix = "-two" if args.minimum == 2 else ""
    args.output = args.output or Path(f"outputs/sensing-world-residential-room-quota{suffix}")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use an empty output directory.")
    meta = json.loads((args.source / "protocol.json").read_text())
    if (meta["cap"], meta["edge_cap"], meta["early_weight"]) != (CAP, EDGES, EARLY_WEIGHT):
        parser.error("Expected the 52-edge, 24-reading, 25% baseline protocol.")
    model = json.loads(Path(meta["model"]).read_text())
    archive = {(r["variant"], r["start"]): r for r in json.loads((Path(meta["source"]) / "routes.json").read_text())
               if r["arm"] == "quarter"}
    with (args.source / "fields.csv").open() as stream:
        previous = {field_key(r): r for r in csv.DictReader(stream) if (r["regime"], r["scenario"]) == ("weak", "mixed")}
    rows, routes = [], []
    with threadpool_limits(limits=1):
        for variant in (0, 1):
            fields, planned = evaluate(variant, model, archive, previous, meta["draws"], args.minimum)
            rows.extend(fields)
            routes.extend(planned)
    if {field_key(r) for r in rows} != previous.keys():
        raise ValueError("The replay must cover all the archived weak mixed fields")
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "fields.csv", rows)
    protocol = dict(source=str(args.source), model=meta["model"], regime="weak", conductance=CONDUCTANCES["weak"],
                    cap=CAP, edge_cap=EDGES, early_weight=EARLY_WEIGHT, rooms=ROOMS, minimum_readings=args.minimum,
                    example=dict(variant=EXAMPLE[0], start=EXAMPLE[1]))
    arms = (*ARMS, TWO_ARM) if args.minimum == 2 else ARMS
    for name, result in (("protocol", protocol), ("routes", routes), ("summary", summarize(rows, routes, arms))):
        (args.output / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
    render(args.output / "route-comparison.png", routes, arms)


if __name__ == "__main__":
    main()
