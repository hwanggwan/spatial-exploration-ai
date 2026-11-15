"""Compare longer walking budgets with a fixed reading cap on the refined corridor grid."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from functools import partial
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from audit_residential_grid_resolution import holdouts, refine_geometry, refined_mapper
from audit_residential_holdout_distance import CAP, FAMILY
from audit_residential_refined_rollout import field_key
from audit_residential_room_quota import EXAMPLE, ROOMS, TWO_ARM, TwoReadingCompletion, paired_resamples, plan_route, score_mixed_fields
from audit_residential_routes import LayoutCase, STARTS, floor_plan
from spatial_exploration.baseline import write_csv


BUDGETS = (52, 104, 156)
METHODS = ("quarter", TWO_ARM)
METRICS = ("common_rmse", "short_rmse", "long_rmse", "original_rmse", "original_short_rmse",
           "coverage", "range_accuracy", "range_mae")


def room_geometry(layout, route):
    sites = np.asarray(route["sites"])
    depth, span = {}, {}
    for room in ROOMS:
        inside = layout.room == room
        doors = np.flatnonzero(inside & layout.movement[:, ~inside].any(axis=1))
        measured = sites[inside[sites]]
        if not len(measured):
            depth[room] = span[room] = None
            continue
        depth[room] = round(layout.distances[np.ix_(measured, doors)].min(axis=1).max() * (layout.size-1))
        span[room] = round(layout.distances[np.ix_(measured, measured)].max() * (layout.size-1))
    return dict(deepest_room_reading=depth, widest_room_pair=span)


def evaluate(variant, model, archive, previous, draws):
    with threadpool_limits(limits=1):
        return evaluate_layout(variant, model, archive, previous, draws)


def evaluate_layout(variant, model, archive, previous, draws):
    coarse = floor_plan(LayoutCase(FAMILY, variant))
    layout, anchors = refine_geometry(coarse)
    held, _ = holdouts(coarse, layout, anchors, variant)
    profiles, kernels, weights, curve = refined_mapper(coarse, layout, anchors, model)
    prior = np.einsum("m,mij->ij", weights, kernels)
    allowed = ~np.isin(np.arange(len(layout.room)), held)
    completion = TwoReadingCompletion(layout, allowed)
    routes = {}
    for start in STARTS:
        for budget in BUDGETS:
            for method in METHODS:
                arm = f"{method}_{budget}"
                if budget == BUDGETS[0]:
                    route = archive[variant, start, method]
                else:
                    before = perf_counter()
                    quota = method == TWO_ARM
                    route = plan_route(layout, anchors, allowed, prior, start, completion if quota else None,
                                       minimum=2 if quota else 0, edge_cap=budget)
                    route["plan_ms"] = (perf_counter() - before) * 1000
                sites = np.asarray(route["sites"])
                if (sites[0] != anchors[start] or not allowed[sites].all() or route["edges"] > budget
                        or route["readings"] > CAP or (method == TWO_ARM and any(route["room_readings"][r] < 2 for r in ROOMS))):
                    raise ValueError("A saved route violates the start, caps, or room requirement")
                routes[start, arm] = route | dict(variant=variant, start=start, arm=arm, method=method, budget=budget,
                                                  **room_geometry(layout, route))
        print(f"Completed six routes for layout {variant}, start {start}", flush=True)
    rows = score_mixed_fields(variant, layout, held, profiles, kernels, weights, curve, routes, previous, draws,
                              baseline_arm="quarter_52")
    return rows, list(routes.values())


def summarize(rows, routes):
    boot = paired_resamples(rows)
    arms = [f"{method}_{budget}" for budget in BUDGETS for method in METHODS]
    output = dict(fields=len(rows), mean_cells=float(np.mean([r["cells"] for r in rows])), scores={}, routes={}, contrasts={})
    comparisons = [(f"{method}_{budget}", f"{method}_52") for method in METHODS for budget in BUDGETS[1:]]
    comparisons += [(f"{TWO_ARM}_{budget}", f"quarter_{budget}") for budget in BUDGETS]
    comparisons += [(f"{method}_156", f"{method}_104") for method in METHODS]
    values = {arm: {metric: np.asarray([r[f"{arm}_{metric}"] for r in rows]) for metric in METRICS} for arm in arms}
    for arm in arms:
        output["scores"][arm] = {metric: float(v.mean()) for metric, v in values[arm].items()}
        selected = [r for r in routes if r["arm"] == arm]
        resource = {metric: float(np.mean([r[metric] for r in selected])) for metric in ("edges", "readings", "plan_ms")}
        resource.update(all_rooms=sum(all(r["room_readings"][room] for room in ROOMS) for r in selected),
                        twice_all_rooms=sum(all(r["room_readings"][room] >= 2 for room in ROOMS) for r in selected),
                        reached_reading_cap=sum(r["readings"] == CAP for r in selected),
                        reached_travel_cap=sum(r["edges"] == r["budget"] for r in selected))
        for metric in ("deepest_room_reading", "widest_room_pair", "first_pair_distance"):
            lengths = [value for r in selected for value in r[metric].values() if value is not None]
            resource[metric] = dict(mean=float(np.mean(lengths)), median=float(np.median(lengths)), rooms=len(lengths))
        output["routes"][arm] = resource
    for a, b in comparisons:
        changes = {}
        for metric in METRICS:
            delta = values[a][metric] - values[b][metric]
            changes[metric] = dict(delta=float(delta.mean()), ci95=np.quantile(delta[boot].mean(axis=1), [.025,.975]).tolist())
        output["contrasts"][f"{a}_minus_{b}"] = changes
    return output


def render(path, routes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    from plot_residential_early_rollout_routes import draw_route
    from plot_residential_zone_ranges import GOLD, INK, MUTED, PAPER

    variant, start = EXAMPLE
    layout, anchors = refine_geometry(floor_plan(LayoutCase(FAMILY, variant)))
    selected = {r["budget"]: r for r in routes if (r["variant"], r["start"], r["method"]) == (*EXAMPLE, TWO_ARM)}
    fig = plt.figure(figsize=(12.7, 5.8), facecolor=PAPER)
    fig.text(.065, .949, "Where longer walking budgets move the readings", color=INK,
             fontsize=22, weight="bold", va="top")
    fig.text(.066, .885, "Same hallway start  /  At least two readings per room  /  At most 24 readings total",
             color=MUTED, fontsize=10, va="top")
    for left, budget in zip((.065,.371,.677), BUDGETS):
        route = selected[budget]
        ax = fig.add_axes([left, .222, .254, .527])
        image = draw_route(ax, layout, anchors[start], route, Normalize(0, 1))
        image.set_array(np.asarray(route["arrivals"][1:]) / route["edges"])
        fig.text(left, .804, f"{budget}-step budget", fontsize=14, color=INK, weight="bold")
        fig.text(left, .772, f"{route['edges']} steps walked  ·  {route['readings']} readings", fontsize=9.2, color=MUTED)
        counts = route["room_readings"]
        depth = {int(room): length for room, length in route["deepest_room_reading"].items()}[0]
        fig.text(left, .174, f"Left room  ·  {counts[0]} readings  ·  max depth {depth} steps", fontsize=8.8, color=MUTED)
    bar = fig.colorbar(image, cax=fig.add_axes([.069,.108,.36,.019]), orientation="horizontal", ticks=(0,.5,1))
    bar.ax.set_xticklabels(("Start", "Halfway", "End"))
    bar.ax.tick_params(labelsize=8, colors=MUTED, pad=2)
    bar.set_label("Progress along each measured route", fontsize=8.5, color=MUTED, labelpad=2)
    bar.outline.set_visible(False)
    markers = [Line2D([], [], marker="D", markerfacecolor=GOLD, markeredgecolor=INK, linestyle="none", label="Start"),
               Line2D([], [], marker="s", markerfacecolor="none", markeredgecolor=INK, linestyle="none", label="End")]
    fig.legend(handles=markers, loc="center right", bbox_to_anchor=(.931,.108), ncol=2, frameon=False, labelcolor=MUTED)
    fig.savefig(path, dpi=190, facecolor=PAPER)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("outputs/sensing-world-residential-room-quota-two"))
    parser.add_argument("--output", type=Path, default=Path("outputs/sensing-world-residential-travel-budget"))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use an empty output directory.")
    source = json.loads((args.source / "protocol.json").read_text())
    if (source["cap"], source["edge_cap"], source["minimum_readings"]) != (CAP, BUDGETS[0], 2):
        parser.error("Expected the 52-edge, 24-reading archive with the two-observation policy.")
    model = json.loads(Path(source["model"]).read_text())
    inputs = Path(source["source"])
    draws = json.loads((inputs / "protocol.json").read_text())["draws"]
    archive = {(r["variant"], r["start"], r["arm"]): r for r in json.loads((args.source / "routes.json").read_text())
               if r["arm"] in METHODS}
    with (inputs / "fields.csv").open() as stream:
        previous = {field_key(r): r for r in csv.DictReader(stream) if (r["regime"],r["scenario"]) == ("weak","mixed")}
    run = partial(evaluate, model=model, archive=archive, previous=previous, draws=draws)
    rows, routes = [], []
    with ProcessPoolExecutor(max_workers=2) as pool:
        for fields, traveled in pool.map(run, (0,1)):
            rows.extend(fields)
            routes.extend(traveled)
    if {field_key(row) for row in rows} != previous.keys():
        raise ValueError("The replay must cover all archived weak mixed fields")
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "fields.csv", rows)
    protocol = dict(source=str(args.source), budgets=BUDGETS, cap=CAP, model=source["model"],
                    regime="weak", scenario="mixed", example=dict(variant=EXAMPLE[0], start=EXAMPLE[1]))
    for name, result in (("protocol", protocol), ("routes", routes), ("summary", summarize(rows, routes))):
        (args.output / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
    render(args.output / "route-comparison.png", routes)


if __name__ == "__main__":
    main()
