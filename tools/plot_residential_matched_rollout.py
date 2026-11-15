"""Show archived fine greedy and rollout paths at the same realized distance."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
import numpy as np
from threadpoolctl import threadpool_limits

from audit_residential_grid_resolution import holdouts, predictions, refine_geometry, refined_mapper
from audit_residential_refined_rollout import MATCHED_ARMS, PLANNER_ARMS, field_batches
from audit_residential_routes import LayoutCase, floor_plan
from plot_residential_zone_ranges import GOLD, INK, MUTED, PAPER, wall_segments


GREEDY = PLANNER_ARMS[1]
ROLLOUT = MATCHED_ARMS[GREEDY]
ARMS = (GREEDY, ROLLOUT)
COLORS = {GREEDY: "#bd5939", ROLLOUT: "#167488"}
LABELS = {GREEDY: "Greedy", ROLLOUT: "Rollout"}
ZONES = ("Lower-left room", "Lower-right room", "Corridor", "Upper room")


def select_case(study, variant, start):
    with (study / "fields.csv").open() as stream:
        rows = [row for row in csv.DictReader(stream)
                if (row["scenario"], int(row["variant"]), int(row["start"])) == ("short", variant, start)]
    changes = np.asarray([float(row[f"{ROLLOUT}_all"]) - float(row[f"{GREEDY}_all"]) for row in rows])
    return rows[int(np.argmin(abs(changes - changes.mean())))]


def replay(study, variant, start, model, case):
    selected = {row["arm"]: row for row in json.loads((study / "routes.json").read_text())
                if (row["variant"], row["start"]) == (variant, start) and row["arm"] in ARMS}
    coarse = floor_plan(LayoutCase("corridor", variant))
    layout, anchors = refine_geometry(coarse)
    held, _ = holdouts(coarse, layout, anchors, variant)
    if selected[GREEDY]["edges"] != selected[ROLLOUT]["edges"]:
        raise ValueError("The example routes must walk the same distance")
    for arm in ARMS:
        route = selected[arm]
        walk = np.asarray(route["walk"])
        if (len(walk) - 1 != route["edges"] or walk[0] != anchors[start]
                or not layout.movement[walk[:-1], walk[1:]].all() or np.isin(route["sites"], held).any()):
            raise ValueError("Incorrect route resources or evaluation mask")
    profiles, kernels, prior, correlogram = refined_mapper(coarse, layout, anchors, model)
    target = (case["regime"], start, int(case["profile"]))
    batches = field_batches(variant, layout, profiles, correlogram)
    truth, noisy = next((truth, readings) for wall, origin, profile, _, truth, readings in batches
                        if (wall, origin, profile) == target)
    draw = int(case["draw"])
    truth, noisy = truth[draw], noisy[draw]
    means = {}
    for arm in ARMS:
        sites = np.asarray(selected[arm]["sites"])
        mean, _ = predictions(kernels, sites, noisy[sites][None, :], np.arange(len(truth)), np.log(prior))
        means[arm] = mean[0]
        np.testing.assert_allclose(np.sqrt(np.mean((mean[0, held] - truth[held])**2)),
                                   float(case[f"{arm}_all"]), rtol=0, atol=1e-10)
    return layout, held, truth, means, selected


def render(path, layout, held, truth, means, routes):
    size = layout.size
    xy = layout.points * (size-1)
    lim = (-.5, size-.5)
    segments = wall_segments(layout)
    shared = np.intersect1d(routes[GREEDY]["sites"], routes[ROLLOUT]["sites"])
    exclusive = {arm: np.setdiff1d(routes[arm]["sites"], shared) for arm in ARMS}
    errors = {arm: abs(values[held] - truth[held]) for arm, values in means.items()}
    rmse = {arm: np.sqrt(np.mean(error**2)) for arm, error in errors.items()}
    field_limit = np.ceil(2*max(np.max(abs(values)) for values in (truth, *means.values()))) / 2

    fig = plt.figure(figsize=(13.4, 9.0), facecolor=PAPER)
    fig.text(.066, .966, "Greedy and rollout at the same walking distance",
             fontsize=23, color=INK, weight="bold", va="top")
    fig.text(.067, .918,
             f"Short-range corridor · 23 × 23 grid · Both walk {routes[GREEDY]['edges']} steps · 100 hold-outs",
             fontsize=11, color=MUTED, va="top")

    def panel(column, top, title, subtitle, *, draw_walls=True):
        left = (.070, .384, .698)[column]
        bottom = .52 if top else .095
        ax = fig.add_axes([left, bottom, .212, .314])
        ax.set(xlim=lim, ylim=lim, xticks=[], yticks=[], aspect="equal")
        ax.set_facecolor("#edf1ef")
        if draw_walls:
            ax.add_collection(LineCollection(segments, color=INK, linewidth=1.65, zorder=3))
        for spine in ax.spines.values():
            spine.set(color=INK, linewidth=1.1)
        fig.text(left, bottom + .359, title, fontsize=12.5, weight="bold", color=INK, va="top")
        fig.text(left, bottom + .335, subtitle, fontsize=9.3, color=MUTED, va="top")
        return ax

    def field(ax, data):
        return ax.imshow(data.reshape(size, size), extent=lim + lim, origin="lower", cmap="RdBu_r",
                         norm=Normalize(-field_limit, field_limit), interpolation="nearest")

    def colorbar(image, rect, label, ticks):
        bar = fig.colorbar(image, cax=fig.add_axes(rect), orientation="horizontal", ticks=ticks)
        bar.set_label(label, fontsize=8.5, color=MUTED, labelpad=2)
        bar.outline.set_visible(False)
        bar.ax.tick_params(labelsize=8, colors=MUTED, length=2, pad=2)

    def stops(ax, indices, color, size=22):
        ax.scatter(*xy[indices].T, s=size, color=color, edgecolors="white", linewidths=.65, zorder=6)

    ax = panel(0, True, "A   True sensor field", "One archived field evaluated under both routes")
    image = field(ax, truth)
    for zone, tag in enumerate(("Lower left", "Lower right", "Corridor", "Upper room")):
        x, y = xy[layout.room == zone].mean(axis=0)
        ax.text(x, y, tag, ha="center", va="center", fontsize=8, color=INK, zorder=5,
                bbox=dict(boxstyle="round,pad=.24", facecolor=PAPER, edgecolor="none", alpha=.94))

    for col, arm in enumerate(ARMS, 1):
        route = routes[arm]
        ax = panel(col, True, f"{chr(65+col)}   {LABELS[arm]} route + reconstruction",
                   f"{route['readings']} readings · Hold-out RMSE {rmse[arm]:.4f}")
        field(ax, means[arm])
        walk = np.asarray(route["walk"])
        ax.plot(*xy[walk].T, color="white", lw=3.5, zorder=4)
        ax.plot(*xy[walk].T, color=COLORS[arm], lw=1.5, zorder=5)
        stops(ax, exclusive[arm], COLORS[arm], size=22)
        stops(ax, shared, GOLD, size=27)
        ax.scatter(*xy[walk[-1]], s=59, marker="s", facecolors="none",
                   edgecolors=INK, linewidths=1.15, zorder=7)
        ax.scatter(*xy[walk[0]], s=48, marker="D", color=GOLD, edgecolors=INK, linewidths=.9, zorder=8)
    colorbar(image, [.336, .489, .306, .012], "Latent sensor value", [-field_limit, 0, field_limit])

    ax = panel(0, False, "D   Where the robot measured", f"{len(shared)} shared stop · colored dots mark exclusive stops")
    for arm in ARMS:
        stops(ax, exclusive[arm], COLORS[arm], size=30)
    stops(ax, shared, GOLD, size=36)
    fig.legend(handles=[
        Line2D([], [], marker="D", color=INK, markerfacecolor=GOLD, linestyle="none", label="Shared start"),
        Line2D([], [], marker="s", color=INK, markerfacecolor="none", linestyle="none", label="Matched stop point"),
        Line2D([], [], marker="o", color=COLORS[GREEDY], linestyle="none", label="Greedy only"),
        Line2D([], [], marker="o", color=COLORS[ROLLOUT], linestyle="none", label="Rollout only"),
    ], loc="upper right", bbox_to_anchor=(.94, .927), fontsize=8, labelcolor=MUTED, ncol=2, frameon=False,
       columnspacing=1.0, handletextpad=.25)

    ax = panel(1, False, "E   Change in hold-out error", "Teal is lower error with rollout")
    delta = errors[ROLLOUT] - errors[GREEDY]
    error_limit = np.ceil(np.max(abs(delta))*10) / 10
    cmap = LinearSegmentedColormap.from_list("relative_error", ["#006d80", "#faf9f5", "#c24232"])
    diff = ax.scatter(*xy[held].T, c=delta, norm=Normalize(-error_limit, error_limit),
                      cmap=cmap, s=21, edgecolors=INK, linewidths=.38, zorder=5)
    colorbar(diff, [.394, .064, .192, .012], "Rollout − greedy absolute error",
             [-error_limit, 0, error_limit])

    ax = panel(2, False, "F   Readings and error by zone", "Hold-out RMSE is computed within each zone", draw_walls=False)
    ax.axis("off")
    ax.text(.55, .91, "Readings", ha="center", transform=ax.transAxes, fontsize=9, color=MUTED)
    ax.text(.85, .91, "RMSE", ha="center", transform=ax.transAxes, fontsize=9, color=MUTED)
    for zone, y in zip((3, 2, 0, 1), (.76, .57, .38, .19)):
        counts = [np.count_nonzero(layout.room[routes[arm]["sites"]] == zone) for arm in ARMS]
        mask = layout.room[held] == zone
        scores = [np.sqrt(np.mean(errors[arm][mask]**2)) for arm in ARMS]
        ax.text(.025, y, ZONES[zone], transform=ax.transAxes, fontsize=9, color=INK, va="center")
        ax.text(.55, y, f"{counts[0]} → {counts[1]}", transform=ax.transAxes,
                fontsize=9, color=INK, ha="center", va="center")
        ax.text(.85, y, f"{scores[0]:.2f} → {scores[1]:.2f}", transform=ax.transAxes,
                fontsize=9, color=INK, ha="center", va="center")
        ax.plot([.025, .98], [y-.082, y-.082], transform=ax.transAxes, lw=.6, color="#d3dbd9")
    ax.text(.025, .03, "Greedy → rollout", transform=ax.transAxes, fontsize=8, color=MUTED)

    fig.savefig(path, dpi=188, facecolor=PAPER)
    plt.close(fig)
    for arm in ARMS:
        print(LABELS[arm], routes[arm]["readings"], rmse[arm],
              [(ZONES[z], int(np.count_nonzero(layout.room[routes[arm]["sites"]] == z)),
                float(np.sqrt(np.mean(errors[arm][layout.room[held] == z]**2)))) for z in range(len(ZONES))])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("outputs/sensing-world-residential-refined-rollout"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    protocol = json.loads((args.study / "protocol.json").read_text())
    model = json.loads(Path(protocol["model"]).read_text())
    # Keep the same floor plan and starting point as the prior resolution figure.
    variant, start = 1, 14
    case = select_case(args.study, variant, start)
    output = args.output or args.study / "short-range-matched-routes.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        render(output, *replay(args.study, variant, start, model, case))
    print(f"Rendered layout {variant}, start {start}, {case['regime']} walls, field draw {case['draw']}")


if __name__ == "__main__":
    main()
