"""Plot the archived 0%, 25%, and 50% rollout routes at two shared starts."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np

from audit_residential_grid_resolution import refine_geometry
from audit_residential_routes import LayoutCase, floor_plan
from plot_residential_zone_ranges import GOLD, INK, MUTED, PAPER, wall_segments


ARMS = (
    ("terminal", "0% · Final uncertainty"),
    ("quarter", "25% · During-walk weight"),
    ("early", "50% · During-walk weight"),
)
EXAMPLES = (
    (1, 14, "Demo start · Corridor B"),
    (0, 64, "Hallway start · Corridor A"),
)


def draw_route(ax, layout, anchor, route, norm):
    coordinates = layout.points * (layout.size - 1)
    walk = np.asarray(route["walk"])
    sites, arrivals = (np.asarray(route[name]) for name in ("sites", "arrivals"))
    if (len(walk) - 1 != route["edges"] or walk[0] != anchor or arrivals[-1] != route["edges"]
            or not layout.movement[walk[:-1], walk[1:]].all()):
        raise ValueError("Archived walk does not match the grid or start")
    np.testing.assert_array_equal(walk[arrivals], sites)
    boundary = (-0.5, layout.size - 0.5)
    ax.set(xlim=boundary, ylim=boundary, aspect="equal", xticks=[], yticks=[])
    ax.set_facecolor("#edf2ef")
    for spine in ax.spines.values():
        spine.set(color=INK, linewidth=1.1)
    ax.add_collection(LineCollection(wall_segments(layout), color=INK, linewidth=1.8, zorder=3))
    ax.plot(*coordinates[walk].T, color="white", linewidth=3.5, solid_capstyle="round", zorder=4)
    ax.plot(*coordinates[walk].T, color="#637d76", linewidth=1.5, solid_capstyle="round", zorder=5)
    image = ax.scatter(*coordinates[sites[1:]].T, c=arrivals[1:], cmap="viridis", norm=norm,
                       s=24, edgecolors=INK, linewidths=.65, zorder=6)
    ax.scatter(*coordinates[walk[-1]], s=65, marker="s", facecolors="none", edgecolors=INK,
               linewidths=1.2, zorder=7)
    ax.scatter(*coordinates[walk[0]], s=47, marker="D", color=GOLD, edgecolors=INK, linewidths=.8, zorder=8)
    return image


def render(output, routes, edge_cap, observation_cap):
    fig = plt.figure(figsize=(12.8, 9.4), facecolor=PAPER)
    fig.text(.065, .977, "Where each rollout takes its readings", color=INK, fontsize=24, weight="bold", va="top")
    fig.text(.066, .932, f"23 × 23 grid  ·  All routes walk {edge_cap} steps  ·  At most {observation_cap} readings",
             color=MUTED, fontsize=10.5, va="top")
    norm = Normalize(0, edge_cap)
    lefts = (.068, .382, .696)
    width, height = .22, .30
    handles = [
        Line2D([], [], marker="D", color=INK, markerfacecolor=GOLD, linestyle="none", label="Start"),
        Line2D([], [], color="#637d76", linewidth=1.5, label="Walking path"),
        Line2D([], [], marker="s", color=INK, markerfacecolor="none", linestyle="none", label="End"),
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(.927,.936), frameon=False, fontsize=9,
               ncol=3, labelcolor=MUTED, columnspacing=1.3)

    for row, (variant, start, label) in enumerate(EXAMPLES):
        layout, anchors = refine_geometry(floor_plan(LayoutCase("corridor", variant)))
        bottom = (.537, .119)[row]
        fig.text(.068, bottom + .349, label, color=INK, fontsize=13, weight="bold")
        for left, (arm, title) in zip(lefts, ARMS):
            route = routes[variant, start, arm]
            if route["edges"] != edge_cap or route["readings"] > observation_cap:
                raise ValueError("The selected routes must share the reported budgets")
            ax = fig.add_axes([left, bottom, width, height])
            image = draw_route(ax, layout, anchors[start], route, norm)
            fig.text(left, bottom + .312, title, color=INK, fontsize=10.5, weight="bold")
            count = route["counts_at_distance"][39]
            fig.text(left, bottom - .010, f"{count} readings by step 39  ·  {route['late_count']} after 39  ·  "
                     f"{route['readings']} total", color=MUTED, fontsize=8.6, va="top")

    colorbar = fig.colorbar(image, cax=fig.add_axes([.33,.057,.34,.014]), orientation="horizontal",
                           ticks=[0, 13, 26, 39, 52])
    colorbar.set_label("Walking step at each measurement", fontsize=9.5, color=INK, labelpad=3)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(labelsize=8.5, colors=MUTED, pad=2, length=2.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=165, facecolor=PAPER)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("outputs/sensing-world-residential-quarter-rollout"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    protocol = json.loads((args.study / "protocol.json").read_text())
    routes = {(r["variant"], r["start"], r["arm"]): r
              for r in json.loads((args.study / "routes.json").read_text())}
    output = args.output or args.study / "rollout-route-comparison.png"
    render(output, routes, protocol["edge_cap"], protocol["cap"])
    print(f"Rendered six archived walks to {output}")


if __name__ == "__main__":
    main()
