"""Render a median mixed-range corridor example from the fixed-route experiment."""

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
from scipy.ndimage import map_coordinates
from threadpoolctl import threadpool_limits

from audit_residential_zone_ranges import evaluate_layout, load_archive


INK = "#172e3c"
MUTED = "#496270"
PAPER = "#fafaf6"
GOLD = "#ffd166"


def representative(path):
    with (path / "fields.csv").open() as stream:
        mixed = [r for r in csv.DictReader(stream) if r["scenario"] == "mixed"]
    changes = [np.sqrt(float(r["local_mid_mse"])) - np.sqrt(float(r["global_mid_mse"])) for r in mixed]
    sample_index = int(np.argmin(abs(np.asarray(changes) - np.median(changes))))
    return mixed[sample_index], len(mixed)


def wall_segments(layout):
    starts, ends = np.where(np.triu(layout.wall_edges, 1))
    points = layout.points * (layout.size - 1)
    midpoints = (points[starts] + points[ends]) / 2
    steps = points[ends] - points[starts]
    perpendiculars = np.column_stack((-steps[:, 1], steps[:, 0])) / 2
    return np.stack((midpoints - perpendiculars, midpoints + perpendiculars), axis=1)


def interpolate_zones(values, zones, subdivisions=11):
    """Upsample colors within each zone while retaining the displayed boundaries."""
    size = zones.shape[0]
    axis = (np.arange(size * subdivisions) + .5) / subdivisions - .5
    coordinates = np.meshgrid(axis, axis, indexing="ij")
    display_zones = map_coordinates(zones, coordinates, order=0, mode="nearest")
    result = np.empty(display_zones.shape, dtype=float)
    for zone in np.unique(zones):
        mask = zones == zone
        inside = display_zones == zone
        numerator = map_coordinates(np.where(mask, values, 0.0), coordinates, order=1, mode="nearest")
        denominator = map_coordinates(mask.astype(float), coordinates, order=1, mode="nearest")
        result[inside] = numerator[inside] / denominator[inside]
    return result


def render(file, case, example, sample_count, *, interpolate_display=False):
    layout = example["layout"]
    points = layout.points * (layout.size - 1)
    size = layout.size
    bounds = (-.5, size - .5)
    segments = wall_segments(layout)
    sites, held, middle = example["sites"], example["held"], example["middle"]
    truth, global_mean, local_mean = (example[k] for k in ("truth", "global_mean", "local_mean"))
    field_limit = np.ceil(2 * np.max(np.abs([truth, global_mean, local_mean]))) / 2
    mid_sites = held[middle]

    fig = plt.figure(figsize=(12.5, 9.35), facecolor=PAPER)
    fig.text(.064, .962, "Spatial reconstruction with regional range weights",
             fontsize=22, weight="bold", color=INK, va="top")
    fig.text(.065, .922, f"Mixed-range corridor  /  Median 2–4-step RMSE change among {sample_count:,} fields  /  "
             f"{case['edges']} steps and {len(sites)} readings", fontsize=10, color=MUTED, va="top")
    lefts = (.075, .388, .701)

    def panel(col, top, title, subtitle):
        left, bottom = lefts[col], .531 if top else .101
        ax = fig.add_axes([left, bottom, .222, .297])
        ax.set(xlim=bounds, ylim=bounds, aspect="equal", xticks=[], yticks=[])
        ax.set_facecolor("#eef1ef")
        for spine in ax.spines.values():
            spine.set(color=INK, linewidth=1.45)
        ax.add_collection(LineCollection(segments, color=INK, linewidth=2.2, zorder=3))
        fig.text(left, bottom + .342, title, fontsize=12, weight="bold", color=INK, va="top")
        fig.text(left, bottom + .319, subtitle, fontsize=9, color=MUTED, va="top")
        return ax

    def grid(ax, values, cmap, limits):
        return ax.imshow(values, origin="lower", extent=bounds + bounds,
                         cmap=cmap, norm=Normalize(*limits), interpolation="nearest", zorder=0)

    def colorbar(artist, rect, label, ticks):
        color = fig.colorbar(artist, cax=fig.add_axes(rect), orientation="horizontal", ticks=ticks)
        color.ax.tick_params(labelsize=8, color=MUTED, labelcolor=INK, length=2.5, pad=2)
        color.set_label(label, fontsize=8.6, labelpad=2, color=MUTED)
        color.outline.set_visible(False)

    a = panel(0, True, "A   True field", "Walking path and measurement stops")
    b = panel(1, True, "B   Global reconstruction", f"Hold-out RMSE  {float(case['global_rmse']):.4f}")
    c = panel(2, True, "C   Regional reconstruction", f"Hold-out RMSE  {float(case['local_rmse']):.4f}")
    for ax, values in zip((a, b, c), (truth, global_mean, local_mean)):
        grid_values = values.reshape(size, size)
        if interpolate_display:
            grid_values = interpolate_zones(grid_values, layout.room.reshape(size, size))
        field_image = grid(ax, grid_values, "RdBu_r", (-field_limit, field_limit))

    walk = [sites[0]]
    for start, end in zip(sites, sites[1:]):
        walk.extend(layout.walk(start, end)[1:])
    a.plot(*points[walk].T, color="white", lw=3.45, zorder=4, solid_capstyle="round")
    a.plot(*points[walk].T, color=INK, lw=1.6, zorder=5, solid_capstyle="round")
    a.scatter(*points[sites].T, s=25, facecolors=GOLD, edgecolors=INK, linewidths=.8, zorder=6)
    a.scatter(*points[sites[0]], s=55, marker="D", facecolors="white", edgecolors=INK, linewidths=1, zorder=7)
    for ax in (b, c):
        ax.scatter(*points[held].T, s=13, facecolors="none", edgecolors="white", linewidths=1, zorder=5)
        ax.scatter(*points[held].T, s=11, facecolors="none", edgecolors=INK, linewidths=.45, zorder=6)
    field_label = "Within-zone interpolation of 12 × 12 values" if interpolate_display else "Latent sensor value"
    colorbar(field_image, [.323, .495, .352, .012], field_label, [-field_limit, 0, field_limit])

    d = panel(0, False, "D   True short-range weights", "Two range conditions in one floor plan")
    e = panel(1, False, "E   Inferred short-range weights", "Posterior mean from the same sensor readings")
    for ax, values in ((d, example["true_weights"]), (e, example["local_weights"])):
        weight_image = grid(ax, values[layout.room].reshape(size, size), "cividis", (0, 1))
        for zone, value in enumerate(values):
            x, y = points[layout.room == zone].mean(axis=0)
            ax.text(x, y, f"{value:.2f}", color=INK, ha="center", va="center", fontsize=11, weight="bold",
                    bbox=dict(boxstyle="round,pad=.22", facecolor=PAPER, edgecolor="none", alpha=.95), zorder=5)
    colorbar(weight_image, [.155, .065, .354, .012], "Short-range weight", [0, .25, .5, .75, 1])

    gmid, lmid = (np.sqrt(float(case[f"{arm}_mid_mse"])) for arm in ("global", "local"))
    f = panel(2, False, "F   Error change at hold-outs", f"2–4-step RMSE  {gmid:.4f} → {lmid:.4f}")
    f.scatter(*points[held[~middle]].T, s=14, color="#aab9ba", linewidth=0, zorder=4)
    delta = abs(local_mean[mid_sites] - truth[mid_sites]) - abs(global_mean[mid_sites] - truth[mid_sites])
    error_limit = max(float(np.ceil(np.max(np.abs(delta)) * 100) / 100), .01)
    error_cmap = LinearSegmentedColormap.from_list("error_change", ["#006d80", "#faf9f5", "#c24232"])
    error_image = f.scatter(*points[mid_sites].T, c=delta, cmap=error_cmap,
                            norm=Normalize(-error_limit, error_limit),
                            s=44, edgecolors=INK, linewidths=.7, zorder=5)
    colorbar(error_image, [.715, .065, .192, .012], "Regional − global absolute error",
             [-error_limit, 0, error_limit])

    handles = [
        Line2D([], [], color=INK, marker="o", markerfacecolor=GOLD, label="Measurement"),
        Line2D([], [], color=INK, marker="D", markerfacecolor="white", linestyle="none", label="Start"),
        Line2D([], [], color=INK, marker="o", markerfacecolor="none", linestyle="none", label="Hold-out"),
    ]
    fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(.937, .9), frameon=False,
               labelcolor=MUTED, fontsize=8.5, ncol=3, handletextpad=.45, columnspacing=1.25)
    file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(file, dpi=190, facecolor=PAPER)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("outputs/sensing-world-residential-zone-ranges"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--interpolate-display", action="store_true",
                        help="Interpolate the field colors within each zone for display.")
    args = parser.parse_args()
    protocol = json.loads((args.study / "protocol.json").read_text())
    case, count = representative(args.study)
    config, model, routes = load_archive(Path(protocol["source"]))
    variant, start, profile, draw = (int(case[k]) for k in ("variant", "start", "profile", "draw"))
    with threadpool_limits(limits=1):
        rows, _, example = evaluate_layout(variant, config, model, routes,
                                           example_key=(case["regime"], start, profile, draw))
    row = next(r for r in rows
               if (r["regime"], r["start"], r["profile"], r["draw"]) == (case["regime"], start, profile, draw))
    keys = ("global_rmse", "global_mid_mse", "local_rmse", "local_mid_mse", "readings", "mid_points")
    np.testing.assert_allclose([row[k] for k in keys], [float(case[k]) for k in keys], atol=1e-12, rtol=0)
    name = "mixed-corridor-maps-interpolated.png" if args.interpolate_display else "mixed-corridor-maps.png"
    output = args.output or args.study / name
    render(output, case, example, count, interpolate_display=args.interpolate_display)
    print(f"Wrote {output} for layout {variant}, profile {profile}, draw {draw}")


if __name__ == "__main__":
    main()
