"""Render the shared-field corridor comparison across grid resolutions."""

from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np

from plot_residential_zone_ranges import GOLD, INK, MUTED, PAPER, wall_segments


def render(path, sample):
    layout = sample["layout"]
    size = layout.size
    points = layout.points * (size-1)
    held = sample["held"]
    truth = sample["truth"]
    panels = [
        ("True field", "Original route", "truth", "fixed"),
        ("12 × 12 reconstruction", "Original route · within-zone interpolation", "coarse_fixed", "fixed"),
        ("23 × 23 reconstruction", "Original route · native prediction", "fine_fixed", "fixed"),
        ("Original measurement candidates", "23 × 23 reconstruction · new route", "coarse_candidates", "coarse_candidates"),
        ("All measurement candidates", "23 × 23 reconstruction · new route", "fine_candidates", "fine_candidates"),
    ]
    limit = np.ceil(2 * max(np.abs(sample[key]).max() for _, _, key, _ in panels)) / 2
    norm = Normalize(-limit, limit)
    extent = (-.5, size-.5, -.5, size-.5)
    walls = wall_segments(layout)
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9.1), facecolor=PAPER)
    fig.subplots_adjust(left=.055, right=.94, bottom=.10, top=.84, wspace=.32, hspace=.32)
    fig.text(.064, .959, "Finer reconstruction and measurement placement", fontsize=22,
             color=INK, weight="bold", va="top")
    fig.text(.065, .914, "One shared sensor field on the refined corridor floor plan",
             fontsize=11, color=MUTED, va="top")
    for ax, (title, subtitle, key, route) in zip(axes.flat, panels):
        data = sample[key]
        sites, walk = sample["routes"][route], sample["walks"][route]
        image = ax.imshow(data.reshape(size, size), origin="lower", extent=extent, cmap="RdBu_r",
                          norm=norm, interpolation="nearest")
        ax.add_collection(LineCollection(walls, linewidth=2, color=INK, zorder=3))
        ax.plot(*points[walk].T, color="white", linewidth=2.6, alpha=.9, zorder=4)
        ax.plot(*points[walk].T, color=INK, linewidth=1.1, zorder=5)
        ax.scatter(*points[sites].T, s=16, facecolors=GOLD, edgecolors=INK, linewidths=.6, zorder=6)
        ax.scatter(*points[sites[0]], marker="D", s=36, facecolors="white",
                   edgecolors=INK, linewidths=.8, zorder=7)
        ax.set_title(title, loc="left", color=INK, fontsize=11, weight="bold", pad=17)
        ax.text(0, 1.016, subtitle, transform=ax.transAxes, fontsize=8, color=MUTED, va="bottom")
        ax.set(xticks=[], yticks=[])
        for spine in ax.spines.values():
            spine.set(color=INK, linewidth=1.1)
    info = axes[1, 2]
    info.axis("off")
    info.text(.04, .92, "This field", transform=info.transAxes, color=INK, fontsize=14, weight="bold")
    lines = []
    for title, route in (("Original route", "fixed"), ("Original candidates", "coarse_candidates"),
                         ("All candidates", "fine_candidates")):
        n = len(sample["routes"][route])
        edges = len(sample["walks"][route]) - 1
        lines.append(f"{title}\n  {n} readings  ·  {edges} refined steps")
    info.text(.04, .8, "\n\n".join(lines), transform=info.transAxes, color=MUTED, fontsize=10, va="top")
    info.text(.04, .24, f"{len(held)} shared evaluation sites\nGold dots mark readings.",
              transform=info.transAxes, color=MUTED, fontsize=9)
    bar = fig.colorbar(image, cax=fig.add_axes([.34, .054, .32, .014]), orientation="horizontal",
                       ticks=[-limit, 0, limit])
    bar.set_label("Latent sensor value", fontsize=9, color=MUTED, labelpad=2)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=8, colors=MUTED, length=2)
    fig.savefig(path, dpi=175, facecolor=PAPER)
    plt.close(fig)
