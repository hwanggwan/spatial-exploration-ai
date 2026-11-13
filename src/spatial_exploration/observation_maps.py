"""Render archived noisy measurements separately from evaluator-only truth."""

import argparse
from pathlib import Path

import numpy as np


def render(reference, field=0, counts=(8, 32, 128), output=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    reference = Path(reference)
    with np.load(reference / "benchmark_observations.npz") as stored:
        points = stored["points"]
        indices = stored["observation_indices"][field]
        values = stored["observed_values"][:, field]
        names = stored["scenario_names"].tolist()
    with np.load(reference / "benchmark_evaluation_only.npz") as stored:
        truth = stored["truth"][:, field]
    if any(count < 1 or count > len(indices) for count in counts):
        raise ValueError("Requested observation count is outside the archive")
    size = len(np.unique(points[:, 0]))
    half = 0.5 / (size - 1)
    limit = float(max(np.abs(truth).max(), np.abs(values[:, :max(counts)]).max()))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    fig, axes = plt.subplots(len(names), len(counts) + 1, figsize=(14, 10),
                             constrained_layout=True, squeeze=False)
    for row, name in enumerate(names):
        axis = axes[row, 0]
        axis.imshow(truth[row].reshape(size, size), origin="lower", interpolation="nearest",
            extent=(-half, 1 + half, -half, 1 + half), cmap="RdBu_r", norm=norm)
        axis.set_ylabel(f"{name.capitalize()} transition\ny coordinate")
        for column, count in enumerate(counts, start=1):
            axis = axes[row, column]
            axis.set_facecolor("#f5f5f5")
            axis.scatter(points[:, 0], points[:, 1], c="#d5d5d5", s=9, linewidths=0)
            sites = points[indices[:count]]
            plot = axis.scatter(sites[:, 0], sites[:, 1], c=values[row, :count], cmap="RdBu_r",
                norm=norm, s=44, edgecolors="#333333", linewidths=0.35)
        for column, axis in enumerate(axes[row]):
            axis.set_xlim(-half, 1 + half)
            axis.set_ylim(-half, 1 + half)
            axis.set_aspect("equal")
            axis.axhline(0.5, color="#777777", linestyle="--", linewidth=0.7, alpha=0.7)
            axis.set_xticks((0, 0.5, 1))
            axis.set_yticks((0, 0.5, 1))
            if row == 0:
                axis.set_title("Latent truth\n(evaluation only)" if column == 0 else
                               f"Observed values: n={counts[column-1]}\n(sensor noise std=0.1)", fontsize=10)
            if row == len(names) - 1:
                axis.set_xlabel("x coordinate")
    fig.colorbar(plot, ax=axes, shrink=0.65, pad=0.02, label="Field / measured value (common scale)")
    fig.suptitle(f"Actual observations — predefined field {field}; no interpolation of measured values\n"
        "Colored circles: measured sites. Gray dots: unmeasured. Dashed line: transition center, not a wall.", fontsize=12)
    output = Path(output) if output else reference / f"observations_field{field}.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output.resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("outputs/transition-study"))
    parser.add_argument("--field", type=int, default=0)
    parser.add_argument("--counts", nargs="+", type=int, default=(8, 32, 128))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(render(args.reference, args.field, args.counts, args.output))


if __name__ == "__main__":
    main()
