"""A small exact-GP mapping baseline; no hidden values are used by planners."""

import argparse
import csv
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Config:
    grid_size: int = 15
    length_scale: float = 0.2
    noise_std: float = 0.1
    travel_budget: float = 2.0
    seeds: int = 5
    max_observations: Optional[int] = None

    def validate(self):
        if self.grid_size < 3 or self.seeds < 1:
            raise ValueError("grid_size must be >= 3 and seeds must be >= 1")
        if not all(np.isfinite(v) for v in (
            self.length_scale, self.noise_std, self.travel_budget
        )):
            raise ValueError("Parameters must be finite")
        if self.length_scale <= 0 or self.noise_std <= 0 or self.travel_budget < 0:
            raise ValueError("Scales must be positive and budget nonnegative")
        if self.max_observations is not None and (not isinstance(self.max_observations, (int, np.integer))
                or isinstance(self.max_observations, (bool, np.bool_)) or self.max_observations < 1):
            raise ValueError("Observation cap must be a positive integer or None")


def make_grid(size):
    x, y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    return np.column_stack((x.ravel(), y.ravel()))


def rbf_covariance(points, length_scale):
    delta = points[:, None, :] - points[None, :, :]
    return np.exp(-np.sum(delta * delta, axis=-1) / (2 * length_scale**2))


class GPBelief:
    def __init__(self, prior_covariance, noise_std):
        self.mean = np.zeros(len(prior_covariance))
        self.covariance = prior_covariance.copy()
        self.noise_variance = noise_std**2

    def observe(self, index, value):
        cross = self.covariance[:, index].copy()
        variance = self.covariance[index, index] + self.noise_variance
        self.mean += cross * ((value - self.mean[index]) / variance)
        self.covariance -= np.outer(cross, cross) / variance
        self.covariance = (self.covariance + self.covariance.T) / 2

    @property
    def std(self):
        return np.sqrt(np.maximum(np.diag(self.covariance), 0))


def explore(points, prior, sensor_values, allowed, config, strategy, rng,
            belief=None, on_step=None):
    if belief is None:
        belief = GPBelief(prior, config.noise_std)
    visited = np.zeros(len(points), dtype=bool)
    current, distance = 0, 0.0
    route, cumulative_distance = [], []
    while True:
        belief.observe(current, sensor_values[current])
        visited[current] = True
        route.append(current)
        cumulative_distance.append(distance)
        if on_step is not None:
            on_step(belief, route, cumulative_distance)
        if config.max_observations is not None and len(route) >= config.max_observations:
            break
        costs = np.linalg.norm(points - points[current], axis=1)
        candidates = np.flatnonzero(
            allowed & ~visited & (costs <= config.travel_budget - distance + 1e-12)
        )
        if not len(candidates):
            break
        if strategy == "random":
            next_index = int(rng.choice(candidates))
        elif strategy == "greedy":
            cross = belief.covariance[:, candidates]
            variance = np.diag(belief.covariance)[candidates] + belief.noise_variance
            expected_reduction = np.mean(cross * cross, axis=0) / variance
            scores = expected_reduction / costs[candidates]
            next_index = int(candidates[np.argmax(scores)])
        else:
            raise ValueError("Unknown strategy: " + strategy)
        distance += float(costs[next_index])
        current = next_index
    return belief, route, cumulative_distance


def evaluate(truth, belief, evaluation_mask):
    residual = truth[evaluation_mask] - belief.mean[evaluation_mask]
    std = belief.std[evaluation_mask]
    return {
        "heldout_rmse": float(np.sqrt(np.mean(residual**2))),
        "heldout_mean_std": float(np.mean(std)),
        "heldout_coverage_95": float(np.mean(np.abs(residual) <= 1.96 * std)),
    }


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render(path, truth, results, size, title=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = make_grid(size)
    half_step = 0.5 / (size - 1)
    fig, axes = plt.subplots(len(results), 3, figsize=(13, 4 * len(results)),
                             constrained_layout=True, squeeze=False)
    limits = {"vmin": float(truth.min()), "vmax": float(truth.max())}
    max_std = max(float(belief.std.max()) for belief, _, _ in results.values())
    for row, (strategy, (belief, route, _)) in enumerate(results.items()):
        for col, (values, panel_title) in enumerate((
            (truth, "Hidden field (evaluation only)"),
            (belief.mean, "GP predicted field"),
            (belief.std, "Latent posterior standard deviation"),
        )):
            ax = axes[row, col]
            options = limits if col < 2 else {"vmin": 0, "vmax": max_std}
            plot = ax.imshow(values.reshape(size, size), origin="lower",
                             extent=(-half_step, 1 + half_step, -half_step, 1 + half_step),
                             cmap="viridis" if col < 2 else "magma", **options)
            trajectory = points[route]
            ax.plot(trajectory[:, 0], trajectory[:, 1], "w.-", linewidth=1, markersize=3)
            ax.plot(*trajectory[0], "wo", markersize=7, markeredgecolor="black")
            ax.plot(*trajectory[-1], "wx", markersize=7)
            ax.set_title(strategy.capitalize() + ": " + panel_title, fontsize=10)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            fig.colorbar(plot, ax=ax, shrink=0.8)
    fig.suptitle(title or "Synthetic spatial exploration — first seed; matched RBF GP model")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(config, output):
    config.validate()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    points = make_grid(config.grid_size)
    prior = rbf_covariance(points, config.length_scale)
    factor = np.linalg.cholesky(prior + 1e-10 * np.eye(len(points)))
    # Deterministic evaluation sites are unavailable to every planner.
    evaluation_mask = np.arange(len(points)) % 5 == 4
    allowed = ~evaluation_mask
    rows, trajectories, example = [], [], {}
    for seed in range(config.seeds):
        field_rng = np.random.default_rng(seed)
        truth = factor @ field_rng.standard_normal(len(points))
        sensor_values = truth + config.noise_std * field_rng.standard_normal(len(points))
        for strategy in ("random", "greedy"):
            belief, route, distances = explore(
                points, prior, sensor_values, allowed, config, strategy,
                np.random.default_rng(seed + 100000),
            )
            rows.append({"seed": seed, "strategy": strategy,
                         **evaluate(truth, belief, evaluation_mask),
                         "travel_distance": distances[-1], "measurements": len(route)})
            for step, (index, distance) in enumerate(zip(route, distances)):
                trajectories.append({"seed": seed, "strategy": strategy, "step": step,
                                     "x": float(points[index, 0]), "y": float(points[index, 1]),
                                     "observed_value": float(sensor_values[index]),
                                     "cumulative_distance": distance})
            if seed == 0:
                example[strategy] = (belief, route, distances)
        if seed == 0:
            render(output / "comparison.png", truth, example, config.grid_size)
    summary = {}
    for strategy in ("random", "greedy"):
        selected = [row for row in rows if row["strategy"] == strategy]
        summary[strategy] = {
            "seeds": config.seeds,
            **{key: float(np.mean([row[key] for row in selected]))
               for key in ("heldout_rmse", "heldout_mean_std", "heldout_coverage_95",
                           "travel_distance", "measurements")},
        }
    write_csv(output / "metrics.csv", rows)
    write_csv(output / "trajectories.csv", trajectories)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "config.json").write_text(json.dumps({
        **asdict(config), "python_version": platform.python_version(),
        "numpy_version": np.__version__, "model": "known zero-mean RBF GP",
        "evaluation": "grid indices where index % 5 == 4; never sampled",
    }, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=15)
    parser.add_argument("--length-scale", type=float, default=0.2)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--travel-budget", type=float, default=2.0)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--max-observations", type=int)
    parser.add_argument("--output", type=Path, default=Path("outputs/baseline"))
    args = parser.parse_args()
    config = Config(args.grid_size, args.length_scale, args.noise_std,
                    args.travel_budget, args.seeds, args.max_observations)
    print(json.dumps(run(config, args.output), indent=2))
    print("Artifacts:", args.output.resolve())


if __name__ == "__main__":
    main()
