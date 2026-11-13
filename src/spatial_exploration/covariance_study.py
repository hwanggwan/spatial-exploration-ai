"""Separate covariance estimation on fixed observations from closed-loop planning."""

import argparse
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import sklearn

from .baseline import Config, GPBelief, evaluate, explore, make_grid, rbf_covariance, render, write_csv
from .estimated_gp import EstimatedGPBelief, EstimatorConfig


MODELS = ("oracle", "fixed_initial", "estimated")
COLORS = {"oracle": "#276b42", "fixed_initial": "#a86c00", "estimated": "#3565b7"}
LABELS = {"oracle": "Oracle covariance", "fixed_initial": "Fixed initial covariance",
          "estimated": "Estimated covariance"}


def make_belief(name, points, true_prior, environment, estimator):
    if name == "oracle":
        return GPBelief(true_prior, environment.noise_std)
    if name == "fixed_initial":
        return GPBelief(estimator.initial_process_variance * rbf_covariance(
            points, estimator.initial_length_scale), environment.noise_std)
    if name == "estimated":
        # The estimator receives neither true covariance parameters nor the field.
        return EstimatedGPBelief(points, environment.noise_std, estimator)
    raise ValueError("Unknown model: " + name)


def parameter_values(name, belief, environment, estimator):
    if name == "oracle":
        return environment.length_scale, 1.0
    if name == "fixed_initial":
        return estimator.initial_length_scale, estimator.initial_process_variance
    return belief.length_scale, belief.process_variance


def collect_step(protocol, seed, name, truth, evaluation_mask, belief, route,
                 distances, environment, estimator):
    length_scale, process_variance = parameter_values(name, belief, environment, estimator)
    return {
        "protocol": protocol, "seed": seed, "model": name,
        "measurements": len(route), "travel_distance": float(distances[-1]),
        **evaluate(truth, belief, evaluation_mask),
        "length_scale": length_scale, "process_variance": process_variance,
        "fit_calls": len(belief.fit_history) if name == "estimated" else 0,
        "boundary_fit_calls": sum(item["boundary_hit"] for item in belief.fit_history)
        if name == "estimated" else 0,
    }


def summarize(rows):
    summary = {}
    metrics = ("heldout_rmse", "heldout_mean_std", "heldout_coverage_95",
               "measurements", "travel_distance", "length_scale", "process_variance",
               "update_seconds", "fit_calls", "boundary_fit_calls")
    for protocol in ("replay", "closed_loop"):
        summary[protocol] = {}
        for name in MODELS:
            selected = [row for row in rows if row["protocol"] == protocol and row["model"] == name]
            summary[protocol][name] = {
                "seeds": len(selected),
                **{key: float(np.mean([row[key] for row in selected])) for key in metrics},
            }
            summary[protocol][name]["heldout_rmse_seed_std"] = float(np.std(
                [row["heldout_rmse"] for row in selected], ddof=1)) if len(selected) > 1 else 0.0
        differences = []
        for seed in sorted({row["seed"] for row in rows}):
            paired = {row["model"]: row for row in rows
                      if row["protocol"] == protocol and row["seed"] == seed}
            differences.append(paired["estimated"]["heldout_rmse"] - paired["oracle"]["heldout_rmse"])
        summary[protocol]["estimated_minus_oracle_rmse"] = {
            "mean_paired_difference": float(np.mean(differences)),
            "seed_differences": differences,
        }
    return summary


def render_replay(path, truth, examples, size, environment, estimator):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = make_grid(size)
    belief, route, _ = examples["oracle"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    lo, hi = float(truth.min()), float(truth.max())
    max_std = max(float(result[0].std.max()) for result in examples.values())
    for row in range(2):
        ax = axes[row, 0]
        plot = ax.imshow(truth.reshape(size, size), origin="lower", extent=(0, 1, 0, 1),
                         vmin=lo, vmax=hi, cmap="viridis")
        trajectory = points[route]
        ax.plot(trajectory[:, 0], trajectory[:, 1], "w.-", markersize=3, linewidth=1)
        ax.set_title("Hidden field + shared route\n" + str(len(route)) + " measurements")
        fig.colorbar(plot, ax=ax, shrink=0.75)
    for column, name in enumerate(MODELS, start=1):
        belief = examples[name][0]
        length_scale, variance = parameter_values(name, belief, environment, estimator)
        for row, values in enumerate((belief.mean, belief.std)):
            ax = axes[row, column]
            bounds = {"vmin": lo, "vmax": hi} if row == 0 else {"vmin": 0, "vmax": max_std}
            plot = ax.imshow(values.reshape(size, size), origin="lower", extent=(0, 1, 0, 1),
                             cmap="viridis" if row == 0 else "magma", **bounds)
            trajectory = points[route]
            ax.plot(trajectory[:, 0], trajectory[:, 1], "w.-", markersize=3, linewidth=1)
            ax.set_title(LABELS[name] + ("\nPosterior mean" if row == 0 else "\nPosterior std")
                         + f"; length={length_scale:.3f}, variance={variance:.3f}", fontsize=9)
            fig.colorbar(plot, ax=ax, shrink=0.75)
    for ax in axes.flat:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.suptitle("Same-observation comparison — seed 0; known zero mean and sensor noise")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def render_curves(path, steps, environment):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis, protocol in zip(axes, ("replay", "closed_loop")):
        for name in MODELS:
            seed_curves = []
            selected = [row for row in steps if row["protocol"] == protocol and row["model"] == name]
            if protocol == "replay":
                x_grid = np.arange(1, max(row["measurements"] for row in selected) + 1)
                x_key = "measurements"
            else:
                x_grid = np.linspace(0, environment.travel_budget, 101)
                x_key = "travel_distance"
            for seed in range(environment.seeds):
                per_seed = [row for row in selected if row["seed"] == seed]
                x = np.array([row[x_key] for row in per_seed])
                y = np.array([row["heldout_rmse"] for row in per_seed])
                indices = np.clip(np.searchsorted(x, x_grid, side="right") - 1, 0, len(y) - 1)
                seed_curves.append(y[indices])
            axis.plot(x_grid, np.mean(seed_curves, axis=0), color=COLORS[name],
                      label=LABELS[name], drawstyle="steps-post")
        axis.set_title("Replay: identical observations" if protocol == "replay"
                       else "Closed loop: model-dependent routes")
        axis.set_xlabel("Number of measurements" if protocol == "replay" else "Travel distance")
        axis.set_ylabel("Mean held-out latent-field RMSE")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.suptitle(f"Covariance estimation study — {environment.seeds} paired fields; descriptive means")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_study(environment, estimator, output):
    environment.validate()
    estimator.validate()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    points = make_grid(environment.grid_size)
    true_prior = rbf_covariance(points, environment.length_scale)
    factor = np.linalg.cholesky(true_prior + 1e-10 * np.eye(len(points)))
    evaluation_mask = np.arange(len(points)) % 5 == 4
    allowed = ~evaluation_mask
    # A fixed design independent of the true kernel and of all observed values.
    design_prior = estimator.initial_process_variance * rbf_covariance(points, estimator.initial_length_scale)
    _, shared_route, shared_distances = explore(
        points, design_prior, np.zeros(len(points)), allowed, environment,
        "greedy", np.random.default_rng(0),
    )
    final_rows, steps, trajectories, fits = [], [], [], []
    examples = {"replay": {}, "closed_loop": {}}
    for seed in range(environment.seeds):
        rng = np.random.default_rng(seed)
        truth = factor @ rng.standard_normal(len(points))
        sensor_values = truth + environment.noise_std * rng.standard_normal(len(points))
        for protocol in ("replay", "closed_loop"):
            for name in MODELS:
                belief = make_belief(name, points, true_prior, environment, estimator)
                step_start = len(steps)

                def record(current_belief, route, distances):
                    steps.append(collect_step(protocol, seed, name, truth, evaluation_mask,
                        current_belief, route, distances, environment, estimator))
                    index = route[-1]
                    trajectories.append({"protocol": protocol, "seed": seed, "model": name,
                        "step": len(route) - 1, "index": index,
                        "x": float(points[index, 0]), "y": float(points[index, 1]),
                        "observed_value": float(sensor_values[index]),
                        "cumulative_distance": float(distances[-1])})

                # Timer wraps actual posterior updates only, excluding plotting/evaluation.
                update_seconds = [0.0]

                def timed_observe(index, value, original_observe=belief.observe):
                    started = time.perf_counter()
                    original_observe(index, value)
                    update_seconds[0] += time.perf_counter() - started

                belief.observe = timed_observe
                if protocol == "replay":
                    route, distances = [], []
                    for index, distance in zip(shared_route, shared_distances):
                        belief.observe(index, sensor_values[index])
                        route.append(index)
                        distances.append(distance)
                        record(belief, route, distances)
                else:
                    belief, route, distances = explore(points, None, sensor_values, allowed,
                        environment, "greedy", np.random.default_rng(0), belief=belief, on_step=record)
                final_rows.append({**steps[-1], "update_seconds": update_seconds[0]})
                if name == "estimated":
                    fits.extend({"protocol": protocol, "seed": seed, **event}
                                for event in belief.fit_history)
                if seed == 0:
                    examples[protocol][name] = (belief, route.copy(), distances.copy())
                assert len(steps) > step_start
        if seed == 0:
            render_replay(output / "mapping_comparison.png", truth, examples["replay"],
                          environment.grid_size, environment, estimator)
            render(output / "exploration_comparison.png", truth, examples["closed_loop"],
                   environment.grid_size, title="Greedy exploration — seed 0; covariance treatment varies")
    summary = summarize(final_rows)
    write_csv(output / "metrics.csv", final_rows)
    write_csv(output / "learning_curves.csv", steps)
    write_csv(output / "trajectories.csv", trajectories)
    if fits:
        write_csv(output / "parameter_fits.csv", fits)
    else:
        # Keep an explicit empty log for budgets too small to trigger a fit.
        (output / "parameter_fits.csv").write_text("protocol,seed,measurements\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "config.json").write_text(json.dumps({
        "environment": asdict(environment), "estimator": asdict(estimator),
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
        "generator_process_variance": 1.0, "generator_cholesky_jitter": 1e-10,
        "known_to_estimator": ["RBF family", "zero mean", "sensor noise_std"],
        "uncertainty": "latent GP posterior conditional on plug-in parameters",
        "shared_design": "greedy with fixed initial kernel; independent of true kernel and values",
        "evaluation": "grid indices where index % 5 == 4; never sampled",
        "timing": "posterior updates and scheduled optimization; excludes metrics, planning, plotting",
    }, indent=2) + "\n")
    render_curves(output / "performance_curves.png", steps, environment)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--grid-size", type=int, default=15)
    parser.add_argument("--length-scale", type=float, default=0.2,
                        help="True generating length scale; never passed to the estimator")
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--travel-budget", type=float, default=2.0)
    parser.add_argument("--initial-length-scale", type=float, default=0.35)
    parser.add_argument("--initial-process-variance", type=float, default=1.5)
    parser.add_argument("--min-observations", type=int, default=4)
    parser.add_argument("--refit-every", type=int, default=3)
    parser.add_argument("--optimizer-restarts", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("outputs/covariance-study"))
    args = parser.parse_args()
    environment = Config(args.grid_size, args.length_scale, args.noise_std,
                         args.travel_budget, args.seeds)
    estimator = EstimatorConfig(initial_length_scale=args.initial_length_scale,
        initial_process_variance=args.initial_process_variance,
        min_observations=args.min_observations, refit_every=args.refit_every,
        optimizer_restarts=args.optimizer_restarts)
    summary = run_study(environment, estimator, args.output)
    for protocol in ("replay", "closed_loop"):
        print(protocol)
        for name in MODELS:
            row = summary[protocol][name]
            print(f"  {name:14s} RMSE={row['heldout_rmse']:.4f} "
                  f"coverage={row['heldout_coverage_95']:.3f} "
                  f"measurements={row['measurements']:.1f}")
    print("Artifacts:", args.output.resolve())


if __name__ == "__main__":
    main()
