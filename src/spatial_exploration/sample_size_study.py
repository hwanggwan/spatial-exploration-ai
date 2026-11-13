"""Nested same-observation mapper comparisons before adding DeepKriging."""

import argparse
from dataclasses import asdict, dataclass
from importlib import metadata
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

from .baseline import GPBelief, evaluate, make_grid, rbf_covariance, write_csv
from .estimated_gp import EstimatedGPBelief, EstimatorConfig


SCENARIOS = ("stationary_rbf", "stationary_mixture", "nonstationary_mixture")
MODELS = ("oracle", "fixed_initial", "estimated")
LABELS = {"stationary_rbf": "Stationary RBF", "stationary_mixture": "Stationary two-scale control",
          "nonstationary_mixture": "Spatially varying two-scale GP"}
COLORS = {"oracle": "#27754d", "fixed_initial": "#b5762b", "estimated": "#376aa3"}


@dataclass(frozen=True)
class SampleSizeConfig:
    grid_size: int = 15
    fields: int = 64
    sample_sizes: tuple = (8, 16, 32, 64, 128)
    stationary_length: float = 0.23
    long_length: float = 0.35
    short_length: float = 0.12
    noise_std: float = 0.1
    field_seed_start: int = 1500000
    design_seed_start: int = 1600000

    def validate(self):
        if self.grid_size < 3 or self.fields < 2:
            raise ValueError("Use a grid of at least three and at least two independent fields")
        allowed_count = int(np.sum(np.arange(self.grid_size**2) % 5 != 4))
        if not self.sample_sizes or any(not isinstance(value, (int, np.integer)) for value in self.sample_sizes):
            raise ValueError("Sample sizes must be nonempty integers")
        if tuple(sorted(set(self.sample_sizes))) != tuple(self.sample_sizes):
            raise ValueError("Sample sizes must be strictly increasing")
        if self.sample_sizes[0] < 1 or self.sample_sizes[-1] > allowed_count:
            raise ValueError("Sample sizes must fit within the permitted sensing sites")
        if not all(np.isfinite(value) and value > 0 for value in
                   (self.stationary_length, self.long_length, self.short_length, self.noise_std)):
            raise ValueError("Scales and noise must be finite and positive")
        if self.short_length >= self.long_length:
            raise ValueError("Short correlation length must be smaller than long length")


def generating_covariance(points, scenario, config):
    if scenario == "stationary_rbf":
        return rbf_covariance(points, config.stationary_length)
    long = rbf_covariance(points, config.long_length)
    short = rbf_covariance(points, config.short_length)
    if scenario == "stationary_mixture":
        return 0.5 * (long + short)
    if scenario == "nonstationary_mixture":
        # Y(s)=a(s)Z_long(s)+b(s)Z_short(s), independent unit-variance GPs.
        # a^2+b^2=1 keeps variance constant; mean is zero everywhere. Vary along
        # y so every withheld x-column covers the same smooth/rough regimes.
        angle = 0.5 * np.pi * points[:, 1]
        a, b = np.cos(angle), np.sin(angle)
        return np.outer(a, a) * long + np.outer(b, b) * short
    raise ValueError("Unknown spatial scenario")


def generate_dataset(config):
    config.validate()
    points = make_grid(config.grid_size)
    evaluation_mask = np.arange(len(points)) % 5 == 4
    allowed = np.flatnonzero(~evaluation_mask)
    covariances = np.asarray([generating_covariance(points, name, config) for name in SCENARIOS])
    factors = [np.linalg.cholesky(covariance + 1e-10 * np.eye(len(points))) for covariance in covariances]
    orders = np.asarray([np.random.default_rng(config.design_seed_start + index).permutation(allowed)[
        :config.sample_sizes[-1]] for index in range(config.fields)])
    truth = np.empty((len(SCENARIOS), config.fields, len(points)))
    values = np.empty((len(SCENARIOS), config.fields, config.sample_sizes[-1]))
    for index in range(config.fields):
        rng = np.random.default_rng(config.field_seed_start + index)
        z, noise = rng.standard_normal((2, len(points)))
        for offset, factor in enumerate(factors):
            truth[offset, index] = factor @ z
            potential = truth[offset, index] + config.noise_std * noise
            values[offset, index] = potential[orders[index]]
    observations = {"points": points, "scenario_names": np.asarray(SCENARIOS),
        "sample_sizes": np.asarray(config.sample_sizes), "field_ids": np.arange(config.fields),
        "observation_indices": orders, "observed_values": values}
    evaluation_only = {"truth": truth, "evaluation_mask": evaluation_mask, "true_covariances": covariances}
    return observations, evaluation_only


def fit_snapshot(model, points, indices, values, noise_std, estimator, oracle_prior=None):
    """Fit only supplied values; oracle covariance is explicit and privileged."""
    if model == "estimated":
        return EstimatedGPBelief.from_observations(points, noise_std, indices, values, estimator)
    if model == "oracle":
        if oracle_prior is None:
            raise ValueError("Oracle needs its explicit generating covariance")
        prior = oracle_prior
    elif model == "fixed_initial":
        prior = estimator.initial_process_variance * rbf_covariance(points, estimator.initial_length_scale)
    else:
        raise ValueError("Unknown mapper")
    belief = GPBelief(prior, noise_std)
    for index, value in zip(indices, values):
        belief.observe(int(index), float(value))
    return belief


def bootstrap_mean(values):
    values = np.asarray(values)
    rng = np.random.default_rng(1700000)
    means = values[rng.integers(0, len(values), size=(5000, len(values)))].mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def summarize(rows, config):
    summaries, comparisons = [], []
    for scenario in SCENARIOS:
        for count in config.sample_sizes:
            oracle = {row["field"]: row for row in rows if row["scenario"] == scenario
                      and row["sample_size"] == count and row["model"] == "oracle"}
            for model in MODELS:
                selected = [row for row in rows if row["scenario"] == scenario
                            and row["sample_size"] == count and row["model"] == model]
                summary = {"scenario": scenario, "sample_size": count, "model": model, "fields": len(selected)}
                for key in ("heldout_rmse", "heldout_coverage_95", "heldout_mse", "heldout_mean_variance",
                    "update_seconds", "fit_seconds", "model_process_variance"):
                    values = [row[key] for row in selected]
                    summary[key] = float(np.mean(values))
                    if key in ("heldout_rmse", "heldout_coverage_95"):
                        summary[key + "_interval"] = bootstrap_mean(values)
                summary["model_length_scale"] = float(np.mean([row["model_length_scale"] for row in selected])) \
                    if all(row["model_length_scale"] is not None for row in selected) else None
                for key in ("length_scale_absolute_error", "process_variance_absolute_error"):
                    values = [row[key] for row in selected if row[key] is not None]
                    summary[key] = float(np.mean(values)) if values else None
                summary["boundary_fit_calls"] = sum(row["boundary_fit_calls"] for row in selected)
                summary["rejected_fit_calls"] = sum(row["rejected_fit_calls"] for row in selected)
                summaries.append(summary)
                comparison = {"scenario": scenario, "sample_size": count, "model": model}
                for key in ("heldout_rmse", "heldout_mse"):
                    deltas = [row[key] - oracle[row["field"]][key] for row in selected]
                    comparison[key + "_difference_vs_oracle"] = float(np.mean(deltas))
                    comparison[key + "_difference_interval"] = bootstrap_mean(deltas)
                comparisons.append(comparison)
    return summaries, comparisons


def render_curves(path, summaries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(SCENARIOS), figsize=(15, 8), constrained_layout=True)
    for column, scenario in enumerate(SCENARIOS):
        for row, metric in enumerate(("heldout_rmse", "heldout_coverage_95")):
            axis = axes[row, column]
            for model in MODELS:
                selected = [item for item in summaries if item["scenario"] == scenario and item["model"] == model]
                counts = [item["sample_size"] for item in selected]
                axis.plot(counts, [item[metric] for item in selected], "o-", color=COLORS[model], label=model)
                intervals = np.asarray([item[metric + "_interval"] for item in selected])
                axis.fill_between(counts, intervals[:, 0], intervals[:, 1], color=COLORS[model], alpha=0.13)
            axis.set_xscale("log", base=2)
            axis.set_xticks(counts, [str(value) for value in counts])
            axis.set_xlabel("Shared acquired measurements")
            axis.grid(alpha=0.2)
            if row == 0:
                axis.set_title(LABELS[scenario], fontsize=11)
                axis.set_ylabel("Mean held-out latent RMSE (lower is better)")
                axis.legend(fontsize=8)
            else:
                axis.axhline(0.95, color="black", linestyle="--", linewidth=1)
                axis.set_ylim(0, 1.02)
                axis.set_ylabel("Mean latent 95% interval coverage")
    fig.suptitle("Nested same-observation comparison — shading: paired-field bootstrap intervals for each mean")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_example(path, scenario, truth, covariance, examples, config, indices):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = make_grid(config.grid_size)
    half_step = 0.5 / (config.grid_size - 1)
    extent = (-half_step, 1 + half_step, -half_step, 1 + half_step)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), constrained_layout=True)
    nearest_horizontal = np.arange(len(points)) + np.where(np.arange(len(points)) % config.grid_size == config.grid_size - 1, -1, 1)
    correlation = covariance[np.arange(len(points)), nearest_horizontal]
    panels = [(axes[0, 0], truth, "Hidden field (evaluation only)", "viridis", truth.min(), truth.max()),
        (axes[1, 0], correlation, "True horizontal-neighbor correlation", "viridis", 0.75, 1.0)]
    maximum_std = max(belief.std.max() for belief in examples.values())
    for column, (name, belief) in enumerate(examples.items(), start=1):
        rmse = evaluate(truth, belief, np.arange(len(points)) % 5 == 4)["heldout_rmse"]
        panels.extend([(axes[0, column], belief.mean, f"{name}: mean; RMSE={rmse:.3f}", "viridis", truth.min(), truth.max()),
            (axes[1, column], belief.std, f"{name}: latent posterior std", "magma", 0, maximum_std)])
    for axis, values, title, cmap, low, high in panels:
        plot = axis.imshow(values.reshape(config.grid_size, config.grid_size), origin="lower", extent=extent,
                           cmap=cmap, vmin=low, vmax=high)
        axis.scatter(points[indices, 0], points[indices, 1], s=9, edgecolors="white", facecolors="none", linewidths=0.5)
        axis.set_title(title, fontsize=9)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        fig.colorbar(plot, ax=axis, shrink=0.75)
    fig.suptitle(f"{LABELS[scenario]}, predefined field 0, n={len(indices)}; same measured sites; no travel planning")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def render_diagnostics(path, comparisons, region_rows, config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for scenario, color in zip(SCENARIOS, ("#27754d", "#b5762b", "#376aa3")):
        selected = [row for row in comparisons if row["scenario"] == scenario and row["model"] == "estimated"]
        counts = [row["sample_size"] for row in selected]
        axes[0].plot(counts, [row["heldout_rmse_difference_vs_oracle"] for row in selected],
                     "o-", color=color, label=LABELS[scenario])
        intervals = np.asarray([row["heldout_rmse_difference_interval"] for row in selected])
        axes[0].fill_between(counts, intervals[:, 0], intervals[:, 1], color=color, alpha=0.13)
    axes[0].axhline(0, color="black", linewidth=1, linestyle="--")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(counts, [str(value) for value in counts])
    axes[0].set_xlabel("Shared acquired measurements")
    axes[0].set_ylabel("Estimated minus oracle: mean latent RMSE")
    axes[0].set_title("Paired oracle gaps; shading: field-bootstrap 95% intervals", fontsize=10)
    axes[0].legend(fontsize=8)
    regions = ("bottom", "middle", "top")
    for model in ("oracle", "estimated"):
        values, intervals = [], []
        for region in regions:
            selected = [float(row["heldout_coverage_95"]) for row in region_rows
                if row["scenario"] == "nonstationary_mixture" and int(row["sample_size"]) == config.sample_sizes[-1]
                and row["model"] == model and row["region"] == region]
            values.append(float(np.mean(selected)))
            intervals.append(bootstrap_mean(selected))
        axes[1].plot(range(3), values, "o-", color=COLORS[model], label=model)
        intervals = np.asarray(intervals)
        axes[1].fill_between(range(3), intervals[:, 0], intervals[:, 1], color=COLORS[model], alpha=0.13)
    axes[1].axhline(0.95, color="black", linewidth=1, linestyle="--", label="Nominal 95%")
    axes[1].set_xticks(range(3), ("Bottom: smooth", "Middle: transition", "Top: rough"))
    axes[1].set_ylim(0.8, 1.005)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set_ylabel("Mean latent 95% interval coverage")
    axes[1].set_title(f"Nonstationary regional calibration at n={config.sample_sizes[-1]}", fontsize=10)
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle("Residual oracle gap and regional coverage — same observations for every mapper")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_study(config, estimator, output):
    config.validate()
    estimator.validate()
    output = Path(output)
    if (output / "config.json").exists():
        raise ValueError("Use a fresh folder to preserve this benchmark")
    output.mkdir(parents=True, exist_ok=True)
    settings = {"environment": asdict(config), "estimator": asdict(estimator), "scenarios": SCENARIOS, "models": MODELS,
        "design": "uniform random permitted-site permutation per field; shared across scenarios/models; nested prefixes",
        "evaluation": "indices % 5 == 4; fixed across sample sizes; latent truth evaluator-only",
        "nonstationarity": "zero mean, unit variance; cos(pi*y/2) and sin(pi*y/2) weighted independent long/short GPs",
        "stationary_mixture_control": "equal-weight covariance mixture of the same two stationary RBF components",
        "fitting": "fresh initial parameters at each n; one maximum-likelihood snapshot fit with two restarts; no smaller-n warm start",
        "uncertainty": "latent posterior conditional on fitted covariance; parameter uncertainty omitted",
        "timing": "fresh complete fit and conditioning at each n, including construction; excludes evaluation/rendering",
        "constraints": "same-observation mapper benchmark; no travel budget, planner, or neural-policy inference",
        "data_status": "new independent field replicates; archived development benchmark for future DeepKriging comparison",
        "generator_cholesky_jitter": 1e-10, "bootstrap": "5000 field resamples, seed 1700000; sites are not IID replicates",
        "cpu_threads": 1, "python_version": platform.python_version(), "numpy_version": np.__version__, "sklearn_version": sklearn.__version__}
    (output / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
    observations, evaluation_only = generate_dataset(config)
    np.savez_compressed(output / "observations.npz", **observations)
    np.savez_compressed(output / "evaluation_only.npz", **evaluation_only)
    (output / "dataset_manifest.json").write_text(json.dumps({name:
        hashlib.sha256((output / name).read_bytes()).hexdigest() for name in ("observations.npz", "evaluation_only.npz")}, indent=2) + "\n")
    (output / "requirements-run.txt").write_text("\n".join(sorted(
        f"{item.metadata['Name']}=={item.version}" for item in metadata.distributions())) + "\n")
    points = observations["points"]
    mask = evaluation_only["evaluation_mask"]
    rows, region_rows, fits = [], [], []
    for offset, scenario in enumerate(SCENARIOS):
        true_prior = evaluation_only["true_covariances"][offset]
        for field in range(config.fields):
            truth = evaluation_only["truth"][offset, field]
            previous_oracle_variance = float("inf")
            for count in config.sample_sizes:
                indices = observations["observation_indices"][field, :count]
                values = observations["observed_values"][offset, field, :count]
                examples = {}
                for model in MODELS:
                    began = time.perf_counter()
                    if model == "oracle":
                        belief = fit_snapshot(model, points, indices, values, config.noise_std, estimator, oracle_prior=true_prior)
                    else:
                        belief = fit_snapshot(model, points, indices, values, config.noise_std, estimator)
                    seconds = time.perf_counter() - began
                    length = getattr(belief, "length_scale", estimator.initial_length_scale if model == "fixed_initial" else
                                     config.stationary_length if scenario == "stationary_rbf" else None)
                    process_variance = float(getattr(belief, "process_variance", estimator.initial_process_variance if model == "fixed_initial" else 1.0))
                    metrics = evaluate(truth, belief, mask)
                    row = {"scenario": scenario, "field": field, "sample_size": count, "model": model,
                        **metrics, "heldout_mse": metrics["heldout_rmse"]**2,
                        "heldout_mean_variance": float(np.mean(np.diag(belief.covariance)[mask])),
                        "update_seconds": seconds, "fit_seconds": float(getattr(belief, "total_fit_seconds", 0.0)),
                        "model_length_scale": length, "model_process_variance": process_variance,
                        "length_scale_absolute_error": abs(length - config.stationary_length)
                            if scenario == "stationary_rbf" and length is not None else None,
                        "process_variance_absolute_error": abs(process_variance - 1),
                        "fit_calls": len(getattr(belief, "fit_history", [])),
                        "boundary_fit_calls": sum(event["boundary_hit"] for event in getattr(belief, "fit_history", [])),
                        "rejected_fit_calls": sum(not event["accepted"] for event in getattr(belief, "fit_history", []))}
                    rows.append(row)
                    for region, region_mask in (("bottom", points[:, 1] <= 0.25),
                        ("middle", (points[:, 1] > 0.25) & (points[:, 1] < 0.75)), ("top", points[:, 1] >= 0.75)):
                        subset = mask & region_mask
                        if subset.any():
                            region_rows.append({"scenario": scenario, "field": field, "sample_size": count, "model": model,
                                "region": region, "heldout_sites": int(subset.sum()), **evaluate(truth, belief, subset)})
                    fits.extend({"scenario": scenario, "field": field, "sample_size": count, **event}
                                for event in getattr(belief, "fit_history", []))
                    if model == "oracle":
                        if row["heldout_mean_variance"] > previous_oracle_variance + 1e-10:
                            raise RuntimeError("Nested oracle covariance failed monotonicity")
                        previous_oracle_variance = row["heldout_mean_variance"]
                    if field == 0 and count in (config.sample_sizes[0], config.sample_sizes[-1]):
                        examples[model] = belief
                if examples:
                    render_example(output / f"example_{scenario}_n{count}.png", scenario, truth,
                                   true_prior, examples, config, indices)
            if (field + 1) % 16 == 0 or field + 1 == config.fields:
                print(f"Completed {scenario}: {field + 1}/{config.fields} fields", flush=True)
    summaries, comparisons = summarize(rows, config)
    write_csv(output / "metrics.csv", rows)
    write_csv(output / "regional_metrics.csv", region_rows)
    if fits:
        write_csv(output / "parameter_fits.csv", fits)
    (output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    (output / "paired_comparisons.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    diagnostics = {"independent_fields_per_scenario": config.fields, "scenario_field_instances": len(SCENARIOS) * config.fields,
        "mapper_evaluations": len(rows), "reserved_evaluation_sites": int(mask.sum()),
        "permitted_observation_sites": int((~mask).sum()), "same_nested_observations_for_all_models": True,
        "all_oracle_variance_curves_nonincreasing": True, "fit_calls": len(fits),
        "boundary_fit_calls": sum(event["boundary_hit"] for event in fits),
        "rejected_fit_calls": sum(not event["accepted"] for event in fits)}
    (output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    render_curves(output / "learning_curves.png", summaries)
    render_diagnostics(output / "oracle_gaps_and_regions.png", comparisons, region_rows, config)
    return summaries, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fields", type=int, default=64)
    parser.add_argument("--grid-size", type=int, default=15)
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    parser.add_argument("--output", type=Path, default=Path("outputs/sample-size-study"))
    args = parser.parse_args()
    config = SampleSizeConfig(grid_size=args.grid_size, fields=args.fields, sample_sizes=tuple(args.sample_sizes))
    with threadpool_limits(limits=1):
        summaries, diagnostics = run_study(config, EstimatorConfig(), args.output)
    for row in summaries:
        print(f"{row['scenario']:25s} n={row['sample_size']:3d} {row['model']:14s} "
              f"RMSE={row['heldout_rmse']:.4f} coverage={row['heldout_coverage_95']:.3f}")
    print(json.dumps(diagnostics, indent=2))
    print("Artifacts:", args.output.resolve())


if __name__ == "__main__":
    main()
