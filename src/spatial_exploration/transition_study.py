"""Frozen, paired mapper comparison as dependence transitions become sharper."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .baseline import write_csv
from .deepkriging import (NeuralMapperConfig, calibration_multiplier, matched_coordinate_width,
                         parameter_count, standardized_features)
from .deepkriging_study import (NEURAL_MODELS, MODEL_COLORS, fit_dataset, flatten_observations,
                               prediction_metrics, summarize, write_fit_diagnostics)
from .estimated_gp import EstimatorConfig
from .sample_size_study import SampleSizeConfig, bootstrap_mean, fit_snapshot
from .transition_fields import (TRANSITIONS, WIDTHS, TRANSITION_LABELS, generate_transition_dataset,
                                transition_weights, transition_covariances)


MODELS = ("oracle", "estimated", *NEURAL_MODELS)


def oracle_gap_changes(rows, counts):
    """Post-hoc diagnostic: separate changed task difficulty from oracle excess."""
    result = []
    for count in counts:
        lookup = {(row["scenario"], row["model"], row["field"]): row["heldout_rmse"]
                  for row in rows if row["sample_size"] == count}
        fields = sorted({field for scenario, model, field in lookup})
        for model in MODELS[1:]:
            base = np.asarray([lookup["gradual", model, field] - lookup["gradual", "oracle", field] for field in fields])
            for scenario in TRANSITIONS[1:]:
                gap = np.asarray([lookup[scenario, model, field] - lookup[scenario, "oracle", field] for field in fields])
                result.append({"scenario": scenario, "sample_size": count, "model": model,
                    "gradual_mean_rmse_excess": float(base.mean()), "condition_mean_rmse_excess": float(gap.mean()),
                    "paired_mean_excess_change": float((gap-base).mean()),
                    "paired_bootstrap_95_interval": bootstrap_mean(gap-base), "analysis_status": "post-hoc descriptive diagnostic"})
    return result


def save_dataset(output, name, observations, evaluator):
    np.savez_compressed(output / f"{name}_observations.npz", **observations)
    np.savez_compressed(output / f"{name}_evaluation_only.npz", **evaluator)


def render_transition_results(output, environment, observations, evaluator, predictions, summaries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for column, scenario in enumerate(TRANSITIONS):
        for model in MODELS:
            subset = [row for row in summaries if row["scenario"] == scenario and row["model"] == model]
            for row, key in enumerate(("heldout_rmse", "heldout_coverage_95")):
                axis = axes[row, column]
                axis.plot(environment.sample_sizes, [item[key] for item in subset], "o-", color=MODEL_COLORS[model], label=model)
                intervals = np.asarray([item[key + "_interval"] for item in subset])
                axis.fill_between(environment.sample_sizes, intervals[:, 0], intervals[:, 1], color=MODEL_COLORS[model], alpha=0.1)
        for row in range(2):
            axis = axes[row, column]
            axis.set_xscale("log", base=2)
            axis.set_xticks(environment.sample_sizes, [str(n) for n in environment.sample_sizes])
            axis.set_xlabel("Identical measurement locations per field")
            axis.grid(alpha=0.2)
        axes[0, column].set_title(TRANSITION_LABELS[scenario], fontsize=10)
        axes[0, column].set_ylabel("Mean held-out latent RMSE")
        axes[0, column].legend(fontsize=8)
        axes[1, column].axhline(0.95, color="black", linestyle="--", linewidth=1)
        axes[1, column].set_ylim(0, 1.02)
        axes[1, column].set_ylabel("95% coverage (neural bands calibrated)")
    fig.suptitle(f"Transition sharpness only — fixed length scales {environment.long_length}/{environment.short_length}, mean 0, variance 1")
    fig.savefig(output / "comparison.png", dpi=150)
    plt.close(fig)

    points = observations["points"]
    yy = np.linspace(0, 1, 501)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for offset, (scenario, width) in enumerate(zip(TRANSITIONS, WIDTHS)):
        a, b = transition_weights(yy, width)
        axes[0].plot(yy, b**2, label=TRANSITION_LABELS[scenario])
        # Prior correlation with a common near-boundary anchor; not an estimated model.
        anchor = int(np.argmin(np.linalg.norm(points - [0.5, 3 / 7], axis=1)))
        column = np.flatnonzero(np.isclose(points[:, 0], points[anchor, 0]))
        axes[1].plot(points[column, 1], evaluator["true_covariances"][offset, anchor, column], "o-", label=scenario)
    axes[0].set_xlabel("y coordinate")
    axes[0].set_ylabel("Short-component variance weight b(y)^2")
    axes[1].axvline(points[anchor, 1], color="gray", linestyle=":")
    axes[1].axvline(0.5, color="black", linestyle="--")
    axes[1].set_xlabel("y coordinate at the same x")
    axes[1].set_ylabel("True correlation with anchor below boundary")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.suptitle("Generating dependence: sharper transition, no impermeable wall")
    fig.savefig(output / "generating_dependence.png", dpi=150)
    plt.close(fig)

    count = environment.sample_sizes[-1]
    truth = evaluator["truth"][:, 0]
    low, high = truth.min(), truth.max()
    fig, axes = plt.subplots(3, 4, figsize=(13, 10), constrained_layout=True)
    half = 0.5 / (environment.grid_size - 1)
    for offset, scenario in enumerate(TRANSITIONS):
        arrays = [truth[offset]] + [predictions[f"{model}_n{count}_mean"][offset * environment.fields]
                                      for model in ("estimated", *NEURAL_MODELS)]
        for column, (label, array) in enumerate(zip(("Truth: evaluator only", "Fitted GP", *NEURAL_MODELS), arrays)):
            axis = axes[offset, column]
            image = axis.imshow(array.reshape(environment.grid_size, environment.grid_size), origin="lower",
                extent=(-half, 1 + half, -half, 1 + half), vmin=low, vmax=high, cmap="viridis")
            observed = observations["observation_indices"][0, :count]
            axis.scatter(points[observed, 0], points[observed, 1], facecolors="none", edgecolors="white", s=8, linewidths=0.5)
            axis.axhline(0.5, color="white", linestyle="--", alpha=0.5)
            if offset == 0:
                axis.set_title(label, fontsize=10)
            if column == 0:
                axis.set_ylabel(TRANSITION_LABELS[scenario], fontsize=9)
    fig.colorbar(image, ax=axes, shrink=0.6)
    fig.suptitle(f"Predefined field 0, n={count}: common components, noise and observed locations")
    fig.savefig(output / "example_field0.png", dpi=150)
    plt.close(fig)


def run_study(output, environment=None, neural=None, estimator=None):
    environment = environment or SampleSizeConfig(field_seed_start=2500000, design_seed_start=2600000)
    neural, estimator = neural or NeuralMapperConfig(), estimator or EstimatorConfig()
    for config in (environment, neural, estimator):
        config.validate()
    if environment.grid_size < 15 and environment.fields > 2:
        raise ValueError("Resolve the sharp transition with at least the recorded 15x15 grid")
    output = Path(output)
    if (output / "protocol.json").exists():
        raise ValueError("Use a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    torch.set_num_threads(1)
    tuning_env = replace(environment, fields=neural.tuning_fields, field_seed_start=2300000, design_seed_start=2310000)
    calibration_env = replace(environment, fields=neural.calibration_fields, field_seed_start=2400000, design_seed_start=2410000)
    seed_sets = [set(range(c.field_seed_start, c.field_seed_start + c.fields))
                 for c in (environment, tuning_env, calibration_env)]
    if any(seed_sets[a] & seed_sets[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Field splits must be disjoint")
    protocol = {"environment": asdict(environment), "neural": asdict(neural), "estimator": asdict(estimator),
        "tuning_environment": asdict(tuning_env), "calibration_environment": asdict(calibration_env),
        "conditions": dict(zip(TRANSITIONS, WIDTHS)),
        "scope": "dependence-transition sharpness only; no physical barrier, terrain, planner or movement",
        "controls": "fixed long/short lengths, zero mean, unit variance, noise, geometry and nested designs",
        "pairing": "identical independent long/short GP component realizations, noise and site permutations across conditions",
        "selection": f"same {len(neural.epochs) * len(neural.weight_decays)} epoch/weight-decay candidates per model/n; pooled independent tuning-field latent MSE",
        "calibration": "pooled independent calibration maps; fixed marginal ensemble interval multiplier per model/n",
        "freeze": "freeze all hyperparameters and scales before generating the new benchmark maps",
        "uncertainty": "raw initialization spread and independently calibrated bands, not a joint posterior covariance",
        "regions": "fixed y<0.35, 0.35<=y<=0.65, y>0.65; boundary band does not change with sharpness",
        "replications": "64 independent field pairs by default; conditions/counts/sites are not independent replicates",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
        "cpu_threads": 1, "component_cholesky_jitter": 1e-10}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    tuning, tuning_eval = generate_transition_dataset(tuning_env)
    calibration, calibration_eval = generate_transition_dataset(calibration_env)
    save_dataset(output, "tuning", tuning, tuning_eval)
    save_dataset(output, "calibration", calibration, calibration_eval)
    points, mask = tuning["points"], tuning_eval["evaluation_mask"]
    features, transforms = {}, {}
    widths = {"coordinate_dnn": matched_coordinate_width(neural), "deepkriging": neural.basis_width}
    for model in NEURAL_MODELS:
        features[model], transforms[model + "_center"], transforms[model + "_scale"] = standardized_features(points, model, neural)
    np.savez_compressed(output / "feature_transforms.npz", **transforms)
    parameters = {model: parameter_count(features[model].shape[1], widths[model]) for model in NEURAL_MODELS}
    if abs(parameters["coordinate_dnn"] / parameters["deepkriging"] - 1) > 0.01:
        raise ValueError("Parameter budgets must match within 1%")
    selected, selection_rows, scales, calibration_rows = {}, [], {}, []
    compute = {"tuning_fit_seconds": 0.0, "calibration_fit_seconds": 0.0, "benchmark_neural_fit_seconds": 0.0,
               "benchmark_gp_fit_seconds": 0.0}
    tuning_truth = tuning_eval["truth"].reshape(-1, len(points))
    calibration_truth = calibration_eval["truth"].reshape(-1, len(points))
    for model in NEURAL_MODELS:
        for count in environment.sample_sizes:
            candidates = []
            for decay in neural.weight_decays:
                snapshots, seconds = fit_dataset(tuning, features[model], model, count, neural, 2320000, decay, neural.epochs)
                compute["tuning_fit_seconds"] += seconds[max(neural.epochs)]
                for epoch, prediction in snapshots.items():
                    residual = tuning_truth[:, mask] - prediction["mean"][:, mask]
                    row = {"model": model, "sample_size": count, "epochs": epoch, "weight_decay": decay,
                           "validation_mse": float(np.mean(residual**2)), "fit_seconds": seconds[epoch]}
                    candidates.append(row)
                    selection_rows.append(row)
            best = min(candidates, key=lambda row: row["validation_mse"])
            choice = {key: best[key] for key in ("epochs", "weight_decay")}
            key = f"{model}_n{count}"
            selected[key] = choice
            snapshots, seconds = fit_dataset(calibration, features[model], model, count, neural, 2420000,
                                             choice["weight_decay"], (choice["epochs"],))
            compute["calibration_fit_seconds"] += seconds[choice["epochs"]]
            prediction = snapshots[choice["epochs"]]
            scales[key] = calibration_multiplier(calibration_truth, prediction["mean"], prediction["raw_variance"], mask)
            calibration_rows.append({"model": model, "sample_size": count, "variance_multiplier": scales[key]})
            print(f"Frozen {key}: {choice}", flush=True)
    write_csv(output / "model_selection.csv", selection_rows)
    write_csv(output / "interval_calibration.csv", calibration_rows)
    (output / "frozen_protocol.json").write_text(json.dumps({"selected": selected, "variance_multipliers": scales,
        "parameters_per_member": parameters, "frozen_before_benchmark_generation": True}, indent=2) + "\n")

    observations, evaluator = generate_transition_dataset(environment)
    save_dataset(output, "benchmark", observations, evaluator)
    fields = environment.fields
    truth = evaluator["truth"].reshape(-1, len(points))
    rows, regions, member_rows, predictions = [], [], [], {}
    region_masks = {"lower": mask & (points[:, 1] < 0.35),
                    "boundary": mask & (np.abs(points[:, 1] - 0.5) <= 0.15),
                    "upper": mask & (points[:, 1] > 0.65)}

    def record(model, count, means, variances, raw_variances, seconds, timing_type):
        for index in range(len(truth)):
            common = {"scenario": TRANSITIONS[index // fields], "field": index % fields,
                      "sample_size": count, "model": model}
            metrics = prediction_metrics(truth[index], means[index], variances[index], mask)
            raw = prediction_metrics(truth[index], means[index], raw_variances[index], mask)
            rows.append({**common, **metrics, "raw_heldout_coverage_95": raw["heldout_coverage_95"],
                         "fit_seconds": seconds[index], "timing_type": timing_type})
            for name, region_mask in region_masks.items():
                regions.append({**common, "region": name,
                    **prediction_metrics(truth[index], means[index], variances[index], region_mask)})

    for count in environment.sample_sizes:
        indices, values = flatten_observations(observations, count)
        for model in ("oracle", "estimated"):
            means, variances, seconds = [], [], []
            for index in range(len(truth)):
                began = time.perf_counter()
                kwargs = {"oracle_prior": evaluator["true_covariances"][index // fields]} if model == "oracle" else {}
                belief = fit_snapshot(model, points, indices[index], values[index], environment.noise_std, estimator, **kwargs)
                seconds.append(time.perf_counter() - began)
                means.append(belief.mean.copy())
                variances.append(np.diag(belief.covariance).copy())
            compute["benchmark_gp_fit_seconds"] += sum(seconds)
            means, variances = np.asarray(means), np.asarray(variances)
            predictions[f"{model}_n{count}_mean"] = means
            predictions[f"{model}_n{count}_variance"] = variances
            record(model, count, means, variances, variances, seconds, "single-map fit and conditioning")
        for model in NEURAL_MODELS:
            key = f"{model}_n{count}"
            choice = selected[key]
            snapshots, seconds = fit_dataset(observations, features[model], model, count, neural, 2520000,
                choice["weight_decay"], (choice["epochs"],), folder=output / "checkpoints")
            prediction = snapshots[choice["epochs"]]
            compute["benchmark_neural_fit_seconds"] += seconds[choice["epochs"]]
            for name, array in prediction.items():
                predictions[key + "_" + name] = array
            record(model, count, prediction["mean"], prediction["raw_variance"] * scales[key], prediction["raw_variance"],
                np.full(len(truth), seconds[choice["epochs"]] / len(truth)), "amortized independent batched fits and grid prediction")
            for index in range(len(truth)):
                for member in range(neural.members):
                    residual = prediction["members"][index, member, mask] - truth[index, mask]
                    member_rows.append({"scenario": TRANSITIONS[index // fields], "field": index % fields,
                        "sample_size": count, "model": model, "member": member,
                        "heldout_rmse": float(np.sqrt(np.mean(residual**2)))})
        print(f"Evaluated n={count}: {fields} paired fields, three transitions, four mappers", flush=True)
    write_csv(output / "metrics.csv", rows)
    write_csv(output / "regional_metrics.csv", regions)
    write_csv(output / "member_metrics.csv", member_rows)
    write_fit_diagnostics(output / "fit_diagnostics.csv", observations, predictions, environment.sample_sizes)
    np.savez_compressed(output / "predictions.npz", **predictions)
    summaries, comparisons = summarize(rows, environment.sample_sizes, scenarios=TRANSITIONS)
    changes = []
    for count in environment.sample_sizes:
        for model in MODELS:
            base = sorted([row for row in rows if row["scenario"] == "gradual" and row["sample_size"] == count
                           and row["model"] == model], key=lambda row: row["field"])
            for scenario in TRANSITIONS[1:]:
                changed = sorted([row for row in rows if row["scenario"] == scenario and row["sample_size"] == count
                                  and row["model"] == model], key=lambda row: row["field"])
                difference = np.asarray([a["heldout_rmse"] - b["heldout_rmse"] for a, b in zip(changed, base)])
                changes.append({"scenario": scenario, "reference": "gradual", "sample_size": count, "model": model,
                    "mean_paired_rmse_change": float(difference.mean()), "paired_bootstrap_95_interval": bootstrap_mean(difference)})
    for filename, value in (("summary.json", summaries), ("paired_model_comparisons.json", comparisons),
                            ("paired_transition_changes.json", changes), ("compute.json", compute),
                            ("oracle_gap_diagnostics.json", oracle_gap_changes(rows, environment.sample_sizes))):
        (output / filename).write_text(json.dumps(value, indent=2) + "\n")
    render_transition_results(output, environment, observations, evaluator, predictions, summaries)
    diagnostics = {"fields_per_condition": fields, "mapper_evaluations": len(rows),
        "neural_member_fits": len(member_rows), "evaluation_sites": int(mask.sum()),
        "shared_component_fields_noise_design": True, "frozen_before_benchmark_generation": True,
        "all_true_covariance_diagonals_unit": bool(np.allclose(np.diagonal(evaluator["true_covariances"], axis1=1, axis2=2), 1)),
        "sharp_transition_10_to_90_width": float(2 * np.log(9) * WIDTHS[-1]),
        "sharp_transition_width_in_grid_spacings": float(2 * np.log(9) * WIDTHS[-1] * (environment.grid_size - 1))}
    (output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    manifest = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in output.glob("*.npz")}
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return summaries, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/transition-study"))
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        summaries, diagnostics = run_study(args.output)
    for row in summaries:
        if row["sample_size"] == 128:
            print(f"{row['scenario']:9s} {row['model']:16s} RMSE={row['heldout_rmse']:.4f} coverage={row['heldout_coverage_95']:.3f}")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
