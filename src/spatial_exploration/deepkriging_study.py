"""Same-observation neural mapper benchmark with independent model selection."""

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
import platform
from pathlib import Path
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .baseline import write_csv
from .deepkriging import (NeuralMapperConfig, calibration_multiplier, fit_ensemble_snapshots,
    matched_coordinate_width, parameter_count, standardized_features)
from .estimated_gp import EstimatorConfig
from .sample_size_study import (SampleSizeConfig, SCENARIOS, LABELS, bootstrap_mean,
    fit_snapshot, generate_dataset)


NEURAL_MODELS = ("coordinate_dnn", "deepkriging")
MODEL_COLORS = {"oracle": "#27754d", "fixed_initial": "#b5762b", "estimated": "#376aa3",
                "coordinate_dnn": "#885c9b", "deepkriging": "#cf5656"}


def read_rows(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def flatten_observations(observations, count):
    fields = len(observations["field_ids"])
    scenarios = observations["observed_values"].shape[0]
    indices = np.tile(observations["observation_indices"][:, :count], (scenarios, 1))
    values = observations["observed_values"][:, :, :count].reshape(scenarios * fields, count)
    return indices, values


def member_seeds(fields, members, base, scenarios=len(SCENARIOS)):
    return [base + (scenario * fields + field) * members + member for scenario in range(scenarios)
            for field in range(fields) for member in range(members)]


def fit_dataset(observations, features, model, count, config, seed_base, decay, checkpoints, folder=None):
    indices, values = flatten_observations(observations, count)
    all_seeds = member_seeds(len(observations["field_ids"]), config.members, seed_base,
                             observations["observed_values"].shape[0])
    predictions = {epoch: {key: [] for key in ("mean", "raw_variance", "members")} for epoch in checkpoints}
    seconds = {epoch: 0.0 for epoch in checkpoints}
    for start in range(0, len(indices), config.chunk_maps):
        stop = min(start + config.chunk_maps, len(indices))
        seeds = all_seeds[start * config.members:stop * config.members]
        checkpoint = folder / f"{model}-n{count}-maps{start}-{stop}.pt" if folder else None
        results, timings, _ = fit_ensemble_snapshots(features, indices[start:stop], values[start:stop],
            model, config, seeds, decay, checkpoints, save_path=checkpoint)
        for epoch, result in results.items():
            seconds[epoch] += timings[epoch]
            for key, array in result.items():
                predictions[epoch][key].append(array)
    predictions = {epoch: {key: np.concatenate(arrays, axis=0) for key, arrays in result.items()}
                   for epoch, result in predictions.items()}
    return predictions, seconds


def prediction_metrics(truth, mean, variance, mask):
    residual = truth[mask] - mean[mask]
    std = np.sqrt(np.maximum(variance[mask], 1e-12))
    return {"heldout_rmse": float(np.sqrt(np.mean(residual**2))),
        "heldout_mse": float(np.mean(residual**2)),
        "heldout_coverage_95": float(np.mean(np.abs(residual) <= 1.96 * std)),
        "heldout_mean_std": float(np.mean(std)),
        "heldout_mean_variance": float(np.mean(variance[mask]))}


def write_fit_diagnostics(path, observations, predictions, counts):
    """Observed-target interpolation diagnostic; no hidden field values used."""
    fields = len(observations["field_ids"])
    rows = []
    for model in NEURAL_MODELS:
        for count in counts:
            indices, values = flatten_observations(observations, count)
            mean = predictions[f"{model}_n{count}_mean"]
            for index in range(len(values)):
                residual = mean[index, indices[index]] - values[index]
                rows.append({"scenario": str(observations["scenario_names"][index // fields]), "field": index % fields,
                    "sample_size": count, "model": model,
                    "observed_noisy_target_rmse": float(np.sqrt(np.mean(residual**2)))})
    write_csv(path, rows)


def summarize(rows, counts, scenarios=SCENARIOS):
    summaries, comparisons = [], []
    models = tuple(dict.fromkeys(row["model"] for row in rows))
    for scenario in scenarios:
        for count in counts:
            by_model = {model: sorted([row for row in rows if row["scenario"] == scenario
                and row["sample_size"] == count and row["model"] == model], key=lambda row: row["field"]) for model in models}
            for model, selected in by_model.items():
                summary = {"scenario": scenario, "sample_size": count, "model": model, "fields": len(selected)}
                for key in ("heldout_rmse", "heldout_mse", "heldout_coverage_95", "raw_heldout_coverage_95",
                            "heldout_mean_std", "heldout_mean_variance", "fit_seconds"):
                    values = [row[key] for row in selected]
                    summary[key] = float(np.mean(values))
                    if key in ("heldout_rmse", "heldout_coverage_95"):
                        summary[key + "_interval"] = bootstrap_mean(values)
                summaries.append(summary)
            for model, reference in (("coordinate_dnn", "estimated"), ("deepkriging", "estimated"), ("deepkriging", "coordinate_dnn")):
                deltas = np.asarray([row["heldout_rmse"] - baseline["heldout_rmse"]
                    for row, baseline in zip(by_model[model], by_model[reference])])
                comparisons.append({"scenario": scenario, "sample_size": count, "model": model,
                    "reference": reference, "mean_paired_rmse_difference": float(deltas.mean()),
                    "paired_field_bootstrap_95_percent_interval": bootstrap_mean(deltas),
                    "lower_rmse_fields": int(np.sum(deltas < -1e-10)),
                    "higher_rmse_fields": int(np.sum(deltas > 1e-10))})
    return summaries, comparisons


def render_results(path, summaries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for column, scenario in enumerate(SCENARIOS):
        for row, key in enumerate(("heldout_rmse", "heldout_coverage_95")):
            axis = axes[row, column]
            for model in ("oracle", "estimated", *NEURAL_MODELS):
                selected = [item for item in summaries if item["scenario"] == scenario and item["model"] == model]
                counts = [item["sample_size"] for item in selected]
                axis.plot(counts, [item[key] for item in selected], "o-", color=MODEL_COLORS[model], label=model)
                intervals = np.asarray([item[key + "_interval"] for item in selected])
                axis.fill_between(counts, intervals[:, 0], intervals[:, 1], color=MODEL_COLORS[model], alpha=0.1)
            axis.set_xscale("log", base=2)
            axis.set_xticks(counts, [str(count) for count in counts])
            axis.set_xlabel("Identical acquired measurements per map")
            axis.grid(alpha=0.2)
            if row == 0:
                axis.set_title(LABELS[scenario], fontsize=10)
                axis.set_ylabel("Mean held-out latent RMSE")
                axis.legend(fontsize=8)
            else:
                axis.axhline(0.95, color="black", linestyle="--", linewidth=1)
                axis.set_ylim(0, 1.02)
                axis.set_ylabel("Empirical 95% interval coverage")
    fig.suptitle("Per-map neural fits — configuration selected on independent fields; existing development benchmark")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_uncertainty(path, summaries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for column, scenario in enumerate(SCENARIOS):
        for row, key in enumerate(("raw_heldout_coverage_95", "heldout_mean_std")):
            axis = axes[row, column]
            for model in ("oracle", "estimated", *NEURAL_MODELS):
                selected = [item for item in summaries if item["scenario"] == scenario and item["model"] == model]
                counts = [item["sample_size"] for item in selected]
                axis.plot(counts, [item[key] for item in selected], "o-", color=MODEL_COLORS[model], label=model)
            axis.set_xscale("log", base=2)
            axis.set_xticks(counts, [str(count) for count in counts])
            axis.set_xlabel("Identical acquired measurements per map")
            axis.grid(alpha=0.2)
            if row == 0:
                axis.set_title(LABELS[scenario], fontsize=10)
                axis.axhline(0.95, color="black", linestyle="--", linewidth=1)
                axis.set_ylim(0, 1.02)
                axis.set_ylabel("Coverage before neural interval calibration")
                axis.legend(fontsize=8)
            else:
                axis.set_yscale("log")
                axis.set_ylabel("Mean reported std after calibration (log scale)")
    fig.suptitle("Uncertainty audit — raw ensemble coverage and calibrated interval width")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_example(path, truth, predictions, observed_indices, size, scenario, count):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = np.column_stack([array.ravel() for array in np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))])
    half = 0.5 / (size - 1)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for axis, (name, values) in zip(axes, {"Latent truth (evaluation only)": truth, **predictions}.items()):
        plot = axis.imshow(values.reshape(size, size), origin="lower", extent=(-half, 1 + half, -half, 1 + half),
                           vmin=truth.min(), vmax=truth.max(), cmap="viridis")
        axis.scatter(points[observed_indices, 0], points[observed_indices, 1], s=12,
                     edgecolors="white", facecolors="none", linewidths=0.6)
        axis.set_title(name, fontsize=10)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        fig.colorbar(plot, ax=axis, shrink=0.75)
    fig.suptitle(f"{LABELS[scenario]}, predefined field 0, n={count}; same observed sites")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_study(reference, output, config=None):
    config = config or NeuralMapperConfig()
    config.validate()
    reference, output = Path(reference), Path(output)
    source, destination = reference.resolve(), output.resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Keep neural outputs separate from the archived GP benchmark")
    if (output / "config.json").exists():
        raise ValueError("Use a fresh output folder")
    source_config = json.loads((reference / "config.json").read_text())
    environment = SampleSizeConfig(**source_config["environment"])
    environment.validate()
    source_hashes = json.loads((reference / "dataset_manifest.json").read_text())
    for filename, digest in source_hashes.items():
        if hashlib.sha256((reference / filename).read_bytes()).hexdigest() != digest:
            raise RuntimeError("Archived benchmark data changed")
    with np.load(reference / "observations.npz") as stored:
        benchmark_observations = {key: stored[key].copy() for key in stored.files}
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    torch.set_num_threads(1)
    settings = {"neural": asdict(config), "environment": asdict(environment),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
        "source": str(reference.resolve()), "source_dataset_sha256": source_hashes,
        "method": "per-map adapted DeepKriging-style ReLU MLP with compact Wendland multiresolution features",
        "status": "core-representation implementation; not full paper reproduction",
        "features": "xy plus basis on 4/7/13 knot grids; support 2.5 times knot spacing; geometry-only standardization",
        "architecture_control": "two ReLU hidden layers; coordinate model width chosen to match parameter count within 1%",
        "fit": "fresh map/member initialization at every n; full observed-data MSE; no hidden targets or cross-map learned weights",
        "selection": f"{len(config.epochs) * len(config.weight_decays)} fixed epoch/weight-decay candidates per model/n; independent tuning-field latent MSE pooled across scenarios",
        "uncertainty": "three-initialization ensemble spread; not the paper's density estimator or a GP posterior",
        "calibration": "independent calibration fields; per-model/n pooled absolute standardized residual 95% quantile; no conformal guarantee",
        "data_status": "archived exposed development benchmark; future final claims require fresh held-out data",
        "paper": "https://www3.stat.sinica.edu.tw/statistica/oldpdf/A34n113.pdf",
        "author_code": "https://github.com/aleksada/DeepKriging/blob/master/PM25_application.py",
        "deviations": "smaller knots/networks; PyTorch independent batched fits; no dropout/batch norm; ensemble intervals instead of density regression"}
    (output / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
    points = benchmark_observations["points"]
    features, transforms = {}, {}
    for model in NEURAL_MODELS:
        features[model], center, scale = standardized_features(points, model, config)
        transforms[model + "_center"], transforms[model + "_scale"] = center, scale
    np.savez_compressed(output / "feature_transforms.npz", **transforms)
    widths = {"coordinate_dnn": matched_coordinate_width(config), "deepkriging": config.basis_width}
    counts_parameters = {model: parameter_count(features[model].shape[1], widths[model]) for model in NEURAL_MODELS}
    if abs(counts_parameters["coordinate_dnn"] / counts_parameters["deepkriging"] - 1) > 0.01:
        raise RuntimeError("Neural parameter budgets are not matched")
    tuning_config = replace(environment, fields=config.tuning_fields, field_seed_start=1800000, design_seed_start=1810000)
    calibration_config = replace(environment, fields=config.calibration_fields, field_seed_start=1900000, design_seed_start=1910000)
    tuning, tuning_evaluation = generate_dataset(tuning_config)
    calibration, calibration_evaluation = generate_dataset(calibration_config)
    for name, obs, ev in (("tuning", tuning, tuning_evaluation), ("calibration", calibration, calibration_evaluation)):
        np.savez_compressed(output / f"{name}_observations.npz", **obs)
        np.savez_compressed(output / f"{name}_evaluation_only.npz", **ev)
    if set(range(environment.field_seed_start, environment.field_seed_start + environment.fields)) & (
        set(range(1800000, 1800000 + config.tuning_fields)) | set(range(1900000, 1900000 + config.calibration_fields))):
        raise RuntimeError("Tuning/calibration maps overlap benchmark seeds")
    selection_rows, selected = [], {}
    compute = {"tuning_fit_seconds": 0.0, "calibration_fit_seconds": 0.0, "benchmark_fit_seconds": 0.0,
        "timing_scope": "initialization, optimization and grid prediction; excludes checkpoint I/O and dataset generation; one CPU thread"}
    tuning_truth = tuning_evaluation["truth"].reshape(-1, len(points))
    mask = tuning_evaluation["evaluation_mask"]
    for model in NEURAL_MODELS:
        for count in environment.sample_sizes:
            candidates = []
            for decay in config.weight_decays:
                predictions, seconds = fit_dataset(tuning, features[model], model, count,
                    config, 1820000, decay, config.epochs)
                compute["tuning_fit_seconds"] += seconds[max(config.epochs)]
                for epoch, result in predictions.items():
                    residual = result["mean"][:, mask] - tuning_truth[:, mask]
                    row = {"model": model, "sample_size": count, "weight_decay": decay, "epochs": epoch,
                        "validation_mse": float(np.mean(residual**2)),
                        "validation_mean_rmse": float(np.sqrt(np.mean(residual**2, axis=1)).mean()),
                        "tuning_fit_seconds": seconds[epoch]}
                    candidates.append(row)
                    selection_rows.append(row)
            winner = min(candidates, key=lambda row: row["validation_mse"])
            selected[f"{model}_n{count}"] = {"epochs": winner["epochs"], "weight_decay": winner["weight_decay"]}
            print(f"Selected {model} n={count}: epochs={winner['epochs']}, decay={winner['weight_decay']}", flush=True)
    write_csv(output / "model_selection.csv", selection_rows)
    scales, calibration_rows = {}, []
    calibration_truth = calibration_evaluation["truth"].reshape(-1, len(points))
    for model in NEURAL_MODELS:
        for count in environment.sample_sizes:
            choice = selected[f"{model}_n{count}"]
            predictions, seconds = fit_dataset(calibration, features[model], model, count,
                config, 1920000, choice["weight_decay"], (choice["epochs"],))
            compute["calibration_fit_seconds"] += seconds[choice["epochs"]]
            result = predictions[choice["epochs"]]
            multiplier = calibration_multiplier(calibration_truth, result["mean"], result["raw_variance"], mask)
            scales[f"{model}_n{count}"] = multiplier
            calibration_rows.append({"model": model, "sample_size": count, "variance_multiplier": multiplier})
    write_csv(output / "interval_calibration.csv", calibration_rows)
    (output / "frozen_protocol.json").write_text(json.dumps({"selected": selected, "variance_multipliers": scales,
        "parameters_per_member": counts_parameters, "widths": widths,
        "tuning_environment": asdict(tuning_config), "calibration_environment": asdict(calibration_config),
        "frozen_before_benchmark_predictions_or_loading_benchmark_targets": True}, indent=2) + "\n")
    # Benchmark targets are loaded only after hyperparameters and interval scales freeze.
    with np.load(reference / "evaluation_only.npz") as stored:
        evaluation = {key: stored[key].copy() for key in stored.files}
    truth = evaluation["truth"].reshape(-1, len(points))
    mask = evaluation["evaluation_mask"]
    rows, regional, member_rows = [], [], []
    for old in read_rows(reference / "metrics.csv"):
        rows.append({"scenario": old["scenario"], "field": int(old["field"]), "sample_size": int(old["sample_size"]),
            "model": old["model"], **{key: float(old[key]) for key in
                ("heldout_rmse", "heldout_mse", "heldout_coverage_95", "heldout_mean_std", "heldout_mean_variance")},
            "raw_heldout_coverage_95": float(old["heldout_coverage_95"]), "fit_seconds": float(old["update_seconds"]),
            "timing_type": "single-map fit", "calibration_multiplier": 1.0})
    fields = environment.fields
    saved_predictions = {}
    for model in NEURAL_MODELS:
        for count in environment.sample_sizes:
            choice = selected[f"{model}_n{count}"]
            predictions, seconds = fit_dataset(benchmark_observations, features[model], model, count,
                config, 2020000, choice["weight_decay"], (choice["epochs"],), folder=output / "checkpoints")
            compute["benchmark_fit_seconds"] += seconds[choice["epochs"]]
            result = predictions[choice["epochs"]]
            multiplier = scales[f"{model}_n{count}"]
            calibrated = result["raw_variance"] * multiplier
            saved_predictions[f"{model}_n{count}_mean"] = result["mean"]
            saved_predictions[f"{model}_n{count}_raw_variance"] = result["raw_variance"]
            saved_predictions[f"{model}_n{count}_members"] = result["members"]
            for index in range(len(truth)):
                scenario, field = SCENARIOS[index // fields], index % fields
                metrics = prediction_metrics(truth[index], result["mean"][index], calibrated[index], mask)
                raw = prediction_metrics(truth[index], result["mean"][index], result["raw_variance"][index], mask)
                rows.append({"scenario": scenario, "field": field, "sample_size": count, "model": model, **metrics,
                    "raw_heldout_coverage_95": raw["heldout_coverage_95"], "fit_seconds": seconds[choice["epochs"]] / len(truth),
                    "timing_type": "amortized independent batched-map fits", "calibration_multiplier": multiplier})
                for region, condition in (("bottom", points[:, 1] <= 0.25),
                    ("middle", (points[:, 1] > 0.25) & (points[:, 1] < 0.75)), ("top", points[:, 1] >= 0.75)):
                    regional.append({"scenario": scenario, "field": field, "sample_size": count, "model": model,
                        "region": region, **prediction_metrics(truth[index], result["mean"][index], calibrated[index], mask & condition),
                        "raw_heldout_coverage_95": prediction_metrics(truth[index], result["mean"][index],
                            result["raw_variance"][index], mask & condition)["heldout_coverage_95"]})
                for member in range(config.members):
                    residual = result["members"][index, member, mask] - truth[index, mask]
                    member_rows.append({"scenario": scenario, "field": field, "sample_size": count, "model": model,
                        "member": member, "heldout_rmse": float(np.sqrt(np.mean(residual**2)))})
            print(f"Evaluated {model} n={count}: {len(truth)} maps, {config.members} members each", flush=True)
    np.savez_compressed(output / "predictions.npz", **saved_predictions)
    write_fit_diagnostics(output / "fit_diagnostics.csv", benchmark_observations, saved_predictions, environment.sample_sizes)
    summaries, comparisons = summarize(rows, environment.sample_sizes)
    write_csv(output / "metrics.csv", rows)
    write_csv(output / "regional_metrics.csv", regional)
    write_csv(output / "member_metrics.csv", member_rows)
    (output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    (output / "paired_comparisons.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    render_results(output / "comparison.png", summaries)
    render_uncertainty(output / "uncertainty.png", summaries)
    for scenario_index, scenario in enumerate(SCENARIOS):
        for count in (environment.sample_sizes[0], environment.sample_sizes[-1]):
            means = {model: saved_predictions[f"{model}_n{count}_mean"][scenario_index * fields] for model in NEURAL_MODELS}
            render_example(output / f"example_{scenario}_n{count}.png", truth[scenario_index * fields], means,
                benchmark_observations["observation_indices"][0, :count], environment.grid_size, scenario, count)
    # Separate single-map timing: batched amortized costs are not online latency.
    timing_rows = []
    count = environment.sample_sizes[-1]
    map_index = 2 * fields  # nonstationary scenario, predefined field 0
    indices, values = flatten_observations(benchmark_observations, count)
    for model in NEURAL_MODELS:
        choice = selected[f"{model}_n{count}"]
        seeds = member_seeds(fields, config.members, 2020000)[map_index * config.members:(map_index + 1) * config.members]
        _, timings, _ = fit_ensemble_snapshots(features[model], indices[map_index:map_index + 1], values[map_index:map_index + 1],
            model, config, seeds, choice["weight_decay"], (choice["epochs"],))
        timing_rows.append({"model": model, "sample_size": count, "scenario": "nonstationary_mixture", "field": 0,
                           "single_map_fit_seconds": timings[choice["epochs"]]})
    began = time.perf_counter()
    fit_snapshot("estimated", points, indices[map_index], values[map_index], environment.noise_std, EstimatorConfig())
    timing_rows.append({"model": "estimated", "sample_size": count, "scenario": "nonstationary_mixture", "field": 0,
                       "single_map_fit_seconds": time.perf_counter() - began})
    write_csv(output / "single_map_timing.csv", timing_rows)
    (output / "compute.json").write_text(json.dumps(compute, indent=2) + "\n")
    diagnostics = {"benchmark_fields_per_scenario": fields, "benchmark_metric_rows": len(rows),
        "neural_map_fits": len(truth) * len(environment.sample_sizes) * len(NEURAL_MODELS),
        "independent_member_fits": len(member_rows), "parameters_per_member": counts_parameters,
        "tuning_maps_per_scenario": config.tuning_fields, "calibration_maps_per_scenario": config.calibration_fields,
        "intervals": "heuristically calibrated marginal ensemble bands; no joint covariance or posterior guarantee",
        "source_data_unchanged": all(hashlib.sha256((reference / name).read_bytes()).hexdigest() == digest
                                     for name, digest in source_hashes.items())}
    (output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    return summaries, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("outputs/sample-size-study"))
    parser.add_argument("--output", type=Path, default=Path("outputs/deepkriging-study"))
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        summaries, diagnostics = run_study(args.reference, args.output)
    for row in summaries:
        if row["model"] in NEURAL_MODELS:
            print(f"{row['scenario']:25s} n={row['sample_size']:3d} {row['model']:16s} "
                  f"RMSE={row['heldout_rmse']:.4f} calibrated coverage={row['heldout_coverage_95']:.3f}")
    print(json.dumps(diagnostics, indent=2))
    print("Artifacts:", args.output.resolve())


if __name__ == "__main__":
    main()
