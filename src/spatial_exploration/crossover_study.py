"""Post-hoc frozen-route crossover: sampling design versus mapper reconstruction."""

import argparse
import csv
import hashlib
import json
from itertools import combinations
from pathlib import Path
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .baseline import write_csv
from .basis_bayes import BayesianBasisModel
from .deepkriging import NeuralMapperConfig, fit_ensemble_snapshots, standardized_features
from .deepkriging_study import prediction_metrics
from .estimated_gp import EstimatorConfig
from .resource_replay import ResourceConfig
from .sample_size_study import bootstrap_mean, fit_snapshot
from .uncertainty_study import MODELS, member_seeds


def summarize(rows, config, models=MODELS):
    summaries, comparisons, decompositions = [], [], []
    for cap in config.observation_caps:
        for budget in config.travel_budgets:
            common = {"observation_cap": cap, "travel_budget": budget}
            lookup = {(r["field"], r["route_model"], r["mapper"]): r["heldout_rmse"] for r in rows
                      if r["observation_cap"] == cap and r["travel_budget"] == budget}
            arrays = {(route, mapper): np.asarray([lookup[field, route, mapper] for field in range(config.fields)])
                      for route in models for mapper in models}
            for (route, mapper), values in arrays.items():
                summaries.append({**common, "route_model": route, "mapper": mapper,
                    "mean_rmse": float(values.mean()), "field_bootstrap_95_interval": bootstrap_mean(values)})
            for reference, candidate in combinations(models, 2):
                for kind, conditions in (("mapper_change_on_fixed_route", models),
                                          ("route_change_with_fixed_mapper", models)):
                    for condition in conditions:
                        difference = (arrays[condition, candidate] - arrays[condition, reference]
                            if kind == "mapper_change_on_fixed_route" else
                            arrays[candidate, condition] - arrays[reference, condition])
                        comparisons.append({**common, "comparison": kind, "condition": condition,
                            "model": candidate, "reference": reference,
                            "mean_paired_rmse_difference": float(difference.mean()),
                            "paired_bootstrap_95_interval": bootstrap_mean(difference)})
            for candidate in models[1:]:
                gg = arrays["estimated", "estimated"]
                gd = arrays["estimated", candidate]
                dg = arrays[candidate, "estimated"]
                dd = arrays[candidate, candidate]
                # Symmetric descriptive allocation, not causal attribution.
                sampling = 0.5 * ((dg - gg) + (dd - gd))
                reconstruction = 0.5 * ((gd - gg) + (dd - dg))
                total = dd - gg
                np.testing.assert_allclose(sampling + reconstruction, total, atol=1e-12)
                row = {**common, "model": candidate, "reference": "estimated",
                       "status": "descriptive; sampling includes locations and count",
                       "mean_total_rmse_difference": float(total.mean())}
                for key, values in (("sampling_design", sampling), ("reconstruction_mapper", reconstruction),
                                    ("interaction", (dd - dg) - (gd - gg))):
                    row[key + "_mean_difference"] = float(values.mean())
                    row[key + "_paired_bootstrap_95_interval"] = bootstrap_mean(values)
                decompositions.append(row)
    return summaries, comparisons, decompositions


def render_results(path, summaries, config, models=MODELS):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(config.observation_caps), len(config.travel_budgets),
                             figsize=(13, 10) if len(models) > 3 else (11, 8.5),
                             constrained_layout=True, squeeze=False)
    all_values = [r["mean_rmse"] for r in summaries]
    for row, cap in enumerate(config.observation_caps):
        for column, budget in enumerate(config.travel_budgets):
            matrix = np.asarray([[next(r["mean_rmse"] for r in summaries if r["observation_cap"] == cap
                and r["travel_budget"] == budget and r["route_model"] == route and r["mapper"] == mapper)
                for mapper in models] for route in models])
            axis = axes[row, column]
            plot = axis.imshow(matrix, cmap="YlOrRd", vmin=min(all_values)-0.01, vmax=max(all_values)+0.01)
            for i in range(len(models)):
                for j in range(len(models)):
                    axis.text(j, i, f"{matrix[i,j]:.4f}", ha="center", va="center", fontsize=9 if len(models) > 3 else 12,
                              color="white" if matrix[i,j] > np.mean(all_values) else "black")
            names = {"estimated": "GP", "deepkriging": "DK", "basis_bayes": "Basis Bayes",
                     "empirical_gp": "Prior GP", "cnp_seed0": "CNP 0", "cnp_seed1": "CNP 1",
                     "attention_seed0": "Attn 0", "attention_seed1": "Attn 1"}
            axis.set_xticks(range(len(models)), [names[m] for m in models], fontsize=8)
            axis.set_yticks(range(len(models)), [names[m].replace("Basis Bayes", "Basis") + " route" for m in models], fontsize=8)
            axis.set_title(f"Caps: readings={cap}, distance={budget:g}", fontsize=10)
    fig.colorbar(plot, ax=axes, shrink=0.7, label="Mean held-out latent RMSE (lower is better)")
    fig.suptitle("Frozen-route crossover — same observed labels within each row\n"
                 "Frozen models; no route changes or hyperparameter selection", fontsize=11)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_study(reference, output, extra_mappers=None):
    reference, output = Path(reference), Path(output)
    if (reference.resolve() == output.resolve() or reference.resolve() in output.resolve().parents
            or output.resolve() in reference.resolve().parents):
        raise ValueError("Keep diagnostic outputs separate from the source archive")
    if (output / "protocol.json").exists():
        raise ValueError("Use a fresh output folder")
    protocol = json.loads((reference / "protocol.json").read_text())
    models = tuple(protocol.get("models", MODELS))
    extra_mappers = extra_mappers or {}
    if (not set(models) <= {"estimated", "deepkriging", "basis_bayes"} | set(extra_mappers)
            or models[:2] != MODELS or len(set(models)) != len(models)):
        raise ValueError("Supply prediction callbacks for every additional frozen model")
    config, neural, estimator = (ResourceConfig(**protocol["resources"]), NeuralMapperConfig(**protocol["neural"]),
                                EstimatorConfig(**protocol["estimator"]))
    for setting in (config, neural, estimator):
        setting.validate()
    manifest = json.loads((reference / "dataset_manifest.json").read_text())
    for filename, digest in manifest.items():
        if hashlib.sha256((reference / filename).read_bytes()).hexdigest() != digest:
            raise RuntimeError("Source archive changed")
    extra_files = ("benchmark_routes.json", "protocol.json", "final_metrics.csv")
    source_hashes = {**manifest, **{name: hashlib.sha256((reference / name).read_bytes()).hexdigest() for name in extra_files}}
    episodes = json.loads((reference / "benchmark_routes.json").read_text())
    with np.load(reference / "benchmark_acquired.npz") as stored:
        acquired = {key: stored[key].copy() for key in stored.files}
    with np.load(reference / "benchmark_predictions.npz") as stored:
        original_means, original_variances = stored["mean"].copy(), stored["variance"].copy()
    output.mkdir(parents=True, exist_ok=True)
    (output / "protocol.json").write_text(json.dumps({"status": (
        "prespecified frozen-route diagnostic on the source benchmark" if protocol.get("crossover_prespecified")
        else "post-hoc diagnosis on exposed fields; not a new test"),
        "reference": str(reference.resolve()), "source_sha256": source_hashes,
        "comparison": "all frozen mappers on each route family's final acquired observations",
        "controls": "same architecture, epochs, regularization, member seed rule and GP fitting settings as source",
        "interpretation": "sampling includes actual count and location differences; decomposition is descriptive, not causal attribution",
        "models": models, "basis": protocol.get("basis"),
        "resources": protocol["resources"], "neural": protocol["neural"], "estimator": protocol["estimator"]}, indent=2) + "\n")
    torch.set_num_threads(1)
    features, _, _ = standardized_features(acquired["points"], "deepkriging", neural)
    basis = (BayesianBasisModel(features, config.noise_std, protocol["basis"]["selected_prior_variance"])
             if "basis_bayes" in models else None)
    original_lookup = {(str(acquired["models"][state]), int(acquired["fields"][state]),
        tuple(acquired["indices"][state, :count])): state for state, count in enumerate(acquired["counts"])}
    requests = {}
    for episode in episodes:
        state = episode["state_ids"][-1]
        indices, values = episode["route"], episode["values"]
        for mapper in models:
            requests.setdefault((mapper, episode["field"], tuple(indices)), {"indices": indices, "values": values,
                "field": episode["field"], "mapper": mapper, "source_state": state})
    cache, new_fits = {}, {model: 0 for model in models}
    elapsed = {model: 0.0 for model in models}
    for key in requests:
        if key in original_lookup:
            state = original_lookup[key]
            cache[key] = {"mean": original_means[state], "variance": original_variances[state], "reused": True}
    for key, request in requests.items():
        model = request["mapper"]
        if key in cache or model == "deepkriging":
            continue
        began = time.perf_counter()
        if model in extra_mappers:
            prediction = extra_mappers[model](request["indices"], request["values"])
            cache[key] = {"mean": np.asarray(prediction["mean"]).copy(),
                          "variance": np.asarray(prediction["variance"]).copy(), "reused": False}
        else:
            belief = (basis.condition(request["indices"], request["values"]) if model == "basis_bayes" else
                fit_snapshot("estimated", acquired["points"], np.asarray(request["indices"]),
                             np.asarray(request["values"]), config.noise_std, estimator))
            cache[key] = {"mean": belief.mean.copy(), "variance": np.diag(belief.covariance).copy(), "reused": False}
        elapsed[model] += time.perf_counter() - began
        new_fits[model] += 1
    for count in sorted({len(r["indices"]) for key, r in requests.items() if key not in cache}):
        selected = [(key, r) for key, r in requests.items() if key not in cache and len(r["indices"]) == count]
        for start in range(0, len(selected), neural.chunk_maps):
            chunk = selected[start:start + neural.chunk_maps]
            seeds = [seed for _, r in chunk for seed in member_seeds(protocol["seeds"]["benchmark_members"],
                r["field"], count, neural.members)]
            result, seconds, _ = fit_ensemble_snapshots(features,
                np.asarray([r["indices"] for _, r in chunk]), np.asarray([r["values"] for _, r in chunk]),
                "deepkriging", neural, seeds, neural.weight_decays[0], neural.epochs)
            elapsed["deepkriging"] += seconds[neural.epochs[0]]
            new_fits["deepkriging"] += len(chunk)
            for index, (key, _) in enumerate(chunk):
                cache[key] = {"mean": result[neural.epochs[0]]["mean"][index],
                    "variance": result[neural.epochs[0]]["raw_variance"][index], "reused": False}
    # Write all fixed predictions before loading benchmark evaluator targets.
    keys = list(requests)
    np.savez_compressed(output / "predictions.npz", mean=np.asarray([cache[key]["mean"] for key in keys]),
        variance=np.asarray([cache[key]["variance"] for key in keys]))
    (output / "prediction_index.json").write_text(json.dumps([{**requests[key], "prediction_id": i,
        "reused_original_fit": cache[key]["reused"]} for i, key in enumerate(keys)], indent=2) + "\n")
    with np.load(reference / "benchmark_evaluation_only.npz") as stored:
        truth, mask = stored["truth"], stored["evaluation_mask"]
    prediction_ids = {key: i for i, key in enumerate(keys)}
    rows = []
    for episode in episodes:
        common = {key: episode[key] for key in ("episode", "field", "observation_cap", "travel_budget")}
        for mapper in models:
            key = (mapper, episode["field"], tuple(episode["route"]))
            prediction = cache[key]
            metrics = prediction_metrics(truth[episode["field"]], prediction["mean"], prediction["variance"], mask)
            rows.append({**common, "route_model": episode["model"], "mapper": mapper,
                "measurements": len(episode["route"]), "travel_distance": episode["distances"][-1],
                "prediction_id": prediction_ids[key], **metrics})
    with (reference / "final_metrics.csv").open() as stream:
        native = {int(row["episode"]): float(row["heldout_rmse"]) for row in csv.DictReader(stream)}
    native_errors = [abs(row["heldout_rmse"] - native[row["episode"]]) for row in rows if row["mapper"] == row["route_model"]]
    if max(native_errors) > 1e-12:
        raise RuntimeError("Native diagonal no longer matches the frozen source result")
    summaries, comparisons, decompositions = summarize(rows, config, models)
    write_csv(output / "metrics.csv", rows)
    for name, value in (("summary.json", summaries), ("paired_comparisons.json", comparisons),
                        ("descriptive_decomposition.json", decompositions)):
        (output / name).write_text(json.dumps(value, indent=2) + "\n")
    render_results(output / "crossover.png", summaries, config, models)
    diagnostics = {"endpoint_evaluations": len(rows), "unique_requested_fits": len(requests),
        "new_fits": new_fits, "new_fit_seconds": elapsed, "max_native_diagonal_error": max(native_errors),
        "source_unchanged": all(hashlib.sha256((reference / name).read_bytes()).hexdigest() == digest for name, digest in source_hashes.items()),
        "routes_or_hyperparameters_changed": False}
    (output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    return summaries, decompositions, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("outputs/uncertainty-study"))
    parser.add_argument("--output", type=Path, default=Path("outputs/uncertainty-crossover"))
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        summaries, decompositions, diagnostics = run_study(args.reference, args.output)
    for row in summaries:
        print(f"n<={row['observation_cap']} d<={row['travel_budget']} {row['route_model']} route, {row['mapper']} mapper: {row['mean_rmse']:.4f}")
    print(json.dumps(decompositions, indent=2))
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
