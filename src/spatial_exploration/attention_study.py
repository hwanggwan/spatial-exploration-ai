"""Capacity-matched deterministic attention-CNP versus frozen pooled CNP."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .baseline import write_csv
from .basis_study import render_results, write_json
from .cnp_study import context_diagnostics, load_predictors, state_predictions, train_models
from .conditional_process import (AttentiveConditionalNeuralProcess, CNPConfig,
                                 ConditionalNeuralProcess, FrozenCNP)
from .deepkriging import NeuralMapperConfig, calibration_multiplier
from .estimated_gp import EstimatorConfig
from .deepkriging_study import prediction_metrics
from .sample_size_study import bootstrap_mean
from .resource_replay import ResourceConfig, generate_fields, geometry
from .uncertainty_study import collect_paths, curves, evaluate_paths, save_split, summarize


SEEDS = {"calibration_fields": 7200000, "benchmark_fields": 7300000,
         "calibration_members": 1020000000, "benchmark_members": 1030000000}


def fixed_step_reconstruction(reference, output, resources, cnp, attention):
    """Same GP-route labels, same last training step; no choices from outcomes."""
    step = max(cnp.checkpoints)
    episodes = [e for e in json.loads((output / "benchmark_routes.json").read_text()) if e["model"] == "estimated"]
    with np.load(output / "benchmark_acquired.npz") as stored:
        observed = {key: stored[key].copy() for key in stored.files if key != "models"}
    state_ids = sorted({e["state_ids"][-1] for e in episodes})
    observed = {key: value if key == "points" else value[state_ids] for key, value in observed.items()}
    lookup = {state: index for index, state in enumerate(state_ids)}
    predictions, hashes = {}, {}
    for seed in cnp.seeds:
        for kind, factory, config, folder in (("cnp", ConditionalNeuralProcess, cnp, reference),
                ("attention", AttentiveConditionalNeuralProcess, attention, output)):
            path = folder / "checkpoints" / f"{kind}-seed{seed}-step{step}.pt"
            with torch.random.fork_rng(devices=[]):
                network = factory(config)
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            network.load_state_dict(checkpoint["state_dict"])
            predictor = FrozenCNP(network, observed["points"])
            predictions[kind, seed] = state_predictions(predictor, observed)
            hashes[f"{kind}_seed{seed}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    np.savez_compressed(output / "fixed_step_predictions.npz", state_ids=state_ids,
        **{f"{kind}_seed{seed}_{key}": values for (kind, seed), result in predictions.items() for key, values in result.items()})
    # All predictions are fixed before scoring latent targets.
    with np.load(output / "benchmark_evaluation_only.npz") as stored:
        truth, mask = stored["truth"], stored["evaluation_mask"]
    rows = []
    for episode in episodes:
        index = lookup[episode["state_ids"][-1]]
        for seed in cnp.seeds:
            row = {key: episode[key] for key in ("field", "observation_cap", "travel_budget")}
            row.update({"training_seed": seed, "updates": step})
            for kind in ("cnp", "attention"):
                prediction = predictions[kind, seed]
                row[kind + "_rmse"] = prediction_metrics(truth[episode["field"]], prediction["mean"][index],
                    prediction["variance"][index], mask)["heldout_rmse"]
                row[kind + "_acquired_noisy_target_rmse"] = float(np.sqrt(np.mean(
                    (prediction["mean"][index, episode["route"]] - episode["values"])**2)))
                row[kind + "_context_mean_latent_std"] = float(np.sqrt(prediction["variance"][index, episode["route"]]).mean())
            rows.append(row)
    write_csv(output / "fixed_step_metrics.csv", rows)
    summaries = []
    for cap in resources.observation_caps:
        for budget in resources.travel_budgets:
            for seed in cnp.seeds:
                chosen = [r for r in rows if r["observation_cap"] == cap and r["travel_budget"] == budget and r["training_seed"] == seed]
                differences = np.asarray([r["attention_rmse"] - r["cnp_rmse"] for r in chosen])
                summaries.append({"observation_cap": cap, "travel_budget": budget, "training_seed": seed,
                    "updates": step, "pooled_cnp_rmse": float(np.mean([r["cnp_rmse"] for r in chosen])),
                    "attention_cnp_rmse": float(np.mean([r["attention_rmse"] for r in chosen])),
                    **{kind + "_" + metric: float(np.mean([r[kind + "_" + metric] for r in chosen]))
                       for kind in ("cnp", "attention") for metric in ("acquired_noisy_target_rmse", "context_mean_latent_std")},
                    "mean_paired_rmse_difference": float(differences.mean()),
                    "paired_field_bootstrap_95_interval": bootstrap_mean(differences)})
    write_json(output / "fixed_step_comparison.json", {"checkpoint_sha256": hashes, "summaries": summaries})
    return summaries


def add_attention_predictors(predictors, output, points, config, selected_step):
    for seed in config.seeds:
        with torch.random.fork_rng(devices=[]):
            network = AttentiveConditionalNeuralProcess(config)
        checkpoint = torch.load(output / "checkpoints" / f"attention-seed{seed}-step{selected_step}.pt",
                                map_location="cpu", weights_only=True)
        network.load_state_dict(checkpoint["state_dict"])
        predictors[f"attention_seed{seed}"] = FrozenCNP(network, points)
    return predictors


def render_attention(output, predictor, episode):
    """Predefined field/route, all available labels only; no selection by error."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = predictor.points
    size = int(round(np.sqrt(len(points))))
    queries = np.array([[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]], dtype=np.float32)
    context = points[episode["route"]]
    with torch.inference_mode():
        weights = predictor.network.attention_weights(torch.as_tensor(context)[None],
                                                       torch.as_tensor(queries)[None])[0].numpy()
    np.savez_compressed(output / "example_attention.npz", query_points=queries,
                        context_points=context, context_indices=episode["route"], weights=weights)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5), constrained_layout=True)
    for axis, query, row in zip(axes, queries, weights):
        axis.plot(context[:, 0], context[:, 1], color="#aaaaaa", linewidth=0.7, zorder=0)
        dots = axis.scatter(context[:, 0], context[:, 1], c=row, s=45 + 250 * row,
                            cmap="viridis", vmin=0, vmax=1)
        axis.plot(*query, "r*", markersize=13)
        axis.set(xlim=(-0.05, 1.05), ylim=(-0.05, 1.05), aspect="equal", xlabel="x", ylabel="y")
        axis.set_title(f"Query ({query[0]:g}, {query[1]:g})", fontsize=9)
    fig.colorbar(dots, ax=axes, shrink=0.65, label="Attention weight")
    fig.suptitle("Attention-CNP seed 0 — predefined field 0, final own route\n"
                 "Red star: query; dots: actual observations. Weights are not causal importance.", fontsize=10)
    fig.savefig(output / "example_attention.png", dpi=150)
    plt.close(fig)


def run_study(reference, output, crossover=True):
    reference, output = Path(reference), Path(output)
    if (reference.resolve() == output.resolve() or reference.resolve() in output.resolve().parents
            or output.resolve() in reference.resolve().parents):
        raise ValueError("Keep new results separate from the frozen CNP source")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh output folder")
    source = json.loads((reference / "protocol.json").read_text())
    resources, cnp = ResourceConfig(**source["resources"]), CNPConfig(**source["cnp"])
    neural, estimator = NeuralMapperConfig(**source["neural"]), EstimatorConfig(**source["estimator"])
    # Reallocate width to the position network rather than adding a larger model.
    # Default: 17,204 parameters, versus pooled CNP's 17,218 (0.081% difference).
    attention = replace(cnp, width=max(1, round(cnp.width * 15 / 16)))
    for config in (resources, cnp, attention, neural, estimator):
        config.validate()
    if tuple(cnp.seeds) != (0, 1):
        raise ValueError("Retain both source training seeds")
    cnp_step = source["selected_cnp_step"]
    source_names = ["protocol.json", "offline_training_observations.npz", "learned_prior_covariance.npy",
                    "validation_acquired.npz", "validation_evaluation_only.npz", "validation_routes.json",
                    "training_results.json"]
    source_names += [f"checkpoints/cnp-seed{s}-step{cnp_step}.pt" for s in cnp.seeds]
    source_names += [name for s in cnp.seeds
                    for name in [f"checkpoints/cnp-seed{s}-step{max(cnp.checkpoints)}.pt"] if name not in source_names]
    source_hashes = {name: hashlib.sha256((reference / name).read_bytes()).hexdigest() for name in source_names}
    manifest = json.loads((reference / "dataset_manifest.json").read_text())
    if any(source_hashes[name] != manifest[name] for name in source_hashes if name in manifest):
        raise RuntimeError("Frozen development data or source weights changed")
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir()
    torch.set_num_threads(1)
    points, mask = geometry(resources)
    with torch.random.fork_rng(devices=[]):
        pooled_parameters = sum(p.numel() for p in ConditionalNeuralProcess(cnp).parameters())
        attention_parameters = sum(p.numel() for p in AttentiveConditionalNeuralProcess(attention).parameters())
    models = ("estimated", "deepkriging", "empirical_gp", "cnp_seed0", "cnp_seed1",
              "attention_seed0", "attention_seed1")
    design = {"reference": str(reference.resolve()), "source_sha256": source_hashes,
        "resources": asdict(resources), "cnp": asdict(cnp), "attention": asdict(attention),
        "neural": asdict(neural), "estimator": asdict(estimator), "models": models, "seeds": SEEDS,
        "crossover_prespecified": True, "selected_cnp_step": cnp_step,
        "method": "deterministic learned-coordinate scaled dot-product cross-attention; no latent variational path or self-attention",
        "paper": "https://arxiv.org/abs/1901.05761",
        "parameter_counts": {"pooled_cnp": pooled_parameters, "attention_cnp": attention_parameters},
        "position_encoder": "shared 2 -> 22 -> representation with ReLU, normalized coordinates",
        "training_controls": "same archived noisy maps, sampler, optimizer, initialization seed rule, per-seed mini-batch RNG, three checkpoints, and maximum updates as pooled CNP",
        "selection": "one attention checkpoint by validation noisy NLL pooled over both seeds, independent of fixed CNP checkpoint selection; no winning-seed selection",
        "offline_data": "reuse training/validation data only; never load previous benchmark targets",
        "acquisition": "native latent marginal variance / Euclidean distance for all models; both physical caps enforced",
        "calibration": "new own-route calibration maps, global variance scalar per model; freeze before new test generation",
        "hypothesis": "query-dependent access to individual observations improves local conditioning and exploration; benefit unknown",
        "fixed_step_diagnostic": "both neural architectures at last common training checkpoint on final fitted-GP observations; predictions archived before scoring",
        "limits": "width changes slightly for parameter matching; same training steps do not equalize FLOPs; factorized marginals are not a coherent GP posterior"}
    write_json(output / "design.json", design)
    for name in source_names:
        if name == "protocol.json":
            continue
        target = output / ("pooled_cnp_training_results.json" if name == "training_results.json" else name)
        shutil.copyfile(reference / name, target)
    with np.load(output / "offline_training_observations.npz") as stored:
        training_sensor = stored["values"]
        np.testing.assert_array_equal(stored["points"], points)
    with np.load(output / "validation_acquired.npz") as stored:
        observed = {key: stored[key].copy() for key in stored.files}
    with np.load(output / "validation_evaluation_only.npz") as stored:
        validation_sensor = stored["potential_sensor_values"]
    ve = json.loads((output / "validation_routes.json").read_text())
    selected, training = train_models(output, resources, attention, training_sensor, validation_sensor,
        observed, ve, network_factory=AttentiveConditionalNeuralProcess, checkpoint_prefix="attention")
    covariance = np.load(output / "learned_prior_covariance.npy")
    predictors = load_predictors(output, points, cnp, cnp_step, covariance, resources.noise_std)
    add_attention_predictors(predictors, output, points, attention, selected)
    protocol = {**design, "selected_attention_step": selected, "offline_sensor_readings": int(training_sensor.size),
                "no_test_time_gradient_updates": True}
    write_json(output / "protocol.json", protocol)
    calibration = generate_fields(resources, resources.calibration_fields, SEEDS["calibration_fields"])
    began = time.perf_counter()
    ce, cs, _ = collect_paths(resources, resources.calibration_fields,
        lambda f, s: calibration["potential_sensor_values"][f, s], neural, estimator,
        SEEDS["calibration_members"], extra_mappers=predictors)
    calibration_wall = time.perf_counter() - began
    save_split(output, "calibration", ce, cs, calibration, resources)
    scales = {}
    for model in models:
        chosen = [s for s in cs if s["model"] == model]
        scales[model] = calibration_multiplier(calibration["truth"][[s["field"] for s in chosen]],
            np.asarray([s["mean"] for s in chosen]), np.asarray([s["variance"] for s in chosen]), mask)
    weight_files = [f"checkpoints/cnp-seed{s}-step{cnp_step}.pt" for s in cnp.seeds]
    weight_files += [f"checkpoints/attention-seed{s}-step{selected}.pt" for s in cnp.seeds]
    frozen_files = weight_files + ["learned_prior_covariance.npy"]
    frozen_hashes = {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in frozen_files}
    write_json(output / "frozen_protocol.json", {"variance_multipliers": scales,
        "selected_cnp_step": cnp_step, "selected_attention_step": selected,
        "frozen_file_sha256": frozen_hashes, "frozen_before_benchmark_generation": True})
    evaluator = generate_fields(resources, resources.fields, SEEDS["benchmark_fields"])
    began = time.perf_counter()
    episodes, states, _ = collect_paths(resources, resources.fields,
        lambda f, s: evaluator["potential_sensor_values"][f, s], neural, estimator,
        SEEDS["benchmark_members"], extra_mappers=predictors)
    benchmark_wall = time.perf_counter() - began
    save_split(output, "benchmark", episodes, states, evaluator, resources)
    steps, endpoints = evaluate_paths(episodes, states, evaluator, resources, estimator, scales)
    summaries, comparisons, route_comparisons = summarize(endpoints, episodes, resources, models)
    curve_rows = curves(steps, episodes, resources, models)
    for name, rows in (("step_metrics.csv", steps), ("final_metrics.csv", endpoints), ("curves.csv", curve_rows)):
        write_csv(output / name, rows)
    for name, value in (("summary.json", summaries), ("paired_comparisons.json", comparisons),
                        ("route_comparisons.json", route_comparisons)):
        write_json(output / name, value)
    write_json(output / "context_diagnostics.json", context_diagnostics(episodes, states, mask, resources, models))
    render_results(output, episodes, states, evaluator, summaries, curve_rows, resources, models)
    example = next(e for e in episodes if e["field"] == 0 and e["model"] == "attention_seed0"
        and e["observation_cap"] == max(resources.observation_caps) and e["travel_budget"] == max(resources.travel_budgets))
    render_attention(output, predictors["attention_seed0"], example)
    fixed_step_reconstruction(reference, output, resources, cnp, attention)
    write_json(output / "dataset_manifest.json", {**frozen_hashes,
        **{p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob("*.npz")}})
    diagnostics = {"benchmark_episodes": len(episodes), "benchmark_unique_prefixes": len(states),
        "all_count_caps_respected": all(len(e["route"]) <= e["observation_cap"] for e in episodes),
        "all_distance_caps_respected": all(e["distances"][-1] <= e["travel_budget"] + 1e-12 for e in episodes),
        "evaluation_sites_never_acquired": not any(mask[e["route"]].any() for e in episodes),
        "new_attention_training_wall_seconds": training["total_wall_seconds"],
        "source_cnp_training_wall_seconds": json.loads((reference / "training_results.json").read_text())["total_wall_seconds"],
        "calibration_loop_wall_seconds": calibration_wall, "benchmark_loop_wall_seconds": benchmark_wall,
        "unique_prefix_fit_seconds": {m: float(sum(s["fit_seconds"] for s in states if s["model"] == m)) for m in models},
        "source_unchanged": all(hashlib.sha256((reference / n).read_bytes()).hexdigest() == d for n, d in source_hashes.items()),
        "weights_and_prior_unchanged": all(hashlib.sha256((output / n).read_bytes()).hexdigest() == d for n, d in frozen_hashes.items()),
        "frozen_before_benchmark_generation": True}
    write_json(output / "diagnostics.json", diagnostics)
    if crossover:
        from .crossover_study import run_study as run_crossover
        run_crossover(output, output.with_name(output.name + "-crossover"), extra_mappers=predictors)
    return summaries, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("outputs/cnp-study"))
    parser.add_argument("--output", type=Path, default=Path("outputs/attention-study"))
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        summaries, diagnostics = run_study(args.reference, args.output)
    for r in summaries:
        print(f"n<={r['observation_cap']} d<={r['travel_budget']} {r['model']}: RMSE={r['heldout_rmse']:.4f}, "
              f"n={r['measurements']:.2f}, d={r['travel_distance']:.3f}")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
