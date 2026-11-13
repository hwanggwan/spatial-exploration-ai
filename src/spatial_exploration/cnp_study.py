"""Offline-trained context-conditioned mapping and sequential exploration."""

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .baseline import GPBelief, write_csv
from .basis_study import render_results, write_json
from .conditional_process import (CNPConfig, ConditionalNeuralProcess, FrozenCNP,
    learned_covariance, observation_nll, sample_training_batch)
from .deepkriging import NeuralMapperConfig, calibration_multiplier
from .estimated_gp import EstimatorConfig
from .resource_replay import ResourceConfig, acquired_states, design_routes, generate_fields, geometry
from .uncertainty_study import collect_paths, curves, evaluate_paths, save_split, summarize


SEEDS = {"training_fields": 6000000, "validation_fields": 6100000, "validation_routes": 6110000,
         "calibration_fields": 6200000, "benchmark_fields": 6300000,
         "calibration_members": 920000000, "benchmark_members": 930000000}


def state_predictions(predictor, observed):
    total, sites = len(observed["counts"]), len(observed["points"])
    predictions = {"mean": np.empty((total, sites)), "variance": np.empty((total, sites))}
    for count in np.unique(observed["counts"]):
        selected = np.flatnonzero(observed["counts"] == count)
        for start in range(0, len(selected), 64):
            chunk = selected[start:start + 64]
            result = predictor.predict_batch(observed["indices"][chunk, :count], observed["values"][chunk, :count])
            for key in predictions:
                predictions[key][chunk] = result[key]
    return predictions


def validation_loss(predictor, observed, sensor_maps, episodes, noise_std, mask):
    """Noisy held-out likelihood, equally weighting maps and route episodes.

    Repeated shared prefixes keep their episode/step weight after cached
    prediction. Only acquired values enter predictor; held-out labels score it.
    """
    predictions = state_predictions(predictor, observed)
    targets = sensor_maps[observed["fields"]][:, mask]
    variance = predictions["variance"][:, mask] + noise_std**2
    nll = 0.5 * (np.log(2 * np.pi * variance) + (targets - predictions["mean"][:, mask])**2 / variance)
    per_state = nll.mean(axis=1)
    return float(np.mean([per_state[e["state_ids"]].mean() for e in episodes]))


def train_models(output, resources, config, training_sensor, validation_sensor, observed, episodes,
                 network_factory=ConditionalNeuralProcess, checkpoint_prefix="cnp"):
    """Train only on offline noisy maps; validation selects a global step count."""
    points, mask = geometry(resources)
    tensor_points = torch.as_tensor(points, dtype=torch.float32)
    tensor_maps = torch.as_tensor(training_sensor, dtype=torch.float32)
    rows, trace = [], []
    began = time.perf_counter()
    for seed in config.seeds:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(18300 + seed)
            network = network_factory(config)
        optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
        rng = np.random.default_rng(940000000 + seed)
        seed_began = time.perf_counter()
        losses = []
        for step in range(1, max(config.checkpoints) + 1):
            fields, context, query = sample_training_batch(points, training_sensor, ~mask, config, rng)
            field_rows = torch.as_tensor(fields[:, None])
            context_tensor, query_tensor = torch.as_tensor(context), torch.as_tensor(query)
            mean, std = network(tensor_points[context_tensor], tensor_maps[field_rows, context_tensor],
                                tensor_points[query_tensor])
            loss = observation_nll(mean, std, tensor_maps[field_rows, query_tensor], resources.noise_std).mean()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Nonfinite CNP training likelihood")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            if step % 100 == 0 or step in config.checkpoints:
                trace.append({"seed": seed, "step": step, "mean_training_nll_recent": float(np.mean(losses[-100:]))})
            if step in config.checkpoints:
                predictor = FrozenCNP(copy.deepcopy(network), points)
                score = validation_loss(predictor, observed, validation_sensor, episodes, resources.noise_std, mask)
                row = {"seed": seed, "step": step, "validation_noisy_nll": score,
                       "elapsed_seed_seconds": time.perf_counter() - seed_began}
                rows.append(row)
                torch.save({"state_dict": network.state_dict(), "cnp": asdict(config), "seed": seed, "step": step,
                            "architecture": checkpoint_prefix},
                           output / "checkpoints" / f"{checkpoint_prefix}-seed{seed}-step{step}.pt")
                write_json(output / "training_progress.json", {"checkpoints": rows, "trace": trace})
                print(f"{checkpoint_prefix.upper()} seed {seed}, step {step}: validation noisy NLL={score:.4f}", flush=True)
    scores = {step: float(np.mean([r["validation_noisy_nll"] for r in rows if r["step"] == step]))
              for step in config.checkpoints}
    selected = min(config.checkpoints, key=lambda step: scores[step])
    result = {"selected_step": selected, "pooled_seed_validation_nll": scores,
        "checkpoint_scores": rows, "total_wall_seconds": time.perf_counter() - began,
        "selection": "one step count minimizing mean validation noisy NLL across all predeclared training seeds; retain every seed",
        "network_parameters": sum(p.numel() for p in network.parameters())}
    write_json(output / "training_results.json", result)
    write_csv(output / "training_trace.csv", trace)
    return selected, result


def load_predictors(output, points, config, step, covariance, noise_std):
    def empirical(indices, values):
        belief = GPBelief(covariance, noise_std)
        for index, value in zip(indices, values):
            belief.observe(index, value)
        return {"mean": belief.mean, "variance": np.maximum(np.diag(belief.covariance), 0)}
    predictors = {"empirical_gp": empirical}
    for seed in config.seeds:
        with torch.random.fork_rng(devices=[]):
            network = ConditionalNeuralProcess(config)
        path = output / "checkpoints" / f"cnp-seed{seed}-step{step}.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        network.load_state_dict(checkpoint["state_dict"])
        predictors[f"cnp_seed{seed}"] = FrozenCNP(network, points)
    return predictors


def context_diagnostics(episodes, states, mask, resources, models):
    """Descriptive conditioning checks; these do not choose model settings."""
    rows = []
    for cap in resources.observation_caps:
        for budget in resources.travel_budgets:
            for model in models:
                selected = [e for e in episodes if e["model"] == model
                            and e["observation_cap"] == cap and e["travel_budget"] == budget]
                errors, context_std, heldout_std, spatial_std = [], [], [], []
                for e in selected:
                    state = states[e["state_ids"][-1]]
                    mean, variance = state["mean"], state["variance"]
                    errors.append(np.sqrt(np.mean((mean[e["route"]] - e["values"])**2)))
                    context_std.append(np.sqrt(variance[e["route"]]).mean())
                    heldout_std.append(np.sqrt(variance[mask]).mean())
                    spatial_std.append(np.std(mean))
                rows.append({"observation_cap": cap, "travel_budget": budget, "model": model,
                    "mean_acquired_noisy_target_rmse": float(np.mean(errors)),
                    "mean_native_std_at_acquired_sites": float(np.mean(context_std)),
                    "mean_native_std_at_heldout_sites": float(np.mean(heldout_std)),
                    "mean_spatial_sd_of_prediction": float(np.mean(spatial_std)),
                    "status": "descriptive diagnostic; acquired noisy-target fit is not generalization accuracy"})
    return rows


def run_study(output, resources=None, cnp=None, neural=None, estimator=None, crossover=True):
    resources = resources or ResourceConfig(calibration_fields=16)
    cnp = cnp or CNPConfig()
    neural = neural or NeuralMapperConfig(epochs=(300,), weight_decays=(0.01,))
    estimator = estimator or EstimatorConfig()
    for setting in (resources, cnp, neural, estimator):
        setting.validate()
    if tuple(cnp.seeds) != (0, 1):
        raise ValueError("This study reports both predeclared training seeds 0 and 1")
    if len(neural.epochs) != 1 or len(neural.weight_decays) != 1:
        raise ValueError("DeepKriging comparator must use frozen settings")
    points, mask = geometry(resources)
    if (max(resources.observation_caps) > cnp.max_context or cnp.max_context > (~mask).sum()
            or cnp.targets > len(points) - cnp.max_context):
        raise ValueError("Training contexts must cover deployment caps and leave enough disjoint targets")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh output folder")
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir()
    torch.set_num_threads(1)
    models = ("estimated", "deepkriging", "empirical_gp", "cnp_seed0", "cnp_seed1")
    design = {"resources": asdict(resources), "cnp": asdict(cnp), "neural": asdict(neural),
        "estimator": asdict(estimator), "models": models, "seeds": SEEDS,
        "crossover_prespecified": True,
        "method": "deterministic CNP adaptation: mean-pool encoded observed pairs, append context count, decode latent Gaussian marginals",
        "paper": "https://proceedings.mlr.press/v80/garnelo18a.html",
        "training": "offline noisy maps only; context includes start 0, uniform count 1..16; equal global/local geometry mixture; disjoint noisy targets",
        "selection": "one checkpoint by validation noisy held-out NLL averaged over route prefixes and both retained seeds",
        "offline_control": "known-zero-mean covariance from same noisy maps; subtract known noise, PSD projection, fixed 5% identity shrinkage",
        "calibration": "separate own-route maps, one global marginal variance scalar per model; freeze before test generation",
        "acquisition": "same native latent marginal variance / Euclidean distance for every model",
        "resources_scope": "online caps include starting measurement; all offline full-map training observations separately reported",
        "truth_access": "offline trainer accepts noisy maps, not latent fields/covariance; deployment accepts acquired pairs only; latent truth evaluator-only",
        "distribution": "same gradual nonstationary simulator law across independent train/validation/calibration/test fields; no distribution-shift claim",
        "timing": "per-prefix fit or frozen-network forward plus grid prediction; neural training reported separately; amortized DK batching; CPU one thread",
        "scope": "adaptation, not exact original-paper reproduction; factorized neural marginals do not support exact integrated-information-gain updates",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__}}
    write_json(output / "design.json", design)
    # Training functions receive only noisy offline readings, never these latent targets.
    training_sensor = generate_fields(resources, cnp.training_fields, SEEDS["training_fields"])["potential_sensor_values"]
    np.savez_compressed(output / "offline_training_observations.npz", points=points, values=training_sensor)
    covariance = learned_covariance(training_sensor, resources.noise_std)
    np.save(output / "learned_prior_covariance.npy", covariance)
    validation = generate_fields(resources, cnp.validation_fields, SEEDS["validation_fields"])
    ve, vs = design_routes(resources, cnp.validation_fields, SEEDS["validation_routes"])
    observed = acquired_states(resources, vs, validation["potential_sensor_values"])
    np.savez_compressed(output / "validation_acquired.npz", **observed)
    np.savez_compressed(output / "validation_evaluation_only.npz", **validation)
    write_json(output / "validation_routes.json", ve)
    selected, training = train_models(output, resources, cnp, training_sensor,
        validation["potential_sensor_values"], observed, ve)
    predictors = load_predictors(output, points, cnp, selected, covariance, resources.noise_std)
    protocol = {**design, "selected_cnp_step": selected,
        "offline_sensor_readings": int(training_sensor.size), "no_test_time_gradient_updates_for_cnp": True,
        "neural_member_rule": "SeedSequence([base,field,count,member]), uint64 modulo 2**63"}
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
        selected_states = [s for s in cs if s["model"] == model]
        scales[model] = calibration_multiplier(calibration["truth"][[s["field"] for s in selected_states]],
            np.asarray([s["mean"] for s in selected_states]), np.asarray([s["variance"] for s in selected_states]), mask)
    checkpoint_hashes = {f"cnp_seed{s}": hashlib.sha256((output / "checkpoints" / f"cnp-seed{s}-step{selected}.pt").read_bytes()).hexdigest()
                         for s in cnp.seeds}
    write_json(output / "frozen_protocol.json", {"variance_multipliers": scales,
        "selected_step": selected, "checkpoint_sha256": checkpoint_hashes,
        "learned_covariance_sha256": hashlib.sha256((output / "learned_prior_covariance.npy").read_bytes()).hexdigest(),
        "frozen_before_benchmark_generation": True})
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
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob("*.npz")}
    manifest["learned_prior_covariance.npy"] = hashlib.sha256((output / "learned_prior_covariance.npy").read_bytes()).hexdigest()
    for seed in cnp.seeds:
        filename = f"checkpoints/cnp-seed{seed}-step{selected}.pt"
        manifest[filename] = checkpoint_hashes[f"cnp_seed{seed}"]
    write_json(output / "dataset_manifest.json", manifest)
    diagnostics = {"benchmark_episodes": len(episodes), "benchmark_unique_prefixes": len(states),
        "all_count_caps_respected": all(len(e["route"]) <= e["observation_cap"] for e in episodes),
        "all_distance_caps_respected": all(e["distances"][-1] <= e["travel_budget"] + 1e-12 for e in episodes),
        "evaluation_sites_never_acquired": not any(mask[e["route"]].any() for e in episodes),
        "calibration_loop_wall_seconds": calibration_wall, "benchmark_loop_wall_seconds": benchmark_wall,
        "offline_training_wall_seconds": training["total_wall_seconds"],
        "offline_sensor_readings": int(training_sensor.size),
        "unique_prefix_fit_seconds": {m: float(sum(s["fit_seconds"] for s in states if s["model"] == m)) for m in models},
        "cnp_weights_unchanged": all(hashlib.sha256((output / "checkpoints" / f"cnp-seed{s}-step{selected}.pt").read_bytes()).hexdigest()
            == checkpoint_hashes[f"cnp_seed{s}"] for s in cnp.seeds), "frozen_before_benchmark_generation": True}
    write_json(output / "diagnostics.json", diagnostics)
    if crossover:
        from .crossover_study import run_study as run_crossover
        run_crossover(output, output.with_name(output.name + "-crossover"), extra_mappers=predictors)
    return summaries, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/cnp-study"))
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        summaries, diagnostics = run_study(args.output)
    for r in summaries:
        print(f"n<={r['observation_cap']} d<={r['travel_budget']} {r['model']}: RMSE={r['heldout_rmse']:.4f}, "
              f"n={r['measurements']:.2f}, d={r['travel_distance']:.3f}")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
