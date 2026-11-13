"""Mapper-driven GP and DeepKriging exploration under identical physical caps."""

import argparse
from dataclasses import asdict
import hashlib
import json
from itertools import combinations
from pathlib import Path
import platform
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .baseline import write_csv
from .basis_bayes import BayesianBasisModel
from .deepkriging import (NeuralMapperConfig, calibration_multiplier,
                         fit_ensemble_snapshots, standardized_features)
from .deepkriging_study import prediction_metrics
from .estimated_gp import EstimatorConfig
from .resource_replay import ResourceConfig, acquired_states, generate_fields, geometry
from .sample_size_study import bootstrap_mean, fit_snapshot
from .uncertainty_planner import choose_uncertainty_action


MODELS = ("estimated", "deepkriging")
COLORS = {"estimated": "#376aa3", "deepkriging": "#cf5656"}


def member_seeds(base, field, count, members):
    """Common deterministic initialization at a field/count across budget cases."""
    return [int(np.random.SeedSequence([base, field, count, member]).generate_state(
        1, dtype=np.uint64)[0]) % (2**63) for member in range(members)]


def collect_paths(config, fields, read_sensor, neural, estimator, seed_base, checkpoint_folder=None,
                  basis_prior_variance=None, extra_mappers=None):
    """No latent truth or true covariance accepted; sensor queried only on arrival.

    Every episode owns its route and measurements. Identical model/field/prefix
    states share numerical fits, but every episode is charged each visited fit.
    """
    points, mask = geometry(config)
    features, _, _ = standardized_features(points, "deepkriging", neural)
    extra_mappers = extra_mappers or {}
    if set(extra_mappers) & {"estimated", "deepkriging", "basis_bayes"}:
        raise ValueError("Extra mapper names must be distinct from built-in models")
    models = MODELS + (("basis_bayes",) if basis_prior_variance is not None else ()) + tuple(extra_mappers)
    basis = None if basis_prior_variance is None else BayesianBasisModel(features, config.noise_std, basis_prior_variance)
    episodes, states, cache = [], [], {}
    checkpoint_index = {}
    for field in range(fields):
        for cap in config.observation_caps:
            for budget in config.travel_budgets:
                for model in models:
                    episodes.append({"episode": len(episodes), "field": field, "model": model,
                        "observation_cap": int(cap), "travel_budget": float(budget),
                        "route": [0], "values": [float(read_sensor(field, 0))],
                        "distances": [0.0], "state_ids": [], "action_scores": [], "active": True})
    epoch, decay = neural.epochs[0], neural.weight_decays[0]
    while any(e["active"] for e in episodes):
        active = [e for e in episodes if e["active"]]
        pending = {}
        for episode in active:
            key = (episode["model"], episode["field"], tuple(episode["route"]))
            if key not in cache:
                pending.setdefault(key, episode)
        for key, episode in pending.items():
            model = episode["model"]
            if model == "deepkriging":
                continue
            began = time.perf_counter()
            if model in extra_mappers:
                prediction = extra_mappers[model](episode["route"], episode["values"])
                mean, variance = np.asarray(prediction["mean"]), np.asarray(prediction["variance"])
                if (mean.shape != (len(points),) or variance.shape != mean.shape
                        or not np.isfinite(mean).all() or not np.isfinite(variance).all() or np.any(variance < 0)):
                    raise ValueError("Extra mapper returned invalid latent predictions")
                cache[key] = len(states)
                states.append({"model": model, "field": episode["field"],
                    "indices": episode["route"].copy(), "values": episode["values"].copy(),
                    "mean": mean.copy(), "variance": variance.copy(),
                    "fit_seconds": time.perf_counter() - began})
                continue
            if model == "basis_bayes":
                belief = basis.condition(episode["route"], episode["values"])
                diagnostics = {"basis_prior_variance": basis_prior_variance}
            else:
                belief = fit_snapshot("estimated", points, np.asarray(episode["route"]),
                    np.asarray(episode["values"]), config.noise_std, estimator)
                diagnostics = {"length_scale": belief.length_scale, "process_variance": belief.process_variance,
                    "fit_calls": len(belief.fit_history),
                    "boundary_fit_calls": sum(event["boundary_hit"] for event in belief.fit_history)}
            cache[key] = len(states)
            states.append({"model": model, "field": episode["field"],
                "indices": episode["route"].copy(), "values": episode["values"].copy(),
                "mean": belief.mean.copy(), "variance": np.maximum(np.diag(belief.covariance), 0),
                "fit_seconds": time.perf_counter() - began,
                **diagnostics})
        neural_pending = [(key, e) for key, e in pending.items() if e["model"] == "deepkriging"]
        for count in sorted({len(e["route"]) for _, e in neural_pending}):
            chosen = [(key, e) for key, e in neural_pending if len(e["route"]) == count]
            for start in range(0, len(chosen), neural.chunk_maps):
                chunk = chosen[start:start + neural.chunk_maps]
                seeds = [seed for _, e in chunk for seed in member_seeds(seed_base, e["field"], count, neural.members)]
                path = None
                if checkpoint_folder and str(count) not in checkpoint_index:
                    path = checkpoint_folder / f"n{count}-first-chunk.pt"
                snapshots, seconds, _ = fit_ensemble_snapshots(features,
                    np.asarray([e["route"] for _, e in chunk]), np.asarray([e["values"] for _, e in chunk]),
                    "deepkriging", neural, seeds, decay, (epoch,), save_path=path)
                prediction = snapshots[epoch]
                chunk_ids = []
                for index, (key, episode) in enumerate(chunk):
                    cache[key] = len(states)
                    chunk_ids.append(len(states))
                    states.append({"model": "deepkriging", "field": episode["field"],
                        "indices": episode["route"].copy(), "values": episode["values"].copy(),
                        "mean": prediction["mean"][index], "variance": prediction["raw_variance"][index],
                        "members": prediction["members"][index], "fit_seconds": seconds[epoch] / len(chunk),
                        "seeds": member_seeds(seed_base, episode["field"], count, neural.members)})
                if path:
                    checkpoint_index[str(count)] = {"file": path.name, "state_ids": chunk_ids}
        for episode in active:
            key = (episode["model"], episode["field"], tuple(episode["route"]))
            state_id = cache[key]
            episode["state_ids"].append(state_id)
            remaining_count = episode["observation_cap"] - len(episode["route"])
            remaining_distance = max(episode["travel_budget"] - episode["distances"][-1], 0)
            visited = np.zeros(len(points), dtype=bool)
            visited[episode["route"]] = True
            current = episode["route"][-1]
            action = choose_uncertainty_action(points, states[state_id]["variance"], current,
                                               visited, ~mask, remaining_distance, remaining_count)
            if action is None:
                episode["active"] = False
                episode["stop_reason"] = "observation_cap" if remaining_count == 0 else "no_affordable_unvisited_site"
            else:
                cost = float(np.linalg.norm(points[action] - points[current]))
                episode["action_scores"].append(float(states[state_id]["variance"][action] / cost))
                episode["route"].append(action)
                episode["distances"].append(episode["distances"][-1] + cost)
                episode["values"].append(float(read_sensor(episode["field"], action)))
    return episodes, states, checkpoint_index


def save_split(output, name, episodes, states, evaluator, config):
    observed = acquired_states(config, states, evaluator["potential_sensor_values"])
    observed["models"] = np.asarray([s["model"] for s in states])
    # Check that the archived values are precisely those delivered through the sensor interface.
    for index, state in enumerate(states):
        np.testing.assert_array_equal(observed["values"][index, :len(state["values"])], state["values"])
    np.savez_compressed(output / f"{name}_acquired.npz", **observed)
    np.savez_compressed(output / f"{name}_evaluation_only.npz", **evaluator)
    np.savez_compressed(output / f"{name}_predictions.npz",
        mean=np.asarray([s["mean"] for s in states]), variance=np.asarray([s["variance"] for s in states]),
        fit_seconds=np.asarray([s["fit_seconds"] for s in states]),
        neural_state_ids=np.asarray([i for i, s in enumerate(states) if s["model"] == "deepkriging"]),
        neural_members=np.asarray([s["members"] for s in states if s["model"] == "deepkriging"]))
    (output / f"{name}_routes.json").write_text(json.dumps(episodes, indent=2) + "\n")


def evaluate_paths(episodes, states, evaluator, config, estimator, scales):
    """Evaluator-only scoring, run after all actions have been collected."""
    points, mask = geometry(config)
    scores = {}
    for state_id, state in enumerate(states):
        truth = evaluator["truth"][state["field"]]
        raw = prediction_metrics(truth, state["mean"], state["variance"], mask)
        calibrated = prediction_metrics(truth, state["mean"], state["variance"] * scales[state["model"]], mask)
        # Oracle reconstruction scores the observations that this route actually acquired.
        oracle = fit_snapshot("oracle", points, np.asarray(state["indices"]), np.asarray(state["values"]),
            config.noise_std, estimator, oracle_prior=evaluator["true_covariance"])
        oracle_metrics = prediction_metrics(truth, oracle.mean, np.diag(oracle.covariance), mask)
        scores[state_id] = {**raw, "calibrated_coverage_95": calibrated["heldout_coverage_95"],
            "calibrated_mean_std": calibrated["heldout_mean_std"],
            "oracle_rmse_on_route": oracle_metrics["heldout_rmse"],
            "oracle_heldout_variance_on_route": oracle_metrics["heldout_mean_variance"]}
    steps, endpoints = [], []
    for episode in episodes:
        common = {key: episode[key] for key in ("episode", "field", "model", "observation_cap", "travel_budget")}
        cumulative = 0.0
        for step, state_id in enumerate(episode["state_ids"]):
            cumulative += states[state_id]["fit_seconds"]
            row = {**common, "step": step, "state_id": state_id, "measurements": step + 1,
                "travel_distance": episode["distances"][step], **scores[state_id],
                "cumulative_fit_seconds": float(cumulative)}
            steps.append(row)
        endpoints.append({**row, "stop_reason": episode["stop_reason"]})
    return steps, endpoints


def summarize(rows, episodes, config, models=MODELS):
    summaries, comparisons, route_diagnostics = [], [], []
    episode_lookup = {(e["field"], e["observation_cap"], e["travel_budget"], e["model"]): e for e in episodes}
    for cap in config.observation_caps:
        for budget in config.travel_budgets:
            common = {"observation_cap": cap, "travel_budget": budget}
            by_model = {}
            for model in models:
                selected = sorted([r for r in rows if r["observation_cap"] == cap and r["travel_budget"] == budget
                                   and r["model"] == model], key=lambda r: r["field"])
                by_model[model] = selected
                summary = {**common, "model": model, "fields": len(selected)}
                for key in ("heldout_rmse", "heldout_coverage_95", "heldout_mean_std", "calibrated_coverage_95",
                    "calibrated_mean_std", "measurements", "travel_distance", "cumulative_fit_seconds",
                    "oracle_rmse_on_route", "oracle_heldout_variance_on_route"):
                    summary[key] = float(np.mean([r[key] for r in selected]))
                summary["rmse_bootstrap_95_interval"] = bootstrap_mean([r["heldout_rmse"] for r in selected])
                summaries.append(summary)
            for reference, model in combinations(models, 2):
                reference_values = np.asarray([r["heldout_rmse"] for r in by_model[reference]])
                model_values = np.asarray([r["heldout_rmse"] for r in by_model[model]])
                comparisons.append({**common, "model": model, "reference": reference,
                    "mean_paired_rmse_difference": float((model_values - reference_values).mean()),
                    "paired_bootstrap_95_interval": bootstrap_mean(model_values - reference_values),
                    "rmse_advantage_percent": float(100 * (reference_values.mean() - model_values.mean()) / reference_values.mean()),
                    "lower_rmse_fields": int(np.sum(model_values < reference_values))})
                equal, overlaps, common_prefix = 0, [], []
                for field in range(config.fields):
                    first = episode_lookup[field, cap, budget, reference]["route"]
                    second = episode_lookup[field, cap, budget, model]["route"]
                    equal += first == second
                    overlaps.append(len(set(first) & set(second)) / len(set(first) | set(second)))
                    length = 0
                    for a, b in zip(first, second):
                        if a != b:
                            break
                        length += 1
                    common_prefix.append(length)
                route_diagnostics.append({**common, "model": model, "reference": reference, "identical_routes": equal, "fields": config.fields,
                    "mean_visited_site_jaccard": float(np.mean(overlaps)), "mean_common_prefix_readings": float(np.mean(common_prefix))})
    return summaries, comparisons, route_diagnostics


def curves(steps, episodes, config, models=MODELS):
    lookup = {(r["episode"], r["step"]): r for r in steps}
    rows = []
    for cap in config.observation_caps:
        for budget in config.travel_budgets:
            for model in models:
                chosen = [e for e in episodes if e["observation_cap"] == cap and e["travel_budget"] == budget and e["model"] == model]
                for axis, values in (("measurement_allowance", range(1, cap + 1)), ("distance_allowance", np.linspace(0, budget, 81))):
                    for value in values:
                        selected = []
                        for episode in chosen:
                            step = min(int(value), len(episode["route"])) - 1 if axis == "measurement_allowance" else int(
                                np.searchsorted(episode["distances"], value + 1e-12, side="right") - 1)
                            selected.append(lookup[episode["episode"], step])
                        rows.append({"observation_cap": cap, "travel_budget": budget, "model": model,
                            "axis": axis, "allowance": float(value),
                            "mean_rmse": float(np.mean([r["heldout_rmse"] for r in selected])),
                            "mean_readings_available": float(np.mean([r["measurements"] for r in selected]))})
    return rows


def render_results(output, episodes, states, evaluator, summaries, curve_rows, config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cap, budget = max(config.observation_caps), max(config.travel_budgets)
    points, _ = geometry(config)
    truth = evaluator["truth"][0]
    half = 0.5 / (config.grid_size - 1)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for row, model in enumerate(MODELS):
        episode = next(e for e in episodes if e["field"] == 0 and e["model"] == model
                       and e["observation_cap"] == cap and e["travel_budget"] == budget)
        final = states[episode["state_ids"][-1]]
        panels = (("Truth (evaluator only)", truth, "viridis", truth.min(), truth.max()),
                  ("Own final prediction", final["mean"], "viridis", truth.min(), truth.max()),
                  ("Native latent std (used for variance score)", np.sqrt(final["variance"]), "magma", 0, None))
        for column, (label, values, cmap, low, high) in enumerate(panels):
            axis = axes[row, column]
            plot = axis.imshow(values.reshape(config.grid_size, config.grid_size), origin="lower",
                extent=(-half, 1 + half, -half, 1 + half), cmap=cmap, vmin=low, vmax=high)
            route = points[episode["route"]]
            axis.plot(route[:, 0], route[:, 1], "w.-", markersize=4, linewidth=1)
            axis.plot(*route[0], "wo", markersize=7, markeredgecolor="black")
            axis.plot(*route[-1], "wx", markersize=8)
            axis.set_title(f"{model}: {label}", fontsize=9)
            axis.set_xlabel("x")
            axis.set_ylabel("y")
            fig.colorbar(plot, ax=axis, shrink=0.7)
        axes[row, 0].text(0.02, 0.98, f"n={len(route)}, distance={episode['distances'][-1]:.2f}",
            transform=axes[row, 0].transAxes, va="top", color="white", fontsize=9,
            bbox={"facecolor": "black", "alpha": 0.5, "edgecolor": "none"})
    fig.suptitle(f"Own uncertainty-driven routes — predefined field 0; n <= {cap}, distance <= {budget:g}\n"
                 "Same variance/distance acquisition rule; models fit only their own acquired observations", fontsize=11)
    fig.savefig(output / "example_routes.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    for axis, model in zip(axes, MODELS):
        episode = next(e for e in episodes if e["field"] == 0 and e["model"] == model
                       and e["observation_cap"] == cap and e["travel_budget"] == budget)
        image = axis.imshow(truth.reshape(config.grid_size, config.grid_size), origin="lower",
            extent=(-half, 1 + half, -half, 1 + half), cmap="viridis", vmin=truth.min(), vmax=truth.max())
        route = points[episode["route"]]
        axis.plot(route[:, 0], route[:, 1], "w.-", markersize=5, linewidth=1.3)
        axis.plot(*route[0], "wo", markersize=8, markeredgecolor="black")
        axis.plot(*route[-1], "wx", markersize=9, markeredgewidth=2)
        label = "Fitted GP" if model == "estimated" else "DeepKriging-style"
        axis.set_title(f"{label}: {len(route)} readings, distance {episode['distances'][-1]:.2f}", fontsize=10)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
    fig.colorbar(image, ax=axes, shrink=0.8, label="Hidden field value (rendering only)")
    fig.suptitle(f"Own uncertainty-driven paths — field 0, caps n={cap}, distance={budget:g}\n"
                 "Circle: common start (0,0). Cross: finish. White dots: acquired measurements.", fontsize=11)
    fig.savefig(output / "paths.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for axis, name in zip(axes, ("measurement_allowance", "distance_allowance")):
        for model in MODELS:
            selected = [r for r in curve_rows if r["observation_cap"] == cap and r["travel_budget"] == budget
                        and r["model"] == model and r["axis"] == name]
            axis.plot([r["allowance"] for r in selected], [r["mean_rmse"] for r in selected], color=COLORS[model],
                label=model, drawstyle="steps-post" if name == "distance_allowance" else "default")
        axis.set_xlabel("Reading allowance" if name == "measurement_allowance" else "Distance allowance")
        axis.set_ylabel("Mean held-out latent RMSE")
        axis.grid(alpha=0.2)
        axis.legend()
    fig.suptitle(f"Closed-loop sequential error — caps n={cap}, distance={budget:g}; carry final estimate after stopping")
    fig.savefig(output / "sequential_curves.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(config.observation_caps), figsize=(10, 4), constrained_layout=True, squeeze=False)
    for column, cap in enumerate(config.observation_caps):
        for model in MODELS:
            selected = sorted([r for r in summaries if r["model"] == model and r["observation_cap"] == cap],
                              key=lambda r: r["travel_budget"])
            axes[0, column].plot([r["travel_budget"] for r in selected], [r["heldout_rmse"] for r in selected],
                                 "o-", color=COLORS[model], label=model)
        axes[0, column].set_title(f"At most {cap} readings")
        axes[0, column].set_xlabel("Travel-distance cap")
        axes[0, column].set_ylabel("Mean final held-out RMSE")
        axes[0, column].set_xticks(config.travel_budgets)
        axes[0, column].legend()
        axes[0, column].grid(alpha=0.2)
    fig.suptitle("Each mapper selects its own route — identical resource upper bounds")
    fig.savefig(output / "budget_comparison.png", dpi=150)
    plt.close(fig)


def run_study(output, config=None, neural=None, estimator=None):
    config = config or ResourceConfig()
    neural = neural or NeuralMapperConfig(epochs=(100,), weight_decays=(0.0001,))
    estimator = estimator or EstimatorConfig()
    for item in (config, neural, estimator):
        item.validate()
    if len(neural.epochs) != 1 or len(neural.weight_decays) != 1:
        raise ValueError("Use frozen neural settings, not a search on benchmark outcomes")
    output = Path(output)
    if (output / "protocol.json").exists():
        raise ValueError("Use a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    torch.set_num_threads(1)
    protocol = {"resources": asdict(config), "neural": asdict(neural), "estimator": asdict(estimator),
        "acquisition": "native latent marginal variance / Euclidean distance; same rule for both mappers",
        "feasibility": "unvisited permitted sites within remaining travel; stop at reading cap or no affordable site",
        "start": "site 0, (0,0); first reading counts, zero initial travel; no return-to-base cost",
        "update": "fresh fit from own acquired prefix after every reading; GP fits parameters from n>=4; neural fixed 100 epochs/decay 0.0001 by default",
        "scope": "mapper-dependent closed loop; no expected integrated variance reduction, learned planner, obstacles, or new hyperparameter search",
        "native_uncertainty": "GP plug-in posterior variance versus three-initialization neural sample variance, floor 1e-6",
        "calibration": "one positive global variance multiplier per model on separate closed-loop calibration maps; does not change within-state action ranking",
        "seeds": {"calibration_fields": 4000000, "benchmark_fields": 4100000,
                  "calibration_members": 100000000, "benchmark_members": 200000000},
        "member_seed_rule": "NumPy SeedSequence([base,field,count,member]), uint64 modulo 2**63; same initialization across budget cases",
        "truth_access": "collection receives only a read-on-arrival sensor callback; truth/covariance used only after paths are fixed",
        "evaluation": "45 fixed withheld sites; 64 independent benchmark fields, paired across model/resource conditions",
        "training_status": "fixed adapted DeepKriging representation and settings from earlier independent tuning; one ensemble construction per field/count",
        "timing": "GP single-prefix fit; neural independent batched fit plus grid prediction amortized per prefix; action selection included in aggregate loop wall time, no standalone latency benchmark",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__}, "cpu_threads": 1}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    calibration_evaluator = generate_fields(config, config.calibration_fields, 4000000)
    began = time.perf_counter()
    calibration_episodes, calibration_states, _ = collect_paths(config, config.calibration_fields,
        lambda field, site: calibration_evaluator["potential_sensor_values"][field, site], neural, estimator, 100000000)
    calibration_wall = time.perf_counter() - began
    save_split(output, "calibration", calibration_episodes, calibration_states, calibration_evaluator, config)
    scales = {}
    for model in MODELS:
        selected = [state for state in calibration_states if state["model"] == model]
        truth = calibration_evaluator["truth"][[state["field"] for state in selected]]
        scales[model] = calibration_multiplier(truth, np.asarray([s["mean"] for s in selected]),
            np.asarray([s["variance"] for s in selected]), calibration_evaluator["evaluation_mask"])
    (output / "frozen_protocol.json").write_text(json.dumps({"variance_multipliers": scales,
        "frozen_before_benchmark_generation": True, "hyperparameters": asdict(neural)}, indent=2) + "\n")
    evaluator = generate_fields(config, config.fields, 4100000)
    began = time.perf_counter()
    episodes, states, checkpoint_index = collect_paths(config, config.fields,
        lambda field, site: evaluator["potential_sensor_values"][field, site], neural, estimator, 200000000,
        checkpoint_folder=output / "checkpoints")
    benchmark_wall = time.perf_counter() - began
    save_split(output, "benchmark", episodes, states, evaluator, config)
    (output / "checkpoint_index.json").write_text(json.dumps(checkpoint_index, indent=2) + "\n")
    steps, endpoints = evaluate_paths(episodes, states, evaluator, config, estimator, scales)
    summaries, comparisons, route_diagnostics = summarize(endpoints, episodes, config)
    curve_rows = curves(steps, episodes, config)
    write_csv(output / "step_metrics.csv", steps)
    write_csv(output / "final_metrics.csv", endpoints)
    write_csv(output / "curves.csv", curve_rows)
    fit_rows = [{"state_id": index, "field": state["field"], "measurements": len(state["indices"]),
        **{key: state[key] for key in ("length_scale", "process_variance", "fit_calls", "boundary_fit_calls")}}
        for index, state in enumerate(states) if state["model"] == "estimated"]
    write_csv(output / "gp_fits.csv", fit_rows)
    for name, value in (("summary.json", summaries), ("paired_comparisons.json", comparisons),
                        ("route_comparisons.json", route_diagnostics)):
        (output / name).write_text(json.dumps(value, indent=2) + "\n")
    render_results(output, episodes, states, evaluator, summaries, curve_rows, config)
    mask = evaluator["evaluation_mask"]
    diagnostics = {"benchmark_episodes": len(episodes), "benchmark_unique_prefixes": len(states),
        "neural_member_fits": sum(neural.members for s in states if s["model"] == "deepkriging"),
        "all_count_caps_respected": all(len(e["route"]) <= e["observation_cap"] for e in episodes),
        "all_distance_caps_respected": all(e["distances"][-1] <= e["travel_budget"] + 1e-12 for e in episodes),
        "evaluation_sites_never_acquired": not any(mask[e["route"]].any() for e in episodes),
        "own_model_uncertainty_drives_actions": True, "frozen_before_benchmark_generation": True,
        "calibration_loop_wall_seconds": calibration_wall, "benchmark_loop_wall_seconds": benchmark_wall,
        "unique_prefix_fit_seconds": {model: float(sum(s["fit_seconds"] for s in states if s["model"] == model)) for model in MODELS}}
    (output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    (output / "dataset_manifest.json").write_text(json.dumps({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in output.glob("*.npz")}, indent=2) + "\n")
    return summaries, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/uncertainty-study"))
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        summaries, diagnostics = run_study(args.output)
    for row in summaries:
        print(f"n<={row['observation_cap']:2d} d<={row['travel_budget']:.1f} {row['model']:12s} "
              f"RMSE={row['heldout_rmse']:.4f}, used n={row['measurements']:.2f}, d={row['travel_distance']:.3f}")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
