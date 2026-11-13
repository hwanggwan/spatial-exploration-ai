"""Same-basis Bayesian linear control with independent tuning and new test maps."""

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
from .basis_bayes import BayesianBasisModel
from .deepkriging import (NeuralMapperConfig, calibration_multiplier,
                         fit_ensemble_snapshots, standardized_features)
from .estimated_gp import EstimatorConfig
from .resource_replay import ResourceConfig, acquired_states, design_routes, generate_fields, geometry
from .uncertainty_study import (collect_paths, curves, evaluate_paths, member_seeds,
                                save_split, summarize)


MODELS = ("estimated", "deepkriging", "basis_bayes")
NAMES = {"estimated": "Fitted RBF GP", "deepkriging": "DeepKriging-style", "basis_bayes": "Basis Bayes"}
COLORS = {"estimated": "#376aa3", "deepkriging": "#cf5656", "basis_bayes": "#25855b"}
NAMES.update({"empirical_gp": "Training-map GP", "cnp_seed0": "CNP seed 0", "cnp_seed1": "CNP seed 1"})
COLORS.update({"empirical_gp": "#25855b", "cnp_seed0": "#7954b3", "cnp_seed1": "#da8b20"})
NAMES.update({"attention_seed0": "Attention-CNP seed 0", "attention_seed1": "Attention-CNP seed 1"})
COLORS.update({"attention_seed0": "#047d91", "attention_seed1": "#be508f"})
SEEDS = {"tuning_fields": 5000000, "tuning_routes": 5010000, "tuning_members": 800000000,
         "calibration_fields": 5100000, "calibration_members": 810000000,
         "benchmark_fields": 5200000, "benchmark_members": 820000000}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def tune(output, config, neural, prior_candidates):
    """Same independent maps and endpoint observations for both candidate grids.

    One pooled held-out latent MSE selects each global setting. Duplicate final
    observation sets share fits, but all planned endpoint cases retain weight.
    """
    evaluator = generate_fields(config, neural.tuning_fields, SEEDS["tuning_fields"])
    episodes, prefixes = design_routes(config, neural.tuning_fields, SEEDS["tuning_routes"])
    final_ids = sorted({e["state_ids"][-1] for e in episodes})
    states = [prefixes[index] for index in final_ids]
    lookup = {old: new for new, old in enumerate(final_ids)}
    endpoint_ids = np.asarray([lookup[e["state_ids"][-1]] for e in episodes])
    observed = acquired_states(config, states, evaluator["potential_sensor_values"])
    np.savez_compressed(output / "tuning_acquired.npz", **observed)
    np.savez_compressed(output / "tuning_evaluation_only.npz", **evaluator)
    write_json(output / "tuning_routes.json", episodes)
    write_json(output / "tuning_endpoint_index.json", {"source_prefix_ids": final_ids,
        "endpoint_prediction_rows": endpoint_ids.tolist()})
    features, _, _ = standardized_features(observed["points"], "deepkriging", neural)
    mask = evaluator["evaluation_mask"]
    truth = evaluator["truth"][observed["fields"]][:, mask]
    rows, saved_predictions = [], {}

    def score(means):
        mse = ((means[:, mask] - truth)**2).mean(axis=1)
        return float(mse[endpoint_ids].mean())

    for prior in prior_candidates:
        began = time.perf_counter()
        model = BayesianBasisModel(features, config.noise_std, prior)
        means = np.asarray([model.condition(s["indices"], observed["values"][i, :observed["counts"][i]]).mean
                            for i, s in enumerate(states)])
        rows.append({"model": "basis_bayes", "prior_variance": prior, "heldout_mse": score(means),
                     "fit_seconds": time.perf_counter() - began})
        saved_predictions[f"basis_prior_{prior}"] = means
    for decay in neural.weight_decays:
        means = {epoch: np.zeros((len(states), len(features))) for epoch in neural.epochs}
        fit_seconds = {epoch: 0.0 for epoch in neural.epochs}
        for count in sorted(set(observed["counts"])):
            selected = np.flatnonzero(observed["counts"] == count)
            for start in range(0, len(selected), neural.chunk_maps):
                chunk = selected[start:start + neural.chunk_maps]
                seeds = [seed for i in chunk for seed in member_seeds(SEEDS["tuning_members"],
                         int(observed["fields"][i]), int(count), neural.members)]
                snapshots, seconds, _ = fit_ensemble_snapshots(features, observed["indices"][chunk, :count],
                    observed["values"][chunk, :count], "deepkriging", neural, seeds, decay, neural.epochs)
                for epoch in neural.epochs:
                    means[epoch][chunk] = snapshots[epoch]["mean"]
                    fit_seconds[epoch] += seconds[epoch]
        for epoch in neural.epochs:
            rows.append({"model": "deepkriging", "epochs": epoch, "weight_decay": decay,
                         "heldout_mse": score(means[epoch]), "fit_seconds": fit_seconds[epoch]})
            saved_predictions[f"neural_decay_{decay}_epochs_{epoch}"] = means[epoch]
    np.savez_compressed(output / "tuning_predictions.npz", **saved_predictions)
    best_basis = min((r for r in rows if r["model"] == "basis_bayes"), key=lambda r: r["heldout_mse"])
    best_neural = min((r for r in rows if r["model"] == "deepkriging"), key=lambda r: r["heldout_mse"])
    write_json(output / "tuning_results.json", {"candidates": rows, "endpoint_cases": len(episodes),
        "unique_observation_sets": len(states), "selected_basis": best_basis, "selected_neural": best_neural,
        "selection": "one global setting per model, pooled endpoint held-out latent MSE"})
    return best_basis["prior_variance"], replace(neural, epochs=(best_neural["epochs"],),
                                               weight_decays=(best_neural["weight_decay"],))


def render_results(output, episodes, states, evaluator, summaries, curve_rows, config, models=MODELS):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cap, budget = max(config.observation_caps), max(config.travel_budgets)
    points, _ = geometry(config)
    truth = evaluator["truth"][0]
    half = 0.5 / (config.grid_size - 1)
    fig, axes = plt.subplots(len(models), 3, figsize=(12, 3.6 * len(models)), constrained_layout=True)
    for row, model in enumerate(models):
        e = next(e for e in episodes if e["field"] == 0 and e["model"] == model
                 and e["observation_cap"] == cap and e["travel_budget"] == budget)
        state = states[e["state_ids"][-1]]
        route = points[e["route"]]
        for column, (label, values, cmap, low, high) in enumerate((
            ("Truth", truth, "viridis", truth.min(), truth.max()),
            ("Prediction", state["mean"], "viridis", truth.min(), truth.max()),
            ("Native latent std", np.sqrt(state["variance"]), "magma", 0, None))):
            axis = axes[row, column]
            plot = axis.imshow(values.reshape(config.grid_size, config.grid_size), origin="lower",
                extent=(-half, 1 + half, -half, 1 + half), cmap=cmap, vmin=low, vmax=high)
            axis.plot(route[:, 0], route[:, 1], "w.-", markersize=4, linewidth=1)
            axis.plot(*route[0], "wo", markersize=7, markeredgecolor="black")
            axis.plot(*route[-1], "wx", markersize=8)
            axis.set_title(f"{NAMES[model]}: {label}", fontsize=9)
            axis.set_xlabel("x"); axis.set_ylabel("y")
            fig.colorbar(plot, ax=axis, shrink=0.7)
        axes[row, 0].text(0.02, 0.98, f"n={len(route)}, distance={e['distances'][-1]:.2f}",
            transform=axes[row, 0].transAxes, va="top", color="white", fontsize=9,
            bbox={"facecolor": "black", "alpha": 0.5, "edgecolor": "none"})
    fig.suptitle(f"Own routes on predefined field 0; n <= {cap}, distance <= {budget:g}\n"
                 "Identical native variance / distance action rule", fontsize=11)
    fig.savefig(output / "example_routes.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(len(config.observation_caps), len(config.travel_budgets),
        figsize=(10, 7), squeeze=False, constrained_layout=True)
    for i, cap in enumerate(config.observation_caps):
        for j, budget in enumerate(config.travel_budgets):
            axis = axes[i, j]
            for model in models:
                rows = [r for r in curve_rows if r["observation_cap"] == cap and r["travel_budget"] == budget
                        and r["model"] == model and r["axis"] == "distance_allowance"]
                axis.plot([r["allowance"] for r in rows], [r["mean_rmse"] for r in rows],
                          color=COLORS[model], label=NAMES[model])
            axis.set_title(f"Caps: readings={cap}, distance={budget:g}")
            axis.set_xlabel("Distance along saved route"); axis.set_ylabel("Mean held-out RMSE")
            axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("New paired fields — saved prefixes of routes planned for the stated caps")
    fig.savefig(output / "error_curves.png", dpi=150); plt.close(fig)


def run_study(output, config=None, neural=None, estimator=None, prior_candidates=(0.05, 0.2, 1.0, 5.0)):
    config, neural, estimator = config or ResourceConfig(), neural or NeuralMapperConfig(), estimator or EstimatorConfig()
    for item in (config, neural, estimator):
        item.validate()
    if not prior_candidates or any(not np.isfinite(p) or p <= 0 for p in prior_candidates):
        raise ValueError("Use finite positive prior candidates")
    if len(set(prior_candidates)) != len(prior_candidates):
        raise ValueError("Prior candidates must be unique")
    if len(set(neural.weight_decays)) != len(neural.weight_decays):
        raise ValueError("Neural regularization candidates must be unique")
    if len(prior_candidates) != len(neural.epochs) * len(neural.weight_decays):
        raise ValueError("Use equal numbers of basis and neural candidate settings")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh output folder")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    design = {"resources": asdict(config), "neural_candidates": asdict(neural), "estimator": asdict(estimator),
        "basis_prior_candidates": prior_candidates, "models": MODELS, "seeds": SEEDS,
        "tuning": "shared random/fixed-GP-greedy route endpoints on independent tuning maps; global held-out MSE",
        "crossover_prespecified": True,
        "comparisons": "own closed loop plus all three frozen mappers on each final route; paired field bootstrap intervals",
        "hypothesis": "same spatial representation with analytic Bayesian linear inference separates representation from nonlinear fitting and ensemble uncertainty; benefit unknown"}
    write_json(output / "design.json", design)
    began = time.perf_counter()
    prior, frozen_neural = tune(output, config, neural, prior_candidates)
    tuning_wall = time.perf_counter() - began
    protocol = {**design, "neural": asdict(frozen_neural),
        "basis": {"selected_prior_variance": prior,
                  "features": "identical standardized xy + Wendland columns; append constant; divide all columns by sqrt(P)",
                  "prior": "beta ~ N(0, tau^2 I); latent K=tau^2 Psi Psi.T; zero prior mean",
                  "noise": "known Gaussian noise std, same as GP; no extra noise in reported latent uncertainty"},
        "acquisition": "native latent marginal variance / Euclidean distance for every model",
        "feasibility": "start site 0 counts; stop at measurement cap or no affordable unvisited permitted site; no return trip",
        "calibration": "global variance scalar per model from separate own-route calibration maps, unique prefix pooled; no action-ranking effect",
        "truth_access": "collector receives only sensor-on-arrival callback; mapper receives acquired sites/values only",
        "evaluation": "fixed index%5==4 sites withheld from sensing; fields paired across models and caps",
        "frozen_prior_routes": "Basis posterior covariance ignores observed values; fixed prior gives same route across fields within a budget condition",
        "timing": "per-prefix fit plus all-site prediction; NN independent fits batched and amortized; geometry/kernel preparation excluded from per-prefix timing",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__}}
    write_json(output / "protocol.json", protocol)
    calibration = generate_fields(config, config.calibration_fields, SEEDS["calibration_fields"])
    began = time.perf_counter()
    ce, cs, _ = collect_paths(config, config.calibration_fields,
        lambda f, s: calibration["potential_sensor_values"][f, s], frozen_neural, estimator,
        SEEDS["calibration_members"], basis_prior_variance=prior)
    calibration_wall = time.perf_counter() - began
    save_split(output, "calibration", ce, cs, calibration, config)
    scales = {}
    for model in MODELS:
        selected = [s for s in cs if s["model"] == model]
        scales[model] = calibration_multiplier(calibration["truth"][[s["field"] for s in selected]],
            np.asarray([s["mean"] for s in selected]), np.asarray([s["variance"] for s in selected]),
            calibration["evaluation_mask"])
    write_json(output / "frozen_protocol.json", {"variance_multipliers": scales,
        "neural": asdict(frozen_neural), "basis_prior_variance": prior, "frozen_before_benchmark_generation": True})
    evaluator = generate_fields(config, config.fields, SEEDS["benchmark_fields"])
    checkpoints = output / "checkpoints"; checkpoints.mkdir()
    began = time.perf_counter()
    episodes, states, checkpoint_index = collect_paths(config, config.fields,
        lambda f, s: evaluator["potential_sensor_values"][f, s], frozen_neural, estimator,
        SEEDS["benchmark_members"], checkpoint_folder=checkpoints, basis_prior_variance=prior)
    benchmark_wall = time.perf_counter() - began
    save_split(output, "benchmark", episodes, states, evaluator, config)
    write_json(output / "checkpoint_index.json", checkpoint_index)
    steps, endpoints = evaluate_paths(episodes, states, evaluator, config, estimator, scales)
    summaries, comparisons, route_comparisons = summarize(endpoints, episodes, config, MODELS)
    curve_rows = curves(steps, episodes, config, MODELS)
    write_csv(output / "step_metrics.csv", steps)
    write_csv(output / "final_metrics.csv", endpoints)
    write_csv(output / "curves.csv", curve_rows)
    for name, value in (("summary.json", summaries), ("paired_comparisons.json", comparisons),
                        ("route_comparisons.json", route_comparisons)):
        write_json(output / name, value)
    render_results(output, episodes, states, evaluator, summaries, curve_rows, config)
    mask = evaluator["evaluation_mask"]
    diagnostics = {"benchmark_episodes": len(episodes), "benchmark_unique_prefixes": len(states),
        "all_count_caps_respected": all(len(e["route"]) <= e["observation_cap"] for e in episodes),
        "all_distance_caps_respected": all(e["distances"][-1] <= e["travel_budget"] + 1e-12 for e in episodes),
        "evaluation_sites_never_acquired": not any(mask[e["route"]].any() for e in episodes),
        "frozen_before_benchmark_generation": True,
        "tuning_wall_seconds": tuning_wall, "calibration_loop_wall_seconds": calibration_wall,
        "benchmark_loop_wall_seconds": benchmark_wall,
        "unique_prefix_fit_seconds": {m: float(sum(s["fit_seconds"] for s in states if s["model"] == m)) for m in MODELS},
        "neural_member_fits": sum(frozen_neural.members for s in states if s["model"] == "deepkriging")}
    write_json(output / "diagnostics.json", diagnostics)
    write_json(output / "dataset_manifest.json", {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob("*.npz")})
    return summaries, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/basis-study"))
    args = parser.parse_args()
    from .crossover_study import run_study as run_crossover
    with threadpool_limits(limits=1):
        summaries, diagnostics = run_study(args.output)
        run_crossover(args.output, args.output.with_name(args.output.name + "-crossover"))
    for r in summaries:
        print(f"n<={r['observation_cap']} d<={r['travel_budget']} {r['model']}: RMSE={r['heldout_rmse']:.4f}, "
              f"n={r['measurements']:.2f}, d={r['travel_distance']:.3f}")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
