"""Frozen exploration policies under oracle, wrong, and fitted GP covariance."""

import argparse
from dataclasses import asdict
from importlib import metadata
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import sklearn
import torch
from threadpoolctl import threadpool_limits

from .baseline import GPBelief, evaluate, make_grid, rbf_covariance, render, write_csv
from .belief_explorer import ExplorationBeliefState, select_action
from .estimated_gp import EstimatedGPBelief, EstimatorConfig
from .exploration_env import ExplorationConfig
from .imitation import ClonedPolicy
from .imitation_study import assert_disjoint_cases, make_cases
from .lookahead_study import read_rows


CONDITIONS = ("oracle", "fixed_initial", "estimated")
LABELS = {"oracle": "Known covariance", "fixed_initial": "Wrong fixed covariance", "estimated": "Fitted covariance"}


def make_mapper(condition, points, noise_std, estimator, oracle_prior=None, oracle_length_scale=None):
    if condition == "estimated":
        return EstimatedGPBelief(points, noise_std, estimator), estimator.initial_length_scale, estimator.initial_process_variance
    if condition == "fixed_initial":
        return GPBelief(estimator.initial_process_variance * rbf_covariance(points,
            estimator.initial_length_scale), noise_std), estimator.initial_length_scale, estimator.initial_process_variance
    if condition == "oracle" and oracle_prior is not None and oracle_length_scale is not None:
        return GPBelief(oracle_prior, noise_std), oracle_length_scale, 1.0
    raise ValueError("Unknown condition or missing explicit oracle reference")


def sample_case(config, case):
    points = make_grid(config.grid_size)
    prior = rbf_covariance(points, case["length_scale"])
    rng = np.random.default_rng(case["field_seed"])
    truth = np.linalg.cholesky(prior + 1e-10 * np.eye(len(points))) @ rng.standard_normal(len(points))
    sensor = truth + config.noise_std * rng.standard_normal(len(points))
    return prior, truth, sensor


def execute_case(config, estimator, case, condition, method, policy, truth, sensor, true_prior):
    """Evaluator owns truth; select_action receives only a modeled belief state."""
    points = make_grid(config.grid_size)
    if condition == "oracle":
        mapper, length, variance = make_mapper(condition, points, config.noise_std, estimator,
            oracle_prior=true_prior, oracle_length_scale=case["length_scale"])
    else:
        # Neither true parameters, true prior, nor unacquired values enter construction.
        mapper, length, variance = make_mapper(condition, points, config.noise_std, estimator)
    start = case["start_index"]
    began = time.perf_counter()
    mapper.observe(start, sensor[start])
    update_seconds = time.perf_counter() - began
    state = ExplorationBeliefState(config, mapper, start, case["budget"], length, variance)
    oracle = GPBelief(true_prior, config.noise_std)
    oracle.observe(start, sensor[start])
    evaluation_mask = np.arange(len(points)) % 5 == 4
    traces, inputs, plans = [], [], []
    planning_seconds, variance_increases = 0.0, 0

    def measurements():
        return {"model_reported_mean_variance": float(np.trace(mapper.covariance) / len(points)),
            "model_heldout_mean_variance": float(np.mean(np.diag(mapper.covariance)[evaluation_mask])),
            "oracle_route_mean_variance": float(np.trace(oracle.covariance) / len(points)),
            "oracle_route_heldout_mean_variance": float(np.mean(np.diag(oracle.covariance)[evaluation_mask])),
            "oracle_route_heldout_rmse": evaluate(truth, oracle, evaluation_mask)["heldout_rmse"],
            **evaluate(truth, mapper, evaluation_mask)}

    def record():
        traces.append({"scenario": case["scenario"], "condition": condition, "method": method,
            "measurements": len(state.route), "index": state.current,
            "x": float(points[state.current, 0]), "y": float(points[state.current, 1]),
            "observed_value": float(sensor[state.current]), "travel_distance": state.travel_distance,
            "model_length_scale": state.length_scale, "model_process_variance": state.process_variance,
            **measurements()})

    record()
    while not state.finished:
        inputs.append({"scenario": case["scenario"], "condition": condition, "method": method,
            "decision": len(state.route) - 1, "current": state.current,
            "remaining_budget": state.budget - state.travel_distance, **state.input_diagnostics()})
        began = time.perf_counter()
        action, plan = select_action(method, policy, state)
        planning_seconds += time.perf_counter() - began
        if plan is not None and method == "rollout":
            if plan.final_mean_variance > plan.greedy_final_mean_variance + 1e-10:
                raise RuntimeError("Rollout completion is worse under the current fixed plug-in model")
            plans.append({"scenario": case["scenario"], "condition": condition,
                "decision": len(state.route) - 1, "first": plan.first,
                "hypothetical_final_model_variance": plan.final_mean_variance,
                "hypothetical_greedy_final_model_variance": plan.greedy_final_mean_variance,
                "route": " ".join(map(str, plan.route)), "candidates_evaluated": plan.candidates_evaluated,
                "simulated_measurements": plan.simulated_measurements})
        previous = float(np.trace(mapper.covariance) / len(points))
        began = time.perf_counter()
        state.move_and_observe(action, sensor[action])
        update_seconds += time.perf_counter() - began
        oracle.observe(action, sensor[action])
        variance_increases += float(np.trace(mapper.covariance) / len(points)) > previous + 1e-10
        record()
    if state.travel_distance > state.budget + 1e-12 or len(state.route) != len(set(state.route)):
        raise RuntimeError("Invalid executed route")
    if not state.allowed[state.route].all():
        raise RuntimeError("A reserved site was measured")
    if condition == "oracle":
        np.testing.assert_allclose(mapper.mean, oracle.mean, atol=1e-10)
        np.testing.assert_allclose(mapper.covariance, oracle.covariance, atol=1e-10)
    fits = [{"scenario": case["scenario"], "condition": condition, "method": method, **event}
            for event in getattr(mapper, "fit_history", [])]
    final_measurements = measurements()
    metrics = {**case, "condition": condition, "method": method, **final_measurements,
        "heldout_mse": final_measurements["heldout_rmse"]**2,
        "travel_distance": state.travel_distance, "measurements": len(state.route),
        "planning_seconds": planning_seconds, "update_seconds": update_seconds,
        "planning_and_update_seconds": planning_seconds + update_seconds,
        "model_length_scale": state.length_scale, "model_process_variance": state.process_variance,
        "fit_calls": len(fits), "boundary_fit_calls": sum(event["boundary_hit"] for event in fits),
        "rejected_fit_calls": sum(not event["accepted"] for event in fits),
        "fit_seconds": float(getattr(mapper, "total_fit_seconds", 0.0)),
        "modeled_variance_increase_steps": int(variance_increases), "decision_count": len(inputs),
        **{key + "_decisions": sum(row[key] for row in inputs) for key in
            ("length_outside_training_range", "posterior_diagonal_above_training_cap", "observation_outside_original_box")},
        "route": " ".join(map(str, state.route))}
    return metrics, traces, inputs, fits, plans, (mapper, state.route, state.cumulative_distances)


def summarize(rows):
    result = {}
    keys = ("heldout_rmse", "heldout_mse", "heldout_coverage_95", "heldout_mean_std",
        "oracle_route_mean_variance", "oracle_route_heldout_mean_variance", "oracle_route_heldout_rmse",
        "model_reported_mean_variance", "model_heldout_mean_variance", "measurements", "travel_distance",
        "planning_seconds", "update_seconds", "planning_and_update_seconds", "fit_calls", "boundary_fit_calls",
        "rejected_fit_calls", "modeled_variance_increase_steps", "model_length_scale", "model_process_variance")
    for condition in CONDITIONS:
        result[condition] = {}
        for method in dict.fromkeys(row["method"] for row in rows):
            selected = [row for row in rows if row["condition"] == condition and row["method"] == method]
            if not selected:
                continue
            decisions = sum(row["decision_count"] for row in selected)
            result[condition][method] = {"cases": len(selected),
                **{key: float(np.mean([row[key] for row in selected])) for key in keys},
                "decision_count": decisions,
                **{key + "_fraction": sum(row[key + "_decisions"] for row in selected) / decisions
                   if decisions else 0.0 for key in ("length_outside_training_range",
                       "posterior_diagonal_above_training_cap", "observation_outside_original_box")}}
    return result


def paired_effect(rows, reference_condition, reference_method, condition, method):
    reference = {row["scenario"]: row for row in rows
                 if row["condition"] == reference_condition and row["method"] == reference_method}
    selected = [row for row in rows if row["condition"] == condition and row["method"] == method]
    result = {"reference_condition": reference_condition, "reference_method": reference_method,
              "condition": condition, "method": method, "cases": len(selected)}
    for key in ("heldout_rmse", "oracle_route_mean_variance"):
        differences = np.asarray([row[key] - reference[row["scenario"]][key] for row in selected])
        rng = np.random.default_rng(960000)
        means = differences[rng.integers(0, len(differences), size=(5000, len(differences)))].mean(axis=1)
        result[key] = {"mean_paired_difference": float(differences.mean()),
            "paired_case_bootstrap_95_percent_interval": np.quantile(means, [0.025, 0.975]).tolist(),
            "lower_count": int(np.sum(differences < -1e-10)),
            "higher_count": int(np.sum(differences > 1e-10)),
            "tied_count": int(np.sum(np.abs(differences) <= 1e-10))}
    return result


def render_summary(path, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = list(summary["oracle"])
    colors = ["#b5762b", "#27754d", "#d3585a", "#885c9b"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for axis, metric, title in zip(axes, ("heldout_rmse", "heldout_coverage_95", "oracle_route_mean_variance"),
        ("Actual prediction error (lower is better)", "Latent 95% interval coverage", "Route value under the true GP (lower is better)")):
        for index, method in enumerate(methods):
            axis.plot(range(3), [summary[condition][method][metric] for condition in CONDITIONS],
                "o-", label=method, color=colors[index % len(colors)], linewidth=1.5)
        axis.set_xticks(range(3), ["Known", "Wrong fixed", "Fitted"])
        axis.set_xlabel("Covariance available to the planner and mapper")
        axis.set_title(title, fontsize=10)
        axis.grid(alpha=0.2)
    axes[1].axhline(0.95, color="black", linestyle="--", linewidth=1, label="Nominal 95%")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend(fontsize=8)
    axes[0].legend(fontsize=8)
    fig.suptitle("Frozen policies — identical new cases and budgets; hidden true GP used only for evaluation")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_study(reference, output, count=96, estimator=None):
    if count < 1:
        raise ValueError("At least one case is required")
    estimator = estimator or EstimatorConfig()
    estimator.validate()
    reference, output = Path(reference), Path(output)
    source, destination = reference.resolve(), output.resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Keep outputs separate from archived imitation files")
    if (output / "config.json").exists():
        raise ValueError("Use a fresh output folder to preserve this study")
    archived = json.loads((reference / "config.json").read_text())
    config = ExplorationConfig(**archived["environment"])
    config.validate()
    torch.set_num_threads(1)
    cases = make_cases(config, count, 940000, 1400000)
    assert_disjoint_cases([cases] + [json.loads((reference / f"{split}_cases.json").read_text())
                                    for split in ("train", "validation", "test")])
    frozen = json.loads((reference / "frozen_models.json").read_text())["sha256"]
    models, fingerprints = {}, {}
    for seed in archived["training"]["seeds"]:
        path = reference / f"rollout-seed-{seed}" / "selected.pt"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != frozen[str(path.resolve())]:
            raise RuntimeError("Archived selected checkpoint changed")
        models[f"bc_rollout_seed_{seed}"] = ClonedPolicy.load(path)
        fingerprints[str(path.resolve())] = digest
    methods = {"greedy": None, "rollout": None, **models}
    output.mkdir(parents=True, exist_ok=True)
    settings = {"environment": asdict(config), "estimator": asdict(estimator),
        "conditions": CONDITIONS, "case_count": count, "scenario_seed": 940000, "field_seed_start": 1400000,
        "source_run": str(reference.resolve()), "source_model_sha256": fingerprints,
        "models": list(methods), "training": "none; frozen selected rollout-cloning checkpoints; both seeds",
        "model_selection": "no threshold, clipping, tuning, or adaptation selected using these test outcomes",
        "neural_input": "current mapper covariance and length scale; original training scales; no clipping",
        "training_parameter_support": {"length_scale": config.length_scale_range, "process_variance": 1.0},
        "known_to_estimator": ["zero mean", "RBF family", "sensor noise standard deviation", "grid and physical budget"],
        "evaluation": "actual held-out latent RMSE/coverage and true-GP route variance; truth never passed to policy",
        "rollout_tail": "freeze current plug-in kernel during each hypothetical completion; future refits not anticipated",
        "guarantee_limit": "policy improvement under a fixed modeled kernel does not guarantee true-GP improvement or improvement across refits",
        "timing": "cumulative planning and mapper updates separate; excludes true-GP evaluation and plotting",
        "python_version": platform.python_version(), "numpy_version": np.__version__, "sklearn_version": sklearn.__version__,
        "cpu_threads": 1, "bootstrap": "5000 paired-case resamples; seed 960000; conditional on frozen models"}
    (output / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
    (output / "evaluation_cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    packages = sorted(f"{item.metadata['Name']}=={item.version}" for item in metadata.distributions())
    (output / "requirements-run.txt").write_text("\n".join(packages) + "\n")
    # A predefined archived case checks the new belief-state evaluator against
    # the original known-covariance Gym workflow before any new-case comparison.
    compatibility_case = json.loads((reference / "test_cases.json").read_text())[0]
    old = {row["method"]: row for row in read_rows(reference / "test" / "evaluation_metrics.csv")
           if row["scenario"] == str(compatibility_case["scenario"])}
    prior, truth, sensor = sample_case(config, compatibility_case)
    for method, policy in methods.items():
        row = execute_case(config, estimator, compatibility_case, "oracle", method, policy, truth, sensor, prior)[0]
        for key, reference_key in (("heldout_rmse", "heldout_rmse"), ("heldout_coverage_95", "heldout_coverage_95"),
            ("oracle_route_mean_variance", "final_mean_variance"), ("travel_distance", "travel_distance"), ("measurements", "measurements")):
            if not np.isclose(row[key], float(old[method][reference_key]), atol=1e-12, rtol=1e-10):
                raise RuntimeError(f"Oracle compatibility failed for {method}, {key}")
    (output / "compatibility_check.json").write_text(json.dumps({"scenario": compatibility_case,
        "methods": list(methods), "all_archived_oracle_outcomes_reproduced": True}, indent=2) + "\n")
    rows, traces, inputs, fits, plans = [], [], [], [], []
    for index, case in enumerate(cases):
        prior, truth, sensor = sample_case(config, case)
        for condition in CONDITIONS:
            examples = {}
            for method, policy in methods.items():
                row, trace, decisions, fit, plan, example = execute_case(config, estimator,
                    case, condition, method, policy, truth, sensor, prior)
                rows.append(row)
                traces.extend(trace)
                inputs.extend(decisions)
                fits.extend(fit)
                plans.extend(plan)
                if index == 0:
                    examples[method] = example
            if index == 0:
                render(output / f"routes_{condition}.png", truth, examples, config.grid_size,
                    title=f"Scenario 0 — {LABELS[condition]}; same field/start/budget; both frozen neural seeds")
        if (index + 1) % 16 == 0 or index + 1 == len(cases):
            print(f"Completed robustness case {index + 1}/{len(cases)}", flush=True)
    summary = summarize(rows)
    comparisons = []
    for condition in CONDITIONS:
        for method in methods:
            if method != "greedy":
                comparisons.append(paired_effect(rows, condition, "greedy", condition, method))
    for method in methods:
        for condition in CONDITIONS[1:]:
            comparisons.append(paired_effect(rows, "oracle", method, condition, method))
        comparisons.append(paired_effect(rows, "fixed_initial", method, "estimated", method))
    for filename, values in (("evaluation_metrics.csv", rows), ("trajectories.csv", traces),
                             ("policy_inputs.csv", inputs), ("parameter_fits.csv", fits), ("rollout_plans.csv", plans)):
        if values:
            write_csv(output / filename, values)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "paired_comparisons.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    if any(hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest for path, digest in fingerprints.items()):
        raise RuntimeError("Frozen weights changed during evaluation")
    diagnostics = {"episodes": len(rows), "all_frozen_checkpoints_unchanged": True,
        "all_routes_feasible_unique_and_permitted": True,
        "maximum_budget_excess": max(0.0, max(row["travel_distance"] - row["budget"] for row in rows)),
        "fit_calls": len(fits), "boundary_fit_calls": sum(event["boundary_hit"] for event in fits),
        "rejected_fit_calls": sum(not event["accepted"] for event in fits),
        "modeled_variance_increase_steps": sum(row["modeled_variance_increase_steps"] for row in rows),
        "policy_decisions": len(inputs), "rollout_decisions": len(plans)}
    (output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    render_summary(output / "comparison.png", summary)
    return summary, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("outputs/imitation-study"))
    parser.add_argument("--output", type=Path, default=Path("outputs/robustness-study"))
    parser.add_argument("--cases", type=int, default=96)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        summary, diagnostics = run_study(args.reference, args.output, args.cases)
    for condition, methods in summary.items():
        print(condition)
        for method, row in methods.items():
            print(f"  {method:20s} RMSE={row['heldout_rmse']:.4f} "
                  f"coverage={row['heldout_coverage_95']:.3f} true_route_V={row['oracle_route_mean_variance']:.4f}")
    print(json.dumps(diagnostics, indent=2))
    print("Artifacts:", args.output.resolve())


if __name__ == "__main__":
    main()
