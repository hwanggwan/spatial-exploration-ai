"""Validate greedy-completion rollout on archived small and larger problems."""

import argparse
from dataclasses import asdict, fields
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
from sb3_contrib import MaskablePPO
import torch

from .baseline import write_csv
from .exact_study import ExactStudyConfig, run_study
from .exploration_env import ExplorationConfig
from .lookahead_study import read_rows
from .rl_study import evaluate_methods


def verify_reproduction(rows, archived, identity_keys, metric_keys):
    previous = {tuple(row[key] for key in identity_keys): row for row in archived}
    current = {tuple(str(row[key]) for key in identity_keys): row for row in rows}
    for identity, reference in previous.items():
        if identity not in current:
            # The optional untrained PPO checkpoints are not part of this run.
            if "untrained" in str(identity):
                continue
            raise RuntimeError(f"Archived result missing: {identity}")
        for key in metric_keys:
            if not np.isclose(float(current[identity][key]), float(reference[key]), rtol=1e-10, atol=1e-12):
                raise RuntimeError(f"Archived outcome changed: {identity}, {key}")


def paired_comparison(rows, reference_method):
    reference = {int(row["scenario"]): row for row in rows if row["method"] == reference_method}
    paired = []
    for row in rows:
        if row["method"] != "rollout":
            continue
        other = reference[int(row["scenario"])]
        paired.append({"scenario": int(row["scenario"]), "reference_method": reference_method,
            "reference_variance": float(other["final_mean_variance"]),
            "rollout_variance": float(row["final_mean_variance"]),
            "variance_difference": float(row["final_mean_variance"]) - float(other["final_mean_variance"]),
            "rmse_difference": float(row["heldout_rmse"]) - float(other["heldout_rmse"])})
    deltas = [row["variance_difference"] for row in paired]
    mean_reference = np.mean([row["reference_variance"] for row in paired])
    comparison = {"scenarios": len(paired),
        "lower_variance_count": int(sum(value < -1e-10 for value in deltas)),
        "higher_variance_count": int(sum(value > 1e-10 for value in deltas)),
        "tied_variance_count": int(sum(abs(value) <= 1e-10 for value in deltas)),
        "mean_paired_variance_difference": float(np.mean(deltas)),
        "relative_reduction_of_mean_variance_percent": float(-100 * np.mean(deltas) / mean_reference)}
    return paired, comparison


def render_comparison(path, paired):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for axis, name in zip(axes, ("greedy", "lookahead_2")):
        selected = [row for row in paired if row["reference_method"] == name]
        values = [row["variance_difference"] for row in selected]
        axis.bar([row["scenario"] for row in selected], values,
            color=["#27754d" if value < -1e-10 else "#b45b4b" if value > 1e-10 else "gray" for value in values])
        axis.axhline(0, color="black", linewidth=0.8)
        axis.axhline(np.mean(values), color="#376aa3", linestyle="--", label="Mean paired difference")
        axis.set_title(f"Rollout minus {name}: final posterior variance", fontsize=10)
        axis.set_xlabel("Archived development scenario index")
        axis.set_ylabel("Negative favors rollout")
        axis.legend(fontsize=8)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Greedy-completion rollout — identical GP, starts, fields, and budget caps")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_comparison(reference, exact_reference, lookahead_reference, output):
    reference, exact_reference = Path(reference), Path(exact_reference)
    lookahead_reference, output = Path(lookahead_reference), Path(output)
    for source in (reference, exact_reference, lookahead_reference):
        source_path, destination = source.resolve(), output.resolve()
        if source_path == destination or source_path in destination.parents or destination in source_path.parents:
            raise ValueError("Use an output folder separate from all archived input runs")
    metadata = json.loads((reference / "config.json").read_text())
    exact_metadata = json.loads((exact_reference / "config.json").read_text())
    cases = json.loads((reference / "evaluation_scenarios.json").read_text())
    if cases != json.loads((lookahead_reference / "evaluation_scenarios.json").read_text()):
        raise ValueError("Archived lookahead and PPO scenarios differ")
    config = ExplorationConfig(**metadata["environment"])
    config.validate()
    exact_config = ExactStudyConfig(**{field.name: exact_metadata[field.name] for field in fields(ExactStudyConfig)})
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(metadata.get("torch_threads", 1))
    models, fingerprints = {}, {}
    for seed in metadata["training_seeds"]:
        path = reference / f"seed-{seed}" / "trained.zip"
        fingerprints[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        models[f"ppo_seed_{seed}"] = MaskablePPO.load(path, device="cpu")
    settings = {"environment": asdict(config), "small_environment": asdict(exact_config),
        "source_runs": [str(path.resolve()) for path in (reference, exact_reference, lookahead_reference)],
        "source_model_sha256": fingerprints, "python_version": platform.python_version(),
        "numpy_version": np.__version__, "torch_threads": torch.get_num_threads(),
        "planner": "all feasible first actions followed by deterministic greedy completion to terminal",
        "base_policy": "whole-grid variance reduction divided by travel distance; lowest-index argmax ties",
        "objective": "minimum final whole-grid mean latent posterior variance",
        "tie_rule": "keep greedy completion unless another decreases variance by more than 1e-12",
        "execution": "execute only the first action, then recompute all completions",
        "search_limits": "none; no first-action shortlist or completion-depth cutoff",
        "training": "none; reuse archived final PPO models",
        "data_status": "reused development cases, including the failure motivating rollout; not a held-out test",
        "limits": "known fixed GP covariance; no field values in planner; no global optimality guarantee"}
    (output / "config.json").write_text(json.dumps(settings, indent=2) + "\n")

    small = output / "exact"
    exact_summary = run_study(exact_config, small, include_rollout=True)
    small_rows = read_rows(small / "metrics.csv")
    if json.loads((small / "cases.json").read_text()) != json.loads((exact_reference / "cases.json").read_text()):
        raise RuntimeError("Small reference cases changed")
    verify_reproduction(small_rows, read_rows(exact_reference / "metrics.csv"), ("method", "case"),
        ("final_mean_variance", "travel_distance", "measurements", "excess_variance_percent"))
    verify_reproduction(read_rows(small / "search_certificates.csv"),
        read_rows(exact_reference / "search_certificates.csv"), ("case",),
        ("prefixes_evaluated", "terminal_routes", "maximum_additional_measurements"))
    small_greedy = {row["case"]: row for row in small_rows if row["method"] == "greedy"}
    for row in small_rows:
        if row["method"] == "rollout" and float(row["final_mean_variance"]) > float(
                small_greedy[row["case"]]["final_mean_variance"]) + 1e-10:
            raise RuntimeError("Rollout is worse than its base policy on a small problem")

    large = output / "grid7"
    large.mkdir(exist_ok=True)
    (large / "evaluation_scenarios.json").write_text(json.dumps(cases, indent=2) + "\n")
    summary = evaluate_methods(config, cases, models, large, include_lookahead=True, include_rollout=True)
    rows = read_rows(large / "evaluation_metrics.csv")
    outcome_keys = ("final_mean_variance", "heldout_rmse", "heldout_coverage_95", "information_gain",
                    "travel_distance", "measurements", "total_reward")
    for source in (reference, lookahead_reference):
        verify_reproduction(rows, read_rows(source / "evaluation_metrics.csv"),
                            ("method", "scenario"), outcome_keys)
    paired, comparisons = [], {}
    for name in ("greedy", "lookahead_2", *models):
        values, comparisons[name] = paired_comparison(rows, name)
        paired.extend(values)
    if comparisons["greedy"]["higher_variance_count"]:
        raise RuntimeError("Rollout performed worse than its base policy")
    if max(float(row["travel_distance"]) - float(row["budget"]) for row in rows) > 1e-12:
        raise RuntimeError("Travel budget exceeded")
    for row in rows:
        if not np.isclose(float(row["total_reward"]), config.reward_scale * float(row["information_gain"]),
                          atol=1e-9, rtol=0):
            raise RuntimeError("Reward no longer equals final variance reduction")
    final_by_case = {row["scenario"]: float(row["final_mean_variance"]) for row in rows if row["method"] == "rollout"}
    plans = read_rows(large / "rollout_plans.csv")
    for plan in plans:
        if float(plan["final_mean_variance"]) > float(plan["greedy_final_mean_variance"]) + 1e-10:
            raise RuntimeError("Selected completion is worse than greedy completion")
        if final_by_case[plan["scenario"]] > float(plan["final_mean_variance"]) + 1e-10:
            raise RuntimeError("Repeated replanning worsened a previously planned completion")
    diagnostics = {"decisions": len(plans),
        "maximum_first_candidates": max(int(plan["candidates_evaluated"]) for plan in plans),
        "total_simulated_measurements": sum(int(plan["simulated_measurements"]) for plan in plans),
        "all_plans_no_worse_than_base": True,
        "all_executed_outcomes_no_worse_than_prior_completions": True}
    comparison = {"archived_outcomes_reproduced": True, "small": exact_summary,
        "grid7": summary, "paired_grid7": comparisons, "rollout_diagnostics": diagnostics,
        "maximum_budget_excess": max(0.0, max(float(row["travel_distance"]) - float(row["budget"]) for row in rows))}
    write_csv(large / "paired_comparison.csv", paired)
    render_comparison(large / "paired_comparison.png", paired)
    (output / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("outputs/rl-study"))
    parser.add_argument("--exact-reference", type=Path, default=Path("outputs/exact-study"))
    parser.add_argument("--lookahead-reference", type=Path, default=Path("outputs/lookahead-study"))
    parser.add_argument("--output", type=Path, default=Path("outputs/rollout-study"))
    args = parser.parse_args()
    comparison = run_comparison(args.reference, args.exact_reference, args.lookahead_reference, args.output)
    for name, row in comparison["grid7"].items():
        print(f"{name:16s} variance={row['final_mean_variance']:.5f} "
              f"RMSE={row['heldout_rmse']:.5f} planning={row['planning_seconds']:.6f}s")
    print(json.dumps(comparison["paired_grid7"], indent=2))
    print("Artifacts:", args.output.resolve())


if __name__ == "__main__":
    main()
