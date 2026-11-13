"""Compare two-step lookahead with archived PPO and the original greedy rule."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO
import torch

from .exploration_env import ExplorationConfig
from .rl_study import evaluate_methods


def read_rows(path):
    import csv
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def compare_paired(rows):
    greedy = {int(row["scenario"]): row for row in rows if row["method"] == "greedy"}
    paired = []
    for row in rows:
        if row["method"] != "lookahead_2":
            continue
        reference = greedy[int(row["scenario"])]
        paired.append({"scenario": int(row["scenario"]),
            "variance_difference": float(row["final_mean_variance"]) - float(reference["final_mean_variance"]),
            "rmse_difference": float(row["heldout_rmse"]) - float(reference["heldout_rmse"]),
            "greedy_variance": float(reference["final_mean_variance"]),
            "lookahead_variance": float(row["final_mean_variance"])})
    return paired


def render_paired(path, paired):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis, key, title in zip(axes, ("variance_difference", "rmse_difference"),
        ("Primary: final posterior variance difference", "Secondary: held-out RMSE difference")):
        values = [row[key] for row in paired]
        axis.bar([row["scenario"] for row in paired], values,
                 color=["#27754d" if value < 0 else "#b45b4b" for value in values])
        axis.axhline(0, color="black", linewidth=0.8)
        axis.axhline(np.mean(values), color="#34549d", linestyle="--", label="Mean paired difference")
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("Predefined scenario index")
        axis.set_ylabel("Lookahead minus greedy; negative favors lookahead")
        axis.legend(fontsize=8)
    fig.suptitle("Two-step lookahead versus greedy — same scenarios and travel budgets")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_comparison(reference, output):
    reference, output = Path(reference), Path(output)
    if reference.resolve() == output.resolve():
        raise ValueError("Use a separate output folder to preserve the archived RL run")
    metadata = json.loads((reference / "config.json").read_text())
    cases = json.loads((reference / "evaluation_scenarios.json").read_text())
    config = ExplorationConfig(**metadata["environment"])
    config.validate()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(metadata.get("torch_threads", 1))
    models = {}
    fingerprints = {}
    for seed in metadata["training_seeds"]:
        path = reference / f"seed-{seed}" / "trained.zip"
        fingerprints[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        models[f"ppo_seed_{seed}"] = MaskablePPO.load(path, device="cpu")
    settings = {"environment": asdict(config), "source_run": str(reference.resolve()),
        "source_model_sha256": fingerprints, "evaluation_cases": len(cases),
        "planner": "exhaustive ordered two-measurement variance-reduction-per-distance",
        "terminal_fallback": "one-move sequence only if no second move is feasible",
        "execution": "execute first move only; replan after every actual measurement",
        "training": "none; reuse all archived final PPO checkpoints",
        "data_status": "same development scenarios; no new out-of-sample claim",
        "limitations": "finite-horizon density heuristic, no remaining-budget tail value or global optimality guarantee"}
    (output / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
    (output / "evaluation_scenarios.json").write_text(json.dumps(cases, indent=2) + "\n")
    summary = evaluate_methods(config, cases, models, output, include_lookahead=True)
    rows = read_rows(output / "evaluation_metrics.csv")
    archived = {(row["method"], row["scenario"]): row
                for row in read_rows(reference / "evaluation_metrics.csv")}
    for row in rows:
        if row["method"] == "lookahead_2":
            continue
        previous = archived[row["method"], row["scenario"]]
        for key in ("final_mean_variance", "heldout_rmse", "heldout_coverage_95",
                    "information_gain", "travel_distance", "measurements"):
            if not np.isclose(float(row[key]), float(previous[key]), rtol=1e-10, atol=1e-12):
                raise RuntimeError(f"Archived baseline changed: {row['method']}, {row['scenario']}, {key}")
    paired = compare_paired(rows)
    deltas = [row["variance_difference"] for row in paired]
    mean_greedy = summary["greedy"]["final_mean_variance"]
    comparison = {"scenarios": len(paired), "archived_baselines_reproduced": True,
        "mean_paired_variance_difference": float(np.mean(deltas)),
        "lower_variance_count": int(sum(delta < -1e-10 for delta in deltas)),
        "higher_variance_count": int(sum(delta > 1e-10 for delta in deltas)),
        "tied_variance_count": int(sum(abs(delta) <= 1e-10 for delta in deltas)),
        "relative_reduction_of_mean_variance_percent": 100 * (
            mean_greedy - summary["lookahead_2"]["final_mean_variance"]) / mean_greedy,
        "mean_total_planning_seconds": {name: values["planning_seconds"] for name, values in summary.items()},
        "maximum_budget_excess": max(0.0, max(float(row["travel_distance"]) - float(row["budget"]) for row in rows))}
    (output / "paired_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    render_paired(output / "paired_comparison.png", paired)
    return summary, comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("outputs/rl-study"))
    parser.add_argument("--output", type=Path, default=Path("outputs/lookahead-study"))
    args = parser.parse_args()
    summary, comparison = run_comparison(args.reference, args.output)
    for method, row in summary.items():
        print(f"{method:20s} variance={row['final_mean_variance']:.4f} "
              f"RMSE={row['heldout_rmse']:.4f} planning={row['planning_seconds']:.5f}s")
    print(json.dumps(comparison, indent=2))
    print("Artifacts:", args.output.resolve())


if __name__ == "__main__":
    main()
