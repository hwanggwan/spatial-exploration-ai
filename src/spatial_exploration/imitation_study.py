"""Train matched greedy/rollout clones; evaluate once on fresh held-out cases."""

import argparse
from dataclasses import asdict
from importlib import metadata
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
from sb3_contrib import MaskablePPO
import torch

from .baseline import write_csv
from .exploration_env import ExplorationConfig, SpatialExplorationEnv
from .imitation import ClonedPolicy, CloningConfig, collect_common_states, train_policy
from .lookahead_study import read_rows
from .rl_study import evaluate_methods, scenarios
from .rollout import rollout_for_environment


def make_cases(config, count, scenario_seed, field_seed_start):
    cases = scenarios(config, count, seed=scenario_seed)
    for index, case in enumerate(cases):
        case["field_seed"] = field_seed_start + index
    return cases


def assert_disjoint_cases(groups):
    seen_parameters, seen_fields = set(), set()
    for cases in groups:
        parameters = {(case["start_index"], case["length_scale"], case["budget"]) for case in cases}
        fields = {case["field_seed"] for case in cases}
        if len(parameters) != len(cases) or len(fields) != len(cases):
            raise ValueError("Duplicate cases within a split")
        if seen_parameters & parameters or seen_fields & fields:
            raise ValueError("Training, validation, test, and archived cases must be disjoint")
        seen_parameters.update(parameters)
        seen_fields.update(fields)


def audit_student_states(config, cases, students, output, evaluation_rows):
    """Teacher queries after model freeze, for diagnosis only; never guide action."""
    records, summaries = [], {}
    env = SpatialExplorationEnv(config)
    final_reference = {(row["method"], int(row["scenario"])): float(row["final_mean_variance"])
                       for row in evaluation_rows}
    started = time.perf_counter()
    for name, (policy, teacher) in students.items():
        agreements, teacher_seconds = [], 0.0
        for case in cases:
            observation, _ = env.reset(seed=case["field_seed"], options=case)
            while not env.finished:
                action, _ = policy.predict(observation, action_masks=env.action_masks())
                began = time.perf_counter()
                label = rollout_for_environment(env).first if teacher == "rollout" else env.greedy_action()
                teacher_seconds += time.perf_counter() - began
                agreements.append(action == label)
                records.append({"method": name, "scenario": case["scenario"],
                    "decision": len(env.route) - 1, "student_action": action,
                    "teacher_action": label, "agrees": action == label})
                observation, _, _, _, info = env.step(action)
                if info["invalid_action"]:
                    raise RuntimeError("Student selected an invalid action during audit")
            if not np.isclose(env.mean_variance, final_reference[name, case["scenario"]], atol=1e-12, rtol=0):
                raise RuntimeError("Audit did not reproduce the frozen student outcome")
        summaries[name] = {"teacher": teacher, "student_visited_states": len(agreements),
            "action_agreement": float(np.mean(agreements)), "teacher_query_seconds": teacher_seconds}
        print(f"Audited {name}: {len(agreements)} student states", flush=True)
    write_csv(output / "student_state_audit.csv", records)
    return {"models": summaries, "audit_seconds": time.perf_counter() - started,
            "use": "post-freeze diagnostic only; teacher actions never executed or used for test adaptation"}


def paired_results(rows, student_names):
    indexed = {(row["method"], int(row["scenario"])): row for row in rows}
    indices = sorted({int(row["scenario"]) for row in rows})
    results = {}
    for name in student_names:
        results[name] = {}
        for reference in ("greedy", "rollout"):
            deltas = np.asarray([float(indexed[name, index]["final_mean_variance"]) -
                float(indexed[reference, index]["final_mean_variance"]) for index in indices])
            rng = np.random.default_rng(950000)
            bootstraps = deltas[rng.integers(0, len(deltas), size=(5000, len(deltas)))].mean(axis=1)
            results[name][reference] = {"mean_variance_difference": float(deltas.mean()),
                "paired_case_bootstrap_95_percent_interval": np.quantile(bootstraps, [0.025, 0.975]).tolist(),
                "lower_variance_cases": int(np.sum(deltas < -1e-10)),
                "higher_variance_cases": int(np.sum(deltas > 1e-10)),
                "tied_variance_cases": int(np.sum(np.abs(deltas) <= 1e-10))}
    return results


def render_training(path, output, models):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for name in models:
        teacher = "rollout" if "rollout" in name else "greedy"
        seed = int(name.rsplit("_", 1)[1])
        rows = read_rows(output / f"{teacher}-seed-{seed}" / "training_history.csv")
        for axis, metric in zip(axes, ("validation_cross_entropy", "validation_action_accuracy")):
            axis.plot([int(row["epoch"]) for row in rows], [float(row[metric]) for row in rows], label=name)
            axis.set_xlabel("Epoch")
            axis.grid(alpha=0.2)
    axes[0].set_title("Validation masked cross entropy (lower is better)")
    axes[1].set_title("Validation teacher-action accuracy (higher is better)")
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=8)
    fig.suptitle("Behavioral cloning: common states; validation episodes excluded from training")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_tradeoff(path, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [("Greedy", ("greedy",), "#b5762b", "o"),
        ("Two-step", ("lookahead_2",), "#376aa3", "o"),
        ("Rollout teacher", ("rollout",), "#27754d", "*"),
        ("Greedy clone", tuple(name for name in summary if name.startswith("bc_greedy")), "#885c9b", "s"),
        ("Rollout clone", tuple(name for name in summary if name.startswith("bc_rollout")), "#d3585a", "D"),
        ("Archived PPO", tuple(name for name in summary if name.startswith("ppo_")), "#737373", "^")]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    for label, names, color, marker in groups:
        for axis, metric in zip(axes, ("final_mean_variance", "heldout_rmse")):
            axis.scatter([1000 * summary[name]["planning_seconds"] for name in names],
                [summary[name][metric] for name in names], color=color, marker=marker,
                s=100 if marker == "*" else 55, label=label, edgecolors="white", linewidths=0.5)
            axis.set_xscale("log")
            axis.set_xlabel("Mean cumulative action-selection time / episode (ms; log scale)")
            axis.grid(alpha=0.2)
    axes[0].set_ylabel("Mean final whole-grid posterior variance")
    axes[1].set_ylabel("Mean held-out latent-field RMSE")
    axes[0].set_title("Primary objective versus deployment cost")
    axes[1].set_title("Realized prediction error versus deployment cost")
    axes[0].legend(fontsize=8)
    fig.suptitle("96 new paired cases — lower and left is better; both neural seeds shown")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_study(config, reference, output):
    config.validate()
    reference, output = Path(reference), Path(output)
    source, destination = reference.resolve(), output.resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Keep the imitation output separate from archived PPO files")
    if (output / "config.json").exists():
        raise ValueError("Use a fresh output folder to preserve the frozen study")
    original = json.loads((reference / "config.json").read_text())
    if tuple(original["network"]["pi"]) != tuple(config.hidden_sizes):
        raise ValueError("Use the archived PPO actor widths for this matched-architecture study")
    environment = ExplorationConfig(**original["environment"])
    environment.validate()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    splits = {"train": make_cases(environment, config.train_episodes, 910000, 1100000),
        "validation": make_cases(environment, config.validation_episodes, 920000, 1200000),
        "test": make_cases(environment, config.test_episodes, 930000, 1300000)}
    assert_disjoint_cases([*splits.values(), json.loads((reference / "evaluation_scenarios.json").read_text())])
    settings = {"environment": asdict(environment), "training": asdict(config),
        "algorithm": "masked behavioral cloning; hard teacher-action cross entropy",
        "observation": "same full GP covariance, visited mask, current xy, remaining budget and length scale as PPO",
        "actor": "MLP with Tanh hidden layers; identical sizes to archived PPO actor; no critic",
        "collection": "common states; alternate rollout and greedy behavior by episode; query both teachers at each state",
        "split_unit": "complete episodes; no state-wise random split",
        "scenario_rng_seeds": {"train": 910000, "validation": 920000, "test": 930000},
        "field_seed_starts": {"train": 1100000, "validation": 1200000, "test": 1300000},
        "selection": "minimum validation cross entropy within a fixed epoch budget; both seeds reported",
        "test_protocol": "freeze all selected model hashes before evaluating fresh test cases; no test adaptation",
        "hidden_field": "no fields in training/validation collection; test fields only for mapping evaluation",
        "distribution_shift": "post-freeze teacher-action agreement on student-visited test states; no DAgger training",
        "ppo_reference": str(reference.resolve()),
        "comparison_limit": "matched actor architecture but different training data, objectives, and compute; not equal-budget algorithm ranking",
        "python_version": platform.python_version(), "numpy_version": np.__version__, "torch_threads": 1}
    (output / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
    for name, cases in splits.items():
        (output / f"{name}_cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    packages = sorted(f"{item.metadata['Name']}=={item.version}" for item in metadata.distributions())
    (output / "requirements-run.txt").write_text("\n".join(packages) + "\n")
    train, train_details = collect_common_states(environment, splits["train"], output / "train.npz")
    validation, validation_details = collect_common_states(environment, splits["validation"], output / "validation.npz")
    dataset_details = {"train": train_details, "validation": validation_details,
        "fingerprints": {name: hashlib.sha256((output / f"{name}.npz").read_bytes()).hexdigest()
                         for name in ("train", "validation")}}
    (output / "dataset.json").write_text(json.dumps(dataset_details, indent=2) + "\n")
    models, students, training = {}, {}, {}
    for teacher in ("greedy", "rollout"):
        for seed in config.seeds:
            name = f"bc_{teacher}_seed_{seed}"
            folder = output / f"{teacher}-seed-{seed}"
            policy, details = train_policy(train, validation, teacher, seed, config, folder)
            models[name] = policy
            students[name] = (policy, teacher)
            training[name] = details
    fingerprints = {str((output / f"{teacher}-seed-{seed}" / "selected.pt").resolve()):
        hashlib.sha256((output / f"{teacher}-seed-{seed}" / "selected.pt").read_bytes()).hexdigest()
        for teacher in ("greedy", "rollout") for seed in config.seeds}
    for seed in original["training_seeds"]:
        path = reference / f"seed-{seed}" / "trained.zip"
        models[f"ppo_seed_{seed}"] = MaskablePPO.load(path, device="cpu")
        fingerprints[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    # Persist the selected checkpoints and decision rule before using test cases.
    (output / "training_summary.json").write_text(json.dumps(training, indent=2) + "\n")
    (output / "frozen_models.json").write_text(json.dumps({"sha256": fingerprints,
        "selection_complete_before_test_evaluation": True}, indent=2) + "\n")
    render_training(output / "training.png", output, students)
    test_output = output / "test"
    test_output.mkdir(exist_ok=True)
    summary = evaluate_methods(environment, splits["test"], models, test_output,
        include_lookahead=True, include_rollout=True,
        example_methods=("greedy", "rollout", "bc_greedy_seed_0", "bc_rollout_seed_0", "bc_rollout_seed_1"))
    rows = read_rows(test_output / "evaluation_metrics.csv")
    if max(float(row["travel_distance"]) - float(row["budget"]) for row in rows) > 1e-12:
        raise RuntimeError("Evaluation exceeded a travel budget")
    for row in rows:
        if not np.isclose(float(row["total_reward"]), environment.reward_scale * float(row["information_gain"]),
                          atol=1e-9, rtol=0):
            raise RuntimeError("Reward/variance identity failed")
    audit = audit_student_states(environment, splits["test"], students, test_output, rows)
    results = {"dataset": dataset_details, "training": training, "test": summary,
        "paired_test_comparisons": paired_results(rows, students), "student_state_audit": audit,
        "all_test_budget_constraints_satisfied": True,
        "bootstrap_unit": "paired scenario, within each fixed trained model; not training-seed uncertainty"}
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    render_tradeoff(output / "tradeoff.png", summary)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-episodes", type=int, default=1024)
    parser.add_argument("--validation-episodes", type=int, default=128)
    parser.add_argument("--test-episodes", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--reference", type=Path, default=Path("outputs/rl-study"))
    parser.add_argument("--output", type=Path, default=Path("outputs/imitation-study"))
    args = parser.parse_args()
    config = CloningConfig(train_episodes=args.train_episodes, validation_episodes=args.validation_episodes,
        test_episodes=args.test_episodes, epochs=args.epochs)
    results = run_study(config, args.reference, args.output)
    for name, row in results["test"].items():
        print(f"{name:24s} variance={row['final_mean_variance']:.5f} "
              f"RMSE={row['heldout_rmse']:.5f} planning={row['planning_seconds']:.6f}s")
    print("Artifacts:", args.output.resolve())


if __name__ == "__main__":
    main()
