"""Apply existing Maskable PPO to a controlled budgeted spatial exploration task."""

import argparse
from dataclasses import asdict
from importlib import metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_checker import check_env
import torch

from .baseline import GPBelief, evaluate, rbf_covariance, render, write_csv
from .exploration_env import ExplorationConfig, SpatialExplorationEnv
from .lookahead import plan_for_environment
from .rollout import rollout_for_environment


class TrainingProgress(BaseCallback):
    def __init__(self, seed):
        super().__init__()
        self.seed = seed

    def _on_step(self):
        if self.num_timesteps % 8192 == 0:
            print(f"training seed={self.seed}, transitions={self.num_timesteps}", flush=True)
        return True


def scenarios(config, count, seed=700000):
    rng = np.random.default_rng(seed)
    allowed = np.flatnonzero(np.arange(config.grid_size**2) % 5 != 4)
    return [{"scenario": index, "field_seed": 800000 + index,
             "start_index": int(rng.choice(allowed)),
             "length_scale": float(rng.uniform(*config.length_scale_range)),
             "budget": float(rng.uniform(*config.budget_range))}
            for index in range(count)]


def choose_action(method, model, env, observation, rng, planning_details=None):
    if method == "random":
        return int(rng.choice(np.flatnonzero(env.action_masks())))
    if method == "greedy":
        return env.greedy_action()
    if method == "lookahead_2":
        plan = plan_for_environment(env, depth=2)
        if planning_details is not None:
            planning_details.update(asdict(plan))
        return plan.first
    if method == "rollout":
        plan = rollout_for_environment(env)
        if planning_details is not None:
            planning_details.update(asdict(plan))
            for key in ("route", "greedy_route"):
                planning_details[key] = " ".join(map(str, planning_details[key]))
        return plan.first
    action, _ = model.predict(observation, action_masks=env.action_masks(), deterministic=True)
    return int(action)


def evaluate_methods(config, cases, models, output, include_lookahead=False, include_rollout=False,
                     example_methods=None):
    rows, traces, examples, plans = [], [], {}, []
    methods = {"random": None, "greedy": None}
    if include_lookahead:
        methods["lookahead_2"] = None
    if include_rollout:
        methods["rollout"] = None
    methods.update(models)
    for case in cases:
        field_rng = np.random.default_rng(case["field_seed"])
        field_env = SpatialExplorationEnv(config)
        field_env.reset(seed=case["field_seed"], options=case)
        prior = rbf_covariance(field_env.points, case["length_scale"])
        factor = np.linalg.cholesky(prior + 1e-10 * np.eye(len(prior)))
        truth = factor @ field_rng.standard_normal(len(prior))
        sensor = truth + config.noise_std * field_rng.standard_normal(len(prior))
        for method, model in methods.items():
            env = SpatialExplorationEnv(config)
            observation, _ = env.reset(seed=case["field_seed"], options=case)
            mapper = GPBelief(prior, config.noise_std)
            mapper.observe(env.current, sensor[env.current])
            rng = np.random.default_rng(case["field_seed"] + 100000)
            planning_seconds, total_reward = 0.0, 0.0

            def record():
                traces.append({"scenario": case["scenario"], "method": method,
                    "measurements": len(env.route), "index": env.current,
                    "x": float(env.points[env.current, 0]), "y": float(env.points[env.current, 1]),
                    "observed_value": float(sensor[env.current]),
                    "travel_distance": env.travel_distance, "mean_variance": env.mean_variance,
                    **evaluate(truth, mapper, env.evaluation_mask)})

            record()
            while not env.finished:
                planning_details = {}
                started = time.perf_counter()
                action = choose_action(method, model, env, observation, rng, planning_details)
                planning_seconds += time.perf_counter() - started
                if planning_details:
                    plans.append({"method": method, "scenario": case["scenario"], "decision": len(env.route) - 1,
                        "current": env.current, "remaining_budget": env.budget - env.travel_distance,
                        **planning_details})
                observation, reward, _, _, info = env.step(action)
                if info["invalid_action"]:
                    raise RuntimeError("An evaluated policy selected an invalid action")
                total_reward += reward
                mapper.observe(action, sensor[action])
                record()
            np.testing.assert_allclose(mapper.covariance, env.belief.covariance, atol=1e-10)
            rows.append({**case, "method": method, "final_mean_variance": env.mean_variance,
                "heldout_mean_variance": float(np.mean(np.diag(mapper.covariance)[env.evaluation_mask])),
                "information_gain": env.initial_variance - env.mean_variance,
                "travel_distance": env.travel_distance, "measurements": len(env.route),
                "total_reward": total_reward, "planning_seconds": planning_seconds,
                **evaluate(truth, mapper, env.evaluation_mask)})
            if case["scenario"] == 0:
                examples[method] = (mapper, env.route.copy(), env.cumulative_distances.copy())
        if case["scenario"] == 0:
            # Show every trained seed, without choosing the best-performing seed.
            shown = {name: examples[name] for name in examples
                     if (name in ("random", "greedy") or "untrained" not in name)
                     and (example_methods is None or name in example_methods)}
            render(output / "routes.png", truth, shown, config.grid_size,
                   title="Budgeted mapping — evaluation scenario 0; all methods use the same oracle GP")
    summary = {}
    greedy_by_case = {row["scenario"]: row for row in rows if row["method"] == "greedy"}
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        deltas = [row["final_mean_variance"] - greedy_by_case[row["scenario"]]["final_mean_variance"]
                  for row in selected]
        summary[method] = {"scenarios": len(selected),
            **{key: float(np.mean([row[key] for row in selected])) for key in
               ("final_mean_variance", "heldout_mean_variance", "information_gain", "heldout_rmse", "heldout_coverage_95",
                "travel_distance", "measurements", "planning_seconds")},
            "paired_variance_difference_vs_greedy": float(np.mean(deltas)),
            "scenarios_lower_variance_than_greedy": int(sum(delta < -1e-10 for delta in deltas))}
    write_csv(output / "evaluation_metrics.csv", rows)
    write_csv(output / "trajectories.csv", traces)
    if plans:
        for method, filename in (("lookahead_2", "lookahead_plans.csv"), ("rollout", "rollout_plans.csv")):
            selected = [plan for plan in plans if plan["method"] == method]
            if selected:
                write_csv(output / filename, selected)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    render_evaluation(output / "evaluation.png", rows, methods)
    return summary


def render_evaluation(path, rows, methods):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(methods)
    fig, axes = plt.subplots(1, 2, figsize=(max(13, 1.6 * len(names)), 5), constrained_layout=True)
    for axis, metric, title in zip(axes,
        ("final_mean_variance", "heldout_rmse"),
        ("Primary: final whole-grid posterior variance", "Secondary: held-out latent-field RMSE")):
        values = [[row[metric] for row in rows if row["method"] == name] for name in names]
        for index, samples in enumerate(values):
            jitter = np.linspace(-0.12, 0.12, len(samples))
            axis.scatter(np.full(len(samples), index) + jitter, samples, alpha=0.35, s=14)
            axis.plot([index - 0.2, index + 0.2], [np.mean(samples)] * 2, color="black", linewidth=2)
        axis.set_xticks(range(len(names)), [name.replace("_", "\n") for name in names], fontsize=8)
        axis.set_title(title, fontsize=10)
        axis.set_ylabel("Lower is better")
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Paired evaluation scenarios; dots are scenarios, black lines are means")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(config, steps, training_seeds, evaluation_count, output):
    config.validate()
    if steps < 1024 or steps % 1024:
        raise ValueError("Training steps must be a positive multiple of 1024")
    if evaluation_count < 1 or not training_seeds or len(set(training_seeds)) != len(training_seeds):
        raise ValueError("Use at least one evaluation case and distinct training seeds")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    check_env(SpatialExplorationEnv(config), warn=True)
    cases = scenarios(config, evaluation_count)
    (output / "evaluation_scenarios.json").write_text(json.dumps(cases, indent=2) + "\n")
    settings = {"environment": asdict(config), "algorithm": "SB3-Contrib MaskablePPO",
        "training_seeds": training_seeds, "transitions_per_seed": steps,
        "evaluation_cases": evaluation_count, "evaluation_scenario_seed": 700000,
        "field_seed_start": 800000, "python_version": platform.python_version(),
        "device": "cpu", "torch_threads": 1, "gamma": 1.0, "gae_lambda": 0.95,
        "n_envs": 4, "n_steps": 256, "batch_size": 128, "n_epochs": 5,
        "learning_rate": 0.0003, "entropy_coefficient": 0.01,
        "network": {"pi": [128, 128], "vf": [128, 128]},
        "observation": "full posterior covariance, visited mask, current xy, remaining budget, length scale",
        "reward": "100 times decrease in mean latent posterior variance; no hidden field values",
        "selection": "fixed final training checkpoint; no evaluation-based model selection",
        "status": "custom-task application of existing PPO; not CAtNIPP reproduction"}
    (output / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
    packages = sorted(f"{item.metadata['Name']}=={item.version}" for item in metadata.distributions())
    (output / "requirements-run.txt").write_text("\n".join(packages) + "\n")
    timings, models = [], {}
    for seed in training_seeds:
        folder = output / f"seed-{seed}"
        folder.mkdir(parents=True, exist_ok=True)
        vec_env = make_vec_env(lambda: SpatialExplorationEnv(config), n_envs=4,
                              seed=seed, monitor_dir=str(folder / "episodes"))
        model = MaskablePPO("MlpPolicy", vec_env, seed=seed, device="cpu", gamma=1.0,
            gae_lambda=0.95, n_steps=256, batch_size=128, n_epochs=5,
            ent_coef=0.01, learning_rate=0.0003,
            policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}}, verbose=0)
        model.save(folder / "untrained.zip")
        model.set_logger(configure(str(folder / "training"), ["csv"]))
        started = time.perf_counter()
        model.learn(total_timesteps=steps, callback=TrainingProgress(seed), log_interval=1)
        elapsed = time.perf_counter() - started
        model.save(folder / "trained.zip")
        models[f"ppo_seed_{seed}_untrained"] = MaskablePPO.load(folder / "untrained.zip", device="cpu")
        models[f"ppo_seed_{seed}"] = model
        timings.append({"training_seed": seed, "transitions": model.num_timesteps,
                        "training_seconds": elapsed})
        vec_env.close()
        print(f"Finished seed {seed}: {elapsed:.1f}s", flush=True)
    write_csv(output / "training_times.csv", timings)
    return evaluate_methods(config, cases, models, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=32768)
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--evaluation-cases", type=int, default=24)
    parser.add_argument("--grid-size", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("outputs/rl-study"))
    args = parser.parse_args()
    summary = run(ExplorationConfig(grid_size=args.grid_size), args.steps,
                  args.training_seeds, args.evaluation_cases, args.output)
    for name, metrics in summary.items():
        print(f"{name:25s} variance={metrics['final_mean_variance']:.4f} "
              f"RMSE={metrics['heldout_rmse']:.4f}")
    print("Artifacts:", args.output.resolve())


if __name__ == "__main__":
    main()
