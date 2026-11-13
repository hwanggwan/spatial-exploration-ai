"""Quantify greedy and two-step planning headroom against tiny exact problems."""

import argparse
from dataclasses import asdict, dataclass
import itertools
import json
from pathlib import Path
import platform
import time

import numpy as np

from .baseline import GPBelief, make_grid, rbf_covariance, write_csv
from .exact_planner import exhaustive_plan
from .lookahead import plan_measurements
from .rollout import plan_rollout


@dataclass(frozen=True)
class ExactStudyConfig:
    grid_size: int = 4
    starts: tuple = (0, 5, 15)
    length_scales: tuple = (0.12, 0.23, 0.35)
    budgets: tuple = (0.75, 1.1, 1.45)
    noise_std: float = 0.1
    max_prefixes: int = 1000000

    def validate(self):
        if self.grid_size < 2 or not self.starts or not self.length_scales or not self.budgets:
            raise ValueError("Specify a grid of at least two and nonempty scenario factors")
        if any(not 0 <= start < self.grid_size**2 or start % 5 == 4 for start in self.starts):
            raise ValueError("Starting sites must be valid permitted grid indices")
        if any(not np.isfinite(value) or value <= 0 for value in
               (*self.length_scales, *self.budgets, self.noise_std)) or self.max_prefixes < 1:
            raise ValueError("Scales, budgets, noise, and search limit must be positive")


def execute_heuristic(prior, distances, start, allowed, budget, noise_std, depth=1, planner=None):
    belief = GPBelief(prior, noise_std)
    belief.observe(start, 0.0)
    visited = np.zeros(len(prior), dtype=bool)
    visited[start] = True
    route, travel, planning_seconds = [start], 0.0, 0.0
    while True:
        available = allowed & ~visited & (distances[route[-1]] <= budget - travel + 1e-12)
        if not available.any():
            break
        started = time.perf_counter()
        arguments = (belief.covariance, distances, route[-1], visited,
            allowed, budget - travel, noise_std**2)
        plan = planner(*arguments) if planner else plan_measurements(*arguments, depth=depth)
        planning_seconds += time.perf_counter() - started
        travel += float(distances[route[-1], plan.first])
        route.append(plan.first)
        visited[plan.first] = True
        belief.observe(plan.first, 0.0)
    return tuple(route), belief, travel, planning_seconds


def reconstruct(prior, route, noise_std):
    belief = GPBelief(prior, noise_std)
    for index in route:
        belief.observe(index, 0.0)
    return belief


def render_gaps(path, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for name, color in (("greedy", "#b5762b"), ("lookahead_2", "#376aa3"), ("rollout", "#27754d")):
        selected = [row for row in rows if row["method"] == name]
        if not selected:
            continue
        axes[0].plot([row["case"] for row in selected],
                     [row["excess_variance_percent"] for row in selected],
                     ".-", label=name, color=color, linewidth=1)
        axes[1].scatter([row["optimal_final_variance"] for row in selected],
                        [row["final_mean_variance"] for row in selected],
                        label=name, color=color, s=25, alpha=0.8)
    values = [row["final_mean_variance"] for row in rows]
    axes[1].plot([min(values), max(values)], [min(values), max(values)], "k--", linewidth=1,
                 label="matches optimum")
    axes[0].set_xlabel("Predefined factorial case index")
    axes[0].set_ylabel("Final variance above optimum (%)")
    axes[0].set_title("Relative gap to the exact discrete optimum")
    axes[1].set_xlabel("Exact optimal final variance")
    axes[1].set_ylabel("Heuristic final variance")
    axes[1].set_title("Points above the diagonal leave room for improvement")
    for axis in axes:
        axis.legend(fontsize=8)
        axis.grid(alpha=0.2)
    fig.suptitle("Complete enumeration on small known-covariance problems")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_routes(path, case, results, config, selection):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = make_grid(config.grid_size)
    allowed = np.arange(len(points)) % 5 != 4
    prior = rbf_covariance(points, case["length_scale"])
    half_step = 0.5 / (config.grid_size - 1)
    names = list(results)
    fig, axes = plt.subplots(1, len(names), figsize=(4.5 * len(names), 4.8), constrained_layout=True)
    for axis, name in zip(axes, names):
        row = results[name]
        route = row["route"]
        belief = reconstruct(prior, route, config.noise_std)
        image = axis.imshow(belief.std.reshape(config.grid_size, config.grid_size),
            origin="lower", extent=(-half_step, 1 + half_step, -half_step, 1 + half_step),
            cmap="magma", vmin=0, vmax=1)
        axis.scatter(points[allowed, 0], points[allowed, 1], s=12, color="gray", alpha=0.5)
        axis.scatter(points[~allowed, 0], points[~allowed, 1], s=28, marker="x", color="cyan",
                     label="reserved evaluation sites")
        locations = points[list(route)]
        axis.plot(locations[:, 0], locations[:, 1], "w.-", linewidth=1)
        for first, second in zip(locations[:-1], locations[1:]):
            axis.annotate("", xy=second, xytext=first,
                          arrowprops={"arrowstyle": "->", "color": "white", "lw": 0.8})
        axis.plot(*locations[0], "wo", markeredgecolor="black", markersize=7)
        axis.plot(*locations[-1], "wx", markersize=8)
        axis.set_title(f"{name}: variance={row['final_mean_variance']:.4f}\n"
                       f"distance={row['travel_distance']:.3f}, measurements={len(route)}", fontsize=10)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        fig.colorbar(image, ax=axis, shrink=0.8, label="Posterior standard deviation")
    fig.suptitle(f"Case {case['case']} ({selection}); length={case['length_scale']}, "
                 f"budget={case['budget']}, start={case['start_index']}")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_study(config, output, include_rollout=False):
    config.validate()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    points = make_grid(config.grid_size)
    distances = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
    allowed = np.arange(len(points)) % 5 != 4
    cases = [{"case": index, "start_index": int(start), "length_scale": float(length),
              "budget": float(budget)} for index, (start, length, budget) in enumerate(
                  itertools.product(config.starts, config.length_scales, config.budgets))]
    (output / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    (output / "config.json").write_text(json.dumps({**asdict(config),
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "objective": "minimum final whole-grid mean latent posterior variance",
        "covariance": "known unit-variance RBF; mean zero and sensor noise known",
        "candidate_sites": "all grid indices except index % 5 == 4; no revisits",
        "constraints": "Euclidean travel budget; no return, obstacles, or measurement cost",
        "solver": "depth-first exhaustive enumeration of every feasible route prefix",
        "budget_tolerance": 1e-12, "objective_tie_tolerance": 1e-12,
        "work_limit": "abort with an exception; never report an interrupted result as optimal",
        "evaluation": "deterministic factorial diagnostic; no sampled field or realized RMSE",
        "include_rollout": include_rollout,
    }, indent=2) + "\n")
    rows, certificates, details = [], [], {}
    for case in cases:
        prior = rbf_covariance(points, case["length_scale"])
        belief = GPBelief(prior, config.noise_std)
        belief.observe(case["start_index"], 0.0)
        initial_variance = float(np.trace(belief.covariance) / len(points))
        visited = np.zeros(len(points), dtype=bool)
        visited[case["start_index"]] = True
        exact = exhaustive_plan(belief.covariance, distances, case["start_index"], visited,
            allowed, case["budget"], config.noise_std**2, config.max_prefixes)
        certificates.append({**case, "search_complete": True,
            "prefixes_evaluated": exact.prefixes_evaluated,
            "terminal_routes": exact.terminal_routes,
            "maximum_additional_measurements": exact.maximum_additional_measurements,
            "search_seconds": exact.search_seconds})
        optimal_belief = reconstruct(prior, exact.route, config.noise_std)
        optimum = float(np.trace(optimal_belief.covariance) / len(points))
        if not np.isclose(optimum, exact.final_mean_variance, atol=1e-12, rtol=0):
            raise RuntimeError("Returned exact route does not reconstruct the reported objective")
        results = {"optimal": (exact.route, optimal_belief, exact.travel_distance, exact.search_seconds)}
        for name, depth in (("greedy", 1), ("lookahead_2", 2)):
            results[name] = execute_heuristic(prior, distances, case["start_index"], allowed,
                                              case["budget"], config.noise_std, depth)
        if include_rollout:
            results["rollout"] = execute_heuristic(prior, distances, case["start_index"], allowed,
                case["budget"], config.noise_std, planner=plan_rollout)
        details[case["case"]] = {}
        for name, (route, final_belief, travel, seconds) in results.items():
            variance = float(np.trace(final_belief.covariance) / len(points))
            gap = variance - optimum
            if gap < -1e-10:
                raise RuntimeError("A heuristic beat the exhaustive result; investigate the solver")
            if travel > case["budget"] + 1e-12 or len(route) != len(set(route)):
                raise RuntimeError("Invalid route in the exact comparison")
            if not np.all(allowed[list(route)]):
                raise RuntimeError("Reserved site was measured")
            optimal_gain = initial_variance - optimum
            row = {**case, "method": name, "initial_mean_variance": initial_variance,
                "optimal_final_variance": optimum, "final_mean_variance": variance,
                "absolute_variance_gap": max(0.0, gap),
                "excess_variance_percent": 100 * max(0.0, gap) / optimum,
                "optimal_information_recovered_percent": 100 * (initial_variance - variance) / optimal_gain
                if optimal_gain > 1e-12 else 100.0,
                "measurements": len(route), "travel_distance": travel, "planning_seconds": seconds,
                "route": " ".join(str(index) for index in route)}
            rows.append(row)
            details[case["case"]][name] = {**row, "route": route}
        if (case["case"] + 1) % 9 == 0:
            print(f"Completed exact case {case['case'] + 1}/{len(cases)}", flush=True)
    summary = {"cases": len(cases), "all_searches_complete": True,
        "total_prefixes_evaluated": sum(row["prefixes_evaluated"] for row in certificates),
        "maximum_prefixes_in_one_case": max(row["prefixes_evaluated"] for row in certificates),
        "total_terminal_routes": sum(row["terminal_routes"] for row in certificates), "methods": {}}
    for name in ("optimal", "greedy", "lookahead_2") + (("rollout",) if include_rollout else ()):
        selected = [row for row in rows if row["method"] == name]
        summary["methods"][name] = {
            **{key: float(np.mean([row[key] for row in selected])) for key in
                ("final_mean_variance", "absolute_variance_gap", "excess_variance_percent",
                 "optimal_information_recovered_percent", "measurements", "travel_distance", "planning_seconds")},
            "max_excess_variance_percent": max(row["excess_variance_percent"] for row in selected),
            "optimal_case_count": sum(row["absolute_variance_gap"] <= 1e-10 for row in selected)}
    worst_case = max((row for row in rows if row["method"] == "greedy"),
                     key=lambda row: row["excess_variance_percent"])["case"]
    summary["largest_greedy_gap_case"] = worst_case
    summary["greedy_by_length_scale"] = {}
    for length in config.length_scales:
        selected = [row for row in rows if row["method"] == "greedy" and row["length_scale"] == length]
        summary["greedy_by_length_scale"][str(length)] = {
            "cases": len(selected),
            "optimal_case_count": sum(row["absolute_variance_gap"] <= 1e-10 for row in selected),
            "mean_excess_variance_percent": float(np.mean([row["excess_variance_percent"] for row in selected]))}
    # Post-hoc explanatory diagnostic, not a new planner or a representative case:
    # find a failure where the entire exact optimal route fits within two moves.
    counterexamples = [row for row in rows if row["method"] == "greedy"
        and row["absolute_variance_gap"] > 1e-10
        and details[row["case"]]["optimal"]["measurements"] <= 3]
    if counterexamples:
        counterexample = min(counterexamples, key=lambda row: row["case"])["case"]
        diagnostic = {"case": cases[counterexample],
            "selection": "first post-hoc failure with an optimum using at most two additional measurements",
            "routes": {}}
        for name, row in details[counterexample].items():
            gain = row["initial_mean_variance"] - row["final_mean_variance"]
            diagnostic["routes"][name] = {"route": row["route"],
                "final_mean_variance": row["final_mean_variance"],
                "total_information_gain": gain, "travel_distance": row["travel_distance"],
                "gain_per_distance": gain / row["travel_distance"]}
        (output / "density_counterexample.json").write_text(json.dumps(diagnostic, indent=2) + "\n")
        render_routes(output / "density_counterexample.png", cases[counterexample],
                      details[counterexample], config, "post-hoc diagnostic: optimum fits within two moves")
        summary["density_counterexample_case"] = counterexample
    write_csv(output / "metrics.csv", rows)
    write_csv(output / "search_certificates.csv", certificates)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    render_gaps(output / "optimality_gaps.png", rows)
    render_routes(output / "routes_case_0.png", cases[0], details[0], config, "predefined example")
    render_routes(output / "largest_gap_routes.png", cases[worst_case], details[worst_case], config,
                  "largest observed greedy gap; selected post hoc for diagnosis")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--starts", type=int, nargs="+", default=[0, 5, 15])
    parser.add_argument("--length-scales", type=float, nargs="+", default=[0.12, 0.23, 0.35])
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.75, 1.1, 1.45])
    parser.add_argument("--max-prefixes", type=int, default=1000000)
    parser.add_argument("--output", type=Path, default=Path("outputs/exact-study"))
    parser.add_argument("--include-rollout", action="store_true")
    args = parser.parse_args()
    config = ExactStudyConfig(args.grid_size, tuple(args.starts), tuple(args.length_scales),
                             tuple(args.budgets), max_prefixes=args.max_prefixes)
    if args.include_rollout and args.output.resolve() == Path("outputs/exact-study").resolve():
        parser.error("Use a separate --output folder for the rollout comparison")
    summary = run_study(config, args.output, include_rollout=args.include_rollout)
    print(json.dumps(summary, indent=2))
    print("Artifacts:", args.output.resolve())


if __name__ == "__main__":
    main()
