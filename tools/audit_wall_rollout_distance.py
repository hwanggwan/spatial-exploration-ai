"""Replay rollout under greedy's realized walking distances as a secondary check."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from spatial_exploration.wall_rollout_study import (
    BUDGETS, REGIMES, TEST_DOORS, dump, load_settings, metrics, plan_route, ridge_models,
    scenes, topology_design,
)
from spatial_exploration.baseline import write_csv


def run(reference, study):
    settings, digest = load_settings(reference)
    frozen = json.loads((study / "frozen.json").read_text())
    if digest != frozen["reference_sha256"]:
        raise ValueError("The ridge settings do not match the main comparison")
    routes = json.loads((study / "routes.json").read_text())
    with (study / "metrics.csv").open() as stream:
        greedy = {(int(r["door"]), int(r["seed"]), r["regime"], int(r["cap"])): r
                  for r in csv.DictReader(stream) if r["planner"] == "greedy"}
    rows, configurations = [], []
    for door in TEST_DOORS:
        design = topology_design(door)
        models = {n: ridge_models(design, spec)["wall_graph"] for n, spec in settings.items()}
        limited = {}
        for base in routes:
            if base["door"] != door or base["planner"] != "greedy":
                continue
            start, cap, limit = base["start"], base["cap"], base["distances"][-1]
            route = plan_route(design, models[cap], start, cap, limit, "rollout")
            assert route.distances[-1] <= limit + 1e-12
            assert route.final_variance <= base["final_variance"] + 1e-10
            limited[start, cap] = route
            configurations.append(dict(door=door, start=start, cap=cap, sites=route.sites,
                rollout_distance=route.distances[-1], greedy_distance=limit,
                rollout_measurements=len(route.sites), greedy_measurements=len(base["sites"]),
                rollout_final_variance=route.final_variance, greedy_final_variance=base["final_variance"]))
        for meta, truth, sensor in scenes(design, frozen["test_fields_per_layout"], frozen["test_seed"]):
            for cap, _ in BUDGETS:
                route = limited[meta["start"], cap]
                posterior = models[cap].condition(route.sites, sensor[list(route.sites)])
                rmse = metrics(truth, posterior, design, 1.0)["rmse"]
                base = greedy[door, meta["seed"], meta["regime"], cap]
                rows.append(dict(**meta, cap=cap, greedy_rmse=float(base["rmse"]), rollout_rmse=rmse,
                    greedy_distance=float(base["distance"]), rollout_distance=route.distances[-1],
                    greedy_measurements=int(base["measurements"]), rollout_measurements=len(route.sites)))

    rng = np.random.default_rng(141002)
    summary = []
    for regime in REGIMES:
        for cap, _ in BUDGETS:
            selected = [r for r in rows if (r["regime"], r["cap"]) == (regime, cap)]
            strata = {}
            for i, row in enumerate(selected):
                strata.setdefault((row["door"], row["start"], row["kappa"]), []).append(i)
            delta = np.array([r["rollout_rmse"] - r["greedy_rmse"] for r in selected])
            boot = np.concatenate([rng.choice(delta[idx], (6000, len(idx))) for idx in strata.values()],
                                  axis=1).mean(axis=1)
            summary.append(dict(regime=regime, cap=cap, fields=len(selected), difference=float(delta.mean()),
                interval=np.quantile(boot, [.025, .975]).tolist(),
                **{k: float(np.mean([r[k] for r in selected])) for k in (
                    "greedy_rmse", "rollout_rmse", "greedy_distance", "rollout_distance",
                    "greedy_measurements", "rollout_measurements")}))
    write_csv(study / "matched-distance-metrics.csv", rows)
    dump(study / "matched-distance.json", dict(
        scope="Secondary check added after primary comparison. Same scenes; no target-dependent tuning.",
        configurations=configurations, summary=summary))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("outputs/wall-topology-study/frozen.json"))
    parser.add_argument("--study", type=Path, default=Path("outputs/wall-rollout-study"))
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        for row in run(args.reference, args.study):
            print(row)


if __name__ == "__main__":
    main()
