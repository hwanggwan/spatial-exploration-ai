"""Test coverage and known-range rollout on the paired residential benchmark."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from functools import partial
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from audit_residential_routes import (
    AREAS, FAMILIES, SEED, SIZE, STARTS, LayoutCase, design_for, field_cases, floor_plan,
)
from spatial_exploration.baseline import write_csv
from spatial_exploration.basis_bayes import BayesianBasisModel
from spatial_exploration.sensing_world import Belief, choose_rollout_site, kernel_basis, rollout_samples
from spatial_exploration.sensing_world_generalization import EDGES, MODES
from spatial_exploration.sensing_world_study import CAP, NOISE, fixed_route_rmse, run_episode
from spatial_exploration.wall_rollout_study import plan_route


ARMS = ("gp", "world", "coverage", "known_range")
COVERAGE_WEIGHT = .025
KEYS = ("family", "variant", "regime", "area", "start", "draw")


def key(row):
    return tuple(str(row[k]) for k in KEYS)


def restricted_priors(covariances):
    _, scales = zip(*MODES)
    return {area: Belief.prior(covariances[np.asarray(scales) == scale], NOISE)
            for area, scale in zip(AREAS, (max(scales), min(scales)))}


def evaluate(case, spec, weights, draws, archived):
    with threadpool_limits(limits=1):
        return evaluate_layout(case, spec, weights, draws, archived)


def evaluate_layout(case, spec, weights, draws, archived):
    layout = floor_plan(case)
    design = design_for(layout, SEED + case.variant)
    intercept = np.sqrt((design.features.shape[1]+1) * spec["intercept"])
    ridge = BayesianBasisModel(design.features, NOISE, spec["prior"], intercept_scale=intercept)
    covariances = np.einsum("mb,bij->mij", weights, kernel_basis(layout))
    world = Belief.prior(covariances, NOISE)
    known = restricted_priors(covariances)
    fixed = {s: plan_route(design, ridge, s, CAP, EDGES/(SIZE-1), "rollout") for s in STARTS}
    planner = partial(choose_rollout_site, innovations=rollout_samples(CAP))
    coverage_planner = partial(planner, coverage_weight=COVERAGE_WEIGHT)
    room_exclusions = {2} if case.family == "corridor" else set()
    rows, diagnostics = [], []
    for row, truth, sensor in field_cases(layout, case, draws, diagnostics):
        start = row["start"]
        source = archived[key(row)]
        route = fixed[start]
        gp_sites = route.sites
        gp_rmse = np.sqrt(np.mean((ridge.condition(gp_sites, sensor[list(gp_sites)]).mean[design.heldout]
                                 - truth[design.heldout])**2))
        gp_common = fixed_route_rmse(world, gp_sites, sensor.__getitem__, truth, design.heldout)
        np.testing.assert_allclose([gp_rmse, gp_common], [float(source["gp_rmse"]), float(source["world_on_gp"])],
                                   rtol=0, atol=1e-10)
        for arm, rmse in (("gp", gp_common), ("world", float(source["world_rmse"]))):
            metrics = {m: float(source[f"{arm}_{m}"]) for m in ("edges", "readings", "rooms", "plan_ms", "coverage", "width")}
            rows.append(dict(**row, arm=arm, rmse=rmse, **metrics))
        for arm, prior, policy in (("coverage", world, coverage_planner), ("known_range", known[row["area"]], planner)):
            episode = run_episode(prior, design, start, EDGES, sensor.__getitem__, planner=policy)
            sites, walk, _, readings, _, forecasts, seconds = episode
            mean, sd = np.asarray(forecasts).T
            rmse = fixed_route_rmse(world, sites, sensor.__getitem__, truth, design.heldout)
            rows.append(dict(**row, arm=arm, rmse=rmse, edges=len(walk)-1, readings=len(sites),
                rooms=len(set(layout.room[sites]) - room_exclusions), plan_ms=seconds*1000,
                coverage=float(np.mean(abs(readings[1:] - mean) <= 1.96*sd)), width=float(np.mean(3.92*sd))))
    return rows


def summarize(rows):
    rng = np.random.default_rng(SEED + 500)
    summary = []
    for family in FAMILIES:
        for area in AREAS:
            by_field = {}
            for row in rows:
                if row["family"] == family and row["area"] == area:
                    by_field.setdefault(key(row), {})[row["arm"]] = row
            groups = list(by_field.values())
            strata = {}
            for i, group in enumerate(groups):
                if set(group) != set(ARMS):
                    raise ValueError("Unmatched rollout arms")
                row = group["world"]
                strata.setdefault((row["variant"], row["regime"], row["start"]), []).append(i)
            boot = np.concatenate([rng.choice(i, (6000, len(i))) for i in strata.values()], axis=1)
            metrics = ("rmse", "edges", "readings", "rooms", "coverage", "width", "plan_ms")
            means = {arm: {m: float(np.mean([group[arm][m] for group in groups])) for m in metrics} for arm in ARMS}
            contrasts = {}
            for arm in ("gp", "coverage", "known_range"):
                diff = np.array([g[arm]["rmse"]-g["world"]["rmse"] for g in groups])
                contrasts[arm] = dict(mean=float(diff.mean()), ci95=np.quantile(diff[boot].mean(axis=1), [.025, .975]).tolist())
            summary.append(dict(family=family, area=area, fields=len(groups), means=means, contrasts_vs_world=contrasts))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("outputs/sensing-world-residential"))
    parser.add_argument("--output", type=Path, default=Path("outputs/sensing-world-residential-coverage"))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("Use at least one worker.")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use an empty output directory.")
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((args.base/"protocol.json").read_text())
    model = json.loads(Path(protocol["model"]).read_text())
    with (args.base/"metrics.csv").open() as stream:
        source = {key(r): r for r in csv.DictReader(stream)}
    settings = dict(**protocol, coverage_weight_per_edge=COVERAGE_WEIGHT,
                    scoring_model="unrestricted original four-mode world model",
                    known_range="Restrict planning to the corresponding two trained modes")
    (args.output/"protocol.json").write_text(json.dumps(settings, indent=2)+"\n")
    cases = [LayoutCase(f, v) for f in FAMILIES for v in range(2)]
    worker = partial(evaluate, spec=model["gp"], weights=np.asarray(model["world_weights"]["world_4"]),
                     draws=protocol["draws_per_start"])
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, case, archived={k: v for k, v in source.items()
                                                      if k[:2] == (case.family, str(case.variant))}): case for case in cases}
        for future in as_completed(futures):
            case = futures[future]
            layout_rows = future.result()
            rows.extend(layout_rows)
            print(f"Verified {case.family} layout {case.variant + 1}: {len(layout_rows)//len(ARMS)} paired fields", flush=True)
    if len(rows) != len(ARMS)*len(source) or len({(*key(r), r["arm"]) for r in rows}) != len(rows):
        raise RuntimeError("Benchmark records do not align with the archived fields")
    rows.sort(key=lambda r: (*key(r), r["arm"]))
    write_csv(args.output/"metrics.csv", rows)
    result = summarize(rows)
    (args.output/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    for row in result:
        print(row["family"], row["area"], {a: round(m["rmse"], 4) for a,m in row["means"].items()})


if __name__ == "__main__":
    main()
