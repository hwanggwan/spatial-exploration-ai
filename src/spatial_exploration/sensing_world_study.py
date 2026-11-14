"""Train a sensing world model and compare its closed-loop routes with a GP."""

import argparse
import csv
from collections import defaultdict
import json
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from .baseline import write_csv
from .basis_bayes import BayesianBasisModel
from .sensing_world import (
    Belief, CONDUCTANCES, HEAT_LENGTHS, choose_rollout_site, choose_site, fit_kernel_weights, kernel_basis, rollout_samples,
)
from .three_room_study import BUDGETS, CAP, KAPPAS, NOISE, REGIMES, STARTS, TEST_LAYOUTS, TEST_SEED, design_for, fields


TRAIN_LAYOUTS = ((1, 10), (10, 1), (3, 8), (9, 4), (5, 6), (6, 5))
TRAIN_SEED = 2810000
TRAIN_FIELDS = 512
MODES = tuple((regime, kappa) for regime in REGIMES for kappa in KAPPAS)
ARMS = ("learned_world", "gp_two_step", "gp_rollout")
SHORTLIST = 5


def learn():
    training = []
    for label, doors in enumerate(TRAIN_LAYOUTS):
        design = design_for(doors)
        readings = defaultdict(list)
        for meta, _, sensor in fields(design, label, TRAIN_FIELDS, TRAIN_SEED):
            readings[meta["regime"], meta["kappa"]].append(sensor)
        training.append((kernel_basis(design.layout), [np.stack(readings[m]) for m in MODES]))
    return fit_kernel_weights(training, NOISE)


def fixed_route_rmse(prior, sites, read, truth, holdout):
    belief = prior
    for site in sites:
        belief = belief.observe(site, read(site))
    mean, _ = belief.moments()
    return float(np.sqrt(np.mean((mean[holdout] - truth[holdout])**2)))


def run_episode(prior, design, start, edges, read, fixed=None, *, planner=choose_site):
    """The planner sees only acquired sensor readings and the known movement graph."""
    hops = np.rint(design.layout.distances * 11).astype(int)
    allowed = ~design.heldout
    states, forecasts, sites, walk, endpoints, readings = [], [], [], [start], [], []
    elapsed, next_site = 0., start
    belief = prior
    while next_site is not None:
        reading = float(read(next_site))
        belief = belief.observe(next_site, reading)
        sites.append(next_site)
        readings.append(reading)
        endpoints.append(len(walk))
        mean, covariance = belief.moments()
        states.append((mean, np.sqrt(np.maximum(np.diag(covariance), 0))))
        before = perf_counter()
        if fixed is None:
            next_site = planner(belief, hops, allowed, sites, edges - (len(walk) - 1), CAP - len(sites))
        else:
            next_site = fixed[len(sites)] if len(sites) < len(fixed) else None
        elapsed += perf_counter() - before
        if next_site is not None:
            # Record the forecast before the sensor is called for this site.
            center, std = belief.reading_moments(next_site)
            forecasts.append((center, std))
            leg = design.layout.walk(sites[-1], next_site)
            if not all(design.layout.movement[u, v] for u, v in zip(leg, leg[1:])):
                raise RuntimeError("Illegal route edge")
            walk.extend(leg[1:])
    if len(sites) > CAP or len(walk) - 1 > edges or len(set(sites)) != len(sites) or design.heldout[sites].any():
        raise RuntimeError("Invalid measurement route")
    return sites, walk, endpoints, readings, states, forecasts, elapsed


def forecast_maps(prior, sites, readings, forecasts):
    """Archive three counterfactual maps per chosen action for the replay."""
    scenarios = []
    belief = prior
    for i, (center, sd) in enumerate(forecasts):
        belief = belief.observe(sites[i], readings[i])
        alternatives = []
        for value in (center - sd, center, center + sd):
            mean, covariance = belief.observe(sites[i + 1], value).moments()
            alternatives.append(dict(reading=round(value, 3),
                mean=np.rint(mean * 1000).astype(int).tolist(),
                std=np.rint(np.sqrt(np.maximum(np.diag(covariance), 0)) * 1000).astype(int).tolist()))
        scenarios.append(alternatives)
    return scenarios


def replay_plan(episode, truth, held, prior, *, include_forecasts=True):
    sites, walk, endpoints, readings, states, forecasts, _ = episode
    plan = dict(sites=sites, walk=walk, endpoints=endpoints, readings=[round(y, 3) for y in readings],
        mean=[np.rint(mean * 1000).astype(int).tolist() for mean, _ in states],
        std=[np.rint(std * 1000).astype(int).tolist() for _, std in states],
        rmse=[float(np.sqrt(np.mean((mean[held] - truth[held])**2))) for mean, _ in states])
    if include_forecasts:
        plan["forecasts"] = [[round(v, 3) for v in pair] for pair in forecasts]
        plan["futures"] = forecast_maps(prior, sites, readings, forecasts)
    return plan


def evaluate(base, weights, count, *, full_rollout=False):
    protocol = json.loads((base / "frozen.json").read_text())
    if (tuple(map(tuple, protocol["test_layouts"])) != TEST_LAYOUTS or protocol["test_seed"] != TEST_SEED
            or protocol["travel_caps"] != BUDGETS or protocol["cap"] != CAP
            or protocol["sensor_noise"] != NOISE or count > protocol["test_fields_per_layout"]):
        raise ValueError("The three-room archive does not match the study")
    fixed = defaultdict(list)
    expected = {}
    with (base / "acquisitions.csv").open() as stream:
        for row in csv.DictReader(stream):
            if row["method"] == "rollout":
                fixed[int(row["layout"]), int(row["seed"]), row["regime"], row["budget"]].append(
                    (int(row["site"]), float(row["value"])))
    with (base / "metrics.csv").open() as stream:
        for row in csv.DictReader(stream):
            if row["method"] == "rollout":
                expected[int(row["layout"]), int(row["seed"]), row["regime"], row["budget"]] = float(row["rmse"])
    archived_routes = json.loads((base / "routes.json").read_text())
    rollout_ms = {(r["layout"], r["start"], r["budget"]): r["planning_ms"]
                  for r in archived_routes if r["method"] == "rollout"}
    with np.load(base / "evaluation-only.npz") as source:
        archive = dict(source)
    rows, replay = [], []
    spec = protocol["ridge"]
    two_step = partial(choose_site, shortlist=SHORTLIST)
    planners = {arm: two_step for arm in ARMS}
    if full_rollout:
        planners["learned_rollout"] = partial(choose_rollout_site, innovations=rollout_samples(CAP))

    for label, doors in enumerate(TEST_LAYOUTS):
        design = design_for(doors)
        ridge = BayesianBasisModel(design.features, NOISE, spec["prior"],
            intercept_scale=np.sqrt((design.features.shape[1] + 1) * spec["intercept"]))
        gp = Belief.prior(ridge.prior_covariance[None], NOISE)
        learned = Belief.prior(np.einsum("mb,bij->mij", weights, kernel_basis(design.layout)), NOISE)
        held = design.heldout
        archived_fields = {(int(s), r): t for s, r, t in zip(
            archive[f"layout{label}_seeds"], archive[f"layout{label}_regimes"], archive[f"layout{label}_truth"])}
        for meta, truth, sensor in fields(design, label, count, TEST_SEED):
            np.testing.assert_allclose(truth, archived_fields[meta["seed"], meta["regime"]], rtol=0, atol=1e-10)
            for budget, limit in BUDGETS.items():
                key = label, meta["seed"], meta["regime"], budget
                route_sites, actual_values = zip(*fixed[key])
                np.testing.assert_allclose(sensor[list(route_sites)], actual_values, rtol=0, atol=1e-10)
                for arm, planner in planners.items():
                    is_learned = arm.startswith("learned")
                    prior = learned if is_learned else gp
                    route = route_sites if arm == "gp_rollout" else None
                    sites, walk, endpoints, readings, states, forecasts, seconds = run_episode(
                        prior, design, meta["start"], round(limit * 11), sensor.__getitem__, fixed=route, planner=planner)
                    rmses = [float(np.sqrt(np.mean((mean[held] - truth[held])**2))) for mean, _ in states]
                    if arm == "gp_rollout":
                        np.testing.assert_allclose(rmses[-1], expected[key], rtol=0, atol=1e-10)
                    mu, sd = np.asarray(forecasts).reshape(-1, 2).T
                    errors = np.asarray(readings[1:]) - mu
                    rows.append(dict(**meta, budget=budget, arm=arm, rmse=rmses[-1], readings=len(sites),
                        edges=len(walk) - 1, rooms=len(np.unique(design.layout.room[sites])),
                        plan_ms=(rollout_ms[label, meta["start"], budget] if arm == "gp_rollout" else seconds * 1000),
                        forecast_gaussian_nll=float(np.mean(.5 * (np.log(2*np.pi*sd**2) + (errors/sd)**2))),
                        forecast_coverage=float(np.mean(np.abs(errors) <= 1.96 * sd)),
                        forecast_width=float(np.mean(3.92 * sd)),
                        learned_rmse_same_route=(rmses[-1] if is_learned else
                            fixed_route_rmse(learned, sites, sensor.__getitem__, truth, held)),
                        gp_rmse_same_route=(rmses[-1] if not is_learned else
                            fixed_route_rmse(gp, sites, sensor.__getitem__, truth, held))))
                    if is_learned and meta["seed"] < TEST_SEED + 1000 * label + len(STARTS):
                        if arm == "learned_world" or budget == "loose":
                            episode = sites, walk, endpoints, readings, states, forecasts, seconds
                            replay.append(dict(**meta, budget=budget, arm=arm,
                                plan=replay_plan(episode, truth, held, prior, include_forecasts=arm == "learned_world")))
        print(f"Evaluated both budgets on layout {label + 1}", flush=True)
    return rows, replay


def summarize(rows, *, full_rollout=False):
    arms = (*ARMS, "learned_rollout") if full_rollout else ARMS
    focus = "learned_rollout" if full_rollout else "learned_world"
    rng = np.random.default_rng(282001)
    results = []
    for regime in REGIMES:
        for budget in BUDGETS:
            groups = {arm: {} for arm in arms}
            for r in rows:
                if r["regime"] == regime and r["budget"] == budget:
                    groups[r["arm"]][r["layout"], r["seed"]] = r
            keys = sorted(groups[focus])
            if not keys or any(set(group) != set(keys) for group in groups.values()):
                raise ValueError("Each policy must evaluate the same fields")
            strata = defaultdict(list)
            for i, key in enumerate(keys):
                r = groups[focus][key]
                strata[r["layout"], r["start"], r["kappa"]].append(i)
            boot = np.concatenate([rng.choice(g, (6000, len(g))) for g in strata.values()], axis=1)
            model_rmse = np.array([groups[focus][k]["rmse"] for k in keys])
            contrasts = {}
            for arm in arms:
                if arm == focus:
                    continue
                delta = model_rmse - [groups[arm][k]["rmse"] for k in keys]
                contrasts[arm] = dict(mean=float(delta.mean()), interval=np.quantile(delta[boot].mean(axis=1), [.025,.975]).tolist())
            metrics = ("rmse", "readings", "edges", "rooms", "plan_ms", "forecast_gaussian_nll", "forecast_coverage",
                       "forecast_width", "learned_rmse_same_route", "gp_rmse_same_route")
            means = {arm: {metric: float(np.mean([group[k][metric] for k in keys])) for metric in metrics}
                     for arm, group in groups.items()}
            results.append(dict(regime=regime, budget=budget, fields=len(keys), means=means, contrasts=contrasts))
    return results


def run(base, output, count, *, full_rollout=False):
    if count < 8 or count % 8:
        raise ValueError("Use a positive field count divisible by eight.")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Use a fresh output directory.")
    with threadpool_limits(limits=1):
        weights = learn()
        frozen = dict(train_seed=TRAIN_SEED, train_fields_per_layout=TRAIN_FIELDS, training_layouts=TRAIN_LAYOUTS,
                      modes=MODES, weights=weights.tolist(), conductances=CONDUCTANCES, lengths=HEAT_LENGTHS,
                      test_layouts=TEST_LAYOUTS, test_seed=TEST_SEED, test_fields_per_layout=count,
                      shortlist=SHORTLIST, cap=CAP, budgets=BUDGETS)
        if full_rollout:
            frozen["rollout"] = dict(tail="Greedy Gaussian moment match, all feasible first stops",
                terminal="Expected mixture variance, analytic within modes plus sampled between modes",
                samples_per_mode=32, sample_seed=283001)
        (output / "frozen.json").write_text(json.dumps(frozen, indent=2) + "\n")
        print("Froze the learned world model before evaluation", flush=True)
        rows, replay = evaluate(base, weights, count, full_rollout=full_rollout)
    write_csv(output / "metrics.csv", rows)
    (output / "replay.json").write_text(json.dumps(replay, separators=(",", ":")))
    summary = summarize(rows, full_rollout=full_rollout)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("outputs/three-room-study"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--full-rollout", action="store_true",
                        help="Compare learned full-budget rollout with both original planners.")
    parser.add_argument("--fields-per-layout", type=int, default=32, help="Per dependence regime.")
    args = parser.parse_args()
    output = args.output or Path("outputs") / ("sensing-world-rollout" if args.full_rollout else "sensing-world-study")
    for row in run(args.base, output, args.fields_per_layout, full_rollout=args.full_rollout):
        print(row["regime"], row["budget"], {a: round(m["rmse"], 4) for a, m in row["means"].items()})


if __name__ == "__main__":
    main()
