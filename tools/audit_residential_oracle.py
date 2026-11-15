"""Use the simulator covariance for routing, then score the common world mapper."""

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits

from audit_residential_coverage import ARMS, key
from audit_residential_routes import (
    AREAS, CONDUCTANCES, LayoutCase, SEED, SIZE, STARTS, design_for, field_cases, field_covariances, floor_plan,
)
from spatial_exploration.baseline import write_csv
from spatial_exploration.basis_bayes import BayesianBasisModel
from spatial_exploration.sensing_world import Belief, kernel_basis
from spatial_exploration.sensing_world_generalization import EDGES
from spatial_exploration.sensing_world_study import CAP, NOISE, fixed_route_rmse, run_episode
from spatial_exploration.wall_rollout_study import plan_route


FAMILIES = ("corridor", "loop")
AREA = min(AREAS)
METRICS = ("rmse", "edges", "readings", "rooms", "plan_ms")


def evaluate(model, draws, archived):
    rows, routes, diagnostics = [], [], []
    limit = EDGES / (SIZE - 1)
    for family in FAMILIES:
        for variant in range(2):
            case = LayoutCase(family, variant)
            layout = floor_plan(case)
            design = design_for(layout, SEED + variant)
            weights = np.asarray(model["world_weights"]["world_4"])
            world = Belief.prior(np.einsum("mb,bij->mij", weights, kernel_basis(layout)), NOISE)
            spec = model["gp"]
            intercept = np.sqrt((design.features.shape[1]+1) * spec["intercept"])
            gp = BayesianBasisModel(design.features, NOISE, spec["prior"], intercept_scale=intercept)
            gp_routes = {start: plan_route(design, gp, start, CAP, limit, "rollout") for start in STARTS}

            covariance_bank = {regime: field_covariances(layout, p) for regime, p in CONDUCTANCES.items()}
            oracle_routes = {}
            for regime, models in covariance_bank.items():
                scale, cov = models[AREA]
                oracle = SimpleNamespace(prior_covariance=cov, noise_std=NOISE)
                for start in STARTS:
                    route = plan_route(design, oracle, start, CAP, limit, "rollout")
                    oracle_routes[regime, start] = route
                    routes.append(dict(family=family, variant=variant, regime=regime, start=start,
                        scale=scale, sites=route.sites, distances=route.distances,
                        final_mean_oracle_variance=route.final_variance, planning_ms=route.planning_ms))

            for row, truth, sensor in field_cases(layout, case, draws, diagnostics, covariances=covariance_bank):
                if row["area"] != AREA:
                    continue
                source = archived[key(row)]
                # Replay the GP route to verify identity with the archived sensor fields.
                start = row["start"]
                same_gp = fixed_route_rmse(world, gp_routes[start].sites, sensor.__getitem__, truth, design.heldout)
                np.testing.assert_allclose(same_gp, source["gp"]["rmse"], atol=1e-10, rtol=0)
                for arm in ARMS:
                    rows.append(dict(**row, arm=arm, **{m: source[arm][m] for m in METRICS},
                                     world_forecast_coverage=None, world_forecast_width=None))
                route = oracle_routes[row["regime"], start]
                sites, walk, _, obs, states, forecasts, _ = run_episode(
                    world, design, start, EDGES, sensor.__getitem__, fixed=route.sites)
                mean, sd = np.asarray(forecasts).T
                rmse = np.sqrt(np.mean((states[-1][0][design.heldout] - truth[design.heldout])**2))
                rooms = set(layout.room[sites]) - ({2} if family == "corridor" else set())
                rows.append(dict(**row, arm="oracle", rmse=float(rmse), edges=len(walk)-1, readings=len(sites),
                    rooms=len(rooms), plan_ms=route.planning_ms,
                    world_forecast_coverage=float(np.mean(abs(np.asarray(obs[1:]) - mean) <= 1.96*sd)),
                    world_forecast_width=float(np.mean(3.92*sd))))
    return rows, routes, diagnostics


def summarize(rows):
    rng = np.random.default_rng(SEED + 600)
    result = {}
    for family in FAMILIES:
        fields = {}
        for row in rows:
            if row["family"] == family:
                fields.setdefault(key(row), {})[row["arm"]] = row
        groups = list(fields.values())
        strata = {}
        arms = (*ARMS, "oracle")
        for i, group in enumerate(groups):
            if set(group) != set(arms):
                raise ValueError("Missing rollout arm")
            r = group["oracle"]
            strata.setdefault((r["variant"], r["regime"], r["start"]), []).append(i)
        bootstrap = np.concatenate([rng.choice(idx, (6000, len(idx))) for idx in strata.values()], axis=1)
        scores = {arm: np.array([g[arm]["rmse"] for g in groups]) for arm in arms}
        contrasts = {}
        for arm in ARMS:
            delta = scores["oracle"] - scores[arm]
            contrasts[arm] = dict(mean=float(delta.mean()),
                ci95=np.quantile(delta[bootstrap].mean(axis=1), [.025, .975]).tolist())
        means = {arm: {m: float(np.mean([g[arm][m] for g in groups])) for m in METRICS} for arm in arms}
        result[family] = dict(fields=len(groups), means=means, oracle_minus=contrasts,
            world_forecast={m: float(np.mean([g["oracle"]["world_forecast_"+m] for g in groups]))
                            for m in ("coverage", "width")})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("outputs/sensing-world-residential-coverage"))
    parser.add_argument("--output", type=Path, default=Path("outputs/sensing-world-residential-oracle"))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use an empty output directory.")
    config = json.loads((args.base/"protocol.json").read_text())
    model = json.loads(Path(config["model"]).read_text())
    with (args.base/"metrics.csv").open() as stream:
        archived = {}
        for row in csv.DictReader(stream):
            if row["family"] in FAMILIES and float(row["area"]) == AREA:
                archived.setdefault(key(row), {})[row["arm"]] = {m: float(row[m]) for m in METRICS}
    args.output.mkdir(parents=True, exist_ok=True)
    settings = dict(reference=str(args.base), families=FAMILIES, area=AREA, conductances=CONDUCTANCES,
                    starts=STARTS, cap=CAP, edges=EDGES, draws_per_start=config["draws_per_start"],
                    planning_covariance="Exact noise-free simulator covariance plus the shared sensor noise at observation",
                    scoring_model="Original unrestricted four-mode world model")
    (args.output/"protocol.json").write_text(json.dumps(settings, indent=2)+"\n")
    with threadpool_limits(limits=1):
        rows, routes, diagnostics = evaluate(model, config["draws_per_start"], archived)
    if len(rows) != len(archived) * (len(ARMS)+1) or len({(*key(r), r["arm"]) for r in rows}) != len(rows):
        raise RuntimeError("Oracle cases do not match the archived benchmark")
    write_csv(args.output/"metrics.csv", rows)
    write_csv(args.output/"range.csv", [r for r in diagnostics if r["target"] == AREA])
    (args.output/"routes.json").write_text(json.dumps(routes, indent=2)+"\n")
    summary = summarize(rows)
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    for family, entry in summary.items():
        print(family, entry)


if __name__ == "__main__":
    main()
