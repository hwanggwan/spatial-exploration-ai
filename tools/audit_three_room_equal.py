"""Secondary coverage-first equal quota on the same three-room test fields."""

import csv
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from spatial_exploration.basis_bayes import BayesianBasisModel
from spatial_exploration.three_room_study import (
    BUDGETS, NOISE, REGIMES, STARTS, TEST_LAYOUTS, TEST_SEED,
    design_for, dump, fields, quota_route,
)


def run(root):
    spec = json.loads((root / "frozen.json").read_text())["ridge"]
    with (root / "metrics.csv").open() as stream:
        reference = {(r["budget"], r["regime"], int(r["layout"]), int(r["seed"])): float(r["rmse"])
                     for r in csv.DictReader(stream) if r["method"] == "priority"}
    rows, plans = [], []
    for label, doors in enumerate(TEST_LAYOUTS):
        design = design_for(doors)
        model = BayesianBasisModel(design.features, NOISE, spec["prior"],
                                   intercept_scale=np.sqrt((design.features.shape[1] + 1) * spec["intercept"]))
        routes = {(s, b): quota_route(design, model, s, cap, "equal_coverage")
                  for s in STARTS for b, cap in BUDGETS.items()}
        for (s, b), route in routes.items():
            sites = list(route.sites)
            assert not design.heldout[sites].any() and route.distances[-1] <= BUDGETS[b] + 1e-12
            assert sites[0] == s and len(sites) == len(set(sites)) <= 16
            counts = np.bincount(design.layout.room[sites], minlength=3).tolist()
            plans.append(dict(layout=label, start=s, budget=b, sites=sites, counts=counts,
                              distance=route.distances[-1]))
        for meta, truth, sensor in fields(design, label, 128, TEST_SEED):
            for b in BUDGETS:
                route = routes[meta["start"], b]
                sites = list(route.sites)
                mean = model.condition(sites, sensor[sites]).mean
                residual = (mean - truth)[design.heldout]
                rows.append(dict(**meta, budget=b, rmse=float(np.sqrt(np.mean(residual**2))),
                                 distance=route.distances[-1], count=len(sites),
                                 priority_rmse=reference[b, meta["regime"], label, meta["seed"]]))
    results = []
    rng = np.random.default_rng(191002)
    for regime in REGIMES:
        for b in BUDGETS:
            subset = [r for r in rows if (r["regime"], r["budget"]) == (regime, b)]
            strata = {}
            delta = np.array([r["priority_rmse"] - r["rmse"] for r in subset])
            for i, r in enumerate(subset):
                strata.setdefault((r["layout"], r["start"], r["kappa"]), []).append(i)
            draws = np.concatenate([rng.choice(delta[idx], (6000, len(idx))) for idx in strata.values()],
                                   axis=1).mean(axis=1)
            results.append(dict(regime=regime, budget=b, priority_minus_coverage_equal=float(delta.mean()),
                                interval=np.quantile(draws, [.025, .975]).tolist(),
                                **{k: float(np.mean([r[k] for r in subset]))
                                   for k in ("rmse", "distance", "count")}))
    result = dict(scope="Post-run sensitivity, new tie priority only; same test data; no tuning",
                  results=results, plans=plans)
    dump(root / "coverage-first-audit.json", result)
    return results


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        print(json.dumps(run(Path("outputs/three-room-study")), indent=2))
