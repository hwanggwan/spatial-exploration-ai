"""Separate simulation shifts while holding GP rollout observations fixed across mappers."""

import argparse
import csv
from itertools import product
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from threadpoolctl import threadpool_limits

from spatial_exploration.barrier_gp import graph_spectrum, unit_covariance
from spatial_exploration.baseline import write_csv
from spatial_exploration.basis_bayes import BayesianBasisModel
from spatial_exploration.sensing_world import Belief, kernel_basis
from spatial_exploration.sensing_world_generalization import (
    ARMS, CAP, EDGES, MODES, NOISE, STARTS, field_covariance, make_design, plan_route, wall_layout,
)
from spatial_exploration.sensing_world_study import fixed_route_rmse


# Match the previous bundled shift and add a control for the source's background rescaling.
FACTORS = ("wall", "rough", "source")
CONDITIONS = {"+".join(k for k, active in zip(FACTORS, flags) if active) or "baseline": flags
              for flags in product((False, True), repeat=3)}
EXTRA = ("source_only", "background_only", "wall+rough+local_only")


def factors_for(layout):
    factors = {}
    for i, (regime, kappa) in enumerate(MODES):
        barrier = regime == "barrier"
        permeability = (.02, .25) if barrier else (1., .45)
        center = (.10 if i % 2 else .90, .22 if i < 2 else .78)
        width = .14
        for condition, (wall, rough, source) in CONDITIONS.items():
            cov = field_covariance(layout, permeability[wall], kappa, power=1.45 if rough else 2.,
                                   source=(center, width) if source else None)
            factors[condition, i] = np.linalg.cholesky(cov)
        base = field_covariance(layout, permeability[0], kappa)
        bump = np.exp(-np.sum((layout.points - center)**2, axis=1) / (2 * width**2))
        local = .40 * np.outer(bump, bump)
        factors["source_only", i] = np.linalg.cholesky(base + local)
        shifted = field_covariance(layout, permeability[1], kappa, power=1.45)
        factors["wall+rough+local_only", i] = np.linalg.cholesky(shifted + local)
        factors["background_only", i] = np.linalg.cholesky(.76 * base)
    return factors


RANGE_CONDITIONS = ("baseline", "rough", "rough_matched_area", "rough_matched_lag5")


def within_room_correlogram(layout):
    """Return a function that averages correlation by within-room walking lag."""
    hops = np.rint(layout.distances * (layout.size - 1)).astype(int)
    same_room = layout.room[:, None] == layout.room[None, :]
    lags = np.arange(1, 9)
    pairs = [same_room & (hops == lag) for lag in lags]
    if any(not mask.any() for mask in pairs):
        raise ValueError("The floor plan must support all eight within-room lags")

    def correlogram(covariance):
        return np.array([1., *[covariance[mask].mean() for mask in pairs]])

    return correlogram


def correlation_area(curve):
    return float((curve[0] + curve[-1]) / 2 + curve[1:-1].sum())


def match_scale(covariance_at_scale, correlogram, measure, target):
    scale = brentq(lambda k: measure(correlogram(covariance_at_scale(k))) - target,
                   .01, 4., xtol=1e-12)
    matched = covariance_at_scale(scale)
    np.testing.assert_allclose(measure(correlogram(matched)), target, atol=1e-10, rtol=0)
    return scale, matched


def range_matched_factors(layout):
    """Match within-room spatial correlation at a prespecified range summary."""
    correlogram = within_room_correlogram(layout)
    definitions = (("rough_matched_area", correlation_area), ("rough_matched_lag5", lambda curve: curve[5]))
    factors, diagnostics = {}, []
    for mode, (regime, kappa) in enumerate(MODES):
        permeability = .02 if regime == "barrier" else 1.
        spectrum, vectors = graph_spectrum(layout, permeability)

        def covariance(scale, power):
            return unit_covariance(vectors, (scale**2 + spectrum)**-power)

        base = covariance(kappa, 2.)
        baseline_curve = correlogram(base)
        alternatives = [("baseline", kappa, base), ("rough", kappa, covariance(kappa, 1.45))]
        for name, measure in definitions:
            target = measure(baseline_curve)
            scale, matched = match_scale(lambda k: covariance(k, 1.45), correlogram, measure, target)
            alternatives.append((name, scale, matched))
        for name, scale, cov in alternatives:
            np.testing.assert_allclose(np.diag(cov), 1., rtol=0, atol=1e-12)
            curve = correlogram(cov)
            factors[name, mode] = np.linalg.cholesky(cov)
            diagnostics.append(dict(condition=name, mode=mode, permeability=permeability, kappa=scale,
                range_area=correlation_area(curve), lag5=float(curve[5]), lag1_semivariance=float(1 - curve[1]),
                correlations=curve.tolist()))
    return factors, diagnostics


def evaluate(config, count, archived, *, match_range=False):
    rows, diagnostics = [], []
    conditions = RANGE_CONDITIONS if match_range else (*CONDITIONS, *EXTRA)
    seed = config["test_seed"]
    for label, spec in enumerate(config["test_geometry"]):
        layout = wall_layout(spec)
        design = make_design(layout, seed + 100 * label)
        gp_config = config["gp"]
        intercept = np.sqrt((design.features.shape[1] + 1) * gp_config["intercept"])
        ridge = BayesianBasisModel(design.features, NOISE, gp_config["prior"], intercept_scale=intercept)
        priors = [Belief.prior(ridge.prior_covariance[None], NOISE)]
        basis = kernel_basis(layout)
        for arm in ARMS[1:]:
            cov = np.einsum("mb,bij->mij", config["weights"][arm], basis)
            priors.append(Belief.prior(cov, NOISE))
        routes = {start: plan_route(design, ridge, start, CAP, EDGES / (layout.size - 1), "rollout")
                  for start in STARTS}
        if match_range:
            factors, layout_diagnostics = range_matched_factors(layout)
            diagnostics.extend(dict(layout=label, **row) for row in layout_diagnostics)
        else:
            factors = factors_for(layout)
        rng = np.random.default_rng(seed + label)
        for case in range(count):
            mode = (case // len(STARTS)) % len(MODES)
            start = STARTS[case % len(STARTS)]
            z, noise = rng.standard_normal((2, len(design.heldout)))
            route = routes[start]
            for condition in conditions:
                truth = factors[condition, mode] @ z
                sensor = truth + NOISE * noise
                scores = [fixed_route_rmse(prior, route.sites, sensor.__getitem__, truth, design.heldout)
                          for prior in priors]
                previous = {"baseline": "benchmark", "wall+rough+source": "shifted"}.get(condition)
                if (label, case, previous) in archived:
                    np.testing.assert_allclose(scores[0], archived[label, case, previous], rtol=0, atol=1e-10)
                rows.append(dict(layout=label, case=case, start=start, mode=mode, condition=condition,
                                 readings=len(route.sites), edges=round(route.distances[-1] * (layout.size - 1)),
                                 **dict(zip(ARMS, scores))))
    return rows, diagnostics


def summarize(rows, *, match_range=False):
    conditions = RANGE_CONDITIONS if match_range else (*CONDITIONS, *EXTRA)
    by_condition = {name: [r for r in rows if r["condition"] == name] for name in conditions}
    base = by_condition["baseline"]
    strata = {}
    for i, row in enumerate(base):
        strata.setdefault((row["layout"], row["start"], row["mode"]), []).append(i)
    rng = np.random.default_rng(295181)
    indexes = np.concatenate([rng.choice(indices, (6000, len(indices))) for indices in strata.values()], axis=1)
    scores = {name: np.array([[r[k] for k in ARMS] for r in group]) for name, group in by_condition.items()}
    gaps = {name: v[:, :1] - v[:, 1:] for name, v in scores.items()}

    def estimate(values):
        return {"mean": values.mean(axis=0).tolist(),
                "ci95": np.quantile(values[indexes].mean(axis=1), [.025, .975], axis=0).T.tolist()}

    result = dict(fields=len(base), arms=ARMS, conditions={
        name: dict(rmse=scores[name].mean(axis=0).tolist(), advantage=estimate(gaps[name]),
                   advantage_lost=estimate(gaps["baseline"] - gaps[name])) for name in by_condition})
    if match_range:
        result["advantage_restored"] = {name: estimate(gaps[name] - gaps["rough"]) for name in RANGE_CONDITIONS[2:]}
    else:
        full = gaps["wall+rough+source"]
        result["restore"] = {factor: estimate(gaps["+".join(k for k in FACTORS if k != factor)] - full)
                             for factor in FACTORS}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("outputs/sensing-world-generalization"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--match-range", action="store_true", help="Match the within-room range before scoring rough fields.")
    parser.add_argument("--fields-per-layout", type=int, default=128)
    args = parser.parse_args()
    if args.fields_per_layout < 32 or args.fields_per_layout % 16:
        parser.error("Use at least 32 fields per layout in multiples of 16.")
    config = json.loads((args.reference / "frozen.json").read_text())
    with (args.reference / "metrics.csv").open() as stream:
        archived = {(int(r["layout"]), int(r["case"]), r["field"]): float(r["rmse"])
                    for r in csv.DictReader(stream) if r["geometry"] == "unseen" and r["arm"] == ARMS[0]}
    study_dir = "sensing-world-range-audit" if args.match_range else "sensing-world-shift-audit"
    output = args.output or Path("outputs") / study_dir
    with threadpool_limits(limits=1):
        rows, diagnostics = evaluate(config, args.fields_per_layout, archived, match_range=args.match_range)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "metrics.csv", rows)
    if diagnostics:
        (output / "correlograms.json").write_text(json.dumps(diagnostics, indent=2, allow_nan=False) + "\n")
    summary = summarize(rows, match_range=args.match_range)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for name, result in summary["conditions"].items():
        print(name, np.round(result["rmse"], 4), np.round(result["advantage_lost"]["mean"], 4),
              np.round(result["advantage_lost"]["ci95"], 4))


if __name__ == "__main__":
    main()
