"""Compare pooled and roughness-split sensing models on identical observations."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import softmax
from threadpoolctl import threadpool_limits

from audit_sensing_world_shifts import correlation_area, match_scale, within_room_correlogram
from spatial_exploration.barrier_gp import graph_spectrum, unit_covariance
from spatial_exploration.baseline import write_csv
from spatial_exploration.basis_bayes import BayesianBasisModel
from spatial_exploration.sensing_world import Belief, fit_kernel_weights, kernel_basis
from spatial_exploration.sensing_world_generalization import (
    CAP, EDGES, MODES, NOISE, STARTS, TRAIN_GEOMETRIES, make_design, plan_route, wall_layout,
)


TRAIN_SEED, EVALUATION_SEED = 298001, 299001
PER_GROUP = 96
POWERS = (2., 1.45)
LABELS = ("smooth", "rough")
ARMS = ("gp", "world_4", "world_8", "world_8_known_roughness")


def field_factors(layout):
    correlogram = within_room_correlogram(layout)
    factors, diagnostics = [], []
    for mode, (regime, kappa) in enumerate(MODES):
        permeability = .02 if regime == "barrier" else 1.
        spectrum, vectors = graph_spectrum(layout, permeability)
        smooth = unit_covariance(vectors, (kappa**2 + spectrum)**-POWERS[0])
        smooth_curve = correlogram(smooth)
        rough_scale, rough = match_scale(
            lambda k: unit_covariance(vectors, (k**2 + spectrum)**-POWERS[1]),
            correlogram, correlation_area, correlation_area(smooth_curve))
        rough_curve = correlogram(rough)
        if rough_curve[1] >= smooth_curve[1]:
            raise ValueError("Range matching must retain higher short-distance roughness")
        factors.append(np.linalg.cholesky(np.stack((smooth, rough))))
        for label, scale, curve in zip(LABELS, (kappa, rough_scale), (smooth_curve, rough_curve)):
            diagnostics.append(dict(mode=mode, condition=label, scale=scale, permeability=permeability,
                range_area=correlation_area(curve), lag1_semivariance=float(1 - curve[1])))
    return np.stack(factors), diagnostics


def training_groups():
    rng = np.random.default_rng(TRAIN_SEED)
    for geometry in TRAIN_GEOMETRIES:
        layout = wall_layout(geometry)
        factors, _ = field_factors(layout)
        groups = []
        for factor in factors.reshape(-1, *factors.shape[-2:]):
            z, noise = rng.standard_normal((2, PER_GROUP, len(factor)))
            groups.append(z @ factor.T + NOISE * noise)
        yield layout, groups


def train():
    pooled, split = [], []
    for layout, groups in training_groups():
        basis = kernel_basis(layout)
        pooled.append((basis, [np.concatenate(groups[i:i+2]) for i in range(0, len(groups), 2)]))
        split.append((basis, groups))
    return fit_kernel_weights(pooled, NOISE), fit_kernel_weights(split, NOISE)


def posterior(covariances, sites, readings, held, *, with_variance=False):
    """Return conditional means and log likelihood for every field and mixture component."""
    means, likelihoods, variances = [], [], []
    for k in covariances:
        factor = np.linalg.cholesky(k[np.ix_(sites, sites)] + NOISE**2 * np.eye(len(sites)))
        z = solve_triangular(factor, readings.T, lower=True, check_finite=False).T
        cross = solve_triangular(factor, k[sites][:, held], lower=True, check_finite=False)
        means.append(z @ cross)
        likelihoods.append(-.5 * np.sum(z**2, axis=1) - np.log(np.diag(factor)).sum())
        if with_variance:
            variances.append(np.maximum(np.diag(k)[held] - np.sum(cross**2, axis=0), 0))
    result = np.stack(means, axis=1), np.stack(likelihoods, axis=1)
    return (*result, np.stack(variances)) if with_variance else result


def combine(means, likelihoods):
    weights = softmax(likelihoods, axis=1)
    return np.einsum("bm,bmn->bn", weights, means), weights


def shared_observations(layout_id, design, ridge, factors, count):
    """Generate paired smooth/rough readings at the same frozen GP stops."""
    rng = np.random.default_rng(EVALUATION_SEED + layout_id)
    layout = design.layout
    limit = EDGES / (layout.size - 1)
    route_cache = {start: plan_route(design, ridge, start, CAP, limit, "rollout") for start in STARTS}
    n = count // (len(MODES) * len(STARTS))
    for mode, mode_factors in enumerate(factors):
        for start, route in route_cache.items():
            sites = np.asarray(route.sites)
            if design.heldout[sites].any() or len(sites) > CAP or route.distances[-1] > limit + 1e-12:
                raise ValueError("Invalid shared observation route")
            edges = round(route.distances[-1] * (layout.size - 1))
            z, noise = rng.standard_normal((2, n, len(design.heldout)))
            for roughness, (label, factor) in enumerate(zip(LABELS, mode_factors)):
                truth = z @ factor.T
                readings = truth[:, sites] + NOISE * noise[:, sites]
                yield mode, start, sites, edges, roughness, label, truth, readings


def evaluate(config, weights, count):
    rows, diagnostics = [], []
    for layout_id, spec in enumerate(config["test_geometry"]):
        layout = wall_layout(spec)
        design = make_design(layout, config["test_seed"] + 100 * layout_id)
        held = design.heldout
        settings = config["gp"]
        intercept = np.sqrt((design.features.shape[1] + 1) * settings["intercept"])
        ridge = BayesianBasisModel(design.features, NOISE, settings["prior"], intercept_scale=intercept)
        basis = kernel_basis(layout)
        covariances = [ridge.prior_covariance[None], *[np.einsum("mb,bij->mij", w, basis) for w in weights]]
        factors, info = field_factors(layout)
        diagnostics.extend(dict(layout=layout_id, **row) for row in info)
        for mode, start, sites, edges, roughness, label, truth, readings in shared_observations(
                layout_id, design, ridge, factors, count):
            fits = [posterior(k, sites, readings, held) for k in covariances]
            predictions = [combine(*fit)[0] for fit in fits[:2]]
            means, likelihoods = fits[2]
            full, probability = combine(means, likelihoods)
            disclosed, _ = combine(means[:, roughness::2], likelihoods[:, roughness::2])
            predictions.extend((full, disclosed))
            # Independent sequential update confirms the batch posterior and group restriction.
            if mode == 0 and start == STARTS[0]:
                for k, prediction in zip((*covariances, covariances[-1][roughness::2]), predictions):
                    belief = Belief.prior(k, NOISE)
                    for site, value in zip(sites, readings[0]):
                        belief = belief.observe(site, value)
                    np.testing.assert_allclose(prediction[0], belief.moments()[0][held], atol=1e-10, rtol=0)
            rmse = np.array([np.sqrt(np.mean((prediction - truth[:, held])**2, axis=1))
                             for prediction in predictions]).T
            p_rough = probability[:, 1::2].sum(axis=1)
            for draw, (scores, p) in enumerate(zip(rmse, p_rough)):
                rows.append(dict(layout=layout_id, mode=mode, start=start, draw=draw, condition=label,
                                 readings=len(sites), edges=edges,
                                 p_rough=float(p), **dict(zip(ARMS, scores))))
    return rows, diagnostics


def summarize(rows):
    rng = np.random.default_rng(299111)
    results = {}
    comparisons = (("world_4", "gp"), ("world_8", "gp"), ("world_8", "world_4"),
                   ("world_8_known_roughness", "world_8"), ("world_8_known_roughness", "world_4"))
    for label in LABELS:
        group = [row for row in rows if row["condition"] == label]
        strata = {}
        for i, row in enumerate(group):
            strata.setdefault((row["layout"], row["mode"], row["start"]), []).append(i)
        sampled = np.concatenate([rng.choice(idx, (6000, len(idx))) for idx in strata.values()], axis=1)
        scores = {arm: np.array([row[arm] for row in group]) for arm in ARMS}
        effects = {}
        for arm, ref in comparisons:
            delta = scores[arm] - scores[ref]
            effects[f"{arm}_minus_{ref}"] = dict(mean=float(delta.mean()),
                ci95=np.quantile(delta[sampled].mean(axis=1), [.025, .975]).tolist())
        p_rough = np.array([r["p_rough"] for r in group])
        results[label] = dict(fields=len(group), rmse={arm: float(v.mean()) for arm, v in scores.items()},
            contrasts=effects, mean_p_rough=float(p_rough.mean()),
            correct_group_rate=float(np.mean((p_rough > .5) == (label == "rough"))),
            mean_readings=float(np.mean([r["readings"] for r in group])),
            mean_edges=float(np.mean([r["edges"] for r in group])))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("outputs/sensing-world-generalization/frozen.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/sensing-world-modes"))
    parser.add_argument("--fields-per-layout", type=int, default=128)
    args = parser.parse_args()
    count = args.fields_per_layout
    if count < 32 or count % (len(MODES) * len(STARTS)):
        parser.error("Use at least 32 fields per layout and balance the modes and starts.")
    config = json.loads(args.reference.read_text())
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use an empty output directory.")
    args.output.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        weights = train()
        protocol = dict(train_seed=TRAIN_SEED, evaluation_seed=EVALUATION_SEED, training_layouts=TRAIN_GEOMETRIES,
            test_layouts=config["test_geometry"], fields_per_layout_and_roughness=count, training_draws_per_group=PER_GROUP,
            roughness_powers=dict(zip(LABELS, POWERS)), modes=MODES, cap=CAP, edges=EDGES, gp=config["gp"],
            world_weights={arm: w.tolist() for arm, w in zip(ARMS[1:3], weights)})
        (args.output / "frozen.json").write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n")
        rows, diagnostics = evaluate(config, weights, count)
    write_csv(args.output / "metrics.csv", rows)
    write_csv(args.output / "range.csv", diagnostics)
    summary = summarize(rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for label, entry in summary.items():
        print(label, entry)


if __name__ == "__main__":
    main()
