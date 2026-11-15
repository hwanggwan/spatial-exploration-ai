"""Compare range-matched graph spectra on the frozen roughness benchmark."""

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import brentq
from threadpoolctl import threadpool_limits

from audit_sensing_world_modes import (
    EVALUATION_SEED, LABELS, PER_GROUP, POWERS, TRAIN_SEED, combine, field_factors, posterior,
    shared_observations, training_groups,
)
from audit_sensing_world_shifts import correlation_area, within_room_correlogram
from spatial_exploration.barrier_gp import graph_spectrum, unit_covariance
from spatial_exploration.baseline import write_csv
from spatial_exploration.basis_bayes import BayesianBasisModel
from spatial_exploration.sensing_world import CONDUCTANCES, HEAT_LENGTHS, fit_kernel_weights, kernel_basis
from spatial_exploration.sensing_world_generalization import (
    CAP, EDGES, MODES, NOISE, STARTS, TRAIN_GEOMETRIES, make_design, wall_layout,
)


ARMS = ("gp", "heat", "spectral_uniform", "spectral_split")
LOW_POWER = 2.
CUTOFF = 1.
HIGH_POWERS = (.5, .75, 1., 1.25, 1.5, 1.75, 2., 2.5, 3., 4.)


def spectrum_power(eigenvalues, scale, high_power):
    """Inverse coefficient penalty with a continuous join at the cutoff."""
    lo = np.minimum(eigenvalues, CUTOFF)
    hi = np.maximum(eigenvalues - CUTOFF, 0)
    return (1 + lo / scale**2)**-LOW_POWER * (1 + hi / (scale**2 + CUTOFF))**-high_power


def match_heat_range(spectra, correlogram, heat, weights, high_power):
    """Match heat variance, conductance mass, and within-room correlation area."""
    marginal_variance = float(heat[0, 0])
    target = correlation_area(correlogram(heat / marginal_variance))
    graph_weights = weights[1:].reshape(len(CONDUCTANCES), len(HEAT_LENGTHS)).sum(axis=1)

    def covariance(scale):
        result = weights[0] * np.eye(len(heat))
        for (values, vectors), weight in zip(spectra, graph_weights):
            if weight > 0:
                result += weight * unit_covariance(vectors, spectrum_power(values, scale, high_power))
        return result

    scale = brentq(lambda k: correlation_area(correlogram(covariance(k) / marginal_variance)) - target,
                   1e-6, 100., xtol=1e-12, rtol=1e-12)
    matched = covariance(scale)
    curve = correlogram(matched / marginal_variance)
    np.testing.assert_allclose(correlation_area(curve), target, atol=1e-9, rtol=0)
    np.testing.assert_allclose(np.diag(matched), np.diag(heat), atol=1e-12, rtol=0)
    return matched, scale, target, float(1 - curve[1])


def train():
    training = [(layout, [np.concatenate(groups[i:i+2]) for i in range(0, len(groups), 2)])
                for layout, groups in training_groups()]
    bases = [kernel_basis(layout) for layout, _ in training]
    heat_weights = fit_kernel_weights([(basis, groups) for basis, (_, groups) in zip(bases, training)], NOISE)
    loss = np.zeros((len(MODES), len(HIGH_POWERS)))
    for basis, (layout, groups) in zip(bases, training):
        spectra = [graph_spectrum(layout, p) for p in CONDUCTANCES]
        correlogram = within_room_correlogram(layout)
        for mode, (samples, weights) in enumerate(zip(groups, heat_weights)):
            empirical = samples.T @ samples / len(samples) - NOISE**2 * np.eye(len(layout.points))
            heat = np.einsum("m,mij->ij", weights, basis)
            for i, high_power in enumerate(HIGH_POWERS):
                candidate, *_ = match_heat_range(spectra, correlogram, heat, weights, high_power)
                loss[mode, i] += np.mean((candidate - empirical)**2) / len(training)
    selected = [HIGH_POWERS[i] for i in loss.argmin(axis=1)]
    return heat_weights, selected, loss


def evaluate(config, weights, powers, count, previous):
    rows, diagnostics = [], []
    for layout_id, spec in enumerate(config["test_geometry"]):
        layout = wall_layout(spec)
        design = make_design(layout, config["test_seed"] + 100 * layout_id)
        held = design.heldout
        gp = config["gp"]
        intercept = np.sqrt((design.features.shape[1] + 1) * gp["intercept"])
        ridge = BayesianBasisModel(design.features, NOISE, gp["prior"], intercept_scale=intercept)
        basis = kernel_basis(layout)
        heat = np.einsum("mb,bij->mij", weights, basis)
        spectra = [graph_spectrum(layout, p) for p in CONDUCTANCES]
        correlogram = within_room_correlogram(layout)
        kernels = {"gp": ridge.prior_covariance[None], "heat": heat}
        for arm, high_powers in (("spectral_uniform", [LOW_POWER] * len(MODES)), ("spectral_split", powers)):
            matrices = []
            for mode, (k, w, power) in enumerate(zip(heat, weights, high_powers)):
                matrix, scale, area, lag1 = match_heat_range(spectra, correlogram, k, w, power)
                matrices.append(matrix)
                diagnostics.append(dict(layout=layout_id, mode=mode, arm=arm, scale=scale,
                    target_area=area, achieved_area=correlation_area(correlogram(matrix / matrix[0, 0])),
                    lag1_semivariance=lag1, heat_lag1_semivariance=float(1 - correlogram(k / k[0, 0])[1])))
            kernels[arm] = np.stack(matrices)
        factors, _ = field_factors(layout)
        for mode, start, sites, edges, _, label, truth, readings in shared_observations(
                layout_id, design, ridge, factors, count):
            stats = {}
            actual = truth[:, held]
            for arm, covariances in kernels.items():
                started = perf_counter()
                means, lls, variance = posterior(covariances, sites, readings, held, with_variance=True)
                estimate, probs = combine(means, lls)
                mixture_var = np.einsum("bm,bmn->bn", probs, variance + (means - estimate[:, None])**2)
                elapsed_ms = (perf_counter()-started) * 1000 / len(readings)
                errors = estimate - actual
                width = 1.96 * np.sqrt(mixture_var)
                stats[arm] = (np.sqrt(np.mean(errors**2, axis=1)), np.mean(abs(errors) <= width, axis=1),
                              (2 * width).mean(axis=1), elapsed_ms)
            for i in range(len(readings)):
                key = (str(layout_id), str(mode), str(start), str(i), label)
                np.testing.assert_allclose([stats[a][0][i] for a in ("gp", "heat")], previous[key],
                                           rtol=0, atol=1e-10)
                row = dict(layout=layout_id, mode=mode, start=start, draw=i, condition=label, readings=len(sites), edges=edges)
                for arm, (rmse, coverage, width, elapsed) in stats.items():
                    row.update({arm: rmse[i], f"{arm}_coverage": coverage[i], f"{arm}_width": width[i],
                                f"{arm}_batch_ms_per_field": elapsed})
                rows.append(row)
    return rows, diagnostics


def summarize(rows):
    rng = np.random.default_rng(300111)
    result = {}
    for label in LABELS:
        group = [r for r in rows if r["condition"] == label]
        strata = {}
        for i, row in enumerate(group):
            strata.setdefault((row["layout"], row["mode"], row["start"]), []).append(i)
        boot = np.concatenate([rng.choice(m, (6000, len(m))) for m in strata.values()], axis=1)
        scores = {arm: np.array([r[arm] for r in group]) for arm in ARMS}
        paired = {}
        for arm, ref in (("spectral_split", "gp"), ("spectral_split", "heat"), ("spectral_split", "spectral_uniform"),
                         ("spectral_uniform", "heat")):
            diff = scores[arm] - scores[ref]
            paired[f"{arm}_minus_{ref}"] = dict(mean=float(diff.mean()),
                ci95=np.quantile(diff[boot].mean(axis=1), [.025, .975]).tolist())
        means = {arm: dict(rmse=float(scores[arm].mean()), **{
            key: float(np.mean([r[f"{arm}_{key}"] for r in group])) for key in ("coverage", "width", "batch_ms_per_field")
        }) for arm in ARMS}
        result[label] = dict(fields=len(group), means=means, contrasts=paired)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("outputs/sensing-world-generalization/frozen.json"))
    parser.add_argument("--previous", type=Path, default=Path("outputs/sensing-world-modes"))
    parser.add_argument("--output", type=Path, default=Path("outputs/sensing-world-spectral"))
    args = parser.parse_args()
    config = json.loads(args.reference.read_text())
    protocol = json.loads((args.previous / "frozen.json").read_text())
    with (args.previous / "metrics.csv").open() as stream:
        previous = {tuple(r[k] for k in ("layout", "mode", "start", "draw", "condition")):
                    (float(r["gp"]), float(r["world_4"])) for r in csv.DictReader(stream)}
    count = protocol["fields_per_layout_and_roughness"]
    if (protocol["test_layouts"] != config["test_geometry"] or protocol["gp"] != config["gp"]
            or protocol["train_seed"] != TRAIN_SEED or protocol["evaluation_seed"] != EVALUATION_SEED
            or protocol["training_draws_per_group"] != PER_GROUP or protocol["cap"] != CAP or protocol["edges"] != EDGES
            or protocol["roughness_powers"] != dict(zip(LABELS, POWERS))):
        parser.error("The previous benchmark does not match this experiment.")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use an empty output directory.")
    args.output.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        weights, powers, loss = train()
        np.testing.assert_allclose(weights, protocol["world_weights"]["world_4"], rtol=0, atol=1e-10)
        settings = dict(train_seed=TRAIN_SEED, evaluation_seed=EVALUATION_SEED, training_layouts=TRAIN_GEOMETRIES,
            test_layouts=config["test_geometry"], fields_per_layout_and_roughness=count, gp=config["gp"], starts=STARTS,
            heat_weights=weights.tolist(), low_power=LOW_POWER, cutoff=CUTOFF, high_power_grid=HIGH_POWERS,
            selected_high_powers=powers, training_covariance_mse=loss.tolist(), cap=CAP, edges=EDGES)
        (args.output / "frozen.json").write_text(json.dumps(settings, indent=2, allow_nan=False) + "\n")
        rows, diagnostics = evaluate(config, weights, powers, count, previous)
    if len(rows) != len(previous):
        raise RuntimeError("Test fields do not match the frozen benchmark")
    write_csv(args.output / "metrics.csv", rows)
    write_csv(args.output / "range.csv", diagnostics)
    result = summarize(rows)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for label, entry in result.items():
        print(label, entry)


if __name__ == "__main__":
    main()
