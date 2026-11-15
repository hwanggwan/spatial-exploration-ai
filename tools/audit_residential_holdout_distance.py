"""Compare frozen mappers by hold-out distance on archived short-range corridor routes."""

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from audit_residential_coverage import key
from audit_residential_oracle_budget import checked_route
from audit_residential_oracle_mapping import route_key
from audit_residential_routes import (
    CONDUCTANCES, LayoutCase, SEED, design_for, field_cases, field_covariances, floor_plan,
)
from audit_sensing_world_shifts import correlation_area, match_scale, within_room_correlogram
from spatial_exploration.barrier_gp import graph_spectrum, unit_covariance
from spatial_exploration.baseline import GPBelief, write_csv
from spatial_exploration.basis_bayes import BayesianBasisModel
from spatial_exploration.sensing_world import Belief, CONDUCTANCES as KERNEL_CONDUCTANCES, HEAT_LENGTHS, kernel_basis
from spatial_exploration.sensing_world_study import NOISE


FAMILY = "corridor"
CAP = 24
MODELS = ("world", "gp", "exact")
BANDS = {"1": (1, 1), "2": (2, 2), "3-4": (3, 4), "5-7": (5, 7), "8+": (8, np.inf)}


def short_mode_indices(modes):
    shortest_group = max(scale for _, scale in modes)
    return [i for i, (_, scale) in enumerate(modes) if scale == shortest_group]


def match_short_components(layout, modes, weights, covariance, target, *, correlogram=None):
    """Shorten graph heat lengths while preserving graph and mixture coefficients."""
    if correlogram is None:
        correlogram = within_room_correlogram(layout)
    spectra = [graph_spectrum(layout, p) for p in KERNEL_CONDUCTANCES]
    lengths = np.asarray(HEAT_LENGTHS) * (layout.size - 1)
    result = covariance.copy()
    diagnostics = []
    for i in short_mode_indices(modes):
        wall, scale = modes[i]
        terms = [(weight, length, eigenvalues, eigenvectors)
                 for (eigenvalues, eigenvectors), group in zip(spectra, weights[i, 1:].reshape(-1, len(lengths)))
                 for length, weight in zip(lengths, group) if weight > 0]

        def scaled_covariance(multiplier):
            matrix = weights[i, 0] * np.eye(len(layout.points))
            for weight, length, eigenvalues, eigenvectors in terms:
                spectrum = np.exp(-0.5 * (multiplier * length)**2 * eigenvalues)
                matrix += weight * unit_covariance(eigenvectors, spectrum)
            return matrix

        diagonal = covariance[i].diagonal()
        normalization = np.sqrt(np.outer(diagonal, diagonal))
        np.testing.assert_allclose(scaled_covariance(1.0), covariance[i], rtol=0, atol=1e-12)
        multiplier, _ = match_scale(lambda value: scaled_covariance(value) / normalization,
                                    correlogram, correlation_area, target)
        if multiplier >= 1:
            raise ValueError("Target must shorten the learned component")
        result[i] = scaled_covariance(multiplier)
        before = correlogram(covariance[i] / normalization)
        after = correlogram(result[i] / normalization)
        diagnostics.append(dict(mode=wall, scale=scale, heat_length_multiplier=multiplier,
                                original_area=correlation_area(before), matched_area=correlation_area(after),
                                original_correlation=before.tolist(), matched_correlation=after.tolist()))
    np.testing.assert_allclose(result.diagonal(axis1=1, axis2=2), covariance.diagonal(axis1=1, axis2=2),
                               rtol=0, atol=1e-12)
    return result, diagnostics


def condition_world(prior, sites, readings, held):
    """Return the latent mixture mean, standard deviation, and time on hold-outs."""
    before = perf_counter()
    state = prior
    for site, value in zip(sites, readings):
        state = state.observe(site, value)
    mean, cov = state.moments()
    sd = np.sqrt(np.maximum(cov.diagonal()[held], 0))
    return mean[held], sd, (perf_counter() - before) * 1000, np.exp(state.log_weights)


def replay(protocol, model, routes, reference, *, short_range_target=None, short_only=False):
    points, fields, calibration = [], [], []
    models = MODELS if short_range_target is None else (*MODELS, "matched_world")
    if short_only:
        models = (*models, "short_only")
        short = short_mode_indices(model["modes"])
        if len(short) != 2 or short_range_target is None:
            raise ValueError("Short-only replay requires two adjusted short modes")
    weights = np.asarray(model["world_weights"]["world_4"])
    spec = model["gp"]
    for variant in sorted({route["variant"] for route in routes.values()}):
        case = LayoutCase(FAMILY, variant)
        layout = floor_plan(case)
        design = design_for(layout, SEED + variant)
        held = np.flatnonzero(design.heldout)
        world_covariance = np.einsum("mb,bij->mij", weights, kernel_basis(layout))
        world = Belief.prior(world_covariance, NOISE)
        if short_range_target is not None:
            covariance, changes = match_short_components(layout, model["modes"], weights,
                                                         world_covariance, short_range_target)
            matched = Belief.prior(covariance, NOISE)
            calibration.extend(dict(variant=variant, **change) for change in changes)
            correlogram = within_room_correlogram(layout)
            # Every basis has a constant diagonal. Mix raw covariances and
            # normalize the resulting lag-two covariance by the mixed variance.
            variances = world_covariance[:, 0, 0]
            original_lag2 = np.array([correlogram(cov)[2] for cov in world_covariance])
            matched_lag2 = np.array([correlogram(cov)[2] for cov in covariance])
            if short_only:
                restricted = Belief.prior(covariance[short], NOISE)
        intercept = np.sqrt((design.features.shape[1] + 1) * spec["intercept"])
        gp = BayesianBasisModel(design.features, NOISE, spec["prior"], intercept_scale=intercept)
        covariances = {regime: field_covariances(layout, p) for regime, p in CONDUCTANCES.items()}
        route_data = {}
        for group, route in routes.items():
            if route["variant"] != variant:
                continue
            previous = checked_route({k: v for k, v in route.items() if k not in ("edges", "readings", "cap")},
                                     design, CAP, protocol["edges"])
            scale, _ = covariances[route["regime"]][protocol["area"]]
            np.testing.assert_allclose(scale, previous["scale"], rtol=0, atol=1e-12)
            sites = np.asarray(previous["sites"])
            steps = layout.distances[np.ix_(held, sites)].min(axis=1) * (layout.size - 1)
            nearest = np.rint(steps).astype(int)
            np.testing.assert_allclose(nearest, steps, rtol=0, atol=1e-10)
            if (nearest < 1).any() or previous["edges"] != protocol["edges"]:
                raise ValueError("Invalid hold-out distances or travel edge count")
            route_data[group] = sites, nearest

        for record, truth, sensor in field_cases(layout, case, protocol["draws_per_start"], [],
                                                 covariances=covariances):
            if record["area"] != protocol["area"]:
                continue
            sites, nearest = route_data[route_key(record)]
            readings = sensor[sites]
            _, covariance = covariances[record["regime"]][record["area"]]

            world_mean, world_sd, world_ms, world_prob = condition_world(world, sites, readings, held)
            before = perf_counter()
            fixed = gp.condition(sites, readings)
            gp_ms = (perf_counter() - before) * 1000

            before = perf_counter()
            oracle = GPBelief(covariance, NOISE)
            for site, value in zip(sites, readings):
                oracle.observe(site, value)
            exact_ms = (perf_counter() - before) * 1000

            means = [world_mean, fixed.mean[held], oracle.mean[held]]
            sds = [world_sd, fixed.std[held], oracle.std[held]]
            times = [world_ms, gp_ms, exact_ms]
            mixture = {}
            if short_range_target is not None:
                mean, sd, elapsed, matched_prob = condition_world(matched, sites, readings, held)
                means.append(mean)
                sds.append(sd)
                times.append(elapsed)
                mixture = dict(world_lag2=float(world_prob @ original_lag2 / (world_prob @ variances)),
                               matched_world_lag2=float(matched_prob @ matched_lag2 / (matched_prob @ variances)))
            if short_only:
                mean, sd, elapsed, prob = condition_world(restricted, sites, readings, held)
                means.append(mean)
                sds.append(sd)
                times.append(elapsed)
                mixture["short_only_lag2"] = float(prob @ matched_lag2[short] / (prob @ variances[short]))
                mixture["matched_world_long_weight"] = float(1 - matched_prob[short].sum())
            means = np.column_stack(means)
            sds = np.column_stack(sds)
            rmse = np.sqrt(np.mean((means - truth[held, None])**2, axis=0))
            prior = reference[key(record)]
            for name, current in (("world_rmse", rmse[0]), ("exact_rmse", rmse[2]), ("readings", len(sites))):
                np.testing.assert_allclose(current, float(prior[name]), rtol=0, atol=1e-10)

            fields.append(dict(**record, **mixture, **{f"{arm}_rmse": value for arm, value in zip(models, rmse)},
                               **{f"{arm}_map_ms": value for arm, value in zip(models, times)}))
            for i, site in enumerate(held):
                points.append(dict(**record, site=int(site), distance=int(nearest[i]), truth=float(truth[site]),
                                   **{f"{arm}_mean": means[i, j] for j, arm in enumerate(models)},
                                   **{f"{arm}_sd": sds[i, j] for j, arm in enumerate(models)}))

    if len(fields) != len(reference) or {key(row) for row in fields} != set(reference):
        raise ValueError("Replay differs from the archived set of fields")
    if len(points) != 28 * len(fields):
        raise ValueError("Expected 28 hold-out points per field")
    return fields, points, calibration


def summarize(fields, points, *, compare_range=False, short_only=False):
    models = (*MODELS, "matched_world") if compare_range else MODELS
    if short_only:
        models = (*models, "short_only")
    rng = np.random.default_rng(SEED + 900)
    scene_index = {key(row): i for i, row in enumerate(fields)}
    strata = {}
    for i, row in enumerate(fields):
        strata.setdefault(route_key(row), []).append(i)
    bootstrap = np.concatenate([rng.choice(idx, (6000, len(idx))) for idx in strata.values()], axis=1)
    results = {}
    bands = BANDS | {"2-4": (2, 4)} if compare_range else BANDS
    for band, (low, high) in bands.items():
        rows = [row for row in points if low <= row["distance"] <= high]
        index = np.array([scene_index[key(row)] for row in rows])
        counts = np.bincount(index, minlength=len(fields))
        boot_counts = counts[bootstrap].sum(axis=1)
        scores, boot_rmse = {}, {}
        for arm in models:
            errors = np.array([row[f"{arm}_mean"] - row["truth"] for row in rows])
            sds = np.array([row[f"{arm}_sd"] for row in rows])
            sums = np.bincount(index, weights=errors**2, minlength=len(fields))
            scores[arm] = dict(rmse=float(np.sqrt(sums.sum() / len(rows))),
                               coverage=float(np.mean(abs(errors) <= 1.96 * sds)),
                               width=float(np.mean(3.92 * sds)))
            boot_rmse[arm] = np.sqrt(sums[bootstrap].sum(axis=1) / boot_counts)
        contrasts = {}
        for other in ("gp", "exact"):
            diff = boot_rmse["world"] - boot_rmse[other]
            contrasts[other] = dict(delta=scores["world"]["rmse"] - scores[other]["rmse"],
                                    ci95=np.quantile(diff, [.025, .975]).tolist())
        results[band] = dict(points=len(rows), fields=int(np.count_nonzero(counts)),
                             routes=len({route_key(row) for row in rows}), scores=scores, world_minus=contrasts)
        if compare_range:
            diff = boot_rmse["matched_world"] - boot_rmse["world"]
            results[band]["matched_minus_original"] = dict(
                delta=scores["matched_world"]["rmse"] - scores["world"]["rmse"],
                ci95=np.quantile(diff, [.025, .975]).tolist())
        if short_only:
            diff = boot_rmse["short_only"] - boot_rmse["matched_world"]
            results[band]["short_only_minus_matched"] = dict(
                delta=scores["short_only"]["rmse"] - scores["matched_world"]["rmse"],
                ci95=np.quantile(diff, [.025, .975]).tolist())
    if sum(results[band]["points"] for band in BANDS) != len(points):
        raise ValueError("Distance bands do not partition the hold-outs")
    means = {arm: {metric: float(np.mean([row[f"{arm}_{metric}"] for row in fields]))
                   for metric in ("rmse", "map_ms")} for arm in models}
    summary = dict(field_means=means, distance_bands=results)
    if short_only:
        delta = np.array([row["short_only_rmse"] - row["matched_world_rmse"] for row in fields])
        summary["field_mean_short_only_minus_matched"] = dict(
            delta=float(delta.mean()), ci95=np.quantile(delta[bootstrap].mean(axis=1), [.025, .975]).tolist())
    if compare_range:
        delta = np.array([row["matched_world_rmse"] - row["world_rmse"] for row in fields])
        summary["field_mean_matched_minus_original"] = dict(
            delta=float(delta.mean()), ci95=np.quantile(delta[bootstrap].mean(axis=1), [.025, .975]).tolist())
        groups = {}
        for row in fields:
            groups.setdefault((row["variant"], row["regime"]), []).append(row)
        diagnostics = ("world_lag2", "matched_world_lag2")
        if short_only:
            diagnostics += ("short_only_lag2", "matched_world_long_weight")
        summary["reweighted_prior_lag2"] = [
            dict(variant=variant, regime=regime,
                 **{name: float(np.mean([row[name] for row in group])) for name in diagnostics})
            for (variant, regime), group in groups.items()
        ]
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("outputs/sensing-world-residential-oracle-budget"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--match-short-range", action="store_true",
                        help="Match the two short modes to the simulator's short-range area.")
    parser.add_argument("--short-only", action="store_true",
                        help="Also compare a range-matched mapper restricted to the two short modes.")
    args = parser.parse_args()
    compare_range = args.match_short_range or args.short_only
    if args.output is None:
        suffix = "range-match" if compare_range else "holdout-distance"
        if args.short_only:
            suffix = "short-only"
        args.output = Path(f"outputs/sensing-world-residential-{suffix}")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use an empty output directory.")
    protocol = json.loads((args.source / "protocol.json").read_text())
    model = json.loads(Path(protocol["model"]).read_text())
    routes = {route_key(row): row for row in json.loads((args.source / "routes.json").read_text())
              if row["family"] == FAMILY and row["cap"] == CAP}
    with (args.source / "metrics.csv").open() as stream:
        reference = {key(row): row for row in csv.DictReader(stream)
                     if row["family"] == FAMILY and int(row["cap"]) == CAP}
    target = protocol["area"] if compare_range else None
    with threadpool_limits(limits=1):
        fields, points, calibration = replay(protocol, model, routes, reference,
                                             short_range_target=target, short_only=args.short_only)
    summary = summarize(fields, points, compare_range=compare_range, short_only=args.short_only)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "fields.csv", fields)
    write_csv(args.output / "points.csv", points)
    run = dict(source=str(args.source), family=FAMILY, cap=CAP, edges=protocol["edges"], area=protocol["area"],
               distance="Minimum legal walking edges from any measurement",
               bands=list(summary["distance_bands"]), model=protocol["model"])
    contrast = "world_minus"
    if compare_range:
        contrast = "matched_minus_original"
        run["short_range_target"] = target
        (args.output / "calibration.json").write_text(json.dumps(calibration, indent=2, allow_nan=False) + "\n")
    if args.short_only:
        contrast = "short_only_minus_matched"
        run["restricted_modes"] = [model["modes"][i] for i in short_mode_indices(model["modes"])]
    for name, payload in (("summary", summary), ("protocol", run)):
        (args.output / f"{name}.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    for band, row in summary["distance_bands"].items():
        print(band, row["points"], {arm: round(stats["rmse"], 4) for arm, stats in row["scores"].items()},
              row[contrast])


if __name__ == "__main__":
    main()
