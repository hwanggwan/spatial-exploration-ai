"""Compare two mappers on the archived exact-covariance rollout measurements."""

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from threadpoolctl import threadpool_limits

from audit_residential_coverage import key
from audit_residential_oracle import AREA, FAMILIES
from audit_residential_routes import CONDUCTANCES, LayoutCase, SEED, design_for, field_cases, field_covariances, floor_plan
from spatial_exploration.baseline import GPBelief, write_csv
from spatial_exploration.sensing_world import Belief, kernel_basis
from spatial_exploration.sensing_world_study import CAP, NOISE

ARMS = ("world", "exact")
METRICS = ("rmse", "coverage", "width", "map_ms")


def route_key(record):
    return tuple(record[name] for name in ("family", "variant", "regime", "start"))


def map_metrics(mean, sd, truth, ms):
    error = mean - truth
    return dict(rmse=float(np.sqrt(np.mean(error**2))), coverage=float(np.mean(abs(error) <= 1.96 * sd)),
                width=float(np.mean(3.92 * sd)), map_ms=ms)


def evaluate(config, model, archive, routes):
    rows = []
    weights = np.asarray(model["world_weights"]["world_4"])
    for family in FAMILIES:
        for variant in range(2):
            case = LayoutCase(family, variant)
            layout = floor_plan(case)
            design = design_for(layout, SEED + variant)
            held = design.heldout
            world = Belief.prior(np.einsum("mb,bij->mij", weights, kernel_basis(layout)), NOISE)
            covariances = {regime: field_covariances(layout, p) for regime, p in CONDUCTANCES.items()}
            for record, truth, sensor in field_cases(layout, case, config["draws_per_start"], [], covariances=covariances):
                if record["area"] != AREA:
                    continue
                reference = archive[key(record)]
                route = routes[route_key(record)]
                scale, covariance = covariances[record["regime"]][AREA]
                np.testing.assert_allclose(route["scale"], scale, rtol=0, atol=1e-12)
                sites = np.asarray(route["sites"])
                edges = round(route["distances"][-1] * (layout.size - 1))
                if (sites[0] != record["start"] or len(sites) != len(set(sites))
                        or len(sites) != int(reference["readings"]) or len(sites) > CAP
                        or held[sites].any() or edges != int(reference["edges"]) or edges > config["edges"]):
                    raise ValueError("Invalid archived oracle route")
                readings = sensor[sites]
                before = perf_counter()
                posterior = world
                for site, value in zip(sites, readings):
                    posterior = posterior.observe(site, value)
                mean, cov = posterior.moments()
                elapsed = (perf_counter() - before) * 1000
                learned = map_metrics(mean[held], np.sqrt(np.maximum(np.diag(cov)[held], 0)), truth[held], elapsed)
                np.testing.assert_allclose(learned["rmse"], float(reference["rmse"]), rtol=0, atol=1e-10)

                before = perf_counter()
                exact = GPBelief(covariance, NOISE)
                for site, value in zip(sites, readings):
                    exact.observe(site, value)
                elapsed = (perf_counter() - before) * 1000
                oracle = map_metrics(exact.mean[held], exact.std[held], truth[held], elapsed)
                observed_cov = covariance[np.ix_(sites, sites)] + NOISE**2 * np.eye(len(sites))
                factor = cho_factor(observed_cov)
                cross = covariance[held][:, sites]
                np.testing.assert_allclose(exact.mean[held], cross @ cho_solve(factor, readings), atol=1e-11, rtol=0)
                variances = covariance.diagonal()[held] - np.einsum("ij,ji->i", cross, cho_solve(factor, cross.T))
                np.testing.assert_allclose(exact.std[held]**2, variances, atol=1e-11, rtol=0)
                rows.append(dict(**record, readings=len(sites), edges=edges,
                    **{f"{arm}_{m}": stats[m] for arm, stats in zip(ARMS, (learned, oracle)) for m in METRICS}))
    if len(rows) != len(archive) or {key(r) for r in rows} != set(archive):
        raise RuntimeError("Replays do not cover the archived oracle fields")
    return rows


def summarize(rows):
    rng = np.random.default_rng(SEED + 700)
    result = {}
    for family in FAMILIES:
        group = [r for r in rows if r["family"] == family]
        strata = {}
        for i, row in enumerate(group):
            strata.setdefault((row["variant"], row["regime"], row["start"]), []).append(i)
        draws = np.concatenate([rng.choice(idx, (6000, len(idx))) for idx in strata.values()], axis=1)
        difference = np.array([r["exact_rmse"] - r["world_rmse"] for r in group])
        means = {arm: {metric: float(np.mean([row[f"{arm}_{metric}"] for row in group]))
                       for metric in METRICS} for arm in ARMS}
        result[family] = dict(fields=len(group), means=means, exact_minus_world=float(difference.mean()),
                              ci95=np.quantile(difference[draws].mean(axis=1), [.025, .975]).tolist())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("outputs/sensing-world-residential-oracle"))
    parser.add_argument("--output", type=Path, default=Path("outputs/sensing-world-residential-oracle-mapping"))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use an empty output directory.")
    config = json.loads((args.source / "protocol.json").read_text())
    comparison = json.loads((Path(config["reference"]) / "protocol.json").read_text())
    model = json.loads(Path(comparison["model"]).read_text())
    with (args.source / "metrics.csv").open() as stream:
        archive = {key(r): r for r in csv.DictReader(stream) if r["arm"] == "oracle"}
    routes = {route_key(r): r for r in json.loads((args.source / "routes.json").read_text())}
    with threadpool_limits(limits=1):
        rows = evaluate(config, model, archive, routes)
    summary = summarize(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "metrics.csv", rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for family, row in summary.items():
        print(family, row)


if __name__ == "__main__":
    main()
