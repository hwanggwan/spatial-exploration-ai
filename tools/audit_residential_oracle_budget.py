"""Vary the reading cap for the short-range exact-covariance rollout."""

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from audit_residential_coverage import key
from audit_residential_oracle import AREA, FAMILIES
from audit_residential_oracle_mapping import ARMS, METRICS, map_metrics, route_key
from audit_residential_routes import (
    CONDUCTANCES, LayoutCase, SEED, STARTS, design_for, field_cases, field_covariances, floor_plan,
)
from spatial_exploration.baseline import GPBelief, write_csv
from spatial_exploration.sensing_world import Belief, kernel_basis
from spatial_exploration.sensing_world_study import NOISE
from spatial_exploration.wall_rollout_study import plan_route


CAPS = (16, 24)


def checked_route(route, design, cap, edge_cap):
    layout = design.layout
    sites = np.asarray(route["sites"])
    if (sites[0] != route["start"] or len(sites) > cap or len(set(sites)) != len(sites)
            or design.heldout[sites].any()):
        raise ValueError("Invalid measurement sites")
    edges = 0
    for step, (start, end) in enumerate(zip(sites, sites[1:]), start=1):
        walk = layout.walk(start, end)
        if not all(layout.movement[a, b] for a, b in zip(walk, walk[1:])):
            raise ValueError("Invalid movement edge")
        edges += len(walk) - 1
        np.testing.assert_allclose(edges, route["distances"][step] * (layout.size - 1), atol=1e-10, rtol=0)
    if edges > edge_cap:
        raise ValueError("Route exceeds movement cap")
    return dict(**route, cap=cap, edges=edges, readings=len(sites))


def score(sites, readings, heldout, truth, world, covariance):
    before = perf_counter()
    posterior = world
    for site, value in zip(sites, readings):
        posterior = posterior.observe(site, value)
    mean, cov = posterior.moments()
    elapsed = (perf_counter() - before) * 1000
    learned = map_metrics(mean[heldout], np.sqrt(np.maximum(cov.diagonal()[heldout], 0)), truth[heldout], elapsed)

    before = perf_counter()
    exact = GPBelief(covariance, NOISE)
    for site, value in zip(sites, readings):
        exact.observe(site, value)
    elapsed = (perf_counter() - before) * 1000
    oracle = map_metrics(exact.mean[heldout], exact.std[heldout], truth[heldout], elapsed)
    return {f"{arm}_{metric}": result[metric]
            for arm, result in zip(ARMS, (learned, oracle)) for metric in METRICS}


def evaluate(config, weights, archive, archived_routes):
    rows, routes = [], {}
    for family in FAMILIES:
        for variant in range(2):
            case = LayoutCase(family, variant)
            layout = floor_plan(case)
            design = design_for(layout, SEED + variant)
            world = Belief.prior(np.einsum("mb,bij->mij", weights, kernel_basis(layout)), NOISE)
            covariance_bank = {regime: field_covariances(layout, p) for regime, p in CONDUCTANCES.items()}
            for regime, models in covariance_bank.items():
                scale, covariance = models[AREA]
                oracle = SimpleNamespace(prior_covariance=covariance, noise_std=NOISE)
                for start in STARTS:
                    group = (family, variant, regime, start)
                    previous = archived_routes[group]
                    np.testing.assert_allclose(scale, previous["scale"], atol=1e-12, rtol=0)
                    routes[(*group, CAPS[0])] = checked_route(previous, design, CAPS[0], config["edges"])
                    planned = plan_route(design, oracle, start, CAPS[1], config["edges"] / (layout.size - 1), "rollout")
                    new = dict(family=family, variant=variant, regime=regime, start=start, scale=scale,
                               sites=planned.sites, distances=planned.distances,
                               final_mean_oracle_variance=planned.final_variance, planning_ms=planned.planning_ms)
                    routes[(*group, CAPS[1])] = checked_route(new, design, CAPS[1], config["edges"])

            for record, truth, sensor in field_cases(layout, case, config["draws_per_start"], [],
                                                     covariances=covariance_bank):
                if record["area"] != AREA:
                    continue
                _, covariance = covariance_bank[record["regime"]][AREA]
                for cap in CAPS:
                    route = routes[(*route_key(record), cap)]
                    sites = np.asarray(route["sites"])
                    metrics = score(sites, sensor[sites], design.heldout, truth, world, covariance)
                    row = dict(**record, cap=cap, readings=route["readings"], edges=route["edges"],
                               planning_ms=route["planning_ms"], **metrics)
                    if cap == CAPS[0]:
                        previous = archive[key(record)]
                        for metric in ("readings", "edges", "world_rmse", "exact_rmse"):
                            np.testing.assert_allclose(row[metric], float(previous[metric]), atol=1e-10, rtol=0)
                    rows.append(row)
    if len(rows) != len(archive) * len(CAPS) or {key(row) for row in rows} != set(archive):
        raise RuntimeError("Budget comparison does not match the archived fields")
    return rows, list(routes.values())


def summarize(rows):
    rng = np.random.default_rng(SEED + 800)
    fields = {}
    for row in rows:
        fields.setdefault(key(row), {})[row["cap"]] = row
    summary = {}
    for family in FAMILIES:
        pairs = [pair for field, pair in fields.items() if field[0] == family]
        strata = {}
        for i, pair in enumerate(pairs):
            strata.setdefault(route_key(pair[CAPS[0]]), []).append(i)
        bootstrap = np.concatenate([rng.choice(idx, (6000, len(idx))) for idx in strata.values()], axis=1)
        metrics = ("readings", "edges", "planning_ms", *(f"{arm}_{m}" for arm in ARMS for m in METRICS))
        means = {cap: {m: float(np.mean([pair[cap][m] for pair in pairs])) for m in metrics} for cap in CAPS}
        differences = {}
        for metric in ("readings", "world_rmse", "exact_rmse"):
            delta = np.asarray([pair[CAPS[1]][metric] - pair[CAPS[0]][metric] for pair in pairs])
            differences[metric] = dict(mean=float(delta.mean()),
                                       ci95=np.quantile(delta[bootstrap].mean(axis=1), [.025, .975]).tolist())
        counts = {}
        for cap in CAPS:
            readings, freq = np.unique([pair[cap]["readings"] for pair in pairs], return_counts=True)
            counts[cap] = dict(zip(readings.tolist(), freq.tolist()))
        edges = {cap: sorted({pair[cap]["edges"] for pair in pairs}) for cap in CAPS}
        summary[family] = dict(fields=len(pairs), means=means, reading_counts=counts,
                               distinct_edges=edges, increased_cap_minus_baseline=differences)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("outputs/sensing-world-residential-oracle"))
    parser.add_argument("--mapping", type=Path, default=Path("outputs/sensing-world-residential-oracle-mapping"))
    parser.add_argument("--output", type=Path, default=Path("outputs/sensing-world-residential-oracle-budget"))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use an empty output directory.")
    config = json.loads((args.source / "protocol.json").read_text())
    if config["cap"] != CAPS[0] or config["edges"] != 26 or config["area"] != AREA:
        parser.error("Expected the 16-reading, 26-edge short-range oracle archive.")
    comparison = json.loads((Path(config["reference"]) / "protocol.json").read_text())
    model = json.loads(Path(comparison["model"]).read_text())
    weights = np.asarray(model["world_weights"]["world_4"])
    with (args.mapping / "metrics.csv").open() as stream:
        archive = {key(row): row for row in csv.DictReader(stream)}
    archived_routes = {route_key(route): route for route in json.loads((args.source / "routes.json").read_text())}
    with threadpool_limits(limits=1):
        rows, routes = evaluate(config, weights, archive, archived_routes)
    summary = summarize(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = dict(source=str(args.source), mapping=str(args.mapping), model=comparison["model"],
                    caps=CAPS, edges=config["edges"], area=AREA, draws_per_start=config["draws_per_start"])
    for name, payload in (("summary", summary), ("routes", routes), ("protocol", protocol)):
        (args.output / f"{name}.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    write_csv(args.output / "metrics.csv", rows)
    for family, result in summary.items():
        print(family, result)


if __name__ == "__main__":
    main()
