"""Sweep effective range on fixed observation routes with frozen predictors."""

import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from audit_sensing_world_modes import combine, posterior
from audit_sensing_world_shifts import correlation_area, match_scale, within_room_correlogram
from spatial_exploration.barrier_gp import graph_spectrum, unit_covariance
from spatial_exploration.baseline import write_csv
from spatial_exploration.basis_bayes import BayesianBasisModel
from spatial_exploration.sensing_world import kernel_basis
from spatial_exploration.sensing_world_generalization import CAP, EDGES, NOISE, STARTS, make_design, plan_route, wall_layout


AREAS = (1.5, 2.5, 3.5, 4.5, 5.5)
REGIMES = {"weak": .02, "persistent": 1.}
POWER, SEED, DRAWS = 2., 301001, 32


def evaluate(config, weights):
    rows, diagnostics = [], []
    for label, geometry in enumerate(config["test_geometry"]):
        layout = wall_layout(geometry)
        design = make_design(layout, config["test_seed"] + 100 * label)
        held = design.heldout
        spec = config["gp"]
        intercept = np.sqrt((design.features.shape[1] + 1) * spec["intercept"])
        gp = BayesianBasisModel(design.features, NOISE, spec["prior"], intercept_scale=intercept)
        world = np.einsum("mb,bij->mij", weights, kernel_basis(layout))
        correlogram = within_room_correlogram(layout)
        for mode, covariance in enumerate(world):
            diagnostics.append(dict(layout=label, kind="prior", regime="learned", value=mode,
                area=correlation_area(correlogram(covariance / covariance[0, 0])), scale=None))
        routes = {start: plan_route(design, gp, start, CAP, EDGES / (layout.size - 1), "rollout") for start in STARTS}
        rng = np.random.default_rng(SEED + label)
        for regime, p in REGIMES.items():
            eig, vectors = graph_spectrum(layout, p)

            def covariance(scale):
                return unit_covariance(vectors, (scale**2 + eig)**-POWER)

            factors = {}
            for area in AREAS:
                scale, matrix = match_scale(covariance, correlogram, correlation_area, area)
                factors[area] = np.linalg.cholesky(matrix)
                diagnostics.append(dict(layout=label, kind="field", regime=regime, value=area,
                    area=correlation_area(correlogram(matrix)), scale=scale))
            for start, route in routes.items():
                sites = np.asarray(route.sites)
                edge_count = round(route.distances[-1] * (layout.size - 1))
                if held[sites].any() or len(sites) > CAP or edge_count > EDGES:
                    raise ValueError("Invalid fixed observation route")
                z, noise = rng.standard_normal((2, DRAWS, len(held)))
                for area, factor in factors.items():
                    truth = z @ factor.T
                    readings = truth[:, sites] + NOISE * noise[:, sites]
                    rmse = []
                    for kernel in (gp.prior_covariance[None], world):
                        prediction, _ = combine(*posterior(kernel, sites, readings, held))
                        rmse.append(np.sqrt(np.mean((prediction - truth[:, held])**2, axis=1)))
                    rows.extend(dict(layout=label, regime=regime, start=start, draw=i, area=area,
                        gp=float(g), world=float(m), edges=edge_count, readings=len(sites))
                        for i, (g, m) in enumerate(zip(*rmse)))
    return rows, diagnostics


def summarize(rows):
    rng = np.random.default_rng(SEED + 100)
    result = {}
    for regime in ("all", *REGIMES):
        summary = []
        for area in AREAS:
            items = [r for r in rows if r["area"] == area and (regime == "all" or r["regime"] == regime)]
            strata = {}
            for i, row in enumerate(items):
                strata.setdefault((row["layout"], row["regime"], row["start"]), []).append(i)
            draws = np.concatenate([rng.choice(indices, (6000, len(indices))) for indices in strata.values()], axis=1)
            gp, world = np.array([[r["gp"], r["world"]] for r in items]).T
            gap = gp - world
            summary.append(dict(range_area=area, fields=len(items), gp=float(gp.mean()), world=float(world.mean()),
                improvement=float(gap.mean()), ci95=np.quantile(gap[draws].mean(axis=1), [.025, .975]).tolist()))
        result[regime] = summary
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, default=Path("outputs/sensing-world-generalization/frozen.json"))
    parser.add_argument("--model", type=Path, default=Path("outputs/sensing-world-modes/frozen.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/sensing-world-ranges"))
    args = parser.parse_args()
    config = json.loads(args.geometry.read_text())
    model = json.loads(args.model.read_text())
    if config["test_geometry"] != model["test_layouts"] or config["gp"] != model["gp"]:
        parser.error("The geometry and model archives must agree.")
    weights = np.asarray(model["world_weights"]["world_4"])
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use an empty output directory.")
    args.output.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        rows, diagnostics = evaluate(config, weights)
    write_csv(args.output / "metrics.csv", rows)
    write_csv(args.output / "range.csv", diagnostics)
    summary = summarize(rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for row in summary["all"]:
        print(row)


if __name__ == "__main__":
    main()
