"""Compare wall and open graph Bayesian ridge on self-chosen measurement routes."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .barrier_deepkriging_study import dump
from .barrier_gp import transport_covariance
from .barrier_study import (BUDGETS, DOORS, KAPPAS, NOISE, REGIMES, SIZE, STARTS,
                            collect, render_example, route_record)
from .baseline import GPBelief, write_csv
from .basis_bayes import BayesianBasisModel
from .wall_topology_study import GRAPH_LENGTH, TEST_DOORS, topology_design


MODELS = ("open_graph", "wall_graph")
CALIBRATION_SEED = 1110000
TEST_SEED = 1210000


def load_settings(reference):
    content = Path(reference).read_bytes()
    frozen = json.loads(content)
    if (tuple(map(tuple, frozen["budgets"])) != BUDGETS or frozen["noise_std"] != NOISE
            or frozen["graph_length"] != GRAPH_LENGTH or frozen["grid"] != SIZE
            or frozen["graph_permeability"] != {"open_graph": 1, "wall_graph": 0}):
        raise ValueError("Frozen settings do not match the closed-loop design")
    settings = frozen["settings"]["shared_settings"]
    if any(settings[f"open_graph-n{n}"] != settings[f"wall_graph-n{n}"] for n, _ in BUDGETS):
        raise ValueError("The two ridge mappers must share all hyperparameters")
    return {n: settings[f"open_graph-n{n}"] for n, _ in BUDGETS}, hashlib.sha256(content).hexdigest()


def ridge_models(design, settings):
    return {model: BayesianBasisModel(design.features[model], NOISE, settings["prior"],
                intercept_scale=np.sqrt((design.features[model].shape[1] + 1) * settings["intercept"]))
            for model in MODELS}


def explore_ridge(design, model, start, budget, sense):
    """Update the exact finite-feature posterior with each newly acquired reading."""
    belief = GPBelief(model.prior_covariance, model.noise_std)

    def update(route, readings):
        belief.observe(route[-1], readings[-1])
        return belief, None

    return collect(design.layout, start, ~design.heldout, budget, sense, update)


def scenes(design, fields, seed_base):
    door = design.layout.door_row
    factors = {(regime, kappa): np.linalg.cholesky(transport_covariance(design.layout, permeability, kappa))
               for regime, permeability in REGIMES.items() for kappa in KAPPAS}
    for field in range(fields):
        seed = seed_base + 1000 * door + field
        start = STARTS[field % len(STARTS)]
        kappa = KAPPAS[(field // len(STARTS)) % len(KAPPAS)]
        z, noise = np.random.default_rng(seed).standard_normal((2, SIZE**2))
        for regime in REGIMES:
            truth = factors[regime, kappa] @ z
            yield dict(door=door, field=field, seed=seed, start=start, regime=regime,
                       kappa=kappa), truth, truth + NOISE * noise


def calibrate(designs, settings, fields):
    ratios = {(m, route, n): [] for m in MODELS for route in MODELS for n, _ in BUDGETS}
    for design in designs.values():
        banks = {n: ridge_models(design, settings[n]) for n, _ in BUDGETS}
        for meta, truth, sensor in scenes(design, fields, CALIBRATION_SEED):
            mask = design.heldout
            for n, distance in BUDGETS:
                for route_model, ridge in banks[n].items():
                    route, _, values, belief, *_ = explore_ridge(
                        design, ridge, meta["start"], (n, distance), sensor.__getitem__)
                    for model, other in banks[n].items():
                        posterior = belief if model == route_model else other.condition(route, values)
                        ratios[model, route_model, n].extend(
                            np.abs(truth[mask] - posterior.mean[mask]) / np.maximum(posterior.std[mask], 1e-8))
    return {key: float(np.quantile(values, .95) / 1.96) for key, values in ratios.items()}


def metrics(truth, belief, design, scale):
    held = design.heldout
    error = np.abs(truth[held] - belief.mean[held])
    half = 1.96 * belief.std[held]
    interval = scale * half
    return dict(rmse=float(np.sqrt(np.mean(error**2))),
                near_wall_rmse=float(np.sqrt(np.mean(error[design.near_wall[held]]**2))),
                raw_coverage=float(np.mean(error <= half)), coverage=float(np.mean(error <= interval)),
                width=float(np.mean(2 * interval)),
                interval_score=float(np.mean(2 * interval + 40 * np.maximum(error - interval, 0))),
                modeled_grid_variance=float(np.diag(belief.covariance).mean()))


def summarize(rows, acquisitions):
    rng = np.random.default_rng(12101)
    results = []
    routes = {}
    for row in acquisitions:
        key = row["door"], row["start"], row["cap"], row["route_model"]
        run = row["seed"], row["regime"]
        routes.setdefault(key, {}).setdefault(run, []).append(row["site"])
    # With frozen Gaussian parameters, covariance updates depend only on locations.
    for by_run in routes.values():
        assert len({tuple(route) for route in by_run.values()}) == 1
    for regime in REGIMES:
        for cap, _ in BUDGETS:
            groups = {(m, r): {} for m in MODELS for r in MODELS}
            for row in rows:
                if (row["regime"], row["cap"]) == (regime, cap):
                    groups[row["model"], row["route_model"]][row["door"], row["seed"]] = row
            keys = sorted(groups[MODELS[0], MODELS[0]])
            assert keys and all(set(keys) == set(g) for g in groups.values())
            strata = {}
            for i, key in enumerate(keys):
                row = groups[MODELS[0], MODELS[0]][key]
                strata.setdefault((row["door"], row["start"], row["kappa"]), []).append(i)

            def compare(a, b, metric):
                delta = np.array([groups[a][k][metric] - groups[b][k][metric] for k in keys])
                boot = np.concatenate([rng.choice(delta[idx], (6000, len(idx)), replace=True)
                                       for idx in strata.values()], axis=1).mean(axis=1)
                return dict(difference=float(delta.mean()), interval=np.quantile(boot, [.025, .975]).tolist(),
                            negative_fields=int(np.sum(delta < -1e-12)))

            open_model, wall_model = MODELS
            closed = {m: {name: float(np.mean([r[name] for r in groups[m, m].values()]))
                          for name in ("rmse", "near_wall_rmse", "distance", "measurements", "crosses_rooms",
                                       "raw_coverage", "coverage", "width", "interval_score", "update_ms",
                                       "planning_ms")}
                      for m in MODELS}
            route_changes = sum(
                tuple(next(iter(runs.values()))) != tuple(next(iter(routes[d, s, cap, wall_model].values())))
                for (d, s, n, m), runs in routes.items() if n == cap and m == open_model)
            results.append(dict(regime=regime, cap=cap, fields=len(keys), closed=closed,
                changed_routes_out_of_eight=route_changes,
                crossover_rmse={m: {r: float(np.mean([row["rmse"] for row in groups[m, r].values()]))
                                    for r in MODELS} for m in MODELS},
                closed_rmse=compare((wall_model, wall_model), (open_model, open_model), "rmse"),
                closed_near_wall=compare((wall_model, wall_model), (open_model, open_model), "near_wall_rmse"),
                route_effect={m: compare((m, wall_model), (m, open_model), "rmse") for m in MODELS},
                mapper_effect={r: compare((wall_model, r), (open_model, r), "rmse") for r in MODELS}))
    return results


def run(reference, output, fields=128, calibration_fields=32):
    if not all(8 <= n < 1000 and n % 8 == 0 for n in (fields, calibration_fields)):
        raise ValueError("Use multiples of eight between 8 and 999 to balance starts and field scales")
    settings, digest = load_settings(reference)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Use a fresh output directory")
    protocol = dict(reference_sha256=digest, settings=settings, budgets=BUDGETS, field_scales=KAPPAS,
                    test_doors=TEST_DOORS, starts=STARTS, regimes=REGIMES, noise_std=NOISE,
                    test_seed=TEST_SEED, test_fields_per_layout=fields,
                    calibration_doors=DOORS, calibration_seed=CALIBRATION_SEED,
                    calibration_fields_per_layout=calibration_fields,
                    acquisition="Greedy whole-grid posterior variance reduction per legal graph distance",
                    primary="Final paired held-out RMSE and actual walking distance; same budgets",
                    evaluation="28 reserved sites; raw covariance drives planning, calibration only scores intervals")
    dump(output / "protocol.json", protocol)
    with threadpool_limits(limits=1):
        scales = calibrate({d: topology_design(d) for d in DOORS}, settings, calibration_fields)
        dump(output / "frozen.json", {**protocol, "std_scales": {
            f"{m}-on-{r}-n{n}": v for (m, r, n), v in scales.items()}})
        rows, acquisitions, field_archive, example = [], [], {}, None
        for door in TEST_DOORS:
            design = topology_design(door)
            banks = {n: ridge_models(design, settings[n]) for n, _ in BUDGETS}
            records = list(scenes(design, fields, TEST_SEED))
            field_archive.update({f"d{door}_{key}": value for key, value in {
                "truth": np.asarray([truth for _, truth, _ in records]), "heldout": design.heldout,
                "seeds": np.asarray([meta["seed"] for meta, _, _ in records]),
                "regimes": np.asarray([meta["regime"] for meta, _, _ in records]),
            }.items()})
            for meta, truth, sensor in records:
                for n, distance in BUDGETS:
                    display = []
                    for route_model, ridge in banks[n].items():
                        route, travel, values, belief, _, updates, planning = explore_ridge(
                            design, ridge, meta["start"], (n, distance), sensor.__getitem__)
                        for i, (site, cost, value) in enumerate(zip(route, travel, values)):
                            acquisitions.append(dict(**meta, cap=n, route_model=route_model, step=i,
                                                     site=site, distance=cost, value=value))
                        for model, mapper in banks[n].items():
                            own = model == route_model
                            posterior = belief if own else mapper.condition(route, values)
                            rows.append(dict(**meta, cap=n, distance_cap=distance, model=model, route_model=route_model,
                                **route_record(design.layout, route, travel),
                                **metrics(truth, posterior, design, scales[model, route_model, n]),
                                update_ms=updates * 1000 if own else None,
                                planning_ms=planning * 1000 if own else None))
                        label = "Open graph ridge" if route_model == MODELS[0] else "Wall graph ridge"
                        display.append((label, belief.mean, route))
                    if (example is None and n == 16 and meta["regime"] == "barrier"
                            and display[0][2] != display[1][2]):
                        example = (design.layout, truth, design.heldout, display, meta["field"])
    summary = summarize(rows, acquisitions)
    dump(output / "summary.json", summary)
    write_csv(output / "metrics.csv", rows)
    write_csv(output / "acquisitions.csv", acquisitions)
    np.savez_compressed(output / "evaluation-only.npz", **field_archive)
    if example is not None:
        render_example(output, *example)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("outputs/wall-topology-study/frozen.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/wall-closed-loop-study"))
    parser.add_argument("--fields-per-layout", type=int, default=128)
    parser.add_argument("--calibration-fields-per-layout", type=int, default=32)
    args = parser.parse_args()
    for row in run(args.reference, args.output, args.fields_per_layout, args.calibration_fields_per_layout):
        print(row["regime"], row["cap"], {m: {key: v[key] for key in ("rmse", "distance", "measurements")}
                                         for m, v in row["closed"].items()})


if __name__ == "__main__":
    main()
