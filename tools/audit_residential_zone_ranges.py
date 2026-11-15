"""Test observation-driven room and corridor range weights on fixed routes."""

import argparse
import json
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from audit_residential_holdout_distance import CAP, FAMILY, match_short_components, short_mode_indices
from audit_residential_oracle_budget import checked_route
from audit_residential_routes import AREAS, CONDUCTANCES, LayoutCase, SEED, STARTS, design_for, field_covariances, floor_plan
from audit_sensing_world_modes import combine, posterior
from spatial_exploration.baseline import write_csv
from spatial_exploration.sensing_world import kernel_basis
from spatial_exploration.sensing_world_study import NOISE


SIM_SEED = 309073
DRAWS = 12
RANGE_WEIGHTS = (0.1, 0.9)
GLOBAL_WEIGHTS = (0.0, *RANGE_WEIGHTS, 0.5, 1.0)
MODELS = ("global", "local", "known_zones", "exact")


def weighted_covariance(short, long, short_weight):
    u = np.sqrt(short_weight)
    v = np.sqrt(1 - short_weight)
    return short * np.outer(u, u) + long * np.outer(v, v)


def range_profiles(n_zones):
    global_profiles = [tuple([weight] * n_zones) for weight in GLOBAL_WEIGHTS]
    mixed_profiles = [profile for profile in product(RANGE_WEIGHTS, repeat=n_zones) if len(set(profile)) > 1]
    return np.array([*global_profiles, *mixed_profiles])


def model_covariances(layout, model, profiles, *, correlogram=None):
    weights = np.asarray(model["world_weights"]["world_4"])
    modes = model["modes"]
    base = np.einsum("mb,bij->mij", weights, kernel_basis(layout))
    adjusted, _ = match_short_components(layout, modes, weights, base, AREAS[0], correlogram=correlogram)
    short = set(short_mode_indices(modes))
    walls = sorted({wall for wall, _ in modes})
    banks = []
    for wall in walls:
        short_idx = next(i for i, (name, _) in enumerate(modes) if name == wall and i in short)
        long_idx = next(i for i, (name, _) in enumerate(modes) if name == wall and i not in short)
        banks.append([weighted_covariance(adjusted[short_idx], adjusted[long_idx], profile[layout.room])
                      for profile in profiles])
    return np.asarray(banks).transpose(1, 0, 2, 3)


def map_predictions(means, log_likelihood, variances, log_prior):
    mean, probs = combine(means, log_likelihood + log_prior)
    variance = probs @ variances + np.einsum("bm,bmn->bn", probs, (means - mean[:, None, :])**2)
    return mean, np.sqrt(np.maximum(variance, 0.0)), probs


def local_mode_prior(n_profiles, n_walls):
    n_global = len(GLOBAL_WEIGHTS)
    profile_prior = np.concatenate((
        np.full(n_global, 0.5 / n_global),
        np.full(n_profiles - n_global, 0.5 / (n_profiles - n_global)),
    ))
    return np.repeat(profile_prior / n_walls, n_walls)


def evaluate_layout(variant, config, model, archived_routes, *, example_key=None):
    rng = np.random.default_rng(SIM_SEED + variant)
    layout = floor_plan(LayoutCase(FAMILY, variant))
    design = design_for(layout, SEED + variant)
    held = np.flatnonzero(design.heldout)
    profiles = range_profiles(int(layout.room.max()) + 1)
    banks = model_covariances(layout, model, profiles)
    n_profiles, n_walls, n_sites, _ = banks.shape
    kernels = banks.reshape(-1, n_sites, n_sites)
    n_global = len(GLOBAL_WEIGHTS)
    scenarios = {GLOBAL_WEIGHTS.index(0.0): "long", GLOBAL_WEIGHTS.index(1.0): "short"}
    scenarios.update({p: "mixed" for p in range(n_global, n_profiles)})
    log_prior = np.log(local_mode_prior(n_profiles, n_walls))
    global_count = n_global * n_walls
    global_prior = np.full(global_count, -np.log(global_count))
    profile_modes = np.repeat(profiles, n_walls, axis=0)
    rows, diagnostics = [], []
    example = None

    for regime, conductance in CONDUCTANCES.items():
        bank = field_covariances(layout, conductance)
        true_cov = {p: weighted_covariance(bank[AREAS[0]][1], bank[AREAS[1]][1], profiles[p, layout.room])
                    for p in scenarios}
        factors = {p: np.linalg.cholesky(cov) for p, cov in true_cov.items()}
        for start in STARTS:
            route = checked_route(archived_routes[variant, start], design, CAP, config["edges"])
            sites = np.asarray(route["sites"])
            if route["edges"] != config["edges"]:
                raise ValueError("Expected the fixed 26-edge route")
            nearest = np.rint(layout.distances[np.ix_(held, sites)].min(axis=1) * (layout.size - 1)).astype(int)
            middle = (nearest >= 2) & (nearest <= 4)
            if not middle.any():
                raise ValueError("Route has no 2–4-edge hold-outs")

            for p, scenario in scenarios.items():
                profile = profiles[p]
                draws = rng.standard_normal((2, DRAWS, n_sites))
                truth = draws[0] @ factors[p].T
                readings = truth[:, sites] + NOISE * draws[1][:, sites]
                held_truth = truth[:, held]
                before = perf_counter()
                means, likelihood, variances = posterior(
                    kernels[:global_count], sites, readings, held, with_variance=True,
                )
                global_prediction = map_predictions(means, likelihood, variances, global_prior)
                global_ms = (perf_counter() - before) * 1000 / DRAWS

                before = perf_counter()
                means, likelihood, variances = posterior(kernels, sites, readings, held, with_variance=True)
                local_prediction = map_predictions(means, likelihood, variances, log_prior)
                local_ms = (perf_counter() - before) * 1000 / DRAWS

                known = slice(p*n_walls, (p+1)*n_walls)
                known_prediction = map_predictions(means[:, known], likelihood[:, known], variances[known], 0.0)
                true_mean, _, true_var = posterior(true_cov[p][None], sites, readings, held, with_variance=True)
                exact_prediction = (true_mean[:, 0], np.broadcast_to(np.sqrt(true_var[0]), held_truth.shape), None)
                mapped = {}
                for arm, (mean, sd, _) in zip(MODELS, (global_prediction, local_prediction, known_prediction, exact_prediction)):
                    error = mean - held_truth
                    mapped.update({
                        f"{arm}_rmse": np.sqrt(np.mean(error**2, axis=1)),
                        f"{arm}_mid_mse": np.mean(error[:, middle]**2, axis=1),
                        f"{arm}_coverage": np.mean(abs(error) <= 1.96 * sd, axis=1),
                        f"{arm}_width": np.mean(3.92 * sd, axis=1),
                    })
                inferred = local_prediction[2] @ profile_modes
                if example_key is not None and (regime, start, p) == example_key[:3]:
                    draw = example_key[3]
                    every_site = np.arange(n_sites)
                    means, likelihood, variances = posterior(
                        kernels, sites, readings[draw:draw+1], every_site, with_variance=True,
                    )
                    global_map, _, _ = map_predictions(means[:, :global_count], likelihood[:, :global_count],
                                                       variances[:global_count], global_prior)
                    local_map, _, _ = map_predictions(means, likelihood, variances, log_prior)
                    np.testing.assert_allclose(global_map[0, held], global_prediction[0][draw], atol=1e-12, rtol=0)
                    np.testing.assert_allclose(local_map[0, held], local_prediction[0][draw], atol=1e-12, rtol=0)
                    example = dict(layout=layout, sites=sites, held=held, middle=middle, truth=truth[draw],
                                   global_mean=global_map[0], local_mean=local_map[0],
                                   true_weights=profile, local_weights=inferred[draw])
                diagnostics.append(dict(variant=variant, regime=regime, start=start, profile=p, scenario=scenario,
                                        global_ms=global_ms, local_ms=local_ms))
                for draw, row in enumerate(inferred):
                    rows.append(dict(variant=variant, regime=regime, start=start, profile=p, scenario=scenario,
                                     draw=draw, readings=len(sites), edges=route["edges"], mid_points=int(middle.sum()),
                                     **{name: value[draw] for name, value in mapped.items()},
                                     **{f"zone_{j}_true": value for j, value in enumerate(profile)},
                                     **{f"zone_{j}_weight": value for j, value in enumerate(row)}))
    return rows, diagnostics, example


def summarize(rows, timing):
    rng = np.random.default_rng(SIM_SEED + 10)
    result = {}
    for scenario in ("mixed", "short", "long"):
        fields = [row for row in rows if row["scenario"] == scenario]
        strata = {}
        for i, row in enumerate(fields):
            strata.setdefault(tuple(row[k] for k in ("variant", "regime", "start", "profile")), []).append(i)
        boot = np.concatenate([rng.choice(ids, (3000, len(ids))) for ids in strata.values()], axis=1)
        result[scenario] = dict(fields=len(fields), mean_readings=float(np.mean([row["readings"] for row in fields])))
        stats = result[scenario]
        counts = np.asarray([row["mid_points"] for row in fields])
        stats["models"] = {}
        mid_sums, rmses = {}, {}
        for arm in MODELS:
            rmse = np.asarray([row[f"{arm}_rmse"] for row in fields])
            mid = np.asarray([row[f"{arm}_mid_mse"] for row in fields]) * counts
            stats["models"][arm] = dict(
                rmse=float(rmse.mean()), mid_rmse=float(np.sqrt(mid.sum() / counts.sum())),
                coverage=float(np.mean([row[f"{arm}_coverage"] for row in fields])),
                width=float(np.mean([row[f"{arm}_width"] for row in fields])),
            )
            mid_sums[arm], rmses[arm] = mid, rmse
        stats["paired_local_minus_global"] = {}
        for metric, delta in (
            ("rmse", (rmses["local"] - rmses["global"])[boot].mean(axis=1)),
            ("mid_rmse", np.sqrt(mid_sums["local"][boot].sum(axis=1) / counts[boot].sum(axis=1))
             - np.sqrt(mid_sums["global"][boot].sum(axis=1) / counts[boot].sum(axis=1))),
        ):
            stats["paired_local_minus_global"][metric] = dict(
                mean=stats["models"]["local"][metric] - stats["models"]["global"][metric],
                ci95=np.quantile(delta, [.025, .975]).tolist(),
            )
        regional = [(row[name], row[name.replace("_true", "_weight")]) for row in fields
                    for name in row if name.startswith("zone_") and name.endswith("_true")]
        actual, estimated = np.asarray(regional).T
        stats["estimated_short_weight"] = {
            str(weight): float(estimated[actual == weight].mean()) for weight in np.unique(actual)
        }
        stats["timing_ms"] = {name: float(np.mean([row[name] for row in timing if row["scenario"] == scenario]))
                              for name in ("global_ms", "local_ms")}
    return result


def load_archive(source):
    config = json.loads((source / "protocol.json").read_text())
    model = json.loads(Path(config["model"]).read_text())
    # Use a single pre-existing route for each layout/start regardless of the
    # new field, the range assignment, or its wall-conductance condition.
    routes = {(row["variant"], row["start"]): {k: v for k, v in row.items() if k not in ("cap", "edges", "readings")}
              for row in json.loads((source / "routes.json").read_text())
              if row["family"] == FAMILY and row["cap"] == CAP and row["regime"] == "weak"}
    return config, model, routes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("outputs/sensing-world-residential-oracle-budget"))
    parser.add_argument("--output", type=Path, default=Path("outputs/sensing-world-residential-zone-ranges"))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use an empty output directory.")
    config, model, routes = load_archive(args.source)
    rows, timing = [], []
    with threadpool_limits(limits=1):
        for variant in sorted({variant for variant, _ in routes}):
            fields, diagnostics, _ = evaluate_layout(variant, config, model, routes)
            rows.extend(fields)
            timing.extend(diagnostics)
    summary = summarize(rows, timing)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "fields.csv", rows)
    protocol = dict(source=str(args.source), family=FAMILY, cap=CAP, edges=config["edges"], model=config["model"],
                    route_archive_condition="weak", sim_seed=SIM_SEED, draws_per_profile_route_and_regime=DRAWS,
                    range_areas=AREAS, mixed_short_weights=RANGE_WEIGHTS, global_short_weights=GLOBAL_WEIGHTS,
                    local_prior_mass_on_uniform_profiles=0.5)
    for name, payload in (("protocol", protocol), ("summary", summary)):
        (args.output / f"{name}.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    for scenario, stats in summary.items():
        print(scenario, stats)


if __name__ == "__main__":
    main()
