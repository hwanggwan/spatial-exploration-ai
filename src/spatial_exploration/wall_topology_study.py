"""Isolate the prediction effect of wall edges at fixed acquisitions and ridge settings."""

import argparse
from dataclasses import replace
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .barrier_deepkriging_study import (COUNTS, GRAPH_LENGTH, TEST_DOORS, NeuralMapperConfig,
                                         dump, generate_split, graph_heat_features, make_design, score)
from .barrier_ridge_study import INTERCEPT_RATIOS, PRIOR_VARIANCES, calibrate, ridge_posterior
from .barrier_study import BUDGETS, DOORS, KAPPAS, NOISE, REGIMES, SIZE
from .baseline import write_csv


MODELS = ("wendland", "open_graph", "wall_graph")
SEEDS = {"tune": 710000, "calibrate": 810000, "test": 910000}
CONTRASTS = {"open_minus_wendland": ("open_graph", "wendland"),
             "wall_minus_open": ("wall_graph", "open_graph")}


def topology_design(door):
    design = make_design(door, NeuralMapperConfig())
    base = design.features["deepkriging"]
    open_features = np.column_stack((base, graph_heat_features(design.layout, permeability=1.0)))
    return replace(design, features={"wendland": base, "open_graph": open_features.astype(np.float32),
                                     "wall_graph": design.features["graph_deepkriging"]})


def tune(designs, acquisitions, truth):
    candidates = []
    for model in MODELS:
        for count in COUNTS:
            for prior in PRIOR_VARIANCES:
                for intercept in INTERCEPT_RATIOS:
                    errors = []
                    for door, design in designs.items():
                        batch = acquisitions[door, count]
                        mean = ridge_posterior(design.features[model], batch.indices, batch.values,
                                               NOISE, prior, intercept, with_variance=False)["mean"]
                        residual = (mean - truth[door])[:, design.heldout]
                        errors.extend(np.sqrt(np.mean(residual**2, axis=1)))
                    candidates.append(dict(model=model, count=count, prior=prior,
                                           intercept=intercept, rmse=float(np.mean(errors))))
    by_key = {(r["count"], r["model"], r["prior"], r["intercept"]): r["rmse"] for r in candidates}
    pooled, independent = {}, {}
    for count in COUNTS:
        pairs = [(prior, intercept) for prior in PRIOR_VARIANCES for intercept in INTERCEPT_RATIOS]
        common = min(pairs, key=lambda p: sum(by_key[(count, m, *p)] for m in MODELS[1:]))
        for model in MODELS:
            independent[model, count] = min(pairs, key=lambda p: by_key[(count, model, *p)])
            pooled[model, count] = common if model in MODELS[1:] else independent[model, count]
    return candidates, {"shared_settings": pooled, "separate_settings": independent}


def fit(designs, acquisitions, settings):
    predictions, timing = {}, {}
    for (door, count), batch in acquisitions.items():
        for model in MODELS:
            key = door, count, model
            before = time.perf_counter()
            predictions[key] = ridge_posterior(designs[door].features[model], batch.indices, batch.values,
                                                NOISE, *settings[model, count])
            timing[key] = time.perf_counter() - before
    return predictions, timing


def summarize(rows):
    summary, contrasts = [], []
    rng = np.random.default_rng(91104)
    metrics = ("rmse", "near_wall_rmse", "coverage", "raw_coverage", "width", "interval_score",
               "distance", "amortized_fit_ms")

    def contrast(count, regime, metric, name, diff, doors):
        samples = np.concatenate([rng.choice(diff[doors == d], (8000, int((doors == d).sum())))
                                  for d in np.unique(doors)], axis=1).mean(axis=1)
        return dict(count=count, regime=regime, metric=metric, contrast=name,
                    difference=float(diff.mean()), interval=np.quantile(samples, [.025, .975]).tolist(),
                    negative_fields=int((diff < 0).sum()),
                    by_door={int(d): float(diff[doors == d].mean()) for d in np.unique(doors)})

    for count in COUNTS:
        grouped = {(regime, model): {(r["door"], r["seed"]): r for r in rows if
                   (r["regime"], r["model"], r["measurements"]) == (regime, model, count)}
                   for regime in REGIMES for model in MODELS}
        keys = sorted(grouped["barrier", "wall_graph"])
        assert keys and all(set(keys) == set(group) for group in grouped.values())
        doors = np.array([d for d, _ in keys])
        for (regime, model), group in grouped.items():
            summary.append(dict(count=count, regime=regime, model=model, fields=len(keys),
                                **{m: float(np.mean([group[k][m] for k in keys])) for m in metrics}))
        for metric in ("rmse", "near_wall_rmse"):
            values = {key: np.array([group[k][metric] for k in keys]) for key, group in grouped.items()}
            for regime in REGIMES:
                for name, (a, b) in CONTRASTS.items():
                    contrasts.append(contrast(count, regime, metric, name,
                                              values[regime, a] - values[regime, b], doors))
            wall_delta = {regime: values[regime, "wall_graph"] - values[regime, "open_graph"]
                          for regime in REGIMES}
            contrasts.append(contrast(count, "barrier_minus_transparent", metric, "wall_minus_open",
                                      wall_delta["barrier"] - wall_delta["transparent"], doors))
    return summary, contrasts


def render(output, contrasts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), sharex=True, sharey=True, layout="constrained")
    labels = ["Wall weakens dependence", "Dependence crosses wall"]
    for ax, metric in zip(axes, ("rmse", "near_wall_rmse")):
        ax.axvline(0, color="0.45", linewidth=1)
        for j, (regime, count) in enumerate((r, n) for r in REGIMES for n in COUNTS):
            result = next(r for r in contrasts if (r["count"], r["regime"], r["metric"], r["contrast"])
                          == (count, regime, metric, "wall_minus_open"))
            color = "#227689" if regime == "barrier" else "#b55238"
            ax.plot(result["interval"], [j, j], color=color, lw=2)
            ax.plot(result["difference"], j, "o", color=color)
        ax.set_yticks(range(4), [f"{label} · {n} readings" for label in labels for n in COUNTS])
        ax.set_title("All reserved sites" if metric == "rmse" else "Reserved sites next to wall")
        ax.grid(axis="x", alpha=0.2)
    axes[0].invert_yaxis()
    fig.supxlabel("Wall graph minus open graph RMSE (negative favors modeling the wall)")
    fig.savefig(output / "wall-effect.png", dpi=175)
    plt.close(fig)


def run(output, fields=128, development_fields=32):
    if not 1 <= min(fields, development_fields) <= max(fields, development_fields) < 1000:
        raise ValueError("Choose 1–999 fields per layout")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Use a fresh output directory")
    with threadpool_limits(limits=1):
        return run_frozen(output, fields, development_fields)


def run_frozen(output, fields, dev_fields):
    protocol = dict(grid=SIZE, development_doors=DOORS, test_doors=TEST_DOORS,
                    development_fields_per_layout=dev_fields, test_fields_per_layout=fields,
                    seeds=SEEDS, budgets=BUDGETS, noise_std=NOISE, true_permeability=REGIMES,
                    kappa=KAPPAS, graph_length=GRAPH_LENGTH, graph_permeability={"open_graph": 1, "wall_graph": 0},
                    priors=PRIOR_VARIANCES, intercept_ratios=INTERCEPT_RATIOS,
                    primary="Shared graph settings pooled on tuning RMSE. Same fixed acquisitions across arms.",
                    primary_contrast="Wall graph minus open graph latent RMSE at n=16 on barrier fields.",
                    calibration="Separate phase, scalar per-arm marginal standard-deviation adjustment.")
    dump(output / "protocol.json", protocol)
    designs = {door: topology_design(door) for door in DOORS}
    acquisitions, truth = generate_split(designs, dev_fields, SEEDS["tune"], output, "tune")
    candidates, settings = tune(designs, acquisitions, truth)
    write_csv(output / "tuning.csv", candidates)
    acquisitions, truth = generate_split(designs, dev_fields, SEEDS["calibrate"], output, "calibrate")
    scales = {}
    for arm, spec in settings.items():
        predictions, _ = fit(designs, acquisitions, spec)
        scales[arm] = calibrate(designs, predictions, truth, COUNTS, models=MODELS)
    dump(output / "frozen.json", {**protocol,
        "settings": {arm: {f"{m}-n{n}": dict(prior=p, intercept=b) for (m, n), (p, b) in spec.items()}
                     for arm, spec in settings.items()},
        "std_multipliers": {arm: {f"{m}-n{n}": v for (m, n), v in spec.items()}
                            for arm, spec in scales.items()}})

    designs = {door: topology_design(door) for door in TEST_DOORS}
    acquisitions, truth = generate_split(designs, fields, SEEDS["test"], output, "test")
    summaries, all_contrasts, archive = {}, {}, {}
    for arm, spec in settings.items():
        predictions, timing = fit(designs, acquisitions, spec)
        rows = score(designs, acquisitions, predictions, truth, timing, scales[arm])
        write_csv(output / f"{arm}-metrics.csv", rows)
        summaries[arm], all_contrasts[arm] = summarize(rows)
        archive.update({f"{arm}_d{d}_n{n}_{m}_{key}": v for (d, n, m), pred in predictions.items()
                        for key, v in pred.items()})
    dump(output / "summary.json", summaries)
    dump(output / "contrasts.json", all_contrasts)
    np.savez_compressed(output / "predictions.npz", **archive)
    render(output, all_contrasts["shared_settings"])
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/wall-topology-study"))
    parser.add_argument("--fields-per-layout", type=int, default=128)
    parser.add_argument("--development-fields", type=int, default=32)
    args = parser.parse_args()
    for row in run(args.output, args.fields_per_layout, args.development_fields)["shared_settings"]:
        print(f"{row['regime']:11s} n={row['count']:2} {row['model']:10s} "
              f"RMSE {row['rmse']:.4f} · near-wall {row['near_wall_rmse']:.4f}")


if __name__ == "__main__":
    main()
