"""Feature-matched Bayesian ridge controls on the frozen graph DeepKriging data."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .barrier_deepkriging_study import Acquisitions, NeuralMapperConfig, dump, make_design, score
from .baseline import write_csv
from .basis_bayes import BayesianBasisModel


FEATURE_MODELS = {"wendland_ridge": "deepkriging", "graph_ridge": "graph_deepkriging"}
PRIOR_VARIANCES = tuple(float(v) for v in np.logspace(-2, 2, 17))
INTERCEPT_RATIOS = (0.0, 0.25, 1.0)
CONTRASTS = {
    "graph_features_in_ridge": ("graph_ridge", "wendland_ridge"),
    "graph_features_in_mlp": ("graph_deepkriging", "deepkriging"),
    "mlp_vs_ridge_with_wendland": ("deepkriging", "wendland_ridge"),
    "mlp_vs_ridge_with_graph": ("graph_deepkriging", "graph_ridge"),
    "graph_ridge_vs_geometry_gp": ("graph_ridge", "geometry_gp"),
}


def ridge_posterior(features, indices, readings, noise_std, prior_variance, intercept_ratio,
                    with_variance=True):
    """Condition the linear coefficient prior, using an n-by-n solve per map."""
    p = features.shape[1]
    model = BayesianBasisModel(features, noise_std, prior_variance,
                               intercept_scale=np.sqrt((p + 1) * intercept_ratio))
    prior = model.prior_covariance
    blocks = prior[indices[..., None], indices[:, None, :]] + noise_std**2 * np.eye(indices.shape[1])
    factors = np.linalg.cholesky(blocks)
    whitened_y = np.linalg.solve(factors, readings[..., None])
    cross = prior[indices]
    if with_variance:
        whitened_cross = np.linalg.solve(factors, cross)
        mean = (whitened_cross.transpose(0, 2, 1) @ whitened_y)[..., 0]
        variance = prior.diagonal() - np.sum(whitened_cross**2, axis=1)
        return dict(mean=mean, raw_variance=np.maximum(variance, 0.0))
    weights = np.linalg.solve(factors.transpose(0, 2, 1), whitened_y)
    return dict(mean=(cross.transpose(0, 2, 1) @ weights)[..., 0])


def read_acquisitions(source, phase, designs, budgets):
    batches = {}
    with np.load(source / f"{phase}-observations.npz") as archive:
        for door, design in designs.items():
            for count, distance in budgets:
                prefix = f"d{door}_n{count}_"
                batch = Acquisitions(**{key: archive[prefix + key] for key in Acquisitions.__dataclass_fields__})
                if (batch.indices.shape != batch.values.shape or batch.indices.shape[1] != count
                        or design.heldout[batch.indices].any()
                        or np.any(batch.travel > distance + 1e-12)
                        or np.any(batch.indices[:, 0] != batch.starts)):
                    raise ValueError("Reference acquisitions violate the frozen design")
                # Compare the exact legal routes, not just their reading count and endpoints.
                for indices, start, travel in zip(batch.indices, batch.starts, batch.travel):
                    route, distances = design.routes[int(start), count]
                    if not np.array_equal(indices, route) or not np.isclose(travel, distances[-1], atol=1e-12, rtol=0):
                        raise ValueError("Acquisition archive does not match the shared routes")
                batches[door, count] = batch
    return batches


def read_truth(source, phase, designs):
    with np.load(source / f"{phase}-evaluation-only.npz") as archive:
        for door, design in designs.items():
            if (not np.array_equal(archive[f"d{door}_heldout"], design.heldout)
                    or not np.array_equal(archive[f"d{door}_near_wall"], design.near_wall)):
                raise ValueError("Reference hold-out mask changed")
        return {door: archive[f"d{door}_truth"] for door in designs}


def select(designs, acquired, truth, counts, noise):
    tuning, selected = [], {}
    for model, features in FEATURE_MODELS.items():
        for count in counts:
            candidates = []
            for variance in PRIOR_VARIANCES:
                for ratio in INTERCEPT_RATIOS:
                    errors = []
                    for door, design in designs.items():
                        batch = acquired[door, count]
                        mean = ridge_posterior(design.features[features], batch.indices, batch.values,
                                               noise, variance, ratio, with_variance=False)["mean"]
                        mask = design.heldout
                        errors.extend(np.sqrt(np.mean((mean[:, mask] - truth[door][:, mask])**2, axis=1)))
                    candidates.append(dict(model=model, count=count, prior_variance=variance,
                                           intercept_ratio=ratio, mean_rmse=float(np.mean(errors))))
            tuning.extend(candidates)
            selected[model, count] = min(candidates, key=lambda row: row["mean_rmse"])
    return tuning, selected


def fit(designs, acquired, settings, noise, features=None):
    predictions, timing = {}, {}
    for (door, count), batch in acquired.items():
        for model, feature_model in FEATURE_MODELS.items():
            key = door, count, model
            spec = settings[model, count]
            matrix = features[door, feature_model] if features is not None else designs[door].features[feature_model]
            before = time.perf_counter()
            predictions[key] = ridge_posterior(matrix, batch.indices, batch.values, noise,
                                                spec["prior_variance"], spec["intercept_ratio"])
            timing[key] = time.perf_counter() - before
    return predictions, timing


def calibrate(designs, predictions, truth, counts, *, models=FEATURE_MODELS):
    scales = {}
    for model in models:
        for count in counts:
            ratios = []
            for door, design in designs.items():
                pred, mask = predictions[door, count, model], design.heldout
                error = np.abs(pred["mean"][:, mask] - truth[door][:, mask])
                std = np.sqrt(np.maximum(pred["raw_variance"][:, mask], 1e-12))
                ratios.append((error / std).ravel())
            scales[model, count] = max(float(np.quantile(np.concatenate(ratios), 0.95) / 1.96), 1e-8)
    return scales


def summarize(rows, regimes, counts):
    summary, effects = [], []
    metrics = ("rmse", "near_wall_rmse", "coverage", "raw_coverage", "width", "interval_score",
               "distance", "amortized_fit_ms")
    rng = np.random.default_rng(640001)
    for regime in regimes:
        for count in counts:
            grouped = {}
            for row in rows:
                if row["regime"] == regime and int(row["measurements"]) == count:
                    model = row["model"]
                    key = int(row["door"]), int(row["seed"])
                    group = grouped.setdefault(model, {})
                    if key in group:
                        raise ValueError("Duplicate paired map in source metrics")
                    group[key] = row
            keys = sorted(grouped["graph_ridge"])
            if not all(set(keys) == set(group) for group in grouped.values()):
                raise ValueError("Different evaluation maps across models")
            doors = np.array([door for door, _ in keys])
            for key in keys:
                if len({(int(g[key]["start"]), float(g[key]["distance"])) for g in grouped.values()}) != 1:
                    raise ValueError("Different acquired route across models")
            for model, group in grouped.items():
                summary.append(dict(regime=regime, count=count, model=model, maps=len(keys),
                                    **{metric: float(np.mean([float(row[metric]) for row in group.values()]))
                                       for metric in metrics}))
            for metric in ("rmse", "near_wall_rmse"):
                values = {m: np.array([float(g[k][metric]) for k in keys]) for m, g in grouped.items()}
                deltas = {name: values[a] - values[b] for name, (a, b) in CONTRASTS.items()}
                deltas["representation_interaction"] = (
                    deltas["graph_features_in_mlp"] - deltas["graph_features_in_ridge"])
                for name, delta in deltas.items():
                    bootstrap = np.concatenate([
                        rng.choice(delta[doors == door], (4000, int((doors == door).sum())), replace=True)
                        for door in np.unique(doors)
                    ], axis=1).mean(axis=1)
                    effects.append(dict(regime=regime, count=count, metric=metric, contrast=name,
                                        mean_difference=float(delta.mean()),
                                        paired_interval=np.quantile(bootstrap, (0.025, 0.975)).tolist(),
                                        negative_fields=int((delta < -1e-12).sum()),
                                        by_door={int(door): float(delta[doors == door].mean())
                                                 for door in np.unique(doors)}))
    return summary, effects


def render(output, effects, regimes, counts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"graph_features_in_ridge": "Add graph basis · ridge",
              "graph_features_in_mlp": "Add graph basis · MLP",
              "mlp_vs_ridge_with_graph": "Switch ridge to MLP · graph basis"}
    fig, axes = plt.subplots(len(counts), len(regimes), figsize=(11, 5.2), sharex=True, sharey=True,
                             squeeze=False, layout="constrained")
    for row, count in enumerate(counts):
        for col, regime in enumerate(regimes):
            ax = axes[row, col]
            ax.axvline(0, color="0.5", linewidth=1)
            for i, (contrast, color) in enumerate(zip(labels, ("#227689", "#b55238", "#6953a1"))):
                result = next(r for r in effects if (r["count"], r["regime"], r["contrast"], r["metric"])
                              == (count, regime, contrast, "rmse"))
                ax.plot(result["paired_interval"], [i, i], color=color, linewidth=2)
                ax.plot(result["mean_difference"], i, "o", color=color)
            ax.set_yticks(range(len(labels)), list(labels.values()))
            ax.set_ylim(len(labels) - 0.5, -0.5)
            ax.grid(axis="x", alpha=0.2)
            ax.set_title(f"{'Wall weakens dependence' if regime == 'barrier' else 'Dependence crosses wall'} · {count} readings")
    fig.supxlabel("Paired change in held-out RMSE (negative favors the change)")
    fig.savefig(output / "paired-effects.png", dpi=170)
    plt.close(fig)


def run(source, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Use a fresh output directory")
    with threadpool_limits(limits=1):
        return run_comparison(source, output)


def run_comparison(source, output):
    reference = json.loads((source / "frozen.json").read_text())
    noise, budgets, regimes = (reference[k] for k in ("sensor_noise_std", "budgets", "regimes"))
    counts = tuple(n for n, _ in budgets)
    config = NeuralMapperConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in reference["neural"].items()})
    protocol = dict(reference=str(source.resolve()), budgets=budgets, sensor_noise_std=noise,
                    features=FEATURE_MODELS, prior_variances=PRIOR_VARIANCES, intercept_ratios=INTERCEPT_RATIOS,
                    scope="Retrospective comparison on the earlier test archive; no new hold-out fields",
                    coefficient_prior="iid normal with variance tau_squared/(p+1); intercept variance tau_squared*ratio",
                    selection="Same development fields and pooled per-field RMSE, independently by feature set/count",
                    calibration="Marginal interval scale from the separate reference calibration phase")
    dump(output / "protocol.json", protocol)
    designs = {door: make_design(door, config) for door in reference["development_doors"]}
    tune_obs = read_acquisitions(source, "tune", designs, budgets)
    tuning, settings = select(designs, tune_obs, read_truth(source, "tune", designs), counts, noise)
    write_csv(output / "tuning.csv", tuning)
    cal_obs = read_acquisitions(source, "calibrate", designs, budgets)
    cal_predictions, _ = fit(designs, cal_obs, settings, noise)
    multipliers = calibrate(designs, cal_predictions, read_truth(source, "calibrate", designs), counts)
    dump(output / "frozen.json", {**protocol, "selected": list(settings.values()),
                                 "std_multipliers": {f"{m}-n{n}": v for (m, n), v in multipliers.items()}})

    designs = {door: make_design(door, config) for door in reference["test_doors"]}
    with np.load(source / "test-features.npz") as archive:
        features = {(door, m): archive[f"d{door}_{m}"] for door in designs for m in FEATURE_MODELS.values()}
    for (door, model), matrix in features.items():
        if not np.array_equal(matrix, designs[door].features[model]):
            raise ValueError("Regenerated feature matrix differs from the neural benchmark")
    test = read_acquisitions(source, "test", designs, budgets)
    predictions, timing = fit(designs, test, settings, noise, features)
    rows = score(designs, test, predictions, read_truth(source, "test", designs), timing, multipliers)
    with (source / "metrics.csv").open() as stream:
        reference_rows = list(csv.DictReader(stream))
    fields = tuple(rows[0])
    combined = [{k: r[k] for k in fields} for r in reference_rows] + rows
    summary, effects = summarize(combined, regimes, counts)
    write_csv(output / "metrics.csv", combined)
    dump(output / "summary.json", summary)
    dump(output / "paired-effects.json", effects)
    dump(output / "reference-manifest.json", {
        name: hashlib.sha256((source / name).read_bytes()).hexdigest()
        for name in ("frozen.json", "metrics.csv", "test-features.npz",
                     *(f"{phase}-{archive}.npz" for phase in ("tune", "calibrate", "test")
                       for archive in ("observations", "evaluation-only")))
    })
    np.savez_compressed(output / "predictions.npz", **{
        f"d{d}_n{n}_{model}_{key}": value for (d, n, model), pred in predictions.items()
        for key, value in pred.items()})
    render(output, effects, regimes, counts)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("outputs/barrier-deepkriging-study"))
    parser.add_argument("--output", type=Path, default=Path("outputs/barrier-ridge-study"))
    args = parser.parse_args()
    for row in run(args.reference, args.output):
        print(f"{row['regime']:11s} n={row['count']:2} {row['model']:18s} "
              f"RMSE={row['rmse']:.4f} coverage={row['coverage']:.3f} width={row['width']:.3f}")


if __name__ == "__main__":
    main()
