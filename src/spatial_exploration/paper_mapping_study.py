"""Leakage-free paper-function comparison as observations decrease to 16."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .baseline import write_csv
from .paper_mapping import (GP_MODELS, MODELS, NEURAL_MODELS, PaperConfig, fit_gp,
    fit_neural, initialization_seed, nested_design, paper_features, paper_signal)


LABELS = {"gp_exponential": "Exponential GP", "gp_rbf": "RBF GP",
          "dk_paper": "DK paper description", "dk_code_clean": "DK clean public-code variant",
          "coordinate_paper": "Coordinate DNN"}
COLORS = dict(zip(MODELS, ("#376aa3", "#25855b", "#cf5656", "#7954b3", "#da8b20")))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def summarize(rows, config):
    summaries, differences = [], []
    for count in config.counts:
        fold_values = {}
        for model in MODELS:
            chosen = [r for r in rows if r["count"] == count and r["model"] == model]
            # Average neural initialization replicates within each shared fold.
            # These are ten dependent folds of ONE function, not independent maps.
            rmse = np.asarray([np.mean([r["test_rmse"] for r in chosen if r["fold"] == f]) for f in range(config.folds)])
            fold_values[model] = rmse
            summaries.append({"count": count, "model": model, "folds": config.folds,
                "initializations_per_fold": len(config.neural_seeds) if model in NEURAL_MODELS else 1,
                "mean_test_rmse": float(rmse.mean()), "sd_across_fold_mean_rmse": float(rmse.std(ddof=1)),
                "mean_test_mae": float(np.mean([r["test_mae"] for r in chosen])),
                "mean_train_rmse": float(np.mean([r["train_rmse"] for r in chosen])),
                "mean_fit_seconds": float(np.mean([r["fit_seconds"] for r in chosen])),
                "mean_predict_seconds": float(np.mean([r["predict_seconds"] for r in chosen])),
                "seed_mean_rmse": {str(s): float(np.mean([r["test_rmse"] for r in chosen if r["member"] == s]))
                                   for s in config.neural_seeds} if model in NEURAL_MODELS else {},
                "interpretation": "descriptive CV on one deterministic function; folds are not independent realizations"})
        for model in MODELS:
            if model == "gp_exponential":
                continue
            delta = fold_values[model] - fold_values["gp_exponential"]
            differences.append({"count": count, "model": model, "reference": "gp_exponential",
                "mean_paired_fold_rmse_difference": float(delta.mean()),
                "lower_rmse_folds": int((delta < 0).sum()), "folds": config.folds,
                "paired_fold_differences": delta.tolist(), "no_independent_field_confidence_interval": True})
    return summaries, differences


def render_results(output, points, truth, designs, predictions, summaries, config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for model in MODELS:
        chosen = [r for r in summaries if r["model"] == model]
        axes[0].plot([r["count"] for r in chosen], [r["mean_test_rmse"] for r in chosen], "o-",
                     label=LABELS[model], color=COLORS[model])
        axes[1].plot([r["count"] for r in chosen], [r["mean_train_rmse"] for r in chosen], "o-",
                     label=LABELS[model], color=COLORS[model])
    for axis, label in zip(axes, ("Held-out RMSE", "Acquired-target RMSE")):
        axis.set_xscale("log"); axis.set_yscale("log")
        axis.set_xticks(config.counts, [str(n) for n in config.counts])
        axis.xaxis.set_minor_locator(NullLocator())
        axis.set_xlabel("Acquired noiseless points"); axis.set_ylabel(label); axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Paper 2D function — fresh models in every fold/count\n"
                 "Same withheld points across sample sizes; means over folds and initialization runs", fontsize=11)
    fig.savefig(output / "sample_size_comparison.png", dpi=150); plt.close(fig)

    # Fixed fold 0, all sizes, first initialization (not the best model/member).
    fig, axes = plt.subplots(len(MODELS) + 1, len(config.counts),
        figsize=(3.2 * len(config.counts), 2.9 * (len(MODELS) + 1)), squeeze=False, constrained_layout=True)
    half = 0.5 / (config.grid_size - 1)
    extent = (-half, 1 + half, -half, 1 + half)
    for col, count in enumerate(config.counts):
        design = next(d for d in designs if d["fold"] == 0 and d["count"] == count)
        series = [("Truth", truth)] + [(LABELS[m], predictions[m][0, col, 0]) for m in MODELS]
        for row, (label, values) in enumerate(series):
            axis = axes[row, col]
            plot = axis.imshow(values.reshape(config.grid_size, config.grid_size), origin="lower", extent=extent,
                               cmap="viridis", vmin=truth.min(), vmax=truth.max())
            sites = points[design["train"]]
            axis.scatter(sites[:, 0], sites[:, 1], s=2 if count > 128 else 5, c="white", alpha=0.65)
            axis.set_title(f"{label}, n={count}", fontsize=8)
            axis.set_xticks((0, 0.5, 1)); axis.set_yticks((0, 0.5, 1))
        fig.colorbar(plot, ax=axes[:, col], shrink=0.45)
    fig.suptitle("Predefined fold 0; first initialization; white points = acquired labels", fontsize=11)
    fig.savefig(output / "example_predictions.png", dpi=150); plt.close(fig)


def run_study(output, config=None):
    config = config or PaperConfig()
    config.validate()
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh output folder")
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir()
    torch.set_num_threads(1)
    protocol = {"config": asdict(config), "models": MODELS,
        "paper": "https://www3.stat.sinica.edu.tw/statistica/password.asp?art=13&num=1&sup=s&vol=34",
        "code": "https://github.com/aleksada/DeepKriging/blob/master/Nonstationary_function-2d.py",
        "scope": "controlled reimplementation, not exact numerical reproduction; no movement, no noise, one fixed function",
        "signal": "sin(30*(t-.9)^4)*cos(2*(t-.9))+(t-.9)/2; t=(x+y)/2",
        "observations": "nested subsets of each outer training pool; same heldout fold at every count and for every model",
        "paper_network": "4x100 ReLU; dropout .5 after first/second hidden; BN after first/third hidden; 200 epochs",
        "code_network": "3x100 ReLU, no dropout/BN; 400 total epochs matching two 200-epoch fit calls; FRESH weights per fold",
        "coordinate_network": "3x100 and supplement-default dropout/BN, 200 epochs, xy inputs only",
        "features": "raw Wendland on 10/19/37 knot grids, support 2.5/m; no extra xy or standardization",
        "optimizer": "Adam lr .001 eps 1e-7; first dense He uniform, later dense Glorot uniform, zero biases; shuffled batches64",
        "bn_deviation": "PyTorch variance/running-stat details may differ from Keras; epsilon and momentum convention matched",
        "gp": "zero mean, fitted amplitude/length/nugget, exponential vs RBF; sklearn bounded MLE with two restarts and jitter1e-10",
        "selection": "all configurations predeclared from cited settings; no test-based selection or warm starts; no extra tuning labels",
        "training_budget": "fixed epochs imply more gradient updates with more observations; do not interpret as pure data-only causal effect",
        "uncertainty": "GP plug-in latent variance archived; neural initialization variability is not calibrated predictive uncertainty; no DDSP replication",
        "statistical_unit": "one deterministic function; outer folds overlap in training, initialization runs are not independent fields",
        "primary_neural_metric": "mean of individual initialization RMSEs, not an ensemble advantage",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__}}
    write_json(output / "protocol.json", protocol)
    write_json(output / "frozen_protocol.json", {"all_settings_fixed_before_fits": True, "config": asdict(config),
        "no_hyperparameter_choice_from_heldout_labels": True})
    points, designs = nested_design(config)
    truth = paper_signal(points)
    basis = paper_features(points, config.knot_grids)
    np.savez_compressed(output / "evaluation_only.npz", points=points, truth=truth)
    train_ids = np.full((len(designs), max(config.counts)), -1, dtype=int)
    values = np.full(train_ids.shape, np.nan)
    for row, design in enumerate(designs):
        count = design["count"]; train_ids[row, :count] = design["train"]; values[row, :count] = truth[design["train"]]
    np.savez_compressed(output / "acquired.npz", points=points, indices=train_ids, values=values,
        counts=[d["count"] for d in designs], folds=[d["fold"] for d in designs])
    write_json(output / "splits.json", [{**d, "train": d["train"].tolist(), "test": d["test"].tolist()} for d in designs])
    shape = (config.folds, len(config.counts), len(config.neural_seeds), len(points))
    predictions = {m: np.empty(shape if m in NEURAL_MODELS else (shape[0], shape[1], 1, shape[3])) for m in MODELS}
    gp_variances = {m: np.empty((shape[0], shape[1], shape[3])) for m in GP_MODELS}
    rows, diagnostics = [], []
    began = time.perf_counter()
    for design in designs:
        fold, count, indices, test = design["fold"], design["count"], design["train"], design["test"]
        count_index = config.counts.index(count)
        acquired = truth[indices].copy()
        for model in MODELS:
            members = config.neural_seeds if model in NEURAL_MODELS else (-1,)
            for member_index, member in enumerate(members):
                if model in GP_MODELS:
                    result = fit_gp(points, indices, acquired, model, config)
                    gp_variances[model][fold, count_index] = result["variance"]
                else:
                    features = points.astype(np.float32) if model == "coordinate_paper" else basis
                    seed = initialization_seed(config, model, fold, member)
                    checkpoint = output / "checkpoints" / f"{model}-n{count}-seed{member}.pt" if fold == 0 else None
                    result = fit_neural(features, indices, acquired, model, config, seed, checkpoint)
                mean = result["mean"]
                predictions[model][fold, count_index, member_index] = mean
                rows.append({"fold": fold, "count": count, "model": model, "member": member,
                    "test_sites": len(test), "test_rmse": float(np.sqrt(np.mean((mean[test] - truth[test])**2))),
                    "test_mae": float(np.mean(np.abs(mean[test] - truth[test]))),
                    "train_rmse": float(np.sqrt(np.mean((mean[indices] - acquired)**2))),
                    "fit_seconds": result["fit_seconds"], "predict_seconds": result["predict_seconds"]})
                diagnostics.append({"fold": fold, "count": count, "model": model, "member": member,
                    **{key: value for key, value in result.items() if key not in ("mean", "variance")}})
            print(f"Fold {fold+1}/{config.folds}, n={count}: {model} complete", flush=True)
        write_csv(output / "progress_metrics.csv", rows)
        write_json(output / "progress.json", {"completed_fold_count_cases": len(rows) // (2 + 3 * len(config.neural_seeds)),
            "total_fold_count_cases": len(designs), "elapsed_seconds": time.perf_counter() - began})
    wall = time.perf_counter() - began
    # All mean/variance arrays are fixed before aggregate scoring or comparisons.
    np.savez_compressed(output / "predictions.npz", **predictions,
                        **{m + "_latent_variance": value for m, value in gp_variances.items()})
    summaries, differences = summarize(rows, config)
    write_csv(output / "metrics.csv", rows)
    write_json(output / "summary.json", summaries)
    write_json(output / "paired_fold_differences.json", differences)
    write_json(output / "fit_diagnostics.json", diagnostics)
    render_results(output, points, truth, designs, predictions, summaries, config)
    neural_diagnostics = [d for d in diagnostics if d["model"] in NEURAL_MODELS]
    restart_groups = {(d["model"], d["fold"], d["member"]) for d in neural_diagnostics}
    matching_initializations = all(len({d["initial_sha256"] for d in neural_diagnostics
        if (d["model"], d["fold"], d["member"]) == group}) == 1 for group in restart_groups)
    checks = {"fits": len(rows), "neural_fits": len([r for r in rows if r["model"] in NEURAL_MODELS]),
        "fit_and_score_wall_seconds": wall, "no_train_test_overlap": all(not set(d["train"]) & set(d["test"]) for d in designs),
        "same_test_set_across_counts": all(len({tuple(d["test"]) for d in designs if d["fold"] == f}) == 1 for f in range(config.folds)),
        "matching_fresh_initializations_across_counts": matching_initializations,
        "all_neural_fits_changed_weights": all(d["initial_sha256"] != d["final_sha256"] for d in neural_diagnostics),
        "all_finite_predictions": all(np.isfinite(value).all() for value in predictions.values()),
        "sum_fit_seconds": {m: float(sum(r["fit_seconds"] for r in rows if r["model"] == m)) for m in MODELS}}
    write_json(output / "diagnostics.json", checks)
    write_json(output / "dataset_manifest.json", {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob("*.npz")})
    write_json(output / "source_manifest.json", {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__), Path(__file__).with_name("paper_mapping.py"))})
    return summaries, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/paper-mapping-study"))
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        summaries, checks = run_study(args.output)
    for row in summaries:
        print(f"n={row['count']:3d} {row['model']:18s} RMSE={row['mean_test_rmse']:.6f}")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
