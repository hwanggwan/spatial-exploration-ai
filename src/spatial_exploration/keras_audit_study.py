"""Same-Keras fold-reuse versus cumulative-training diagnostic."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

from .baseline import write_csv
from .keras_audit import (AuditConfig, CONDITIONS, KerasTrainer, source_features,
                         stage_orders, tensorflow, weight_hash)


LABELS = {"reuse_weights": "Weight reuse (contaminated CV)", "fresh_400": "Fresh, 400 epochs",
          "fresh_matched": "Fresh, cumulative epochs matched"}
COLORS = {"reuse_weights": "#c65b55", "fresh_400": "#7755ac", "fresh_matched": "#277eaf"}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def load_acquisitions(reference, config):
    designs = [d for d in json.loads((reference / "splits.json").read_text()) if d["count"] == config.count]
    acquired = np.load(reference / "acquired.npz")
    points = acquired["points"]
    if len(designs) != config.folds or sorted(d["fold"] for d in designs) != list(range(config.folds)):
        raise ValueError("Reference must have exactly the requested folds/count")
    rows = []
    for design in sorted(designs, key=lambda d: d["fold"]):
        fold = design["fold"]
        record = np.flatnonzero((acquired["folds"] == fold) & (acquired["counts"] == config.count))
        if len(record) != 1:
            raise ValueError("Reference acquisitions are ambiguous")
        indices = np.asarray(design["train"], dtype=int)
        test = np.asarray(design["test"], dtype=int)
        stored = acquired["indices"][record[0], :config.count]
        if (len(indices) != config.count or not np.array_equal(indices, stored)
                or set(indices) & set(test) or set(indices) | set(test) != set(range(len(points)))):
            raise ValueError("Require the full outer training pool and disjoint held-out fold")
        rows.append({"fold": fold, "train": indices, "test": test,
                     "values": acquired["values"][record[0], :config.count].copy()})
    if sorted(i for row in rows for i in row["test"]) != list(range(len(points))):
        raise ValueError("Outer evaluation folds must partition the grid")
    return points, rows


def summarize(rows, config):
    summary = []
    for condition in CONDITIONS:
        selected = [r for r in rows if r["condition"] == condition]
        seed_means = {str(seed): float(np.mean([r["test_rmse"] for r in selected if r["seed"] == seed]))
                      for seed in config.seeds}
        summary.append({"condition": condition, "mean_test_rmse": float(np.mean([r["test_rmse"] for r in selected])),
            "mean_test_mse": float(np.mean([r["test_mse"] for r in selected])),
            "mean_train_rmse": float(np.mean([r["train_rmse"] for r in selected])),
            "seed_mean_test_rmse": seed_means, "folds": config.folds,
            "mean_epoch_budget": float(np.mean([r["total_epochs"] for r in selected])),
            "valid_heldout": condition != "reuse_weights",
            "mean_test_rmse_excluding_first_fold": float(np.mean([r["test_rmse"] for r in selected if r["fold"] > 0]))})
    return summary


def render_results(output, rows, histories, config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for condition in CONDITIONS:
        values = np.array([[next(r["test_rmse"] for r in rows if r["condition"] == condition
                         and r["seed"] == seed and r["fold"] == fold)
                         for fold in range(config.folds)] for seed in config.seeds])
        x = np.arange(1, config.folds + 1)
        axes[0].plot(x, values.mean(axis=0), "o-", label=LABELS[condition], color=COLORS[condition])
        axes[0].fill_between(x, values.min(axis=0), values.max(axis=0), alpha=.12, color=COLORS[condition])
    axes[0].set_yscale("log"); axes[0].set_xlabel("Outer fold in source order")
    axes[0].set_ylabel("RMSE on current fold (reuse is contaminated)"); axes[0].set_xticks(np.arange(1, config.folds+1))
    axes[0].legend(fontsize=8); axes[0].grid(alpha=.2)
    for seed in config.seeds:
        selected = [r for r in histories if r["stream"] == "fresh" and r["fold"] == config.folds - 1 and r["seed"] == seed]
        axes[1].plot([r["total_epochs"] for r in selected], [np.sqrt(r["train_mse"]) for r in selected], label=f"Initialization {seed}")
    axes[1].set_yscale("log"); axes[1].set_xlabel("Epochs on final fold's 810 acquired points")
    axes[1].set_ylabel("Acquired-point RMSE"); axes[1].axvline(config.stage_epochs, color="gray", ls=":")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=.2)
    fig.suptitle("Same Keras model: fold reuse versus independent, matched training\n"
                 "Shading = initialization range; one deterministic surface", fontsize=11)
    fig.savefig(output / "fold_reuse_comparison.png", dpi=150); plt.close(fig)


def run_study(output, reference, source, config=None):
    config = config or AuditConfig()
    config.validate()
    output, reference, source = Path(output), Path(reference), Path(source)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh output directory")
    if not source.is_file():
        raise ValueError("Archive the authors' public script before fitting")
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir()
    tf = tensorflow()
    source_bytes = source.read_bytes()
    (output / "author_source.py.txt").write_bytes(source_bytes)
    protocol = {"config": asdict(config), "conditions": CONDITIONS,
        "source_url": "https://github.com/aleksada/DeepKriging/blob/master/Nonstationary_function-2d.py",
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "architecture": "3x100 ReLU; first He-uniform, later Glorot-uniform; no dropout/BN; raw 1830 Wendland values",
        "optimizer": "Keras Adam .001, beta1 .9, beta2 .999, epsilon1e-7, float32, no weight decay",
        "source_semantics": "one stage = two consecutive 200-epoch source fits; new Adam each stage; retain weights only in reuse arm",
        "matched_arm": "fold f (zero-based): fresh original initialization, repeat ONLY current acquired pool for f+1 stages",
        "coupling": "identical initial weights within seed; same position-order permutation for stage k in both streams",
        "reuse_warning": "intentionally invalid CV from second fold; previously seen test labels; diagnostic only",
        "selection": "fixed 10 folds, 3 seeds and 400 epochs/stage; no test selection or early stopping",
        "loop": "compiled GradientTape loop uses actual Keras model/Adam; equivalence checked against train_on_batch",
        "deviations": "modern TensorFlow/Keras, not pinned original software; explicit seeds, coupled batch orders; omit validation logging, no effect on selection",
        "scope": "single deterministic function, no movement; training-history comparison, not a causal leakage-only decomposition",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "tensorflow": tf.__version__, "keras": tf.keras.__version__}}
    write_json(output / "frozen_protocol.json", protocol)
    write_json(output / "reference_manifest.json", {name: hashlib.sha256((reference / name).read_bytes()).hexdigest()
        for name in ("acquired.npz", "splits.json", "evaluation_only.npz", "summary.json")})
    points, designs = load_acquisitions(reference, config)
    features = source_features(points, config.knot_grids)
    # Evaluation is separate; no full-grid labels enter a KerasTrainer call.
    evaluator = np.load(reference / "evaluation_only.npz")
    if not np.array_equal(points, evaluator["points"]):
        raise ValueError("Reference geometry mismatch")
    truth = evaluator["truth"]
    np.savez_compressed(output / "geometry.npz", points=points, features=features)
    predictions = np.empty((len(CONDITIONS), len(config.seeds), config.folds, len(points)))
    rows, histories, stage_records = [], [], []
    started = time.perf_counter()
    actual_fit_seconds = 0.

    def fit_stage(trainer, design, seed, stage, stream):
        nonlocal actual_fit_seconds
        result = trainer.fit_stage(features[design["train"]], design["values"],
                                   stage_orders(config, seed, stage, len(design["train"])))
        actual_fit_seconds += result["fit_seconds"]
        for h in result["histories"]:
            histories.append({"seed": seed, "fold": design["fold"], "stream": stream, "stage": stage,
                "total_epochs": stage * config.stage_epochs + h["epoch_in_stage"], **h})
        stage_records.append({"seed": seed, "fold": design["fold"], "stage": stage, "stream": stream,
                              **{k: v for k, v in result.items() if k != "histories"}})
        print(f"seed={seed} fold={design['fold']+1}/{config.folds} {stream} stage={stage+1} done", flush=True)
        return result

    def record(condition, trainer, design, seed, seed_index, stages, result, cumulative_seconds, seen):
        mean, query_seconds = trainer.predict(features)
        fold, test, train = design["fold"], design["test"], design["train"]
        predictions[CONDITIONS.index(condition), seed_index, fold] = mean
        test_mse = float(np.mean((mean[test] - truth[test])**2))
        train_mse = float(np.mean((mean[train] - design["values"])**2))
        rows.append({"condition": condition, "seed": seed, "fold": fold,
            "total_epochs": stages * config.stage_epochs,
            "total_updates": stages * result["updates"], "prior_seen_test_sites": len(set(test) & seen),
            "test_sites": len(test), "train_rmse": float(np.sqrt(train_mse)), "test_rmse": float(np.sqrt(test_mse)),
            "train_mse": train_mse, "test_mse": test_mse, "cumulative_fit_seconds": cumulative_seconds,
            "predict_seconds": query_seconds, "initial_sha256": trainer.initial_hash,
            "start_last_stage_sha256": result["start_stage_sha256"], "final_sha256": result["final_sha256"]})
        trainer.model.save_weights(output / "checkpoints" / f"{condition}-seed{seed}-fold{fold}.weights.h5")

    for seed_index, seed in enumerate(config.seeds):
        reuse = KerasTrainer(features.shape[1], config, seed)
        initial_weights = [v.copy() for v in reuse.model.get_weights()]
        past_seen, reuse_seconds = set(), 0.
        for design in designs:
            fold = design["fold"]
            result = fit_stage(reuse, design, seed, fold, "reuse")
            reuse_seconds += result["fit_seconds"]
            record("reuse_weights", reuse, design, seed, seed_index, fold+1, result, reuse_seconds, past_seen)
            if fold == 0:
                # Identical stage/data/start; reuse its numerical result, no
                # redundant fitting or triple-counted actual compute.
                record("fresh_400", reuse, design, seed, seed_index, 1, result, result["fit_seconds"], set())
                record("fresh_matched", reuse, design, seed, seed_index, 1, result, result["fit_seconds"], set())
            else:
                fresh = KerasTrainer(features.shape[1], config, seed, initial_weights)
                seconds = 0.
                for stage in range(fold + 1):
                    trained = fit_stage(fresh, design, seed, stage, "fresh")
                    seconds += trained["fit_seconds"]
                    if stage == 0:
                        record("fresh_400", fresh, design, seed, seed_index, 1, trained, seconds, set())
                record("fresh_matched", fresh, design, seed, seed_index, fold+1, trained, seconds, set())
                del fresh
            past_seen.update(design["train"])
            write_csv(output / "progress_metrics.csv", rows)
            write_csv(output / "training_history.csv", histories)
            write_json(output / "progress.json", {"completed_seed_fold_cases": len(rows)//3,
                "total_seed_fold_cases": len(config.seeds)*config.folds, "actual_stage_fits": len(stage_records),
                "elapsed_seconds": time.perf_counter() - started})
    wall = time.perf_counter() - started
    summary = summarize(rows, config)
    write_csv(output / "metrics.csv", rows)
    write_json(output / "summary.json", summary)
    write_json(output / "stage_diagnostics.json", stage_records)
    np.savez_compressed(output / "predictions.npz", predictions=predictions, conditions=CONDITIONS, seeds=config.seeds)
    checks = {"recorded_results": len(rows), "actual_stage_fits": len(stage_records),
        "actual_fit_seconds": actual_fit_seconds, "training_loop_wall_seconds": wall,
        "all_finite_predictions": bool(np.isfinite(predictions).all()),
        "all_fresh_test_sites_unseen": all(r["prior_seen_test_sites"] == 0 for r in rows if r["condition"] != "reuse_weights"),
        "all_later_reuse_test_sites_previously_seen": all(r["prior_seen_test_sites"] == r["test_sites"]
            for r in rows if r["condition"] == "reuse_weights" and r["fold"] > 0),
        "shared_fold_zero_predictions": all(np.array_equal(predictions[0, s, 0], predictions[c, s, 0])
            for s in range(len(config.seeds)) for c in (1, 2)),
        "matched_update_budgets": all(next(r["total_updates"] for r in rows if r["condition"] == "reuse_weights"
            and r["seed"] == seed and r["fold"] == fold) == next(r["total_updates"] for r in rows
            if r["condition"] == "fresh_matched" and r["seed"] == seed and r["fold"] == fold)
            for seed in config.seeds for fold in range(config.folds))}
    write_json(output / "diagnostics.json", checks)
    render_results(output, rows, histories, config)
    write_json(output / "source_manifest.json", {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__), Path(__file__).with_name("keras_audit.py"))})
    return summary, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("outputs/paper-mapping-study"))
    parser.add_argument("--output", type=Path, default=Path("outputs/keras-reuse-audit"))
    parser.add_argument("--source", type=Path, default=Path("external/deepkriging/Nonstationary_function-2d.py"))
    args = parser.parse_args()
    summary, checks = run_study(args.output, args.reference, args.source)
    print(json.dumps({"summary": summary, "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
