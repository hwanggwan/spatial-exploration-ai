"""Compare matched per-map DeepKriging-style ensembles and a geometry GP."""

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .barrier_gp import KernelBank, RoomLayout, graph_spectrum, kernel_banks, room_layout, transport_covariance
from .barrier_study import (BUDGETS, DOORS, KAPPAS, NOISE, REGIMES, SIZE, STARTS,
                            evaluation_sites, fixed_design)
from .baseline import write_csv
from .deepkriging import (NeuralMapperConfig, fit_ensemble_snapshots, parameter_count,
                          standardized_features)


NEURAL = ("deepkriging", "graph_deepkriging")
MODELS = (*NEURAL, "geometry_gp")
COUNTS = tuple(count for count, _ in BUDGETS)
TEST_DOORS = (2, 9)
WIDTHS = dict(zip(NEURAL, (33, 22)))
GRAPH_LENGTH = 0.22
SEEDS = {"tune": 150000, "calibrate": 250000, "test": 350000}


@dataclass(frozen=True)
class LayoutDesign:
    layout: RoomLayout
    heldout: np.ndarray
    near_wall: np.ndarray
    features: dict
    bank: KernelBank
    routes: dict


@dataclass(frozen=True)
class Acquisitions:
    indices: np.ndarray
    values: np.ndarray
    travel: np.ndarray
    seeds: np.ndarray
    starts: np.ndarray
    regimes: np.ndarray


def graph_heat_features(layout, length=GRAPH_LENGTH, *, permeability=0.0):
    """Normalized, grid-anchored square root of the graph heat covariance."""
    eig, vectors = graph_spectrum(layout, permeability)
    t = 0.5 * (length * (layout.size - 1))**2
    root = (vectors * np.exp(-0.5 * t * eig)) @ vectors.T
    # The graph block has mean squared feature magnitude one, like standardized Wendland.
    return np.sqrt(len(eig)) * root / np.linalg.norm(root, axis=1, keepdims=True)


def make_design(door, config):
    layout = room_layout(SIZE, door)
    heldout, near = evaluation_sites(layout)
    banks = kernel_banks(layout)
    euclidean, _, _ = standardized_features(layout.points, "deepkriging", config)
    features = {NEURAL[0]: euclidean, NEURAL[1]: np.column_stack(
        (euclidean, graph_heat_features(layout))).astype(np.float32)}
    routes = {(start, count): fixed_design(layout, start, ~heldout, (count, cap), banks["stationary"])
              for start in STARTS for count, cap in BUDGETS}
    return LayoutDesign(layout, heldout, near, features, banks["geometry"], routes)


def generate_split(designs, fields, seed_base, output, phase):
    """Generate truth in the evaluator, then expose only the fixed acquired rows."""
    acquired, evaluator = {}, {}
    obs_archive, eval_archive = {}, {}
    for door, design in designs.items():
        factors = {(regime, scale): np.linalg.cholesky(transport_covariance(design.layout, permeability, scale))
                   for regime, permeability in REGIMES.items() for scale in KAPPAS}
        seeds, starts, regimes, truth, readings = [], [], [], [], []
        for field in range(fields):
            seed = seed_base + door * 1000 + field
            start = STARTS[field % len(STARTS)]
            scale = KAPPAS[(field // len(STARTS)) % len(KAPPAS)]
            z, noise = np.random.default_rng(seed).standard_normal((2, SIZE**2))
            for regime in REGIMES:
                latent = factors[regime, scale] @ z
                seeds.append(seed)
                starts.append(start)
                regimes.append(regime)
                truth.append(latent)
                readings.append(latent + NOISE * noise)
        truth, readings = np.asarray(truth), np.asarray(readings)
        evaluator[door] = truth
        eval_archive[f"d{door}_truth"] = truth
        eval_archive[f"d{door}_heldout"] = design.heldout
        eval_archive[f"d{door}_near_wall"] = design.near_wall
        for count, cap in BUDGETS:
            routes, distances = zip(*(design.routes[start, count] for start in starts))
            indices, travel = np.asarray(routes), np.asarray([d[-1] for d in distances])
            assert indices.shape == (len(seeds), count)
            assert not design.heldout[indices].any() and (travel <= cap + 1e-12).all()
            values = np.take_along_axis(readings, indices, axis=1)
            batch = Acquisitions(indices, values, travel, np.asarray(seeds), np.asarray(starts),
                                 np.asarray(regimes))
            acquired[door, count] = batch
            obs_archive.update({f"d{door}_n{count}_{key}": value for key, value in vars(batch).items()})
    np.savez_compressed(output / f"{phase}-observations.npz", **obs_archive)
    np.savez_compressed(output / f"{phase}-evaluation-only.npz", **eval_archive)
    return acquired, evaluator


def neural_predictions(features, acquisitions, model, config, decay, epochs, checkpoint_prefix=None):
    """Fresh independent member fits; no simulator truth in this interface."""
    config = replace(config, basis_width=WIDTHS[model])
    member_seeds = [10 * int(seed) + member for seed in acquisitions.seeds
                    for member in range(config.members)]
    predictions = {epoch: [] for epoch in epochs}
    seconds = 0.0
    for start in range(0, len(acquisitions.indices), config.chunk_maps):
        stop = min(start + config.chunk_maps, len(acquisitions.indices))
        path = (checkpoint_prefix.parent / f"{checkpoint_prefix.name}-{start}-{stop}.pt"
                if checkpoint_prefix is not None else None)
        results, timings, _ = fit_ensemble_snapshots(
            features, acquisitions.indices[start:stop], acquisitions.values[start:stop], "deepkriging",
            config, member_seeds[start * config.members:stop * config.members], decay, epochs, save_path=path)
        seconds += timings[max(epochs)]
        for epoch, result in results.items():
            predictions[epoch].append(result)
    return {epoch: {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}
            for epoch, chunks in predictions.items()}, seconds


def gp_predictions(bank, acquisitions):
    before = time.perf_counter()
    means, variances, parameters = [], [], []
    for indices, values in zip(acquisitions.indices, acquisitions.values):
        belief, params = bank.posterior(indices, values, NOISE)
        means.append(belief.mean)
        variances.append(belief.std**2)
        parameters.append(params)
    return dict(mean=np.asarray(means), raw_variance=np.asarray(variances),
                parameters=np.asarray(parameters)), time.perf_counter() - before


def tune(designs, acquired, truth, config):
    records, selection = [], {}
    total_seconds = 0.0
    for model in NEURAL:
        for count in COUNTS:
            candidates = []
            for decay in config.weight_decays:
                rmses = {epoch: [] for epoch in config.epochs}
                for door, design in designs.items():
                    result, elapsed = neural_predictions(design.features[model], acquired[door, count],
                                                         model, config, decay, config.epochs)
                    total_seconds += elapsed
                    mask = design.heldout
                    for epoch, pred in result.items():
                        residual = pred["mean"][:, mask] - truth[door][:, mask]
                        rmses[epoch].extend(np.sqrt(np.mean(residual**2, axis=1)))
                for epoch, rmse in rmses.items():
                    candidates.append(dict(model=model, count=count, decay=decay,
                                           epochs=epoch, mean_rmse=float(np.mean(rmse))))
            records.extend(candidates)
            selection[model, count] = min(candidates, key=lambda r: (r["mean_rmse"], r["epochs"], r["decay"]))
    return records, selection, total_seconds


def fit_selected(designs, acquired, config, settings, checkpoints=None):
    predictions, timing = {}, {}
    for (door, count), observations in acquired.items():
        design = designs[door]
        for model in MODELS:
            key = door, count, model
            if model == "geometry_gp":
                predictions[key], timing[key] = gp_predictions(design.bank, observations)
            else:
                spec = settings[model, count]
                prefix = checkpoints / f"d{door}-n{count}-{model}" if checkpoints is not None else None
                output, timing[key] = neural_predictions(design.features[model], observations, model,
                    config, spec["decay"], (spec["epochs"],), prefix)
                predictions[key] = output[spec["epochs"]]
    return predictions, timing


def calibrate(designs, predictions, truth):
    """Marginal validation scaling; not a joint posterior or conformal guarantee."""
    multipliers = {}
    for count in COUNTS:
        for model in MODELS:
            ratios = []
            for door, design in designs.items():
                pred, mask = predictions[door, count, model], design.heldout
                residual = np.abs(truth[door][:, mask] - pred["mean"][:, mask])
                std = np.sqrt(np.maximum(pred["raw_variance"][:, mask], 1e-12))
                ratios.append((residual / std).ravel())
            multipliers[model, count] = max(float(np.quantile(np.concatenate(ratios), 0.95) / 1.96), 1e-8)
    return multipliers


def score(designs, acquired, predictions, truth, timing, multipliers):
    rows = []
    for (door, count, model), pred in predictions.items():
        design, data = designs[door], acquired[door, count]
        mask = design.heldout
        near = design.near_wall[mask]
        std = np.sqrt(np.maximum(pred["raw_variance"][:, mask], 1e-12))
        raw_half = 1.96 * std
        half = raw_half * multipliers[model, count]
        error = np.abs(truth[door][:, mask] - pred["mean"][:, mask])
        coverage = error <= half
        interval_score = 2 * half + 40 * np.maximum(error - half, 0)
        rmse = np.sqrt(np.mean(error**2, axis=1))
        wall_rmse = np.sqrt(np.mean(error[:, near]**2, axis=1))
        acquired_rmse = np.sqrt(np.mean((np.take_along_axis(pred["mean"], data.indices, axis=1)
                                         - data.values)**2, axis=1))
        for i, seed in enumerate(data.seeds):
            rows.append(dict(door=door, seed=int(seed), start=int(data.starts[i]), regime=str(data.regimes[i]),
                             measurements=count, model=model, distance=float(data.travel[i]),
                             rmse=float(rmse[i]), near_wall_rmse=float(wall_rmse[i]),
                             acquired_rmse=float(acquired_rmse[i]),
                             raw_coverage=float(np.mean(error[i] <= raw_half[i])),
                             raw_width=float(np.mean(2 * raw_half[i])),
                             coverage=float(np.mean(coverage[i])), near_wall_coverage=float(np.mean(coverage[i, near])),
                             width=float(np.mean(2 * half[i])), interval_score=float(np.mean(interval_score[i])),
                             amortized_fit_ms=1000 * timing[door, count, model] / len(data.seeds)))
    return rows


def summarize(rows):
    metrics = ("rmse", "near_wall_rmse", "acquired_rmse", "coverage", "near_wall_coverage", "width",
               "raw_coverage", "raw_width", "interval_score", "distance", "amortized_fit_ms")
    summary, contrasts = [], []
    rng = np.random.default_rng(415000)
    for regime in REGIMES:
        for count in COUNTS:
            grouped = {model: {(r["door"], r["seed"]): r for r in rows if r["model"] == model
                       and r["regime"] == regime and r["measurements"] == count} for model in MODELS}
            keys = sorted(grouped[MODELS[0]])
            assert all(set(keys) == set(g) for g in grouped.values())
            doors = np.array([door for door, _ in keys])
            for model, selected in grouped.items():
                summary.append(dict(regime=regime, count=count, model=model, fields=len(selected),
                                    **{metric: float(np.mean([r[metric] for r in selected.values()]))
                                       for metric in metrics}))
            for reference in (NEURAL[0], "geometry_gp"):
                for metric in ("rmse", "near_wall_rmse"):
                    delta = np.array([grouped[NEURAL[1]][key][metric] - grouped[reference][key][metric]
                                      for key in keys])
                    boot = np.concatenate([
                        rng.choice(delta[doors == door], (4000, int((doors == door).sum())), replace=True)
                        for door in np.unique(doors)
                    ], axis=1).mean(axis=1)
                    contrasts.append(dict(regime=regime, count=count, reference=reference, metric=metric,
                                          geometry_neural_minus_reference=float(delta.mean()),
                                          paired_interval=np.quantile(boot, (0.025, 0.975)).tolist(),
                                          geometry_neural_wins=int(np.sum(delta < -1e-12)),
                                          by_door={int(door): float(delta[doors == door].mean())
                                                   for door in np.unique(doors)}))
    return summary, contrasts


def render(output, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = dict(zip(MODELS, ("Wendland DeepKriging", "Graph DeepKriging", "Geometry GP")))
    fig, axes = plt.subplots(3, 2, figsize=(9.5, 8), sharex=True, sharey="row", layout="constrained")
    for col, regime in enumerate(REGIMES):
        for row, (metric, ylabel) in enumerate((("rmse", "Held-out RMSE"),
                                                ("coverage", "95% interval coverage"),
                                                ("width", "Mean interval width"))):
            ax = axes[row, col]
            for model, color in zip(MODELS, ("#7355a0", "#b54e31", "#217383")):
                cell = [r for r in summary if r["model"] == model and r["regime"] == regime]
                ax.plot([r["count"] for r in cell], [r[metric] for r in cell], marker="o",
                        color=color, label=labels[model])
            ax.grid(alpha=0.2)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title("Wall weakens dependence" if regime == "barrier" else "Dependence crosses wall")
            if metric == "coverage":
                ax.axhline(.95, color="0.3", linestyle=":", linewidth=1)
                ax.set_ylim(0.75, 1.005)
            if row == 2:
                ax.set_xticks(COUNTS)
                ax.set_xlabel("Identical acquisitions per map")
    handles, legend = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend, loc="upper center", bbox_to_anchor=(0.5, 0), ncol=3, frameon=False)
    fig.suptitle("Frozen evaluation on new fields and door locations")
    fig.savefig(output / "comparison.png", dpi=175, bbox_inches="tight")
    plt.close(fig)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run(output, fields_per_layout=48, development_fields=16, config=None):
    if not (1 <= min(fields_per_layout, development_fields)
            and max(fields_per_layout, development_fields) < 1000):
        raise ValueError("Use 1–999 fields per layout and split")
    config = config or NeuralMapperConfig(members=5, epochs=(100, 300, 600), chunk_maps=16)
    config.validate()
    if config.members > 10:
        raise ValueError("At most ten members per reserved seed block")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Use a fresh output directory")
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        return run_frozen(output, fields_per_layout, development_fields, config)


def run_frozen(output, fields, dev_fields, config):
    designs = {door: make_design(door, config) for door in DOORS}
    sizes = {model: designs[DOORS[0]].features[model].shape[1] for model in NEURAL}
    protocol = dict(development_doors=DOORS, test_doors=TEST_DOORS, development_fields_per_layout=dev_fields,
                    test_fields_per_layout=fields, budgets=BUDGETS, regimes=REGIMES, kappa=KAPPAS,
                    sensor_noise_std=NOISE, seed_bases=SEEDS, seed_rule="base + 1000 * door + field",
                    member_seed_rule="10 * field_seed + member index; paired across regimes and neural arms",
                    graph_length=GRAPH_LENGTH, widths=WIDTHS, feature_counts=sizes,
                    parameters={m: parameter_count(sizes[m], WIDTHS[m]) for m in NEURAL},
                    neural=asdict(config), model_selection="held-out RMSE pooled across tuning layouts/regimes",
                    calibration="separate fields; pooled 0.95 residual/std quantile per model and count; all arms",
                    test_metric="mean per-field latent RMSE at 28 reserved sites (8 near the wall)",
                    acquisition="same target-blind stationary-prior route for all mappers; no closed loop",
                    versions=dict(torch=str(torch.__version__), numpy=np.__version__))
    dump(output / "protocol.json", protocol)
    acquired, truth = generate_split(designs, dev_fields, SEEDS["tune"], output, "tune")
    tuning, selected, tuning_seconds = tune(designs, acquired, truth, config)
    write_csv(output / "tuning.csv", tuning)
    acquired, truth = generate_split(designs, dev_fields, SEEDS["calibrate"], output, "calibrate")
    predictions, calibration_timing = fit_selected(designs, acquired, config, selected)
    multipliers = calibrate(designs, predictions, truth)
    dump(output / "frozen.json", {**protocol, "selected": list(selected.values()),
         "std_multipliers": {f"{model}-n{count}": value for (model, count), value in multipliers.items()}})

    # Test fields and their truth do not exist until settings are frozen.
    designs = {door: make_design(door, config) for door in TEST_DOORS}
    acquired, truth = generate_split(designs, fields, SEEDS["test"], output, "test")
    checkpoints = output / "checkpoints"
    checkpoints.mkdir()
    predictions, timing = fit_selected(designs, acquired, config, selected, checkpoints)
    rows = score(designs, acquired, predictions, truth, timing, multipliers)
    summary, contrasts = summarize(rows)
    write_csv(output / "metrics.csv", rows)
    dump(output / "summary.json", summary)
    dump(output / "paired.json", contrasts)
    dump(output / "compute.json", dict(tuning_neural_seconds=tuning_seconds,
                                      calibration_seconds=sum(calibration_timing.values()),
                                      test_seconds={m: sum(v for k, v in timing.items() if k[2] == m)
                                                    for m in MODELS}))
    np.savez_compressed(output / "predictions.npz",
        **{f"d{d}_n{n}_{m}_{key}": val for (d, n, m), pred in predictions.items() for key, val in pred.items()})
    np.savez_compressed(output / "test-features.npz",
        **{f"d{d}_{m}": feat for d, design in designs.items() for m, feat in design.features.items()})
    render(output, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/barrier-deepkriging-study"))
    parser.add_argument("--fields-per-layout", type=int, default=48)
    parser.add_argument("--development-fields-per-layout", type=int, default=16)
    args = parser.parse_args()
    for row in run(args.output, args.fields_per_layout, args.development_fields_per_layout):
        print(f"{row['regime']:11s} n={row['count']:2} {row['model']:16s} "
              f"RMSE={row['rmse']:.4f} coverage={row['coverage']:.3f} width={row['width']:.3f}")


if __name__ == "__main__":
    main()
