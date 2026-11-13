"""Secondary comparison: select the GP kernel on the same development fields."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from spatial_exploration.barrier_gp import KernelBank, kernel_banks, room_layout


def phase_arrays(root, phase, doors, count):
    data = {}
    with np.load(root / f"{phase}-observations.npz") as obs, np.load(root / f"{phase}-evaluation-only.npz") as eva:
        for door in doors:
            prefix = f"d{door}_n{count}_"
            data[door] = (obs[prefix + "indices"], obs[prefix + "values"], obs[prefix + "regimes"],
                          obs[prefix + "seeds"], eva[f"d{door}_truth"], eva[f"d{door}_heldout"])
    return data


def predict(data, parameters, noise):
    output = {}
    for door, (indices, values, _, _, truth, mask) in data.items():
        layout = room_layout(int(np.sqrt(len(mask))), door)
        full = kernel_banks(layout)["geometry"]
        selected = full.covariances[full.parameters.index(parameters)]
        fixed = KernelBank((parameters,), selected[None, ...], 0)
        beliefs = [fixed.posterior(sites, y, noise)[0] for sites, y in zip(indices, values)]
        output[door] = (np.asarray([p.mean[mask] for p in beliefs]) - truth[:, mask],
                        np.asarray([p.std[mask] for p in beliefs]))
    return output


def run(root):
    protocol = json.loads((root / "frozen.json").read_text())
    dev, test = protocol["development_doors"], protocol["test_doors"]
    noise = protocol["sensor_noise_std"]
    candidates = kernel_banks(room_layout(12, dev[0]))["geometry"].parameters
    with (root / "metrics.csv").open() as stream:
        neural = {(int(row["door"]), int(row["measurements"]), row["regime"], int(row["seed"])): float(row["rmse"])
                  for row in csv.DictReader(stream) if row["model"] == "graph_deepkriging"}
    rng = np.random.default_rng(530001)
    settings = {}
    for count, _ in protocol["budgets"]:
        tune = phase_arrays(root, "tune", dev, count)
        tuning = {}
        for candidate in candidates:
            errors = predict(tune, candidate, noise)
            tuning[candidate] = float(np.concatenate([
                np.sqrt(np.mean(residual**2, axis=1)) for residual, _ in errors.values()]).mean())
        winner = min(tuning, key=tuning.get)
        cal = predict(phase_arrays(root, "calibrate", dev, count), winner, noise)
        ratios = np.concatenate([(np.abs(error) / np.maximum(std, 1e-6)).ravel() for error, std in cal.values()])
        settings[count] = (winner, tuning[winner], float(np.quantile(ratios, 0.95) / 1.96))
    # Fixed GP choices above use only the original tuning and calibration splits.
    records = []
    for count, (winner, tuning_rmse, multiplier) in settings.items():
        data = phase_arrays(root, "test", test, count)
        errors = predict(data, winner, noise)
        for regime in protocol["regimes"]:
            rmse, raw_coverage, coverage, width, interval_score, delta = [], [], [], [], [], []
            for door in test:
                residual, std = errors[door]
                selected = data[door][2] == regime
                error, half = np.abs(residual[selected]), 1.96 * std[selected] * multiplier
                rmse_door = np.sqrt(np.mean(error**2, axis=1))
                rmse.extend(rmse_door)
                raw_coverage.extend((error <= 1.96 * std[selected]).mean(axis=1))
                coverage.extend((error <= half).mean(axis=1))
                width.extend((2 * half).mean(axis=1))
                interval_score.extend((2 * half + 40 * np.maximum(error - half, 0)).mean(axis=1))
                delta.append(np.array([neural[door, count, regime, int(seed)] - r
                                       for seed, r in zip(data[door][3][selected], rmse_door)]))
            draws = np.concatenate([rng.choice(d, (4000, len(d)), replace=True) for d in delta], axis=1)
            records.append(dict(regime=regime, count=count, length=winner[0], graph_weight=winner[1],
                                tuning_rmse=tuning_rmse, std_multiplier=multiplier, rmse=float(np.mean(rmse)),
                                raw_coverage=float(np.mean(raw_coverage)), coverage=float(np.mean(coverage)),
                                width=float(np.mean(width)), interval_score=float(np.mean(interval_score)),
                                graph_neural_minus_fixed_gp=float(np.concatenate(delta).mean()),
                                paired_interval=np.quantile(draws.mean(axis=1), [0.025, 0.975]).tolist()))
    return dict(scope="Secondary, added after primary scores; GP kernel tuned and calibrated only on original development splits",
                comparisons=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        result = run(args.root)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    (args.root / "gp-selection-audit.json").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
