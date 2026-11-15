"""Rescore archived rollout observations at every jointly unmeasured grid cell."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from functools import partial
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from audit_residential_adaptive_rollout import EDGES, MATCHED_ARM, MATCHED_OUTPUT
from audit_residential_grid_resolution import holdouts, predictions, refine_geometry, refined_mapper
from audit_residential_holdout_distance import CAP, FAMILY
from audit_residential_refined_rollout import SCENARIOS, field_batches, field_key
from audit_residential_routes import CONDUCTANCES, LayoutCase, floor_plan
from audit_residential_zone_ranges import GLOBAL_WEIGHTS
from spatial_exploration.baseline import write_csv


ARMS = ("quarter", MATCHED_ARM)
BINS = ("1", "2-4", "5-8", "9+")
BIN_CUTS = (2, 5, 9)


def reading_sites(layout, held, routes, record):
    sites = [np.asarray(route["sites"]) for route in routes]
    if sites[0][0] != sites[1][0]:
        raise ValueError("The archived starts differ")
    for arm, arm_sites in zip(ARMS, sites):
        edges = round(layout.distances[arm_sites[:-1], arm_sites[1:]].sum() * (layout.size-1))
        if (len(arm_sites) != int(float(record[f"{arm}_readings"])) or len(arm_sites) > CAP
                or len(np.unique(arm_sites)) != len(arm_sites) or np.isin(arm_sites, held).any()
                or edges != int(float(record[f"{arm}_edges"])) or edges > EDGES):
            raise ValueError("Archived observation sites or resources differ")
    return sites


def evaluate(variant, model, fixed, oracle, previous_fields, draws, regime_filter=None):
    with threadpool_limits(limits=1):
        return evaluate_layout(variant, model, fixed, oracle, previous_fields, draws, regime_filter)


def evaluate_layout(variant, model, fixed, oracle, previous_fields, draws, regime_filter=None):
    coarse = floor_plan(LayoutCase(FAMILY, variant))
    layout, anchors = refine_geometry(coarse)
    held, _ = holdouts(coarse, layout, anchors, variant)
    profiles, kernels, prior, curve = refined_mapper(coarse, layout, anchors, model)
    log_prior = np.log(prior)
    targets = np.vstack((profiles[len(GLOBAL_WEIGHTS):], np.ones_like(profiles[0]), np.zeros_like(profiles[0])))
    grid = np.arange(len(layout.room))
    held_mask = np.isin(grid, held)
    rows, distance_stats = [], []
    for regime, start, profile, scenario, truth, noisy in field_batches(variant, layout, profiles, curve):
        if regime_filter is not None and regime != regime_filter:
            continue
        short = targets[profile, layout.room] > 0.5
        reference = fixed[variant, start]
        fixed_sites = np.asarray(reference["sites"])
        fixed_means, fixed_sd = predictions(kernels, fixed_sites, noisy[:draws, fixed_sites], grid, log_prior)
        for draw in range(draws):
            row = dict(variant=variant, regime=regime, start=start, profile=profile, draw=draw, scenario=scenario)
            key = field_key(row)
            archived = oracle[key]
            np.testing.assert_array_equal(targets[profile], archived["true_zone_range_weights"])
            _, sites = reading_sites(layout, held, (reference, archived), previous_fields[key])
            common = np.ones(len(grid), dtype=bool)
            common[np.union1d(fixed_sites, sites)] = False
            common_sites = grid[common]
            if not common[held].all():
                raise ValueError("The common evaluation set must include the original hold-outs")
            before = perf_counter()
            mean, sd = predictions(kernels, sites, noisy[draw:draw+1, sites], common_sites, log_prior)
            row["oracle_mapping_ms"] = (perf_counter() - before) * 1000
            errors = np.stack((fixed_means[draw, common], mean[0])) - truth[draw, common][None, :]
            squared = errors**2
            sds = np.stack((fixed_sd[draw, common], sd[0]))
            old = held_mask[common]
            short_common = short[common]
            groups = dict(original=old, common=np.ones(len(common_sites), dtype=bool), short=short_common,
                          original_short=old & short_common, long=~short_common)
            for group, mask in groups.items():
                row[f"{group}_cells"] = int(mask.sum())
                if not mask.any():
                    row.update({f"{arm}_{group}_rmse": None for arm in ARMS})
                    continue
                rmses = np.sqrt(squared[:, mask].mean(axis=1))
                for arm, rmse in zip(ARMS, rmses):
                    row[f"{arm}_{group}_rmse"] = float(rmse)
                if group in ("original", "original_short"):
                    suffix = "rmse" if group == "original" else "short_rmse"
                    expected = [float(previous_fields[key][f"{arm}_{suffix}"]) for arm in ARMS]
                    np.testing.assert_allclose(rmses, expected, rtol=0, atol=1e-10)
            for i, arm in enumerate(ARMS):
                row[f"{arm}_coverage"] = float(np.mean(abs(errors[i]) <= 1.96 * sds[i]))
                row[f"{arm}_interval_width"] = float(np.mean(3.92 * sds[i]))
            rows.append(row)

            nearest = np.stack([layout.distances[np.ix_(common_sites, arm_sites)].min(axis=1)
                                for arm_sites in (fixed_sites, sites)])
            nearest = np.rint(nearest * (layout.size-1)).astype(int)
            if np.any(nearest < 1):
                raise ValueError("A measured site entered the common evaluation set")
            bins = np.digitize(nearest, BIN_CUTS)
            # In each anchor's distance bins, compare both predictions at the same cells.
            field_stats = np.zeros((2, 2, len(BINS), 3))
            for anchor, indices in enumerate(bins):
                for group, mask in enumerate((groups["common"], short_common)):
                    field_stats[anchor, group, :, 0] = np.bincount(indices[mask], minlength=len(BINS))
                    for i in range(len(ARMS)):
                        field_stats[anchor, group, :, i+1] = np.bincount(indices[mask], weights=squared[i, mask],
                                                                       minlength=len(BINS))
            distance_stats.append(field_stats)
    return rows, np.asarray(distance_stats)


def summarize(rows, distance_stats):
    rng = np.random.default_rng(314803)
    output, distances = {}, []
    for scenario in SCENARIOS:
        selected = np.flatnonzero([r["scenario"] == scenario for r in rows])
        fields = [rows[i] for i in selected]
        strata = {}
        for idx, row in enumerate(fields):
            key = tuple(row[k] for k in ("variant", "regime", "start", "profile"))
            strata.setdefault(key, []).append(idx)
        boot = np.concatenate([rng.choice(ids, (3000, len(ids))) for ids in strata.values()], axis=1)
        report = dict(fields=len(fields), oracle_mapping_ms=float(np.mean([r["oracle_mapping_ms"] for r in fields])))
        for group in ("original", "common", "short", "original_short", "long"):
            if not fields[0][f"{group}_cells"]:
                continue
            counts = [r[f"{group}_cells"] for r in fields]
            values = np.array([[r[f"{arm}_{group}_rmse"] for arm in ARMS] for r in fields])
            changes = values[:, 1] - values[:, 0]
            report[group] = dict(mean_cells=float(np.mean(counts)), cell_range=[min(counts), max(counts)],
                                 fixed=float(values[:, 0].mean()), oracle=float(values[:, 1].mean()),
                                 delta=float(changes.mean()), ci95=np.quantile(changes[boot].mean(axis=1), [.025, .975]).tolist())
        for metric in ("coverage", "interval_width"):
            report[metric] = {arm: float(np.mean([r[f"{arm}_{metric}"] for r in fields])) for arm in ARMS}
        selected_stats = distance_stats[selected]
        for a, anchor in enumerate(ARMS):
            for g, group in enumerate(("common", "short")):
                for b, distance in enumerate(BINS):
                    cells, fixed, oracle = selected_stats[:, a, g, b].T
                    total = int(cells.sum())
                    if not total:
                        continue
                    scores = np.sqrt(np.array([fixed.sum(), oracle.sum()]) / total)
                    boot_count = cells[boot].sum(axis=1)
                    keep = boot_count > 0
                    changes = (np.sqrt(oracle[boot].sum(axis=1)[keep] / boot_count[keep])
                               - np.sqrt(fixed[boot].sum(axis=1)[keep] / boot_count[keep]))
                    low, high = np.quantile(changes, [.025, .975])
                    distances.append(dict(scenario=scenario, anchor=anchor, group=group, distance=distance,
                                          field_cells=total, fixed=float(scores[0]), oracle=float(scores[1]),
                                          delta=float(scores[1]-scores[0]), ci95_low=float(low), ci95_high=float(high)))
        output[scenario] = report
    return output, distances


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=MATCHED_OUTPUT)
    parser.add_argument("--regime", choices=tuple(CONDUCTANCES))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    suffix = f"-{args.regime}" if args.regime else ""
    args.output = args.output or Path(f"outputs/sensing-world-residential-dense-holdout{suffix}")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use an empty output directory.")
    protocol = json.loads((args.source / "protocol.json").read_text())
    if (protocol["cap"], protocol["edge_cap"], protocol["short_priority_source"]) != (CAP, EDGES, "rank_matched_oracle"):
        parser.error("Expected a 52-edge, 24-reading matched-oracle archive.")
    model = json.loads(Path(protocol["model"]).read_text())
    fixed = {(r["variant"], r["start"]): r for r in json.loads((Path(protocol["source"]) / "routes.json").read_text())
             if r["arm"] == "quarter"}
    oracle = {field_key(r): r for r in json.loads((args.source / "routes.json").read_text())}
    with (args.source / "fields.csv").open() as stream:
        previous_fields = {field_key(r): r for r in csv.DictReader(stream)}
    if oracle.keys() != previous_fields.keys():
        raise ValueError("Archived routes and field metrics differ")
    if args.regime:
        previous_fields = {key: row for key, row in previous_fields.items() if row["regime"] == args.regime}
        oracle = {key: oracle[key] for key in previous_fields}
    run = partial(evaluate, model=model, fixed=fixed, oracle=oracle, previous_fields=previous_fields,
                  draws=protocol["draws"], regime_filter=args.regime)
    rows, all_stats = [], []
    with ProcessPoolExecutor(max_workers=2) as pool:
        for variant, (fields, stats) in enumerate(pool.map(run, (0, 1))):
            rows.extend(fields)
            all_stats.append(stats)
            print(f"Replayed both policies across {len(fields)} fields on layout {variant}", flush=True)
    if {field_key(r) for r in rows} != previous_fields.keys():
        raise ValueError("Unexpected replay field keys")
    summary, distances = summarize(rows, np.concatenate(all_stats))
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "fields.csv", rows)
    write_csv(args.output / "distance-summary.csv", distances)
    output_protocol = dict(source=str(args.source), model=protocol["model"], edge_cap=EDGES, cap=CAP,
                           distance_bins=BINS, distance="Shortest legal walk to a measured cell",
                           test_cells="Exclude the union of both policies' measurement locations in each field")
    if args.regime:
        output_protocol.update(regime=args.regime, conductance=CONDUCTANCES[args.regime])
    for name, data in (("summary", summary), ("protocol", output_protocol)):
        (args.output / f"{name}.json").write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
