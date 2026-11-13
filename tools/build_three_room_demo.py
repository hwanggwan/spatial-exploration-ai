"""Build an offline HTML replay from the verified three-room benchmark."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from spatial_exploration.baseline import GPBelief
from spatial_exploration.basis_bayes import BayesianBasisModel
from spatial_exploration.three_room_study import PRIMARY, design_for


ROOT = Path(__file__).resolve().parents[1]
TOKEN = "__BENCHMARK_REPLAY_DATA__"


def build_data(source):
    protocol = json.loads((source / "frozen.json").read_text())
    chosen = {(layout, protocol["test_seed"] + 1000 * layout + offset)
              for layout in range(len(protocol["test_layouts"]))
              for offset in range(len(protocol["starts"]))}
    observations = {}
    with (source / "acquisitions.csv").open() as stream:
        for row in csv.DictReader(stream):
            layout, seed = int(row["layout"]), int(row["seed"])
            if (layout, seed) in chosen and row["method"] in PRIMARY:
                key = layout, seed, row["regime"], row["budget"], row["method"]
                steps = observations.setdefault(key, [])
                if int(row["step"]) != len(steps):
                    raise ValueError("Unexpected observation order")
                steps.append((int(row["site"]), float(row["distance"]), float(row["value"])))
    scores = {}
    with (source / "metrics.csv").open() as stream:
        for row in csv.DictReader(stream):
            layout, seed = int(row["layout"]), int(row["seed"])
            if (layout, seed) in chosen and row["method"] in PRIMARY:
                scores[layout, seed, row["regime"], row["budget"], row["method"]] = float(row["rmse"])

    layouts, cases = [], []
    ridge = protocol["ridge"]
    with np.load(source / "evaluation-only.npz") as archive, threadpool_limits(limits=1):
        for layout_id, doors in enumerate(protocol["test_layouts"]):
            design = design_for(doors)
            layout = design.layout
            prefix = f"layout{layout_id}_"
            held = archive[prefix + "heldout"]
            np.testing.assert_array_equal(held, design.heldout)
            layouts.append({"doors": doors, "held": np.flatnonzero(held).tolist()})
            model = BayesianBasisModel(design.features, protocol["sensor_noise"], ridge["prior"],
                        intercept_scale=np.sqrt((design.features.shape[1] + 1) * ridge["intercept"]))
            for seed, regime, truth in zip(*(archive[prefix + suffix] for suffix in ("seeds", "regimes", "truth"))):
                if (layout_id, seed) not in chosen:
                    continue
                start = protocol["starts"][int(seed) - protocol["test_seed"] - 1000 * layout_id]
                for budget, limit in protocol["travel_caps"].items():
                    plans = {}
                    for method in PRIMARY:
                        key = layout_id, seed, str(regime), budget, method
                        steps = observations[key]
                        sites = [s for s, _, _ in steps]
                        if (sites[0] != start or len(set(sites)) != len(sites) or held[sites].any()
                                or steps[-1][1] > limit + 1e-12 or len(sites) > protocol["cap"]):
                            raise ValueError("Invalid archived route")
                        walk, endpoints = [start], []
                        mean, std, rmse = [], [], []
                        belief = GPBelief(model.prior_covariance, model.noise_std)
                        for i, (site, distance, value) in enumerate(steps):
                            if i:
                                leg = layout.walk(sites[i - 1], site)
                                if not all(layout.movement[u, v] for u, v in zip(leg, leg[1:])):
                                    raise ValueError("Illegal walking edge")
                                walk.extend(leg[1:])
                            np.testing.assert_allclose(distance, (len(walk) - 1) / (layout.size - 1), atol=1e-12)
                            endpoints.append(len(walk))
                            belief.observe(site, value)
                            mean.append(np.rint(belief.mean * 1000).astype(int).tolist())
                            std.append(np.rint(belief.std * 1000).astype(int).tolist())
                            rmse.append(float(np.sqrt(np.mean((belief.mean[held] - truth[held]) ** 2))))
                        np.testing.assert_allclose(rmse[-1], scores[key], atol=1e-12, rtol=0)
                        plans[method] = dict(sites=sites, walk=walk, endpoints=endpoints,
                                             mean=mean, std=std, rmse=rmse)
                    cases.append(dict(layout=layout_id, start=start, regime=str(regime), budget=budget,
                                      truth=np.rint(truth * 1000).astype(int).tolist(), plans=plans))
    if len(cases) != len(chosen) * len(protocol["regime_permeability"]) * len(protocol["travel_caps"]):
        raise ValueError("Incomplete replay cases")
    return dict(layouts=layouts, cases=cases, caps=protocol["travel_caps"], count=protocol["cap"],
                starts=protocol["starts"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/three-room-study")
    parser.add_argument("--output", type=Path, default=ROOT / "demo/index.html")
    args = parser.parse_args()
    template = (ROOT / "demo/index.template.html").read_text(encoding="utf8")
    if template.count(TOKEN) != 1:
        raise ValueError("Expected a single replay data placeholder")
    data = build_data(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(template.replace(TOKEN, json.dumps(data, separators=(",", ":"))), encoding="utf8")
    print(f"Exported {len(data['cases'])} paired replays to {args.output}")


if __name__ == "__main__":
    main()
