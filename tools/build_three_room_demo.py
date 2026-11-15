"""Build the standalone HTML replay from the verified benchmark archives."""

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
WORLD_TOKEN = "__WORLD_RENDERER__"
REFINED_TOKEN = "__REFINED_RENDERER__"
REFINED_DATA_TOKEN = "__REFINED_REPLAY_DATA__"


def build_data(source, learned_replay=None, rollout_replay=None):
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

    learned = {}
    if learned_replay is not None:
        rows = json.loads(learned_replay.read_text(encoding="utf8"))
        learned = {(r["layout"], r["seed"], r["regime"], r["budget"]): r for r in rows}
        if len(learned) != len(rows):
            raise ValueError("Duplicate learned scene")
    rollouts = {}
    if rollout_replay is not None:
        rows = json.loads(rollout_replay.read_text(encoding="utf8"))
        records = [r for r in rows if r.get("arm") == "learned_rollout"]
        rollouts = {(r["layout"], r["seed"], r["regime"], r["budget"]): r for r in records}
        if len(rollouts) != len(records):
            raise ValueError("Duplicate rollout scene")
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
                                             mean=mean, std=std, rmse=rmse,
                                             readings=[round(value, 3) for _, _, value in steps])
                    if learned_replay is not None:
                        record = learned.pop((layout_id, seed, str(regime), budget))
                        learned_plan = record["plan"]
                        verify_learned_plan(learned_plan, start, layout, held, truth, protocol["cap"], limit)
                        plans["learned_world"] = learned_plan
                    if rollout_replay is not None and budget == "loose":
                        plan = rollouts.pop((layout_id, seed, str(regime), budget))["plan"]
                        verify_learned_plan(plan, start, layout, held, truth, protocol["cap"], limit)
                        plans["learned_rollout"] = plan
                    cases.append(dict(layout=layout_id, start=start, regime=str(regime), budget=budget,
                                      truth=np.rint(truth * 1000).astype(int).tolist(), plans=plans))
    if learned or rollouts:
        raise ValueError("Unexpected learned scenes")
    if len(cases) != len(chosen) * len(protocol["regime_permeability"]) * len(protocol["travel_caps"]):
        raise ValueError("Incomplete replay cases")
    return dict(layouts=layouts, cases=cases, caps=protocol["travel_caps"], count=protocol["cap"],
                starts=protocol["starts"])


def verify_learned_plan(plan, start, layout, held, truth, cap, limit):
    sites, walk, endpoints = (plan[k] for k in ("sites", "walk", "endpoints"))
    count = len(sites)
    if (not 1 <= count <= cap or sites[0] != start or walk[0] != start
            or len(set(sites)) != count or held[sites].any()
            or (len(walk) - 1) / (layout.size - 1) > limit + 1e-12
            or len(endpoints) != count or endpoints[0] != 1 or endpoints[-1] != len(walk)
            or any(a >= b for a, b in zip(endpoints, endpoints[1:]))
            or [walk[i - 1] for i in endpoints] != sites
            or not all(layout.movement[u, v] for u, v in zip(walk, walk[1:]))):
        raise ValueError("Invalid learned route")
    for key in ("mean", "std"):
        if np.asarray(plan[key]).shape != (count, len(held)):
            raise ValueError("Invalid learned map shape")
    if len(plan["readings"]) != count:
        raise ValueError("Missing sensor values")
    if any(k in plan for k in ("forecasts", "futures")) and not all(
            len(plan.get(k, ())) == count - 1 for k in ("forecasts", "futures")):
        raise ValueError("Missing action forecasts")
    rounded = np.asarray(plan["mean"])[:, held] / 1000
    np.testing.assert_allclose(np.sqrt(np.mean((rounded - truth[held])**2, axis=1)), plan["rmse"],
                               atol=.0005, rtol=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/three-room-study")
    parser.add_argument("--learned-replay", type=Path, default=ROOT / "outputs/sensing-world-study/replay.json")
    parser.add_argument("--rollout-replay", type=Path, default=ROOT / "outputs/sensing-world-rollout/replay.json")
    parser.add_argument("--refined-study", type=Path, default=ROOT / "outputs/sensing-world-residential-refined-rollout")
    parser.add_argument("--output", type=Path, default=ROOT / "demo/index.html")
    args = parser.parse_args()
    template = (ROOT / "demo/index.template.html").read_text(encoding="utf8")
    data = build_data(args.source, args.learned_replay, args.rollout_replay)
    refined = None
    if args.refined_study.is_dir():
        from build_residential_replay import build_residential_replay

        refined = build_residential_replay(args.refined_study, ROOT)
    replacements = {
        TOKEN: json.dumps(data, separators=(",", ":")),
        REFINED_DATA_TOKEN: json.dumps(refined, separators=(",", ":")),
        WORLD_TOKEN: (ROOT / "demo/world.js").read_text(encoding="utf8"),
        REFINED_TOKEN: (ROOT / "demo/refined.js").read_text(encoding="utf8"),
    }
    for placeholder, content in replacements.items():
        if template.count(placeholder) != 1:
            raise ValueError(f"Expected one placeholder for {placeholder}")
        template = template.replace(placeholder, content)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(template, encoding="utf8")
    n_refined = len(refined["cases"]) if refined else 0
    print(f"Exported {n_refined} corridor and {len(data['cases'])} three-room replays to {args.output}")


if __name__ == "__main__":
    main()
