"""Compare a room quota against greedy and rollout with the same wall-aware ridge."""

import argparse
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .baseline import GPBelief, write_csv
from .room_budget import RoomQuota
from .wall_rollout_study import (DOORS, REGIMES, STARTS, TEST_DOORS, Route, dump, load_settings,
                                 metrics, plan_route, ridge_models, scenes, topology_design)

CAP, DISTANCE = 16, 2.0
METHODS = ("greedy", "room_quota", "rollout")
CAL_SEED, TEST_SEED = 1510000, 1610000


def quota_route(design, ridge, start, cap=CAP, distance=DISTANCE):
    belief = GPBelief(ridge.prior_covariance, ridge.noise_std)
    before = time.perf_counter()
    quota = RoomQuota(design.layout, ~design.heldout, start, cap, distance)
    elapsed = time.perf_counter() - before
    route, walked = [start], [0.0]
    visited = np.zeros_like(design.heldout)
    visited[start] = True
    belief.observe(start, 0.)
    while len(route) < cap:
        before = time.perf_counter()
        nxt = quota.next(belief, route[-1], visited, distance - walked[-1])
        elapsed += time.perf_counter() - before
        if nxt is None:
            break
        walked.append(walked[-1] + float(design.layout.distances[route[-1], nxt]))
        route.append(nxt)
        visited[nxt] = True
        belief.observe(nxt, 0.)
    if quota.active and np.count_nonzero(visited & ~quota.home) < quota.away_target:
        raise RuntimeError("Failed to preserve a feasible quota completion")
    return Route(tuple(route), tuple(walked), float(belief.covariance.diagonal().mean()),
                 elapsed * 1000, 0, 0)


def make_routes(design, model, matched=False):
    results = {(s, p): (quota_route(design, model, s) if p == "room_quota"
                        else plan_route(design, model, s, CAP, DISTANCE, p))
               for s in STARTS for p in METHODS}
    if matched:
        # A secondary check caps rollout at the quota policy's realized distance.
        results.update({(s, "rollout_matched"): plan_route(
            design, model, s, CAP, results[s, "room_quota"].distances[-1], "rollout") for s in STARTS})
    return results


def evaluate(designs, settings, count, seed, scales=None, matched=False):
    rows, route_rows, ratios = [], [], {}
    for door, design in designs.items():
        model = ridge_models(design, settings)["wall_graph"]
        routes = make_routes(design, model, matched=matched)
        held = design.heldout
        for (start, method), route in routes.items():
            sites = list(route.sites)
            right = int(np.count_nonzero(design.layout.room[sites]))
            if held[sites].any() or len(sites) != len(set(sites)) or len(sites) > CAP or sites[0] != start:
                raise RuntimeError("Invalid sensing route")
            distance_cap = (routes[start, "room_quota"].distances[-1]
                            if method == "rollout_matched" else DISTANCE)
            if route.distances[-1] > distance_cap + 1e-12:
                raise RuntimeError("Travel cap exceeded")
            for a, b, d in zip(sites, sites[1:], np.diff(route.distances)):
                walk = design.layout.walk(a, b)
                if not (np.isclose(d, (len(walk)-1)/(design.layout.size-1), atol=1e-12, rtol=0)
                        and all(design.layout.movement[u,v] for u,v in zip(walk, walk[1:]))):
                    raise RuntimeError("Illegal walking leg")
            route_rows.append(dict(door=door, start=start, method=method, left=len(sites)-right,
                                   right=right, distance_cap=distance_cap, **asdict(route)))
        route_index = {(r["start"], r["method"]): r for r in route_rows if r["door"] == door}
        for meta, truth, sensor in scenes(design, count, seed):
            for (start, method), route in routes.items():
                if start != meta["start"]:
                    continue
                sites = list(route.sites)
                belief = model.condition(sites, sensor[sites])
                if not np.isclose(belief.covariance.diagonal().mean(), route.final_variance, atol=1e-12, rtol=0):
                    raise RuntimeError("Route replay has a different posterior covariance")
                if scales is None:
                    ratios.setdefault(method, []).extend(
                        np.abs(truth[held] - belief.mean[held]) / np.maximum(belief.std[held], 1e-8))
                else:
                    r = route_index[start, method]
                    rows.append(dict(**meta, method=method, distance=route.distances[-1],
                                     measurements=len(sites), left=r["left"], right=r["right"],
                                     **metrics(truth, belief, design, scales[method])))
    return rows, route_rows, {m: float(np.quantile(r, .95) / 1.96) for m,r in ratios.items()}


def summarize(rows, routes):
    rng = np.random.default_rng(161001)
    results = []
    methods = (*METHODS, "rollout_matched")
    fields = ("rmse", "near_wall_rmse", "distance", "measurements", "left", "right",
              "modeled_grid_variance", "coverage", "raw_coverage", "width", "interval_score")
    for regime in REGIMES:
        groups = {m: {(r["door"],r["seed"]):r for r in rows if r["regime"] == regime and r["method"] == m}
                  for m in methods}
        keys = sorted(groups["greedy"])
        if not all(set(g) == set(keys) for g in groups.values()):
            raise RuntimeError("Field pairs do not match")
        strata = {}
        for i, k in enumerate(keys):
            r = groups["greedy"][k]
            strata.setdefault((r["door"],r["start"],r["kappa"]), []).append(i)
        comparisons, near_wall_comparisons = {}, {}
        for a,b in (("room_quota","greedy"), ("rollout","room_quota"),
                    ("rollout_matched","room_quota"), ("rollout","greedy")):
            for metric, contrasts in (("rmse", comparisons), ("near_wall_rmse", near_wall_comparisons)):
                delta = np.array([groups[a][k][metric] - groups[b][k][metric] for k in keys])
                samples = np.concatenate(
                    [rng.choice(delta[idx], (6000, len(idx))) for idx in strata.values()], axis=1).mean(axis=1)
                contrasts[f"{a}_minus_{b}"] = dict(difference=float(delta.mean()),
                    interval=np.quantile(samples, [.025,.975]).tolist(), lower_rmse_fields=int(np.sum(delta < -1e-12)))
        results.append(dict(regime=regime, fields=len(keys), comparisons=comparisons,
            near_wall_comparisons=near_wall_comparisons,
            means={m:{f:float(np.mean([r[f] for r in g.values()])) for f in fields} for m,g in groups.items()},
            route_stats={m:dict(mean_planning_ms=float(np.mean([r["planning_ms"] for r in routes if r["method"] == m])),
                both_rooms=sum(r["left"]>0 and r["right"]>0 for r in routes if r["method"] == m),
                hit_count_cap=sum(len(r["sites"])==CAP for r in routes if r["method"]==m))
                for m in methods}))
    return results


def render(output, routes, designs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    door, start = TEST_DOORS[0], STARTS[0]
    design = designs[door]
    layout = design.layout
    y = door / (layout.size - 1)
    pad = .5 / (layout.size - 1)
    fig, axes = plt.subplots(1, 3, figsize=(10.3, 4), layout="constrained")
    for ax, method, label in zip(axes, METHODS, ("Greedy", "Room-quota greedy", "Rollout")):
        record = next(r for r in routes if (r["door"],r["start"],r["method"]) == (door,start,method))
        sites = list(record["sites"])
        walk = [sites[0]]
        for a,b in zip(sites,sites[1:]):
            walk.extend(layout.walk(a,b)[1:])
        ax.axvspan(-pad,.5,color="#edf4f6")
        ax.axvspan(.5,1+pad,color="#faf3e8")
        ax.vlines([.5,.5],[-pad,y+pad],[y-pad,1+pad],color="#171717",lw=2)
        ax.scatter(*layout.points[design.heldout].T,s=10,color="#a0a0a0")
        ax.plot(*layout.points[walk].T,color="#23677b",lw=1.7)
        ax.scatter(*layout.points[sites].T,s=24,facecolor="white",edgecolor="#23677b",zorder=3)
        ax.scatter(*layout.points[sites[0]],s=55,marker="s",color="#ffc751",edgecolor="#202020",zorder=4)
        ax.set(xlim=(-pad,1+pad),ylim=(-pad,1+pad),xticks=[],yticks=[],aspect="equal")
        ax.set_title(f"{label}\n{record['left']} left / {record['right']} right · distance {record['distances'][-1]:.3f}")
    fig.suptitle("Same start and budget · circles are readings, gray dots are held out")
    fig.savefig(output / "routes.png", dpi=170)
    plt.close(fig)


def run(reference, output):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Use a fresh output directory")
    settings,digest = load_settings(reference)
    spec = settings[CAP]
    protocol = dict(reference_sha256=digest, ridge=spec, caps=[CAP,DISTANCE], starts=STARTS,
                    calibration_doors=DOORS, calibration_seed=CAL_SEED, calibration_fields_per_door=32,
                    test_doors=TEST_DOORS, test_seed=TEST_SEED, test_fields_per_door=128, regimes=REGIMES,
                    quota="Aim for 8 in start room, preserve a feasible nearest-neighbor witness for 8 in other room; switch early if needed",
                    sensitivity="Also cap rollout at room-quota's deterministic actual distance; never use field outcomes",
                    metric="Final mean per-field latent RMSE at 28 common holdout sites; lower is better")
    dump(output/"protocol.json",protocol)
    with threadpool_limits(limits=1):
        _,_,scales = evaluate({d:topology_design(d) for d in DOORS},spec,32,CAL_SEED,matched=True)
        dump(output/"frozen.json",{**protocol,"std_scales":scales})
        designs = {d:topology_design(d) for d in TEST_DOORS}
        rows,routes,_ = evaluate(designs,spec,128,TEST_SEED,scales,matched=True)
    summary = summarize(rows,routes)
    write_csv(output/"metrics.csv",rows)
    dump(output/"routes.json",routes)
    dump(output/"summary.json",summary)
    render(output,routes,designs)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference",type=Path,default=Path("outputs/wall-topology-study/frozen.json"))
    parser.add_argument("--output",type=Path,default=Path("outputs/wall-room-quota-study"))
    args = parser.parse_args()
    for row in run(args.reference,args.output):
        print(row["regime"], {m:{k:round(v[k],5) for k in ("rmse","distance","measurements","left","right")}
                              for m,v in row["means"].items()})


if __name__ == "__main__":
    main()
