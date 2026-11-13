"""Illustrate the first equal-distance 16-cap route pair with fewer rollout readings."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from spatial_exploration.wall_rollout_study import dump, load_settings, ridge_models, scenes, topology_design


def walking_path(layout, sites):
    path = [sites[0]]
    for a, b in zip(sites, sites[1:]):
        path.extend(layout.walk(a, b)[1:])
    return layout.points[path]


def render(path, design, truth, routes, means, example, by_room_counts, distance):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    layout = design.layout
    size = layout.size
    pad = 0.5 / (size - 1)
    door_y = layout.door_row / (size - 1)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), layout="constrained")
    limit = max(float(np.max(np.abs(values))) for values in (truth, *means.values()))
    shared = set.intersection(*(set(route) for route in routes.values()))
    colors = {"greedy": "#056589", "rollout": "#875026"}
    for ax, method, values in zip(axes, ("truth", *routes), (truth, *means.values())):
        img = ax.imshow(values.reshape(size, size), origin="lower",
                        extent=(-pad, 1 + pad, -pad, 1 + pad),
                        cmap="RdBu_r", vmin=-limit, vmax=limit)
        ax.vlines([.5, .5], [-pad, door_y + pad], [door_y - pad, 1 + pad], color="black", lw=2.6)
        ax.scatter(*layout.points[design.heldout].T, s=14, c="#737373",
                   edgecolors="#e6e6e6", linewidths=.5, zorder=3)
        ax.set(xticks=[], yticks=[], xlim=(-pad, 1 + pad), ylim=(-pad, 1 + pad))
        if method == "truth":
            ax.set_title("Hidden field\nGray dots: held out")
            continue
        sites = routes[method]
        color = colors[method]
        ax.plot(*walking_path(layout, sites).T, color="white", lw=3.5, zorder=4)
        ax.plot(*walking_path(layout, sites).T, color=color, lw=2, zorder=5)
        only = [site for site in sites if site not in shared]
        ax.scatter(*layout.points[only].T, s=33, color="white", edgecolors=color, lw=1.3, zorder=6)
        ax.scatter(*layout.points[sorted(shared - {sites[0]})].T, marker="D",
                   s=35, color="#ffcf58", edgecolors="#222222", zorder=7)
        ax.scatter(*layout.points[sites[0]], marker="s", s=70, color="#ffcf58",
                   edgecolors="#222222", zorder=8)
        ax.scatter(*layout.points[sites[1]], marker="^", s=65, color=color,
                   edgecolors="white", lw=1.0, zorder=8)
        left, right = by_room_counts[method]
        ax.set_title(f"{method.capitalize()} · {len(sites)} readings ({left} left, {right} right)\n"
                     f"Held-out RMSE {example[method]['all']:.3f}")
    fig.colorbar(img, ax=axes, shrink=.77, label="Latent sensor value")
    handles = [
        Line2D([], [], marker="s", linestyle="", markerfacecolor="#ffcf58", markeredgecolor="#222222",
               label="Shared start / first reading"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor="white", markeredgecolor="#222222",
               label="Measurement stop"),
        Line2D([], [], marker="^", linestyle="", markerfacecolor="#222222", markeredgecolor="white",
               label="First stop after start"),
        Line2D([], [], color="#555555", label="Legal walking path"),
    ]
    if shared - {next(iter(routes.values()))[0]}:
        handles.insert(1, Line2D([], [], marker="D", linestyle="", markerfacecolor="#ffcf58",
                                 markeredgecolor="#222222", label="Other shared reading"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .01), ncol=len(handles), frameon=False)
    fig.suptitle(f"Equal-distance route example · 16-reading cap · both walk {distance:.3f} map units")
    fig.savefig(path, dpi=170, bbox_inches="tight", pad_inches=.13)
    plt.close(fig)


def run(study, reference):
    frozen = json.loads((study / "frozen.json").read_text())
    settings, digest = load_settings(reference)
    if digest != frozen["reference_sha256"]:
        raise ValueError("Ridge settings do not match the archived comparison")
    configs = json.loads((study / "matched-distance.json").read_text())["configurations"]
    candidates = [r for r in configs if r["cap"] == 16
                  and r["rollout_measurements"] < r["greedy_measurements"]
                  and abs(r["rollout_distance"] - r["greedy_distance"]) < 1e-12]
    chosen = min(candidates, key=lambda r: (r["door"], r["start"]))
    door, start, cap = (chosen[k] for k in ("door", "start", "cap"))
    base = next(r for r in json.loads((study / "routes.json").read_text())
                if (r["door"], r["start"], r["cap"], r["planner"]) == (door, start, cap, "greedy"))
    routes = {"greedy": base["sites"], "rollout": chosen["sites"]}
    design = topology_design(door)
    layout, held = design.layout, design.heldout
    room_masks = {"all": held, "left": held & ~layout.room, "right": held & layout.room}
    ridge = ridge_models(design, settings[cap])["wall_graph"]
    distance = chosen["greedy_distance"]
    for route in routes.values():
        walk = walking_path(layout, route)
        assert route[0] == start and not held[route].any()
        np.testing.assert_allclose(np.linalg.norm(np.diff(walk, axis=0), axis=1).sum(), distance, atol=1e-12)

    with (study / "matched-distance-metrics.csv").open() as stream:
        saved = {int(row["seed"]): row for row in csv.DictReader(stream)
                 if (int(row["door"]), int(row["start"]), int(row["cap"]), row["regime"])
                 == (door, start, cap, "barrier")}
    scores = {p: [] for p in routes}
    example = None
    for meta, truth, sensor in scenes(design, frozen["test_fields_per_layout"], frozen["test_seed"]):
        if meta["start"] != start or meta["regime"] != "barrier":
            continue
        means = {p: ridge.condition(sites, sensor[sites]).mean for p, sites in routes.items()}
        row = {}
        for p, mean in means.items():
            row[p] = {k: float(np.sqrt(np.mean((truth[mask] - mean[mask])**2)))
                      for k, mask in room_masks.items()}
            np.testing.assert_allclose(row[p]["all"], float(saved[meta["seed"]][f"{p}_rmse"]), atol=1e-12)
            scores[p].append(row[p])
        if example is None:
            example = (meta["seed"], truth, means, row)
    seed, truth, means, example_scores = example
    assert len(scores["greedy"]) == len(saved) == len(scores["rollout"])
    by_room_counts = {p: [int(np.sum(~layout.room[sites])), int(np.sum(layout.room[sites]))]
                      for p, sites in routes.items()}
    summary = dict(selection="Lowest door/start among equal-distance 16-cap pairs with fewer rollout "
                             "readings, then first barrier seed; no error-based selection",
                   door=door, start=start, seed=seed, cap=cap, distance=distance,
                   shared_sites=sorted(set.intersection(*(set(sites) for sites in routes.values()))),
                   routes=routes, measurement_counts_left_right=by_room_counts,
                   example_rmse=example_scores, fields=len(saved),
                   mean_rmse={p: {k: float(np.mean([r[k] for r in values])) for k in room_masks}
                              for p, values in scores.items()})
    dump(study / "matched-distance-route-example.json", summary)
    render(study / "matched-distance-routes.png", design, truth, routes, means, example_scores,
           by_room_counts, distance)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("outputs/wall-rollout-study"))
    parser.add_argument("--reference", type=Path, default=Path("outputs/wall-topology-study/frozen.json"))
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        result = run(args.study, args.reference)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
