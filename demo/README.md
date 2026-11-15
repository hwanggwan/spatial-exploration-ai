# Field Routes

Open the [live demo](https://hwanggwan.github.io/spatial-exploration-ai/) in a browser, or open `index.html` locally to use it offline. The **Room coverage** replay shows a selected field where the minimum-two-per-room rollout lowers whole-map and short-zone error. Both routes start together and use 156 walking steps and 24 readings. The displayed RMSE uses the 484 cells unmeasured by both completed routes. [Room-coverage benchmark](../docs/residential-floor-plan-study.md#increasing-the-walking-budget-with-a-fixed-reading-cap)

Select **Greedy vs. rollout** for six replays on the same 23 × 23 grid spanning two corridor layouts and short, long, and mixed-range fields. **Shared budget caps** lets both planners run under the 52-step and 24-reading limits. **Same walking distance** replays the archived rollout up to greedy's stopping distance. The timeline aligns by walking steps, and the table below the scene reports aggregate results across the benchmark fields. [Corridor benchmark](../docs/residential-floor-plan-study.md#remaining-budget-rollout-on-the-refined-grid)

The true field and predictions share a sensor scale with blue for negative values, white for zero, and red for positive values. Each predicted map has a top-down uncertainty map beneath it, with a shared posterior standard deviation scale for methods in the same view. Dots mark readings taken so far. The README animation shows the room-coverage replay at sampled walking distances.

**World model vs. GP** shows the earlier 26-step, 16-reading comparison. Its timeline aligns by measurement count. Stop colors distinguish shared locations from sites that only one planner has measured at the current frame.

**3D world** shows the archived walking path and updates the floor color at each reading. Drag the scene or use the camera buttons to rotate it. **Compare maps** shows the Bayesian planners and the learned world model together. With the learned model selected, **Low**, **Mean**, and **High** show the prediction and uncertainty after a hypothetical reading at the next stop. The route metrics reflect readings taken so far. The arrows and slider let you jump to an individual reading. Set the floor plan, start, travel cap, and sensor dependence to select one of 32 paired scenes. The true field stays visible on the same color scale as the predictions. The timeline aligns measurement count, and a completed route stays on its final frame.

The page replays archived experiments. The equal-allocation, priority-allocation, and GP rollout planners share a Bayesian ridge model. The learned world model uses a mixture of covariances trained offline. All planners know the movement layout. Each displayed posterior is computed from that route's archived sensor values at the selected prefix. The true field and the gray hold-out sites support evaluation. Ground truth does not participate in selecting a route or updating the posterior.

The two evaluation layouts, four starting sites, two travel budgets, and two field regimes produce 32 cases (128 individual routes). Scenes use the first archived field per layout and start, without selection by RMSE. Posterior means, standard deviations, truth, and the recorded noisy sensor readings are quantized to 0.001 for display only. RMSE uses the unrounded predictions and the same 28 evaluation sites. Planning code and full paired results are in the [three-room study](../docs/three-room-study.md).

To regenerate the standalone file after editing the template or renderers, use the three-room and learned world model archives in the project's Python environment.

```sh
PYTHONPATH=src .venv/bin/python -m spatial_exploration.sensing_world_study
PYTHONPATH=src .venv/bin/python -m spatial_exploration.sensing_world_study --full-rollout
PYTHONPATH=src .venv/bin/python tools/build_three_room_demo.py
```

With `outputs/sensing-world-residential-refined-rollout` available, the builder also exports the corridor scenes. To regenerate that archive, first generate the wall topology settings as described in the [repository README](../README.md), then run the corridor studies from the repository root with fresh output directories.

```sh
export PYTHONPATH=src
python -m spatial_exploration.sensing_world_generalization
python tools/audit_sensing_world_modes.py
python tools/audit_residential_routes.py
python tools/audit_residential_coverage.py
python tools/audit_residential_oracle.py
python tools/audit_residential_oracle_mapping.py
python tools/audit_residential_oracle_budget.py
python tools/audit_residential_grid_resolution.py
python tools/audit_residential_refined_rollout.py
python tools/build_three_room_demo.py
```

The builder checks terminal and equal-travel RMSEs against the benchmark archive and writes `demo/index.html`.

To generate the room-coverage replay from the refined corridor archive, run the following benchmarks. The adaptive and oracle runs provide paired field scores and reference routes for the room and travel budget studies.

```sh
python tools/audit_residential_early_rollout.py
python tools/audit_residential_early_rollout.py --include-quarter
python tools/audit_residential_room_rollout.py
python tools/audit_residential_adaptive_rollout.py
python tools/audit_residential_adaptive_rollout.py --prioritize-short
python tools/audit_residential_adaptive_rollout.py --oracle-short
python tools/audit_residential_adaptive_rollout.py --match-oracle-bonus
python tools/audit_residential_room_quota.py --minimum 2
python tools/audit_residential_travel_budget.py
python tools/build_three_room_demo.py
```

The live demo is served by GitHub Pages from the `gh-pages` branch. That branch contains the standalone replay as `index.html` and a `.nojekyll` marker. To update it, regenerate the replay on `main` and copy the new HTML to the publishing branch, pointing the experiment links to the repository reports.
