# Field Routes

Open the [live demo](https://hwanggwan.github.io/spatial-exploration-ai/) in a browser, or open `index.html` locally to use it offline. The **3D rollout comparison** opens with the full-budget learned rollout and GP rollout side by side. They share the layout, starting site, field, 26-edge travel cap, and 16-reading cap. One timeline advances both by measurement count, and one camera angle applies to both. Stop colors distinguish shared locations from sites that only one planner has measured at the current frame. The 16 paired cases are the 26-edge cases from the same 32-scene archive.

 **3D world** shows the archived walking path and updates the floor color at each reading. Drag the scene or use the camera buttons to rotate it. **Compare maps** shows the Bayesian planners and the learned world model together. With the learned model selected, **Low**, **Mean**, and **High** show the map after a hypothetical reading at the next stop. The route metrics reflect readings taken so far. The arrows and slider let you jump to an individual reading. Set the floor plan, start, travel cap, and sensor dependence to select one of 32 paired scenes. The true field stays visible, on the same color scale as the predictions. **Uncertainty** changes only the planner maps and shows a separate scale. The timeline aligns measurement count; an arm that runs out of feasible measurements stays on its final frame.

The page replays archived experiments. The equal-allocation, priority-allocation, and GP rollout planners share a Bayesian ridge model. The learned world model uses a mixture of covariances trained offline. All planners know the movement layout. Each displayed posterior is computed from that route's archived sensor values at the selected prefix. The true field and the gray hold-out sites support evaluation. Ground truth does not participate in selecting a route or updating the posterior.

The two evaluation layouts, four starting sites, two travel budgets, and two field regimes produce 32 cases (128 individual routes). Scenes use the first archived field per layout and start, without selection by RMSE. Posterior means, standard deviations, truth, and the recorded noisy sensor readings are quantized to 0.001 for display only. RMSE uses the unrounded predictions and the same 28 evaluation sites. Planning code and full paired results are in the [three-room study](../docs/three-room-study.md).

To regenerate the standalone file after editing `index.template.html` or `world.js`, use the three-room and learned world model archives in the project's Python environment:

```sh
PYTHONPATH=src .venv/bin/python -m spatial_exploration.sensing_world_study
PYTHONPATH=src .venv/bin/python -m spatial_exploration.sensing_world_study --full-rollout
PYTHONPATH=src .venv/bin/python tools/build_three_room_demo.py
```

The exporter checks all archived route segments, budgets, evaluation masks, and terminal RMSEs against the benchmark. It writes only `demo/index.html`.

The live demo is served by GitHub Pages from the `gh-pages` branch. That branch contains the standalone replay as `index.html` and a `.nojekyll` marker. To update it, regenerate the replay on `main` and copy the new HTML to the publishing branch, with the Experiment notes link pointing to the GitHub report.
