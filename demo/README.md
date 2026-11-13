# Field Routes

Open the [live demo](https://hwanggwan.github.io/spatial-exploration-ai/) in a browser, or open `index.html` locally to use it offline. **Play** advances one reading at a time; the arrows or slider let you compare an individual step. Set the floor plan, start, travel cap, and sensor dependence to select one of 32 paired scenes. The true field stays visible, on the same color scale as the predictions. **Uncertainty** changes only the planner maps and shows a separate scale. The timeline aligns measurement count; an arm that runs out of feasible measurements stays on its final frame.

The page replays archived experiments. The equal-allocation, priority-allocation, and rollout planners use identical mapper parameters and know the movement layout. Each displayed posterior is computed from that route's archived sensor values at the selected prefix. The true field and the gray hold-out sites support evaluation. Ground truth does not participate in selecting a route or updating the posterior.

The two evaluation layouts, four starting sites, two travel budgets, and two field regimes produce 32 cases (96 individual routes). Scenes use the first archived field per layout and start, without selection by RMSE. Posterior means, standard deviations, and truth are quantized to 0.001 for display only. RMSE uses the unrounded predictions and the same 28 evaluation sites. Planning code and full paired results are in the [three-room study](../docs/three-room-study.md).

To regenerate the standalone file after editing `index.template.html`, use the existing three-room archive and the project's neural environment:

```sh
PYTHONPATH=src .venv/bin/python tools/build_three_room_demo.py
```

The exporter checks all archived route segments, budgets, evaluation masks, and terminal RMSEs against the benchmark. It writes only `demo/index.html`.

The live demo is served by GitHub Pages from the `gh-pages` branch. That branch contains the standalone replay as `index.html` and a `.nojekyll` marker. To update it, regenerate the replay on `main` and copy the new HTML to the publishing branch, with the Experiment notes link pointing to the GitHub report.
