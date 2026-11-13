# Closed-loop exploration with wall and open graph ridge

The [fixed-observation topology study](wall-topology-study.md) found a point-prediction gain from cutting graph edges where the wall weakens dependence. Here the two ridge mappers choose their own measurements. On barrier fields the wall-aware system gives a lower final hold-out RMSE, but swapping the routes while holding the mapper fixed does not establish that its routes are better.

## Design

- The two mappers have identical 380-input basis dimensions, diffusion length 0.22, prior variance 0.31623, intercept ratio 0.25, and sensor noise SD 0.12. The prior settings are the previous study's frozen shared values. Only the prediction graph changes: non-door wall edges have weight 1 (open) or 0 (wall). Neither gets the true field permeability.
- Both use the same movement graph, with four-neighbor walking and door-only wall crossings. They start with the same reading at the same location, and can measure at most 8 sites within 1 distance unit or 16 sites within 2 units. The start counts. Transiting a site does not measure it. No return to start is required.
- After each reading, condition the Bayesian ridge posterior on the acquired observations. The shared greedy rule chooses the affordable unmeasured site with greatest expected reduction in average whole-grid latent variance per walking distance. For covariance `C`, the reduction from site `i` is `mean(C[:, i]^2)/(C[i, i] + noise_variance)`. The next reading is provided only after choosing its site.
- With fixed Gaussian settings, the posterior covariance update depends on where the readings were taken; their numerical values affect only the mean. Consequently this rule replans after each measurement but is deterministic for a given graph, layout, start and cap. It does not use the field regime or infer permeability online.
- Test on new seed blocks with 128 fields at each of two door positions (rows 2 and 9), with four balanced starts and two hidden spatial scales. Under the two field regimes, the simulator uses the same driving Gaussian draw and sensor noise for each paired field, with non-door field edge weight 0.02 or 1. Within each regime both models read the same noisy value at common sites.
- Score latent truth at the same 28 never-sensed sites per layout, including eight near the wall. Marginal standard-deviation scales are learned on separate calibration fields (32 per door at rows 3 and 8), per mapper, route source, and cap. These scales are frozen before testing and used only for evaluating intervals, not to steer the planner.

## Closed-loop results

Averages across 256 fields per regime and cap. Distances are in units of the map side. Differences and intervals are wall minus open.

| Field dependence | Reading / distance cap | Open RMSE | Wall RMSE | Paired RMSE difference (95% interval) | Open actual distance | Wall actual distance |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| Weak across wall | 8 / 1 | 0.8981 | 0.8886 | −0.0095 [−0.0169, −0.0021] | 0.7841 | 0.7727 |
| Weak across wall | 16 / 2 | 0.8265 | 0.8115 | −0.0150 [−0.0280, −0.0024] | 1.6250 | 1.6591 |
| Persists across wall | 8 / 1 | 0.8687 | 0.8773 | +0.0086 [+0.0008, +0.0164] | 0.7841 | 0.7727 |
| Persists across wall | 16 / 2 | 0.7772 | 0.7949 | +0.0176 [+0.0042, +0.0314] | 1.6250 | 1.6591 |

All runs reach their observation cap. On barrier fields the wall-aware system lowers mean RMSE by 1.1% at eight readings while walking 1.4% less; at 16 readings it lowers RMSE by 1.8% while walking 2.1% more. The wall-aware route differs in 4/8 layout/start configurations at eight readings and 8/8 at 16. At 16 readings, the wall planner measures in both rooms in two of eight configurations; the open planner does so in three. Distance is deterministic for each configuration; table averages are over those eight balanced configurations. RMSE intervals use 6,000 paired bootstrap resamples within door/start/spatial-scale strata. They describe uncertainty across fields for this set of layouts and starts. They are unadjusted.

The barrier result applies to mean RMSE across all held-out sites. Near the wall at 16 readings, the wall-aware system's closed-loop RMSE is 0.8497 versus 0.8336 for open (difference +0.0161, interval [−0.0047, +0.0371]). The near-wall gain from the fixed-observation study does not carry over to the self-chosen acquisition comparison here.

## Isolating the route change

For each route, replay its acquired values with both mappers. For the 16-reading barrier case:

| Mapper | Open graph's route | Wall graph's route |
| --- | ---: | ---: |
| Open graph ridge | 0.8265 | 0.8184 |
| Wall graph ridge | 0.8137 | 0.8115 |

The wall mapper improves on either fixed route (paired differences −0.0128 [−0.0174, −0.0083] on open routes and −0.0069 [−0.0103, −0.0036] on wall routes). Holding the mapper fixed and changing the route gives a smaller or less certain result: −0.0081 [−0.0208, +0.0049] for the open mapper and −0.0023 [−0.0146, +0.0099] for the wall mapper. Both route intervals cross zero. No other regime/cap yields a clear route-only advantage either. The end-to-end system improves on barrier fields; the evidence for that improvement is clearer for the mapper than for route selection. Replay contrasts include differences in actual walking distance.

At 16 barrier readings, calibrated coverage is 94.0% for open and 94.2% for wall, with widths 3.16 and 3.11, respectively. Uncalibrated coverage is 78.8% and 80.7%. Combined posterior update and planning time averages about 2.5 ms per run at this cap on the small grid, excluding geometry and prior construction, physical sensing, and movement. Marginal interval calibration does not validate the joint covariance used for acquisition. This is a static synthetic study, without an optimal-planner claim or measured physical performance.

## Reproduce and verify

```sh
PYTHONPATH=src .venv/bin/python -m spatial_exploration.wall_closed_loop_study \
  --reference outputs/wall-topology-study/frozen.json --output outputs/wall-closed-loop-study
```

Use a fresh output directory. Artifacts include `protocol.json`, `frozen.json`, `metrics.csv`, `acquisitions.csv`, `evaluation-only.npz`, `summary.json`, and `routes.png` (the first 16-reading barrier case with differing routes, selected by route rather than score). The verification replay checks all 2,048 test routes and 22,528 movement legs for budgets, no reserved-site sensing, and legal walking; common readings are identical. It reproduces all 4,096 mapper-on-route metric records with zero discrepancy and confirms the frozen settings hash. The Torch suite passes 96 tests with two optional TensorFlow skips; system Python passes 50 with 48 optional dependency skips.
