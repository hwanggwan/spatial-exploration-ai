# Greedy versus rollout with wall-aware ridge

**Rollout lowers final held-out error at both common budget caps. It walks farther under those caps, but its 16-reading advantage persists when constrained to greedy's actual walking distance.** This comparison follows the [wall ridge closed-loop study](wall-closed-loop-study.md) and holds the mapper fixed.

## Design

Both planners use the same frozen wall graph ridge, sensor noise, 12-by-12 layout, measurement mask, start, and legal walking distances. Only the decision rule changes. The first reading at the start counts. Budgets are 8 total readings within 1 map-side unit or 16 within 2 units. Neither planner may measure a reserved site, and travel between measurements follows four-neighbor shortest paths through the door. Traversing a site does not automatically acquire its value.

- **Greedy:** Select the feasible next reading with the highest expected whole-grid variance reduction per unit of walking distance.
- **Rollout:** Consider **all feasible** next readings, with no forced first move or candidate pruning. For each candidate, simulate its covariance update and greedy completion until no legal move remains or the reading cap is reached. Choose the first move of the completion with the lowest final whole-grid variance. Break numerical ties in favor of the greedy first move. Execute only that move and replan.

The ridge covariance is fixed and its Gaussian posterior covariance depends only on acquired locations. Thus covariance-only sequential replanning gives one deterministic route per layout, start, budget, and planner. We compute it once, then replay each chosen acquisition using that field's actual simulated sensor values. The replay produces the same mean and covariance that updating online would produce, and no values are needed to choose the route. This comparison does not test adapting to measured values or learning the wall's permeability online.

Test fields use a new seed block, 128 fields at each of the two test door positions, and the same 28 reserved latent-truth sites per layout as before. Four starts and two hidden field scales are balanced. Both field regimes have a physical wall for movement; dependence is weak across it in one regime and persists across it in the other. Both planners read the same noisy value at any shared site. Standard-deviation calibration uses a separate seed block and the original development layouts, for reporting intervals only. The settings are frozen before test generation.

## Same budget caps

All numbers average 256 matched fields per regime and budget. Every run reaches its reading cap. Differences are rollout minus greedy; the 95% paired percentile intervals use 6,000 bootstrap draws within door/start/field-scale strata.

| Field dependence | Reading / travel cap | Held-out RMSE, greedy → rollout | Paired RMSE difference (95% interval) | Actual walking distance, greedy → rollout |
| --- | ---: | ---: | --- | ---: |
| Weak across wall | 8 / 1 | 0.8654 → 0.8358 | −0.0297 [−0.0412, −0.0183] | 0.7727 → 1.0000 |
| Weak across wall | 16 / 2 | 0.7978 → 0.7420 | −0.0558 [−0.0738, −0.0378] | 1.6591 → 2.0000 |
| Persists across wall | 8 / 1 | 0.8652 → 0.8347 | −0.0304 [−0.0413, −0.0201] | 0.7727 → 1.0000 |
| Persists across wall | 16 / 2 | 0.7886 → 0.7438 | −0.0448 [−0.0629, −0.0271] | 1.6591 → 2.0000 |

For 16 barrier readings, RMSE falls 7.0%, near-wall RMSE falls from 0.8526 to 0.7707, and modeled final grid variance falls from 0.2583 to 0.2344. Rollout has lower modeled variance in all eight layout/start configurations at each cap, and changes the first move in all eight. At 16 readings, it measures on both sides in 8/8 configurations, versus 2/8 for greedy. At 8 readings those counts are 2/8 and 0/8. These are observed route patterns, not crossing requirements or evidence that crossing is generally beneficial.

At 16 barrier readings, calibrated coverage is 94.2% for greedy and 94.7% for rollout; widths are 3.08 and 2.96. The corresponding raw coverage is 80.9% and 81.9%. With one CPU thread on this grid, mean cumulative action-selection time is on the order of 1–3 ms for greedy versus 0.6 s for rollout at the larger cap (excluding prior construction, sensor reads, and physical movement). The measured compute cost is specific to the exhaustive-first-candidate rollout implementation. Rollout optimizes the model's terminal variance through candidate greedy completions, not the globally optimal route or true held-out RMSE.

## Secondary control at matched walking distance

After the primary comparison, limit each rollout to the distance that greedy actually walked from that same start on that layout. The cap depends only on greedy's route, not on field targets. Keep the reading cap and evaluate the same test fields, making this a sensitivity check rather than an independent test. Rollout uses the same actual distance as greedy in every tested configuration.

| Field dependence | Reading cap | RMSE, greedy → distance-matched rollout | Paired difference (95% interval) | Mean rollout readings |
| --- | ---: | ---: | --- | ---: |
| Weak across wall | 8 | 0.8654 → 0.8583 | −0.0071 [−0.0158, +0.0012] | 7.75 |
| Weak across wall | 16 | 0.7978 → 0.7650 | −0.0328 [−0.0508, −0.0146] | 15.25 |
| Persists across wall | 8 | 0.8652 → 0.8606 | −0.0045 [−0.0130, +0.0040] | 7.75 |
| Persists across wall | 16 | 0.7886 → 0.7689 | −0.0197 [−0.0360, −0.0036] | 15.25 |

At 16 readings, rollout uses the same distance and no more observations, yet improves RMSE under both regimes. At eight readings the distance-matched intervals include zero. This supports the value of terminal-budget planning here, but the comparison changes both the scoring objective and planning horizon. It does not isolate which change matters more. The intervals are conditional on the tested layouts/starts and unadjusted across contrasts.

The equal-distance route figure illustrates the first layout/start pair (in sorted order) where rollout uses fewer readings at the 16-reading cap, using its first barrier field. Selection does not use RMSE. Both routes walk 17 grid edges (1.545 map units) and share only the starting observation. Greedy takes all 16 readings in the right room; rollout takes three there and ten in the left room. On this example field, final held-out RMSE is 0.892 versus 0.539. Across the 32 barrier fields at that same layout and start, the means are 0.822 versus 0.788. Average left-room RMSE improves (0.956 to 0.647) while right-room RMSE worsens (0.625 to 0.873). The gray dots in the plot are reserved test sites; the ridge fits use only readings taken at the marked stops.

## Reproduction and verification

```sh
PYTHONPATH=src .venv/bin/python -m spatial_exploration.wall_rollout_study
PYTHONPATH=src .venv/bin/python tools/audit_wall_rollout_distance.py
PYTHONPATH=src .venv/bin/python tools/plot_wall_rollout_matched_routes.py
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

Use a fresh primary output directory. The output includes the frozen protocol, per-field metrics, route configurations, acquired measurements, evaluator-only truths, the paired comparison plot and a route example, plus the matched-distance audit. Verification checks the frozen settings hash, 32 route configurations, all 2,048 episodes and 22,528 walking legs. All acquisitions respect the movement graph, both caps, and the held-out mask. The final rerun reproduces all non-timing acquisitions and metrics exactly, and scoring replay has zero discrepancy. The Torch suite passes 97 tests with two optional TensorFlow skips; system Python passes 51 with 48 optional skips.
