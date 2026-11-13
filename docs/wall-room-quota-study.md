# Room quotas versus greedy and rollout

A cheap room quota recovers most of rollout's improvement in whole-map error in the known two-room task. At the same actual walking distance and measurement count, rollout still yields lower error near the wall. This extends the [wall-aware ridge rollout experiment](wall-rollout-study.md) without changing the mapper.

## Protocol

The known floor plan determines legal four-neighbor movement. Both field regimes have the same physical wall and door; only the field's cross-wall dependence differs. The wall-aware Bayesian ridge, sensor noise, and regularization are held fixed across all methods. The robot can take 16 readings including the start, and walk up to 2 map-side units. Walking through a site does not take a reading. We score the same 28 never-sensed grid sites per layout; eight of those are next to the wall.

- **Greedy** maximizes modeled variance reduction per unit walking distance for the next reading.
- **Room-quota greedy** aims to take eight readings in the start room and eight in the other room. Before each start-room action it checks that a cheap nearest-neighbor route through the door can still collect the eight other-room readings. If no such start-room action fits, it crosses early. In the other room, it continues the same greedy ranking while preserving a feasible completion for its outstanding quota. It does not cross back. If the initial nearest-neighbor check cannot certify the away-room quota, it uses ordinary greedy. The check is a sufficient feasibility witness, not an exhaustive proof of infeasibility. No truth values or field regime are used.
- **Rollout** considers every feasible next reading, then compares greedy completions by final whole-grid modeled variance under the caps. It makes no room allocation in advance.
- **Distance-matched rollout** uses the room quota's actual walking distance for that layout and start as its distance cap. The distance is determined by the layout and modeled covariance, without seeing any field values. This planned comparison checks whether differences survive at the same realized walking distance.

The two layouts have doors at rows 2 and 9 with four balanced starts each. Each regime has 256 new test fields across the layouts (seed base 1610000). Calibration uses distinct fields and the original development door positions (seed base 1510000); interval scale is for evaluation only. Because fixed Gaussian posterior covariance depends only on measurement locations, replanning yields deterministic routes for each layout and start. Each route is computed once and replayed with noisy readings on every matched field.

## Results

All arms take 16 readings in all eight test configurations. Counts per room below describe the individual routes, not just balanced starting positions across the suite.

| Method | Mean actual distance | Both rooms (of 8 configurations) | Barrier RMSE | Cross-wall dependence RMSE | Mean planning time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Greedy | 1.6591 | 2 | 0.8317 | 0.8066 | 1.8 ms |
| Room-quota greedy | 1.9205 | 8 | 0.7694 | 0.7588 | 2.1 ms |
| Rollout | 2.0000 | 8 | 0.7674 | 0.7586 | 585 ms |
| Distance-matched rollout | 1.9205 | 8 | 0.7698 | 0.7589 | 710 ms |

Room-quota greedy takes eight readings per room in every test configuration. Its barrier RMSE is 7.5% lower than greedy's, with paired difference −0.0622 [−0.0826, −0.0425], but it spends 0.2614 more distance. This shows the benefit at the same caps, without attributing all improvement to balancing the rooms independent of travel.

Distance-matched rollout uses exactly the same walking distance and number of readings in every configuration. Its whole-map advantage over room-quota greedy is unresolved: the paired differences (rollout minus quota) are +0.0003 [−0.0120, +0.0125] with a dependence barrier and +0.0002 [−0.0113, +0.0119] without one. These intervals do not establish equivalence. At the wall-adjacent test sites, however, matched rollout reduces RMSE from 0.8063 to 0.7682 with the barrier (difference −0.0381 [−0.0532, −0.0232]) and from 0.7682 to 0.7282 without it (−0.0400 [−0.0560, −0.0239]). Therefore covering both rooms captures most of the observed whole-map gain in this experiment; the chosen sites still matter for local accuracy. Intervals are paired percentile bootstrap estimates with 6,000 resamples within layout, start, and field-scale strata, unadjusted for multiple comparisons.

Barrier-regime calibrated coverage is 95.0% for quota and 94.9% for matched rollout, with mean widths 3.03 and 3.02. Without calibration, coverage is 80.5% and 81.0%. Computation times are per route, including quota feasibility checks, excluding mapper updates, geometry construction, sensing and physical movement. The lightweight rule is a strong baseline for whole-map accuracy on these two equal-sized rooms. Its fixed 8/8 target has not been validated on unequal rooms, many-room layouts, or very tight budgets.

## Reproduce

```sh
PYTHONPATH=src .venv/bin/python -m spatial_exploration.wall_room_quota_study
```

Use a fresh output directory. The run archives the protocol, frozen scales, route tables, per-field metrics, summary and a fixed layout/start plot. Routes are checked for legal walking, both resource caps, and no sensing at held-out sites; replayed covariance agrees with the planner's covariance. The finalized rerun produced identical per-field metrics and frozen settings. The system suite passed 52 tests with 48 optional skips; the Torch suite passed 98 tests with two optional skips. The new test checks that preserving the door reserve can override a nearer measurement and that an unverified quota falls back to greedy.
