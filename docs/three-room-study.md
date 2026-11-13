# Measurement allocation across three unequal rooms

A priority rule improves whole-map prediction over uniform quotas in these layouts. Rollout improves further, especially when it can spend enough distance to reach the far room. At the same realized walking distance as the priority rule, rollout's remaining advantage is smaller.

## Design

The 12-by-12 grid is split into three rooms, with widths 3, 6, and 3 columns. The robot starts in the left room. It knows the two door locations; the staggered doors make the right room more expensive to reach. The start counts as the first reading. The robot can collect up to 16 readings and walk up to 18 or 26 grid edges (1.6364 or 2.3636 map-side units). Movement follows the legal four-neighbor shortest path between selected sensing stops; transiting a site does not measure it. Every method uses the same wall-aware ridge settings and noise SD of 0.12.

- **Equal quota** enumerates orders that start in the left room and visit a subset of the other rooms. It divides readings as evenly as possible across selected rooms. It picks the plan with the most readings for which it can construct a legal completion, then the most rooms, then the shortest completion witness. A rule that prioritizes visiting rooms over filling the reading cap is examined separately.
- **Priority quota** enumerates the same room orders and uses the same feasibility check. Room weight is the sum of posterior variances after the first reading, divided by one plus walking distance from the start to the closest allowed stop in that room. Summing over the room gives larger rooms more weight when per-site uncertainty is similar. The policy assigns at least one reading per selected room, apportions the rest by weight, and chooses a feasible allocation by the fixed proxy `sum(weight * (1 - exp(-readings/4)))`. This has diminishing returns and can leave some readings unused. The room order and quota are fixed after the start; the policy does not learn transport parameters or revise quotas using measured values.
- Both quota methods choose stops using the same within-room variance-reduction-per-distance greedy rule. Every accepted stop preserves a concrete nearest-neighbor completion for all remaining room quotas. This is a sufficient feasibility check, not a proof that an alternative allocation is impossible.
- **Rollout** compares all feasible next stops by simulating greedy continuations to either cap, chooses the first stop with the best terminal whole-grid modeled variance, and replans. It has no room quotas. The study also runs rollout with each quota policy's actual walking distance as its cap. No test outcomes enter the cap.

The prior is fixed, so covariance and route choice depend on selected locations, not sensor values. Routes are computed once per layout, start, method and budget, then replayed using the actual noisy readings. The mapper sees only those acquired values. Calibration uses 32 fields per development layout, with door row pairs (3, 8) and (9, 4), zero indexed. Test uses distinct seed blocks and 128 fields per layout, with door row pairs (2, 9) and (8, 3). Four starts and two hidden field scales are balanced. Under each of the two regimes there are 256 paired fields per budget: one regime weakens dependence across walls and the other allows it to continue across walls. Movement is the same in both regimes.

Evaluation uses 28 common reserved sites per layout (7, 14, and 7 by room, proportional to area). We compute latent RMSE per field over these sites and average the field scores. Per-room RMSE is computed separately, so the aggregate RMSE is not an average of the three room RMSEs. Intervals for paired differences resample fields within layout, start, and scale (6,000 draws). They are descriptive, unadjusted, and conditional on the chosen layout family.

## Results

Every number is averaged over the same balanced fields within the indicated budget and regime.

| Travel cap | Method | Actual distance | Readings | Mean readings by room (left / middle / right) | Barrier RMSE | Cross-wall dependence RMSE |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| 1.6364 | Equal quota | 1.6364 | 15.75 | 11.75 / 4.00 / 0.00 | 0.8519 | 0.8210 |
| 1.6364 | Priority quota | 1.5682 | 13.50 | 5.38 / 8.12 / 0.00 | 0.7817 | 0.7621 |
| 1.6364 | Rollout | 1.6364 | 15.50 | 4.88 / 9.62 / 1.00 | 0.7519 | 0.7398 |
| 2.3636 | Equal quota | 2.0795 | 16.00 | 9.00 / 7.00 / 0.00 | 0.7834 | 0.7591 |
| 2.3636 | Priority quota | 1.8864 | 15.88 | 6.00 / 9.12 / 0.75 | 0.7470 | 0.7346 |
| 2.3636 | Rollout | 2.3636 | 16.00 | 3.00 / 9.38 / 3.62 | 0.6835 | 0.6729 |

On barrier fields the priority rule improves over equal allocation by 0.0702 [0.0531, 0.0879] at the tight cap and 0.0364 [0.0252, 0.0480] at the loose cap while walking less. With the loose cap, rollout reaches all three rooms for six of eight layout/start combinations, compared with two for priority and zero for equal. On barrier fields, right-room RMSE falls from 0.9486 with priority to 0.6888 with rollout; left-room RMSE increases from 0.6019 to 0.6697. This is a regional tradeoff. It does not establish that visiting the far room is always best.

**At the priority rule's realized distance,** rollout also walks exactly the same distance in every tested configuration. The reduction in barrier RMSE is 0.0174 [0.0052, 0.0299] at the tight cap and 0.0139 [0.0037, 0.0248] at the loose cap. Mean reading counts are 13.50 vs. 15.38 at the tight cap, and 15.88 vs. 16.00 at the loose cap (priority vs. matched rollout). Thus the effect is not from more travel, but some of it may be from fitting more readings into the same distance. When rollout is matched to equal quota's actual distance at the loose cap, both take 16 readings in every configuration; rollout still lowers barrier RMSE from 0.7834 to 0.7278. With cross-wall dependence, the priority-distance comparison reduces RMSE by 0.0116 [0.0019, 0.0216] at the loose cap, while the tight-cap interval crosses zero.

The equal baseline prioritizes count and can remain in the starting room. In a post-run sensitivity check, a coverage-first equal quota instead maximizes rooms visited before reading count, retaining the same allocation and feasibility machinery. Priority still has lower mean RMSE in both regimes and at both caps. The coverage-first rule takes only 8.25 or 9.25 readings on average. Neither comparison isolates the contribution of uncertainty mass from the priority rule's allocation, scoring, and stopping choices.

The quota policies take roughly 4 ms to plan a full route on this grid, compared with about 350–670 ms for full rollout (one CPU thread; geometry and belief updates excluded from these timings). On the loose-cap barrier fields, calibrated interval coverage is 95.2% for priority and 96.5% for rollout, with widths 2.92 and 2.88; the interval scale is for reporting only. These are synthetic results on two layouts with 25% / 50% / 25% room area. They do not establish general allocation rules for real buildings or arbitrary floor plans.

## Reproduce and verify

```sh
PYTHONPATH=src .venv/bin/python -m spatial_exploration.three_room_study
PYTHONPATH=src .venv/bin/python tools/audit_three_room_equal.py
```

The primary study takes its mapper settings from the frozen wall-topology study. Use a fresh output directory. The output stores the protocol, interval scales, deterministic route tables, per-field metrics, acquired values and evaluator-only truth. The route figure shows the first prespecified layout and start. Verification checks all 80 unique routes and 1,164 walking legs, including door-only crossings, both caps, and the exclusion of held-out sites. All 32 distance-matched route pairs agree in actual distance. Replaying all 5,120 method/field episodes (79,616 acquisitions) from the archives reproduces the recorded aggregate scoring metrics exactly. The system suite passed 53 tests with 48 optional skips; the Torch suite passed 99 with two optional skips.
