# Active sensing on residential floor plans

The earlier three-room experiments placed rooms in a row. This follow-up compares that arrangement with two residential patterns on the same 12 × 12 grid. The full floor plan, including doors, is available to both planners at the start.

| Layout family | Geometry |
| --- | --- |
| Linear rooms | Three rooms in a row. Two variants change both doorway positions. |
| Apartment corridor | Three rooms open into one shared hallway. Two variants change a partition and doorway positions. |
| Connected rooms | Four rooms form a loop with one doorway to each neighboring room. Two variants change the partitions and door locations. |

Each layout has the same four starting cells, at most 26 walking edges and 16 readings including the first, and 28 held-out cells allocated by zone area. The corridor is included in the field and its holdout. It is excluded from the reported room counts. The matched models use the frozen GP and four-mode heat-mixture priors from the range audit. Geometry is provided directly to each model through the graph. The residential layouts enter evaluation without retraining.

On each layout, the simulator generates a short- and long-range field from the same innovations and noise. It fixes the spectral exponent at 2 and solves for the scale that gives a within-zone 0–8-edge correlogram area of 2.5 or 4.5. Within each range condition it tests weak (0.02) and persistent (1.0) cross-wall conductance. The terminology refers to the generator's edge conductance. The room boundaries still block movement for both conditions. Each layout, conductance and start has six field draws per range, for 96 paired fields per layout family and range.

Both closed-loop planners evaluate their own sensing stops. On the GP route the two mappers receive exactly the same observations. On the learned route they also replay the same observations in a crossover. Each planner uses the shared move graph. GP rollout's covariance updates depend on site locations, so the experiment caches those routes by start. The learned rollout updates mixture probabilities with acquired values. Neither scorer supplies unobserved field values to the planner.

For each layout family and range, bootstrap intervals are paired and stratified by layout variant, conductance and start. They quantify field-to-field variation on these fixed layouts. Room counts, door arrangements, and room sizes differ by family, so across-family contrasts combine those differences. The layouts are stylized residential examples with free movement within each room.

## Reproduce

From the repository root, after generating the frozen model in the [range and roughness studies](sensing-world-generalization.md), run:

```sh
PYTHONPATH=src python tools/audit_residential_routes.py
```

Outputs include the protocol, per-field metrics, achieved ranges, example routes and a plan comparison image under `outputs/sensing-world-residential`. The image uses the first draw at the same starting cell in the long-range, weak cross-wall condition.

## Results

Each row averages 96 paired fields over two layouts, four starts, and two wall conductances. Both policies walk exactly 26 edges on every field. Each pair of values in the last two columns is GP / world. The corridor column counts rooms with at least one measurement and excludes the hallway.

| Layout | Range | GP on GP route | World on GP route | World on own route | Readings | Rooms sampled |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Linear | 2.5 | 0.842 | 0.833 | 0.817 | 16.00 / 15.82 | 2.38 / 2.75 |
| Linear | 4.5 | 0.672 | 0.637 | 0.612 | 16.00 / 15.49 | 2.38 / 2.77 |
| Corridor | 2.5 | 0.848 | 0.842 | 0.844 | 16.00 / 14.91 | 2.00 / 2.35 |
| Corridor | 4.5 | 0.705 | 0.674 | 0.649 | 16.00 / 14.66 | 2.00 / 2.42 |
| Loop | 2.5 | 0.817 | 0.812 | 0.826 | 15.88 / 15.92 | 4.00 / 4.00 |
| Loop | 4.5 | 0.627 | 0.591 | 0.583 | 15.88 / 15.84 | 4.00 / 4.00 |

At range 4.5, closed-loop world minus GP RMSE is −0.056 [−0.081, −0.032] in corridors and −0.044 [−0.060, −0.029] in loops. At range 2.5, the corresponding differences are −0.004 [−0.024, +0.015] and +0.009 [−0.006, +0.025]. On long-range corridor fields, the world mapper improves on GP with the same observations and improves by a further 0.025 [0.004, 0.048] on its own observations. In the long-range loops, most of the closed-loop gain is already present when both mappers replay the GP readings. For short-range loops both planners sample every room, and the mean error goes up when the world mapper switches to its own stops. The paired interval for that route change includes zero.

On long-range corridor fields, observed coverage of moment-based 95% next-reading intervals is 86.2% for GP and 97.0% for the world model. Average interval widths are 1.94 and 2.78 sensor units. Mean summed planning times in the six-worker run are approximately 0.92 and 2.00 seconds on those layouts. Timing values reflect concurrent CPU load. The audit verified both caps, paired scene identity, achieved range, and the legality of the displayed paths.

## Follow-up: coverage and known-range routing

The follow-up reuses every field, sensor draw, starting cell and holdout from the six-plan benchmark. All four policies share the 26-edge and 16-reading caps. Every recorded RMSE in this follow-up uses the same unrestricted four-mode world mapper on the specified planner's acquired observations. This includes the GP route. The goal is to measure the value of the acquired observations to a fixed mapper.

- **World rollout** uses the original expected terminal variance objective.
- **World rollout with coverage** adds 0.025 times the mean number of legal walking edges from each grid cell to its nearest measured stop at the end of the candidate route. The candidate suffixes are generated by the existing greedy continuation. The penalty reranks those suffixes using all grid coordinates. Its coefficient is fixed across the two range conditions and the six layouts.
- **Known-range rollout** uses the original objective with only the two trained components for the correct short or long range group. The corresponding fitted prior correlogram areas across the six layouts span 2.77–2.98 for the short group and 4.30–4.93 for the long group. The range label guides planning. It leaves the actual generating covariance unknown, and the scoring mapper still uses all four components.
- **GP rollout** uses the original fixed-prior GP navigation policy.

The two original arms use archived per-field metrics and scores from the earlier crossover. The audit rebuilds the observation process and reproduces the GP route's scores under both mappers before evaluating the new arms. Moment-based next-reading coverage in the output uses each arm's planning belief. It uses the same observations as the corresponding rollout, and it is separate from the common-mapper RMSE.

```sh
PYTHONPATH=src python tools/audit_residential_coverage.py
```

Outputs are in `outputs/sensing-world-residential-coverage`. The bootstrap contrasts are paired within layout variant, wall conductance, and start, with 6,000 resamples and unadjusted 95% intervals.

| Layout | Range area | GP route | Original world route | Coverage route | Known-range route |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear | 2.5 | 0.8325 | 0.8171 | 0.8165 | 0.8231 |
| Corridor | 2.5 | 0.8419 | 0.8443 | 0.8637 | 0.8446 |
| Loop | 2.5 | 0.8123 | 0.8262 | 0.8229 | 0.8297 |
| Linear | 4.5 | 0.6369 | 0.6124 | 0.6104 | 0.5792 |
| Corridor | 4.5 | 0.6744 | 0.6491 | 0.6646 | 0.6615 |
| Loop | 4.5 | 0.5905 | 0.5825 | 0.5826 | 0.5818 |

All 2,304 episodes walk exactly 26 edges and remain within the reading cap. The table uses 96 paired fields per row. On short-range loop layouts, coverage minus original is −0.0033 [−0.0090, +0.0026]. The GP route still has the smallest sample mean. On short-range corridors, coverage minus original is +0.0194 [+0.0058, +0.0337]. The coverage arm takes 14.43 readings in 2.47 rooms on average, compared with 14.91 readings in 2.35 rooms for the original world route. The difference is consistent with collecting fewer observations while spreading them across more rooms. On long-range corridors, the coverage penalty also raises RMSE by 0.0155 [0.0018, 0.0295]. The long-range linear and loop comparisons have coverage differences close to zero with intervals spanning zero.

All short-range known-group contrasts against the original rollout have intervals spanning zero. The largest known-group improvement is on long-range linear layouts (−0.0332 [−0.0492, −0.0181]). This is a diagnostic with an evaluation label, and its payoff depends on layout and range. The original and coverage scores use the same mixture and candidate generation. The coverage result is for one penalty weight. The known-group arm can still have a mismatch to the actual field range.

The metric archive includes realized reading counts, sampled rooms, and planner-belief coverage, interval width and planning times for all four routes. On short-range corridors the original and coverage arms have 92.3% and 92.2% nominal 95% next-reading coverage with widths 2.93 and 2.92. The coverage planner takes approximately 2.09 seconds per field in the parallel run, reflecting worker contention. Original planner timings come from the preceding archive.

## Follow-up: exact-covariance rollout in the short-range layouts

On the corridor and loop layouts at range area 2.5, the oracle receives the exact simulator latent covariance for the layout and wall conductance. It uses the existing GP rollout planner and shared observation noise variance. With a Gaussian field and fixed covariance, the uncertainty update depends on the selected sites. The planner therefore constructs one route per layout, conductance and starting cell, before seeing any field realization, and scores each greedy completion to the end of the available budget. The true realized field enters only the simulator and the scorer. The oracle uses the same rollout heuristic as GP, so its route is an informed reference rather than a globally optimized route.

The oracle uses the same 26-edge and 16-reading caps and the same sensor samples and holdouts as the four-policy comparison. Each route is scored using the unrestricted original four-mode world mapper. The audit reproduces the GP route's common-mapper score for each paired field. In total there are 32 oracle routes and 192 paired fields, 96 per family. Paired intervals use 6,000 bootstrap resamples within layout variant, wall conductance and start.

```sh
PYTHONPATH=src python tools/audit_residential_oracle.py
```

The experiment writes the protocol, oracle routes, range diagnostics, common-mapper metrics and contrast summaries to `outputs/sensing-world-residential-oracle`.

| Layout | GP route | Original world route | Coverage route | Known-range route | Oracle route |
| --- | ---: | ---: | ---: | ---: | ---: |
| Corridor | 0.8419 | 0.8443 | 0.8637 | 0.8446 | 0.8530 |
| Loop | 0.8123 | 0.8262 | 0.8229 | 0.8297 | 0.8075 |

The oracle minus original world RMSE contrast is +0.0086 [−0.0133, +0.0308] on corridors and −0.0188 [−0.0379, +0.0004] on loops. Relative to the GP route the corresponding contrasts are +0.0110 [−0.0075, +0.0291] and −0.0048 [−0.0217, +0.0126]. The most pronounced difference on loops is between exact covariance and the known-range group (−0.0222 [−0.0369, −0.0069]). This comparison jointly changes the covariance and removes mixture-aware route scoring, so it compares planners.

All arms walk exactly 26 edges. Oracle routes average 15.94 readings and 2.06 rooms in corridors, and 16.00 readings and 3.81 rooms in loops. The audit verifies 32 legal oracle routes and reproduces their average model variances by sequential conditioning. Planning times for the oracle average 0.66 and 0.77 seconds per route on corridors and loops respectively; each route is reused across the six field draws. On oracle-selected observations the unrestricted world model's nominal 95% next-reading intervals have 91.6% and 92.1% coverage, with widths 2.89 and 2.95. These results evaluate one rollout heuristic and a common learned mapper at a fixed range and budget.

### Exact-covariance mapping on the same oracle observations

Replay the 32 archived oracle routes on the same 192 short-range fields. The unrestricted world mapper and the exact-covariance GP both receive the same acquired locations and noisy readings. The exact GP uses the simulator's latent covariance and observation noise variance, and both predictions are evaluated on the same 28 withheld latent values. All world mapper RMSEs reproduce the route-only archive to within 1e-10. Sequential exact-GP mean and variance agree with direct batch conditioning on every replay.

| Layout | Fields | World mapper RMSE | Exact GP RMSE | Exact minus world [paired 95% interval] |
| --- | ---: | ---: | ---: | ---: |
| Corridor | 96 | 0.8530 | 0.8421 | −0.0108 [−0.0170, −0.0047] |
| Loop | 96 | 0.8075 | 0.8039 | −0.0035 [−0.0095, +0.0017] |

The average gains are about 1.3% on corridors and 0.4% on loops. These comparisons hold the realized routes fixed. The exact Gaussian posterior mean minimizes conditional mean squared error on those routes under the simulator. The estimated gain therefore describes the headroom from using the generating covariance for mapping on those routes. It does not establish the minimum error over all feasible measurement sets.

Nominal 95% pointwise latent interval coverage on corridors is 93.3% for the world mapper and 95.1% for the exact GP, with mean widths 3.09 and 3.24. The corresponding loop values are 93.0% and 95.0% with widths 2.97 and 3.20. Replaying sequential updates takes about 3.3 ms per field with the four-mode mapper and 0.7 ms with the exact GP on one CPU thread, excluding route planning and independent verification.

```sh
PYTHONPATH=src python tools/audit_residential_oracle_mapping.py
```

Per-field scores and paired summaries are written to `outputs/sensing-world-residential-oracle-mapping`.

### Increasing the reading cap at fixed travel

Compare reading caps of 16 and 24 under a 26-edge travel cap. Replay the archived 16-reading oracle routes and replan the 24-reading routes with the same exact-covariance rollout heuristic. Keep the same layout variants, starting cells, short-range fields, noisy sensor draws, and 28 hold-out cells. The start counts as the first reading. Each mapper receives the same observations within each route. Changing the cap can change the route from its first move, so the comparison measures the budget change with replanning.

The replay verifies legal walking edges, excludes the hold-out cells from measurement, and reproduces both mappers' archived RMSEs at the 16-reading cap to within 1e-10. The paired 95% intervals summarize 6,000 field resamples within layout variant, wall conductance, and start.

```sh
PYTHONPATH=src python tools/audit_residential_oracle_budget.py
```

The protocol, per-field metrics, route traces, and summary are saved to `outputs/sensing-world-residential-oracle-budget`.

| Layout | Mean readings, 16 → 24 | World RMSE, 16 → 24 | World change [paired 95% interval] | Exact GP RMSE, 16 → 24 | Exact GP change [paired 95% interval] |
| --- | ---: | ---: | ---: | ---: | ---: |
| Corridor | 15.94 → 20.94 | 0.8530 → 0.8266 | −0.0264 [−0.0446, −0.0075] | 0.8421 → 0.8191 | −0.0230 [−0.0405, −0.0058] |
| Loop | 16.00 → 22.50 | 0.8075 → 0.7920 | −0.0155 [−0.0317, +0.0005] | 0.8039 → 0.7914 | −0.0125 [−0.0270, +0.0019] |

Every route walks exactly 26 edges. With the higher cap, corridor routes take 18–24 readings and loop routes take 20–24. Two of the 16 corridor routes and four of the 16 loop routes reach 24 readings. World RMSE drops by 3.1% on corridors and 1.9% on loops. Both mapper intervals exclude zero on corridors and include zero on loops. The estimates condition on the tested layouts, wall conductances, and starts.

At the 24-reading cap, nominal 95% hold-out coverage is 93.1% and 92.2% for the world mapper on corridors and loops, versus 95.0% and 94.8% for exact GP. The corresponding mean interval widths are 3.02 and 2.91 for the world mapper and 3.16 and 3.12 for exact GP. Planning takes 1.12 s per corridor route and 1.51 s per loop route on one CPU thread in this run. Mapping takes 4.5–4.8 ms for the world mapper and about 1 ms for exact GP.

### Hold-out RMSE by distance to a reading

Replay the 16 short-range corridor routes from the 24-reading cap experiment. Each route uses 26 walking edges and contributes six fields. The unrestricted four-mode world mapper, frozen single-prior basis GP, and exact-covariance GP receive identical locations and noisy observations for each field. The frozen GP uses the same prior and feature construction as the earlier residential comparison. The world and exact GP reproduce all 96 archived field RMSEs to within 1e-10.

Distance is the minimum number of legal walking edges from a hold-out cell to any measured cell, following the floor plan's doors. A distance band pools squared errors over all its hold-out predictions before taking the square root. The intervals keep all predictions from a field together and use 6,000 paired bootstrap resamples within layout variant, wall conductance, and start. They are unadjusted for multiple bands.

| Nearest reading (edges) | Hold-out predictions | World RMSE | Frozen GP RMSE | Exact GP RMSE | World minus exact GP [paired 95% interval] |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1,218 | 0.6675 | 0.6728 | 0.6625 | +0.0050 [+0.0001, +0.0100] |
| 2 | 420 | 0.8829 | 0.8739 | 0.8649 | +0.0180 [+0.0041, +0.0319] |
| 3–4 | 342 | 0.9316 | 0.9224 | 0.9100 | +0.0216 [−0.0020, +0.0455] |
| 5–7 | 324 | 0.9582 | 0.9571 | 0.9581 | +0.0001 [−0.0172, +0.0172] |
| 8–14 | 384 | 1.0614 | 1.0806 | 1.0601 | +0.0013 [−0.0106, +0.0137] |

Cells two to four edges away account for 28% of hold-out predictions and 74% of the total world minus exact GP excess squared error in this replay. The two-edge band has the clearest difference. At five or more edges, both mappers have relatively high absolute RMSE and similar averages. Their distance to observations may limit reconstruction there. The world minus frozen GP intervals include zero in all five bands. Mean field RMSE is 0.8266 for the world model, 0.8298 for the frozen GP, and 0.8191 for exact GP.

For nominal 95% latent intervals, coverage across distance bands is 92.0–94.8% for the world mapper, 70.3–78.4% for the frozen GP, and 94.0–95.7% for exact GP. The full output includes band-specific interval widths and paired world-minus-GP contrasts. Sequential conditioning and posterior evaluation take roughly 5 ms per field for the world model and 1 ms for either GP on one CPU thread. Route planning is excluded.

```sh
PYTHONPATH=src python tools/audit_residential_holdout_distance.py
```

Per-point predictions, per-field errors, and distance-band summaries are saved to `outputs/sensing-world-residential-holdout-distance`.

### Short-range covariance adjustment on fixed routes

Replay the same 16 corridor routes, 96 fields, noisy readings, and hold-outs from the distance audit. For each layout, multiply the heat lengths within each of the two short modes by a constant so its within-room correlogram area falls from 2.82–2.91 to 2.50. The adjustment preserves the graph wall conductances, trained kernel coefficients, latent marginal variances, observation noise, long-mode kernels, and initial mode probabilities. Both mappers update mode probabilities from the same sensor values. The simulator's known range area sets the target for this diagnostic.

| Nearest reading (edges) | Original world RMSE | Adjusted world RMSE | Exact GP RMSE | Adjusted minus original [paired 95% interval] |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.6675 | 0.6673 | 0.6625 | −0.0003 [−0.0021, +0.0016] |
| 2 | 0.8829 | 0.8798 | 0.8649 | −0.0031 [−0.0077, +0.0018] |
| 3–4 | 0.9316 | 0.9300 | 0.9100 | −0.0016 [−0.0070, +0.0036] |
| 2–4 combined | 0.9051 | 0.9027 | 0.8855 | −0.0024 [−0.0062, +0.0014] |
| 5–7 | 0.9582 | 0.9592 | 0.9581 | +0.0010 [−0.0035, +0.0060] |
| 8–14 | 1.0614 | 1.0632 | 1.0601 | +0.0018 [−0.0009, +0.0047] |
| Mean field RMSE | 0.8266 | 0.8262 | 0.8191 | −0.0004 [−0.0022, +0.0015] |

The heat lengths shrink by 11.6–13.5%. At two edges the adjusted short components have correlations of 0.466–0.480. Weighting the original component kernels by their observation-updated probabilities gives an average two-edge correlation of 0.555–0.603 across the four layout and conductance groups. With the adjustment and its updated probabilities, those group averages are 0.503–0.570. The simulator has a two-edge correlation of 0.489–0.495. The change closes about 12% of the original 2–4-edge RMSE gap to exact GP, and the uncertainty interval includes zero. The result tests one adjustment to the shortest modes.

At 2–4 edges, nominal 95% hold-out coverage changes from 93.6% to 94.4%, and mean interval width from 3.28 to 3.37. Both world mappers take about 4–5 ms per field for posterior prediction on one CPU thread. Travel and readings are identical to the distance audit.

```sh
PYTHONPATH=src python tools/audit_residential_holdout_distance.py --match-short-range
```

The `outputs/sensing-world-residential-range-match` directory contains the two mappers' paired predictions, distance summaries, and the calibration of the adjusted kernels.

### Restrict the range-matched mapper to the two short modes

Replay the same 16 corridor routes and 96 fields. Compare the range-matched four-mode mapper with the same mapper restricted to its two calibrated short modes. Restriction sets the two long modes' prior probabilities to zero and puts equal initial probability on the remaining wall modes. Sensor readings still update their probabilities. Both short kernels, observation noise, routes, and hold-outs are identical across arms. This test supplies the short-range group to the mapper using the simulator condition.

| Nearest reading (edges) | Four modes RMSE | Short modes only RMSE | Exact GP RMSE | Short only minus four [paired 95% interval] |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.6673 | 0.6662 | 0.6625 | −0.0011 [−0.0047, +0.0021] |
| 2 | 0.8798 | 0.8703 | 0.8649 | −0.0096 [−0.0200, +0.0006] |
| 3–4 | 0.9300 | 0.9198 | 0.9100 | −0.0102 [−0.0220, +0.0012] |
| 2–4 combined | 0.9027 | 0.8928 | 0.8855 | −0.0098 [−0.0185, −0.0015] |
| Mean field RMSE | 0.8262 | 0.8220 | 0.8191 | −0.0041 [−0.0079, −0.0005] |

The primary 2–4-edge comparison pools 762 hold-out predictions and closes about 57% of the gap from the four-mode mapper to exact GP. The frozen GP scores 0.8960 on those same predictions. After conditioning on readings, the four-mode mapper assigns 12–33% mean probability to long modes across layout and wall-conductance groups. Restriction shifts the two-edge average correlation of the reweighted prior kernels from 0.50–0.57 to about 0.47. The paired RMSE intervals resample complete fields within layout variant, wall conductance, and start.

At 2–4 edges, nominal 95% latent interval coverage goes from 94.4% to 95.5% and mean width from 3.37 to 3.53. Mapping time goes from roughly 4.1 ms to 2.6 ms per field on one CPU thread. Both arms use the same travel and readings. The improvement is measured with the short group supplied to the mapper.

```sh
PYTHONPATH=src python tools/audit_residential_holdout_distance.py --short-only
```

The comparison saves per-field and per-point predictions, calibration, and paired summaries in `outputs/sensing-world-residential-short-only`.

### Range weights by room and corridor

Use new simulated fields on the two corridor floor-plan variants. Combine independent short- and long-range latent fields by multiplying them by the square roots of their location-specific weights before adding them. The simulator uses the resolvent-squared graph covariances with reference within-room areas of 2.5 and 4.5. Each of the four zones has a short-component weight of 0.1 or 0.9 in mixed fields. The test enumerates all 14 mixed assignments. It also includes fields generated uniformly from either single range. Each layout, conductance, start, and assignment has 12 field draws. The simulation seed differs from earlier audits.

Both mappers use the same learned heat-kernel components with the earlier short-mode range adjustment. The global mapper considers five spatially constant short weights (0, 0.1, 0.5, 0.9, and 1). The regional mapper considers the same profiles and the 14 mixed assignments. It assigns half its prior probability to each profile family and gives the two wall modes equal prior probability. Both mappers update the probabilities of their profiles and wall modes by the joint likelihood of the readings. The known-zone reference receives the generator's short-weight assignment and still infers the wall mode. Exact GP receives the generating covariance. The test reuses one archived route per layout and start across all simulations. Each route walks 26 edges, the mean realized count is 20.5 readings, and every mapper predicts the same 28 hold-outs.

| Field type | Fields | Global mean field RMSE | Regional mean field RMSE | Global 2–4-edge RMSE | Regional 2–4-edge RMSE | Regional minus global at 2–4 edges [paired 95% interval] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mixed | 2,688 | 0.7508 | 0.7484 | 0.8211 | 0.8175 | −0.0036 [−0.0044, −0.0027] |
| Uniform short | 192 | 0.8268 | 0.8280 | 0.9362 | 0.9391 | +0.0029 [+0.0005, +0.0053] |
| Uniform long | 192 | 0.6280 | 0.6295 | 0.6741 | 0.6762 | +0.0021 [−0.0007, +0.0050] |

On mixed fields, known zones score 0.8095 at 2–4 edges and exact GP scores 0.8018. The regional mapper's mean inferred short weights are 0.422 and 0.532 in zones with true weights of 0.1 and 0.9. Field-level bootstrap intervals use 3,000 paired resamples within layout variant, wall conductance, start, and assignment. The experiment evaluates zone-wise mixing on two layouts with fixed routes and a fixed prior.

On mixed fields, nominal 95% latent interval coverage is 94.8% globally and 95.1% regionally, with mean widths 2.91 and 2.92. Batched one-thread prediction costs about 0.05 ms per field globally and 0.16 ms regionally under the experiment's repeated-route batching.

```sh
PYTHONPATH=src python tools/audit_residential_zone_ranges.py
```

The protocol, field metrics, inferred zone weights, and summaries are written to `outputs/sensing-world-residential-zone-ranges`.

### Finer grid for reconstruction and measurement placement

Use a 23 × 23 grid with half the original step length and all 144 original grid nodes embedded. The address of each wall or door comes from the corresponding coarse-cell boundary. Half-step points on a zone boundary are assigned to the positive row or column side. The original walking path expands to two legal fine edges for every coarse edge. New fields are generated on the fine grid from the same resolvent-squared family, with short and long correlogram areas matched at the original physical lags on the original nodes. The 2,688 paired draws cover the two corridor layouts, two wall conductances, four starts, and 14 mixed assignments. Each field has the original 28 hold-outs and 72 additional fine-only hold-outs, with 18 in each zone.

For **reconstruction resolution**, keep the original archived walk, locations, and noisy sensor values. Compare the native coarse and refined regional mappers with the same mode weights and priors. Evaluate the coarse prediction between nodes using within-zone bilinear interpolation. For **measurement placement**, use the same native 23 × 23 mapper and variance-reduction-per-distance greedy planner in both arms, with the same fixed prior, fine movement graph, start, 52-edge travel budget, and 24-reading cap. The only planner input that differs is the eligible measurement mask. A matched-travel check truncates both executed walks at the shorter realized travel length for each pair and includes only readings acquired by that distance.

| Change | Original-node hold-out RMSE | Fine-only hold-out RMSE | Fine-only difference [paired 95% interval] |
| --- | ---: | ---: | ---: |
| Finer reconstruction with fixed readings | 0.7380 → 0.7372 | 0.7651 → 0.7631 | −0.0019 [−0.0022, −0.0016] |
| More measurement candidates, shared caps | 0.7718 → 0.7782 | 0.7854 → 0.8112 | +0.0258 [+0.0231, +0.0284] |
| More measurement candidates, equal realized travel | 0.7912 → 0.7819 | 0.8067 → 0.8131 | +0.0064 [+0.0039, +0.0087] |

All fixed-reading routes use 52 refined edges and average 20.5 readings. Under shared caps the mean actual travel is 51.50 and 43.75 edges and the mean realized readings are 18.38 and 23.75 for the original and expanded candidate sets. The expanded routes use an average of 17.25 fine-only measurement sites. With the same travel prefixes, both arms walk 43.50 edges on average and collect 15.13 and 23.63 readings, respectively. The fine candidate set improves original-node RMSE at the matched horizon and increases fine-only RMSE. These candidate-set effects apply to the greedy policy and the tested floor plans.

The intervals use 3,000 paired resamples within layout, wall conductance, start, and mixed assignment. Nominal 95% interval coverage on original nodes is 95.3% for coarse reconstruction and 95.0% for fine reconstruction with fixed readings. It is 94.8% and 94.7% for the native fine mapper under the two candidate masks with shared caps. Batched mapping takes around 0.2–0.3 ms per field on one CPU thread, excluding path planning.

```sh
PYTHONPATH=src python tools/audit_residential_grid_resolution.py
```

The output in `outputs/sensing-world-residential-grid-resolution` includes per-field scores, routes, paired summaries, and one fixed-index example map.

### Remaining-budget rollout on the refined grid

Compare three planners on the same 23 × 23 movement graph and regional world mapper. Coarse-candidate greedy can measure only original grid nodes. Fine-candidate greedy and fine-candidate rollout can measure any unreserved node. All arms share the same start, 52-edge travel cap, and 24-reading cap. The planners use the same frozen moment-matched prior covariance. Acquired sensor values update the world mapper's predictions and mixture weights during scoring. They do not affect these planners' routes.

At each decision, the rollout searches the union of the eight feasible first moves with the highest immediate whole-map variance reduction and the eight with the highest reduction per walking distance. It includes the baseline greedy first move. It runs a full greedy completion from each of these first moves until the remaining travel or reading budget is exhausted, selects the lowest terminal whole-map variance, executes its first move, and replans. Each completion can use the entire fine candidate set.

The test reproduces both greedy routes for each of the eight layout/start combinations. Their mixed-field RMSEs on original-node and fine-only hold-outs agree with all 2,688 archived results to within 1e-10. The test adds 192 uniformly short and 192 uniformly long fields, with separate innovations. All arms see the same truth and realized sensor noise where they measure in common. The primary RMSE averages field-level RMSE over the same 100 reserved locations on each layout. The paired 95% intervals use 3,000 resamples within layout, wall conductance, start, and range assignment. To control realized travel, a second comparison takes only the measurements that rollout had collected by each greedy route's stopping distance.

```sh
PYTHONPATH=src python tools/audit_residential_refined_rollout.py
```

The protocol, field metrics, full walks, equal-travel prefixes, and paired summaries are written to `outputs/sensing-world-residential-refined-rollout`.

| Field range | Fields | Coarse-candidate greedy RMSE | Fine-candidate greedy RMSE | Fine rollout RMSE | Fine rollout minus fine greedy [paired 95% interval] |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uniform short | 192 | 0.8530 | 0.8712 | 0.8164 | −0.0548 [−0.0669, −0.0426] |
| Mixed | 2,688 | 0.7833 | 0.8038 | 0.7533 | −0.0504 [−0.0542, −0.0470] |
| Uniform long | 192 | 0.6861 | 0.7092 | 0.6352 | −0.0740 [−0.0895, −0.0587] |

With the same budget caps, average travel is 51.50, 43.75, and 52.00 fine edges for coarse greedy, fine greedy, and fine rollout. Their mean reading counts are 18.38, 23.75, and 21.50. Consecutive one-edge stops account for 79% of fine greedy legs and 55% of fine rollout legs. Fine rollout measures in 3.25 zones on average versus 3.00 for fine greedy. It uses 16.0 new measurement locations on average. End-to-end planning takes about 21 ms, 41 ms, and 3.15 s per route, respectively, on a single CPU thread. Nominal 95% interval coverage for fine greedy and fine rollout is 93.7% and 93.6% on uniformly short fields, 94.8% and 94.9% on mixed fields, and 96.5% and 97.0% on uniformly long fields. The rollout uses slightly narrower intervals in all three conditions.

At fine greedy's actual distance, the averaged rollout prefix uses 15.63 readings versus 23.75 for fine greedy. The all-site RMSE differences are −0.0151 [−0.0271, −0.0034] on short fields, −0.0025 [−0.0062, +0.0010] on mixed fields, and −0.0128 [−0.0292, +0.0037] on long fields. On the 72 fine-only hold-outs in mixed fields the matched-travel contrast is −0.0049 [−0.0089, −0.0012]. At coarse greedy's actual distance the all-site rollout prefix contrasts remain below zero in all three conditions. Those prefixes average 21.00 readings versus coarse greedy's 18.38. These comparisons hold realized travel fixed by construction and retain the measurements actually made by that point in each route.
