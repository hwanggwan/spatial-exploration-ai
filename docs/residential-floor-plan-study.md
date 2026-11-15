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

### Earlier uncertainty reduction during rollout

Replay the same 3,072 refined-grid fields with the same eight layout/start pairs, 52-edge travel cap, 24-reading cap, candidate shortlist, and regional prediction model. The two rollout policies share the fixed moment-matched covariance and greedy continuation. The previous policy scores terminal whole-grid variance. The new policy scores an equal-weight blend of terminal variance and mean whole-grid variance over the walking budget.

For a completed route, the score is \(\tfrac12 V(52)+\tfrac{1}{104}\int_0^{52}V(d)\,\mathrm{d}d\), where \(V(d)\) is the mean latent variance after readings acquired by edge \(d\). Variance stays constant between readings and after a route ends. At each replanning step, the already-traversed portion contributes the same constant for all candidates. The divisor remains the original 52-edge budget. The audit computes each candidate's variance reductions from a Cholesky factor of its observation covariance and verifies the integral for all executed routes against sequential conditioning.

The curves use the field-level RMSE on the same 100 hold-outs at every edge. The prediction at each edge conditions on the observations acquired by that point. The distance-average RMSE averages over the 52 unit walking intervals. Terminal RMSE includes all readings acquired by edge 52. The late-reading share excludes the starting reading and counts new readings after 75% of each route's actual walking distance. The paired intervals resample fields within layout, regime, start, and range assignment.

```sh
PYTHONPATH=src python tools/audit_residential_early_rollout.py
```

The output directory `outputs/sensing-world-residential-early-rollout` contains the per-distance curves, paired summaries, walking routes, and diagnostic plots. The terminal-only arm reproduces the eight exact archived walks and all final field RMSEs.

| Field range | Distance-average RMSE, old → new | Paired difference [95% interval] | Final RMSE, old → new | Paired final difference [95% interval] |
| --- | ---: | ---: | ---: | ---: |
| Short | 0.9096 → 0.9034 | −0.0062 [−0.0112, −0.0008] | 0.8164 → 0.8150 | −0.0015 [−0.0099, +0.0078] |
| Mixed | 0.8646 → 0.8592 | −0.0054 [−0.0068, −0.0040] | 0.7533 → 0.7583 | +0.0050 [+0.0021, +0.0079] |
| Long | 0.7920 → 0.7812 | −0.0108 [−0.0158, −0.0058] | 0.6352 → 0.6438 | +0.0086 [−0.0017, +0.0193] |

Both arms walk 52 edges on all paired routes. Their average counts change from 3.63 to 4.38 by edge 13, from 8.88 to 9.75 by edge 26, from 12.50 to 14.25 by edge 39, and from 21.50 to 23.13 at the end. The late share falls from 43.7% to 39.8%, while the absolute late count stays close to nine (9.00 versus 8.88). The new score raises the number of observations before the final quarter. Both arms respect the 24-reading cap. The RMSE comparison includes the difference in the realized reading counts.

The mean fixed-prior variance along the walk falls from 0.7536 to 0.7337, and its terminal value from 0.5489 to 0.5429. The mixture mapper's empirical hold-out RMSE improves earlier on all three field groups. It rises at the end on mixed fields. Terminal nominal 95% hold-out coverage stays similar within each group (93.6% on short, approximately 94.8–94.9% on mixed, and 96.8–97.0% on long). One-thread planning time averages 3.22 s for the terminal objective and 3.69 s for the new objective.

### Reducing the distance-average weight to 25%

Use 75% terminal whole-grid variance and 25% distance-average variance, with the same original 52-edge horizon and shortlist. Replan the eight routes for this weight and replay the same 3,072 fields and 100 hold-outs. The two reference arms use their archived routes. Their distance-average and terminal RMSEs are reproduced for every field. The direct 25%-minus-50% intervals use the same paired resampling stratification as the baseline comparison.

```sh
PYTHONPATH=src python tools/audit_residential_early_rollout.py --include-quarter
```

The comparison writes the three curves, route resources, and paired contrasts to `outputs/sensing-world-residential-quarter-rollout`.

| Field range | Distance-average RMSE, 0% / 25% / 50% | Fraction of the 50% gain retained at 25% | Final RMSE, 0% / 25% / 50% |
| --- | ---: | ---: | ---: |
| Short | 0.9096 / 0.9093 / 0.9034 | 5% | 0.8164 / 0.8133 / 0.8150 |
| Mixed | 0.8646 / 0.8637 / 0.8592 | 16% | 0.7533 / 0.7529 / 0.7583 |
| Long | 0.7920 / 0.7879 / 0.7812 | 38% | 0.6352 / 0.6351 / 0.6439 |

The distance-average contrasts for 25% minus 0% are −0.0003 [−0.0032, +0.0030] on short, −0.0009 [−0.0017, −0.00004] on mixed, and −0.0041 [−0.0075, −0.0008] on long fields. At edge 26 in mixed fields, RMSE is 0.8641, 0.8609, and 0.8557 for 0%, 25%, and 50%. The 25% weight retains 38% of the midpoint gain and 16% of the distance-average gain. At the end, the direct mixed-field difference for 25% minus 50% is −0.0054 [−0.0079, −0.0030], and the 25% minus 0% difference is −0.0004 [−0.0019, +0.0011].

All routes walk 52 edges. Mean reading counts at the end are 21.50, 22.13, and 23.13 for 0%, 25%, and 50%. The respective late shares are 43.7%, 40.8%, and 39.8%. Nominal 95% coverage on mixed fields is 94.9%, 94.9%, and 94.8%. The 25% planner takes about 4.29 seconds per route in this run, while timings for 0% and 50% are taken from the archived run.

### Penalizing the largest remaining room variance

Keep 25% of the score on mean whole-grid variance during the walk. Reallocate one quarter of the terminal portion to the largest of the three room-average terminal latent variances. The resulting weights are 56.25% for terminal whole-grid variance, 18.75% for the largest terminal room variance, and 25% for the distance average. The hallway remains part of the whole-grid terms. A room's variance is the mean posterior diagonal over all its cells.

Replan both policies at each of the eight layout/start combinations with the same 52-edge and 24-reading caps, frozen covariance, shortlist, and greedy completion. The rollout uses observation locations to update this covariance. The mapper then conditions on the realized sensor values on each route. Evaluate the same 3,072 fields on 100 reserved sites per layout. Worst-room RMSE is the maximum of the three room-level hold-out RMSEs for each field. The intervals are based on 3,000 paired resamples within layout, conductance, start, and range assignment. The baseline reproduces the archived routes and all final field RMSEs.

```sh
PYTHONPATH=src python tools/audit_residential_room_rollout.py
```

The audit writes the protocol, routes, field metrics, and paired summaries to `outputs/sensing-world-residential-room-rollout`.

| Field range | Final whole-map RMSE, baseline → room-weighted | Paired difference [95% interval] | Final worst-room RMSE, baseline → room-weighted | Paired difference [95% interval] |
| --- | ---: | ---: | ---: | ---: |
| Short | 0.8133 → 0.8291 | +0.0158 [+0.0032, +0.0283] | 1.0118 → 1.0196 | +0.0078 [−0.0191, +0.0346] |
| Mixed | 0.7529 → 0.7487 | −0.0043 [−0.0082, −0.0003] | 0.9706 → 0.9409 | −0.0296 [−0.0394, −0.0204] |
| Long | 0.6351 → 0.6281 | −0.0070 [−0.0238, +0.0094] | 0.8341 → 0.8263 | −0.0078 [−0.0446, +0.0276] |

| Zone | Baseline readings by edge 39 | Room-weighted readings by edge 39 |
| --- | ---: | ---: |
| Lower left room | 0.625 | 0.625 |
| Lower right room | 0.750 | 1.750 |
| Hallway | 2.250 | 3.500 |
| Upper room | 9.875 | 4.250 |

The mean number of physical rooms measured by edge 39 rises from 1.875 to 2.125. Average readings by that point fall from 13.50 to 10.125. Mixed-field whole-map RMSE at edge 39 rises from 0.8198 to 0.8257, with paired change +0.0059 [+0.0031, +0.0086]. The mixed-field worst-room RMSE change at edge 39 is −0.0020 [−0.0077, +0.0031]. All routes walk 52 edges. Mean final readings fall from 22.125 to 17.375, so the comparison holds travel and both budget caps fixed and includes differences in realized readings.

The frozen planning covariance gives mean terminal worst-room variance of 0.8847 versus 0.7064 and whole-grid terminal variance of 0.5448 versus 0.5755. Some modeled gains are specific to the room objective, and empirical results vary by field range. Nominal 95% hold-out coverage is 93.6% versus 92.9% on short fields, 94.86% versus 94.89% on mixed fields, and 96.9% versus 97.1% on long fields. One-thread mean planning time is 4.67 versus 3.49 seconds per route and the mean number of first-move branches evaluated is 305 versus 234 across decisions on a route.

### Updating local range weights from each sensor reading

After each reading, evaluate the marginal Gaussian likelihood of the accumulated readings for each range-profile and wall-mode pair. Normalize with the frozen mapper's prior and sum over wall modes to obtain the posterior mass for each range profile. A zone's estimated short-scale weight is the posterior mean of that profile's short-scale coefficient. Form a plug-in Gaussian planning prior by averaging each profile's covariance over the original wall prior, using the newly inferred profile masses. Condition that covariance on the measured locations and run the original 25% distance-aware rollout. The greedy continuations use that covariance for their hypothetical readings. This approximation updates profile weights after real observations and does not integrate future changes to the profile weights inside the rollout.

The primary comparator is the archived 25% distance-aware rollout with fixed prior weights. The archived worst-room-weighted rollout provides a second fixed-route reference. All arms share the same starting locations, 52-edge and 24-reading caps, held-out cells, covariance bank, and final mixture mapper. The rollout consumes only readings at its chosen sites. True simulator coefficients above 0.5 label short zones for scoring, including the hallway when its coefficient is short. Final RMSE uses the same 100 held-out sites. The paired bootstrap resamples 3,000 times within layout, wall conductance, start, and range assignment.

```sh
PYTHONPATH=src python tools/audit_residential_adaptive_rollout.py
```

The default run includes the first four archived draws in each of the 256 strata. The output in `outputs/sensing-world-residential-adaptive-rollout` contains the paired field metrics, aggregate comparisons, and successive inferred short-scale weights along each adaptive route.

| Field range | Fields | Fixed 25% RMSE | Fixed worst-room rollout RMSE | Adaptive RMSE | Adaptive minus fixed 25% [paired 95% interval] |
| --- | ---: | ---: | ---: | ---: | ---: |
| Short | 64 | 0.81925 | 0.82967 | 0.82352 | +0.00426 [−0.00516, +0.01358] |
| Mixed | 896 | 0.75019 | 0.73965 | 0.75139 | +0.00120 [−0.00093, +0.00330] |
| Long | 64 | 0.66778 | 0.63872 | 0.66960 | +0.00182 [−0.00433, +0.00778] |

| Mixed-field metric | Fixed 25% | Fixed worst-room | Adaptive |
| --- | ---: | ---: | ---: |
| Actual travel | 52 | 52 | 52 |
| All readings | 22.125 | 17.375 | 21.576 |
| Readings in true short zones | 11.063 | 8.688 | 10.865 |
| Share of readings in true short zones | 50.00% | 50.00% | 49.88% |
| Hold-out RMSE in true short zones | 0.79534 | 0.79441 | 0.79805 |
| Worst-room hold-out RMSE | 0.97152 | 0.92954 | 0.97231 |

The adaptive-minus-fixed-25% paired change is −0.198 short-zone readings [−0.282, −0.113] and −0.12 percentage points in short-zone share [−0.43, +0.17]. Excluding the hallway, mixed-field short-room counts are 9.75, 6.44, and 9.50 for fixed 25%, fixed worst-room, and adaptive rollout. The adaptive site sequence changes on 849 of 896 mixed fields. Its average posterior short-scale coefficient in true short versus long zones is 0.510 versus 0.496 after five readings and 0.546 versus 0.458 at the end. By comparison, the true coefficients are 0.9 and 0.1. The final separation is 0.088 for adaptive observations and 0.086 for fixed 25% observations under the same inference procedure.

On mixed fields, nominal 95% interval coverage is 94.86%, 95.02%, and 94.91% for fixed 25%, fixed worst-room, and adaptive rollout, respectively. Mean adaptive planning time is 1.40 seconds per mixed-field route during the six-worker run. The fixed paths are replayed from disk. The benchmark reproduces their archived hold-out scores and checks compact and original 25% rollout decisions against the eight archived paths.

### Prioritizing variance reduction in estimated short-range zones

Use the same observation-driven range-profile inference and planning covariance. In zone \(r\), give each cell raw weight \(1+4\max(\widehat w_r-0.5,0)\), where \(\widehat w_r\) is the posterior mean short-scale coefficient. A coefficient of 0.60 gives a cell 1.4 times the raw weight of a cell at or below 0.50. Normalize the weights over the grid. Use their weighted mean variance for the first-move shortlist, the greedy continuations, and both the terminal and during-walk terms. The planner holds the weights fixed within each simulated continuation and updates them after each real reading. The hallway is eligible for weighting as its own zone.

Compare on the same 1,024 fields, starts, 52-edge and 24-reading caps, and 100 hold-outs per layout. Use the same final mixture mapper on each policy's observations. The fixed 25% and covariance-adaptive rollouts are the main comparators. The latter isolates the effect of changing the objective. The fixed worst-room route is retained as an additional reference. The simulator uses coefficients above 0.5 to label short zones for evaluation. The acquisition function consumes only the posterior estimates.

```sh
PYTHONPATH=src python tools/audit_residential_adaptive_rollout.py --prioritize-short --workers 8
```

The output directory `outputs/sensing-world-residential-short-priority-rollout` contains per-field metrics, adaptive routes with posterior weight traces, and paired bootstrap summaries.

| Mixed-field metric (896 fields) | Fixed 25% | Covariance-adaptive | Short-priority |
| --- | ---: | ---: | ---: |
| Actual movement steps | 52 | 52 | 52 |
| Total readings | 22.125 | 21.576 | 21.681 |
| Readings in true short zones | 11.063 | 10.865 | 11.330 |
| Share of readings in true short zones | 50.00% | 49.88% | 51.54% |
| Distinct true short zones measured | 1.625 | 1.625 | 1.576 |
| True-short-zone hold-out RMSE | 0.79534 | 0.79805 | 0.79350 |
| Whole-map hold-out RMSE | 0.75019 | 0.75139 | 0.75348 |

Against fixed 25% rollout, the paired changes are +0.268 short-zone readings [+0.146, +0.392], +1.54 percentage points of short-zone share [+1.11, +1.98], −0.00185 short-zone RMSE [−0.00599, +0.00235], and +0.00329 whole-map RMSE [+0.00006, +0.00660]. Excluding hallway measurements, the short-room counts are 9.75, 9.50, and 9.97 across the three arms. Against covariance adaptation alone, the paired changes are +0.465 short-zone readings [+0.345, +0.590], −0.00455 short-zone RMSE [−0.00813, −0.00096], and +0.00209 whole-map RMSE [−0.00081, +0.00521]. The number of readings within true short zones rises, while the number of distinct short zones measured falls.

On the 64 uniformly short fields, whole-map RMSE is 0.81925, 0.82352, and 0.82854. On the 64 uniformly long fields, it is 0.66778, 0.66960, and 0.66880. The paired whole-map RMSE contrast between short-priority and fixed rollout includes zero in both uniform conditions. On mixed fields, nominal 95% hold-out coverage is 94.86%, 94.91%, and 94.93%. Mean short-priority planning time per mixed field is 1.49 seconds during the eight-worker run. This comparison replays the reference arms from the previous audit.

### Oracle range coefficients for short-zone priority

Replace the estimated coefficient in the priority score with the simulator's true coefficient. The same function \(1+4\max(w_r-0.5,0)\) gives raw cell weights of 2.6 in true short zones and 1.0 in true long zones on mixed fields. Uniform true coefficients reduce to uniform priority after normalization. The oracle supplies the score only. The planner still builds its covariance from observed sensor values, and the final mixture mapper uses the same inference procedure as the other arms. The true field covariance stays with the simulator.

The comparison replays the same 1,024 fields, starts, 52-edge and 24-reading caps, and held-out sites from the estimated-priority run. The 896 mixed fields measure targeted allocation. The 128 uniform fields check the uniform-weight case. Reference field metrics are read from the previous run. Paired intervals use the same 3,000 resamples within layout, wall conductance, start, and range assignment.

```sh
PYTHONPATH=src python tools/audit_residential_adaptive_rollout.py --oracle-short --workers 8
```

The output in `outputs/sensing-world-residential-short-oracle-rollout` includes the oracle routes, inferred coefficient traces, per-field results, and paired interval summaries.

| Mixed-field metric (896 fields) | Fixed 25% | Estimated short-priority | Oracle short-priority |
| --- | ---: | ---: | ---: |
| Actual movement steps | 52 | 52 | 52 |
| Total readings | 22.125 | 21.681 | 22.330 |
| Readings in true short zones | 11.063 | 11.330 | 18.174 |
| Share of readings in true short zones | 50.00% | 51.54% | 81.11% |
| Distinct true short zones measured | 1.625 | 1.576 | 1.839 |
| True-short-zone hold-out RMSE | 0.79534 | 0.79350 | 0.68320 |
| True-long-zone hold-out RMSE | 0.66698 | 0.67908 | 0.82316 |
| Whole-map hold-out RMSE | 0.75019 | 0.75348 | 0.77538 |

The oracle gains 7.112 short-zone readings over the fixed 25% arm [7.019, 7.209] and 6.844 over estimated priority [6.690, 6.998]. Excluding hallways, short-room counts are 9.75, 9.97, and 15.87. Relative to fixed 25%, the paired oracle difference is −0.11214 [−0.12064, −0.10341] for true-short-zone RMSE and +0.02519 [+0.01781, +0.03266] for whole-map RMSE. Relative to estimated priority, these differences are −0.11030 [−0.11865, −0.10144] and +0.02190 [+0.01463, +0.02920]. The long-zone RMSE is calculated from the per-field whole-map and short-zone squared-error sums and the number of held-out cells in each zone set. Long-zone readings fall from 11.06 to 4.16 versus the fixed 25% arm.

The oracle reproduces covariance-adaptive routes on all 128 uniform fields. On mixed fields, nominal 95% interval coverage is 94.86%, 94.93%, and 94.72% across the three arms. Mean oracle planning time is 1.90 seconds per mixed-field route in the eight-worker run. The oracle comparison combines correct zone assignments and the raw score differences produced by the true coefficients.

### Oracle priority with matched bonus strengths

At each decision on a mixed field, compute the four raw zone weights from the sensor-based coefficients using the estimated-priority rule. Preserve their sorted values and assign the largest weights to the true short zones. Within each of the true short and long groups, assign larger weights to zones with higher estimated coefficients. Normalize over map cells and apply the same weighted rollout as before. Thus the two policies share the same relative weight magnitudes at a common observation history. The oracle provides the assignment of the weights only. The policies can infer different weights along their own routes as they collect different observations. Uniform oracle profiles yield uniform weights.

The audit compares the same 1,024 fields, starts, 52-edge and 24-reading caps, and held-out cells against the archived fixed, estimated-priority, and full-strength oracle routes. The planning covariance and mapper inference use the same sensor-based procedure in both oracle arms. The paired intervals use 3,000 resamples within layout, wall conductance, start, and range assignment.

```sh
PYTHONPATH=src python tools/audit_residential_adaptive_rollout.py --match-oracle-bonus --workers 8
```

The output in `outputs/sensing-world-residential-matched-oracle-rollout` contains the paired results and the sensor-estimated range coefficients along each matched-strength route.

| Mixed-field metric (896 fields) | Fixed 25% | Estimated short priority | Full-strength oracle | Matched-strength oracle |
| --- | ---: | ---: | ---: | ---: |
| Actual movement steps | 52 | 52 | 52 | 52 |
| Total readings | 22.125 | 21.681 | 22.330 | 21.516 |
| Readings in true short zones | 11.063 | 11.330 | 18.174 | 11.616 |
| Share of readings in true short zones | 50.00% | 51.54% | 81.11% | 53.23% |
| True-short-zone hold-out RMSE | 0.79534 | 0.79350 | 0.68320 | 0.78701 |
| True-long-zone hold-out RMSE | 0.66698 | 0.67908 | 0.82316 | 0.68385 |
| Whole-map hold-out RMSE | 0.75019 | 0.75348 | 0.77538 | 0.75309 |

The average per-route ratio of largest to smallest raw zone weights across decisions is 1.214 for estimated priority, 1.222 for the matched oracle, and 2.600 for the full-strength oracle. Against estimated priority, the matched oracle gains 0.286 short-zone readings [0.170, 0.412] and lowers short-zone RMSE by 0.00649 [−0.01071, −0.00251 for the signed difference]. Its whole-map RMSE difference is −0.00039 [−0.00266, +0.00191]. Against fixed 25%, the matched oracle's short-zone RMSE difference is −0.00834 [−0.01372, −0.00316], and its whole-map difference is +0.00290 [−0.00089, +0.00667]. It retains 7.4% of the full-strength oracle's short-zone RMSE improvement and removes 88.5% of its whole-map RMSE penalty, using fixed 25% as the reference for both percentages.

The matched oracle reproduces covariance-adaptive routes on all 128 uniform fields. On mixed fields, nominal 95% hold-out coverage is 94.96% for the matched oracle and 94.93% for estimated priority. Mean planning time is 1.75 seconds per mixed-field route during the eight-worker run.

### Dense evaluation on common unmeasured cells

Replay the archived fixed-25% and matched-strength oracle routes and regenerate their original sensor readings on the same 1,024 fields. For each pair, exclude the union of both policies' measured cells from the 23 × 23 grid and evaluate both maps on every remaining cell. The original 100 hold-outs remain in the expanded set. Their whole-map and short-zone RMSEs reproduce the archive for every field to within 1e-10. Predicted means and uncertainty use the same mixture mapper as before.

The primary metrics average field-level RMSEs. Distance bins use the minimum legal walking distance to a measured cell. The distance summaries pool squared errors over field/cell pairs within each bin. For each choice of distance anchor, both predictions are evaluated at the same cells. The published file includes bins anchored to each route, so their diagonal values also describe each policy at its own nearest-observation distances. Resampling uses 3,000 paired draws within layout, wall conductance, start, and range assignment.

```sh
PYTHONPATH=src python tools/audit_residential_dense_holdout.py
```

The output in `outputs/sensing-world-residential-dense-holdout` includes field metrics, distance-bin summaries, and the protocol.

| Mixed-field evaluation | Mean cells per field | Fixed RMSE | Matched oracle RMSE | Oracle minus fixed [paired 95% interval] |
| --- | ---: | ---: | ---: | ---: |
| Original hold-outs, all | 100.0 | 0.75019 | 0.75309 | +0.00290 [−0.00086, +0.00674] |
| Original hold-outs, short zones | 50.0 | 0.79534 | 0.78701 | −0.00834 [−0.01353, −0.00327] |
| Expanded common set, all | 496.8 | 0.74085 | 0.74308 | +0.00223 [−0.00131, +0.00575] |
| Expanded common set, short zones | 247.5 | 0.79816 | 0.78697 | −0.01119 [−0.01665, −0.00586] |
| Expanded common set, long zones | 249.3 | 0.66130 | 0.67978 | +0.01848 [+0.01220, +0.02496] |

| Nearest fixed-route observation | Pooled RMSE, all cells, fixed → oracle | Pooled RMSE, short-zone cells, fixed → oracle |
| --- | ---: | ---: |
| 1 step | 0.3974 → 0.4947 | 0.4675 → 0.5551 |
| 2–4 steps | 0.6292 → 0.6448 | 0.7165 → 0.7150 |
| 5–8 steps | 0.8495 → 0.8340 | 0.9258 → 0.8933 |
| 9+ steps | 0.9648 → 0.9430 | 0.9793 → 0.9454 |

Among short-zone cells nine or more steps from the fixed route, the mean distance to a measurement is 13.53 steps under fixed rollout and 12.70 under the matched oracle. Among short-zone cells one step from the fixed route, the two means are 1.00 and 1.82 steps. At each route's own nearest-observation distance, whole-map RMSEs at one step are 0.3974 and 0.3999, and at nine or more steps they are 0.9648 and 0.9648. These comparisons reflect changes in measurement placement on the common grid.

On mixed fields, empirical coverage of nominal 95% intervals is 94.91% and 95.01% on the expanded set. The matched oracle's posterior mapping takes 6.20 ms per field on average in this replay, excluding route planning and scoring. Both route archives use 52 steps on every field and average 22.13 and 21.52 readings, respectively.

Restricting the replay to weak cross-wall conductance (0.02) gives 448 mixed fields with a mean of 497.2 common evaluation cells. The oracle-minus-fixed RMSE differences are +0.00431 [+0.00052, +0.00826] across all cells, −0.00459 [−0.01097, +0.00140] in short zones, and +0.01516 [+0.00843, +0.02188] in long zones. In short zones, the paired difference is +0.08442 at one step from a fixed-route measurement, −0.02244 at five to eight steps, and −0.02870 at nine or more steps. The regime-specific tables are written to `outputs/sensing-world-residential-dense-holdout-weak`.

```sh
PYTHONPATH=src python tools/audit_residential_dense_holdout.py --regime weak
```

### Requiring one reading in each physical room

Add a coverage constraint to the fixed-covariance 25% distance-aware rollout. The movement graph, first-move shortlist sizes, variance objective, and final mixture mapper match the existing rollout. After any proposed reading, compute the exact minimum travel to obtain one reading in each remaining room. The calculation considers every order of outstanding rooms and all eligible cells in them through a subset recurrence. Check both the residual travel and reading budgets at the first move and throughout each greedy continuation. A physical-room start counts as the first reading in that room. A hallway start counts against the total reading cap.

```sh
PYTHONPATH=src python tools/audit_residential_room_quota.py
```

Both methods start at the same eight layout/start combinations and walk 52 edges on every route. The coverage constraint measures all three rooms on eight of eight routes, compared with two of eight for the existing rollout. Their respective mean reading counts are 20.75 and 22.13. Mean short-zone counts are 10.38 and 11.06 per field. Single-thread mean planning time in the paired run is 0.47 and 0.41 seconds per route. The baseline reproduces the archived paths, and its original hold-out scores are verified on every field.

On the illustrated corridor start, the existing rollout first measures the upper room at step 19 and the right room at step 39. The coverage-constrained rollout first measures the left room at step 3, the upper room at step 25, and the right room at step 45. Their reading counts for left, upper, and right are 0/10/12 and 1/10/8, respectively. Each route also makes two hallway readings. The coverage constraint leaves room order flexible. For the hallway start on the other layout, the constrained route measures the upper room first and the left room at step 52.

Compare both maps on the same 448 weak mixed fields. For the primary evaluation, exclude every cell measured by either trajectory and average the resulting field-level RMSEs. The common evaluation set averages 493.9 cells. The original 100 hold-outs are included as a sensitivity analysis. Paired 95% intervals use 3,000 resamples stratified by layout, start, and range assignment.

| Evaluation | Existing rollout RMSE | Room-quota RMSE | Quota minus existing [95% interval] |
| --- | ---: | ---: | ---: |
| Expanded common set | 0.74104 | 0.74142 | +0.00038 [−0.00850, +0.00866] |
| Expanded common set, short zones | 0.79476 | 0.80690 | +0.01214 [+0.00097, +0.02306] |
| Expanded common set, long zones | 0.66231 | 0.64028 | −0.02203 [−0.03512, −0.00917] |
| Original 100 hold-outs | 0.75478 | 0.74389 | −0.01089 [−0.02061, −0.00169] |
| Original short-zone hold-outs | 0.79645 | 0.80420 | +0.00775 [−0.00403, +0.01950] |

Nominal 95% interval coverage is 94.96% for the baseline and 94.95% for the quota. The output in `outputs/sensing-world-residential-room-quota` includes archived measurement sequences, first entries and first readings by room, field scores, and the illustrated route comparison.

### Requiring two distinct readings in each physical room

The two-reading policy allows two eligible cells in a single room visit. The route planner retains the fixed initial covariance and the existing variance objective. To certify a candidate, check both budgets and evaluate completion routes that group pending readings in each room. The check enumerates every order of the outstanding rooms and uses doorway-to-doorway costs over all eligible cells within each room. A cell already measured in a room with one remaining reading is excluded. The planner can replan after each reading and choose any new path with a certified completion. Direct enumeration on small candidate sets verifies the grouped completion costs.

```sh
PYTHONPATH=src python tools/audit_residential_room_quota.py --minimum 2
```

The comparison uses the same 448 weak mixed fields and conditions the same mapper on each route's own observations. All three policies use common evaluation cells obtained by excluding the union of their measured cells. This set averages 481.9 cells per field. To evaluate range identification, classify a physical room as short if its final posterior mean short-scale coefficient exceeds 0.5. Score against the simulator labels and average across fields and rooms. The true labels are used by the evaluator.

| Metric | Existing rollout | Minimum one | Minimum two |
| --- | ---: | ---: | ---: |
| Actual mean walking steps | 52.00 | 52.00 | 52.00 |
| Mean readings | 22.13 | 20.75 | 21.63 |
| Routes with at least two readings in every room | 0/8 | 1/8 | 8/8 |
| Common-set whole-map RMSE | 0.74431 | 0.74508 | 0.74410 |
| Common-set short-zone RMSE | 0.79750 | 0.81000 | 0.81121 |
| Final room-range accuracy | 62.57% | 61.90% | 61.98% |

Changing the minimum from one to two gives a paired whole-map RMSE difference of −0.00098 [−0.00687, +0.00460] and a short-zone RMSE difference of +0.00121 [−0.00824, +0.00997]. The range classification change is +0.07 percentage points [−2.23, +2.38]. The first two readings in a room are one walking step apart in 15 of the 24 room-route pairs. The illustrated route measures the upper room first at step 16 and the right room at step 37. It gets three lower-left readings at steps 50–52, at distances of zero, one, and two steps from that room's doorway. The previous one-reading route measured the lower-left room at step 3.

Nominal 95% interval coverage is 94.97% with the one-reading requirement and 94.99% with the two-reading requirement. One-thread mean planning times are 0.50 and 0.88 seconds per route. Protocol, route, error, and range-identification results are in `outputs/sensing-world-residential-room-quota-two`.

### Increasing the walking budget with a fixed reading cap

Compare the existing fixed-covariance rollout and the minimum-two-per-room policy at 52, 104, and 156 edges on the same two corridor layouts and four starts. The reading cap is 24 at all horizons. The original 52-edge paths are replayed from the previous audit. All runs use the same variance objective and final mixture mapper.

Use the same 448 weak mixed fields throughout. Each start's evaluation set includes cells unmeasured by all six policies. The original 100 hold-outs are retained as an additional evaluation set. Spatial coverage is measured by the longest room-side doorway-to-reading distance, the widest pair of measured cells in each room, and the distance between the first two readings in a room. Averages omit rooms where the statistic is undefined. Room-range accuracy uses the same threshold on the final posterior coefficient as the preceding experiment.

```sh
PYTHONPATH=src python tools/audit_residential_travel_budget.py
```

The output in `outputs/sensing-world-residential-travel-budget` includes route resources, paired field metrics, and a route comparison for the previously illustrated hallway start.

| Walking budget | Policy | Mean readings | Whole-map RMSE | Short-zone RMSE | Room-range accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| 52 | Existing rollout | 22.13 | 0.74348 | 0.79766 | 62.57% |
| 52 | Minimum two | 21.63 | 0.74565 | 0.81274 | 61.98% |
| 104 | Existing rollout | 23.75 | 0.66221 | 0.73530 | 63.32% |
| 104 | Minimum two | 23.75 | 0.64512 | 0.72430 | 63.76% |
| 156 | Existing rollout | 23.63 | 0.62586 | 0.70304 | 60.94% |
| 156 | Minimum two | 24.00 | 0.61177 | 0.68869 | 65.77% |

All 48 routes use their entire walking budget. The common set averages 431.4 evaluation cells. For the two-per-room policy, mean first-pair spacing is 1.88, 3.50, and 5.63 edges. Mean deepest reading by room is 5.63, 9.88, and 11.17 edges from the doorway. On the illustrated start the left room is measured to depths of 2, 8, and 10 edges. The 104-to-156-edge comparison restricted to the six starts with 24 readings at both budgets covers 336 fields. The two-per-room policy's paired changes are −0.03490 [−0.04158, −0.02825] for whole-map RMSE and −0.03959 [−0.04981, −0.02915] for short-zone RMSE. The room-range accuracy difference from 52 to 156 edges is +3.79 percentage points [+1.26, +6.25].

At 156 edges the two-per-room policy's whole-map difference relative to existing rollout is −0.01409 [−0.02015, −0.00802] on the common set and +0.00454 [−0.00277, +0.01185] on the original 100 hold-outs. Restricting the common-set comparison to the five starts where both 156-edge routes use 24 readings gives −0.00313 [−0.01017, +0.00389]. Nominal 95% interval coverage is 94.95–95.24% across the six arms. Mean planning time for 104 and 156 edges is 1.13 and 1.57 seconds for existing rollout and 1.68 and 2.12 seconds for the two-per-room variant during the two-worker run.

### Additional floor plans in the replay

The 156-step room-coverage replay also includes three rooms in a row and four rooms connected in a loop. Both comparisons use the same frozen mixture mapper, rollout score, and 24-reading cap. The constrained route requires two distinct readings in each physical room, including all four rooms in the loop. Each example uses a mixed-range field with weak cross-wall dependence and the same starting site for the two routes. The builder verifies the realized travel and reading counts and scores predictions on the 486 cells both routes leave unmeasured in each example.

| Selected replay | Whole-map RMSE, existing → minimum two | Short-zone RMSE, existing → minimum two |
| --- | ---: | ---: |
| Three rooms in a row | 0.5565 → 0.5268 | 0.6836 → 0.6185 |
| Four-room loop | 0.6020 → 0.5566 | 0.8921 → 0.7880 |

The examples were selected from paired fields where the minimum-two policy improved both scores. They show how the routes behave in these layouts. The 448-field travel-budget benchmark above covers the corridor layouts.
