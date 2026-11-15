# Generalization to unseen walls and shifted sensor fields

The [learned rollout](sensing-world-study.md) fits a mixture of spatial covariance models on simulated sensor maps. This follow-up asks how well the frozen models reconstruct a field when the robot is given a new floor plan and can measure only at the sites it chooses. All planners see the test walls and openings. They infer the field from acquired measurements.

## Train and test conditions

The original world model learns from 6,144 noisy maps on six layouts with two straight walls and one door per wall. The diverse world model also uses 6,144 noisy maps and the same architecture, with eight training layouts that vary wall positions, bends, orientation, and openings. Within each of the four covariance modes, it trains across multiple cross-wall permeabilities, ranges, covariance spectra, and occasional local sources. Geometry and sensor pattern diversity are changed together during training. The comparison measures their combined effect.

The two test factors are crossed. Standard walls use two straight walls with one door per wall. New walls use three held-out geometries with changes to wall positions, bends, orientation, or openings. Every new geometry differs from every training geometry. Benchmark fields use the original simulation family with fresh draws. Shifted fields use intermediate cross-wall permeabilities (0.25 and 0.45), a rougher covariance spectrum, and a local source whose position is outside the training source range. The paired benchmark and shifted simulations share their Gaussian innovations and sensor noise. Each of the four cells has 48 fields across three fixed layouts, four starts, and four field-mode settings.

All methods operate on a 12 × 12 grid with at most 26 walking edges and 16 readings. The starting reading counts toward the cap. GP rollout uses the frozen wall-aware Bayesian ridge configuration and replans to the end of the budget. Both learned models use the same end-of-budget rollout with 32 sampled measurement sequences per posterior component. The planner uses the known floor plan and sensor noise scale. The evaluator reserves 28 sites for latent-field RMSE and supplies a noisy value only when the planner selects its location. Intervals below use unadjusted paired bootstrap resampling within each layout and start and are conditional on the fixed layout set.

## Holdout RMSE

| Test walls | Test sensor field | GP rollout | Original world model | Diverse world model |
| --- | --- | ---: | ---: | ---: |
| Standard | Benchmark | 0.737 | 0.707 | 0.689 |
| Standard | Shifted | 0.762 | 0.740 | 0.740 |
| New | Benchmark | 0.735 | 0.709 | 0.697 |
| New | Shifted | 0.741 | 0.739 | 0.742 |

On new walls with benchmark fields, diverse training lowers RMSE by 0.038 relative to GP rollout (paired difference −0.038, 95% interval [−0.063, −0.012]). The difference relative to the original world model is −0.012 [−0.023, −0.0001]. With both shifts, the difference from GP is +0.002 [−0.015, 0.019] and the difference from the original world model is +0.004 [−0.007, 0.014]. The 22 jointly resource-matched runs in this last cell also give similar mean RMSEs (0.731 GP, 0.730 original, 0.732 diverse). This subset is descriptive because inclusion depends on how the planners used their budgets.

All episodes use exactly 26 walking edges. The respective mean reading counts on new walls with shifted fields are 15.92, 15.33, and 15.75. GP samples an average of 2.92 rooms on those fields. Both learned models sample all three rooms. Moment-based 95% next-reading intervals cover 78.5%, 92.9%, and 91.0% of realized readings at the corresponding policies' chosen sites, with average widths 1.93, 2.94, and 2.72. In the parallel evaluation, route planning takes approximately 1.14, 2.20, and 2.34 seconds per episode in this cell. These timings reflect worker contention and include planner calls only.

This evaluation uses synthetic fields and three fixed new test layouts. The diverse training and test sets retain a common spatial simulator family. Paired RMSE differences evaluate the complete closed-loop policies and capture both where they measured and how they mapped the readings.

## Reproduce

Use the frozen reference from the [wall topology study](wall-topology-study.md).

```sh
python -m spatial_exploration.sensing_world_generalization
```

The experiment writes the frozen training and evaluation settings, per-episode metrics, summary, and comparison plot under `outputs/sensing-world-generalization`. All 576 episodes were checked for paired scene identity, feasible movement, resource caps, and hold-out exclusion.

## Fixed-route diagnosis of the sensor shift

Keep the three new layouts, their holdouts, and one GP rollout route per layout and start. The GP's fixed covariance makes those routes independent of the readings. Feed all three mappers the same measurements on that route under each condition. Keep the standard-normal innovations, sensor noise, and latent correlation scale paired. Across conditions, the field changes and the sensor readings at those same locations change with it. This audit uses 128 draws per layout (384 per condition), balanced over the original four modes and four starts. The first 16 per layout reuse the prior benchmark draws. The audit checks that baseline and bundled-shift GP scores reproduce the prior scores on those 48 draws.

The factorial changes are:

- **Cross-wall dependence.** The generator's wall conductance changes from 0.02 to 0.25 for the weak regime and 1 to 0.45 for the persistent regime. The navigation graph stays fixed.
- **Roughness.** The spectral exponent decreases from 2 to 1.45 at the same nominal correlation scale and unit marginal background variance. This increases relative high-frequency content and also changes effective correlations between neighboring sites.
- **Source and background.** The original shift adds a localized covariance component near an outer edge and scales the background covariance by 0.76. Separate controls distinguish adding the localized component from that background rescaling.

Frozen parameters are loaded from the previous study. Bootstrap intervals use 6,000 paired resamples within layout, start, and simulated mode and are unadjusted. They describe these layouts, this observation route, and these shift sizes.

```sh
PYTHONPATH=src python tools/audit_sensing_world_shifts.py
```

| Fixed-route condition | GP | Original world | Diverse world | Diverse improvement over GP |
| --- | ---: | ---: | ---: | ---: |
| Baseline field | 0.7325 | 0.7036 | 0.7033 | 0.0292 |
| Wall conductance only | 0.7326 | 0.7056 | 0.7046 | 0.0280 |
| Roughness only | 0.8503 | 0.8447 | 0.8470 | 0.0033 |
| Local source only | 0.7370 | 0.7077 | 0.7072 | 0.0298 |
| Background scaling only | 0.6393 | 0.6148 | 0.6143 | 0.0250 |
| Original bundled shift | 0.7477 | 0.7445 | 0.7459 | 0.0018 |

The roughness change removes 0.0259 [0.0229, 0.0289] of the diverse model's RMSE advantage by itself. Changing conductance alone removes 0.0012 [−0.0002, 0.0026]. Adding the localized source alone changes the gap by less than 0.001. Restoring baseline roughness while retaining the other bundled changes raises the diverse model's RMSE advantage from 0.0018 to about 0.0244. These contrasts concern the specified synthetic changes on one fixed set of routes. The exponent changes effective correlation range as well as high-frequency content, and the wall conductance effects are averaged over increases and decreases in the two regimes.

### Roughness at matched effective range

On the same layouts and fixed routes, keep wall conductance, unit marginal variance, and sensor noise at baseline settings, with no localized source. Lower the graph spectral exponent from 2 to 1.45. For each layout and simulated mode, tune the scale parameter by root finding to match the baseline area under the within-room correlogram from 0 to 8 shortest-path grid steps. Average site pairs within each lag and use the trapezoidal rule. As a sensitivity check, match correlation at five steps instead. The code verifies both targets and the marginal variances. It reports the realized lag-one semivariance (one minus mean correlation at one step) to check that the short-distance pattern actually gets rougher. Matching one range statistic leaves the rest of the correlogram free to change.

```sh
PYTHONPATH=src python tools/audit_sensing_world_shifts.py --match-range
```

The audit reuses the 384 paired draws and records per-field RMSE, paired contrasts, and per-layout, per-mode correlograms under `outputs/sensing-world-range-audit`.

| Fixed-route condition | GP | Original world | Diverse world | Diverse improvement over GP |
| --- | ---: | ---: | ---: | ---: |
| Baseline field | 0.7325 | 0.7036 | 0.7033 | 0.0292 |
| Rougher at original scale parameter | 0.8503 | 0.8447 | 0.8470 | 0.0033 |
| Rougher at matched correlogram area | 0.7560 | 0.7398 | 0.7405 | 0.0155 |
| Rougher at matched five-step correlation | 0.7676 | 0.7526 | 0.7535 | 0.0140 |

At matched area, the loss in the diverse model's advantage is 0.0137 [0.0121, 0.0153]. At matched five-step correlation, it is 0.0151 [0.0135, 0.0169]. Both matches retain increased lag-one semivariance in all 12 layout-mode combinations. Matching area restores 0.0122 of the advantage relative to the rough field with the original scale parameter. The remaining loss is consistent with sensitivity to the short-distance shape of the correlation function. A scalar range match does not isolate an effect independent of all other aspects of the spatial covariance.

### Pooled versus separate roughness modes

Both world models fit the same 6,144 noisy training fields (eight training layouts × four wall/range combinations × two roughness levels × 96 draws). They use the same 13 kernel basis matrices and the same nonnegative covariance fitting procedure. The four-mode model concatenates the two roughness groups by wall/range combination before fitting. The eight-mode model fits them separately. Both use a uniform prior over their own modes. The diagnostic with known roughness restricts the eight-mode model to the corresponding four modes before conditioning on readings, leaving wall dependence and correlation range to be inferred.

Training and test generators use spectral exponents 2 and 1.45. Within each layout × mode the rough field's scale is adjusted to match the smooth field's within-room correlogram area. Evaluation uses three held-out layouts, new random seeds, 384 smooth/rough field pairs, and paired field innovations and noise. All arms receive the same readings on the frozen GP rollout route. The route uses 26 walking edges and 15 or 16 readings per field. RMSE uses the same 28 held-out sites. This comparison uses a newly matched training set, so the four-mode model has newly fitted weights.

| Condition | GP | Four modes | Eight modes | Eight modes, known roughness |
| --- | ---: | ---: | ---: | ---: |
| Smooth | 0.72231 | 0.70295 | 0.70291 | 0.70211 |
| Rough | 0.74872 | 0.73780 | 0.73773 | 0.73677 |

For rough fields, eight modes minus four modes is −0.000063 (paired 95% interval [−0.000355, +0.000226]). Revealing the group reduces eight-mode RMSE by 0.000964 [0.000190, 0.001738]. The highest marginal roughness probability identifies the group on 56.5% of smooth fields and 65.6% of rough fields. Mean posterior rough-group mass is 0.475 and 0.544, respectively. The fitted rough components increase mean within-room lag-one semivariance by 0.047 compared with 0.051 for the generating process, averaged across matched layout and mode pairs. For this basis, regime separation changes the fitted local covariance, while revealing the group yields a small prediction gain on these routes. This is a fixed-route comparison and does not evaluate whether a roughness-aware model would choose better locations.

```sh
PYTHONPATH=src python tools/audit_sensing_world_modes.py
```

The audit writes frozen weights, per-field metrics, achieved ranges, and paired summaries to `outputs/sensing-world-modes`.

### Frequency-dependent regularization

Use the same 6,144 training fields and exact 384 smooth/rough test pairs from the pooled-versus-split comparison. The test geometry, GP-selected 15–16 observations, sensor values, and 28 holdout sites are shared by all models. The paired smooth and rough simulator fields still match within-room correlogram area. GP and the four-mode heat mixture reproduce their preceding per-field RMSEs.

For graph Laplacian eigenvalue `mu` and cutoff `c = 1`, the new covariance weights each eigenvector by the reciprocal of a generalized ridge penalty. The spectral variance at scale `k` is

```text
(1 + min(mu, c) / k²)^(-a_low) * (1 + max(mu - c, 0) / (k² + c))^(-a_high)
```

Both spectral arms keep `a_low = 2`. The uniform-decay control also sets `a_high = 2`. The split-decay arm selects `a_high` for each of four modes by minimizing mean covariance squared error against the shared training groups. Its selected values are 1.25, 1.75, 1.50 and 1.75 in the frozen mode order. Both arms retain the heat model's mixture mass on each graph conductance and the independent latent term. On each layout and mode, solve for the scale that matches the heat covariance's within-room correlogram area from 0 to 8 walking steps. The model uses its trained heat prior and the known layout to compute that target. Both spectral arms match the heat prior's marginal variances and area, with area errors below 1e-9. For the split-decay arm, mean lag-one semivariance rises by 0.0138 relative to heat across the 12 layout-mode combinations. The match controls a scalar summary of range.

| Evaluation field | GP | Heat world model | Spectral, uniform decay | Spectral, split decay |
| --- | ---: | ---: | ---: | ---: |
| Smooth | 0.72231 | 0.70295 | 0.70215 | 0.70456 |
| Rough | 0.74872 | 0.73780 | 0.73906 | 0.73697 |

On rough fields, split minus heat is −0.00082 [−0.00192, +0.00028], split minus GP is −0.01175 [−0.01678, −0.00680], and split minus uniform decay is −0.00209 [−0.00334, −0.00089]. On smooth fields, split minus heat is +0.00162 [+0.00059, +0.00267]. Intervals are unadjusted paired 95% intervals with 6,000 bootstrap replicates stratified by layout, start, and mode. There is evidence of a rough-field improvement from splitting decay within this spectral family. The comparison with heat is inconclusive, and heat gives lower error on the smooth fields. The modes and regularization are frozen before evaluation. The results measure prediction at fixed observations.

For rough fields, moment-based 95% latent pointwise intervals cover 80.4% of holdouts for GP, 94.2% for heat, 94.0% for uniform decay, and 94.3% for split decay. Average widths are 1.92, 2.88, 2.87, and 2.88 sensor units. Batched posterior scoring costs roughly 0.01 ms per field for GP and 0.03 ms for each mixture on one CPU thread, excluding layout, prior construction, and route planning.

```sh
PYTHONPATH=src python tools/audit_sensing_world_spectral.py
```

Run the pooled-versus-split audit first to generate the frozen comparison. The spectral audit saves training choices, per-field metrics, matched-range diagnostics and paired summaries in `outputs/sensing-world-spectral`.

### Effective range sweep

Fix the graph spectral exponent at 2 and the marginal field variance at one. On each of the three unseen layouts, vary the scale parameter to reach within-room 0–8-step correlogram areas of 1.5, 2.5, 3.5, 4.5, and 5.5. Wall conductance is fixed within each regime (0.02 or 1). Use 768 test fields per area, paired across areas through their innovations and noise. The four-mode heat world model and GP settings are frozen. Both receive identical readings from the same GP route (26 edges, 15–16 readings) and are scored on the same 28 held-out sites.

| Correlogram area | GP RMSE | World model RMSE | GP minus world |
| --- | ---: | ---: | ---: |
| 1.5 | 0.9443 | 0.9466 | −0.0023 |
| 2.5 | 0.8249 | 0.8223 | +0.0026 |
| 3.5 | 0.7267 | 0.7105 | +0.0161 |
| 4.5 | 0.6394 | 0.5996 | +0.0398 |
| 5.5 | 0.5553 | 0.4913 | +0.0640 |

The improvement grows with correlogram area in both wall regimes. At the shortest setting the GP minus world difference has a paired 95% interval of [−0.0044, −0.0003]. At the longest setting the interval is [0.0593, 0.0687]. Intervals are unadjusted, with 6,000 paired resamples stratified by layout, wall regime and start. The exponent is fixed throughout, but varying scale also changes adjacent-site correlation. These results quantify prediction on fixed routes, and do not measure how the rollout responds to a change in range.

```sh
PYTHONPATH=src python tools/audit_sensing_world_ranges.py
```

The [residential floor-plan study](residential-floor-plan-study.md) compares fixed-route mapping and closed-loop sensing on corridor and loop layouts at two effective ranges.
