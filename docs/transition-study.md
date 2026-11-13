# Dependence-transition sharpness study

## Controlled question

Does making a spatial dependence transition more abrupt change the relative accuracy of a fitted stationary GP, coordinate DNN, and DeepKriging-style network? Change the transition only, retaining long/short RBF length scales 0.35/0.12, zero mean, unit marginal variance, noise std 0.1, the 15-by-15 grid, and observation counts 8/16/32/64/128.

This study does not represent an impermeable wall, an obstacle-constrained route, or a terrain dataset. It is a controlled continuation of the [neural mapper comparison](deepkriging-study.md). Real barrier applications require an additional treatment of connectivity; see [application evidence and the proposed barrier experiment](barrier-applications.md).

## Generating conditions

Generate independent unit-variance stationary components `Z_long` and `Z_short`, then

`Y(s) = cos(pi*q(y)/2)*Z_long(s) + sin(pi*q(y)/2)*Z_short(s)`.

| Condition | q(y) | Width from q=0.1 to q=0.9 |
| --- | --- | ---: |
| Gradual | y | 0.8 |
| Moderate | logistic((y-0.5)/0.10) | 0.4394 |
| Sharp | logistic((y-0.5)/0.05) | 0.2197 |

The weights satisfy `a(y)^2+b(y)^2=1`. Generating covariance is `a(s)a(t)K_long(s,t)+b(s)b(t)K_short(s,t)`, preserving positive semidefiniteness and constant marginal variance. Greater sharpness concentrates changes around y=0.5; it is not a formal ordering of every possible measure of nonstationarity. It also changes how much of the domain is almost purely long/short scale, so it need not increase integrated prediction difficulty monotonically.

On this grid the narrowest 10–90% transition spans about 3.08 grid spacings. We deliberately stop there rather than claim to resolve a discontinuity. Even a very sharp continuous switch is different from a physical wall with a door or an impermeable boundary.

## Pairing and frozen evaluation

For every independent field, reuse the same **long and short component realizations**, sensor-noise realization, and observation permutation across the three conditions. Their values differ only through the spatial mixing weights. This explicit component pairing differs from the earlier benchmark's common Cholesky-normal-vector pairing; comparisons against that old run are not paired or directly attributable to sharpness.

The same 45 sites (index mod 5 equals 4) remain evaluation-only; all models receive identical acquired values within each condition/field/count. Observation prefixes are nested. No model receives hidden benchmark field values. Oracle GP is explicitly privileged with the true generating covariance, while the fitted GP estimates a single stationary RBF from acquired data, with known zero mean/noise and the existing fitting bounds/restarts.

- Tuning: eight independent fields per condition, field/design seeds 2,300,000/2,310,000.
- Calibration: eight other fields per condition, seeds 2,400,000/2,410,000.
- Benchmark: 64 new paired fields per condition, seeds 2,500,000/2,600,000.

Both neural models retain their previously documented geometry, architectures, parameter counts (8,737 coordinate; 8,673 spatial), three initialization members, optimizer, and four epoch/weight-decay candidates. Select one setting per model/count pooled over all three tuning conditions. Calibrate one marginal interval scale per model/count pooled over all three independent calibration conditions. Thus the configuration is held fixed across transition levels. Member seed bases are 2,320,000/2,420,000/2,520,000 for tuning/calibration/benchmark, respectively.

Write `frozen_protocol.json` before **generating** benchmark data. A workflow test enforces this ordering. This is a new frozen synthetic benchmark for the stated comparison, not validation of a deployed real-world application. Do not retune against its outcomes and continue calling it an untouched test set.

Metrics include latent held-out RMSE/MSE, raw and calibrated coverage, interval width, acquired-target interpolation error, individual member accuracy, and compute. Save regional metrics for fixed bands y<0.35, 0.35<=y<=0.65, y>0.65 across all conditions. Paired field-bootstrap intervals compare models within conditions and changes relative to gradual; they resample fields, not sites. Conditions and sample sizes are dependent comparisons, and no multiplicity-adjusted significance claim is made.

Initialization ensembles and pooled interval scaling remain heuristic; they do not supply a joint conditional covariance for the exploration planner. GP and neural timing scopes differ (conditioning versus fit plus grid prediction), and neural batched amortized time is throughput rather than per-map latency.

## Recorded results

Recorded November 4, 2026. All five coordinate configurations select 300 epochs (weight decay 0.01 at n=8/16/32 and 0.0001 at n=64/128). All five spatial configurations select 100 epochs and weight decay 0.0001. This differs from the earlier study's selection because the development fields and conditions differ. Selection is shared across all three transition levels in this run, and no settings were changed after benchmarking.

Mean held-out latent RMSE at n=128, across 64 paired fields:

| Transition | Oracle GP | Fitted stationary GP | Coordinate DNN | DeepKriging-style |
| --- | ---: | ---: | ---: | ---: |
| Gradual | 0.1986 | 0.2147 | 0.4600 | 0.3373 |
| Moderate | 0.1909 | 0.2127 | 0.4632 | 0.3390 |
| Sharp | 0.1885 | 0.2152 | 0.4604 | 0.3417 |

The fitted GP has the lowest non-oracle mean RMSE at every tested condition/count. At n=128, the spatial model remains better than the coordinate model and worse than fitted GP in all three conditions. In the sharp condition its paired RMSE difference is +0.1266 versus fitted GP (bootstrap 95% interval [0.1102, 0.1419]; improves 2/64 fields), and -0.1187 versus coordinates (interval [-0.1486, -0.0895]; improves 53/64).

Changing gradual to sharp at n=128 produces a fitted-GP paired RMSE change of +0.00041, interval [-0.00450, 0.00553], and a spatial-network change of +0.00446, interval [-0.00750, 0.01668]. These comparisons do not show a clear whole-map accuracy change. A sharper transition did not expose a neural advantage.

### Why sharper did not simply mean harder

Oracle RMSE actually decreases by 0.01013, paired interval [-0.01461, -0.00601], from gradual to sharp. This is consistent with the larger almost-pure long-scale region becoming easier to interpolate: oracle lower-band mean RMSE decreases from 0.1204 to 0.0728, while upper-band RMSE stays similar (0.2402 to 0.2438). These observations support a mechanism interpretation; we did not independently randomize transition width and the area occupied by nearly pure components.

An additional **post-hoc descriptive diagnostic** separates absolute RMSE from excess above the condition-specific oracle. Fitted-GP mean excess is 0.01613 in gradual and 0.02666 in sharp; the paired excess increase is 0.01053, interval [0.00645, 0.01469]. Equivalently, excess of mean RMSE over oracle increases from about 8.1% to 14.1%. Spatial-network excess also increases, from 0.13866 to 0.15324. This indicates a larger gap to the matched-model reference, despite little change in absolute non-oracle RMSE. It does not rescue the neural model's ranking or establish a general monotonic law of nonstationarity.

The prespecified boundary band also needs cautious interpretation. Fitted-GP boundary mean RMSE changes from 0.2075 to 0.2165, but the paired change interval [-0.00081, 0.01899] crosses zero. Lower/boundary/upper fitted-GP coverage in the sharp condition is 99.27%/91.98%/91.46%; whole-map coverage of 94.24% hides regional imbalance.

### Uncertainty and compute

At n=128 in the sharp condition:

| Mapper | Raw 95% coverage | Coverage after neural calibration | Mean std after calibration |
| --- | ---: | ---: | ---: |
| Fitted GP | 94.24% | 94.24% (unchanged) | 0.2053 |
| Coordinate DNN | 43.06% | 95.42% | 0.8526 |
| DeepKriging-style | 76.42% | 93.06% | 0.5032 |

The neural intervals remain broad and require substantial calibration. Spatial-network intervals are about 2.45 times GP width on average in this condition; matching nominal coverage alone would not establish comparable uncertainty quality. The experiment still supplies no neural posterior suitable for hypothetical-measurement information gain.

Recorded summed fit times are 24.90 seconds for tuning, 8.49 seconds for calibration, 72.69 seconds for benchmark neural ensembles, and 24.47 seconds for both benchmark GP treatments. These are the scoped timings described above, not total experiment wall time or per-map deployment latency. Benchmark outputs contain 3,840 mapper evaluations and 5,760 neural member fits. All 59 tests pass with optional dependencies; system Python passes 36 and skips 23 optional tests. Checks cover valid covariances, constant variance, sharper cross-boundary correlation changes, shared noise/design/component values, freeze-before-generation ordering, and common mapper observations.

## Consequence for the application

Do not keep making this synthetic transition harsher until a neural model wins. Retain the result: this representation does not outperform the fitted GP under these data and tuning resources. A room/door or water-network experiment should be justified by the physical task, with transport geometry supplied fairly to both model classes, rather than introduced only to disadvantage the existing GP. The [barrier application proposal](barrier-applications.md) records the next hypothesis without claiming it has been implemented or validated.

## Reproduction

```sh
PYTHONPATH=src .venv/bin/python -m spatial_exploration.transition_study --output outputs/transition-study
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

Use a fresh output folder. The archive contains the prespecified protocol, feature transforms, tuning/calibration/benchmark observations separated from evaluator-only data, frozen settings, all neural benchmark checkpoints and predictions, metrics, regional and member diagnostics, model/transition paired comparisons, compute, hashes, and figures. Existing benchmark archives are preserved.
