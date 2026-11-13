# Per-map DeepKriging-style mapper study

## Question and scope

Does a multiresolution spatial embedding improve prediction over a coordinate-input neural network with a similar parameter budget, and does either beat the fitted stationary GP as observations increase? Reuse the exact archived observations from the [sample-size benchmark](sample-size-study.md). This is a mapper comparison with no route, movement budget, or policy retraining.

The benchmark has already been inspected during development. Independently generated tuning and calibration maps prevent direct configuration selection on its outcomes, but do not turn it into a fresh final test set. A subsequent confirmatory experiment must freeze the complete design before generating new test maps.

## Method audit

A later [controlled paper-function reimplementation](paper-mapping-study.md) separately tests the original 2D function and source-based network settings at n=810/128/32/16. It documents discrepancies between the supplement and public code, removes cross-fold weight reuse, and includes both exponential and RBF GP controls. It does not replace or retroactively tune the adapted results below.

The [DeepKriging paper](https://www3.stat.sinica.edu.tw/statistica/oldpdf/A34n113.pdf) motivates multiresolution compact spatial basis features as inputs to a neural predictor. The [author's implementation](https://github.com/aleksada/DeepKriging/blob/master/PM25_application.py) supplies a concrete Wendland-feature and neural-network reference. We implement that core representation in PyTorch and document our small-data adaptation; this is **not a full paper reproduction**.

For a knot u and normalized distance r = ||s-u||/theta, our feature is

`phi(r) = (1-r)^6 * (35*r^2 + 18*r + 3)/3` for `0 <= r <= 1`, zero otherwise.

Square knot grids of 4, 7, and 13 per axis give 234 basis features. Add x and y for 236 inputs. The support radius is 2.5 times the knot spacing, `2.5/(m-1)`. All features are standardized using the known grid geometry alone. The author's application uses larger knot grids, three hidden layers, dropout, batch normalization, and a separate density-estimation procedure for uncertainty. Our smaller two-hidden-layer networks omit dropout and batch normalization and use an initialization ensemble for uncertainty. These changes matter for any interpretation of failure or success.

## Fitting and capacity control

- DeepKriging-style network: 236 inputs, two 32-unit ReLU layers, scalar output; 8,673 parameters per member.
- Coordinate control: two inputs, two 91-unit ReLU layers, scalar output; 8,737 parameters per member, within 1% of the spatial model.
- Each prediction averages three independently initialized members. Both models have the same ensemble size, optimizer, candidate training settings, and observation sets.
- Every map and observation count receives a fresh fit. No network weights are transferred between maps or reused from a larger observation set. The noisy measurements on that map are the only fitting targets.
- Use Adam, learning rate 0.003, full observed-data MSE, and candidate weight decays 0.0001/0.01 and epoch counts 100/300. The optimizer's weight decay penalizes weights and biases.
- Batched matrix operations accelerate independent fits; there is no weight sharing, cross-map normalization, or loss averaging that changes individual gradients. Tests compare vectorized and separate-network gradients.

The fitted GP retains its previously recorded results and assumptions: known zero mean and sensor noise, a fitted single-RBF length scale and variance, and plug-in conditional uncertainty. Neural models do not receive the true covariance or explicitly use the known noise variance in their MSE loss. Parameter matching controls size, but does not match inductive bias, width, tuning cost, or optimization difficulty.

## Independent selection, calibration, and evaluation

The domain, three field regimes, nested observation counts 8/16/32/64/128, and fixed 45 evaluation sites are unchanged. There are 64 benchmark fields per regime. Regimes share underlying random draws, so they are paired conditions, not 192 independent field replications.

1. Generate eight new tuning fields per regime (field seeds start at 1,800,000, design seeds at 1,810,000). Select one epoch/weight-decay pair per model and observation count by latent evaluation-site MSE pooled across all three regimes. This is four candidates per model/count and 24 tuning maps; the choice is not specialized to a field or regime.
2. Generate eight other calibration fields per regime (field seeds start at 1,900,000, design seeds at 1,910,000). Estimate one variance multiplier per model/count, pooled across regimes. No weights or training settings are selected on these maps.
3. Write `frozen_protocol.json` with all selected settings and interval multipliers **before loading the archived benchmark targets or making benchmark predictions**. An end-to-end test enforces this ordering. Hashes verify that the source observation and evaluator archives are unchanged.
4. Fit the neural models to the archived measurements and evaluate their latent predictions at the same withheld sites as the GP. Archive individual-member predictions, weights, model selection, interval calibration, and paired comparisons.

Initialization seeds are disjoint between tuning, calibration, and benchmark fits. Within each split, candidates reuse the same independent member seeds. The training API accepts observed indices/values and geometry, not hidden field targets. Separate development-map evaluator values are used only for hyperparameter selection and interval scaling. Benchmark evaluator values are used only for scoring and rendering.

Bootstrap intervals resample the 64 fields within each regime, not spatial sites. Comparisons use paired field RMSE differences. The five observation counts share data prefixes and are not independent experiments. Three ensemble members characterize initialization sensitivity only; this is not a broad study of training variability.

## Uncertainty interpretation

At each site, raw variance is the sample variance of three member predictions, floored at 1e-6. It captures some initialization sensitivity and is **not a GP posterior variance**. All members see the same data; spread can collapse or remain large for reasons unrelated to actual error. No observation-noise variance is added, since the evaluation target is the latent field.

On independent calibration fields compute `q = quantile_0.95(abs(truth-mean)/raw_std)` over evaluation sites and use variance multiplier `(q/1.96)^2`. Report both raw coverage and coverage after this fixed scaling. This pooled heuristic has no conformal or posterior guarantee; spatial sites and regimes are dependent, and overall coverage can hide regional failures. Wider calibrated intervals do not improve mean predictions.

Marginal ensemble bands do not provide the joint conditional covariance or expected variance reduction after a hypothetical new measurement. The current GP information-gain planner therefore cannot be connected to this mapper merely by replacing its displayed standard deviations.

## Results

Recorded on November 4, 2026, with Python 3.12.15, NumPy 2.5.4, and PyTorch 2.15.0. All ten selected model/count configurations use 300 epochs. Coordinate DNN selects weight decay 0.01 at n=8–64 and 0.0001 at n=128; the spatial model selects 0.01 at n=8 and 0.0001 thereafter. No settings were changed after benchmark evaluation.

Mean held-out latent RMSE on the nonstationary regime (64 paired fields; lower is better):

| Observations | Oracle GP | Fitted RBF GP | Coordinate DNN | DeepKriging-style |
| --- | ---: | ---: | ---: | ---: |
| 8 | 0.7751 | 0.8353 | 1.1659 | 0.8585 |
| 16 | 0.6526 | 0.7182 | 0.8872 | 0.7776 |
| 32 | 0.5037 | 0.5491 | 0.6725 | 0.6552 |
| 64 | 0.3479 | 0.3821 | 0.5208 | 0.5007 |
| 128 | 0.2059 | 0.2245 | 0.4601 | 0.3520 |

At n=128 the spatial model reduces mean RMSE by 23.5% relative to the coordinate model, but has 56.8% higher mean RMSE than the fitted GP. Its paired difference against the coordinate model is -0.1081, field-bootstrap 95% interval [-0.1359, -0.0810], improving 51/64 fields. Against the fitted GP its paired difference is +0.1275, interval [0.1078, 0.1476], improving only 3/64 fields. At n=32/64 the mean improvement over coordinates has an interval crossing zero; do not claim a clear advantage at every observation count. The n=8 spatial-versus-GP interval also crosses zero.

The n=128 comparison across regimes:

| Regime | Fitted GP | Coordinate DNN | DeepKriging-style |
| --- | ---: | ---: | ---: |
| Stationary RBF | 0.1067 | 0.2281 | 0.2824 |
| Stationary two-scale | 0.2291 | 0.4740 | 0.3597 |
| Nonstationary two-scale | 0.2245 | 0.4601 | 0.3520 |

Both neural models have higher mean RMSE than the fitted GP in every regime/count cell. Spatial features help over coordinates in the two-scale cases, but hurt in the stationary RBF case at n=32/64/128. Representing nonstationarity is a capability, not a guarantee of better finite-data predictions.

### Uncertainty and fitting diagnostics

Nonstationary n=128:

| Mapper | Raw 95% coverage | Coverage after neural calibration | Mean reported latent std after calibration |
| --- | ---: | ---: | ---: |
| Fitted GP | 94.13% | 94.13% (unchanged) | 0.1950 |
| Coordinate DNN | 36.56% | 92.40% | 0.6888 |
| DeepKriging-style | 73.68% | 93.06% | 0.5456 |

The spatial model's n=128 variance multiplier is 6.36 (standard-deviation multiplier about 2.52); at n=8 it is about 2,465. Raw initialization spread is particularly unreliable under heavy regularization. Coverage approaching 95% after calibration does not make the models equally informative: spatial-model intervals are about 2.8 times as wide as GP intervals on average at n=128. Its bottom/middle/top coverage is 95.83%/92.11%/91.93%, so pooled calibration does not remove regional imbalance. The independent calibration set has only eight independent fields per regime, which limits precision.

Across all regimes at n=128, mean RMSE against the **observed noisy training targets** is 0.000254 for the spatial ensemble versus 0.1495 for the coordinate ensemble. The spatial model nearly interpolates measurements despite sensor noise std 0.1. This is a useful warning about interpolation and generalization, not evidence of accurate latent-field recovery. We have not isolated how much of the gap comes from representation, regularization, optimization, or the field family.

### Compute

Actual summed fit times are 24.89 seconds for tuning, 11.36 seconds for calibration, and 95.27 seconds for benchmark neural fits under the timing scope below. At n=128, amortized batched fit-plus-grid-prediction time per map is about 104 ms for the coordinate ensemble and 56 ms for the spatial ensemble. These are throughput figures, not online update latency.

The separate predefined single-map audit records 136 ms for the coordinate ensemble, 118 ms for the spatial ensemble, and 41 ms for the fitted GP. Timings are descriptive and include the scope differences below; they do not establish a general speed ratio. A sequential explorer would also need to pay repeated refitting or implement and validate an update strategy.

### Interpretation and next experiment

This run supports a limited conclusion: spatial embedding helps this neural predictor on two-scale fields, but the adapted neural mapper does not close the fitted-GP gap, and its raw ensemble uncertainty is unsuitable as a drop-in planning posterior. The fields are still Gaussian-process draws; even the misspecified stationary GP has a useful inductive bias. This is not evidence that all nonstationary problems favor GPs, nor a verdict on the published DeepKriging method.

The search is deliberately small: knot grids, support, widths, optimizer, and ensemble size were fixed; only epochs and weight decay were selected. All winners reach the maximum candidate epoch count, so the coordinate network may benefit from further optimization. The spatial network's near-exact interpolation means simply adding epochs is not an established remedy. Do not change these settings against the same benchmark and call the outcome a fresh test.

A useful next diagnostic is a regularized linear predictor on the **same spatial basis**, tuned on separate maps, to ask whether nonlinearity adds value beyond the basis itself. Add an estimated nonstationary GP for a stronger modeling comparison before attributing any eventual gain specifically to avoiding spatial covariance parameters. Freeze candidates and use new independent test fields before making final claims. Closed-loop integration should wait for a credible measurement-update and uncertainty design.

## Reproduction and artifacts

```sh
.venv/bin/python -m pip install -e '.[neural]'
# Generate the reference first if it is not already archived.
PYTHONPATH=src .venv/bin/python -m spatial_exploration.sample_size_study --output outputs/sample-size-study
PYTHONPATH=src .venv/bin/python -m spatial_exploration.deepkriging_study --reference outputs/sample-size-study --output outputs/deepkriging-study
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

Use a fresh output directory for each neural run. Configurations, library versions, source hashes, selection/calibration observations, evaluator-only archives, feature transforms, frozen settings, and all benchmark checkpoints are saved. The benchmark contains 1,920 neural map fits and 5,760 independent member fits; combined GP/neural metrics have 4,800 rows.

`compute.json` records the actual tuning, calibration, and benchmark fit totals without double counting shared epoch checkpoints. Timings include initialization, optimization, and grid prediction, excluding dataset generation and checkpoint I/O. All numerical libraries use one CPU thread in the CLI. Batched amortized costs are explicitly separated from a predefined single-map timing audit (nonstationary field 0 at n=128); the latter is one descriptive run, not a stable latency benchmark. GP archival update timings and neural fit-plus-prediction timings have slightly different scopes, so precise speedup claims are inappropriate.

Main files: `metrics.csv`, `regional_metrics.csv`, `member_metrics.csv`, `fit_diagnostics.csv`, `summary.json`, `paired_comparisons.json`, `comparison.png`, `uncertainty.png`, six predefined field-0 examples, `predictions.npz`, `single_map_timing.csv`, and `diagnostics.json`. Source observations remain in the reference directory. All 55 tests pass in the optional-dependency environment; the system-Python run passes 33 and skips 22 optional tests. Tests cover basis values/support, parameter matching, independence of batched gradients, learning and checkpoint reloads, calibration scaling, and the full freeze-before-evaluation workflow.
