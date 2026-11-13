# Sample size before DeepKriging

## Question and scope

How much of the fitted-GP mapping error and undercoverage is associated with having very few observations, and how much remains when a stationary single-scale RBF covariance is structurally misspecified? Increase acquired observations under a fixed spatial design before introducing a neural mapper.

This is a **same-observation mapper benchmark**, without travel planning or a physical travel budget. It does not retrain or evaluate the frozen 7-by-7 neural policies, establish that their robustness failure is repaired, or compare planning efficiency. All observations count toward the stated sample size; there is no extra free starting measurement.

## Prespecified design

- Domain: a fixed 15-by-15 grid in the unit square, 225 sites.
- Withhold indices `index % 5 == 4` throughout: 45 evaluation sites, with 180 permitted observation sites.
- Sample sizes: 8, 16, 32, 64, and 128 acquired noisy measurements.
- Replicates: 64 independent field/noise draws within each scenario. Field seeds start at 1500000.
- For each field, use a uniform random permutation of permitted sites, design seed starting at 1600000. Take nested prefixes so every smaller observation set is contained in every larger set.
- Share the exact indices and values across all mappers at each sample size. Observation selection is independent of measured values and true covariance.
- Share observation orders and underlying standard-normal field/noise draws across scenarios as common random numbers. Scenarios are paired, not 192 mutually independent field replicates.
- All latent fields have zero mean and unit pointwise variance. Sensor noise standard deviation is 0.1, known to every mapper.
- Factors, generator parameters, sample sizes, fit settings, and seeds are recorded before fitting or viewing results. No parameters are changed to obtain a favorable nonstationary result.

The domain and grid are held fixed as sampling density increases. This is not an expanding-domain experiment, and denser observations on one realization are not independent field replicates. Do not assume separate covariance parameter errors must decrease monotonically.

## Three generating scenarios

1. **Stationary RBF:** a single unit-variance RBF covariance with length scale 0.23. This is correctly specified for the fitted model family.
2. **Stationary two-scale control:** equal covariance weights on independent unit-variance RBF components with lengths 0.35 and 0.12. Its covariance is `0.5 * K_long + 0.5 * K_short`. It is stationary but outside the single-RBF model family; this checks whether a gap can also arise from mixing scales without nonstationarity.
3. **Spatially varying two-scale GP:** `Y(s) = a(s) Z_long(s) + b(s) Z_short(s)`, with independent components as above, `a(s) = cos(pi*y/2)`, and `b(s) = sin(pi*y/2)`. Its covariance is `a(s)*a(t)*K_long(s,t) + b(s)*b(t)*K_short(s,t)`. The bottom is smooth and the top varies more rapidly. Since `a(s)^2 + b(s)^2 = 1`, pointwise variance remains 1 and the mean remains 0.

The nonstationary construction changes dependence rather than introducing a spatial mean trend or a variance trend. It is positive semidefinite by construction as a sum of diagonally weighted valid covariance matrices. Cholesky generation uses jitter 1e-10. Vary along y because the existing held-out pattern forms x-columns; all y-regimes then have the same held-out-column coverage. Border prediction remains part of this fixed design for every model.

The two-scale stationary control is important for interpretation: outperforming a single-RBF model on the nonstationary case would not by itself attribute a future DeepKriging gain specifically to nonstationarity. Multi-scale covariance-family mismatch can also matter.

## Three mappers and fitting protocol

- **Oracle GP:** condition the actual generating covariance on the supplied noisy measurements. In the mixture scenarios, this is the true mixture covariance, not a single RBF with a privileged scalar length. It knows covariance, not the latent realization.
- **Fixed initial GP:** use stationary RBF length 0.35 and process variance 1.5, matching the earlier fixed-initial baseline.
- **Estimated GP:** fit a stationary RBF length and process variance by maximum marginal likelihood, knowing the zero mean and sensor noise. Use the existing estimator settings: initial 0.35/1.5, bounds [0.05, 1] and [0.05, 5], initial-point optimization plus two randomized restarts.

At **each requested sample size**, start the estimated model afresh from the stated initial parameters and perform exactly one batch snapshot fit using that observation prefix. This deliberately avoids the previous online `4, 7, 10, ...` refit schedule and avoids inheriting the smaller-sample fitted parameters. The added `from_observations` API does not change that original online schedule. Record initial and optimized likelihood, accepted/rejected status, bounds, warnings, and optimization time for every snapshot.

In the correctly specified single-RBF scenario, report absolute errors for length 0.23 and process variance 1. For mixture scenarios there is no unique true global RBF length, so its parameter error is left undefined. An estimated scalar length there is a model-dependent compromise. The reported amplitude difference from 1 compares with the true pointwise variance; it is not a claim that a misspecified model's likelihood-optimal parameter must equal 1.

Fresh mapper construction, covariance fitting if applicable, and conditioning on all n measurements are included in each runtime. Earlier sample-size fits, metrics, bootstrap calculations, and rendering are excluded. Use one numerical-library thread. These are descriptive local timings, not a controlled hardware benchmark.

## Metrics and uncertainty

Evaluate latent-field RMSE, MSE, mean posterior latent variance, and coverage of mean plus/minus 1.96 posterior standard deviations on the **same 45 withheld sites** at every sample size. Estimated-GP intervals condition on fitted parameters and omit their uncertainty. Also record prediction and coverage metrics separately in bottom, middle, and top y-regions.

Bootstrap 64 complete fields with 5,000 resamples and seed 1700000 for mean RMSE, mean coverage, and paired excess error relative to oracle. Reuse field-resampling indices for paired comparisons. Grid sites, sample sizes, and models are repeated measurements, not independent replicates. These intervals describe the fixed generator, design, and fitting setup, not uncertainty over all possible spatial processes or optimizer configurations.

The oracle's held-out posterior variance must be nonincreasing for each field's nested designs. Realized RMSE, even for the oracle, need not decrease at every step on every field. A residual gap to oracle is evidence of remaining limitation in this model/fit setup; it alone does not identify which part is structural versus finite-sample or optimization error. The correctly specified and stationary-mixture controls, fit diagnostics, calibration, and regional errors help interpret that gap.

## Data isolation and future neural comparison

`observations.npz` contains points, scenario names, sample sizes, field IDs, nested observation indices, and the **observed values along the maximum prefix**. Each mapper receives only the first n indices and values. It does not receive unacquired sensor values, evaluation targets, or the true covariance, except for the explicitly privileged oracle.

`evaluation_only.npz` separately contains full latent fields, evaluation masks, and true covariance matrices. These support oracle/evaluation/figures and must never enter a deployed mapper or a future neural model's training features. Store SHA-256 fingerprints for both files. The archive makes the exact observation sets reusable for the planned coordinate-DNN and DeepKriging development comparisons.

This is an exposed development benchmark once analyzed. Choose future neural architectures and hyperparameters with separate training/validation fields, and reserve a fresh test set for final claims. No DeepKriging model is implemented or validated by this GP-only study.

## Validation and reproduction

Tests check unit diagonal and positive semidefiniteness, location-varying dependence at equal displacement, unchanged stationary dependence, dataset reproducibility, fixed holdouts, common/nested observation prefixes, isolation of evaluator truth, one-fit snapshot behavior, agreement with independent batch GP conditioning, input ownership, and invalid observation/sample-size rejection. The small end-to-end workflow verifies all three models receive exactly the same data and unknown models never receive an oracle covariance argument.

## Results — November 4, 2026

All 2,880 mapper evaluations complete: three scenarios, 64 fields per scenario, five sample sizes, and three mappers. There are 960 snapshot optimizations, 23 boundary-hit events, and no rejected fits. Every oracle held-out variance curve is nonincreasing for its nested design. All 49 tests pass in the full environment; system Python passes 33 and skips 16 optional RL/imitation tests.

Mean held-out latent RMSE; each cell lists **oracle / fitted stationary RBF** on exactly the same measurements:

| Measurements | Stationary RBF | Stationary two-scale control | Nonstationary two-scale GP |
| ---: | ---: | ---: | ---: |
| 8 | 0.6897 / 0.7602 | 0.7907 / 0.8526 | 0.7751 / 0.8353 |
| 16 | 0.4754 / 0.5118 | 0.6830 / 0.7194 | 0.6526 / 0.7182 |
| 32 | 0.3028 / 0.3085 | 0.5191 / 0.5412 | 0.5037 / 0.5491 |
| 64 | 0.1684 / 0.1740 | 0.3635 / 0.3741 | 0.3479 / 0.3821 |
| 128 | 0.1055 / 0.1067 | 0.2250 / 0.2291 | 0.2059 / 0.2245 |

Increasing observations substantially improves fitted-GP prediction in **every** scenario. In the correctly specified RBF setting, mean absolute length error drops from 0.08024 at n=8 to 0.01257 at n=128. Mean amplitude error also declines, but is still 0.2841 at n=128; good predictions do not require every parameter estimate to equal the generating value on each field realization.

At n=128, the estimated-minus-oracle mean RMSE gap is:

| Scenario | Paired absolute RMSE gap | Field-bootstrap 95% interval | Ratio of mean RMSEs: excess over oracle |
| --- | ---: | ---: | ---: |
| Stationary RBF | 0.00118 | [-0.00064, 0.00309] | 1.12% |
| Stationary two-scale control | 0.00412 | [-0.00139, 0.01020] | 1.83% |
| Nonstationary two-scale GP | 0.01860 | [0.01211, 0.02527] | 9.03% |

The relative percentages divide the overall mean fitted RMSE by the overall mean oracle RMSE and subtract one; they are not means of per-field percentages. The first two intervals include zero, which does not establish exact equivalence. The nonstationary residual remains positive under this paired diagnostic, and no n=128 optimization hits its parameter bounds. This is consistent with a remaining limitation of the global single-RBF model for this construction. It is not proof that optimization is globally optimal or that DeepKriging will close the gap.

The nonstationary **absolute** gap shrinks from 0.06013 at n=8 to 0.01860 at n=128. Do not claim that more observations fail to help there: fitted RMSE falls from 0.8353 to 0.2245. The larger residual relative to oracle is the relevant remaining challenge.

### Calibration and the fixed-wrong baseline

Mean coverage of fitted-model latent 95% intervals:

| Measurements | Stationary RBF | Stationary two-scale control | Nonstationary two-scale GP |
| ---: | ---: | ---: | ---: |
| 8 | 84.03% | 81.15% | 84.13% |
| 16 | 90.35% | 86.81% | 87.33% |
| 32 | 91.25% | 91.42% | 90.63% |
| 64 | 92.95% | 91.32% | 91.49% |
| 128 | 94.72% | 93.23% | 94.13% |

More observations markedly reduce aggregate undercoverage in this benchmark. The fixed wrong covariance also yields lower RMSE as n grows, but its intervals become more overconfident. At n=128 its RMSE/coverage pairs are 0.1702/57.57% for stationary RBF, 0.4795/20.42% for the stationary mixture, and 0.4794/27.12% for the nonstationary mixture. More data alone do not repair a fixed wrong uncertainty model.

Aggregate coverage can hide regional imbalance. In the nonstationary scenario at n=128:

| Region | Oracle RMSE | Fitted RMSE | Oracle coverage | Fitted coverage |
| --- | ---: | ---: | ---: | ---: |
| Bottom: smooth | 0.1199 | 0.1793 | 94.53% | 98.57% |
| Middle: transition | 0.2010 | 0.2089 | 95.01% | 94.12% |
| Top: rough | 0.2499 | 0.2587 | 94.79% | 89.71% |

The stationary RBF fit uses one global estimated length, averaging about 0.1291 at n=128. Its smooth-region error and conservative intervals coexist with rough-region undercoverage. This spatial imbalance is a more specific target for a spatial-basis mapper than the aggregate 94.13% coverage alone. The regional plot uses all three prespecified regions, at the prespecified largest sample size, and is a descriptive diagnostic of the recorded results.

### Compute, interpretation, and next comparison

Fresh fitted snapshots average about 11–12 ms at n=8 and 35–40 ms at n=128 across scenarios, including complete conditioning after optimization. Timing varies by fit and machine; the beginning of the evaluation overlapped short unit-test runs. The old online schedule has different cumulative costs and is not benchmarked here.

These results support examining sample size before adding model complexity: the correctly specified fitted GP becomes a strong benchmark once observations are less sparse. The spatially varying case retains modest but measurable oracle headroom and regional calibration imbalance. A future DeepKriging comparison should use the exact archived prefixes, include a coordinate-input DNN, and report sparse as well as denser regimes. GP models with nonstationary covariance are also possible comparators; this experiment evaluates a specific stationary family rather than the entire class of GP models.

Do not infer that the previously frozen neural planner's input-shift failure has been fixed. This study changes observation count, grid, and fitting protocol in a mapper-only comparison. Policy adaptation or performance with larger observation sets requires its own matched experiment. No DeepKriging advantage has yet been observed.

```sh
PYTHONPATH=src .venv/bin/python -m spatial_exploration.sample_size_study --output outputs/sample-size-study
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

The study itself requires the base NumPy/Matplotlib/scikit-learn dependencies and no optional RL packages. Use a fresh output folder; the workflow refuses to overwrite an existing study configuration.

- `config.json`, `requirements-run.txt`: complete fixed protocol and dependencies.
- `observations.npz`, `evaluation_only.npz`, `dataset_manifest.json`: reusable observation prefixes and separate privileged evaluator data with fingerprints.
- `metrics.csv`, `regional_metrics.csv`: all field/sample-size/model results, parameters, and costs.
- `parameter_fits.csv`: every batch snapshot fit, likelihood, bounds, warnings, and time.
- `summary.json`, `paired_comparisons.json`, `diagnostics.json`: aggregate curves, paired oracle gaps, validation counts, and fitting diagnostics.
- `learning_curves.png`: RMSE and coverage by sample size in all three scenarios, with field-bootstrap intervals.
- `oracle_gaps_and_regions.png`: paired fitted-minus-oracle error curves and nonstationary regional coverage at the largest sample size.
- `example_{scenario}_n8.png` and `example_{scenario}_n128.png`: predefined field 0, same measured sites, true field/dependence, and mapper predictions/uncertainty. Points show the experimental design; no travel trajectory is implied.
