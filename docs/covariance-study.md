# Covariance estimation: protocol and first development run

## What changed

The original experiment supplies the exact generating covariance to the mapper. This experiment compares it against an assumed fixed covariance and a fitted covariance, without changing the field generator or greedy acquisition score.

| Model | RBF length scale | Process variance |
| --- | --- | --- |
| Oracle | True value: 0.2 | True value: 1.0 |
| Fixed initial | Fixed at 0.35 | Fixed at 1.5 |
| Estimated | Starts at 0.35; fits from acquired observations | Starts at 1.5; fits from acquired observations |

All three know the zero mean, RBF family, and sensor noise standard deviation of 0.1. This is partial removal of oracle information, not a fully unknown covariance model.

## Estimation

Use scikit-learn's GaussianProcessRegressor with ConstantKernel times RBF, `normalize_y=False`, and known observation variance supplied through `alpha`. Fit by maximum log marginal likelihood. The initial fit uses four observations; subsequent fits occur at counts 7, 10, 13, etc. Between fits, use exact sequential GP updates with the current parameters. After each accepted fit, reconstruct the posterior from all acquired observations under the new kernel.

Use the previous parameter values as the first optimization start and two reproducible random restarts. Bounds are [0.05, 1.0] for length scale and [0.05, 5.0] for process variance. These are demonstration settings; they were not selected by a validation search. Reject an optimization result if its likelihood is nonfinite or worse than the starting likelihood beyond numerical tolerance. Log convergence warnings, boundary hits, and fit times.

The estimated mapper is never passed the true generating parameters, unobserved field values, or held-out labels. Known sensor noise is passed explicitly. Predictive intervals are for the latent field, not noisy future observations.

Reference: [scikit-learn GP regression documentation](https://scikit-learn.org/stable/modules/gaussian_process.html).

## Two protocols

**Replay:** construct one greedy route using the fixed initial covariance, independent of the true parameters and observed values. All mappers receive exactly the same observations in the same order. This isolates mapper differences. It is one sampling design, so results need not generalize to other observation layouts.

**Closed loop:** each model uses its current posterior covariance to choose its own next measurement. Start, field, potential sensor observations, permitted sites, and travel budget are paired across models. This compares whole systems, including differing numbers and locations of measurements.

The estimated covariance can change with measured values, so its closed-loop route can now depend on the measurements. Oracle and fixed-initial routes remain value-independent because their kernels stay fixed.

## First run

Run on November 3, 2026 with ten development fields (seeds 0–9), a 15-by-15 grid, and travel budget 2.0. No parameters were adjusted after viewing these results. The original baseline's five-seed numerical summary remains exactly unchanged after the shared-code refactor.

### Identical-observation replay

All models acquire 21 measurements along the same route. The following are averages across the ten fields; grid sites within one field are not independent replications.

| Model | Held-out RMSE | Empirical latent-field 95% interval coverage |
| --- | ---: | ---: |
| Oracle | 0.5791 | 94.4% |
| Fixed initial | 0.8392 | 58.4% |
| Estimated | 0.5938 | 89.8% |

Here, estimation reduces error relative to the fixed initial model, while its interval coverage falls below the nominal 95%. Average final fitted length scale is 0.2133 and average fitted process variance is 1.0564; these averages do not establish accurate recovery on each individual field.

### Model-dependent exploration

| Model | Held-out RMSE | Empirical latent-field 95% interval coverage | Mean measurements |
| --- | ---: | ---: | ---: |
| Oracle | 0.6508 | 96.2% | 24.0 |
| Fixed initial | 0.8392 | 58.4% | 21.0 |
| Estimated | 0.6232 | 90.0% | 22.5 |

The estimated model has slightly lower observed RMSE than the oracle in this small closed-loop run. This does not establish that covariance estimation beats knowing the covariance: routes differ, the planner is greedy rather than globally optimal, and ten realized fields give a noisy comparison. The same-observation protocol is the cleaner check of mapper behavior.

Across both protocols, 129 parameter fits were performed, with two boundary hits and no rejected fits. Raw diagnostics are in `outputs/covariance-study/parameter_fits.csv`.

## Interpretation limits

- Plug-in GP intervals ignore uncertainty in estimated covariance parameters; sparse early observations and model assumptions can also affect calibration. The observed undercoverage is consistent with overconfidence but does not identify a single cause.
- Sampling has no measurement-time or energy cost. Closed-loop measurement counts differ, so an equal-travel comparison is not an equal-measurement comparison.
- Parameters can be weakly identified from a short local trajectory. Larger or more diverse observation sets and additional fitting strategies may change the result.
- The kernel family and zero mean are still correctly specified, and noise is still known.
- The shared replay path uses one fixed initial kernel. Test other observation designs and unseen parameter settings before making broad claims.
- Seed-level uncertainty and a larger, separately held-out evaluation are required for substantive performance claims. These are development results, not a final test set.

## Outputs and reproduction

```sh
PYTHONPATH=src python3 -m spatial_exploration.covariance_study --seeds 10 --output outputs/covariance-study
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

- `mapping_comparison.png`: seed-0 predictions and uncertainties under identical observations.
- `exploration_comparison.png`: seed-0 model-dependent routes, predictions, and uncertainties.
- `performance_curves.png`: mean replay RMSE by measurement count and mean closed-loop RMSE by travel distance. Between observations, retain the last available prediction.
- `metrics.csv`: final metrics for each field, model, and protocol, including posterior-update time (optimization included; plotting, evaluation, and planning excluded).
- `learning_curves.csv`: per-observation metrics and current parameters.
- `trajectories.csv`: actual acquired sites and noisy values.
- `parameter_fits.csv`: optimizer diagnostics and parameter histories.
- `summary.json` and `config.json`: descriptive summaries, paired differences, and run settings.

Proceed next to selecting and reproducing a learning-based exploration method. Keep this covariance study as a controlled baseline; DeepKriging integration remains a later hypothesis.
