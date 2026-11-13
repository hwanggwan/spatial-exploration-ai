# Does the graph basis need a neural network?

The [graph DeepKriging experiment](barrier-deepkriging-study.md) found a point-prediction gain with 16 readings on fields where a wall weakens dependence. This follow-up adds two linear controls: ridge on the Wendland inputs, and ridge on exactly the inputs used by graph DeepKriging. It replays the same acquisitions, fields, and reserved sites; it does not add new hold-out fields or compare exploration policies.

## Model

Each map gets its own linear fit. For a location with archived input row `x(s)` and `p` input columns, the latent prediction is

```text
f(s) = b + x(s)ᵀβ
y(s) = f(s) + ε,  ε ~ N(0, σ²),  σ = 0.12
β ~ N(0, τ²/(p+1) I),  b ~ N(0, τ² r)
```

The Wendland control uses 236 inputs (coordinates and multiresolution Wendland features), and the graph control uses the same 236 plus 144 graph heat features. Coefficients are learned from the 8 or 16 noisy values actually acquired on the current map. Given the Gaussian model, the posterior mean equals the ridge estimate with objective `SSE + (σ²/τ²)[(p+1)||β||² + b²/r]`. At `r=0`, fix the intercept to zero. The `p+1` scaling keeps feature prior scale comparable across input dimensions, and the intercept has its own prior because a smooth field can have a nonzero level on one map. With far fewer readings than coefficients, the penalty specifies how to shrink the many solutions that could fit those readings.

This model can also be read as a finite-feature GP with kernel `τ² [x(s)ᵀx(s')/(p+1) + r]`. The archived feature columns themselves do not change. The graph block contributes an additive term to the induced covariance, and the predictor can be solved in observation space with an 8-by-8 or 16-by-16 Cholesky factorization. The model is linear in the features, while its predictions can be nonlinear in the original coordinates.

Tune 17 log-spaced prior variances from 0.01 to 100 and intercept ratios 0, 0.25, and 1 by mean per-map held-out RMSE on the original development fields, separately for each representation and reading count. Use the same regime-pooled selection criterion as the neural benchmark. The winners all have `τ²=100` and `r=0.25`. A development-only check at 1,000 and 10,000 finds the objective close to its weak-regularization limit: increasing `τ²` from 100 to 10,000 decreases tuning RMSE by at most 4.3e-6. The archived test used the original winner, not the extended grid.

Compute latent posterior marginal intervals and apply the original study's scalar residual-to-standard-deviation calibration on the separate calibration fields. Calibration changes interval widths only, and the multipliers and ridge settings are saved before loading the test acquisitions and test truth. No benchmark target selects a setting. Because the benchmark was previously seen and the new arms were motivated by those results, treat this as a retrospective paired comparison.

## Results

Every cell uses 96 fields and the same 28 reserved evaluation locations per layout. Values are mean per-field latent RMSE.

| Field dependence | Readings | Wendland ridge | Graph ridge | Wendland DeepKriging | Graph DeepKriging | Geometry GP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Wall weakens dependence | 8 | 0.9128 | 0.9035 | 0.9083 | 0.9044 | 0.9237 |
| Wall weakens dependence | 16 | 0.8410 | 0.8184 | 0.8440 | 0.8160 | 0.8449 |
| Dependence crosses wall | 8 | 0.8927 | 0.8870 | 0.8974 | 0.8952 | 0.9093 |
| Dependence crosses wall | 16 | 0.8089 | 0.7983 | 0.8159 | 0.8125 | 0.8333 |

On the barrier fields with 16 readings, adding the graph basis to ridge lowers RMSE by **0.0227 (2.7%)**, with a paired 95% interval **[−0.0310, −0.0139]** for graph ridge minus Wendland ridge. The reduction has the same sign on both test door placements and occurs on 72 of 96 fields. Near-wall RMSE drops from **0.8261 to 0.7984**. On the same graph features, switching from ridge to the MLP lowers mean RMSE by just **0.0024**, with interval **[−0.0185, +0.0133]** for MLP minus ridge. This interval does not establish equivalence, but the comparison does not resolve an additional nonlinear-model benefit. The representation-by-model interaction also spans zero (−0.0054 [−0.0164, +0.0060]).

The linear fit reproduces most of the earlier point-prediction improvement in this setting. Its gain with graph features even when dependence crosses the wall means the contrast does not isolate the wall topology itself: the appended block also supplies a spatially smooth prior component, rescales the total feature kernel, and changes its shape. A graph basis with wall edges restored at the same feature dimension could isolate topology in a further experiment. The separate neural and ridge settings are chosen by the same development metric, but this is a comparison of fitted estimators rather than an intervention on nonlinearity alone.

With 16 barrier readings, calibrated 95% coverage is **94.6%** for graph ridge and **94.9%** for graph DeepKriging. Graph ridge has a narrower mean width (**3.08 vs. 5.70**) and lower interval score (**3.99 vs. 6.76**; lower is better). Raw graph ridge intervals cover 100% of the test values because point tuning selects a very large prior variance, so their useful width is due to held-out calibration. The graph ridge standard-deviation multiplier is 0.086 at 16 readings. These marginal results do not validate posterior cross-location covariance or establish a planning advantage. Mean travel is identical across all arms (0.693 and 1.636 units at 8 and 16 readings), and all point and interval comparisons use the same acquired observations.

Paired intervals use 4,000 bootstrap resamples of fields within fixed layouts. They are descriptive, unadjusted, and conditional on these two test layouts. Archived neural timings include batched optimization; ridge timings include feature covariance construction and batched conditioning on the same CPU, not sensor costs or actual online latency.

## Reproduce

```sh
PYTHONPATH=src .venv/bin/python -m spatial_exploration.barrier_ridge_study \
  --reference outputs/barrier-deepkriging-study --output outputs/barrier-ridge-study
```

The follow-up uses the existing graph DeepKriging benchmark's exact test feature matrices and observations. It checks regenerated features and routes against those archives. Use a fresh output directory. Outputs include `frozen.json`, `metrics.csv`, `summary.json`, `paired-effects.json`, `paired-effects.png`, `predictions.npz`, and the source archive checksums. `boundary-check.json` records the development-only regularization sensitivity. `verification.json` records disjoint phase seeds, 384 verified test acquisition sequences, and agreement with 768 independent sequential posterior replays (maximum differences 3.11e-15 in the mean and 1.14e-13 in variance). The unit test additionally compares with an independent coefficient-space ridge solve. The Torch suite passes 94 tests, with two optional TensorFlow skips; system Python passes 50, with 46 optional dependency skips.
