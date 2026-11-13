# Graph DeepKriging at fixed measurement budgets

This study extends the [room-and-door GP experiment](barrier-study.md) and the existing [DeepKriging-style mapper](deepkriging-study.md). It asks two separate questions: whether a graph basis helps the neural mapper, and whether a neural mapper with geometry improves on the geometry-aware GP. This is a per-map representation experiment, not a numerical reproduction of the DeepKriging paper or a closed-loop exploration comparison.

## Model and design

- **Wendland DeepKriging.** Two coordinates plus 234 compact Wendland basis features at three resolutions, standardized over known grid coordinates. The model is a two-hidden-layer ReLU MLP with 33 units in each layer and 8,977 trainable parameters.
- **Graph DeepKriging.** Keeps all 236 inputs and appends 144 graph basis values, one channel anchored to each grid cell. Each layer has 22 units, for 8,911 parameters (0.74% fewer). This is an approximate parameter-budget control, not an equal-width ablation. Both ensembles have five independently initialized members and are fitted from scratch on each map, without learned weights shared across maps.
- **Geometry GP.** The previous GP's mixture of a Euclidean RBF and a normalized graph heat kernel. It selects the mixture and length by marginal likelihood on the acquired labels and assumes zero mean, unit prior variance, and known sensor noise. It is a non-oracle comparator: simulator permeability and correlation scale are hidden. Its kernel family and parameter search are distinct from the neural model's fixed graph feature scale.

For the graph basis, remove direct adjacency through the wall except at the door. Let the resulting graph Laplacian be `L`, and compute the symmetric square root of its heat kernel, `B = exp(-tL/2)`. With `N=144` and `F_i = sqrt(N) B_i / ||B_i||`, the feature Gram matrix `FFᵀ/N` is exactly the GP's unit-diagonal graph heat component at length 0.22. The features depend on the known floor plan alone, and each block's scale is set before inspecting targets. The neural model can combine them nonlinearly with Wendland features. The graph GP and graph neural model both see the same floor plan, although they use it differently.

All three models receive the same sensor readings from a target-blind stationary-prior greedy route on each map, with legal distances through the door. The fixed route reaches exactly 8 readings within 1 distance unit or 16 within 2 units, counting the start. Thus no model can gain by collecting a different location or spending more. The routes at the two caps can differ, so changing caps is not a pure observation-count ablation. There are 28 reserved evaluation sites per layout, eight in the columns adjacent to the wall. Scores use latent, noise-free truth at those sites.

The field generator uses a graph resolvent covariance at two spatial scales, not the fitted graph heat covariance. Both a weakly permeable wall and a field with dependence crossing the wall are kept. Each pair shares the Gaussian driving draw and sensor noise (SD 0.12) across regimes. The three phases use independent seed ranges:

| Phase | Door rows (zero-indexed) | Fields per layout per regime | Purpose |
| --- | --- | ---: | --- |
| Tuning | 3, 8 | 16 | Select epochs (100, 300, 600) and Adam weight decay (0.0001, 0.01) for each neural mapper/count by mean held-out RMSE, pooled over development layouts and regimes |
| Calibration | 3, 8 | 16 | Fit a marginal interval multiplier per mapper/count on separate development maps, pooled over regimes |
| Test | 2, 9 | 48 | Evaluate frozen settings on new fields and new door positions |

The test fields are generated only after writing the selected settings and calibration multipliers. Every test map is independently fitted on its own noisy acquisitions; no tuning, calibration, or selection uses its held-out values. The new door positions are within the same two-room layout family, so this tests limited transfer rather than generalization to arbitrary floor plans.

For uncertainty, raw neural variance is initialization spread, not a GP posterior. Raw GP variance is the fitted plug-in latent variance. Each model's reported calibrated interval scales its raw standard deviation by a scalar estimated on calibration maps from the 95th percentile of the absolute residual-to-std ratio divided by 1.96. The same procedure is applied to the GP; uncalibrated coverage and widths are retained. This is empirical marginal calibration, not a conformal guarantee or a way to predict variance reduction after a new measurement.

## Results on the frozen test

Each row aggregates 96 fields over two new door placements. The values are means of per-field RMSEs at the reserved sites.

| Field dependence | Readings | Wendland DeepKriging | Graph DeepKriging | Geometry GP (per-map likelihood) |
| --- | ---: | ---: | ---: | ---: |
| Wall weakens dependence | 8 | 0.9083 | 0.9044 | 0.9237 |
| Wall weakens dependence | 16 | 0.8440 | 0.8160 | 0.8449 |
| Dependence crosses wall | 8 | 0.8974 | 0.8952 | 0.9093 |
| Dependence crosses wall | 16 | 0.8159 | 0.8125 | 0.8333 |

At 16 readings with a barrier, graph DeepKriging reduces mean RMSE by 3.3% relative to the Wendland network and 3.4% relative to the geometry GP. It wins on 69/96 and 59/96 fields, respectively, and the mean differences have the same sign in both test layouts. Paired differences (graph neural minus reference) are −0.0280 [−0.0419, −0.0131] versus the Wendland network and −0.0289 [−0.0454, −0.0141] versus the GP. Near-wall RMSE is 0.7970 versus 0.8347 and 0.8303, respectively. At 16 readings without a dependence barrier, the graph neural model also improves on this fitted GP (difference −0.0207 [−0.0347, −0.0071]), but is essentially tied with the Wendland network. Neither 8-reading condition has a clear paired RMSE advantage. Intervals are stratified by fixed layout and resample fields within each layout (4,000 draws, unadjusted); they do not describe uncertainty across arbitrary new layouts.

These are comparisons of the separately selected mapper configurations. They do not isolate network nonlinearity from the basis, regularization, or training schedule. Selected (epochs, weight decay) for the Wendland network are (100, 0.01) at n=8 and (600, 0.0001) at n=16; for the graph network they are (600, 0.0001) and (100, 0.01), respectively.

## Uncertainty and resource costs

At 16 barrier readings, the graph network's raw nominal 95% coverage is **30.8%**. Separate calibration raises this to **94.9%**, close to the GP's calibrated 94.6%, by multiplying its standard deviations by 9.25. The graph model's mean interval width is **5.70**, versus **3.78** for the GP. The GP also has a better calibrated interval score (4.57 vs. 6.76, lower is better). The neural predictor improves point accuracy on this test while giving a less informative interval. Its initialization spread remains unsuitable as a drop-in posterior for planning. Across the two caps, both neural models use separately selected fits and calibration scales, so their interval widths need not decrease with the acquisition count.

Average actual distances are 0.693 and 1.636 units for the 8- and 16-reading caps, exactly matched across mappers. At 16 readings, the graph neural ensemble takes about 21 ms per map in batched fit-plus-prediction throughput, compared with 0.21 ms per GP fit and prediction and 124 ms per Wendland ensemble (selected for more epochs). These are CPU timings on this 144-site grid, not single-map online latency measurements.

## Secondary GP selection check

The primary GP selects kernel parameters separately on each map using the acquired likelihood. The neural hyperparameters benefit from separate labeled development fields. After the primary run, a secondary comparison gave the GP the same development fields: choose one fixed length and graph weight per count by the same tuning RMSE, then calibrate on the separate calibration fields. No test outcome chooses a candidate, but adding this comparison after seeing primary outcomes makes it a sensitivity check.

The selected kernel uses length 0.14, with graph weight 1 at 8 readings and 0.5 at 16. At 16 barrier readings the fixed GP improves from 0.8449 to **0.8369**, while graph DeepKriging stays at **0.8160** (paired difference −0.0209 [−0.0323, −0.0094]). The graph neural advantage also persists on the transparent fields (0.8125 vs. 0.8285), but without a clear benefit from graph features over the Wendland network in that regime. The fixed GP's intervals remain narrower (3.69 vs. 5.70 on barrier fields) and have a better interval score (4.26 vs. 6.76). This does not rule out stronger GP or linear basis estimators.

The [matched-input Bayesian ridge follow-up](barrier-ridge-study.md) compares the same Wendland and graph inputs with and without a neural network. On these archived fields, the graph basis improves ridge, and the additional gain from switching graph ridge to the neural network is unresolved. The uncertainty results do not yet motivate replacing the GP planner.

## Reproduce

```sh
PYTHONPATH=src .venv/bin/python -m spatial_exploration.barrier_deepkriging_study --output outputs/barrier-deepkriging-study
PYTHONPATH=src .venv/bin/python tools/audit_barrier_gp_selection.py outputs/barrier-deepkriging-study
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

Use a fresh output directory and the existing neural dependencies. The study uses one CPU thread and saves `protocol.json`, `frozen.json`, and `tuning.csv`; phase-specific acquisition and evaluator archives; test feature matrices and all neural checkpoints; `predictions.npz`, `metrics.csv`, `summary.json`, `paired.json`, `compute.json`, and `comparison.png`. Fit timings cover initialization, optimization or GP likelihood selection, and prediction on the grid; they exclude feature construction, checkpoint I/O, and acquisition. Neural timings are amortized batched throughput, not one-map online latency.

`verification.json` confirms distinct phase seeds and checks the 384 test acquisition records against the reserved sites and movement constraints. All 48 checkpoint batches (3,840 neural members) reload to exactly the archived predictions. Rebuilding all 384 GP fits and independently recomputing all 1,152 score records gives a maximum metric discrepancy of 4.45e-16. The test suite reports 93 passes and two optional TensorFlow skips in the Torch environment; system Python reports 50 passes and 45 optional dependency skips.
