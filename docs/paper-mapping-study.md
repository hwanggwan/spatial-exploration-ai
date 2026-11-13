# DeepKriging paper-function study

This study asks whether a controlled reimplementation recovers the DeepKriging paper's advantage on its deterministic 2D function, and what happens when acquired observations fall from 810 to 128, 32, and 16. It is a mapper comparison with shared observations, without route selection or travel costs. It does not establish a resource-constrained exploration advantage.

A subsequent [same-Keras audit](keras-reuse-audit.md) compares cross-fold weight reuse with independent fitting at matched cumulative gradient-update budgets. It retains this study's exact observations and folds, while keeping contaminated reuse scores separate from genuine held-out predictions. It does not alter these recorded PyTorch results.

The later [provenance audit](paper-provenance-audit.md) finds that our exponential GP mean and population SD actually match the published **Table S4** baseline at its printed precision, and nearly reproduce the saved 2020 GP fold outputs. The numerical-reproduction failure discussed below concerns **Table S2** and the neural entries, rather than every published GP result.

## Source audit and scope

The primary references are the [published supplement, Sections S2 and S3.2 and Table S2](https://www3.stat.sinica.edu.tw/statistica/password.asp?art=13&num=1&sup=s&vol=34) and the [authors' public 2D script](https://github.com/aleksada/DeepKriging/blob/master/Nonstationary_function-2d.py). The supplement describes four 100-unit hidden layers with its default dropout/batch normalization. The script instead has three hidden layers with both regularizers commented out. The script also makes two consecutive 200-epoch training calls per fold.

More seriously, the public script constructs its neural models before the cross-validation loop and keeps their weights between folds. If run as written, an earlier training fold can expose labels that are later evaluated as held out. This is a source-code observation, **not evidence that the paper's tables were produced by that exact script**. This implementation constructs fresh weights and a fresh optimizer for every fold, count, architecture, and initialization run.

Two source-based neural configurations were declared before fitting, rather than choosing one using held-out performance. The result is a controlled reimplementation, **not an exact numerical reproduction**. TensorFlow/Keras and GPy are replaced by PyTorch and scikit-learn. Network initialization draws, batch order, batch-normalization internals, and GP optimization details can change results. The source's reported architecture search is not repeated. Test data select no hyperparameters, epochs, or final models; there are no extra tuning labels in the sparse settings.

## Frozen protocol

The noiseless response on a 30-by-30 grid over the unit square is

\[
t=(x+y)/2,\qquad f(x,y)=\sin\{30(t-0.9)^4\}\cos\{2(t-0.9)\}+(t-0.9)/2.
\]

There is one fixed analytic surface, not a collection of independent Gaussian-process realizations. Ten shuffled folds use split seed 123. Each fold holds out 90 sites. A separate seeded permutation of its remaining 810 sites defines strictly nested acquired sets of 16, 32, 128, and 810. **The same 90 test sites remain fixed at all four counts**, and every model receives exactly the same acquired locations and labels. Smaller-count results score those same 90 points, rather than silently changing the evaluation set to all unobserved points.

| Model | Inputs | Fixed fitting settings |
| --- | --- | --- |
| Exponential GP | xy | Zero mean; fitted amplitude, length scale, and nugget |
| RBF GP | xy | Same GP optimization; smoother-kernel control |
| DK paper description | 1,830 spatial basis values | 4 hidden layers of 100 ReLUs; dropout and batch normalization; 200 epochs |
| DK clean public-code variant | Same 1,830 values | 3 hidden layers of 100 ReLUs; no dropout or batch normalization; 400 epochs |
| Coordinate DNN | xy only | Supplement-default 3 hidden layers of 100 ReLUs; dropout and batch normalization; 200 epochs |

The basis uses Wendland kernels on 10-by-10, 19-by-19, and 37-by-37 knot grids, with support radius 2.5/m at resolution m, matching the public script. Inputs are raw basis values, without extra xy columns or feature standardization. All 1,830 columns have nonzero support somewhere on the known candidate grid and remain available at every count; support and feature count are not retuned for sparse data. The earlier project adaptation uses different features, support, widths, and regularization, so its error levels are not directly comparable.

The supplement recommends removing all-zero columns for irregular data. Here the dictionary is built on the full known query grid, following the public script, so none is removed. This choice becomes consequential when extending the task to sparse acquired sets: the mean numbers of columns nonzero at an acquired site are only 521.4 / 834.3 / 1,580.1 / 1,830.0 at n=16 / 32 / 128 / 810. This geometry-only diagnostic was examined after fitting began; it changes no settings. Columns inactive at every acquired site receive no direct first-layer data gradient. Dropping them using acquired locations or choosing a coarser dictionary is a separate, untested sparse-data adaptation. This observation does not establish how much of any error gap the dictionary causes, and it cannot explain a dense n=810 discrepancy, where all columns are active.

Neural fitting uses Adam (learning rate 0.001, epsilon 1e-7), shuffled batches of 64, first-layer He-uniform initialization, subsequent Glorot-uniform initialization, and zero biases. Dropout rate is 0.5 after hidden layers one and two; batch normalization follows hidden layers one and three, with epsilon 0.001 and PyTorch momentum 0.01. The fourth hidden layer in the paper-description version has no extra regularization. Three initialization seeds run independently for every neural configuration. The same architecture/fold/member starts from identical weights across observation counts; all optimizer state is reset. The primary neural metric averages **individual-run RMSEs**, rather than obtaining an extra ensemble-prediction advantage.

Both GPs optimize marginal likelihood with two random restarts in addition to the initial fit. Initial amplitude/length/noise are 1; bounds are amplitude [1e-6, 1000], length [1e-3, 100], and noise variance [1e-10, 1]. Numerical diagonal jitter is 1e-10; targets are not centered or standardized. The fitted nugget is retained despite noiseless target generation, consistent with the source GPRegression model. Exponential is Matérn with smoothness 1/2; RBF is an added control rather than the paper baseline. Archived GP latent variances subtract the fitted observation nugget from the predictive variance. Plug-in parameter uncertainty is not integrated.

Fixed epochs imply more gradient updates at larger counts: at n=810, 13 batches per epoch, compared with one at n=16 or 32. Thus the count comparison changes both available data and optimizer updates under the source-style training rule. It is not a pure data-only causal effect. The two DK variants also differ in architecture, regularization, and training duration; their difference does not isolate a single mechanism.

The coordinate control follows the supplement's default architecture rather than matching the spatial models' parameter counts. In particular, comparing the unregularized 400-epoch DK variant with the regularized 200-epoch coordinate control cannot isolate the benefit of the basis alone. This source-oriented comparison complements, rather than replaces, the earlier capacity-matched project study.

## Results

Recorded November 5, 2026, using Python 3.12.15, NumPy 2.5.4, PyTorch 2.15.0, scikit-learn 1.9.1, and SciPy 1.18.1. The run contains 80 GP fits and 360 independent neural fits. Settings were not changed in response to these outcomes.

Mean held-out RMSE (lower is better; neural entries average individual runs across three initializations and ten folds):

| Acquired points | Exponential GP | RBF GP | DK paper description | DK clean public-code variant | Coordinate DNN |
| --- | ---: | ---: | ---: | ---: | ---: |
| 810 | 0.015136 | **0.004264** | 0.069060 | 0.017772 | 0.131799 |
| 128 | 0.058881 | **0.043076** | 0.198650 | 0.095497 | 0.241245 |
| 32 | **0.137431** | 0.140637 | 0.280218 | 0.194843 | 0.277664 |
| 16 | **0.197032** | 0.197333 | 0.287478 | 0.237263 | 0.286982 |

**The published DeepKriging advantage is not recovered under this frozen, fresh-fold implementation, even at n=810.** The clean public-code variant improves substantially with more observations, but its mean error stays above the exponential GP at every count. It beats that GP in 2/10 folds at n=810 and in 0/10 folds at the three smaller counts, after averaging neural initialization RMSE within each fold. The smoother RBF control improves over exponential in all ten dense folds. These are descriptive counts, not independent-field significance tests.

For numerical context, Table S2 reports dense test RMSE 0.003466 for DeepKriging, 0.006934 for its coordinate baseline, and 0.008552 for exponential Kriging. The current run reproduces neither those numerical values nor that ranking. Source discrepancies, removing fold-to-fold neural weight reuse, and framework/optimization differences remain possible contributors. The present experiment does **not** isolate which causes the discrepancy or show that the published table used contaminated code.

Training errors distinguish two unresolved issues. At n=810, the paper-description network has mean training RMSE 0.066171 and test RMSE 0.069060: substantial fitting error remains under this fixed regularized 200-epoch rule. The clean variant has training RMSE 0.004061 and test RMSE 0.017772. At n=16, the clean variant nearly interpolates its acquired labels (training RMSE 0.00000353) while test RMSE remains 0.237263. Thus neither simply assuming sparse observations are the sole problem nor interpreting good training fit as good unseen-site prediction is supported. The clean variant's three seed-wise dense mean RMSEs range only from 0.017709 to 0.017891, although three seeds are a limited optimization audit.

GP optimization is also not a fully converged oracle reference: 44/80 fits record warnings, mostly a fitted nugget at its 1e-10 lower bound; some optimizer starts terminate abnormally. All fitted predictions are finite and no warnings or fits were discarded. Learned parameters and every warning remain in `fit_diagnostics.json`. Bounds or optimization strategies were not adjusted after looking at these errors.

The recorded fit/scoring loop took 524.34 seconds, including checkpoint/progress I/O but excluding final plots. Mean n=810 fit time per individual model is 4.682 seconds (exponential GP), 5.750 seconds (RBF GP), 3.672 seconds (paper-description DK), 5.484 seconds (clean-code DK), and 1.741 seconds (coordinate DNN). Neural initialization replicates are experimental repeats, not a required three-model ensemble. Query timing is archived separately. This single-machine run does not establish a scaling or latency advantage.

![Fixed-test error versus acquired count](../outputs/paper-mapping-study/sample_size_comparison.png)

The main conclusion is limited: increasing observations alone does not recover a DeepKriging advantage for these source-based configurations. Before attributing the discrepancy to the method or integrating it into exploration, an independently planned implementation/optimization audit is more informative than further favorable scenario selection.

## Uncertainty and interpretation limits

This task evaluates point prediction. It does not reproduce the paper's separate density-estimation uncertainty method. Variation among initialization runs is not a calibrated posterior or a reliable exploration acquisition function. GP latent variances are archived for inspection; no uncertainty winner is asserted here.

Ten folds share much of their training data and come from one surface. Fold-level variation and paired differences are descriptive; they are not independent-field confidence intervals. Three initializations measure some optimization sensitivity, not uncertainty over new environments. Randomly scattered test sites mainly evaluate interpolation and do not mimic an explorer's large unvisited region.

Changing the GP kernel, neural regularization, training duration, and sample count can affect the outcome. A stationary covariance can interpolate a deterministic surface with changing local variation well; naming the surface nonstationary does not guarantee that a neural mapper wins. Conversely, a negative result for these fixed source-based configurations would not rule out a better tuned DeepKriging model.

Any subsequent tuning must use separate development labels/surfaces before a new final comparison. If the goal returns to sequential exploration, validate sparse-data conditioning and predictive uncertainty, then compare routes on shared starts, fields, and both measurement and distance caps. Dense interpolation success alone would not resolve those requirements.

## Reproduction and artifacts

```sh
.venv/bin/python -m pip install -e '.[neural]'
PYTHONPATH=src .venv/bin/python -m spatial_exploration.paper_mapping_study --output outputs/paper-mapping-study
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

Use a fresh output folder. All fitting uses one CPU thread in the CLI. `protocol.json` and `frozen_protocol.json` are written before fitting. `splits.json` and `acquired.npz` preserve every actual acquired set; hidden full-grid truth is separate in `evaluation_only.npz` and is never passed into a fitting function. Known query geometry is available to all models. `predictions.npz` preserves every full-grid prediction and GP latent variance. `metrics.csv`, `summary.json`, `paired_fold_differences.json`, and `fit_diagnostics.json` retain all outcomes, timing scopes, learned GP parameters/warnings, and neural initial/final weight hashes. Fold-0 checkpoints include every observation count and neural initialization; examples use the predefined first initialization rather than a selected best run. Dataset and source hashes are archived.

`sample_size_comparison.png` compares held-out and training errors; `example_predictions.png` shows the predefined fold-0 surface maps. Fit timing includes initialization and optimization; query timing is separate, and checkpoint I/O is excluded. These are recorded CPU costs, not hardware-independent speed claims.

All 85 tests pass with optional dependencies. System Python passes 44 and skips 41 optional tests. The new tests check source-function/basis values, fixed nested acquisitions, restart independence, checkpoint reconstruction, and an end-to-end acquired-label-only fitting workflow. After the full run, all 440 held-out RMSEs were recomputed from archived predictions with zero discrepancy; all 36 saved checkpoints reproduce their archived grid predictions exactly. Ninety distinct initialization hashes cover the independent neural model/fold/member groups, reused only as fresh initial states across counts. Dataset hashes and finite nonnegative GP latent variances are verified in `verification.json`. `basis_coverage.json` records the post-hoc geometry diagnostic. Minor x-axis ticks were cleaned up after visual inspection, without rerunning fits or changing predictions; `fitting_source_manifest.json` retains fit-time source hashes and `source_manifest.json` records final source hashes.
