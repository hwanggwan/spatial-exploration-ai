# Frozen-policy robustness to covariance estimation

## Question and fixed comparison

Does the rollout-cloned policy retain useful behavior when the exact covariance is unavailable? Evaluate frozen selected networks under three covariance treatments while holding the field, potential sensor values, starting location, and physical budget fixed within each scenario.

- **Oracle:** the mapper and planner know the generating unit-variance RBF covariance and its length scale.
- **Fixed initial:** use RBF length scale 0.35 and process variance 1.5 throughout. These are the initial assumptions already used in the covariance-estimation study, not choices tuned on this experiment.
- **Estimated:** start with those same assumptions, then fit length scale and process variance by maximum marginal likelihood after 4 observations and every 3 thereafter. Use the existing estimator unchanged: bounds [0.05, 1] and [0.05, 5], warm start plus two optimizer restarts, known mean zero and sensor noise standard deviation 0.1. Predictive uncertainty is conditional on plug-in parameters and omits parameter uncertainty.

Compare greedy, greedy-completion rollout, and both selected rollout-cloning seeds (0 and 1) in every treatment. There is no retraining, clipping, fallback policy, normalization change, or model selection using these results. Record selected-checkpoint SHA-256 hashes before evaluation and verify they remain unchanged afterward. The greedy-cloning and PPO policies are outside this focused frozen-rollout-clone comparison.

Use 96 new 7-by-7 cases with scenario seed 940000 and field/noise seeds starting at 1400000. As in training, sample starts from permitted sites, length scales uniformly from [0.12, 0.35], and budgets uniformly from [1.5, 2.5]. Check that parameter triples and seeds are disjoint from the imitation training, validation, and previous test sets. This relaxes parameter knowledge within the same RBF family and environment; it does not test nonstationarity, covariance-family misspecification, unknown sensor noise, obstacles, or robot dynamics.

Only endpoint measurements are acquired, each once. The starting measurement is free. Reserved grid indices `index % 5 == 4` cannot be measured and supply the held-out evaluation sites. Sensor measurements have no cost beyond movement. There is no return-to-base requirement.

## Planner interface and input shift

The planner-facing belief state receives only the current mapper, physical state, permitted-site mask, budget, and measurements already acquired. The evaluator keeps the true field, potential unacquired sensor values, and true kernel separately. Unknown-covariance mappers are constructed without the true prior or true length scale. The oracle treatment explicitly supplies them as a benchmark.

Use the same observation encoder as the original Gym environment: full modeled posterior covariance, visited flags, current coordinates, remaining budget scaled by the original training maximum 2.5, and the **current mapper length scale** scaled by the original training maximum 0.35. The estimated treatment reads the current estimated length scale after each scheduled fit. Fixed-initial uses 0.35; oracle uses the known true value only in that treatment. Measured values influence an estimated policy through covariance-parameter refits, not through a hidden true-kernel feature.

Preserve raw inputs without clipping. Training used a unit process variance and length scales in [0.12, 0.35]; estimated parameters may leave that support, and modeled posterior variances may exceed 1. At every decision log:

- Whether current model length is outside the training parameter range.
- Whether a posterior diagonal exceeds the training prior-variance cap 1 by more than 1e-6.
- Whether any encoded feature exceeds the original declared observation box [-1.1, 1.1], allowing tolerance 1e-6.
- Current model parameters and maximum absolute covariance entry.

These are coarse input-support diagnostics, not a statistical out-of-distribution detector or a causal attribution of any performance loss.

## Objective and guarantee boundaries

Use actual held-out latent-field RMSE and interval coverage to evaluate deployed predictions. Separately reconstruct an evaluator-only GP with the **true covariance** on each executed route. Its final mean latent posterior variance measures route information under the generating model; its held-out variance and RMSE are also recorded. It receives the same measurements along that route and never guides action selection. Under the correctly specified GP, its posterior covariance quantifies conditional expected squared prediction error; realized RMSE can rank methods differently.

The deployed mapper's own reported variance remains in the logs for diagnosis. It is not the cross-treatment performance objective: a misspecified or fitted covariance can report low variance while making inaccurate predictions. Count actual update steps where modeled variance increases. A covariance refit can increase uncertainty even after acquiring a new measurement; the original fixed-kernel variance-reduction reward is not interpreted as true information gain here. This is an evaluation study, with no RL reward or training.

At each rollout decision, freeze the current modeled kernel throughout each hypothetical greedy completion. Future refits and their uncertain future values are not simulated. Check that the chosen completion is no worse than the base greedy completion **under that frozen current model**. The fixed-model policy-improvement property does not guarantee a better route under the true kernel, better realized RMSE, or improvement over an entire exploration with future parameter refits. Do not assert a true-model or across-refit guarantee from that local check.

All planner comparisons are closed loop: different methods can choose different observations and therefore obtain different parameter estimates. This measures the complete mapper/planner system. It does not isolate a mapper's accuracy on identical observations; the earlier replay study addresses that separate question.

## Validation and reporting

Before new-case comparisons, a predefined archived oracle case (scenario 0 of the imitation test set) must reproduce the original numerical variance, RMSE, coverage, travel, and measurement count for all four methods. Unit tests check observation/mask/route equivalence with the original known-covariance Gym workflow, raw estimated features without clipping, independent joint-GP route variance, hidden-kernel and unacquired-value isolation, covariance increases after refitting, and rejection of invalid measurements without state mutation.

Record all scenario-level outcomes, actual observed values and routes, decision-level input-support flags, optimizer fits/boundary events/warnings, and hypothetical rollout completions. Report actual travel and measurement count alongside each case's common budget cap. Split cumulative online action-selection time from actual mapper-update time, including scheduled optimization; evaluator-only true-GP updates, metrics, and rendering are excluded from both. Use one PyTorch and one numerical-library thread. Single-pass local timings are descriptive rather than a hardware benchmark.

For comparisons within each treatment and treatment changes within each fixed planner, bootstrap paired scenario-level RMSE and true-route variance differences with 5,000 resamples and RNG seed 960000. Intervals condition on each frozen model and this scenario distribution; they do not describe training-seed uncertainty. Do not treat grid sites or trajectory states as independent replicates. This test set becomes exposed after analysis; future method changes need separate model-selection data and fresh evaluation.

## Results — November 4, 2026

All 1,152 episodes complete: 96 paired cases, three treatments, four methods. The predefined archived oracle case reproduces all checked outcomes. Every executed route respects its budget, reserved sites, and no-revisit rule; every frozen checkpoint hash remains unchanged. The full suite passes 42 tests; system Python passes 26 and skips 16 optional RL/imitation tests.

| Covariance | Planner | Held-out RMSE | Latent 95% coverage | True-GP route variance | Mapper's reported variance | Measurements | Actual travel |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Known | Greedy | 0.51721 | 96.41% | 0.36283 | 0.36283 | 11.208 | 1.929 |
| Known | Rollout | 0.50218 | 95.49% | 0.33333 | 0.33333 | 10.990 | 1.974 |
| Known | Neural seed 0 | 0.50433 | 96.18% | 0.35004 | 0.35004 | 10.875 | 1.918 |
| Known | Neural seed 1 | 0.50428 | 96.06% | 0.35181 | 0.35181 | 11.000 | 1.928 |
| Wrong fixed | Greedy | 0.68643 | 72.92% | 0.37780 | 0.24576 | 10.927 | 1.932 |
| Wrong fixed | Rollout | 0.63296 | 73.73% | 0.35805 | 0.19781 | 9.896 | 1.976 |
| Wrong fixed | Neural seed 0 | 0.66044 | 74.65% | 0.40049 | 0.25443 | 8.750 | 1.931 |
| Wrong fixed | Neural seed 1 | 0.72418 | 72.57% | 0.41052 | 0.27468 | 8.854 | 1.930 |
| Fitted | Greedy | 0.60627 | 87.38% | 0.37595 | 0.34031 | 11.240 | 1.927 |
| Fitted | Rollout | 0.60799 | 83.33% | 0.36166 | 0.29440 | 10.021 | 1.978 |
| Fitted | Neural seed 0 | 0.66127 | 84.03% | 0.43888 | 0.36000 | 8.271 | 1.930 |
| Fitted | Neural seed 1 | 0.65994 | 84.84% | 0.44376 | 0.39310 | 8.188 | 1.931 |

Numbers average the same 96 cases in each row. Report both neural seeds rather than selecting a favorable one. This fresh known-covariance control retains the neural policies' earlier advantage over greedy, but neither unknown-covariance treatment supports a robust improvement by the frozen neural policies in true-GP route variance.

In the fitted treatment, neural RMSE exceeds greedy's by 0.05500 and 0.05367. Paired-case bootstrap 95% intervals are [0.01480, 0.09371] and [0.00966, 0.10017]. Their true-route variance exceeds greedy's by 0.06293 and 0.06781, with intervals [0.04913, 0.07749] and [0.05319, 0.08312]; they are worse in 79 and 80 cases, respectively. This is a substantial loss under the stated frozen-policy deployment, not just a change in the mapper's self-reported confidence.

Fitted rollout still improves true-route variance over fitted greedy on average by 0.01429, interval [-0.02178, -0.00684]. Actual RMSE is nearly tied (difference +0.00172, interval [-0.04068, +0.04290]), and rollout's coverage is lower. Better route information does not imply a uniformly better fitted mapper or a lower realized error on each field.

Covariance fitting helps several closed-loop systems relative to the wrong fixed assumptions: mean RMSE changes by -0.08016 for greedy, -0.02497 for rollout, +0.00083 for neural seed 0, and -0.06424 for neural seed 1. The rollout and neural-seed-0 RMSE intervals include zero. Fitting worsens the neural policies' true-route variance relative to wrong fixed covariance, even as their model estimates and some prediction metrics improve. Do not infer a pure mapper effect from these model-dependent routes.

## Calibration and input-support diagnostics

The wrong fixed treatment's nominal 95% intervals cover only about 73–75% of held-out latent values. Fitting raises this to about 83–87%, still below nominal coverage. The data are sparse and uncertainty conditions on fitted parameters; this experiment does not establish how much of the undercoverage is due specifically to parameter uncertainty versus its interaction with estimation and adaptive data collection.

The fixed-model reported variance can be actively misleading across treatments. For rollout, wrong fixed covariance reports a mean final variance of 0.19781, lower than the known treatment's 0.33333, while the evaluator-only true-route variance is worse (0.35805) and RMSE rises from 0.50218 to 0.63296. Treating reported variance alone as the performance score would reward this overconfidence.

Under fitted covariance, the neural policies receive observations outside the original declared box on 66.33% and 66.23% of decisions; estimated length scales leave the training parameter range on 28.51% and 27.54%. About 61% of decisions have posterior diagonal entries above the unit training prior cap. Under wrong fixed covariance, about 88% of neural decisions leave the declared box because the assumed process variance differs from training. These diagnostics establish a meaningful input shift, but they do not identify its causal contribution without separate controlled ablations.

Neural measurement counts fall from about 11 in the known treatment to about 8.2 in the fitted treatment at similar traveled distances. Routes, data quantity, parameter estimation, and later neural inputs interact. It would be premature to attribute the entire failure to any single feature or to infer that all learned planning policies fail with estimated covariance.

Across fitted runs there are 948 scheduled optimizations, 110 boundary-hit events, and no rejected fits. The mapper's whole-grid reported variance increases on 340 actual measurement updates. Re-estimation therefore changes the uncertainty model enough that fixed-kernel variance-reduction interpretations need care.

## Computational cost

Mean cumulative milliseconds per episode in the fitted treatment:

| Planner | Action selection | Mapper updates, including fits | Sum |
| --- | ---: | ---: | ---: |
| Greedy | 0.554 | 26.668 | 27.222 |
| Rollout | 42.842 | 20.719 | 63.561 |
| Neural seed 0 | 0.855 | 16.185 | 17.040 |
| Neural seed 1 | 0.726 | 15.838 | 16.564 |

Neural action selection remains inexpensive, but fitting dominates its online cost. Including mapper updates, rollout takes about 3.7–3.8 times the neural runs' time here; action-selection-only speed ratios must not be presented as total-system speedups. The neural runs also acquire fewer measurements and trigger fewer fits, which contributes to their lower cost and poorer route information. No equal-measurement-count efficiency claim is made. Timings are descriptive; short unit-test jobs also ran during the beginning of this evaluation.

## Interpretation and next modeling step

The prior neural-policy gain depends materially on the known-covariance input regime. Frozen deployment with plug-in estimates loses that gain. A future policy experiment can train on estimated-belief trajectories, with explicit normalization and sufficient training-state coverage, then freeze settings before a new test set. This study does not perform that adaptation or select a remedy using the test outcomes.

For the planned DeepKriging extension, establish a same-observation mapper benchmark with GP, coordinate-input neural, and spatial-basis neural models. Specify how each neural mapper conditions on a newly encountered field and represents uncertainty. Evaluate sparse-data prediction error, calibration, and update cost before connecting the mapper to a learned policy. This failure does not show that DeepKriging will help; it provides a concrete baseline and a robustness limitation that any extension must address.

## Reproduction and artifacts

```sh
PYTHONPATH=src .venv/bin/python -m spatial_exploration.robustness_study --output outputs/robustness-study
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

Requires the archived imitation-study selected checkpoints and manifests. Use a fresh output folder. The run refuses to overwrite a study configuration or overlap the archived source folder.

- `config.json`, `evaluation_cases.json`, `requirements-run.txt`: fixed protocol, model fingerprints, paired scenarios, and dependency versions.
- `compatibility_check.json`: reproduction of the predefined archived oracle case.
- `evaluation_metrics.csv`, `trajectories.csv`: every executed route, model parameters, prediction and true-route metrics, and costs.
- `policy_inputs.csv`: current modeled parameters and raw input-support diagnostics at each actual decision.
- `parameter_fits.csv`: all scheduled fits, accepted/rejected flags, bounds, warnings, and optimization times.
- `rollout_plans.csv`: completion values under each decision's frozen current modeled kernel.
- `summary.json`, `paired_comparisons.json`, `diagnostics.json`: means, paired conditional intervals, feasibility, fitting diagnostics, and checkpoint integrity.
- `comparison.png`: RMSE, coverage, and true-GP route variance across treatments.
- `routes_{condition}.png`: predefined scenario 0, showing all four methods and both neural seeds in each treatment.
