# Same-basis Bayesian linear control

## Question

Does replacing the small-data nonlinear mapper and initialization ensemble with analytic Bayesian linear inference on the **same spatial representation** improve reconstruction or resource-constrained exploration?

This control was motivated by the earlier [frozen-route diagnosis](crossover-study.md), where the reconstruction contrast was larger than the sampling contrast under caps (16,2). It tests one specific linear prior against a retuned neural model, rather than presuming that fewer parameters or a Bayesian label must improve performance. Observation-conditioned neural models remain a later step.

## Model

Extract the original neural representation into a shared geometry-only module without changing its calculations: two coordinates plus compact Wendland features on 4×4, 7×7, and 13×13 knot grids. Standardize each column using the known candidate-grid geometry, without measured values or field truth. The DeepKriging-style network still receives exactly these 236 columns.

The linear control appends a constant column and divides all 237 columns by `sqrt(237)`. The constant is needed because centered features alone would force every reconstructed field to have zero spatial average. With design vector ψ(s), assume:

\[
f(s)=\psi(s)^T\beta,\qquad \beta\sim N(0,\tau^2 I),\qquad
y(s)=f(s)+\epsilon,\quad \epsilon\sim N(0,0.1^2).
\]

Condition on the acquired positions and values analytically. The implementation uses the equivalent finite-feature GP kernel `K = tau² Ψ Ψᵀ` and sequential Gaussian conditioning. Report **latent** covariance without adding measurement noise. An independent coefficient-posterior calculation checks the implementation in tests.

This is Bayesian linear regression and a finite-feature GP at the same time. It does not eliminate a spatial prior or guarantee calibration under a different generating process. The mean prior marginal variance is approximately τ², but the diagonal varies spatially with feature norms. All standardized columns have the same coefficient prior variance, so the 169 finest-scale columns contribute much more total prior power than the 16 coarsest-scale columns. The Bayesian intercept also has a prior variance; it is not an unpenalized mean estimate. The network's effective regularization is different. Consequently, this comparison controls the input representation, **not every inductive bias or prior**.

For fixed τ² and known noise, posterior covariance depends on observed positions rather than values. The uncertainty-only linear policy therefore follows the same route on every field within a budget condition. This is expected behavior, not a failure to deliver observations to the mapper.

## Prespecified development and evaluation

Write `design.json` before generating tuning maps. Use eight new independent tuning fields, eight other calibration fields, and 64 new benchmark fields. Seed bases are 5,000,000 / 5,100,000 / 5,200,000; neural initialization bases are 800,000,000 / 810,000,000 / 820,000,000. These are different from previous experiments.

Give the linear and neural models **four candidate settings each**, on identical tuning observations:

- Linear coefficient prior variance τ²: 0.05, 0.2, 1, 5.
- Neural epochs: 100 or 300, crossed with Adam weight decay 0.0001 or 0.01. Keep the original architecture, three initialization members, learning rate, and variance floor.

Use endpoint observations from shared random and fixed-GP-greedy reference routes across the four physical budget conditions. Select one global setting per model by pooled held-out latent MSE over all 64 tuning endpoint cases. Identical endpoint observation sets share numerical fits (40 unique sets), while all planned endpoint cases retain equal weight. This equalizes the number of settings and the validation data, not computational effort or flexibility of the candidate families. GP retains the existing per-prefix stationary-RBF marginal-likelihood estimator and known zero mean/noise.

Selection gives **τ²=5**, and **300 epochs / weight decay 0.01** for DeepKriging. Tuning MSE is 0.81818 and 0.83106 respectively. The selected linear variance is the search boundary; values 1 and 5 have very similar tuning losses. No wider search is performed after seeing test outcomes. Neural settings differ from the previous 100-epoch/0.0001 experiment, so a change across those studies cannot be attributed to one cause or compared as a paired treatment.

Freeze model settings, then obtain one global interval-variance multiplier per model on separate own-route calibration maps. Freeze those multipliers **before benchmark generation**. Calibration pools unique acquired prefixes and held-out sites; it is a heuristic with correlated observations, not a conformal or posterior guarantee. Positive global scaling cannot change an action ranking.

The simulator remains the gradual nonstationary long/short GP mixture (lengths 0.35/0.12, spatially varying mixture weights, constant mean and marginal variance). Every model starts at site 0, pays for the starting measurement, senses sequentially, and obeys both a count cap (8 or 16) and travel cap (1 or 2). The 45 evaluation sites are unavailable for sensing. Each model uses **native latent marginal variance divided by distance** to choose among affordable unvisited sites. This is not expected integrated information gain. There are no obstacles or return-to-base costs.

Predeclare a three-by-three frozen-route crossover: reconstruct every final acquired route with all three frozen mappers. Archive predictions before loading benchmark truth in that diagnostic. Reuse native fits exactly, preserve observed values, and verify source hashes. Bootstrap differences over 64 independent fields, not individual spatial sites. The multiple intervals are descriptive and have no multiplicity adjustment.

## Closed-loop results

Lower held-out latent RMSE is better. Entries show mean error and actual resource consumption across 64 paired fields.

| Count cap | Travel cap | Fitted GP RMSE | DeepKriging RMSE | Basis Bayes RMSE |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 1 | 0.8944 | 0.8842 | 0.9107 |
| 8 | 2 | 0.7954 | 0.8815 | 0.9107 |
| 16 | 1 | 0.8944 | 0.8749 | 0.9156 |
| 16 | 2 | 0.7888 | 0.8509 | 0.9185 |

At caps (16,2):

| System | Actual readings | Actual travel | Mean cumulative prefix fitting time |
| --- | ---: | ---: | ---: |
| Fitted GP | 8.06 | 1.962 | 0.0810 s |
| DeepKriging-style | 15.39 | 1.702 | 0.6937 s |
| Basis Bayes | 16.00 | 1.320 | 0.0185 s |

Relative to GP, DeepKriging's paired RMSE difference is +0.06214, 95% field-bootstrap interval [0.02974, 0.09349]. Basis Bayes differs by +0.12967 [0.08967, 0.17075]. Under the tighter travel cap of 1, neural mean RMSE is slightly lower than GP, but the paired intervals include zero. Do not claim a uniform ordering across resource regimes.

All 768 benchmark episodes obey both caps and avoid evaluation sites. Collection uses 2,637 unique prefix fits, including 3,099 neural member fits. Benchmark collection wall time is 54.15 s, after 6.09 s tuning and 8.52 s calibration collection in the recorded run. Prefix timings exclude geometry/kernel preparation and archive/plotting work; neural independent fits are batched and amortized, while analytic fits are single-prefix. They are not isolated deployment latency measurements.

Increasing observations can worsen realized prediction error under a misspecified model; neither the mean curves nor an individual realization must decrease monotonically. Distance curves show saved prefixes of routes planned for the indicated full caps. They do not rerun a new optimized policy at every intermediate allowance.

## Same-data reconstruction and route diagnosis

At caps (16,2), rows share exactly the same acquired positions and noisy values; columns change only the frozen mapper:

| Original route | GP mapper | DeepKriging mapper | Basis Bayes mapper |
| --- | ---: | ---: | ---: |
| GP | 0.7888 | 0.7992 | 0.8204 |
| DeepKriging | 0.8434 | 0.8509 | 0.8714 |
| Basis Bayes | 0.8914 | 0.8965 | 0.9185 |

On GP-selected data, the neural-minus-GP difference is +0.01043 [-0.01790, 0.03530]; the interval includes zero. Basis-minus-neural is +0.02114 [0.00870, 0.03477]. This particular Bayesian linear prior does not improve the neural reconstruction even on shared data in this condition. It also does not establish that all linear regularization choices fail.

Holding the neural mapper fixed, replacing its own acquired observations with GP-selected observations reduces mean RMSE by 0.05171. Conversely, using neural-selected data with the GP mapper raises error by 0.05462 [0.00694, 0.10392]. The symmetric descriptive decomposition of the +0.06214 own-system gap allocates +0.05317 [0.02553, 0.08197] to sampling design and +0.00897 [-0.01350, 0.03021] to reconstruction. These are conditional arithmetic contrasts, not causal percentages. Sampling includes actual count as well as positions.

This result differs from the earlier weakly regularized neural run: in the current **retuned** experiment the larger contrast is route selection. It supports checking uncertainty and policy behavior after tuning reconstruction; it does not invalidate the earlier conditional diagnosis or isolate regularization as the cause of the change.

The linear policy tends to collect nearby sites and retain high uncertainty over most of the domain. Evaluator-only true-GP conditional held-out variance at caps (16,2) is 0.58395 for GP routes, 0.75523 for neural routes, and 0.84765 for basis routes. More readings alone do not establish better spatial coverage. Analytic uncertainty under an assumed basis prior does not make the marginal-variance/distance heuristic optimal.

## Uncertainty

| System, caps (16,2) | Native 95% coverage | Calibrated coverage | Native mean latent std | Calibrated mean std |
| --- | ---: | ---: | ---: | ---: |
| Fitted GP | 81.94% | 87.99% | 0.6275 | 0.8072 |
| DeepKriging-style | 8.85% | 96.53% | 0.0401 | 3.9478 |
| Basis Bayes | 100.00% | 98.19% | 2.2397 | 1.1769 |

The frozen variance multipliers are 1.6547, 9708.8166, and 0.27611. Their very different magnitudes expose different failures: the neural initialization spread is extremely small, while the basis prior is overly diffuse across unsensed regions. Scalar neural calibration achieves broad bands, not useful spatial ranking or informative uncertainty. GP calibration does not reach nominal coverage on these new endpoint cases either. Do not call any of these fully calibrated joint posteriors under the true field law.

In predefined example field 0, strong neural regularization yields a largely flat prediction despite spatially varying truth. This is an example, not proof that every neural fit collapses. The model was selected by fixed-reference-route reconstruction MSE, not closed-loop accuracy or uncertainty quality. That selection objective does not certify deployment as an uncertainty-driven policy.

## Interpretation and next step

The experiment rejects the easy explanation that the existing network merely needs to be replaced by same-basis Bayesian linear inference. That particular replacement is fast, supplies coherent conditional covariance under its own prior, and still performs worse here. The common representation does not make the priors equivalent. Uncertainty ranking, spatial scale allocation, mean regularization, and the local variance/distance rule remain separate issues.

The subsequent [CNP experiment](cnp-study.md) implements an observation-conditioned architecture trained across independent maps, with a new test set and a GP control learned from the same noisy offline data. It accepts the acquired context on a new map without fresh weight fitting. The later [attention extension](attention-study.md) improves same-data reconstruction while still failing to improve the complete exploration system relative to fitted GP in its larger-distance condition. Merely outputting a variance or using attention does not demonstrate useful spatial exploration.

## Reproduction and artifacts

```sh
PYTHONPATH=src .venv/bin/python -m spatial_exploration.basis_study --output outputs/basis-study
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

Use a fresh output folder; the command also writes the sibling `outputs/basis-study-crossover`. Records include prespecified design, candidate losses, selected settings, tuning endpoint indices, calibration and benchmark acquired/evaluator archives, prefix predictions, step/final metrics, paired contrasts, routes, curves, representative weights, checksums, and runtime diagnostics. The crossover evaluates 2,304 endpoints from 1,584 unique requested fits; its native diagonal matches exactly, source hashes remain unchanged, and no routes or hyperparameters change.

Validation: 72 tests pass in the full environment. System Python runs 72 tests with 28 optional tests skipped. Tests cover analytic posterior agreement, noise separation, covariance invariants, legal observations, freeze-before-test ordering, both resource caps, exact same-data crossover, and source preservation. Figures were rendered and visually inspected.
