# Observation-conditioned spatial mapping

## What changed

Implement a conditional neural process (CNP) that learns across independent maps. A frozen network receives only the current map's acquired `(x, y, value)` pairs and query positions. It predicts the new field without fitting weights on that map. This is a substantive change from the per-map DeepKriging-style network, which restarts optimization after each acquired prefix.

The core method is based on [Garnelo et al., Conditional Neural Processes, ICML 2018](https://proceedings.mlr.press/v80/garnelo18a.html), with the authors' [official implementation](https://github.com/google-deepmind/neural-processes/blob/master/conditional_neural_process.ipynb) as a reference. The present 2D noisy-sensing task, architecture, context-count channel, training distribution, and evaluation differ from the source experiments. **This is a task adaptation, not a paper reproduction.** Attention and a latent variational neural process are not implemented in this step.

## Model and uncertainty

The encoder maps each observed pair to a 64-dimensional vector using a `3 → 64 → 64 → 64` MLP with ReLU between hidden layers. Mean pooling makes the context representation invariant to observation order. For each query, concatenate that representation, the query coordinates, and `log(1+n)/log(17)` for context count n. A `67 → 64 → 64 → 2` decoder produces mean and latent standard deviation `0.02 + softplus(raw_std)`. Coordinates are mapped from [0,1] to [-1,1]. There are **17,218 trainable parameters**.

For noisy targets, minimize Gaussian negative log likelihood with variance `latent_std² + known_noise_std²`. The sensor standard deviation remains 0.1, shared with GP. Thus the reported latent variance excludes the known measurement-noise term. This separation is a modeling assumption: the learned latent variance can still include approximation error, poor conditioning, and task uncertainty. It is not guaranteed to be an exact epistemic posterior.

The network returns factorized Gaussian marginals rather than a coherent spatial joint posterior. It cannot directly supply the GP's exact hypothetical-observation covariance update or expected integrated information gain. More context need not monotonically reduce its learned variance. Scalar interval calibration cannot correct within-state spatial ranking.

## Offline data and controls

Use the existing 15×15 gradual nonstationary Gaussian-mixture simulator, with long/short scales 0.35/0.12, zero mean, unit marginal variance, and spatially changing dependence. The five deployed systems are:

1. The existing stationary single-RBF GP fitted from the current map's observations.
2. The existing DeepKriging-style mapper with frozen 300 epochs, weight decay 0.01, and three initialization members, selected in the earlier independent basis study.
3. A **training-map GP**: estimate the full-grid covariance from the same offline noisy maps available to CNP. Assume known zero mean, subtract known sensor noise, project negative eigenvalues to zero, and apply a prespecified 5% isotropic shrinkage. There is no true covariance input to this estimator or deployed mapper.
4. CNP training seed 0.
5. CNP training seed 1.

The training-map GP is necessary to distinguish the value of extra offline data from the value of the CNP architecture. It learns a fixed-grid spatial prior and does not claim off-grid generalization. CNP and this control have a substantial offline-information advantage over the two fresh-per-map baselines. Their different model families and fitting resources are not matched in every respect.

Generate **4,096 independent training maps**: 921,600 distinct noisy full-grid readings. Only noisy values enter the trainer and covariance learner; no latent field targets or simulator covariance are passed to them. Latent fields are used for later evaluation. Offline training includes labels at the spatial coordinates that will be withheld on new maps; fields are independent. This tests new **map realizations at a known geometry**, not transfer to unseen locations or a new spatial law.

At each training step, use 32 maps and a common random context count uniformly from 1 to 16. Every context includes start site 0. Each row uses either globally sampled permitted sites or samples from a local neighborhood, with equal probability; half the local neighborhoods are anchored at the deployment start. Draw 64 distinct query targets disjoint from that row's context. Contexts exclude the fixed evaluation-site mask. Training geometry is not a cost-constrained robot trajectory, and all offline readings are accounted separately from online resources.

## Selection and freeze

Write `design.json` before data generation. Seed bases are 6,000,000 for training, 6,100,000 for validation, 6,200,000 for calibration, and 6,300,000 for benchmark fields. They do not overlap previous studies. Train both predeclared seeds with Adam at 0.001, gradient norm clipping at 5, and candidate checkpoints 1,000 / 3,000 / 6,000 updates. Do not select a winning seed.

Use 32 separate validation maps and common random/fixed-GP-greedy reference-route prefixes across all resource conditions. Predictor inputs contain only each acquired prefix. Score noisy held-out target likelihood, averaging across steps within a route and then routes; every field has the same route conditions. Shared prefixes reuse predictions but keep their repeated episode weights. Choose **one checkpoint count averaged across both seeds**:

| Updates per seed | Mean validation noisy NLL |
| ---: | ---: |
| 1,000 | 1.35858 |
| **3,000** | **1.34646** |
| 6,000 | 1.34764 |

The selected seeds have validation NLL 1.34757 / 1.34535 at 3,000 updates. Selection uses likelihood rather than latent test error or closed-loop outcomes. Both full training runs through 6,000, validation, and checkpoint saving take 41.60 s in the recorded single-CPU-thread environment; data simulation is outside that timer.

On 16 other calibration maps, collect own-model trajectories and compute one global latent variance multiplier per system. Freeze weights, covariance, checkpoint choice, and interval scales **before generating 64 benchmark maps**. No test-time CNP gradients or target-dependent hyperparameter changes are allowed. Checkpoint and prior hashes are retained.

## Physical task and comparison

Every system starts at site 0 and senses sequentially. The starting reading counts. Cross caps of 8/16 readings and travel distance 1/2. The 45 fixed evaluation sites cannot be sensed. Every system maximizes native current latent marginal variance divided by Euclidean distance over affordable, unvisited permitted sites. Stop at the reading cap or when no further site is affordable. No obstacles or return-to-base cost are modeled.

This is the same uncertainty-seeking heuristic, not an integrated-variance planner or a learned policy. A better covariance model need not produce a better route under this local score. CNP means/variances can depend on observed values; the training-map GP's fixed covariance gives value-independent variance-only routes.

Predeclare a five-by-five frozen-route crossover: fit/query all five models on the exact final positions and noisy values from every route family. Preserve native results exactly, write predictions before loading latent targets, and verify source hashes. Within-row comparisons fix observations; own-system comparisons permit different actual resource use under the same upper bounds. Paired bootstrap intervals resample 64 fields, conditional on the fixed training runs. They do not quantify all training-seed uncertainty and are not adjusted for multiple comparisons.

## New-map results

Mean held-out latent RMSE, lower is better:

| Count cap | Travel cap | Fitted GP | DeepKriging | Training-map GP | CNP seed 0 | CNP seed 1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 1 | 0.9802 | 0.9692 | 0.9464 | 0.9293 | 0.9761 |
| 8 | 2 | 0.8397 | 0.9678 | 0.9459 | 0.9293 | 0.9761 |
| 16 | 1 | 0.9802 | 0.9583 | 0.9464 | 0.9232 | 0.9870 |
| 16 | 2 | **0.8188** | 0.9356 | 0.8475 | 0.9197 | 0.9885 |

At caps (16,2), CNP-minus-fitted-GP paired differences are +0.10093 [0.05483, 0.14503] and +0.16979 [0.11837, 0.22012]. The training-map GP's own-system difference is +0.02870 [-0.01707, 0.07677]. CNP does not improve the system in this resource condition. There is no claim of a universal ranking over all physical budgets; tighter-distance outcomes differ.

| System, caps (16,2) | Actual readings | Actual distance | Mean cumulative online prefix computation |
| --- | ---: | ---: | ---: |
| Fitted GP | 8.16 | 1.968 | 68.3 ms |
| DeepKriging | 15.77 | 1.679 | 656.9 ms |
| Training-map GP | 15.00 | 1.947 | 14.7 ms |
| CNP seed 0 | 16.00 | 1.131 | 7.0 ms |
| CNP seed 1 | 16.00 | 1.071 | 3.5 ms |

CNP performs frozen forwards and is fast in this run, but increased reading count does not compensate for restricted spatial coverage. Timing scopes include prefix fit/query, not offline training, simulation, rendering, or geometry setup; DeepKriging batching is amortized. Differences between the two CNP timing measurements are not a controlled architecture-speed comparison.

All 1,280 benchmark episodes obey both caps and avoid evaluation sites. There are 4,704 unique prefix states. Calibration collection takes 14.02 s and benchmark collection 50.80 s. The weights remain frozen.

## Same-observation results

At caps (16,2), give every mapper the **same GP-selected measurements**:

| Mapper | RMSE on fixed GP observations |
| --- | ---: |
| Fitted GP | 0.8188 |
| DeepKriging | 0.8657 |
| **Training-map GP** | **0.7527** |
| CNP seed 0 | 0.8775 |
| CNP seed 1 | 0.8770 |

The training-map GP improves on fitted GP by -0.06602 [-0.09424, -0.03776], about 8.1% lower mean error on these observations. CNP instead differs by +0.05879 [0.02200, 0.09563] and +0.05824 [0.02207, 0.09373]. Thus the same offline maps contain useful spatial information, but this CNP does not exploit it as effectively in reconstruction. Offline data availability alone does not explain away its deficit.

The training-map GP's own route has error 0.8475, while its reconstruction on fitted-GP-selected data has error 0.7527. Its evaluator-only true-GP route variance is 0.74628 versus 0.58956 on fitted-GP routes. Learning a useful nonstationary prior does not make the present marginal-variance/distance heuristic optimal. The fixed-GP-planner/learned-prior-mapper combination is a prespecified crossover result, not a separately optimized or newly tested hybrid policy.

CNP also has a route problem: replacing its own observations with fitted-GP-selected observations reduces mean error from 0.9197/0.9885 to 0.8775/0.8770. The two seeds have nearly equal error on shared GP data but substantially different own-system errors. Both reconstruction and uncertainty-driven sampling matter; attributing everything to one component would be unwarranted.

The full crossover contains 6,400 endpoint evaluations and 4,270 unique requested fits. Native diagonal error is exactly zero, original source hashes remain unchanged, and no route or setting is selected after inspecting test results.

## Does CNP uncertainty actually condition on observations?

At caps (16,2), native marginal 95% coverage is 81.87% for fitted GP, 6.77% for DeepKriging, 95.52% for training-map GP, and 94.38%/90.83% for the CNP seeds. Corresponding mean held-out latent standard deviations are 0.6667 / 0.0412 / 0.8660 / 0.9092 / 0.9176. CNP's broader uncertainty is more realistic than initialization disagreement, but broad coverage is not sufficient for useful action ranking.

Frozen variance multipliers are 4.3163 / 6062.8487 / 0.9070 / 0.9712 / 1.0511. Calibrated endpoint coverage is 95.56% / 94.17% / 94.44% / 93.99% / 91.63%. Calibration pools correlated sites and unique prefixes; it offers no formal new-map coverage guarantee and cannot alter the selected routes.

An additional **descriptive conditioning diagnostic**, unused for tuning, examines acquired sites. Mean RMSE against acquired noisy readings is 0.0209 for fitted GP, 0.0138 for training-map GP, and **0.5968 / 0.6579 for CNP**. Native latent standard deviation at acquired sites is approximately 0.097 / 0.099 versus **0.688 / 0.717**. This does not demand interpolating measurement noise. It shows that CNP's response to nearby actual observations is much weaker than analytic conditioning in this setup. Representative maps likewise show strongly smoothed predictions and relatively broad uncertainty near the route.

Global mean pooling compresses all observations into one query-independent representation, making loss of local detail a plausible mechanism. Architecture size, optimization, training geometry, and objective choice are other possible contributors. These diagnostics do not isolate one causal failure or prove that the CNP method generally cannot learn spatial fields.

## Interpretation and next experiment

The new implementation succeeds at **conditioning a frozen neural mapper on a new context set**, with no fresh per-map weight fitting. It does not establish a mapping or exploration advantage. Keep the negative result in the portfolio record.

The subsequent [attention-CNP experiment](attention-study.md) now implements query-dependent attention while retaining these exact development data and pooled checkpoints. [Kim et al., Attentive Neural Processes](https://arxiv.org/abs/1901.05761) motivates this direction and discusses underfitting in neural processes. The new implementation is deterministic attention-CNP, not full latent ANP. On a fresh benchmark, it improves equal-duration, same-observation reconstruction but still fails to beat fitted GP in own-route exploration. Keep the two benchmark sets distinct; this page records the original pooled experiment.

Do not assume attention repairs uncertainty, choose a model from a favorable example map, or add planning complexity before checking local conditioning and spatial uncertainty ranking. The offline covariance control demonstrates useful information in the training data, while its weak own route also motivates a later integrated-information-gain comparison under a fixed common mapper.

## Reproduction and validation

```sh
PYTHONPATH=src .venv/bin/python -m spatial_exploration.cnp_study --output outputs/cnp-study
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

Use a fresh folder; the main command also creates sibling `outputs/cnp-study-crossover`. Artifacts include offline noisy observations, learned covariance, model checkpoints, learning traces, validation contexts, selected/frozen settings, calibration/test acquired and evaluator archives, step/final metrics, full crossover predictions, source hashes, routes, curves, and conditioning diagnostics. `example_routes.png` uses predefined field 0 with separate native-uncertainty scales; it is not a selected best case.

The full environment passes **77 tests**. System Python runs 77 with 33 optional tests skipped. New tests cover context-order invariance, context-value response, query independence, frozen parameters, invalid observations, disjoint training labels, sensing-mask compliance, known-noise likelihood, offline covariance recovery/PSD behavior, both retained seeds, freeze-before-test ordering, both online budgets, exact crossover diagonal reproduction, identical observed values, and source preservation. Figures are rendered and visually checked.
