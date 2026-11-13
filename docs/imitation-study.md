# Behavioral cloning of exploration planners

## Question

Can a neural policy retain the rollout teacher's exploration quality while reducing online planning cost? A strong planner does not automatically make an effective teacher: copying a discrete action is a different objective from minimizing final mapping variance, and a student's mistakes change the states it subsequently encounters.

This is a custom-task implementation of masked behavioral cloning, not a new algorithm or a reproduction of a particular robotics paper. The teacher is the project's greedy-completion rollout; the comparison teacher is its original gain-per-distance greedy policy. PPO remains an archived custom-task baseline. The GP mapper and physical exploration task are unchanged.

## Common inputs and matched actors

Both cloning models use exactly the same observation arrays and feasible-action masks. At each collected state, query both teachers and store their action labels separately. Alternate rollout and greedy collection behavior by episode, using a predetermined 50/50 schedule. Thus teacher comparisons change labels while holding the supervised input distribution fixed.

Use the same flattened observation as PPO: full posterior covariance, visited flags, current coordinates, remaining budget, and known RBF length scale. The actor is a two-hidden-layer MLP with 128 units per layer and Tanh activation, followed by 49 action logits. These sizes and activations match the archived PPO actor; cloning has no critic. Both teachers' models start with identical weights for each seed, and share training batches, optimizer settings, and epoch budget.

Apply the exact feasible-action mask before cross entropy and deterministic deployment argmax. The mask enforces the physical budget, reserved sites, and no revisits. It does not rank feasible actions. No teacher fallback, hidden field, future measurement value, or search is used during student deployment. The known fixed GP still performs the mapping updates; this study learns the planner rather than replacing the GP with a neural mapper.

The trained policy approximates the teacher's action, not the teacher's complete value function. Different actions can have similar or identical route value, so exact action agreement alone is not a measure of exploration quality. States with only one feasible action are included in reported action accuracies.

## Frozen protocol

- Grid: 7 by 7, known unit-variance RBF covariance, noise standard deviation 0.1.
- Starts uniformly sampled from permitted sites; length scales uniform on [0.12, 0.35] and travel budgets uniform on [1.5, 2.5].
- Training: 1,024 complete episodes; scenario RNG seed 910000, episode seed range starting at 1100000.
- Validation: 128 complete episodes; scenario RNG seed 920000, episode seed range starting at 1200000.
- Test: 96 new paired episodes; scenario RNG seed 930000, field/noise seeds starting at 1300000.
- Assert that split parameter triples and seed ranges are disjoint from one another and from the archived 24 development cases. Never split adjacent states from one episode across training and validation.
- No latent fields are sampled for training or validation: covariance and teacher actions do not depend on sensor values under this fixed-kernel task.
- Adam learning rate 0.0003, batch size 128, 40 epochs, CPU with one PyTorch thread, seeds 0 and 1 for each teacher.
- Choose each checkpoint by minimum validation masked cross entropy within the fixed epoch budget. Persist all selected checkpoint hashes before any test evaluation. Report both seeds; do not choose a winner using test results.
- Evaluate random, greedy, two-step density lookahead, rollout, four cloned policies, and both archived final PPO policies on identical test cases and potential sensor values.

The architecture is matched, but PPO and cloning do not have equal training data or compute budgets. Teacher queries involve expensive planning; report that cost in addition to neural optimization and deployment. A comparison of these runs is not an equal-resource ranking of learning algorithms.

## Distribution shift and uncertainty

After all models are frozen, replay each student's own test trajectory and query its teacher solely for diagnosis. Report action agreement on student-visited states and verify the replay reproduces the original final variance. Teacher labels do not influence student actions and are not used for test-time adaptation. Teacher-query time in this audit is excluded from student deployment timing and recorded separately.

This measures behavior outside the teacher/mixture collection trajectories; it does not solve the distribution shift. Dataset aggregation on student-visited **training** states is a possible later experiment. The distinction between standard supervised imitation and policy-induced state distributions is discussed in [Ross, Gordon, and Bagnell (2011)](https://proceedings.mlr.press/v15/ross11a.html). DAgger is not implemented here.

Primary metric: final whole-grid mean latent posterior variance. Also report held-out latent RMSE, descriptive coverage, actual travel, measurement count, and cumulative online action-selection time. A lower GP variance implies lower model-based expected squared error under the matched model, not a guaranteed lower realized RMSE on each field.

Bootstrap paired scenario-level variance differences with 5,000 resamples and seed 950000, within each fixed trained model. These intervals reflect variation across the 96 test configurations, conditional on that checkpoint; they do not measure uncertainty across training seeds or cover algorithm selection. Individual grid sites and adjacent trajectory states are not independent experimental replications.

## Results — November 4, 2026

Training collection yields 10,386 common states and validation yields 1,299. The teachers agree on 55.50% and 53.19% of actions, respectively. Collecting both labelled datasets takes 80.07 seconds, including 76.77 seconds of rollout teacher queries. Every actor has 337,073 parameters. Each training run takes 5.36–5.70 seconds, excluding shared dataset generation and initial/final diagnostics.

| Teacher | Seed | Selected epoch | Initial validation action accuracy | Selected validation action accuracy | Agreement on student-visited test states |
| --- | ---: | ---: | ---: | ---: | ---: |
| Greedy | 0 | 38 | 5.54% | 84.76% | 90.02% |
| Greedy | 1 | 39 | 6.77% | 85.45% | 89.93% |
| Rollout | 0 | 39 | 6.39% | 54.58% | 40.22% |
| Rollout | 1 | 39 | 7.24% | 54.12% | 47.89% |

These accuracy columns use different state distributions and should not be read as a paired causal estimate of distribution-shift damage. Rollout's actions are harder for this flat MLP to reproduce, and student-visited agreement is limited. Equally good alternative actions and states with forced single actions also affect exact-match accuracy. Assess the deployed trajectory objective directly:

| Method | Mean final variance | Mean held-out RMSE | Held-out 95% coverage | Measurements | Actual travel | Mean planning / episode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random | 0.62053 | 0.75439 | 96.06% | 4.958 | 2.005 | 0.037 ms |
| Greedy | 0.37287 | 0.54319 | 95.37% | 11.719 | 1.994 | 0.172 ms |
| Two-step density | 0.36402 | 0.54424 | 94.56% | 12.052 | 1.995 | 6.632 ms |
| Rollout teacher | **0.34519** | 0.53033 | 94.33% | 11.208 | 2.041 | 55.840 ms |
| Greedy clone, seed 0 | 0.37788 | 0.54974 | 96.06% | 11.542 | 1.992 | 0.864 ms |
| Greedy clone, seed 1 | 0.37976 | 0.54456 | 95.95% | 11.552 | 1.994 | 0.741 ms |
| Rollout clone, seed 0 | 0.35785 | **0.51990** | 95.95% | 11.385 | 1.994 | 0.715 ms |
| Rollout clone, seed 1 | 0.35756 | 0.52322 | 95.25% | 11.115 | 1.995 | 0.690 ms |
| Archived PPO, seed 0 | 0.53039 | 0.64984 | 95.83% | 7.073 | 2.002 | 1.313 ms |
| Archived PPO, seed 1 | 0.48968 | 0.62596 | 95.49% | 8.281 | 1.996 | 1.341 ms |

Each row uses all 96 new paired test cases. The neural seeds are reported separately rather than selecting one using test performance. All methods obey the case-specific budget cap; actual travel differs. Single-pass cumulative action-selection timings exclude mapping updates, field evaluation, rendering, and teacher queries made only for the later audit. They are local CPU measurements rather than a controlled performance benchmark or large-grid scalability result.

Both rollout clones improve mean final variance over greedy by **4.03% and 4.11%**. They beat greedy in 67 and 68 of the 96 individual cases, respectively. The paired mean variance differences and conditional scenario-bootstrap 95% intervals are:

- Seed 0 versus greedy: -0.01502, interval [-0.02012, -0.00995].
- Seed 1 versus greedy: -0.01531, interval [-0.02041, -0.01018].
- Seed 0 versus rollout teacher: +0.01267, interval [+0.00802, +0.01742].
- Seed 1 versus rollout teacher: +0.01237, interval [+0.00830, +0.01654].

The clones' mean final variance is **3.67% and 3.58% higher than their rollout teacher**, while their recorded online action selection is about **78 and 81 times faster**. The neural policies are still slower than the extremely cheap greedy rule. Teacher planning cost has moved offline rather than disappeared.

The greedy clones are slightly worse than their own teacher, despite higher action accuracy. Under matched states, architecture, initial weights, training batches, and budgets, the rollout-labelled clones produce lower mean final variance than the greedy-labelled clones. This supports a useful teacher effect in this initial experiment; it does not establish general superiority of cloning over PPO under equal resources.

Realized RMSE happens to be lower for the rollout clones than for the rollout teacher in these sampled fields, although their whole-grid posterior variance is higher. That does not make them better optimizers of the stated objective or imply lower expected squared error. Keep the variance objective, finite-sample RMSE, and measurement-site subset separate.

The remaining teacher gap and limited student-state agreement motivate investigating training-state coverage or value-aware targets. They do not establish which mechanism caused the gap. Any new variant should be chosen on training/validation data and assessed on a new test set; these 96 cases have now been used for reporting and diagnosis.

## Validation and reproduction

Tests check the masked categorical loss against a hand calculation, zero gradient through invalid logits, rejection of infeasible teacher labels, episode-split separation, common-state data persistence, checkpoint action preservation, learning on a separable synthetic problem, and feasible student deployment without teacher fallback.

All 36 tests pass in the full RL environment. System Python passes 26 tests and skips the 10 optional RL/imitation tests. Post-run checks verify selected checkpoints are unchanged from their pre-test hashes, selection follows minimum validation loss, teacher clones share initial weights per seed, and actor parameter shapes match the archived PPO actor exactly. Stored training/validation labels are feasible and all test budgets hold.

```sh
PYTHONPATH=src .venv/bin/python -m spatial_exploration.imitation_study --output outputs/imitation-study
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

Use a fresh output folder for each run. The script refuses to overwrite an existing study configuration or overlap the archived PPO input folder. The optional RL dependencies and original PPO checkpoints are required.

- `config.json`, `*_cases.json`, `requirements-run.txt`: fixed protocol, complete split identities, and dependencies.
- `train.npz`, `validation.npz`, `dataset.json`: common inputs, masks, teacher labels, episode IDs, collection cost, and fingerprints.
- `*.episodes.csv`: collection behavior and complete routes by episode.
- `{teacher}-seed-{seed}/`: untrained, selected, and final checkpoints and epoch histories.
- `training_summary.json`, `frozen_models.json`, `training.png`: training/validation performance and selected-checkpoint provenance.
- `test/evaluation_metrics.csv`, `trajectories.csv`, `summary.json`: all frozen-model outcomes and actual routes.
- `test/evaluation.png`, `routes.png`: all methods' metrics and a predefined subset of routes for scenario 0, without choosing by performance.
- `test/student_state_audit.csv`, `results.json`: student-state diagnostics, paired intervals, training cost, and test results.
- `tradeoff.png`: final variance and realized RMSE against online action-selection time; both neural seeds are shown.
