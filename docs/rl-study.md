# Learning a budgeted exploration policy

## Purpose and status

Apply a maintained implementation of an existing reinforcement-learning algorithm to the project's controlled exploration task, and compare its final mapping quality with the existing greedy planner. This is an application of Maskable PPO to a custom task, not reproduction of the CAtNIPP paper or a claim of global optimality.

DeepKriging is not included. Keep the GP mapper fixed while testing a different planner.

## Existing-method audit

Reviewed the [CAtNIPP paper](https://proceedings.mlr.press/v205/cao23b.html) and [authors' implementation](https://github.com/marmotlab/CAtNIPP) at commit `eee8c458093697523493665f024a1495fdb1f8c2`. A read-only local checkout is ignored under `external/CAtNIPP`.

The upstream implementation uses a probabilistic roadmap, a destination condition, samples along movement segments, and optionally adapts its region of interest. The provided training configuration uses Ray workers and a CUDA global network. Our current simulator uses discrete point measurements, unrestricted straight-line movement, whole-grid variance reduction, and no destination requirement. Running the upstream model without reconciling those differences would not be a controlled comparison. Full paper reproduction is deferred and remains a separate task.

For this learning baseline, use [SB3-Contrib Maskable PPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html). Invalid-action masks exclude already sampled, evaluation-only, and unaffordable locations. The maintained library supplies the PPO algorithm; the environment and evaluation harness are project code.

## Task

- Grid: 7 by 7 for a small CPU experiment. These absolute metrics should not be compared directly with the earlier 15-by-15 experiments.
- Covariance: oracle unit-variance RBF GP with known sensor noise standard deviation 0.1, as in the original baseline. This intentionally isolates planning from covariance estimation.
- Episode randomization: length scale uniform in [0.12, 0.35], initial travel budget uniform in [1.5, 2.5], and a random permitted starting site.
- Measurement sites: indices with `index % 5 == 4` are reserved for evaluation. Every other site can be observed once. All methods use the same restrictions.
- Cost: Euclidean distance, with no measurement cost, obstacles, dynamics, or return-to-base constraint.
- Action: index of the next measurement location.
- Termination: no permitted unvisited location is affordable with the remaining budget.

## Observation and reward

The policy receives the full GP posterior covariance, visited mask, current xy position, remaining budget, and known length scale. It does not receive field values, posterior mean, held-out labels, or the eventual evaluation metrics.

For fixed known covariance parameters, both covariance transitions and the variance-reduction objective are independent of observed values. Thus training can update the covariance using zero pseudo-observations: it does not need to sample or reveal a hidden field. This is an information-planning task with known model dynamics. It is not yet value-adaptive hotspot search or unknown-covariance exploration.

Per-step reward is 100 times the decrease in mean latent posterior variance over all grid sites. Set `gamma=1`, so total undiscounted reward telescopes to 100 times the reduction between the post-start and final posterior variance. Travel is a hard budget constraint; the reward is not divided by distance. The greedy benchmark retains its existing one-step reduction-per-distance rule.

Because the full transition model is known, model-based lookahead is also a natural future benchmark. Reinforcement learning is being evaluated as a way to learn a reusable policy, not assumed necessary for this simplified task.

## Training and evaluation protocol

Start with two training seeds (0 and 1), each with 32,768 transitions. Use four synchronous environments, a two-layer 128-unit policy network and value network, rollout length 256 per environment, batch size 128, five optimization epochs, learning rate 0.0003, and entropy coefficient 0.01. Train on CPU with one Torch thread. Save both the untrained and final models.

Evaluate on 24 pre-generated scenarios, using scenario seed 700000 and field seeds starting at 800000. Starts, budgets, kernels, latent fields, and potential noisy observations are paired across random, greedy, untrained PPO, and trained PPO. Every masked policy acts deterministically during evaluation. Use a fixed training budget and final checkpoint; do not select checkpoints or hyperparameters using evaluation performance. This is an initial development evaluation within the training parameter ranges, not an out-of-distribution claim.

Primary metric: final whole-grid mean GP posterior variance, matching the reward objective. Secondary metrics: latent-field RMSE and pointwise 95% interval coverage on reserved sites, travel, number of measurements, and action-selection time. A separate GP reconstructs the actual field from acquired noisy values at evaluation; its covariance is checked against the training environment's covariance.

Report every training seed, including failures to beat greedy. The scenario-0 route figure is selected by its predefined index, not by outcome. Distinguish independent evaluation scenarios from repeated evaluations of different training seeds on those same scenarios.

## Reproduction

The recorded run uses a dedicated Python 3.12 environment:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[rl]'
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python -m spatial_exploration.rl_study --steps 32768 --training-seeds 0 1 --evaluation-cases 24 --output outputs/rl-study
```

The output records the package versions, full configuration, evaluation scenarios, training logs, timing, untrained/final checkpoints, paired metrics, trajectories, and figures. Installation of the optional RL dependencies is required for the environment tests; the base test suite explicitly skips them otherwise.

## First development result — November 4, 2026

Executed the prespecified two seeds, 32,768 transitions per seed, and 24 paired evaluation scenarios. No training settings or checkpoint choice were changed after inspecting these results.

| Method | Mean final whole-grid variance | Held-out RMSE | Mean measurements |
| --- | ---: | ---: | ---: |
| Random | 0.6592 | 0.7584 | 4.50 |
| Greedy | 0.3967 | 0.5530 | 11.29 |
| PPO seed 0, untrained | 0.6738 | 0.8128 | 4.54 |
| PPO seed 0, trained | 0.5398 | 0.6805 | 7.46 |
| PPO seed 1, untrained | 0.6163 | 0.7579 | 5.33 |
| PPO seed 1, trained | 0.5050 | 0.6649 | 8.38 |

Both trained policies improve on their own untrained checkpoints in average variance, but neither beats greedy on any of the 24 scenarios under the primary variance objective. Mean travel is similar (about 1.96–1.98), while the learned policies acquire fewer measurements. This is consistent with less effective use of travel budget; it does not isolate a unique cause of the gap.

The two training runs took about 9.7 and 9.2 seconds on this machine, excluding initial dependency installation. This is a short learning baseline, not an exhaustive study of PPO's capabilities. Final model reloads reproduce the saved scenario-0 routes for both seeds. All 144 method/scenario evaluations respect their budgets, and cumulative reward matches 100 times total variance reduction up to floating-point tolerance. All 14 tests pass in the RL environment with no skipped tests.

## Interpretation and next decision

This run establishes a functioning train/save/load/evaluate pipeline and a concrete performance gap. It provides no evidence that the current learned planner improves on greedy. The state is a flattened full covariance matrix plus basic navigation features; it has no attention-based graph structure or geometric inductive bias. These and training duration are possible factors to investigate, not established explanations.

Before integrating DeepKriging, use a validation protocol to investigate planning improvements. A model-based lookahead benchmark can first check whether considering future measurements helps in this simplified known-covariance task. If pursuing PPO improvements, allocate a fixed tuning budget and separate validation scenarios; retain all seeds and obtain fresh final test scenarios after decisions are frozen. The current 24 cases are now development diagnostics.

Artifacts in `outputs/rl-study/`:

- `evaluation.png`: all scenario-level results and method means, including untrained checkpoints.
- `routes.png`: random, greedy, and both trained-policy routes for predefined scenario 0.
- `evaluation_metrics.csv` and `trajectories.csv`: paired final metrics and acquired measurements.
- `seed-0/` and `seed-1/`: untrained/final models, training metrics, and episode returns.
- `config.json`, `evaluation_scenarios.json`, `requirements-run.txt`, and `training_times.csv`: reproducibility records.
