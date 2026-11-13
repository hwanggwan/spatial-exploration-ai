# Two-step lookahead: controlled comparison

## Question

Does evaluating two future measurements improve budgeted mapping over the existing one-step greedy rule, and what is the planning-time cost?

This is a finite-horizon model-based heuristic, not another trained model and not a globally optimal solver. The GP mapper, known covariance assumptions, available measurement sites, and travel constraint are unchanged.

## Decision rule

At the current state, let C be the latent GP posterior covariance over N grid sites and let epsilon be the known observation-noise variance. For a feasible first site a, condition the covariance as

`C_a = C - C[:, a] C[a, :] / (C[a, a] + epsilon)`.

Then enumerate feasible second sites b. Each ordered pair must fit in the remaining travel budget, neither site may have been measured, and neither site may be reserved for evaluation. Compute the second variance reduction using C_a, so correlated measurements are not double-counted.

For each affordable pair, score:

`[mean_variance(C) - mean_variance(C_ab)] / [distance(current, a) + distance(a, b)]`.

Choose the pair with the largest score. If a first move leaves no feasible second move, evaluate that terminal one-move sequence using the original one-step score. Otherwise, a first move is evaluated through its best affordable second move. Stable array order breaks exact ties.

Execute only a, acquire the actual measurement, and plan again. The previously proposed b is not committed. In the known-covariance setting, hypothetical updates require measurement locations only, so this calculation uses no future observed values or hidden field values.

At depth one, the helper reproduces the original greedy reduction-per-distance criterion. At depth two, it exhaustively enumerates the feasible ordered pairs, with roughly quadratic candidate growth. It does not evaluate the rest of the remaining-budget trajectory. Consequently, greater short-horizon density need not give a better final map.

## Protocol

Use the exact 24 scenarios saved in `outputs/rl-study/evaluation_scenarios.json`: the same starts, budgets, length scales, latent fields, and potential noisy observations. Use the 7-by-7 oracle-GP environment. Reload both archived final PPO checkpoints; do not retrain or select between them.

Preserve `outputs/rl-study/` and write new artifacts to `outputs/lookahead-study/`. Record SHA-256 hashes of the reused checkpoints. Verify the newly evaluated random, greedy, and PPO metrics against the archived results; the experiment stops if their numerical outcomes change. Timing is remeasured and is not expected to be identical.

The two-step density rule was specified before viewing the new results. These are already-seen development scenarios, not a new held-out test set.

## First result — November 4, 2026

All values below are means across the 24 paired scenarios.

| Method | Final whole-grid posterior variance | Held-out RMSE | Measurements | Distance traveled |
| --- | ---: | ---: | ---: | ---: |
| Random | 0.6592 | 0.7584 | 4.50 | 1.9819 |
| Greedy | 0.3967 | 0.5530 | 11.29 | 1.9625 |
| Two-step lookahead | 0.3847 | 0.5438 | 12.04 | 1.9894 |
| PPO, training seed 0 | 0.5398 | 0.6805 | 7.46 | 1.9689 |
| PPO, training seed 1 | 0.5050 | 0.6649 | 8.38 | 1.9792 |

Relative to greedy, mean final variance is 3.05% lower, with mean paired difference -0.01209. Lookahead has lower final variance in 15 scenarios and higher final variance in 9, with no ties under the recorded numerical tolerance. Its slight aggregate improvement is not a guarantee of an advantage on every instance.

Mean cumulative action-selection time over one complete exploration is approximately 0.154 ms for greedy and 6.428 ms for lookahead, about 42 times higher in this single run. These are small, one-pass CPU timings, not a rigorous hardware benchmark. Physical travel is still the only budgeted cost. Computational time is measured separately, and performance at larger grid sizes has not been evaluated here.

The budget upper bound is paired, but actual travel and measurement counts differ. The improvement therefore includes how effectively a strategy uses the available budget; this is not a comparison at an exactly equal realized path length or measurement count.

## Verification

- All 20 tests pass in the project's RL environment.
- Sequential hypothetical reductions match an independent joint two-observation GP calculation.
- Depth one matches the original greedy scoring rule.
- Planning leaves the actual covariance and masks unchanged.
- A terminal one-move fallback works when a second measurement is unaffordable.
- All archived baseline and trained-PPO outcomes reproduce under the new evaluation harness.
- No executed route or hypothetical selected pair exceeds its budget.
- The 265 actual lookahead decisions include 24 terminal one-move plans. The largest first-step search evaluates 1,482 feasible sequences.

## Reproduction and artifacts

Requires the project's RL environment and archived checkpoints from the first PPO run:

```sh
PYTHONPATH=src .venv/bin/python -m spatial_exploration.lookahead_study --reference outputs/rl-study --output outputs/lookahead-study
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

- `evaluation.png`: all methods' scenario-level outcomes and means.
- `paired_comparison.png`: lookahead-minus-greedy differences for each scenario; negative favors lookahead.
- `routes.png`: all compared routes on predefined scenario 0.
- `lookahead_plans.csv`: proposed first/second sites, scores, expected gains, distances, and sequence counts at each actual decision.
- `evaluation_metrics.csv`, `trajectories.csv`, `summary.json`, and `paired_comparison.json`: raw and descriptive results.
- `config.json` and `evaluation_scenarios.json`: decision rule, checkpoint provenance, and shared evaluation cases.

The result supplies a modest model-based improvement over greedy on these cases and a stronger target for learned policies. Validate on fresh scenarios before claiming general improvement. Keep uncertainty-model extensions such as DeepKriging separate from this planner comparison.
