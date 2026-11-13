# Greedy-completion rollout

## Motivation and decision rule

The exact diagnostic exposed a mismatch between information per distance and the final-budget objective. A slightly less efficient route can collect more total information before the budget runs out. Test a policy-improvement baseline that evaluates complete feasible continuations, holding the mapper, covariance, measurement rules, and physical budget fixed.

At each state, enumerate **every** feasible next measurement `a`. Simulate that measurement, then follow the existing deterministic gain-per-distance greedy policy until no further measurement is affordable. Rank the completions by their final whole-grid mean latent posterior variance:

`Q_rollout(state, a) = trace(C_after_a_and_greedy_completion) / number_of_grid_sites`.

Choose the action with the lowest value, execute **only its first measurement**, and repeat the calculation at the next state. There is no first-action shortlist or tail-depth cutoff. The hypothetical route is an evaluation of one greedy continuation after each first action; this is not enumeration of all complete routes.

Covariance is already conditioned on all current observations. Every hypothetical measurement uses the exact rank-one GP covariance update, with the same sensor noise. The known fixed kernel makes covariance independent of observed values, so no future measurements or hidden field values enter planning. All grid sites contribute to the objective, including reserved evaluation sites that cannot be measured.

Always include greedy's own first action. Evaluate that completion first and retain it unless another completion reduces final variance by more than 1e-12. The base greedy rule breaks score ties using the lowest candidate index, as before. With full feasible actions, the fixed deterministic model, and exact arithmetic, rollout has a policy-improvement property relative to its base: greedy's continuation is always an available alternative, and further rollout replanning cannot worsen the continuation's value. Numerical checks use a 1e-10 tolerance. This does **not** guarantee the global optimum or an improvement in realized field RMSE.

This follows the established rollout construction; it is a model-based planning implementation, not a new learned AI model or a paper reproduction. See [Bertsekas, constrained rollout and policy improvement](https://arxiv.org/abs/2002.07407). PPO's reward was already aligned with total variance reduction (`gamma=1`); no PPO reward change or retraining is made here.

## Protocol and provenance

1. Reuse the 27 prespecified 4-by-4 exact-diagnostic cases, including the counterexample that motivated this change. Recompute exact optima and compare exact/greedy/two-step/rollout. Verify archived objectives, travel, measurement counts, and exact enumeration counts reproduce.
2. Reuse the 24 archived 7-by-7 development cases, fields, sensor values, starts, and budget caps. Compare random, greedy, two-step density lookahead, rollout, and both archived final PPO seeds. Record checkpoint SHA-256 fingerprints. Verify all archived baseline objectives and mapping metrics reproduce.

The original runs are preserved. Outputs are written under `outputs/rollout-study/exact` and `outputs/rollout-study/grid7`. These are reused development cases, not a newly held-out test set. Do not interpret the results as unbiased generalization estimates after method development.

All methods know the same unit-variance RBF covariance and noise standard deviation 0.1. Each site can be measured once; the starting measurement is free. Observations occur only at selected endpoints. There are no obstacles, return-to-base requirement, measurement-time cost, or dynamics. Budget caps are identical by case; actual travel and measurement counts can differ.

## Small-case results — November 4, 2026

All 27 exhaustive searches complete and reproduce the original reference: 1,899 feasible prefixes, 1,407 terminal routes, and at most 202 prefixes in a case.

| Method | Mean final variance | Cases matching optimum | Mean case-level excess variance | Worst excess variance |
| --- | ---: | ---: | ---: | ---: |
| Exact | 0.650970 | 27 / 27 | 0% | 0% |
| Greedy | 0.663723 | 20 / 27 | 2.4494% | 10.7342% |
| Two-step density | 0.663723 | 20 / 27 | 2.4494% | 10.7342% |
| Rollout | 0.651166 | 26 / 27 | 0.0475% | 1.2830% |

Percentages average `(V_method - V_optimal) / V_optimal` over cases; they are not ratios of overall means. Rollout recovers a mean 99.9481% of the attainable optimal information reduction. Its remaining nonoptimal case is case 17: start 5, length scale 0.35, budget 1.45. Final variance is 0.418719 versus optimum 0.413415. This directly demonstrates the limit of using a greedy tail.

In the earlier density counterexample (case 6), rollout chooses `0 -> 1 -> 5`, matching the optimal final variance 0.650317. Greedy and two-step density still choose `0 -> 5`, finishing at 0.713207. The motivating failure is fixed under the unchanged cost model.

## Larger development-case results

All values below average over the same 24 archived cases. Planning time is the sum of action-selection calls across an entire exploration episode, averaged across episodes; it is **not** time per decision or training time.

| Method | Whole-grid final variance | Held-out latent RMSE | Held-out 95% coverage | Measurements | Actual travel | Planning / episode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random | 0.65923 | 0.75838 | 95.83% | 4.500 | 1.982 | 0.033 ms |
| Greedy | 0.39675 | 0.55302 | 97.22% | 11.292 | 1.962 | 0.163 ms |
| Two-step density | 0.38466 | 0.54376 | 98.15% | 12.042 | 1.989 | 6.396 ms |
| Rollout | **0.36878** | 0.55168 | 96.30% | 11.000 | 2.022 | 53.073 ms |
| PPO seed 0 | 0.53978 | 0.68046 | 96.30% | 7.458 | 1.969 | 1.427 ms |
| PPO seed 1 | 0.50502 | 0.66487 | 93.98% | 8.375 | 1.979 | 1.352 ms |

Rollout reduces the overall mean final variance by **7.05% relative to greedy**, improving all 24 cases. Relative to two-step density, it reduces the mean by **4.13%**, improving 17 cases and worsening 7. Its policy-improvement property applies to the greedy base, not to every other planner. Both archived trained PPO seeds remain worse on the variance objective in all cases; they have not been retrained for this comparison.

Rollout uses about 325 times greedy's and 8.3 times two-step's cumulative planning time in the recorded run. The absolute mean is about 53 ms per episode on this small grid. These are single-pass local CPU timings, including Python action dispatch and excluding GP updates in the real environment, field evaluation, and rendering. They are descriptive, not a controlled hardware benchmark or evidence of large-grid scalability. The final recorded evaluation ran without concurrent test jobs; runtime varies between runs.

There are 240 actual rollout decisions, up to 39 feasible first actions at a decision, and 31,905 simulated measurements across all completions. Every selected completion is no worse than its current greedy continuation. Every executed final variance is no worse than each earlier selected completion for that episode, within numerical tolerance. All methods obey their budget caps, and the reward identity remains intact.

## Why the RMSE result needs separate interpretation

Rollout's mean held-out RMSE is almost unchanged from greedy and slightly worse than two-step lookahead, despite its lower variance objective. The primary objective averages posterior variance over the entire grid; RMSE uses only the nine reserved sites and one realized field/noise sample per configuration. These are different metrics and domains. Under the matched GP, latent variance measures expected **squared** prediction error, not the realized RMSE of an individual map or the mean of square roots.

For diagnosis, final mean posterior variance on just the held-out sites is also recorded: greedy 0.39292, two-step 0.37895, rollout 0.36464. Thus the model-implied expected squared error improves on that subset too, while this finite sample's realized RMSE does not rank the methods identically. This is not evidence that the deployed prediction error must improve. Assess realized error and coverage across a larger, freshly held-out field set before making a general claim.

## Validation and next AI milestone

Independent tests compute batch GP posteriors and all greedy completions to check candidate ranking and final variance without reusing the planner's rank-one update. Additional checks cover the density counterexample, repeated policy improvement, feasible/unique/permitted routes, deterministic ties, input preservation, and terminal/invalid states. The full RL-environment suite passes 31 tests; the system-Python suite passes the available tests with five optional RL tests skipped.

The next learned-policy experiment can use rollout as a teacher for imitation learning. Generate training cases separately from these development cases, include states visited by the learned policy to examine distribution shift, compare with equal-architecture PPO and a supervised greedy teacher, and report both training-data generation cost and online speed. Freeze model choices before a new test set. Improving the teacher on known covariance does not validate DeepKriging or solve unknown-covariance mapping; that remains a separate mapper experiment.

## Reproduction and artifacts

```sh
PYTHONPATH=src .venv/bin/python -m spatial_exploration.rollout_study --output outputs/rollout-study
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

The study requires the three archived source runs; the CLI accepts alternate source paths. To run only the small comparison without RL dependencies:

```sh
PYTHONPATH=src python3 -m spatial_exploration.exact_study --include-rollout --output outputs/rollout-exact-only
```

- `config.json`, `comparison.json`: method definition, provenance, fingerprints, results, and validation diagnostics.
- `exact/optimality_gaps.png`, `exact/density_counterexample.png`: all optimality gaps and the motivating case including rollout.
- `exact/metrics.csv`, `exact/search_certificates.csv`: routes, exact references, complete-search counts, and timings.
- `grid7/paired_comparison.png`, `.csv`: all scenario-level variance differences against greedy, two-step, and PPO.
- `grid7/evaluation.png`, `grid7/routes.png`: variance/RMSE comparison and the predefined scenario 0.
- `grid7/evaluation_metrics.csv`, `trajectories.csv`: final outcomes and actual measurement sequences.
- `grid7/rollout_plans.csv`: each selected hypothetical completion, greedy continuation, final values, candidate count, and simulated-measurement count. Selected hypothetical routes may change after replanning.
