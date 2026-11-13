# Exact small-problem planning diagnostic

## Question

Is greedy already close to the best possible route, or does the two-step heuristic fail to use available improvement? Obtain a globally optimal reference on deliberately small discrete problems before expanding model complexity.

## Objective and exactness

Minimize final whole-grid mean latent posterior variance subject to the travel budget, permitted sensing sites, and no repeated measurements. This is the final objective, not information gain divided by distance. The GP covariance, process variance, mean, and sensor noise are known. With fixed covariance, no observed or hidden field values are needed to evaluate any route's posterior variance.

Starting with the free initial measurement, enumerate every feasible non-repeating route prefix using depth-first search. For each next measurement, update the covariance exactly with Gaussian conditioning. Discard only revisits and moves that exceed the remaining budget. Every feasible final route is included among these prefixes. Compare final objectives across the entire search, treating differences below 1e-12 as numerical ties and preferring the shorter route.

The selected route is therefore optimal within the specified finite candidate set, budget tolerance, model, and numerical objective tolerance. It is not a claim of continuous-space or real-world optimality. A configurable prefix-count cap raises an exception if reached; interrupted searches never return a result labelled optimal.

Tests independently enumerate permutations and evaluate their posterior variances with a joint batch GP calculation, checking both the optimum and enumeration counts. Tests also cover monotonicity in budget, terminal cases, mask preservation, and explicit failure on an incomplete search.

## Prespecified small cases

Use a 4-by-4 grid in the unit square, with grid spacing 1/3. As before, reserve indices where `index % 5 == 4` for evaluation; this leaves 13 permitted sensing sites. Starting position is measured at no travel cost. Only endpoint measurements are taken.

Take the full cross product of:

- Starts: indices 0, 5, and 15 (two corners and an interior point).
- RBF length scales: 0.12, 0.23, and 0.35.
- Travel budgets: 0.75, 1.10, and 1.45.

This produces 27 deterministic configurations. Process variance is 1 and sensor noise standard deviation is 0.1. There are no obstacles, return-to-base condition, measurement cost, or robot dynamics. Compare exact optimization, one-step greedy, and the existing two-step density heuristic. The 7-by-7 PPO models have a different action/observation dimension and are not evaluated on this grid.

These factors were fixed before viewing results. Cases are a small factorial diagnostic, not independent random field realizations or a final statistical test set. No field is sampled, so realized field RMSE is not reported. Under the correctly specified GP, the variance objective is expected mapping error; it is distinct from error for any one realized field.

## Result — November 4, 2026

All 27 searches complete. Across them, 1,899 feasible route prefixes and 1,407 terminal routes are evaluated; the largest case has 202 prefixes. At most four additional measurements fit in the selected budgets.

| Method | Mean final variance | Cases matching optimum | Mean case-level excess variance over optimum | Mean fraction of optimal information recovered |
| --- | ---: | ---: | ---: | ---: |
| Exact optimum | 0.6510 | 27 / 27 | 0% | 100% |
| Greedy | 0.6637 | 20 / 27 | 2.45% | 95.81% |
| Two-step density lookahead | 0.6637 | 20 / 27 | 2.45% | 95.81% |

For each case, excess variance is `(V_method - V_optimal) / V_optimal`. The table averages these case-level percentages; it is not the percentage difference between the two overall mean variances. Optimal information recovered is `(V_initial - V_method) / (V_initial - V_optimal)` for each case. The worst excess variance is 10.73%.

Greedy and the current two-step method take identical routes in all 27 cases. Break down greedy's results by the prespecified length-scale factor:

| Length scale | Cases matching optimum | Mean case-level excess variance |
| --- | ---: | ---: |
| 0.12 | 9 / 9 | 0% |
| 0.23 | 9 / 9 | 0% |
| 0.35 | 2 / 9 | 7.35% |

The gap is concentrated in the strongest-correlation settings of this small suite. This supports neither a general claim that greedy is optimal nor a claim that long-horizon planning has no value.

## A concrete scoring failure

Post hoc, select the first failing case where the entire exact optimum uses at most two additional measurements. This is case 6: start 0, length scale 0.35, budget 0.75. The selection is for diagnosis, not to represent the average outcome.

| Route | Additional measurements | Distance | Total variance reduction | Reduction / distance | Final variance |
| --- | ---: | ---: | ---: | ---: | ---: |
| Greedy/lookahead: 0 → 5 | 1 | 0.4714 | 0.1602 | **0.3397** | 0.7132 |
| Exact optimum: 0 → 1 → 5 | 2 | 0.6667 | **0.2230** | 0.3346 | **0.6503** |

Moving diagonally to site 5 has slightly higher information per distance but leaves too little budget for another measurement. Moving through site 1 fits within the budget and collects an extra useful observation. The two-step heuristic evaluates this two-move route but rejects it because its density score is slightly smaller. In this counterexample, extending the planning horizon is not required to see the optimum: the density ranking itself disagrees with minimizing final variance under the budget.

The lack of measurement cost is substantive here. Intermediate measurements are free apart from their travel, so a feasible path that samples an intermediate site may outperform a shorter direct path. The result follows the project's stated cost model and should not be generalized to sensors with substantial sampling time or energy costs.

## Interpretation and limits

- Greedy is exactly optimal in many of these tiny problems, leaving no possible improvement there under the same assumptions.
- There is verified improvement room in other cases. The current two-step density rule can miss it because it optimizes a local ratio rather than the final-budget objective.
- Grid spacing is coarser than in the 7-by-7 study. The same absolute length scale gives a different correlation between neighboring candidates, so these optimality gaps cannot be transferred to the larger task.
- This is a model-based discrete optimum with known covariance. Unknown covariance, noisy parameter learning, moving fields, and additional physical costs can change the problem.
- Candidate counts and maximum route depths are deliberately small. Completing this search does not make exhaustive search a practical general solution for the larger grids.
- Any future planner change should be evaluated under a prespecified objective and validation protocol. The observed failure suggests testing a remaining-budget or terminal-variance value, without assuming it will improve every setting.

## Reproduction and artifacts

```sh
PYTHONPATH=src .venv/bin/python -m spatial_exploration.exact_study --output outputs/exact-study
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

The exact solver and its tests use NumPy and do not require the optional RL libraries. The full project suite passes 26 tests in the RL environment.

- `optimality_gaps.png`: all case-level gaps and comparison with the exact optimum; greedy/lookahead curves coincide.
- `routes_case_0.png`: the predefined first case.
- `largest_gap_routes.png`: the largest observed greedy-gap case, explicitly labelled as a post-hoc diagnostic.
- `density_counterexample.png` and `.json`: the short-horizon density-versus-final-objective counterexample above.
- `metrics.csv`: every method's objective, gap, information fraction, route, travel, and runtime.
- `search_certificates.csv`: completed-search flag, prefix/terminal counts, maximum depth, and runtime for every case.
- `cases.json`, `config.json`, and `summary.json`: factors, numerical tolerances, work limit, and descriptive summaries.
