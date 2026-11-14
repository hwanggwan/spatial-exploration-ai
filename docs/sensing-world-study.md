# Learned world model for active sensing

The robot knows the three-room floor plan, including which movements through doors are legal. It learns a model of the sensor field and uses that model to imagine the effects of possible measurements. The simulator's field is static. The modeled transition is the update to the robot's belief after taking a reading.

## Model and planning

Four spatial modes represent combinations of weak or persistent cross-wall dependence and two correlation scales. Each mode fits 13 nonnegative weights by least squares against empirical covariance from noisy, fully observed training simulations, subtracting the known sensor noise variance from the empirical diagonal. The basis comprises identity and twelve heat kernels that span four spatial scales on three graphs with different wall conductances. Training uses simulator labels to group examples by mode. It uses 512 paired field draws per training layout over six layouts, for 6,144 noisy maps across both regimes. The layouts, field seeds and noisy measurements used for testing are separate. The test modes come from the same simulation family, and the planner starts with equal weights over modes.

For each candidate location, the current mixture predicts a distribution of noisy readings. A new reading updates each spatial component and its probability. The planner first shortlists five legal sites by moment-matched one-step variance reduction per walking edge. For each shortlisted site it integrates over three possible readings per component using Gaussian quadrature and, for each resulting belief, chooses the best feasible continuation of one additional reading. The continuation uses a covariance moment match to estimate its reduction. The two-stop reward is cumulative mean grid variance reduction divided by travel. If no second reading fits the budget, the planner evaluates just the first. It executes only the first move and replans after the simulated sensor responds.

The matched GP two-step uses the same shortlist and objective with a single frozen wall-aware ridge covariance. The previous GP rollout uses that same ridge mapper and greedy completions to the full available horizon. The two-step reward can leave some travel budget unused after reaching the observation cap. The learned model uses covariance estimation and Bayesian conditioning, with no neural network.

## Held-out results

There are 64 paired fields per dependence regime and budget, over two unseen floor plans and four starts. The first reading counts towards the cap of 16. RMSE uses latent values on 28 fixed sites unavailable for sampling. Every policy uses the same walking and measurement caps. Counts and distances below are realized averages.

| Cross-wall dependence | Walk cap | Learned RMSE | GP two-step RMSE | GP rollout RMSE | Learned readings / edges | GP rollout readings / edges |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Weak | 18 | 0.701 | 0.743 | 0.751 | 14.78 / 17.70 | 15.50 / 18.00 |
| Weak | 26 | 0.691 | 0.732 | 0.687 | 16.00 / 19.31 | 16.00 / 26.00 |
| Persists | 18 | 0.712 | 0.744 | 0.740 | 14.83 / 17.73 | 15.50 / 18.00 |
| Persists | 26 | 0.702 | 0.735 | 0.678 | 16.00 / 19.31 | 16.00 / 26.00 |

At 18 edges, learned minus GP rollout RMSE is −0.050 [−0.073, −0.027] for weak cross-wall dependence and −0.028 [−0.052, −0.007] for persistent dependence. At 26 edges the difference is +0.004 [−0.019, 0.026] and +0.024 [0.002, 0.046], respectively. These are unadjusted paired 95% intervals from 6,000 resamples within each fixed layout, start and correlation scale. They reflect the joint choice of mapper and planner. The learned model lowers RMSE against GP two-step at both budgets, conditional on these layouts.

### Same-route predictions

Replaying the GP rollout measurements with the learned mapper at 26 edges gives mean RMSE of 0.661 versus 0.687 under weak dependence and 0.651 versus 0.678 when dependence persists. Under the same conditions, the learned mapper on its own shorter routes has RMSE 0.691 and 0.702. Thus the full-budget GP route is valuable to the learned mapper when travel is less constrained. The experiment does not match actual travel between the learned and full-horizon GP routes or isolate the value of branching from the learned prior.

At the 18-edge cap with weak cross-wall dependence, next-observation coverage using mean ±1.96 standard deviations is 94.6% for the learned model, 82.4% for GP two-step, and 82.8% for GP rollout. Mean interval widths are 2.59, 1.82, and 1.84 sensor units. These are moment-based intervals at each policy's chosen locations. They are assessed before acquiring the reading and include observation noise. The learned two-step search takes roughly 65–106 ms per route on this CPU, while GP two-step takes roughly 14–25 ms. The GP rollout timing in the artifact comes from the earlier benchmark's route archive.

## Replay and reproduction

From the repository root, generate the three-room archive as described in the [three-room benchmark](three-room-study.md), then run:

```sh
python -m spatial_exploration.sensing_world_study
python tools/build_three_room_demo.py
```

The study writes frozen weights, per-field metrics, aggregate results, and the learned plans for 32 demo scenes. The offline demo replays these model-generated plans. **Low**, **Mean**, and **High** show the map for an illustrative next observation at the forecast mean minus one standard deviation, at the mean, or at the mean plus one standard deviation. These scenarios are generated using the observations accumulated before the selected action.

## Follow-up: rollout to the remaining budget

The learned rollout reuses the existing learned parameters and the archived benchmark fields. It evaluates all feasible first stops. For each first stop, the GP rollout routine builds a greedy suffix using the current mixture's moment-matched covariance. The suffix continues until it reaches the reading cap or runs out of feasible travel. Each resulting path is scored by the expected mean latent variance over all 144 grid sites at the end of the suffix.

The suffix is fixed during each hypothetical evaluation. The score integrates the Gaussian mixture's posterior within-mode variance analytically. It estimates the expected between-mode variance by drawing 32 correlated noisy measurement sequences per mode with shared scrambled Sobol innovations and weighting mode averages by current posterior probabilities. The calculation conditions on all hypothetical readings in that suffix. Online, the planner takes only the first step, updates the mixture with the measured value, and evaluates the remaining budget again. The routine shares the GP rollout's first-move search and greedy completion logic. The approximate score preserves the mixture in terminal evaluation. Brute-force contingent planning would also optimize the later stops separately for every hypothetical outcome.

```sh
python -m spatial_exploration.sensing_world_study --full-rollout
```

This run saves frozen settings, per-field metrics, paired comparisons, and the original two-step learned replays and the 16 full-rollout replays for the 26-edge comparison under `outputs/sensing-world-rollout`. The learning and test protocols retain the original budgets of 18 or 26 walking edges and at most 16 readings. All contrasts use the same 64 paired fields per cell and 6,000 within-stratum bootstrap resamples. The comparisons reuse simulator test fields from the earlier study.

| Cross-wall dependence | Walk cap | Two-step learned RMSE | Learned rollout RMSE | GP rollout RMSE | Learned rollout readings / edges | GP rollout readings / edges |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Weak | 18 | 0.701 | 0.692 | 0.751 | 14.33 / 18 | 15.50 / 18 |
| Weak | 26 | 0.691 | 0.645 | 0.687 | 15.83 / 26 | 16.00 / 26 |
| Persists | 18 | 0.712 | 0.715 | 0.740 | 14.36 / 18 | 15.50 / 18 |
| Persists | 26 | 0.702 | 0.644 | 0.678 | 15.84 / 26 | 16.00 / 26 |

Learned and GP rollout use exactly the walking cap on all paired fields. At the 26-edge cap, learned rollout minus GP rollout RMSE is −0.043 [−0.062, −0.023] where walls weaken dependence and −0.034 [−0.052, −0.016] where dependence persists. Against the original learned two-step planner the corresponding differences are −0.046 [−0.070, −0.022] and −0.057 [−0.082, −0.035]. The two-step learned route covers 19.31 edges at this cap. The learned rollout measures in all three rooms on every field at 26 edges. The associated change in RMSE reflects the planning horizon, suffix objective, and candidate search together. Differences between the learned planners at 18 edges remain small, with both paired intervals including zero.

On the descriptive subset where both rollout policies used exactly 16 readings and 26 edges (56 of 64 fields per regime, selected by the realized measurement counts), RMSEs were 0.643 vs. 0.681 under weak dependence and 0.635 vs. 0.671 under persistent dependence. At 26 edges under weak dependence, switching the learned mapper to the GP rollout observations raises RMSE from 0.645 to 0.661. On those same GP observations the frozen ridge mapper gives 0.687.

At 26 edges, the learned rollout takes about 1.19 s to plan an episode, versus 0.67 s from the GP rollout archive and roughly 0.09–0.10 s for the learned two-step planner in the same follow-up. At locations selected by the respective rollout policies, the learned model's moment-based 95% noisy-reading intervals cover about 94–95% of the next readings, versus 82–84% for GP, with mean widths of about 2.83 and 1.95 sensor units. The verification reproduces the 768 baseline rows' resource and accuracy metrics and checks both caps on all 1,024 episodes. The neural test environment passed 109 tests with two optional skips.
