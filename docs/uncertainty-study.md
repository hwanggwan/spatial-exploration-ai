# Mapper-driven uncertainty exploration

## Question

Can GP and DeepKriging each choose the next sensing location using their own uncertainty, and does either map the field more accurately under the same observation-count and travel limits? Unlike the [shared-route resource replay](resource-study.md), this is an actual mapper-dependent closed loop: routes and measurements may differ between the two systems.

The action rule is deliberately simple and identical for both models:

`score(s) = current native latent marginal variance at s / distance(current,s)`.

Choose the highest score among affordable, unvisited, permitted sites. Ties use the lowest site index. Stop when the reading cap is reached or no next site is affordable. This rule seeks uncertain locations at low travel cost; it does **not** calculate expected reduction of uncertainty over the whole map, solve the optimal two-resource route, or train a new neural policy.

## Matched physical task

Use the same 15-by-15 unit square, gradual nonstationary long/short GP mixture, lengths 0.35/0.12, zero mean, unit variance, and sensor noise std 0.1. Start at site 0, (0,0), with one reading and zero travel. Test caps of 8/16 readings and 1/2 travel distance, all combinations. There are no obstacles, robot dynamics, measurement-time costs, or return-to-base requirements. The 45 evaluation-only sites remain unavailable for sensing.

Generate 64 new fields (seed base 4,100,000), with the same potential sensor values shared by both systems and all budget settings. Each system collects its own ordered observations, so actual reading counts and travel use can differ under the common upper bounds. The earlier replay's `design_length`/`design_variance` settings are inherited in the resource-config archive but do not define this planner; current fitted-model variances drive every action.

The collection function receives only geometry/configuration and a `read_sensor(field,site)` callback. It queries that callback on arrival, including the initial site. It accepts no latent field or true covariance. Training gets only the acquired prefix. Evaluator-only values are used after **all** paths have been fixed. Tests alter never-visited potential measurements and verify that the paths and predictions remain unchanged.

## Mappers, uncertainty and updates

**Fitted GP:** fresh RBF snapshot at every acquired prefix, known zero mean and noise, initial length 0.35 and variance 1.5. From four readings onward, fit covariance parameters using the existing bounded marginal-likelihood procedure with two optimizer restarts. Use the diagonal of the current plug-in latent posterior covariance, excluding sensor noise. This does not integrate parameter uncertainty.

**DeepKriging-style:** the existing adapted 236-input Wendland/coordinate network, two 32-unit ReLU layers, three independent initialization members. Keep the previously independently selected settings fixed: 100 epochs, weight decay 0.0001, Adam learning rate 0.003. Fit fresh from each acquired prefix, including sparse n<8 prefixes outside the earlier tuning counts. Use member sample variance, floored at 1e-6, excluding measurement noise. This is an initialization-spread proxy, not a guaranteed posterior uncertainty or a full paper reproduction.

Both use **variance**, not standard deviation, in the score. Both refit after every reading, including the final reading used for scoring. The acquisition rule does not see prediction means or hidden values. It accounts for the count cap through feasibility/termination, not by looking ahead over the remaining number of readings.

Member seeds use `SeedSequence([base,field,count,member])`, converted from uint64 modulo 2^63. Initialization is shared across resource conditions at a given field/count, without sharing learned weights; identical acquired prefixes reuse exact numerical predictions. This prevents resource-cap changes from causing an unrelated initialization change on identical data. Benchmark member seed base is 200,000,000; calibration base is 100,000,000. There is one ensemble construction per field/count, not a separate repeated-training-seed experiment.

## Independent interval calibration

Before generating benchmark fields, collect mapper-driven routes on eight independent calibration fields (seed base 4,000,000), using the same physical task and native acquisition rule. Fit one positive global variance multiplier **for each model**, pooling its acquired-prefix predictions and withheld sites. Freeze these multipliers and all settings in `frozen_protocol.json` before benchmark generation.

Report native and calibrated coverage and interval width for both GP and neural models. Scalar calibration multiplies all scores within one state by the same positive value, so it cannot change action ranking. It can improve interval scale but cannot fix a map that puts uncertainty in the wrong locations. Spatial sites, prefixes and budget conditions are correlated; longer routes contribute more calibration states, and there are only eight independent calibration fields. No conformal or posterior coverage guarantee is claimed.

## Evaluation

The primary comparison is paired final held-out latent RMSE under each common resource pair. Report actual reading count, travel, native/calibrated coverage, interval width and repeated fitting cost. Bootstrap fields, not sites, for paired RMSE differences. Resource conditions are correlated; do not interpret them as independent replications or claim multiplicity-adjusted significance.

After routes are fixed, compute an evaluator-only **oracle GP reconstruction on each acquired route**, including its held-out RMSE and conditional variance. This reference uses the correct generating covariance only at evaluation time. Its conditional variance helps describe sensing-location quality separately from each mapper's own reported uncertainty; it does not enter decisions, and a favorable oracle reconstruction does not establish the deployed mapper's accuracy. The original frozen run did not include both deployed mappers on both routes; a separate subsequent [post-hoc crossover diagnosis](crossover-study.md) now records that comparison without changing the original run.

Record exact route equality, visited-site Jaccard overlap and common-prefix length. Plot a predefined field-0 example rather than selecting a favorable map after seeing results. Error-versus-distance and count curves carry forward the last available prediction after an episode stops. Intermediate distance plots audit paths planned with the stated full episode budget, rather than reoptimizing for each plotted distance.

GP costs are single-prefix fitting/conditioning; neural costs are independent batched fitting plus prediction, amortized over states. Per-episode sums charge every visited prefix even if a numerical fit is cached across resource settings. Total collection wall time also includes acquisitions and orchestration. These are not directly comparable online latency benchmarks or a matched compute budget. No neural speedup is inferred from batching.

## Recorded results

Recorded November 4, 2026, without changing the frozen settings. All 512 benchmark episodes respect both caps and exclude evaluation sites. There are 1,432 unique fitted prefix states and 2,529 independent neural member fits. Across all 256 paired field/resource cases, **no GP and DeepKriging routes are identical**; their common prefix contains only the starting reading. They are no longer following the shared reference routes.

Mean final held-out latent RMSE and actual resource use over 64 paired fields:

| Reading cap | Distance cap | GP readings | DK readings | GP distance | DK distance | GP RMSE | DK RMSE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 1 | 4.00 | 6.83 | 0.995 | 0.935 | 0.9069 | 0.9263 |
| 8 | 2 | 7.30 | 7.98 | 1.867 | 1.284 | 0.8504 | 0.9065 |
| 16 | 1 | 4.00 | 7.14 | 0.995 | 0.963 | 0.9069 | 0.9216 |
| 16 | 2 | 8.20 | 12.20 | 1.966 | 1.966 | 0.8363 | 0.8806 |

DeepKriging collects more observations on average but does not achieve lower mean error in any of the four resource conditions. At caps (16,2), its mean RMSE is 5.29% higher; the paired difference is +0.04424 with field-bootstrap 95% interval [-0.00292, 0.08848], improving 23/64 fields. This interval includes zero, so this condition alone does not establish a clear accuracy ranking. At caps (8,2) the paired difference is +0.05613, interval [0.00781, 0.10048], improving 20/64 fields. The two distance-1 comparison intervals cross zero. These are descriptive paired intervals across four correlated settings, not a multiplicity-adjusted general superiority claim.

The predefined field-0 illustration at caps (16,2) has seven GP measurements over distance 1.93 and thirteen neural measurements over distance 1.96. It illustrates distinct routes; the aggregate result comes from all 64 fields rather than this one map. At caps (16,2), mean visited-site Jaccard overlap is only 0.0639, including the common start.

### Sensing-location quality versus own-model prediction

At caps (16,2), the evaluator-only true-GP reconstruction gives mean held-out conditional variance 0.6010 on GP-selected routes and 0.6535 on neural-selected routes, despite the latter's larger reading count. The paired variance difference is +0.05252, field-bootstrap interval [0.02220, 0.08482]. Under this generating model, the neural routes leave more reference uncertainty on average; more nearby readings do not necessarily provide more useful spatial coverage.

Oracle RMSE on those same routes is 0.7693 versus 0.7972. Its paired difference interval [-0.00896, 0.06509] includes zero. These conditional reference diagnostics help identify a route-quality limitation but do not fully separate all sources of deployed-mapper error. They are computed only after routes are fixed, with no true covariance exposed to either planner. The later [frozen-route crossover](crossover-study.md) finds a larger mean reconstruction-model contrast than sampling-design contrast at caps (16,2), with important uncertainty and post-hoc qualifications.

### Uncertainty remains a limitation

Native 95% coverage at caps (16,2) is 86.22% for GP and 53.85% for neural initialization spread. Independent global variance multipliers are 4.7067 and 47.4648 respectively (std multipliers about 2.17 and 6.89). After scaling, coverage is 97.95% and 97.01%, with mean std 1.5719 and 2.3256. These broad, somewhat overcovering endpoint intervals come from a calibration pool that also includes earlier prefixes; they are not evidence of precise or spatially well-ranked uncertainty.

The neural uncertainty map can be patchy and can disagree with reference uncertainty. Multiplying every location by a common number adjusts scale but cannot improve where the planner believes information is missing. The three-member ensemble, small-data fresh fitting and sparse early prefixes are important limitations of this implementation. Do not generalize this outcome to the published DeepKriging method or all neural uncertainty methods.

### Compute and validation

Calibration collection takes 4.58 seconds wall time and benchmark collection 19.47 seconds in the recorded one-thread run, excluding evaluator-only oracle reconstruction and rendering. Unique-prefix scoped fit totals are 5.46 seconds for GP and 13.76 seconds for neural ensembles. Mean accumulated scoped fitting cost per caps-(16,2) episode is 0.073 seconds GP versus 0.199 seconds neural, with the batching/scope qualifications above; no deployment speed claim follows.

All 67 tests pass with optional dependencies. System Python passes 41 and skips 26 optional tests. Tests cover the variance/distance ranking, feasibility and termination, invariance to positive scalar calibration, rejection of invalid uncertainty values, sensor calls only at acquired sites, reproducibility when never-visited measurements change, own-prefix fitting, reconstruction of each chosen action from the saved variance, and calibration freezing before benchmark generation. Archived predictions and representative checkpoints can be reloaded and independently checked against every recorded next action.

The immediate conclusion is that mapper-dependent exploration is operational and physically feasible, but this uncertainty proxy has not delivered a DeepKriging advantage. A useful next diagnostic is spatial uncertainty ranking against evaluator-only prediction errors and oracle conditional variance before increasing model complexity or searching acquisition settings against these exposed fields. Any new variant requires separate development data and a new frozen benchmark.

## Reproduction and artifacts

```sh
PYTHONPATH=src .venv/bin/python -m spatial_exploration.uncertainty_study --output outputs/uncertainty-study
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

Use a fresh output folder. Save protocols, calibration and benchmark routes, acquired-only prefix arrays, separate evaluator-only data, all means/variances and member predictions, representative checkpoints with explicit state-ID indices, GP fit diagnostics, step/final metrics, curves, paired comparisons, route comparisons and data hashes. `paths.png` shows just the two paths on the same evaluator-only field; `example_routes.png` also includes predictions and native uncertainty. Other figures show sequential error and final errors under the resource matrix. Earlier benchmarks are preserved.

The new test fields and new acquisition rule mean results are **not** a paired comparison against the old information-gain greedy routes or the earlier shared-route study. A future acquisition-method comparison must run those alternatives on the same new fields and constraints.
