# Sequential mapping under two resource limits

## Objective

Return to the project's original question: how accurately can a spatial field be mapped from a fixed start when both the number of measurements and travel distance are limited? The primary outcome is final held-out latent RMSE under **both constraints simultaneously**, rather than accuracy at a randomly chosen observation count or a model's reported variance alone.

The first stage isolates mapping on shared sequential routes. It does not yet demonstrate a neural planner or a complete DeepKriging exploration system. The same route feeds oracle GP, fitted stationary GP, and the existing adapted DeepKriging mapper. A subsequent closed-loop comparison must let each system select its own actions on matched maps, starts, and resources.

## Resources and reference routes

- A 15-by-15 unit-square grid, with the same 45 evaluation-only sites as previous studies.
- Fixed start at grid site 0, coordinate (0,0). Its reading counts as measurement one and costs zero travel.
- Measurement caps: 8 and 16. Travel caps: 1 and 2 unit-square distance units. All four combinations are evaluated.
- Each unvisited, permitted location can be measured once. Movement is a straight segment; there are no walls, obstacles, or return-to-base costs. Every future sensing action consumes one count and its Euclidean travel distance.
- Stop when the measurement cap is reached or no unvisited sensing site is affordable. A resource cap is an upper bound: one route may leave unused measurements or distance.
- Two reference designs: uniformly random affordable next site, and greedy whole-grid variance-reduction per distance under a fixed initial stationary covariance (length 0.35, variance 1.5, noise std 0.1).

Reference routes are generated using **no field measurements or true covariance**. They are independent of the alternative mappers and do not adapt to measured values. Greedy knows the remaining travel budget but does not optimize a full future route or explicitly price the remaining observation count. It is a simple constrained heuristic, not an optimal solution to the two-resource planning problem.

For each route every mapper receives identical acquired positions and noisy values, in the same order. Models can access all grid coordinates but not unobserved field values. The full latent field and potential sensor readings are simulator/evaluator-only; the fitting interface contains only acquired prefixes. Shared canonical prefixes are evaluated once and reused numerically, so merely changing a resource cap cannot change predictions on identical data. Per-episode compute accounting still charges every prefix that episode would fit.

## Fields, fitting and calibration

Use the gradual nonstationary mixture from previous work, retaining long/short lengths 0.35/0.12, zero mean, unit variance, and noise std 0.1. This is a representative controlled sensing task, not a new severity sweep or a real-world deployment.

DeepKriging retains the same 236 features, two 32-unit hidden layers, and three-member initialization ensemble. Prespecify 100 epochs and weight decay 0.0001, carried from independent tuning in the transition study. These settings are applied unchanged to every sequential prefix; n<8 is an explicit extrapolation beyond the earlier tuning counts. No new benchmark-based model selection is performed. Each prefix is fitted fresh, rather than warm-starting from future or past fitted networks.

The fitted GP also uses a fresh snapshot at every prefix, starting from the established initial covariance and fitting its RBF length/variance once n>=4. Below four observations it retains the initial covariance. Both mappers fit only measurements acquired so far; this snapshot protocol deliberately retains the previous mapper comparison's conditioning rule and is not the older online GP's every-three-observations refit schedule. Oracle receives the true covariance as an explicit privileged reference.

Generate eight independent calibration fields, with field/route/member seed bases 3,000,000/3,010,000/3,020,000. Use the same resource matrix, but independent random routes. Estimate a separate pooled marginal variance multiplier for each available prefix length using only these calibration maps. Spatial sites and prefixes are dependent; this is heuristic scaling, not a posterior or conformal guarantee. Raw and scaled coverage and interval widths are reported.

Freeze settings and calibration **before generating** the 64 benchmark fields, whose field/route/member seeds start at 3,100,000/3,110,000/3,120,000. Tests enforce that ordering. Every resource/design combination uses the same field realization and potential measurement noise. These are new synthetic field realizations for this frozen comparison, not proof of performance outside this simulator.

## Evaluation and interpretation

Record error, coverage, interval width, distance and measurement count after every reading. At termination, compare models on paired fields under each resource pair and route design. Report percentage RMSE advantage as `100*(mean_GP_RMSE-mean_DK_RMSE)/mean_GP_RMSE`; positive favors DeepKriging and negative favors GP. Bootstrap 64 fields for paired differences; grid sites, budget settings and prefixes are not independent replicates.

Show error versus distance allowance and measurement allowance, always retaining the other resource constraint. After an episode stops, carry its last available prediction forward in the curve. This does not invent new readings or assert that another observation was collected. Distance curves audit a fixed route designed for its full episode budget; they are not separately replanned optimal routes for every plotted intermediate budget.

Random and greedy routes may spend different amounts of the allowed resources. Their final error comparison is a comparison of feasible resource allocation under common upper bounds, not equal numbers of actual readings. For a mapper comparison on any one route, the actual readings and distance are identical. No claim of globally optimal exploration or learned-policy advantage follows from these reference routes.

Both models pay repeated refitting cost. GP timings are single-prefix construction/fitting/conditioning; neural times include initialization, optimization and grid prediction, amortized across independent batched states. Per-episode sums charge all visited prefixes even when their numerical results were cached across budget conditions. These have different scopes and batching advantages, so they are not a reliable online latency speedup benchmark. Computation is measured but is not currently a third enforced resource limit.

## Recorded results

Recorded November 4, 2026, without changing the prescribed model settings. The 64 paired fields produce 512 route episodes, 1,478 unique acquired-prefix states, 4,434 independent neural member fits, 11,946 step-level model evaluations and 1,536 endpoint evaluations. All episodes respect both caps, begin at (0,0), and exclude all 45 evaluator sites.

Mean resource consumption and final held-out RMSE (lower is better):

| Observation cap | Distance cap | Reference route | Actual readings | Actual distance | Fitted GP RMSE | DeepKriging RMSE |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 8 | 1 | Random | 3.70 | 0.971 | 0.9119 | 0.9312 |
| 8 | 1 | Greedy | 8.00 | 0.647 | 0.9097 | 0.9264 |
| 8 | 2 | Random | 5.41 | 1.972 | 0.9288 | 0.9244 |
| 8 | 2 | Greedy | 8.00 | 0.647 | 0.9097 | 0.9264 |
| 16 | 1 | Random | 3.70 | 0.971 | 0.9119 | 0.9312 |
| 16 | 1 | Greedy | 12.00 | 0.933 | 0.8727 | 0.9159 |
| 16 | 2 | Random | 5.41 | 1.972 | 0.9288 | 0.9244 |
| 16 | 2 | Greedy | 16.00 | 1.379 | 0.7879 | 0.8357 |

The cap combinations expose different bottlenecks. The greedy route with an eight-reading cap uses only 0.647 distance, so increasing its travel cap does not help. With cap 16 and distance 1 it stops at 12 readings because another site is unaffordable. At distance 2 it reaches 16 readings after 1.379 travel. Random movement consumes most travel with far fewer observations, so raising its count cap from 8 to 16 changes none of these endpoints. The same realized consumption is shared exactly by GP and DeepKriging on each route.

On the greedy route with caps (16,2), DeepKriging has 6.07% higher mean RMSE than fitted GP; the paired difference is +0.04783 with field-bootstrap 95% interval [0.01911, 0.07409], improving 21/64 fields. With caps (16,1) the difference is +0.04323, interval [0.00735, 0.07836]. Other mapper comparison intervals cross zero, including the small numerical spatial-network advantage on random distance-2 routes. These data provide no clear DeepKriging advantage under the tested physical resource limits.

Within caps (16,2), fitted GP on the greedy reference route has 15.17% lower mean RMSE than on the random route, which is a resource-allocation result under common upper bounds; actual reading counts are different. This demonstrates the relevance of route choice without establishing a learned-policy advantage. The current greedy route still optimizes a local density heuristic and need not be the best feasible route under both constraints.

### Sequential uncertainty and compute

At caps (16,2) on the greedy route, oracle RMSE is 0.7342 and nominal 95% coverage is 95.56%. Fitted GP coverage is 85.83%, versus DeepKriging raw coverage of 52.81% and calibrated coverage of 93.19%. Mean reported latent std is 0.6768 for fitted GP and 1.4266 for calibrated DeepKriging. The apparent coverage improvement comes with much wider intervals; neither non-oracle uncertainty treatment is a substitute for a validated planner posterior.

The curves show transient error increases, including at the first GP parameter fit. Parameter refitting and realized prediction error are not guaranteed to improve monotonically after every reading. The longer-distance random routes are redesigned for their new full budget rather than guaranteed nested prefixes, so their endpoint errors also need not improve monotonically with a larger budget. Do not interpret more acquired points as automatically informative coverage of the whole domain.

Calibration neural fitting takes 3.96 seconds wall time in this run. Benchmark unique-prefix fit totals are 21.21 seconds for neural ensembles and 14.79 seconds for fitted GP under their respective timing scopes. Charging every visited prefix, mean accumulated scoped fit time on the caps-(16,2) greedy episode is 0.229 seconds neural versus 0.180 seconds GP. Neural values are batched amortized throughput and do not establish sequential deployment latency; no compute-budget superiority is claimed.

### What this establishes and what remains

This run evaluates the existing mappers at actual sequential resource states rather than IID observation snapshots. It finds no evidence of a DeepKriging benefit under these caps and shows that measurement count and travel constraints can bind differently. It does not establish that DeepKriging cannot help after a different prespecified training/update design, nor that these reference routes are optimal.

The next full-system experiment must make action selection explicit: fit/retrain a policy for remaining count and distance, or use a planner that evaluates both resources. For a DeepKriging-driven system, first define a valid observation-conditioned uncertainty/update rule; current marginal ensemble spread cannot simply replace the GP covariance in the existing information-gain formula. Keep shared-route mapper results separate from any future mapper-dependent closed-loop trajectories.

All 63 tests pass with optional dependencies; system Python passes 39 and skips 24 optional tests. The baseline capped example was run on five seeds. Tests check both caps and starting-point accounting, true path length, unique allowed sensing sites, canonical prefix reuse, acquired-only values, freeze-before-benchmark generation, and resource curves that never invent observations.

## Reproduction and artifacts

```sh
# Baseline now also accepts a measurement cap; omit it for the old distance-only behavior.
PYTHONPATH=src python3 -m spatial_exploration.baseline --max-observations 8 --travel-budget 2 --output outputs/baseline-capped
PYTHONPATH=src .venv/bin/python -m spatial_exploration.resource_study --output outputs/resource-study
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

Use a fresh study directory. Archive `protocol.json`, frozen scales, calibration and benchmark routes, acquired prefix arrays, separate simulator/evaluator-only fields and potential measurements, predictions, representative neural checkpoints for each prefix length, GP fit parameters, step/final metrics, curves, summaries, paired mapper comparisons, data hashes and diagnostics. Member seeds are reconstructible from state IDs and the recorded seed base. Representative checkpoints do not cover every state; all state predictions are saved.

Figures: `example_routes.png` shows a predefined field and both routes; `sequential_curves.png` shows error as resources accrue; `budget_comparison.png` compares final error under the resource matrix. Existing benchmark archives are preserved.
