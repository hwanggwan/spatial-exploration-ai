# Experiment plan

## Question

Does a different spatial representation improve mapping and exploration under a limited travel budget, and under which conditions?

## Baseline protocol

- Domain: a unit square with discrete candidate sensing locations.
- Latent field: static, zero-mean, unit-variance RBF GP.
- Sensor: independent Gaussian noise at each candidate location. Each location can be observed once.
- A seed fixes both the latent field and the potential noisy observations shared across planners.
- A fixed subset of locations is withheld from all planners and used for evaluation only. Its measurements and true values never enter planning.
- The first sensing location is the lower-left corner and incurs no travel cost.
- Movement is a straight line without obstacles. The initial protocol budgets travel distance and has no measurement-count cap.
- The original experiments above budget only travel. The [two-resource sequential study](resource-study.md) now also caps measurement count, including the starting observation, and stops when either limit prevents further sensing. Do not interpret the older distance-only studies as controlling actual measurement count.
- Greedy score: expected integrated latent posterior variance reduction over the full grid, divided by travel distance. It uses the model covariance, not hidden field values.
- With fixed known covariance parameters, this variance-only planner's route is independent of measured values. It is an uncertainty-reduction reference, not a value-adaptive hotspot-search policy.
- Metrics: latent-field RMSE on held-out sites, mean posterior standard deviation there, descriptive 95% interval coverage there, distance traveled, and number of measurements.
- The entire realized field is available only to the evaluator and renderer. Final-budget RMSE is the initial comparison; add error-versus-distance curves before drawing efficiency conclusions.

## Limits of the initial experiment

The original GP mapper uses the known generating kernel and parameters. This is a strong matched-model baseline and an implementation check. It does not establish real-world robustness, and it deliberately does not favor DeepKriging.

The [covariance estimation study](covariance-study.md) now adds maximum marginal likelihood estimation of the RBF length scale and process variance. The mean, noise, and covariance family are still known. Kernel misspecification remains a later condition. Separate same-observation replay from model-dependent closed-loop exploration.

Five seeds are a smoke experiment, not evidence of superiority. Report paired seed-level differences and uncertainty across more independent maps for substantive claims. Grid sites within one field are not independent experimental replications. Per-seed coverage is descriptive; evaluate calibration across independent fields.

## Separate comparisons

1. **Planner:** same GP mapper, compare random / greedy / reproduced learned planner.
2. **Mapper:** identical recorded observations, compare GP / DNN with coordinates / DNN with multiresolution RBF embeddings (DeepKriging).
3. **Full system:** allow each planner to collect its own observations, compare outcomes on the same unseen maps, starts, and budgets.

The [sequential resource replay](resource-study.md) returns the mapper comparison to ordered acquisition from a fixed start under both travel and measurement caps. It reuses each acquired prefix across mappers, reports final error and resource-use curves, and recalibrates marginal neural bands on separate route-distributed maps. Reference routes remain independent of mapper predictions, so this is not the full-system comparison. The next planner experiment must match both upper bounds, report actual consumption, and establish a valid hypothetical-measurement update for the neural mapper.

The [mapper-driven uncertainty experiment](uncertainty-study.md) now allows each system to choose its own route using native marginal variance divided by distance. This rule requires current marginal uncertainty, not hypothetical conditioning or joint covariance. It is a feasible first closed-loop comparison under both resource caps; it must not be described as the earlier GP expected-integrated-variance acquisition. Independent calibration rescales reported intervals without changing action ranking. Score route differences and conditional oracle route quality after collection, with no true covariance in decisions. Hypothetical-measurement updates remain necessary for a future information-gain/lookahead version.

The frozen run yields distinct paths in every paired case but no demonstrated DeepKriging accuracy advantage. At caps (16,2), neural sensing acquires more points (12.20 versus 8.20) yet has higher mean error (0.8806 versus 0.8363), with a paired interval crossing zero; evaluator-only conditional route variance is higher as well. Native initialization uncertainty is poorly calibrated. Examine spatial uncertainty ranking and route redundancy before expanding the neural model or tuning acquisition variants; keep all new model selection separate from these now-exposed fields.

The subsequent [post-hoc crossover](crossover-study.md) fixes the recorded routes and fits each alternative mapper to exactly the same acquired values. At caps (16,2), its symmetric mean reconstruction component (0.03536) exceeds its sampling-design component (0.00888), while the latter's interval crosses zero. This shifts the diagnosis toward small-data reconstruction/regularization as well as uncertainty ranking. It is conditional descriptive analysis, not a new independent test or causal allocation of the total gap.

The [same-basis Bayesian control](basis-study.md) now implements the proposed linear diagnostic on 64 new fields, with four independent tuning candidates for both the basis prior and the neural mapper. The selected 300-epoch/0.01 neural setting differs from the previous run. At caps (16,2), mean own-system RMSE is 0.7888 / 0.8509 / 0.9185 for GP / neural / basis; on identical GP observations it is 0.7888 / 0.7992 / 0.8204. The Bayesian replacement does not help here. The current neural sampling contrast exceeds its reconstruction contrast, while native initialization uncertainty is extremely overconfident. This is a different frozen experiment, not causal evidence that regularization alone repaired reconstruction. Analytic posterior covariance under an isotropic basis prior does not automatically improve spatial exploration. An observation-conditioned neural model trained across independent maps is the next candidate; uncertainty and closed-loop validation remain required.

The [CNP study](cnp-study.md) now implements that observation-conditioned model with 4,096 offline training maps, 32 validation maps, two retained training seeds, separate calibration, and 64 new benchmark maps. Include a full-grid covariance GP learned from the same noisy offline data to control for prior learning. At caps (16,2), own-system RMSE is 0.8188 for fitted GP, 0.8475 for training-map GP, and 0.9197/0.9885 for CNP. On fixed fitted-GP observations the learned-prior GP reaches 0.7527, while CNP remains 0.8775/0.8770. Both offline spatial information and route selection matter; a useful learned prior does not make marginal-variance/distance planning optimal. The CNP's large observed-site error/uncertainty motivates a controlled query-dependent attention extension on a fresh frozen test set, not a claim that more complex AI must win.

The [attention-CNP study](attention-study.md) now adds deterministic query-specific attention with 17,204 parameters versus pooled CNP's 17,218, reusing identical offline data and training schedules. On 64 new fields at caps (16,2), same-GP-observation RMSE with both models at 6,000 updates improves from 0.8397/0.8230 to 0.7644/0.7667. Own-route attention errors remain 0.9035/0.8585 versus fitted GP's 0.7878. This separates a useful reconstruction change from an unresolved sensing-decision problem. The primary selected checkpoints differ (3,000 pooled versus 6,000 attention); the equal-duration diagnostic is secondary and separately recorded. Next hold the mapper fixed while comparing feasible acquisition rules, using the learned-prior GP's deployable covariance for an integrated-variance reference and fresh test maps. Neural factorized marginals alone do not establish a correct hypothetical-conditioning model.

The [two-step lookahead comparison](lookahead-study.md) adds a model-based planner under the same GP and physical budget. It uses combined variance reduction per travel distance over a two-measurement horizon and executes only the first move. This separates short-horizon planning from changes to the mapper and from policy training.

The [exact small-problem diagnostic](exact-study.md) enumerates complete feasible routes on tiny grids to quantify the remaining planning headroom. The exact objective is minimum final posterior variance under the budget; a local information-per-distance score is a heuristic for that objective. Separate failures of the score from failures caused by the planning horizon.

The [greedy-completion rollout study](rollout-study.md) evaluates every feasible first action followed by a terminal greedy continuation, minimizing final posterior variance. Replanning improves the greedy base under the fixed deterministic covariance model but is not globally optimal. On reused development cases it matches 26/27 tiny optima and reduces mean variance on the larger grid; realized RMSE does not improve comparably. Preserve separate measures of variance, realized prediction error, coverage, travel, and compute. PPO's existing undiscounted reward already targets total variance reduction.

The [behavioral-cloning study](imitation-study.md) now trains matched actors on common states labelled by either rollout or greedy, with independent training/validation episodes and 96 new paired test cases. Both rollout clones reduce mean variance relative to greedy while sacrificing some teacher quality for much lower online compute. Account for teacher-data generation and audit costs, and examine agreement on student-visited states. Architecture matching does not equalize PPO and cloning training resources. The reported test cases are now exposed; keep them separate from future model selection and use a new test set for new variants.

The [frozen-policy robustness study](robustness-study.md) tests three covariance treatments on 96 additional paired cases, using the actual current mapper covariance and length scale as neural inputs. Frozen clones lose their known-covariance advantage under fitted covariance; reported variance alone is unreliable across models, and fitted uncertainty undercovers. Evaluate route quality with an evaluator-only true GP, keep parameter-learning cost separate from action-selection cost, and do not transfer the fixed-model rollout guarantee across future covariance refits. Any adaptation must use separate training/validation data and a new frozen test set.

The [sample-size mapper benchmark](sample-size-study.md) increases shared observations from 8 to 128 on a fixed 15-by-15 grid, with 64 fields in each of three spatial scenarios. Fresh stationary-RBF fits nearly recover the oracle in the correctly specified case and improve in every regime. Spatially varying dependence retains a positive paired oracle gap and regional calibration imbalance at the largest sample size. A stationary two-scale control guards against attributing every covariance-family mismatch to nonstationarity. Archive observation prefixes separately from evaluator truth for the planned coordinate-DNN/DeepKriging comparison. This mapper-only experiment does not establish recovery of the frozen exploration policy.

To attribute a gain to RBF embedding, include a comparable coordinate-input neural network. To attribute a gain to policy learning, hold the mapper fixed. Retrain learned policies for each mapper using matched training resources and report both training and deployment costs.

## Before adding DeepKriging

The separate [paper-function study](paper-mapping-study.md) now audits the original deterministic 2D benchmark before interpreting the adapted mapper's negative results. It fixes 90 evaluation sites per fold, nests acquired sets from 810 down to 16, tests two source-based neural configurations and two GP kernels, and resets all neural weights and optimizer state for every fit. No test-driven model selection or cross-fold weight reuse is allowed. The one-surface cross-validation result must be kept separate from both independent-field evidence and movement-budget experiments.

The 440-fit run does not recover the published ranking: at n=810, exponential/RBF GP errors are 0.01514/0.00426, versus 0.06906 for the supplement-description network and 0.01777 for the clean public-code variant. Both neural means remain above exponential GP at all smaller counts. Dense regularized-network training error remains large; sparse unregularized-network interpolation does not generalize. Resolve source/optimization discrepancies on an independently planned audit before attributing these outcomes to nonstationarity or assuming additional observations will produce a neural advantage.

The follow-up [same-Keras audit](keras-reuse-audit.md) now fixes architecture, initialization, batch orders, and optimizer resets while comparing cross-fold weight reuse, fresh 400-epoch fitting, and fresh fitting with matched cumulative updates. Across three seeds and ten folds, current-fold RMSE is 0.00541 for contaminated reuse, versus valid held-out RMSE 0.01767/0.01481 for the fresh controls. Added training helps, but retained history changes scores much more in this diagnostic. The valid longer fit is close to exponential GP and remains above RBF GP. The public notebook and Python script differ, and historical table provenance remains unresolved; do not equate this observed code behavior with proof about the paper's published numbers. Keep subsequent tuning, uncertainty validation, and exploration decisions separate from this one-function audit.

The first [DeepKriging-style mapper comparison](deepkriging-study.md) now implements per-map neural fitting, a capacity-matched coordinate control, and initialization-ensemble bands calibrated on independent fields. Hyperparameters and interval scales freeze before benchmark evaluation. This addresses the implementation questions below, but does not provide the joint covariance needed by the current information-gain planner. The implementation adapts the published spatial representation; it does not reproduce the complete paper setup or density-estimation method.

On the exposed nonstationary benchmark at n=128, the spatial network improves over coordinates (RMSE 0.3520 versus 0.4601) but remains worse than fitted GP (0.2245). Raw ensemble uncertainty substantially undercovers; calibration yields wider intervals. The regularized same-basis linear diagnostic is now implemented in the separate sparse sequential experiment above; it does not reproduce this high-count mapper experiment. A stronger estimated nonstationary GP remains a useful later comparison. Do not integrate the neural mapper into an information-gain planner based on these results alone.

- Baseline and selected existing AI method run reproducibly, with documented deviations from the source method.
- Specify how the neural mapper conditions on new measurements: per-map fitting/fine-tuning versus an observation-conditioned architecture. A coordinate-only network trained on one map cannot infer the realization of an unseen map without observations.
- Specify predictive uncertainty and distinguish measurement noise from uncertainty about the unobserved field.
- First test the mapper on replayed observation sets, including sparse early-exploration regimes.
- Compare update time and uncertainty calibration alongside prediction error.

## Later scenario matrix

Vary correlation length, sensor noise, sampling budget, anisotropy, and nonstationarity. Keep training, validation, and test fields separate. Select configurations on validation results, then freeze them before test evaluation. Do not design only favorable scenarios after inspecting results.

The [transition-sharpness study](transition-study.md) fixes correlation lengths and all measurement settings, varying only how abruptly the long/short mixture changes. It uses common component realizations and new benchmark fields generated after neural settings and calibration freeze. Compare paired field errors and fixed regional bands; do not assume sharper transitions must increase whole-map error or favor a neural model.

The recorded comparison finds little absolute RMSE change at n=128 for fitted GP or the spatial network, while oracle RMSE improves. The fitted GP retains its non-oracle lead. A post-hoc excess-over-oracle diagnostic increases with sharpness, clarifying why raw error alone does not isolate model mismatch from changed task difficulty. Preserve these unfavorable neural results when moving to application-driven barrier conditions.

[Barrier applications](barrier-applications.md) motivate a separate proposed experiment with walls/doorways or water-network connectivity. Keep movement constraints separate from field transport. Provide layout information to both the GP and neural model, include a fitted geometry-aware GP and a same-feature linear control, and reserve unseen layouts for evaluation. A coordinate-only win on one fixed map does not establish learning of physical connectivity or transfer to new terrain.

## Useful outcomes

An accurate reproduction, a functioning demo, and a defensible analysis are deliverables even if DeepKriging does not improve exploration. Report where it helps, where it fails, and what the computational tradeoff is.
