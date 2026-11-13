# Same-Keras fold-reuse and training-duration audit

This diagnostic follows the [paper-function reimplementation](paper-mapping-study.md), which did not recover the published DeepKriging ranking. It asks whether retaining neural weights across cross-validation folds changes the reported error beyond what is obtained by independently training for the same number of updates. This is an audit of training history, not a new exploration experiment or an estimate of genuine held-out performance for the reused model.

A later [historical provenance audit](paper-provenance-audit.md) connects saved 2020 notebook outputs to the earlier manuscript's Table 2 means and to the published Table S4 GP mean/SD. The final revised Table S2 execution remains unidentified. This narrows the historical uncertainty without changing this study's controlled training-history comparison.

## Source and frozen design

The [authors' public 2D script](https://github.com/aleksada/DeepKriging/blob/master/Nonstationary_function-2d.py) constructs one three-hidden-layer neural network outside its fold loop. Each fold recompiles with a fresh Adam optimizer, then makes two consecutive 200-epoch fits while retaining weights. Its dropout/batch-normalization lines are commented out. The original script is downloaded, archived as text, and hashed; unrelated notebook/GP plotting code is not executed.

A post-hoc inspection of the repository's [saved notebook](https://github.com/aleksada/DeepKriging/blob/master/Nonstationary_function-2d.ipynb), performed after the current fits began, finds further source ambiguity. It uses RBF GP and current-test-fold validation to choose an additional training duration, whereas the Python script uses exponential GP and fixed durations. Its displayed cells construct `model_4` but later fit `model`, and append layers to `model_base` before its displayed definition. Execution counts are nonsequential. Stored output means (DK 0.010540, coordinate DNN 0.088055, GP 0.004432) also differ from published Table S2. Those saved outputs cannot be assumed to correspond to a clean execution of the current cells. These observations do not change this frozen experiment or establish how the published table was generated; they strengthen the need to distinguish the inspected Python script, saved notebook, and publication. The notebook hash and observations are archived separately in `notebook_provenance.json`.

The current study uses the same Keras/TensorFlow model family instead of porting the network to PyTorch. It is still a modern-version adaptation, **not a locked historical-environment reproduction**. TensorFlow 2.20.0 and Keras 3.15.1 are recorded. The previous paper-function archive supplies the exact noiseless function values, 900 known grid positions, ten outer folds, and 810 acquired labels per fold. All existing reference artifacts remain unchanged.

Three initialization seeds and all conditions are fixed before fitting:

| Condition | Starting weights | Data in training history | Epochs before scoring fold j, numbered 1–10 | Valid held-out score? |
| --- | --- | --- | --- | --- |
| Reuse weights | Continue from previous fold | All preceding training pools, then current pool | 400j cumulative | Only first fold |
| Fresh, 400 | Original seed's initialization | Current fold's 810 acquisitions only | 400 | Yes |
| Fresh, matched | Original seed's initialization | Current fold's 810 acquisitions only, repeated | 400j | Yes |

The first fold is numerically identical across conditions, so its fit is shared rather than redundantly repeated. Within each seed, all new models start from the exact same weight arrays. At stage k, both streams use the same sequence of permutations of the 810 training-row positions. Each stage starts with zero Adam iteration/moment state, matching the source recompilation behavior; weights reset only for fresh models. The fresh matched control repeats its current acquired set for j stages and has exactly the same cumulative number of gradient updates as the reused model at fold j. Its first stage also supplies the fresh-400 result, preserving a common prefix. Matched budgets range from 400 to 4,000 epochs, averaging 2,200 across folds; the mean is not a uniform 4,000-epoch experiment.

The network has three 100-unit ReLU hidden layers, first-layer He-uniform initialization and later Glorot-uniform initialization, scalar output, and no dropout or batch normalization. Inputs are the same raw 1,830 Wendland features as the previous study, with support 2.5/m at resolutions 10/19/37. Adam uses learning rate 0.001, betas 0.9/0.999, epsilon 1e-7, float32, and no weight decay. Batches contain 64 points, with the final 42-point batch retained: 13 updates per epoch and 5,200 per 400-epoch stage. CPU intra/inter-op threads are one, with deterministic operations enabled.

For efficiency, a compiled TensorFlow loop applies the actual Keras model and Adam to each minibatch. An equivalence test compares its final weights against Keras `train_on_batch` on identical acquired batches and initial weights. This replaces `model.fit` input-pipeline overhead; it is a documented loop adaptation. Source validation logging is omitted because no early stopping or validation-driven selection is performed. Batch orders and initialization seeds are explicitly coupled rather than attempting to recreate unspecified historical RNG state. Training RMSE is logged every 50 epochs; test values choose no settings, weights, training durations, or stop points.

A separate post-freeze check also compares the compiled loop directly with literal Keras `model.fit` on the actual fold-0 acquired set: 810 points, 1,830 inputs, three 100-unit hidden layers, four epochs, and identical explicit minibatches. Both make 52 updates and yield a maximum weight difference of exactly zero. This finite numerical check does not recreate unspecified historical software behavior, but checks the current loop adaptation at the real network/data size. It changes no fitting settings; results are in `fit_equivalence.json`.

## Why reuse is contaminated

Ten-fold test sets are disjoint. Consequently, every test point in fold two was part of fold one's training pool. The same holds for every later test fold. From fold two onward, the reused model's current test values have already influenced its weights. Its training history also covers all 900 grid sites after its second stage. A low score on these points can reflect prior exposure and must not be interpreted as independent prediction.

The fitting API still accepts only each current pool's acquired features and labels. The contamination arises deliberately through retained weights, not by directly passing the full evaluation vector into a training function. Each result records how many current test points were previously seen; fresh models have zero such points. The reused curve is clearly labeled contaminated CV.

The matched comparison controls initial weights, cumulative update counts, stagewise optimizer resets, and batch-index orders. It changes the identities and frequencies of labels encountered earlier. Thus it assesses the effect of retained training history beyond duration on this surface; it does not identify a universal causal percentage attributable solely to leakage.

## Results

Recorded November 5, 2026. All 30 seed/fold combinations completed with the predeclared settings, producing 90 condition-level results. Numerical fitting actually executes 192 stages (76,800 epochs and 998,400 minibatch updates); common prefixes are fitted once. Actual stage execution took 1,258.46 seconds, and the training/scoring/checkpoint loop took 1,263.93 seconds (21.07 minutes), excluding final plots. Stage timings include graph tracing and optimization, but exclude model construction and checkpoint I/O; the loop timing includes those costs.

Mean of individual seed/fold RMSEs, with no prediction ensembling:

| Condition | Mean training RMSE | Mean current-fold RMSE | Seed-wise mean current-fold RMSEs | Interpretation |
| --- | ---: | ---: | --- | --- |
| Reuse weights | 0.001819 | 0.005414 | 0.005526 / 0.005431 / 0.005284 | Contaminated from fold two |
| Fresh, 400 epochs | 0.003513 | 0.017667 | 0.017938 / 0.017412 / 0.017652 | Valid held-out predictions |
| Fresh, matched 400–4,000 epochs | 0.002017 | 0.014807 | 0.015244 / 0.014998 / 0.014180 | Valid held-out predictions |

**Additional training improves mean held-out error by 16.19%, but does not reproduce the reused model's low scores.** Relative to the matched control, the reused model's mean current-fold RMSE is 63.44% lower. All 27 later-fold seed/fold cases have lower reused-model errors than their matched fresh counterparts; the three first-fold cases are identical by construction. The matched fresh model improves over the 400-epoch prefix in 24/30 cases. These are descriptive comparisons on one function, not independent-field significance tests or a causal allocation of error to leakage.

The same-fold GP references from the prior archive remain unchanged:

| Valid mapper | Mean held-out RMSE | Folds lower than exponential GP, averaging neural seeds within fold |
| --- | ---: | ---: |
| Previous exponential GP | 0.015136 | Reference |
| Previous RBF GP | 0.004264 | 10/10 |
| Fresh Keras, 400 epochs | 0.017667 | 2/10 |
| Fresh Keras, matched epochs | 0.014807 | 6/10 |

The longer independent Keras fit is close to the exponential GP on average (difference -0.000328), rather than an established general advantage. Both independent neural conditions remain above the RBF GP in every fold after averaging seeds. A reused-model score is not an admissible held-out entry in this table.

The earlier independent PyTorch public-code variant had RMSE 0.017772 at 400 epochs, compared with 0.017667 for fresh Keras here. Merely switching to Keras does not recover the paper's reported 0.003466. Because initialization and permutation rules differ between these studies, this is not an isolated framework-effect experiment. Within the current Keras experiment, retaining cross-fold history has a much larger observed score difference than matching training duration alone.

The final-fold acquired-label learning curves also fluctuate substantially under constant learning rate 0.001 and repeated optimizer resets. More epochs under this rule do not establish full convergence. No learning-rate schedule, training-loss checkpoint selection, or alternative regularization was introduced after inspecting these traces.

![Fold-by-fold scores and acquired-label learning curves](../outputs/keras-reuse-audit/fold_reuse_comparison.png)

The supported conclusion is that the inspected script's weight reuse materially changes its evaluation, and the difference persists under matched cumulative updates. This is evidence about this public-code behavior and surface. It does not demonstrate that the published Table S2 was generated by that exact behavior, and the paper's exact numerical values remain unreproduced.

## Interpretation boundaries

This experiment can establish how the inspected public-code behavior affects this implementation's score. It cannot establish that the original paper's tables used that exact script, nor reconstruct the authors' unpublished environment or model-selection decisions. It audits only the unregularized public-code architecture; the four-layer regularized supplement interpretation is a separate configuration. The paper's reported numbers and metric units must be read as published rather than assuming every printed public-code MSE is an RMSE.

All ten folds come from one deterministic function and have overlapping training sets. Initialization ranges are descriptive, not independent-field confidence intervals. Earlier PyTorch scores use a different initialization/permutation protocol; their difference from fresh Keras scores is context, not an isolated framework effect. The GP reference is the previous scikit-learn fit, not a new GPy reproduction. No movement, acquisition policy, or predictive-uncertainty claim is evaluated here.

## Reproduction and artifacts

```sh
# A separate environment avoids changing the existing Torch/RL environment.
.venv/bin/python -m venv .venv-keras
.venv-keras/bin/python -m pip install -e '.[keras-audit]'
mkdir -p external/deepkriging
curl --fail --location https://raw.githubusercontent.com/aleksada/DeepKriging/master/Nonstationary_function-2d.py --output external/deepkriging/Nonstationary_function-2d.py
PYTHONPATH=src .venv-keras/bin/python -m spatial_exploration.keras_audit_study --reference outputs/paper-mapping-study --output outputs/keras-reuse-audit
PYTHONPATH=src .venv-keras/bin/python -m unittest discover -s tests -v
```

The paper-function reference must already exist; use a fresh output directory. `frozen_protocol.json` precedes all fitting and evaluator loading. `reference_manifest.json` identifies immutable source observations and truth. `author_source.py.txt` and its SHA-256 identify the inspected public script. `metrics.csv` includes both MSE and RMSE, training errors, actual exposure counts, update budgets, and weight hashes; `training_history.csv` contains acquired-label-only learning curves. `stage_diagnostics.json` records weight continuity and optimizer steps. `predictions.npz` stores every full-grid prediction. Every result has an HDF5 weights checkpoint, including shared numerical prefixes. Timings count actual stage execution once, while result-level cumulative training costs can share prefixes and must not be summed as independent work.

All 89 test cases are covered across the existing Torch and separate TensorFlow environments: Torch passes 87 and skips the two TensorFlow-specific cases; TensorFlow passes 48 and skips 41 optional Torch/RL cases. System Python passes 46 and skips 43 optional cases. New checks cover source basis support, coupled shuffled batches, acquisition/test separation, equivalence with Keras batch training, optimizer-state resets without weight resets, fold-to-fold weight continuity, shared initial states, update-budget matching, explicit exposure counts, and freeze-before-fit ordering. `environment.txt` records the actual installed package versions.

Post-run verification independently recomputes all 90 reported MSE/RMSE pairs and reloads all 90 HDF5 checkpoints: both score discrepancies and maximum full-grid prediction discrepancies are exactly zero. It verifies every stage's weight continuity, fresh initialization, update/epoch budget, intended test exposure, and unchanged reference hashes. Results are in `verification.json`; `paired_diagnostics.json` adds descriptive paired contrasts and comparisons with the unchanged GP fits. All 90 predictions are finite. These checks validate the implemented comparison, not the historical provenance of the published table.
