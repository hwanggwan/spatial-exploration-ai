# Re-review of the original 2D cross-validation code

Reviewed November 5, 2026, with an explicit attempt to distinguish confirmed defects from harmless uses of evaluation data and unknown publication provenance.

**The inspected Python script retains trained neural weights across folds, contaminating its held-out interpretation from fold two. This does not establish that the final published Table S2 used this script or that DeepKriging itself is invalid.**

## Source and execution scope

The frozen source is commit `0ff987f416cb634a3191719cb61443694397a5c6` of the authors' repository, file `Nonstationary_function-2d.py`, SHA-256 `bc17b535e2607da534aa47c22171f868ab35aef002dd670ecde5aa977a8db1aa`. The corresponding notebook and the older `f321daa` Python source were also inspected. The selected helper functions and model declarations have identical ASTs in both commits.

The new diagnostic, `tools/review_original_cv.py`, extracts the original `deep_model`, `test_model`, `optimal_epoch`, and both model declarations using Python's AST. It executes those function bodies and model declarations unchanged, with a tracing subclass that delegates `compile` and `fit` to Keras. It drives the first two original splits and reduces the two consecutive fits from 200+200 to 2+2 epochs. The 900 locations, 1,830 raw basis features, and 810 training/90 test sites are the archived benchmark data. A separate check confirms all ten archived training and test index arrays exactly match `KFold(10, shuffle=True, random_state=123)`, including row order.

This uses TensorFlow 2.20.0/Keras 3.15.1, fixed diagnostic seeds, and modern dependencies. Notebook magics, deprecated imports, plotting, and GP fitting are not executed. It is a direct helper-function/state check, **not execution of the entire unmodified file, a historical-environment reproduction, or a new accuracy comparison**. Only the evaluator-style review harness accesses the full target vector to trace the original validation/evaluation calls.

## Finding 1: trained weights cross the fold boundary

**Confirmed in the inspected script; high impact for its cross-validation interpretation.**

- The DeepKriging network is constructed at [lines 155–163](https://github.com/aleksada/DeepKriging/blob/0ff987f416cb634a3191719cb61443694397a5c6/Nonstationary_function-2d.py#L155-L163), and the coordinate DNN at lines 171–179, before the fold loop.
- [Lines 217–228](https://github.com/aleksada/DeepKriging/blob/0ff987f416cb634a3191719cb61443694397a5c6/Nonstationary_function-2d.py#L217-L228) pass those same model objects into both fitting helpers on every fold. Neither helper nor the loop recreates the models or restores initial weights.
- `deep_model` recompiles the existing object at lines 100–102. This configures training; it does not reset its Dense-layer weights. In our trace, the maximum weight change caused by every `compile` call is exactly **0**, including at the second fold. The first fold's final weight hash exactly equals the second fold's starting hash for both networks.
- Recompilation does create a new Adam optimizer in this trace: its iteration counter resets from **52 to 0**. Optimizer reset and weight reset are different operations.

For ten disjoint test folds, every second-fold test point belongs to the first-fold training set. More generally, all 90 test points in each later fold have appeared in earlier training pools. The checked exposure counts are `[0, 90, 90, 90, 90, 90, 90, 90, 90, 90]`; within-fold train/test intersections are all zero. Thus the issue is cross-fold training history, even though each current split is internally disjoint.

Both neural methods are affected. In contrast, the [GP constructor is inside the loop](https://github.com/aleksada/DeepKriging/blob/0ff987f416cb634a3191719cb61443694397a5c6/Nonstationary_function-2d.py#L233-L238) and receives only the current training labels. This review identifies no equivalent cross-fold GP state carryover there. Extra neural training time is an additional comparison difference; extra epochs alone are not the definition of leakage.

As a historical semantics cross-check, [Keras 2.2.4's `Model.compile` implementation](https://github.com/keras-team/keras/blob/2.2.4/keras/engine/training.py#L34-L104) configures the optimizer/loss without recreating layer weights. That tag is illustrative historical source, **not an established version of the authors' actual environment**.

For independent CV, construct and compile both networks afresh within every fold, using only that fold's training labels. Resetting the optimizer or metric counters alone is insufficient. The earlier [same-Keras study](keras-reuse-audit.md) quantifies the training-history difference with a cumulative-update control; the present short trace establishes the state transition rather than its universal numerical impact.

## Finding 2: distinguish notebook selection from fixed script epochs

The Python script sets `NB_START_EPOCHS=200` and assigns `model_optim=200` / `model_optim_base=200` at [lines 223–224](https://github.com/aleksada/DeepKriging/blob/0ff987f416cb634a3191719cb61443694397a5c6/Nonstationary_function-2d.py#L223-L224). The `optimal_epoch(...)` text on those lines is commented out. Therefore **test-based epoch selection is not an executed operation in this Python loop**, despite the presence of that helper elsewhere in the file.

The [current notebook](https://github.com/aleksada/DeepKriging/blob/0ff987f416cb634a3191719cb61443694397a5c6/Nonstationary_function-2d.ipynb) differs. In zero-based cell 9, current test targets supply validation loss, `optimal_epoch(history)` selects the additional training duration, and the same test targets provide the reported score. If that intended loop executes, test responses influence a fitted setting, so the resulting score is not an untouched test evaluation. A separate validation split inside the training pool would address this selection issue.

However, the displayed notebook creates `model_4` while the loop uses `model`; cell 5 also appends BatchNormalization layers to `model_base` before its displayed definition in cell 6. Its saved execution counts are nonsequential. It cannot be certified as a clean top-to-bottom execution in its displayed state. The notebook observation is a source-level problem in the intended workflow, not proof that its saved outputs or the final paper used that exact execution.

## Negative controls and non-findings

Passing labels as `validation_data` does **not by itself** train on them. The [Keras API documentation](https://keras.io/api/models/model_training_apis/#fit-method) explicitly separates validation evaluation from training. The Python helpers have no validation-driven callback or active epoch selection in their fixed-epoch loop.

The direct control starts two original networks with identical seeds and training data. It changes only validation/evaluation labels, adding 1,000 to them in the second run. Validation losses change dramatically, while the final learned weight arrays remain **exactly identical, maximum absolute difference 0**. Calling `evaluate` changes neither weights nor optimizer iterations. This control applies to the inspected fixed-settings architecture; it does not excuse using validation results later to choose settings, as in the notebook loop.

Other points that are not response leakage here:

- Training normally uses training labels; that alone is expected supervised learning.
- Current-fold training and test indices are disjoint.
- The Wendland basis uses location geometry and fixed knots, not response values. Knowing all grid coordinates is part of this mapping setup.
- Supplying `targets` to this ordinary `KFold.split` call does not train a model on those targets.
- Continuing a model on genuinely acquired past observations in an online task can be valid. Here the intended claim is held-out CV, and the retained history already contains the current test labels.

## What remains unknown

The [provenance audit](paper-provenance-audit.md) links earlier notebook output means to an earlier manuscript table, but it does not identify the exact execution behind final Table S2. The inspected source supports a specific public-code defect. It does not prove misconduct, invalidate the representation, determine the size or direction of bias in every setting, or establish that every published comparison is wrong.

An independent mapper comparison should start each fold with fresh weights, keep tuning within training data, and freeze settings before scoring the outer test set. This addresses the confirmed defects without assuming the paper's final execution history.

## Local artifacts and rerun

`outputs/original-code-review/review.json` records source identity, function locations, all ten exposure counts, compile/fit traces, model weight hashes, the validation-label control, and notebook cell source. `extracted_helpers.py.txt` preserves the selected original declarations. `split_verification.json` records the independent KFold index check.

```sh
PYTHONPATH=src .venv-keras/bin/python tools/review_original_cv.py --output outputs/original-code-review-rerun
```

Use a fresh output directory; the helper trace refuses to overwrite prior review evidence. The original author files and prior numerical-study outputs are unchanged.
