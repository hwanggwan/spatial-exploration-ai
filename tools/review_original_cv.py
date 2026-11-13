"""Trace the author's actual helper functions; do not execute notebook/GP code.

This review tests state semantics with 2+2 epochs, not historical numerical
reproduction. Source text/model declarations are extracted unchanged via AST.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from spatial_exploration.keras_audit import AuditConfig, source_features, tensorflow, weight_hash
from spatial_exploration.keras_audit_study import load_acquisitions


def extract_blocks(source):
    module = ast.parse(source)
    chosen = []
    locations = {}
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name in {"deep_model", "test_model", "optimal_epoch"}:
            chosen.append(node)
            locations[node.name] = [node.lineno, node.end_lineno]
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {"model", "model_base"} for t in node.targets):
            chosen.append(node)
            locations[ast.unparse(node.targets[0])] = [node.lineno, node.end_lineno]
        elif (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
              and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "add"
              and isinstance(node.value.func.value, ast.Name) and node.value.func.value.id in {"model", "model_base"}):
            chosen.append(node)
    if set(locations) != {"deep_model", "test_model", "optimal_epoch", "model", "model_base"}:
        raise ValueError("Unexpected author source structure; inspect manually")
    selected = ast.Module(body=chosen, type_ignores=[])
    return compile(selected, "author_extracted_helpers", "exec"), ast.unparse(selected), locations


def run_review(output, repository, reference):
    output, repository, reference = Path(output), Path(repository), Path(reference)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh review directory")
    output.mkdir(parents=True, exist_ok=True)
    source_path = repository / "Nonstationary_function-2d.py"
    source = source_path.read_text()
    compiled, extracted, locations = extract_blocks(source)
    (output / "extracted_helpers.py.txt").write_text(extracted + "\n")
    tf = tensorflow()
    log = []

    class TracedSequential(tf.keras.Sequential):
        def compile(self, *args, **kwargs):
            before = self.get_weights()
            old = getattr(self, "optimizer", None)
            old_iterations = None if old is None else int(old.iterations.numpy())
            result = super().compile(*args, **kwargs)
            after = self.get_weights()
            log.append({"event": "compile", "name": getattr(self, "review_name", None),
                "fold": getattr(self, "review_fold", None), "object_id": id(self),
                "weights_before": weight_hash(before), "weights_after": weight_hash(after),
                "largest_weight_change": max(float(np.max(np.abs(a-b))) for a, b in zip(before, after)),
                "old_optimizer_iterations": old_iterations, "new_optimizer_iterations": int(self.optimizer.iterations.numpy()),
                "new_optimizer_object": old is not self.optimizer})
            return result

        def fit(self, *args, **kwargs):
            before = weight_hash(self.get_weights())
            begin_steps = int(self.optimizer.iterations.numpy())
            result = super().fit(*args, **kwargs)
            log.append({"event": "fit", "name": getattr(self, "review_name", None),
                "fold": getattr(self, "review_fold", None), "object_id": id(self),
                "weights_before": before, "weights_after": weight_hash(self.get_weights()),
                "optimizer_steps_before": begin_steps, "optimizer_steps_after": int(self.optimizer.iterations.numpy()),
                "epochs": kwargs["epochs"], "acquired_labels": len(args[1]),
                "validation_supplied": kwargs.get("validation_data") is not None})
            return result

    def namespace(seed, features):
        tf.keras.utils.set_random_seed(seed)
        values = {"np": np, "Sequential": TracedSequential, "Dense": tf.keras.layers.Dense,
                  "Dropout": tf.keras.layers.Dropout, "BatchNormalization": tf.keras.layers.BatchNormalization,
                  "K": features, "NB_START_EPOCHS": 2, "BATCH_SIZE": 64}
        exec(compiled, values)
        return values

    config = AuditConfig()
    points, designs = load_acquisitions(reference, config)
    features = source_features(points, config.knot_grids)
    truth = np.load(reference / "evaluation_only.npz")["truth"]
    seen, exposure = set(), []
    for design in designs:
        exposure.append({"fold": design["fold"], "within_fold_overlap": len(set(design["train"]) & set(design["test"])),
            "test_sites": len(design["test"]), "previous_training_overlap": len(seen & set(design["test"]))})
        seen.update(design["train"])
    assert [r["previous_training_overlap"] for r in exposure] == [0] + [90]*9
    assert all(r["within_fold_overlap"] == 0 for r in exposure)

    values = namespace(20261105, features.shape[1])
    states = []
    for design in designs[:2]:
        for name, matrix in (("model", features), ("model_base", points.astype(np.float32))):
            model = values[name]
            model.review_name, model.review_fold = name, design["fold"]
            previous = weight_hash(model.get_weights())
            train, test = design["train"], design["test"]
            values["deep_model"](model, matrix[train], design["values"], matrix[test], truth[test], "continuous")
            values["test_model"](model, matrix[train], design["values"], matrix[test], truth[test], 2)
            states.append({"name": name, "fold": design["fold"], "object_id": id(model),
                "before_deep_model": previous, "after_test_model": weight_hash(model.get_weights())})
    for name in ("model", "model_base"):
        stages = [r for r in states if r["name"] == name]
        assert stages[0]["object_id"] == stages[1]["object_id"]
        assert stages[0]["after_test_model"] == stages[1]["before_deep_model"]
        assert stages[0]["before_deep_model"] != stages[0]["after_test_model"]
    source_trace = list(log)
    compiles = [r for r in source_trace if r["event"] == "compile"]
    assert len(compiles) == 4
    assert all(r["largest_weight_change"] == 0 for r in compiles)
    assert all(r["new_optimizer_iterations"] == 0 and r["new_optimizer_object"] for r in compiles)
    assert [r["old_optimizer_iterations"] for r in compiles if r["fold"] == 1] == [52, 52]

    # Negative control: change ONLY validation/evaluation labels, keeping current
    # training labels, initialization, seeds, and original function bodies fixed.
    counterfactual = []
    design = designs[0]
    for shift in (0., 1000.):
        trial = namespace(772211, features.shape[1])
        model = trial["model"]
        model.review_name = "validation_negative_control"
        model.review_fold = 0
        train, test = design["train"], design["test"]
        history = trial["deep_model"](model, features[train], design["values"], features[test], truth[test] + shift, "continuous")
        before_evaluation = weight_hash(model.get_weights())
        iterations = int(model.optimizer.iterations.numpy())
        scores = model.evaluate(features[test], truth[test] + shift, verbose=0)
        assert before_evaluation == weight_hash(model.get_weights())
        assert iterations == int(model.optimizer.iterations.numpy())
        counterfactual.append({"validation_label_shift": shift, "final_weights_sha256": before_evaluation,
            "weights": [v.copy() for v in model.get_weights()], "validation_losses": history.history["val_loss"],
            "evaluation_loss": float(scores[0]), "evaluate_changed_weights": False})
    difference = max(float(np.max(np.abs(a-b))) for a, b in zip(counterfactual[0]["weights"], counterfactual[1]["weights"]))
    assert difference == 0, difference
    assert counterfactual[0]["validation_losses"] != counterfactual[1]["validation_losses"]
    for record in counterfactual:
        del record["weights"]

    notebook = json.loads((repository / "Nonstationary_function-2d.ipynb").read_text())
    notebook_review = []
    for i, cell in enumerate(notebook["cells"]):
        text = "".join(cell.get("source", []))
        if "kfold.split" in text or "def optimal_epoch" in text:
            notebook_review.append({"cell_index": i, "execution_count": cell.get("execution_count"), "source": text})
    historical = {}
    for revision in ("f321daa", "0ff987f"):
        old_source = subprocess.check_output(["git", "-C", str(repository), "show", revision + ":Nonstationary_function-2d.py"]).decode()
        _, old_blocks, old_locations = extract_blocks(old_source)
        historical[revision] = {"selected_source_same_as_head": old_blocks == extracted, "locations": old_locations}
    report = {"scope": "review of inspected public artifacts, not proof of final paper execution",
        "commit": subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"]).decode().strip(),
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(), "source_locations": locations,
        "runtime": {"tensorflow": tf.__version__, "keras": tf.keras.__version__},
        "runtime_deviations": "Only original helper functions and model declarations extracted unchanged; globals set to 2+2 epochs; modern Keras; two folds; notebook/import/GP code not executed",
        "compile_preserves_all_neural_weights": True, "compile_resets_adam_state": True,
        "learned_weights_cross_fold_boundary": True, "affected_models": ["DeepKriging", "coordinate DNN"],
        "fold_exposure": exposure, "source_helper_trace": source_trace, "source_model_states": states,
        "validation_negative_control": {"largest_final_weight_difference": difference, "trials": counterfactual,
            "conclusion": "Supplying validation labels and evaluate alone do not train weights under fixed settings"},
        "historical_source_blocks": historical, "notebook_relevant_cells": notebook_review,
        "not_claimed": ["That validation_data alone performs gradient training", "That optimal_epoch is called by the .py CV loop", "That the entire unmodified file runs on modern dependencies", "That final Table S2 used this code", "That this confirms all reported comparisons are wrong"]}
    (output / "review.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"commit": report["commit"], "compile_preserves_all_neural_weights": True,
        "learned_weights_cross_fold_boundary": True, "prior_test_exposure_counts": [r["previous_training_overlap"] for r in exposure],
        "validation_label_shift_final_weight_difference": difference, "original_compile_events": compiles}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/original-code-review"))
    parser.add_argument("--repository", type=Path, default=Path("external/deepkriging-history"))
    parser.add_argument("--reference", type=Path, default=Path("outputs/paper-mapping-study"))
    args = parser.parse_args()
    run_review(args.output, args.repository, args.reference)
