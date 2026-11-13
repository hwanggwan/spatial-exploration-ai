import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.model_selection import KFold

from spatial_exploration.baseline import make_grid
from spatial_exploration.keras_audit import AuditConfig, KerasTrainer, source_features, stage_orders, weight_hash, tensorflow
from spatial_exploration.keras_audit_study import load_acquisitions, run_study

HAS_TENSORFLOW = importlib.util.find_spec("tensorflow") is not None


def tiny_reference(path):
    path.mkdir()
    points = make_grid(3)
    truth = np.sin(points[:, 0]*3) + points[:, 1]**2
    designs = [{"fold": f, "count": 6, "train": train.tolist(), "test": test.tolist()}
               for f, (train, test) in enumerate(KFold(3, shuffle=True, random_state=123).split(points))]
    (path / "splits.json").write_text(json.dumps(designs))
    np.savez_compressed(path / "acquired.npz", points=points, indices=[d["train"] for d in designs],
        values=[truth[d["train"]] for d in designs], folds=np.arange(3), counts=[6]*3)
    np.savez_compressed(path / "evaluation_only.npz", points=points, truth=truth)
    (path / "summary.json").write_text("[]")
    return points, truth, designs


class AuditDesignTests(unittest.TestCase):
    def test_public_basis_and_coupled_stage_permutations(self):
        features = source_features(np.array([[0., 0.], [.25, 0.]]), (10, 19, 37))
        self.assertEqual(features.shape, (2, 1830))
        np.testing.assert_array_equal(features[0, [0, 100, 461]], 1.)
        self.assertEqual(features[1, 0], 0.)
        config = AuditConfig(stage_epochs=3)
        orders = stage_orders(config, 0, 0, 810)
        np.testing.assert_array_equal(np.sort(orders, axis=1), np.tile(np.arange(810), (3, 1)))
        np.testing.assert_array_equal(orders, stage_orders(config, 0, 0, 810))
        self.assertFalse(np.array_equal(orders, stage_orders(config, 0, 1, 810)))

    def test_reference_uses_stored_acquisitions_and_rejects_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "reference"
            points, truth, designs = tiny_reference(reference)
            actual, acquisitions = load_acquisitions(reference, AuditConfig(folds=3, count=6))
            np.testing.assert_array_equal(actual, points)
            for row in acquisitions:
                np.testing.assert_array_equal(row["values"], truth[row["train"]])
            designs[0]["test"][0] = designs[0]["train"][0]
            (reference / "splits.json").write_text(json.dumps(designs))
            with self.assertRaises(ValueError):
                load_acquisitions(reference, AuditConfig(folds=3, count=6))


@unittest.skipUnless(HAS_TENSORFLOW, "Install TensorFlow for same-Keras audit")
class KerasTrainingTests(unittest.TestCase):
    def test_compiled_loop_matches_keras_batches_and_optimizer_reset(self):
        tf = tensorflow()
        config = AuditConfig(folds=3, count=6, seeds=(0,), width=8, batch_size=4,
                             stage_epochs=3, chunk_epochs=2, knot_grids=(2, 3))
        first = KerasTrainer(13, config, 7)
        second = KerasTrainer(13, config, 8, first.model.get_weights())
        second.model.compile(optimizer=second.optimizer, loss="mse", jit_compile=False)
        rng = np.random.default_rng(2)
        x, y = rng.normal(size=(6, 13)).astype(np.float32), rng.normal(size=6).astype(np.float32)
        orders = stage_orders(config, 0, 0, 6)
        result = first.fit_stage(x, y, orders)
        for order in orders:
            for start in range(0, len(x), config.batch_size):
                ids = order[start:start+config.batch_size]
                second.model.train_on_batch(x[ids], y[ids])
        for actual, expected in zip(first.model.get_weights(), second.model.get_weights()):
            np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=2e-6)
        self.assertEqual(result["updates"], 6)
        digest = weight_hash(first.model.get_weights())
        first.reset_optimizer()
        self.assertEqual(int(first.optimizer.iterations.numpy()), 0)
        self.assertAlmostEqual(float(first.optimizer.learning_rate.numpy()), .001, places=8)
        self.assertEqual(digest, weight_hash(first.model.get_weights()))
        for variable in first.optimizer.variables:
            if variable is not first.optimizer.learning_rate:
                self.assertTrue(np.all(variable.numpy() == 0))

    def test_full_workflow_preserves_folds_budget_coupling_and_weight_history(self):
        config = AuditConfig(folds=3, count=6, seeds=(0,), width=8, batch_size=4,
                             stage_epochs=2, chunk_epochs=1, knot_grids=(2, 3))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, output, source = root / "reference", root / "output", root / "source.py"
            tiny_reference(reference)
            source.write_text("# source snapshot for workflow test\n")
            original_fit = KerasTrainer.fit_stage
            def guarded_fit(trainer, features, acquired, orders):
                self.assertTrue((output / "frozen_protocol.json").exists())
                self.assertEqual(len(acquired), 6)
                return original_fit(trainer, features, acquired, orders)
            with patch.object(KerasTrainer, "fit_stage", guarded_fit), \
                 patch("spatial_exploration.keras_audit_study.render_results"), patch("builtins.print"):
                summary, checks = run_study(output, reference, source, config)
            self.assertEqual(len(summary), 3)
            self.assertEqual(checks["recorded_results"], 9)
            self.assertEqual(checks["actual_stage_fits"], 8)  # reuse3 + fresh2+3
            for name in ("all_finite_predictions", "all_fresh_test_sites_unseen",
                         "all_later_reuse_test_sites_previously_seen", "shared_fold_zero_predictions", "matched_update_budgets"):
                self.assertTrue(checks[name], name)
            records = json.loads((output / "stage_diagnostics.json").read_text())
            reuse = [r for r in records if r["stream"] == "reuse"]
            for previous, current in zip(reuse, reuse[1:]):
                self.assertEqual(previous["final_sha256"], current["start_stage_sha256"])
            initial = reuse[0]["start_stage_sha256"]
            self.assertTrue(all(r["start_stage_sha256"] == initial for r in records
                                if r["stream"] == "fresh" and r["stage"] == 0))
            predictions = np.load(output / "predictions.npz")["predictions"]
            self.assertEqual(predictions.shape, (3, 1, 3, 9))
            self.assertEqual(len(list((output / "checkpoints").glob("*.weights.h5"))), 9)


if __name__ == "__main__":
    unittest.main()
