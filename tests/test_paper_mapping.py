import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from spatial_exploration.paper_mapping import (PaperConfig, PaperNetwork, fit_neural,
        fit_gp, initialization_seed, nested_design, paper_features, paper_signal)
    from spatial_exploration.paper_mapping_study import run_study


@unittest.skipUnless(HAS_TORCH, "Install PyTorch for paper mapping study")
class PaperMappingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = PaperConfig(grid_size=5, folds=3, counts=(4, 8, 16),
            neural_seeds=(0, 1), knot_grids=(2, 3), width=8, batch_size=8,
            paper_epochs=2, code_epochs=3, gp_restarts=0)

    def test_signal_and_public_code_basis_support(self):
        points = np.array([[0., 0.], [.2, .6], [.4, .4], [1., 1.]])
        shifted = points.sum(axis=1) / 2 - .9
        expected = np.sin(30 * shifted**4) * np.cos(2 * shifted) + shifted / 2
        np.testing.assert_allclose(paper_signal(points), expected)
        self.assertEqual(paper_signal(points)[1], paper_signal(points)[2])
        features = paper_features(np.array([[0., 0.], [.25, 0.]]), (10, 19, 37))
        self.assertEqual(features.shape, (2, 1830))
        np.testing.assert_array_equal(features[0, [0, 100, 461]], 1.)
        self.assertEqual(features[1, 0], 0.)  # radius exactly 2.5/10
        self.assertTrue(np.all((features >= 0) & (features <= 1)))

    def test_nested_acquisitions_share_test_without_overlap(self):
        points, designs = nested_design(self.config)
        all_test = []
        for fold in range(self.config.folds):
            chosen = [d for d in designs if d["fold"] == fold]
            all_test.extend(chosen[0]["test"])
            previous = set()
            for design in chosen:
                self.assertEqual(len(design["train"]), design["count"])
                self.assertTrue(previous <= set(design["train"]))
                self.assertFalse(set(design["train"]) & set(design["test"]))
                np.testing.assert_array_equal(design["test"], chosen[0]["test"])
                previous = set(design["train"])
        self.assertEqual(sorted(all_test), list(range(len(points))))
        with self.assertRaises(ValueError):
            nested_design(PaperConfig(grid_size=5, folds=3, counts=(17,)))

    def test_fits_restart_independently_and_checkpoint_reconstructs_prediction(self):
        points, designs = nested_design(self.config)
        features = paper_features(points, self.config.knot_grids)
        indices = designs[0]["train"]
        values = paper_signal(points)[indices]
        seed = initialization_seed(self.config, "dk_paper", 0, 0)
        first = fit_neural(features, indices, values, "dk_paper", self.config, seed)
        fit_neural(features, indices, values + 2., "dk_paper", self.config, seed + 1)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "network.pt"
            repeated = fit_neural(features, indices, values, "dk_paper", self.config, seed, checkpoint)
            state = torch.load(checkpoint, weights_only=True)
            network = PaperNetwork(state["inputs"], state["model"], state["width"])
            network.load_state_dict(state["state_dict"])
            network.eval()
            with torch.inference_mode():
                restored = network(torch.tensor(features)).numpy()
        np.testing.assert_array_equal(first["mean"], repeated["mean"])
        np.testing.assert_array_equal(first["mean"], restored)
        self.assertEqual(first["initial_sha256"], repeated["initial_sha256"])
        self.assertEqual(first["final_sha256"], repeated["final_sha256"])
        self.assertNotEqual(first["initial_sha256"], first["final_sha256"])
        changed = fit_neural(features, indices, values + 2., "dk_paper", self.config, seed)
        self.assertFalse(np.allclose(first["mean"], changed["mean"]))

    def test_gp_and_end_to_end_archive_keep_acquired_labels_separate(self):
        points, designs = nested_design(self.config)
        truth = paper_signal(points)
        indices = designs[2]["train"]
        for model in ("gp_exponential", "gp_rbf"):
            result = fit_gp(points, indices, truth[indices], model, self.config)
            self.assertTrue(np.isfinite(result["mean"]).all())
            self.assertTrue(np.all(result["variance"] >= 0))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "study"
            def guarded_fit(features, train, acquired, *args):
                self.assertTrue((output / "frozen_protocol.json").exists())
                self.assertEqual(len(acquired), len(train))
                np.testing.assert_allclose(acquired, truth[train])
                return fit_neural(features, train, acquired, *args)
            with patch("spatial_exploration.paper_mapping_study.fit_neural", side_effect=guarded_fit), \
                 patch("spatial_exploration.paper_mapping_study.render_results"), \
                 patch("builtins.print"):
                summaries, checks = run_study(output, self.config)
            self.assertEqual(len(summaries), 15)
            self.assertEqual(checks["fits"], 72)
            self.assertTrue(checks["same_test_set_across_counts"])
            self.assertTrue(checks["no_train_test_overlap"])
            self.assertTrue(checks["all_finite_predictions"])
            predictions = np.load(output / "predictions.npz")
            self.assertEqual(predictions["dk_paper"].shape, (3, 3, 2, 25))
            diagnostics = json.loads((output / "fit_diagnostics.json").read_text())
            hashes = {r["initial_sha256"] for r in diagnostics
                      if r["model"] == "dk_paper" and r["fold"] == 0 and r["member"] == 0}
            self.assertEqual(len(hashes), 1)
            other = next(r["initial_sha256"] for r in diagnostics
                         if r["model"] == "dk_paper" and r["fold"] == 1 and r["member"] == 0)
            self.assertNotIn(other, hashes)


if __name__ == "__main__":
    unittest.main()
