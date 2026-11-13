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
    from spatial_exploration.deepkriging import (BatchedSpatialMLP, NeuralMapperConfig, calibration_multiplier,
        fit_ensemble_snapshots, matched_coordinate_width, parameter_count, raw_features, standardized_features, wendland)
    from spatial_exploration.deepkriging_study import run_study as run_neural_study
    from spatial_exploration.estimated_gp import EstimatorConfig
    from spatial_exploration.sample_size_study import SampleSizeConfig, run_study as run_gp_study


@unittest.skipUnless(HAS_TORCH, "Install PyTorch to test neural mappers")
class DeepKrigingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_wendland_matches_formula_and_has_exact_compact_support(self):
        np.testing.assert_allclose(wendland([0, 0.5, 1, 2, 1e200]), [1, 83 / 768, 0, 0, 0])
        with self.assertRaises(ValueError):
            wendland([-0.1])

    def test_geometry_features_and_parameter_matched_control(self):
        config = NeuralMapperConfig()
        xx, yy = np.meshgrid(np.linspace(0, 1, 15), np.linspace(0, 1, 15))
        points = np.column_stack((xx.ravel(), yy.ravel()))
        features, center, scale = standardized_features(points, "deepkriging", config)
        self.assertEqual(features.shape, (225, 236))
        np.testing.assert_allclose(features.mean(axis=0), 0, atol=1e-6)
        np.testing.assert_allclose(features.astype(float).std(axis=0), 1, atol=1e-6)
        np.testing.assert_allclose(features * scale + center, raw_features(points, "deepkriging", config), atol=1e-6)
        two_points = raw_features(np.array([[0, 0], [1, 1]]), "deepkriging", config)
        self.assertEqual(two_points[0, 2], 1)
        self.assertEqual(two_points[1, 2], 0)
        coordinate = parameter_count(2, matched_coordinate_width(config))
        spatial = parameter_count(236, config.basis_width)
        self.assertLess(abs(coordinate / spatial - 1), 0.01)

    def test_batched_gradients_equal_independent_fits(self):
        inputs = torch.tensor(np.random.default_rng(0).normal(size=(2, 5, 3)), dtype=torch.float32)
        targets = torch.tensor(np.random.default_rng(1).normal(size=(2, 5)), dtype=torch.float32)
        batch = BatchedSpatialMLP(3, 4, [11, 22])
        batch_output = batch(inputs)
        ((batch_output - targets)**2).mean(1).sum().backward()
        for index, seed in enumerate((11, 22)):
            independent = BatchedSpatialMLP(3, 4, [seed])
            output = independent(inputs[index:index + 1])
            torch.testing.assert_close(output[0], batch_output[index])
            ((output - targets[index:index + 1])**2).mean().backward()
            for combined, single in zip(batch.parameters(), independent.parameters()):
                torch.testing.assert_close(combined.grad[index], single.grad[0], rtol=1e-5, atol=1e-7)

    def test_ensemble_learns_and_checkpoint_reloads_the_same_predictions(self):
        config = NeuralMapperConfig(knot_grids=(2, 3), basis_width=16, learning_rate=0.01, epochs=(1, 100))
        features = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1], [0, 0]], dtype=np.float32)
        indices = np.array([[0, 1, 2, 3]])
        values = np.array([[0, -2, 2, 0]], dtype=float)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            results, _, _ = fit_ensemble_snapshots(features, indices, values, "coordinate_dnn",
                config, [0, 1, 2], 0.0001, (1, 100), save_path=path)
            initial = np.mean((results[1]["mean"][0, :4] - values[0])**2)
            final = np.mean((results[100]["mean"][0, :4] - values[0])**2)
            self.assertLess(final, initial * 0.2)
            self.assertTrue((results[100]["raw_variance"] > 0).all())
            saved = torch.load(path, weights_only=True)
            restored = BatchedSpatialMLP(saved["features"], saved["width"], saved["seeds"])
            restored.load_state_dict(saved["state_dict"])
            with torch.inference_mode():
                np.testing.assert_allclose(restored(torch.as_tensor(features)).numpy(),
                    results[100]["members"][0], atol=1e-7)

    def test_calibration_scales_with_validation_residuals(self):
        mean = np.zeros((2, 20))
        truth = np.linspace(-2, 2, 40).reshape(2, 20)
        variance = np.ones_like(mean) * 0.1
        mask = np.ones(20, dtype=bool)
        first = calibration_multiplier(truth, mean, variance, mask)
        second = calibration_multiplier(2 * truth, mean, variance, mask)
        self.assertAlmostEqual(second, 4 * first)

    @patch("spatial_exploration.deepkriging_study.render_results")
    @patch("spatial_exploration.deepkriging_study.render_uncertainty")
    @patch("spatial_exploration.deepkriging_study.render_example")
    @patch("spatial_exploration.sample_size_study.render_curves")
    @patch("spatial_exploration.sample_size_study.render_diagnostics")
    @patch("spatial_exploration.sample_size_study.render_example")
    def test_selection_freezes_before_benchmark_targets_are_loaded(self, *_):
        environment = SampleSizeConfig(grid_size=5, fields=2, sample_sizes=(4, 8))
        neural = NeuralMapperConfig(knot_grids=(2, 3), basis_width=16, members=2, epochs=(1, 2),
            weight_decays=(0.0001,), tuning_fields=2, calibration_fields=2, chunk_maps=2)
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "gp", Path(directory) / "neural"
            run_gp_study(environment, EstimatorConfig(optimizer_restarts=0), source)
            original_load = np.load

            def checked_load(path, *args, **kwargs):
                if Path(path) == source / "evaluation_only.npz":
                    self.assertTrue((output / "frozen_protocol.json").exists())
                return original_load(path, *args, **kwargs)

            with patch("spatial_exploration.deepkriging_study.np.load", side_effect=checked_load):
                summary, diagnostics = run_neural_study(source, output, neural)
            self.assertEqual(diagnostics["benchmark_metric_rows"], 60)
            self.assertEqual(diagnostics["independent_member_fits"], 48)
            self.assertTrue(diagnostics["source_data_unchanged"])
            self.assertEqual(len(summary), 30)
            frozen = json.loads((output / "frozen_protocol.json").read_text())
            self.assertEqual(len(frozen["selected"]), 4)
            self.assertEqual(len(frozen["variance_multipliers"]), 4)


if __name__ == "__main__":
    unittest.main()
