import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from spatial_exploration.baseline import make_grid, rbf_covariance
from spatial_exploration.sample_size_study import SampleSizeConfig
from spatial_exploration.transition_fields import (WIDTHS, transition_weights, transition_covariances,
                                                  generate_transition_dataset)

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    from spatial_exploration.deepkriging import NeuralMapperConfig
    from spatial_exploration.estimated_gp import EstimatorConfig
    from spatial_exploration.transition_study import run_study


class TransitionFieldTests(unittest.TestCase):
    def test_weights_keep_variance_constant_and_sharpen_without_changing_lengths(self):
        y = np.array([0, 0.4, 0.5, 0.6, 1])
        for width in WIDTHS:
            a, b = transition_weights(y, width)
            np.testing.assert_allclose(a*a + b*b, 1, atol=1e-14)
            self.assertTrue((np.diff(b) > 0).all())
            self.assertAlmostEqual(a[2], b[2])
        a, b = transition_weights(y, None)
        np.testing.assert_allclose(a, np.cos(np.pi*y/2))
        np.testing.assert_allclose(b, np.sin(np.pi*y/2))
        self.assertLess(transition_weights(y, 0.05)[1][1], transition_weights(y, 0.1)[1][1])
        self.assertGreater(transition_weights(y, 0.05)[1][3], transition_weights(y, 0.1)[1][3])
        for bad in (0, -1, np.nan):
            with self.assertRaises(ValueError):
                transition_weights(y, bad)

    def test_covariances_are_valid_and_cross_boundary_correlation_decreases(self):
        config = SampleSizeConfig()
        points = make_grid(9)
        covariance = transition_covariances(points, config)
        for matrix in covariance:
            np.testing.assert_allclose(matrix, matrix.T, atol=1e-14)
            np.testing.assert_allclose(np.diag(matrix), 1, atol=1e-14)
            self.assertGreater(np.linalg.eigvalsh(matrix).min(), -1e-12)
        points = np.array([[0.5, 0.4], [0.5, 0.6]])
        bridge = transition_covariances(points, config)[:, 0, 1]
        self.assertTrue((np.diff(bridge) < 0).all())
        original = transition_covariances(points, config)[0]
        a,b = transition_weights(points[:,1],None)
        np.testing.assert_allclose(original, np.outer(a,a)*rbf_covariance(points,0.35)
                                   + np.outer(b,b)*rbf_covariance(points,0.12))

    def test_paired_dataset_shares_noise_design_and_center_field_values(self):
        config = SampleSizeConfig(grid_size=5, fields=2, sample_sizes=(4,8))
        obs, ev = generate_transition_dataset(config)
        repeated, repeated_ev = generate_transition_dataset(config)
        for key in obs:
            np.testing.assert_array_equal(obs[key],repeated[key])
        for key in ev:
            np.testing.assert_array_equal(ev[key],repeated_ev[key])
        self.assertNotIn("truth",obs)
        self.assertNotIn("true_covariances",obs)
        self.assertFalse(ev["evaluation_mask"][obs["observation_indices"]].any())
        residual = obs["observed_values"] - np.take_along_axis(ev["truth"],
            np.broadcast_to(obs["observation_indices"],obs["observed_values"].shape),axis=2)
        np.testing.assert_allclose(residual[0],residual[1],atol=1e-14)
        np.testing.assert_allclose(residual[0],residual[2],atol=1e-14)
        center = np.isclose(obs["points"][:,1],0.5)
        np.testing.assert_allclose(ev["truth"][0,:,center],ev["truth"][1,:,center],atol=1e-14)
        np.testing.assert_allclose(ev["truth"][0,:,center],ev["truth"][2,:,center],atol=1e-14)


@unittest.skipUnless(HAS_TORCH,"Install PyTorch to test the full transition comparison")
class TransitionWorkflowTests(unittest.TestCase):
    @patch("spatial_exploration.transition_study.render_transition_results")
    def test_benchmark_generation_follows_freeze_and_all_mappers_share_observations(self, *_):
        env = SampleSizeConfig(grid_size=5,fields=2,sample_sizes=(4,8),field_seed_start=2500000,design_seed_start=2600000)
        neural = NeuralMapperConfig(knot_grids=(2,3),basis_width=16,members=2,epochs=(1,2),
            weight_decays=(0.0001,),tuning_fields=2,calibration_fields=2,chunk_maps=2)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            def checked_generator(config):
                if config.field_seed_start == env.field_seed_start:
                    self.assertTrue((output/'frozen_protocol.json').exists())
                    self.assertFalse((output/'benchmark_evaluation_only.npz').exists())
                return generate_transition_dataset(config)
            from spatial_exploration.transition_study import fit_snapshot
            with patch("spatial_exploration.transition_study.generate_transition_dataset",side_effect=checked_generator), patch(
                "spatial_exploration.transition_study.fit_snapshot",wraps=fit_snapshot) as fitting:
                summaries,diagnostics=run_study(output,env,neural,EstimatorConfig(optimizer_restarts=0))
            self.assertEqual(len(summaries),24)
            self.assertEqual(diagnostics['mapper_evaluations'],48)
            self.assertEqual(diagnostics['neural_member_fits'],48)
            self.assertTrue(diagnostics['all_true_covariance_diagonals_unit'])
            observed=np.load(output/'benchmark_observations.npz')
            # Fit calls are grouped by count, then mapper, then condition/field.
            for count_index,count in enumerate(env.sample_sizes):
                for row in range(6):
                    oracle=fitting.call_args_list[count_index*12+row]
                    estimated=fitting.call_args_list[count_index*12+6+row]
                    np.testing.assert_array_equal(oracle.args[2],estimated.args[2])
                    np.testing.assert_array_equal(oracle.args[3],estimated.args[3])
                    np.testing.assert_array_equal(estimated.args[3],observed['observed_values'][row//2,row%2,:count])
                    self.assertNotIn('oracle_prior',estimated.kwargs)
            self.assertTrue(json.loads((output/'frozen_protocol.json').read_text())['frozen_before_benchmark_generation'])


if __name__ == '__main__':
    unittest.main()
