import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

from spatial_exploration.baseline import make_grid
from spatial_exploration.estimated_gp import EstimatedGPBelief, EstimatorConfig
from spatial_exploration.sample_size_study import (SampleSizeConfig, SCENARIOS, generate_dataset,
    generating_covariance, fit_snapshot, run_study)


class SampleSizeStudyTests(unittest.TestCase):
    def test_nonstationarity_changes_dependence_without_changing_marginal_variance(self):
        config = SampleSizeConfig(grid_size=5, fields=2, sample_sizes=(4, 8))
        points = make_grid(5)
        for scenario in SCENARIOS:
            covariance = generating_covariance(points, scenario, config)
            np.testing.assert_allclose(covariance, covariance.T, atol=1e-14)
            np.testing.assert_allclose(np.diag(covariance), 1, atol=1e-14)
            self.assertGreater(np.linalg.eigvalsh(covariance).min(), -1e-12)
            bottom, top = covariance[0, 1], covariance[20, 21]
            if scenario == "nonstationary_mixture":
                self.assertGreater(bottom, top + 0.1)
            else:
                self.assertAlmostEqual(bottom, top)

    def test_dataset_is_reproducible_and_holds_out_the_same_sites(self):
        config = SampleSizeConfig(grid_size=5, fields=2, sample_sizes=(4, 8, 16))
        observations, evaluation = generate_dataset(config)
        repeated, repeated_evaluation = generate_dataset(config)
        for key in observations:
            np.testing.assert_array_equal(observations[key], repeated[key])
        for key in evaluation:
            np.testing.assert_array_equal(evaluation[key], repeated_evaluation[key])
        self.assertFalse(evaluation["evaluation_mask"][observations["observation_indices"]].any())
        self.assertEqual(observations["observed_values"].shape, (3, 2, 16))
        self.assertNotIn("truth", observations)
        self.assertNotIn("true_covariances", observations)
        for order in observations["observation_indices"]:
            self.assertEqual(len(set(order)), 16)

    def test_fresh_snapshot_matches_batch_gp_and_has_one_fit(self):
        points = make_grid(5)
        indices = np.array([0, 1, 3, 6, 10, 13, 17, 21])
        values = np.array([0.2, -0.3, 0.7, -0.2, 0.1, -0.7, 0.8, -0.2])
        config = EstimatorConfig(optimizer_restarts=0)
        belief = EstimatedGPBelief.from_observations(points, 0.1, indices, values, config)
        self.assertEqual(len(belief.fit_history), 1)
        self.assertEqual(belief.fit_history[0]["measurements"], 8)
        kernel = ConstantKernel(belief.process_variance, "fixed") * RBF(belief.length_scale, "fixed")
        reference = GaussianProcessRegressor(kernel=kernel, alpha=0.01, optimizer=None)
        reference.fit(points[indices], values)
        mean, covariance = reference.predict(points, return_cov=True)
        np.testing.assert_allclose(belief.mean, mean, atol=1e-10)
        np.testing.assert_allclose(belief.covariance, covariance, atol=1e-10)
        indices[0], values[0] = 2, 1000
        self.assertEqual(belief.indices[0], 0)
        self.assertEqual(belief.values[0], 0.2)

    def test_snapshot_rejects_invalid_observations(self):
        points = make_grid(3)
        for indices, values in (([0, 0], [1, 2]), ([0, 20], [1, 2]),
            ([0.0, 1.0], [1, 2]), ([0, 1], [1, np.nan]), ([0, 1], [1]), ([], [])):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                EstimatedGPBelief.from_observations(points, 0.1, indices, values)

    def test_oracle_matches_joint_conditioning_and_unknown_fit_ignores_true_kernel(self):
        config = SampleSizeConfig(grid_size=5, fields=2, sample_sizes=(4, 8, 16))
        observed, evaluation = generate_dataset(config)
        points = observed["points"]
        estimator = EstimatorConfig(optimizer_restarts=0)
        prior = evaluation["true_covariances"][2]
        indices, values = observed["observation_indices"][0, :8], observed["observed_values"][2, 0, :8]
        copies = [item.copy() for item in (points, indices, values, prior)]
        oracle = fit_snapshot("oracle", points, indices, values, 0.1, estimator, oracle_prior=prior)
        cross = prior[:, indices]
        noisy = prior[np.ix_(indices, indices)] + 0.01 * np.eye(8)
        np.testing.assert_allclose(oracle.covariance, prior - cross @ np.linalg.solve(noisy, cross.T), atol=1e-12)
        np.testing.assert_allclose(oracle.mean, cross @ np.linalg.solve(noisy, values), atol=1e-12)
        first = fit_snapshot("estimated", points, indices, values, 0.1, estimator)
        second = fit_snapshot("estimated", points, indices, values, 0.1, estimator, oracle_prior=100 * np.eye(25))
        np.testing.assert_array_equal(first.mean, second.mean)
        np.testing.assert_array_equal(first.covariance, second.covariance)
        for before, after in zip(copies, (points, indices, values, prior)):
            np.testing.assert_array_equal(before, after)

    @patch("spatial_exploration.sample_size_study.render_diagnostics")
    @patch("spatial_exploration.sample_size_study.render_curves")
    @patch("spatial_exploration.sample_size_study.render_example")
    def test_workflow_gives_every_mapper_the_same_nested_prefixes(self, *_):
        config = SampleSizeConfig(grid_size=5, fields=2, sample_sizes=(4, 8))
        with tempfile.TemporaryDirectory() as directory, patch(
            "spatial_exploration.sample_size_study.fit_snapshot", wraps=fit_snapshot) as spy:
            summaries, diagnostics = run_study(config, EstimatorConfig(optimizer_restarts=0), directory)
            self.assertEqual(diagnostics["mapper_evaluations"], 36)
            self.assertEqual(diagnostics["fit_calls"], 12)
            self.assertEqual(len(summaries), 18)
            data = np.load(Path(directory) / "observations.npz")
            for position in range(0, len(spy.call_args_list), 3):
                calls = spy.call_args_list[position:position + 3]
                for call in calls[1:]:
                    np.testing.assert_array_equal(call.args[2], calls[0].args[2])
                    np.testing.assert_array_equal(call.args[3], calls[0].args[3])
                    self.assertNotIn("oracle_prior", call.kwargs)
            for scenario in range(3):
                for field in range(2):
                    start = (scenario * 2 + field) * 6
                    small = spy.call_args_list[start]
                    large = spy.call_args_list[start + 3]
                    np.testing.assert_array_equal(small.args[2], large.args[2][:4])
                    np.testing.assert_array_equal(small.args[3], large.args[3][:4])
                    np.testing.assert_array_equal(large.args[2], data["observation_indices"][field])
            with (Path(directory) / "metrics.csv").open() as stream:
                metrics = list(csv.DictReader(stream))
            self.assertEqual(len(metrics), 36)
            self.assertTrue(json.loads((Path(directory) / "diagnostics.json").read_text())["all_oracle_variance_curves_nonincreasing"])
            with self.assertRaises(ValueError):
                run_study(config, EstimatorConfig(), directory)

    def test_sample_sizes_fit_the_fixed_domain_and_increase_strictly(self):
        SampleSizeConfig().validate()
        for sizes in ((8, 8), (8, 4), (0, 8), (8, 200)):
            with self.subTest(sizes=sizes), self.assertRaises(ValueError):
                SampleSizeConfig(sample_sizes=sizes).validate()


if __name__ == "__main__":
    unittest.main()
