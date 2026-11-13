import unittest

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

from spatial_exploration.baseline import Config, GPBelief, explore, make_grid, rbf_covariance
from spatial_exploration.estimated_gp import EstimatedGPBelief, EstimatorConfig


class EstimatedGPTests(unittest.TestCase):
    def test_no_parameter_fit_before_minimum_observations(self):
        points = make_grid(4)
        config = EstimatorConfig(min_observations=4, optimizer_restarts=0)
        belief = EstimatedGPBelief(points, 0.1, config)
        reference = GPBelief(config.initial_process_variance * rbf_covariance(
            points, config.initial_length_scale), 0.1)
        for index, value in ((0, 0.2), (5, -0.5), (10, 0.8)):
            belief.observe(index, value)
            reference.observe(index, value)
        self.assertEqual(belief.fit_history, [])
        np.testing.assert_allclose(belief.mean, reference.mean)
        np.testing.assert_allclose(belief.covariance, reference.covariance)

    def test_refitted_posterior_matches_independent_batch_gp(self):
        points = make_grid(4)
        config = EstimatorConfig(min_observations=4, refit_every=3, optimizer_restarts=0)
        belief = EstimatedGPBelief(points, 0.1, config)
        indices = [0, 1, 5, 6, 9, 10, 14]
        values = [0.2, 0.4, 0.3, -0.2, 0.1, -0.5, 0.2]
        for index, value in zip(indices, values):
            belief.observe(index, value)
        self.assertEqual([item["measurements"] for item in belief.fit_history], [4, 7])
        kernel = ConstantKernel(belief.process_variance, "fixed") * RBF(
            belief.length_scale, "fixed")
        reference = GaussianProcessRegressor(kernel=kernel, alpha=0.01, optimizer=None)
        reference.fit(points[indices], values)
        mean, covariance = reference.predict(points, return_cov=True)
        np.testing.assert_allclose(belief.mean, mean, atol=1e-10)
        np.testing.assert_allclose(belief.covariance, covariance, atol=1e-10)
        self.assertTrue(all(item["accepted"] for item in belief.fit_history))
        self.assertTrue(np.isfinite(belief.std).all())

    def test_unacquired_values_cannot_change_estimated_planner(self):
        points = make_grid(5)
        allowed = np.arange(len(points)) % 5 != 4
        config = Config(grid_size=5, travel_budget=2, seeds=1)
        estimator = EstimatorConfig(optimizer_restarts=0)
        values = np.sin(points[:, 0] * 8) + points[:, 1]
        first_belief = EstimatedGPBelief(points, 0.1, estimator)
        _, first_route, distances = explore(points, None, values, allowed, config,
            "greedy", np.random.default_rng(0), belief=first_belief)
        altered_values = values.copy()
        untouched = np.ones(len(points), dtype=bool)
        untouched[first_route] = False
        altered_values[untouched] = 10000
        second_belief = EstimatedGPBelief(points, 0.1, estimator)
        _, second_route, _ = explore(points, None, altered_values, allowed, config,
            "greedy", np.random.default_rng(0), belief=second_belief)
        self.assertEqual(first_route, second_route)
        np.testing.assert_allclose(first_belief.mean, second_belief.mean)
        self.assertLessEqual(distances[-1], config.travel_budget + 1e-12)
        self.assertTrue(np.all(allowed[first_route]))

    def test_invalid_estimator_configuration_is_rejected(self):
        for config in (EstimatorConfig(initial_length_scale=0),
                       EstimatorConfig(refit_every=0),
                       EstimatorConfig(min_observations=1)):
            with self.assertRaises(ValueError):
                config.validate()


if __name__ == "__main__":
    unittest.main()
