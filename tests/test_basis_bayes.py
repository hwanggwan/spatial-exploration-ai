import unittest

import numpy as np

from spatial_exploration.basis_bayes import BayesianBasisModel


class BayesianBasisTests(unittest.TestCase):
    def test_function_posterior_matches_independent_coefficient_posterior(self):
        features = np.random.default_rng(70).normal(size=(9, 3))
        model = BayesianBasisModel(features, 0.15, 1.7)
        indices, values = np.array([0, 3, 7]), np.array([0.8, -0.5, 1.1])
        design = model.features[indices]
        beta_covariance = np.linalg.inv(np.eye(4) / 1.7 + design.T @ design / 0.15**2)
        beta_mean = beta_covariance @ design.T @ values / 0.15**2
        result = model.condition(indices, values)
        np.testing.assert_allclose(result.mean, model.features @ beta_mean, atol=1e-12)
        np.testing.assert_allclose(result.covariance, model.features @ beta_covariance @ model.features.T, atol=1e-12)
        # Observation noise enters conditioning, not the returned latent bands.
        self.assertLess(result.covariance[0, 0], 0.15**2)

    def test_covariance_is_value_independent_and_updates_reduce_variance(self):
        model = BayesianBasisModel(np.random.default_rng(3).normal(size=(12, 5)), 0.1, 1.0)
        previous = np.diag(model.prior_covariance)
        for count in range(1, 8):
            belief = model.condition(np.arange(count), np.arange(count) * 0.2)
            changed = model.condition(np.arange(count), np.arange(count) * -100)
            np.testing.assert_array_equal(belief.covariance, changed.covariance)
            self.assertTrue(np.all(np.diag(belief.covariance) <= previous + 1e-12))
            self.assertGreaterEqual(np.linalg.eigvalsh(belief.covariance).min(), -1e-12)
            previous = np.diag(belief.covariance)

    def test_intercept_allows_nonzero_grid_mean_and_invalid_observations_fail(self):
        features = np.array([[-1.0], [1.0]])
        model = BayesianBasisModel(features, 0.1, 1)
        self.assertGreater(model.condition([0, 1], [1, 1]).mean.mean(), 0.9)
        for indices, values in (([0, 0], [1, 2]), ([2], [1]), ([0.5], [1]), ([0], [np.nan])):
            with self.assertRaises(ValueError):
                model.condition(indices, values)
        for noise, prior in ((0, 1), (0.1, -1), (0.1, np.nan)):
            with self.assertRaises(ValueError):
                BayesianBasisModel(features, noise, prior)


if __name__ == "__main__":
    unittest.main()
