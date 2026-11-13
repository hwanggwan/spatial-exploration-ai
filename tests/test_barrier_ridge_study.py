import importlib.util
import unittest

import numpy as np


HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    from spatial_exploration.barrier_ridge_study import ridge_posterior


@unittest.skipUnless(HAS_TORCH, "Install PyTorch for shared graph benchmark tooling")
class BarrierRidgeTests(unittest.TestCase):
    def test_batched_dual_fit_matches_independent_primal_ridge(self):
        rng = np.random.default_rng(1701)
        features = rng.normal(size=(13, 21))
        indices = np.array([[0, 1, 2, 3], [2, 4, 6, 9]])
        readings = rng.normal(size=indices.shape)
        noise, prior_variance = 0.12, 1.7
        for ratio in (0.0, 0.25):
            phi = np.column_stack((features / np.sqrt(22), np.full(13, np.sqrt(ratio))))
            result = ridge_posterior(features, indices, readings, noise, prior_variance, ratio)
            mean_only = ridge_posterior(features, indices, readings, noise, prior_variance, ratio,
                                        with_variance=False)
            np.testing.assert_allclose(result["mean"], mean_only["mean"], atol=1e-12)
            for i, (idx, y) in enumerate(zip(indices, readings)):
                design = phi[idx]
                precision = np.eye(phi.shape[1]) / prior_variance + design.T @ design / noise**2
                covariance = np.linalg.inv(precision)
                coefficient_mean = np.linalg.solve(precision, design.T @ y / noise**2)
                np.testing.assert_allclose(result["mean"][i], phi @ coefficient_mean, atol=1e-12)
                np.testing.assert_allclose(result["raw_variance"][i],
                                           np.diag(phi @ covariance @ phi.T), atol=1e-12)
