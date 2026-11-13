import importlib.util
import unittest

import numpy as np


HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    from spatial_exploration.barrier_ridge_study import ridge_posterior
    from spatial_exploration.wall_closed_loop_study import NOISE, explore_ridge, ridge_models
    from spatial_exploration.wall_topology_study import topology_design


@unittest.skipUnless(HAS_TORCH, "Install PyTorch for shared graph benchmark tooling")
class WallClosedLoopTests(unittest.TestCase):
    def test_same_route_for_different_readings_and_batch_agreement(self):
        design = topology_design(2)
        spec = {"prior": 0.31622776601683794, "intercept": 0.25}
        results = {}
        for model, ridge in ridge_models(design, spec).items():
            called = []

            def read(index):
                self.assertFalse(design.heldout[index])
                self.assertNotIn(index, called)
                called.append(index)
                return np.sin(index)

            route, distances, y, belief, *_ = explore_ridge(design, ridge, 13, (16, 2.0), read)
            self.assertEqual(route, called)
            self.assertEqual(route[0], 13)
            self.assertLessEqual(len(route), 16)
            self.assertLessEqual(distances[-1], 2.0 + 1e-12)
            reroute, _, _, changed, *_ = explore_ridge(design, ridge, 13, (16, 2.0), lambda i: -np.sin(i))
            self.assertEqual(route, reroute)
            np.testing.assert_array_equal(belief.covariance, changed.covariance)
            np.testing.assert_allclose(belief.mean, -changed.mean, atol=1e-12)
            expected = ridge_posterior(design.features[model], np.array([route]), np.array([y]),
                                       NOISE, spec["prior"], spec["intercept"])
            np.testing.assert_allclose(belief.mean, expected["mean"][0], atol=1e-12)
            np.testing.assert_allclose(belief.std**2, expected["raw_variance"][0], atol=1e-12)
            results[model] = route
        self.assertNotEqual(results["open_graph"], results["wall_graph"])
