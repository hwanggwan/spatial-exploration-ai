import unittest

import numpy as np

from spatial_exploration.baseline import Config, GPBelief, explore, make_grid, rbf_covariance


class BaselineTests(unittest.TestCase):
    def test_observation_cap_and_travel_cap_both_apply_including_start(self):
        points = make_grid(5)
        prior = rbf_covariance(points,0.3)
        allowed = np.arange(len(points)) % 5 != 4
        for strategy in ('random','greedy'):
            config = Config(grid_size=5,travel_budget=10,max_observations=3)
            _,route,distances = explore(points,prior,np.zeros(25),allowed,config,strategy,np.random.default_rng(0))
            self.assertEqual(len(route),3)
            self.assertEqual(distances[0],0)
            config = Config(grid_size=5,travel_budget=0,max_observations=3)
            _,route,distances = explore(points,prior,np.zeros(25),allowed,config,strategy,np.random.default_rng(0))
            self.assertEqual(route,[0])
            self.assertEqual(distances,[0])
        for cap in (0,-1,1.5,True):
            with self.assertRaises(ValueError):Config(max_observations=cap).validate()

    def test_sequential_gp_matches_batch_conditioning(self):
        points = make_grid(4)
        prior = rbf_covariance(points, 0.3)
        indices = [0, 5, 14]
        values = np.array([0.7, -0.2, 1.3])
        belief = GPBelief(prior, 0.1)
        for index, value in zip(indices, values):
            belief.observe(index, value)
        cross = prior[:, indices]
        observed_cov = prior[np.ix_(indices, indices)] + 0.01 * np.eye(3)
        expected_mean = cross @ np.linalg.solve(observed_cov, values)
        expected_cov = prior - cross @ np.linalg.solve(observed_cov, cross.T)
        np.testing.assert_allclose(belief.mean, expected_mean, atol=1e-12)
        np.testing.assert_allclose(belief.covariance, expected_cov, atol=1e-12)

    def test_planners_obey_budget_and_exclude_evaluation_sites(self):
        config = Config(grid_size=5, travel_budget=1.2, seeds=1)
        points = make_grid(config.grid_size)
        prior = rbf_covariance(points, config.length_scale)
        allowed = np.arange(len(points)) % 5 != 4
        for strategy in ("random", "greedy"):
            arguments = (points, prior, np.zeros(len(points)), allowed, config, strategy)
            _, route, distances = explore(*arguments, np.random.default_rng(3))
            _, repeated, repeated_distances = explore(*arguments, np.random.default_rng(3))
            self.assertEqual(route, repeated)
            self.assertEqual(distances, repeated_distances)
            self.assertEqual(len(route), len(set(route)))
            self.assertTrue(np.all(allowed[route]))
            actual_distance = np.linalg.norm(np.diff(points[route], axis=0), axis=1).sum()
            self.assertAlmostEqual(actual_distance, distances[-1])
            self.assertLessEqual(actual_distance, config.travel_budget + 1e-12)

    def test_greedy_choices_do_not_depend_on_hidden_field_values(self):
        points = make_grid(5)
        prior = rbf_covariance(points, 0.2)
        allowed = np.arange(len(points)) % 5 != 4
        config = Config(grid_size=5, seeds=1)
        _, first, _ = explore(points, prior, np.zeros(len(points)), allowed, config,
                              "greedy", np.random.default_rng(0))
        _, second, _ = explore(points, prior, np.arange(len(points)), allowed, config,
                               "greedy", np.random.default_rng(0))
        self.assertEqual(first, second)

    def test_zero_budget_only_observes_start(self):
        config = Config(grid_size=3, travel_budget=0, seeds=1)
        points = make_grid(3)
        prior = rbf_covariance(points, 0.2)
        _, route, distances = explore(points, prior, np.zeros(9), np.ones(9, dtype=bool),
                                      config, "greedy", np.random.default_rng(0))
        self.assertEqual(route, [0])
        self.assertEqual(distances, [0.0])


if __name__ == "__main__":
    unittest.main()
