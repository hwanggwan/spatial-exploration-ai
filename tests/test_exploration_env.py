import unittest

import numpy as np

from spatial_exploration.baseline import Config, explore, rbf_covariance

try:
    import gymnasium
    from spatial_exploration.exploration_env import ExplorationConfig, SpatialExplorationEnv
except ImportError:
    gymnasium = None


@unittest.skipIf(gymnasium is None, "Install the rl extra to test the learning environment")
class ExplorationEnvironmentTests(unittest.TestCase):
    def test_greedy_environment_matches_original_planner(self):
        env = SpatialExplorationEnv(ExplorationConfig(grid_size=5))
        env.reset(seed=0, options={"start_index": 0, "budget": 2.0, "length_scale": 0.2})
        prior = rbf_covariance(env.points, 0.2)
        original_belief, original_route, distances = explore(env.points, prior,
            np.zeros(env.count), env.allowed, Config(grid_size=5),
            "greedy", np.random.default_rng(0))
        while not env.finished:
            env.step(env.greedy_action())
        self.assertEqual(env.route, original_route)
        self.assertAlmostEqual(env.travel_distance, distances[-1])
        np.testing.assert_allclose(env.belief.covariance, original_belief.covariance)

    def test_reward_telescopes_to_total_variance_reduction(self):
        env = SpatialExplorationEnv(ExplorationConfig(grid_size=5))
        obs, _ = env.reset(seed=10)
        self.assertTrue(env.observation_space.contains(obs))
        initial_variance = env.mean_variance
        total_reward = 0.0
        done = False
        while not done:
            action = env.greedy_action()
            self.assertTrue(env.action_masks()[action])
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            self.assertFalse(truncated)
            self.assertFalse(info["invalid_action"])
            self.assertTrue(env.observation_space.contains(obs))
        self.assertAlmostEqual(total_reward, env.config.reward_scale * (initial_variance - env.mean_variance))
        self.assertLessEqual(env.travel_distance, env.budget + 1e-12)
        self.assertEqual(len(env.route), len(set(env.route)))
        self.assertTrue(np.all(env.allowed[env.route]))

    def test_masks_exclude_visited_heldout_and_unreachable_sites(self):
        env = SpatialExplorationEnv(ExplorationConfig(grid_size=5))
        env.reset(seed=0, options={"start_index": 0, "budget": 1.5, "length_scale": 0.2})
        while not env.finished:
            mask = env.action_masks()
            self.assertFalse(mask[env.visited].any())
            self.assertFalse(mask[env.evaluation_mask].any())
            self.assertFalse(mask[env.distances[env.current] > env.budget - env.travel_distance + 1e-12].any())
            env.step(env.greedy_action())
        self.assertFalse(env.action_masks().any())

    def test_seeded_reset_and_route_are_reproducible(self):
        env = SpatialExplorationEnv(ExplorationConfig(grid_size=5))
        first, _ = env.reset(seed=22)
        first_parameters = (env.current, env.budget, env.length_scale)
        first_route = []
        while not env.finished:
            action = env.greedy_action()
            first_route.append(action)
            env.step(action)
        second, _ = env.reset(seed=22)
        self.assertEqual(first_parameters, (env.current, env.budget, env.length_scale))
        np.testing.assert_array_equal(first, second)
        second_route = []
        while not env.finished:
            action = env.greedy_action()
            second_route.append(action)
            env.step(action)
        self.assertEqual(first_route, second_route)

    def test_invalid_action_terminates_without_moving(self):
        env = SpatialExplorationEnv(ExplorationConfig(grid_size=5))
        env.reset(seed=0)
        _, reward, done, _, info = env.step(env.current)
        self.assertTrue(done)
        self.assertTrue(info["invalid_action"])
        self.assertLess(reward, 0)
        self.assertEqual(env.travel_distance, 0)


if __name__ == "__main__":
    unittest.main()
