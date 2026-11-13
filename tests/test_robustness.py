import importlib.util
import unittest

import numpy as np

HAS_RL = all(importlib.util.find_spec(name) is not None for name in ("torch", "gymnasium", "sb3_contrib"))
if HAS_RL:
    import torch
    from spatial_exploration.baseline import GPBelief, make_grid, rbf_covariance
    from spatial_exploration.belief_explorer import ExplorationBeliefState, select_action
    from spatial_exploration.estimated_gp import EstimatedGPBelief, EstimatorConfig
    from spatial_exploration.exploration_env import ExplorationConfig, SpatialExplorationEnv
    from spatial_exploration.imitation import ClonedPolicy
    from spatial_exploration.robustness_study import execute_case, make_mapper, sample_case


@unittest.skipUnless(HAS_RL, "Install the rl extra to test frozen-policy robustness")
class RobustnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.config = ExplorationConfig(grid_size=3)
        self.estimator = EstimatorConfig(optimizer_restarts=0)
        self.case = {"scenario": 0, "field_seed": 1400000, "start_index": 0,
                     "length_scale": 0.23, "budget": 1.5}
        self.prior, self.truth, self.sensor = sample_case(self.config, self.case)

    def execute(self, condition="estimated", method="greedy", policy=None, sensor=None, prior=None, truth=None):
        return execute_case(self.config, self.estimator, self.case, condition, method, policy,
            self.truth if truth is None else truth, self.sensor if sensor is None else sensor,
            self.prior if prior is None else prior)

    def test_known_belief_state_matches_gym_inputs_masks_and_routes(self):
        gym = SpatialExplorationEnv(self.config)
        observation, _ = gym.reset(seed=10, options=self.case)
        mapper = GPBelief(self.prior, self.config.noise_std)
        mapper.observe(0, self.sensor[0])
        state = ExplorationBeliefState(self.config, mapper, 0, 1.5, 0.23, 1.0)
        while not gym.finished:
            np.testing.assert_array_equal(state.observation(), observation)
            np.testing.assert_array_equal(state.action_masks(), gym.action_masks())
            action, _ = select_action("greedy", None, state)
            self.assertEqual(action, gym.greedy_action())
            state.move_and_observe(action, self.sensor[action])
            observation, _, _, _, _ = gym.step(action)
        self.assertTrue(state.finished)
        self.assertEqual(state.route, gym.route)
        self.assertAlmostEqual(state.travel_distance, gym.travel_distance)

    def test_estimated_policy_input_uses_current_estimate_without_clipping(self):
        points = make_grid(3)
        mapper = EstimatedGPBelief(points, 0.1, self.estimator)
        mapper.observe(0, 0.2)
        state = ExplorationBeliefState(self.config, mapper, 0, 1.5, 0.35, 1.5)
        mapper.length_scale = 0.8
        mapper.covariance *= 2
        self.assertAlmostEqual(float(state.observation()[-1]), 0.8 / 0.35, places=6)
        np.testing.assert_allclose(state.observation()[:81], mapper.covariance.ravel(), rtol=1e-6, atol=1e-7)
        self.assertTrue(state.input_diagnostics()["length_outside_training_range"])
        self.assertTrue(state.input_diagnostics()["observation_outside_original_box"])
        self.assertTrue(state.input_diagnostics()["posterior_diagonal_above_training_cap"])

    def test_true_route_variance_matches_independent_joint_gp_conditioning(self):
        metrics, _, _, _, _, _ = self.execute(condition="fixed_initial")
        route = [int(index) for index in metrics["route"].split()]
        cross = self.prior[:, route]
        observed = self.prior[np.ix_(route, route)] + 0.01 * np.eye(len(route))
        posterior = self.prior - cross @ np.linalg.solve(observed, cross.T)
        self.assertAlmostEqual(metrics["oracle_route_mean_variance"], np.trace(posterior) / 9)
        self.assertAlmostEqual(metrics["oracle_route_heldout_mean_variance"], np.diag(posterior)[4])
        self.assertGreater(abs(metrics["model_reported_mean_variance"] - metrics["oracle_route_mean_variance"]), 1e-4)

    def test_unobserved_values_and_hidden_true_kernel_do_not_change_unknown_model_routes(self):
        torch.manual_seed(0)
        neural = ClonedPolicy(94, 9, hidden_sizes=(8,))
        for condition in ("fixed_initial", "estimated"):
            for method, policy in (("greedy", None), ("rollout", None), ("student", neural)):
                with self.subTest(condition=condition, method=method):
                    first = self.execute(condition=condition, method=method, policy=policy)[0]
                    acquired = [int(index) for index in first["route"].split()]
                    altered = self.sensor.copy()
                    mask = np.ones(9, dtype=bool)
                    mask[acquired] = False
                    altered[mask] = 10000
                    second = self.execute(condition=condition, method=method, policy=policy,
                        sensor=altered, prior=rbf_covariance(make_grid(3), 0.7), truth=self.truth + 100)[0]
                    self.assertEqual(first["route"], second["route"])
                    self.assertAlmostEqual(first["model_reported_mean_variance"], second["model_reported_mean_variance"])
                    self.assertAlmostEqual(first["model_length_scale"], second["model_length_scale"])

    def test_covariance_refits_can_increase_reported_variance(self):
        mapper = EstimatedGPBelief(make_grid(3), 0.1, self.estimator)
        for index, value in ((0, 10.0), (1, -10.0), (3, 10.0)):
            mapper.observe(index, value)
        before = np.trace(mapper.covariance) / 9
        mapper.observe(6, -10.0)
        self.assertEqual(len(mapper.fit_history), 1)
        self.assertGreater(np.trace(mapper.covariance) / 9, before)

    def test_invalid_measurement_is_rejected_without_changing_state(self):
        mapper, length, variance = make_mapper("fixed_initial", make_grid(3), 0.1, self.estimator)
        mapper.observe(0, 0.2)
        state = ExplorationBeliefState(self.config, mapper, 0, 1.5, length, variance)
        before = state.observation().copy()
        for action in (0, 4, 100):
            with self.subTest(action=action), self.assertRaises(ValueError):
                state.move_and_observe(action, 10000)
            np.testing.assert_array_equal(before, state.observation())
            self.assertEqual(state.route, [0])
            self.assertEqual(state.travel_distance, 0.0)


if __name__ == "__main__":
    unittest.main()
