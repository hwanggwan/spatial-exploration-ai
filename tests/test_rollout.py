import unittest

import numpy as np

from spatial_exploration.baseline import GPBelief, make_grid, rbf_covariance
from spatial_exploration.exact_planner import exhaustive_plan
from spatial_exploration.exact_study import execute_heuristic
from spatial_exploration.rollout import plan_rollout


class RolloutTests(unittest.TestCase):
    def setUp(self):
        self.points = make_grid(4)
        self.distances = np.linalg.norm(self.points[:, None] - self.points[None, :], axis=-1)
        self.prior = rbf_covariance(self.points, 0.35)
        self.belief = GPBelief(self.prior, 0.1)
        self.belief.observe(0, 0.0)
        self.visited = np.zeros(16, dtype=bool)
        self.visited[0] = True
        self.allowed = np.arange(16) % 5 != 4

    def plan(self, **overrides):
        arguments = dict(covariance=self.belief.covariance, distances=self.distances,
            current=0, visited=self.visited, allowed=self.allowed,
            remaining_budget=1.45, noise_variance=0.01)
        arguments.update(overrides)
        return plan_rollout(**arguments)

    def batch_variance(self, route):
        cross = self.prior[:, list(route)]
        observed = self.prior[np.ix_(route, route)] + 0.01 * np.eye(len(route))
        return float(np.trace(self.prior - cross @ np.linalg.solve(observed, cross.T)) / 16)

    def test_candidate_ranking_matches_independent_batch_gp_completions(self):
        # Independently recompute the full batch posterior at every hypothetical
        # step and compute each one-measurement gain with another batch solve.
        completions = []
        simulation_count = 0
        for first in np.flatnonzero(self.allowed & ~self.visited & (self.distances[0] <= 1.45)):
            route, distance = [0, int(first)], float(self.distances[0, first])
            while True:
                available = [int(index) for index in np.flatnonzero(self.allowed)
                    if index not in route and distance + self.distances[route[-1], index] <= 1.45 + 1e-12]
                if not available:
                    break
                variance = self.batch_variance(route)
                scores = [(variance - self.batch_variance(route + [index])) /
                          self.distances[route[-1], index] for index in available]
                chosen = available[int(np.argmax(scores))]
                distance += self.distances[route[-1], chosen]
                route.append(chosen)
            completions.append((self.batch_variance(route), tuple(route), distance))
            simulation_count += len(route) - 1
        result = self.plan()
        self.assertAlmostEqual(result.final_mean_variance, min(item[0] for item in completions))
        self.assertAlmostEqual(result.final_mean_variance, self.batch_variance(result.route))
        self.assertAlmostEqual(result.greedy_final_mean_variance, self.batch_variance(result.greedy_route))
        self.assertEqual(result.candidates_evaluated, len(completions))
        self.assertEqual(result.simulated_measurements, simulation_count)
        self.assertLessEqual(result.planned_distance, 1.45 + 1e-12)
        self.assertEqual(result.first, result.route[1])
        self.assertEqual(len(result.route), len(set(result.route)))
        self.assertTrue(np.all(self.allowed[list(result.route)]))

    def test_rollout_fixes_the_density_counterexample(self):
        result = self.plan(remaining_budget=0.75)
        optimal = exhaustive_plan(self.belief.covariance, self.distances, 0,
            self.visited, self.allowed, 0.75, 0.01)
        self.assertAlmostEqual(result.final_mean_variance, optimal.final_mean_variance)
        self.assertLess(result.final_mean_variance, result.greedy_final_mean_variance - 0.05)
        self.assertEqual(len(result.route), 3)

    def test_one_remaining_reading_maximizes_terminal_gain_not_gain_per_distance(self):
        allowed = np.zeros(16, dtype=bool)
        allowed[[0, 1, 5]] = True
        # The farther site has higher variance and larger absolute gain, but
        # the nearer one has higher gain per unit distance.
        covariance = np.diag([0.01, 1.0, 1.0, 1.0, 1.0, 1.2] + [1.0] * 10)
        result = self.plan(covariance=covariance, allowed=allowed, remaining_observations=1)
        self.assertEqual(result.greedy_route, (0, 1))
        self.assertEqual(result.route, (0, 5))
        self.assertEqual(result.simulated_measurements, 2)
        with self.assertRaises(RuntimeError):
            self.plan(remaining_observations=0)

    def test_replanning_is_no_worse_than_greedy_and_initial_completion(self):
        for budget in (0.34, 0.75, 1.1, 1.45):
            with self.subTest(budget=budget):
                initial = self.plan(remaining_budget=budget)
                greedy = execute_heuristic(self.prior, self.distances, 0, self.allowed, budget, 0.1)
                rollout = execute_heuristic(self.prior, self.distances, 0, self.allowed,
                    budget, 0.1, planner=plan_rollout)
                final = self.batch_variance(rollout[0])
                self.assertLessEqual(final, self.batch_variance(greedy[0]) + 1e-10)
                self.assertLessEqual(final, initial.final_mean_variance + 1e-10)
                self.assertLessEqual(rollout[2], budget + 1e-12)

    def test_single_candidate_and_objective_ties_keep_the_base_policy(self):
        result = self.plan(remaining_budget=0.34)
        self.assertEqual(result.route, (0, 1))
        self.assertEqual(result.route, result.greedy_route)
        # Sites 1 and 4 are equidistant in an identity covariance; all complete
        # two-observation objectives tie, so preserve the base's index tie break.
        allowed = np.zeros(16, dtype=bool)
        allowed[[0, 1, 4]] = True
        tied = self.plan(covariance=np.eye(16), allowed=allowed, remaining_budget=0.34)
        self.assertEqual(tied.first, 1)
        self.assertEqual(tied.route, tied.greedy_route)

    def test_search_preserves_inputs_and_rejects_invalid_or_terminal_states(self):
        arrays = (self.belief.covariance, self.distances, self.visited, self.allowed)
        copies = [item.copy() for item in arrays]
        self.plan()
        for before, after in zip(copies, arrays):
            np.testing.assert_array_equal(before, after)
        with self.assertRaises(RuntimeError):
            self.plan(remaining_budget=0)
        for changes in (dict(noise_variance=0), dict(remaining_budget=-1),
                        dict(visited=np.zeros(16, dtype=bool)), dict(distances=np.zeros((16, 16)))):
            with self.subTest(changes=list(changes)), self.assertRaises(ValueError):
                self.plan(**changes)


if __name__ == "__main__":
    unittest.main()
