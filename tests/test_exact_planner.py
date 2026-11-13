import itertools
import unittest

import numpy as np

from spatial_exploration.baseline import GPBelief, make_grid, rbf_covariance
from spatial_exploration.exact_planner import SearchLimitExceeded, exhaustive_plan


class ExactPlannerTests(unittest.TestCase):
    def setUp(self):
        self.points = make_grid(3)
        self.distances = np.linalg.norm(self.points[:, None] - self.points[None, :], axis=-1)
        self.prior = rbf_covariance(self.points, 0.35)
        self.belief = GPBelief(self.prior, 0.1)
        self.belief.observe(0, 0.0)
        self.visited = np.zeros(9, dtype=bool)
        self.visited[0] = True
        self.allowed = np.zeros(9, dtype=bool)
        self.allowed[[0, 1, 3, 4]] = True

    def solve(self, **overrides):
        arguments = dict(covariance=self.belief.covariance, distances=self.distances,
            current=0, visited=self.visited, allowed=self.allowed,
            remaining_budget=1.1, noise_variance=0.01)
        arguments.update(overrides)
        return exhaustive_plan(**arguments)

    def test_exact_solution_matches_permutations_and_joint_gp_prediction(self):
        result = self.solve()
        available = [1, 3, 4]
        minimum, feasible_prefixes, terminal_routes = float("inf"), 0, 0
        for depth in range(len(available) + 1):
            for sequence in itertools.permutations(available, depth):
                route = (0,) + sequence
                distance = sum(self.distances[a, b] for a, b in zip(route[:-1], route[1:]))
                if distance > 1.1 + 1e-12:
                    continue
                feasible_prefixes += 1
                observation_cov = self.prior[np.ix_(route, route)] + 0.01 * np.eye(len(route))
                posterior = self.prior - self.prior[:, route] @ np.linalg.solve(
                    observation_cov, self.prior[list(route), :])
                minimum = min(minimum, float(np.trace(posterior) / len(self.points)))
                if not any(distance + self.distances[route[-1], index] <= 1.1 + 1e-12
                           for index in available if index not in route):
                    terminal_routes += 1
        self.assertAlmostEqual(result.final_mean_variance, minimum)
        self.assertEqual(result.prefixes_evaluated, feasible_prefixes)
        self.assertEqual(result.terminal_routes, terminal_routes)
        self.assertLessEqual(result.travel_distance, 1.1 + 1e-12)
        self.assertEqual(len(result.route), len(set(result.route)))

    def test_more_budget_cannot_worsen_the_optimal_objective(self):
        results = [self.solve(remaining_budget=budget) for budget in (0, 0.5, 1.1, 1.6)]
        for first, second in zip(results[:-1], results[1:]):
            self.assertLessEqual(second.final_mean_variance, first.final_mean_variance + 1e-12)

    def test_zero_budget_returns_current_position(self):
        result = self.solve(remaining_budget=0)
        self.assertEqual(result.route, (0,))
        self.assertEqual(result.prefixes_evaluated, 1)
        self.assertEqual(result.terminal_routes, 1)
        self.assertAlmostEqual(result.final_mean_variance, np.trace(self.belief.covariance) / 9)

    def test_incomplete_search_raises_instead_of_claiming_optimality(self):
        with self.assertRaises(SearchLimitExceeded):
            self.solve(max_prefixes=1)

    def test_solver_does_not_modify_current_covariance_or_masks(self):
        copies = [item.copy() for item in (self.belief.covariance, self.visited, self.allowed)]
        self.solve()
        for before, after in zip(copies, (self.belief.covariance, self.visited, self.allowed)):
            np.testing.assert_array_equal(before, after)

    def test_all_reported_measurements_are_allowed(self):
        result = self.solve(remaining_budget=1.6)
        self.assertTrue(np.all(self.allowed[list(result.route)]))


if __name__ == "__main__":
    unittest.main()
