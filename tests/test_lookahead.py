import unittest

import numpy as np

from spatial_exploration.baseline import GPBelief, make_grid, rbf_covariance
from spatial_exploration.lookahead import plan_measurements


class LookaheadTests(unittest.TestCase):
    def setUp(self):
        self.points = make_grid(4)
        self.distances = np.linalg.norm(self.points[:, None] - self.points[None, :], axis=-1)
        self.belief = GPBelief(rbf_covariance(self.points, 0.3), 0.1)
        self.belief.observe(0, 0.0)
        self.visited = np.zeros(16, dtype=bool)
        self.visited[0] = True
        self.allowed = np.arange(16) % 5 != 4

    def plan(self, **overrides):
        arguments = dict(covariance=self.belief.covariance, distances=self.distances,
            current=0, visited=self.visited, allowed=self.allowed,
            remaining_budget=1.5, noise_variance=0.01, depth=2)
        arguments.update(overrides)
        return plan_measurements(**arguments)

    def test_two_step_scores_match_independent_joint_gp_conditioning(self):
        plan = self.plan()
        covariance = self.belief.covariance
        available = np.flatnonzero(self.allowed & ~self.visited)
        independent_scores = []
        for first in available:
            first_cost = self.distances[0, first]
            if first_cost > 1.5 + 1e-12:
                continue
            seconds = [second for second in available if second != first
                and first_cost + self.distances[first, second] <= 1.5 + 1e-12]
            for second in seconds or [None]:
                indices = [first] if second is None else [first, second]
                observed_cov = covariance[np.ix_(indices, indices)] + 0.01 * np.eye(len(indices))
                posterior = covariance - covariance[:, indices] @ np.linalg.solve(
                    observed_cov, covariance[indices, :])
                gain = (np.trace(covariance) - np.trace(posterior)) / len(covariance)
                cost = first_cost + (0 if second is None else self.distances[first, second])
                independent_scores.append(gain / cost)
                if first == plan.first and second == plan.second:
                    self.assertAlmostEqual(gain, plan.expected_variance_reduction)
                    self.assertAlmostEqual(cost, plan.planned_distance)
        self.assertAlmostEqual(plan.score, max(independent_scores))
        self.assertEqual(plan.sequences_evaluated, len(independent_scores))

    def test_depth_one_matches_original_reduction_per_distance(self):
        plan = self.plan(depth=1)
        candidates = np.flatnonzero(self.allowed & ~self.visited & (self.distances[0] <= 1.5))
        cross = self.belief.covariance[:, candidates]
        gains = np.mean(cross**2, axis=0) / (np.diag(self.belief.covariance)[candidates] + 0.01)
        expected = candidates[np.argmax(gains / self.distances[0, candidates])]
        self.assertEqual(plan.first, expected)
        self.assertIsNone(plan.second)

    def test_terminal_one_move_is_used_when_no_second_is_affordable(self):
        allowed = np.zeros(16, dtype=bool)
        allowed[[0, 1]] = True
        plan = self.plan(remaining_budget=0.34, allowed=allowed)
        self.assertEqual(plan.first, 1)
        self.assertIsNone(plan.second)
        self.assertLessEqual(plan.planned_distance, 0.34)

    def test_hypothetical_search_does_not_mutate_the_current_belief_or_masks(self):
        original = [array.copy() for array in
                    (self.belief.covariance, self.distances, self.visited, self.allowed)]
        self.plan()
        for before, after in zip(original,
                (self.belief.covariance, self.distances, self.visited, self.allowed)):
            np.testing.assert_array_equal(before, after)

    def test_no_budget_produces_no_plan(self):
        with self.assertRaises(RuntimeError):
            self.plan(remaining_budget=0)

    def test_only_supported_depths_are_accepted(self):
        with self.assertRaises(ValueError):
            self.plan(depth=3)


if __name__ == "__main__":
    unittest.main()
