import unittest

import numpy as np

from spatial_exploration.baseline import GPBelief
from spatial_exploration.three_room_quota import STARTS, choose_allocation, quota_next, three_rooms


class ThreeRoomQuotaTests(unittest.TestCase):
    def test_allocation_remains_feasible_and_walk_uses_only_the_two_doors(self):
        layout = three_rooms((2, 9))
        start = STARTS[0]
        allowed = np.ones(len(layout.points), dtype=bool)
        allowed[[26, 117]] = False  # Door cells can be traversed but not sensed.
        for method in ("equal", "priority"):
            for budget in (18 / 11, 26 / 11):
                with self.subTest(method=method, budget=budget):
                    belief = GPBelief(np.eye(len(allowed)), 0.12)
                    visited = np.zeros_like(allowed)
                    visited[start] = True
                    belief.observe(start, 0)
                    order, pending = choose_allocation(
                        layout, allowed, start, 16, budget, belief.covariance, method)
                    self.assertLessEqual(pending.sum() + 1, 16)
                    current, travel = start, 0.0
                    while pending.any():
                        nxt = quota_next(layout, allowed, visited, current, budget - travel,
                                         order, pending, belief)
                        self.assertTrue(allowed[nxt] and not visited[nxt])
                        for a, b in zip(layout.walk(current, nxt), layout.walk(current, nxt)[1:]):
                            if layout.room[a] != layout.room[b]:
                                self.assertEqual(a // layout.size, (2, 9)[min(layout.room[a], layout.room[b])])
                        travel += layout.distances[current, nxt]
                        pending[layout.room[nxt]] -= 1
                        visited[nxt] = True
                        belief.observe(nxt, 0)
                        current = nxt
                    self.assertLessEqual(travel, budget + 1e-12)
