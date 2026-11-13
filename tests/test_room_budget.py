import unittest

import numpy as np

from spatial_exploration.barrier_gp import room_layout
from spatial_exploration.baseline import GPBelief
from spatial_exploration.room_budget import RoomQuota


class RoomBudgetTests(unittest.TestCase):
    def test_door_reserve_overrides_nearer_home_measurement(self):
        layout = room_layout(6, 2)
        start, tempting, door_site, next_away = 13, 12, 15, 16
        allowed = np.zeros(36, dtype=bool)
        allowed[[start, tempting, door_site, next_away]] = True
        visited = np.zeros_like(allowed)
        visited[start] = True
        belief = GPBelief(np.eye(36), .12)
        belief.observe(start, 0.)
        # Three edges allow both away readings, but not a detour to site 12.
        quota = RoomQuota(layout, allowed, start, 4, .6)
        self.assertTrue(quota.active)
        self.assertEqual(quota.next(belief, start, visited, .6), door_site)
        visited[door_site] = True
        self.assertEqual(quota.next(belief, door_site, visited, .2), next_away)
        fallback = RoomQuota(layout, allowed, start, 4, .2)
        self.assertFalse(fallback.active)
        visited[door_site] = False
        self.assertEqual(fallback.next(belief, start, visited, .2), tempting)
