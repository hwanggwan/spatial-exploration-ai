import unittest

import numpy as np

from spatial_exploration.barrier_gp import kernel_banks, room_layout, transport_covariance
from spatial_exploration.barrier_study import NOISE, collect, evaluation_sites
from spatial_exploration.baseline import GPBelief


class BarrierGPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.layout = room_layout(12, 3)
        cls.banks = kernel_banks(cls.layout)

    def test_shortest_walk_crosses_only_at_door(self):
        layout = self.layout
        left, right = 5, 6  # Euclidean neighbors on opposite sides of the wall.
        walk = layout.walk(left, right)
        self.assertAlmostEqual(layout.distances[left, right], 7 / 11)
        self.assertEqual([(a, b) for a, b in zip(walk, walk[1:]) if layout.room[a] != layout.room[b]],
                         [(41, 42)])
        self.assertTrue(all(layout.movement[a, b] for a, b in zip(walk, walk[1:])))

    def test_barrier_changes_dependence_without_changing_unit_variance(self):
        layout = self.layout
        permeable = transport_covariance(layout, 1.0, 0.42)
        blocked = transport_covariance(layout, 0.02, 0.42)
        self.assertLess(blocked[5, 6], 0.4 * permeable[5, 6])
        for covariance in (permeable, blocked, *self.banks['geometry'].covariances):
            np.testing.assert_allclose(covariance.diagonal(), 1)
            self.assertGreater(np.linalg.eigvalsh(covariance).min(), -1e-10)

    def test_batch_posterior_matches_existing_sequential_conditioning(self):
        indices = [17, 18, 42, 54, 93, 94]
        values = [-1.4, 0.2, 0.6, 0.9, -0.4, -1.0]
        for bank in self.banks.values():
            fitted, params = bank.posterior(indices, values, NOISE)
            sequential = GPBelief(bank.covariances[bank.parameters.index(params)], NOISE)
            for index, value in zip(indices, values):
                sequential.observe(index, value)
            np.testing.assert_allclose(fitted.mean, sequential.mean, atol=1e-11)
            np.testing.assert_allclose(fitted.covariance, sequential.covariance, atol=1e-11)

    def test_closed_loop_reads_only_selected_legal_sites(self):
        layout = self.layout
        heldout, _ = evaluation_sites(layout)
        for bank in self.banks.values():
            seen = []

            def sense(index):
                self.assertNotIn(index, seen)
                self.assertFalse(heldout[index])
                seen.append(index)
                return float(np.sin(index))

            route, distances, readings, *_ = collect(
                layout, 13, ~heldout, (16, 0.38), sense,
                lambda indices, values: bank.posterior(indices, values, NOISE))
            self.assertEqual(route, seen)
            self.assertEqual(route[0], 13)
            self.assertGreater(len(route), 1)
            self.assertLess(len(route), 16)  # The distance cap binds first.
            self.assertLessEqual(distances[-1], 0.38)
            self.assertEqual(readings, [float(np.sin(index)) for index in route])
