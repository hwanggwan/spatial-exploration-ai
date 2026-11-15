import importlib.util
import unittest

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("torch"), "Study fixture requires PyTorch")
class GeneralizationGeometryTests(unittest.TestCase):
    def test_train_and_test_wall_graphs_are_distinct_and_connected(self):
        from spatial_exploration.sensing_world_generalization import (
            STARTS, TEST_GEOMETRIES, TRAIN_GEOMETRIES, make_design, wall_layout,
        )

        trained = {wall_layout(spec).wall_edges.tobytes() for spec in TRAIN_GEOMETRIES}
        tested = set()
        for spec in TEST_GEOMETRIES:
            layout = wall_layout(spec)
            adjacency = layout.wall_edges.tobytes()
            self.assertNotIn(adjacency, trained | tested)
            tested.add(adjacency)
            self.assertTrue(np.isfinite(layout.distances).all())
            self.assertFalse(np.any(layout.wall_edges & layout.movement))
            across_rooms = layout.room[:, None] != layout.room[None, :]
            self.assertEqual(np.sum(layout.movement & across_rooms), 2 * sum(map(len, spec[2])))
            design = make_design(layout, seed=4)
            self.assertEqual(design.heldout.sum(), 28)
            self.assertFalse(design.heldout[list(STARTS)].any())

    def test_shifted_field_shares_sensor_noise_but_changes_truth(self):
        from spatial_exploration.sensing_world_generalization import (
            Job, TEST_GEOMETRIES, make_design, simulate, wall_layout,
        )

        design = make_design(wall_layout(TEST_GEOMETRIES[0]), seed=4)
        rows = list(simulate(design, Job("unseen", 0, 16)))
        for benchmark, shifted in zip(rows[::2], rows[1::2]):
            self.assertEqual(benchmark[:3], shifted[:3])
            self.assertFalse(np.array_equal(benchmark[4], shifted[4]))
            np.testing.assert_allclose(benchmark[5] - benchmark[4], shifted[5] - shifted[4], atol=1e-12)
