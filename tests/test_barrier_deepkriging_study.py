import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    from spatial_exploration import barrier_deepkriging_study as study
    from spatial_exploration.barrier_gp import kernel_banks, room_layout
    from spatial_exploration.deepkriging import NeuralMapperConfig


@unittest.skipUnless(HAS_TORCH, "Install PyTorch to test graph DeepKriging")
class GraphDeepKrigingTests(unittest.TestCase):
    def test_features_reconstruct_the_gp_heat_kernel(self):
        layout = room_layout(12, 2)
        features = study.graph_heat_features(layout)
        bank = kernel_banks(layout)["geometry"]
        heat = bank.covariances[bank.parameters.index((study.GRAPH_LENGTH, 1.0))]
        np.testing.assert_allclose(features @ features.T / len(features), heat, atol=1e-12)
        alternative = study.graph_heat_features(room_layout(12, 9))
        self.assertFalse(np.allclose(features, alternative))

    def test_open_graph_control_restores_cross_wall_adjacency(self):
        from spatial_exploration.wall_topology_study import topology_design

        a, b = (topology_design(door) for door in study.TEST_DOORS)
        for design in (a, b):
            base = design.features["wendland"]
            opened, wall = (design.features[key] for key in ("open_graph", "wall_graph"))
            self.assertEqual(opened.shape, wall.shape)
            for matrix in (opened, wall):
                np.testing.assert_array_equal(matrix[:, :base.shape[1]], base)
                graph = matrix[:, base.shape[1]:].astype(float)
                np.testing.assert_allclose(np.mean(graph**2, axis=1), 1, atol=1e-7)
            # Adjacent nodes on opposite sides of the wall, far from either door.
            x, y = np.flatnonzero(design.layout.wall_edges[5])[0], 5
            self.assertGreater(opened[x, 236:] @ opened[y, 236:],
                               wall[x, 236:] @ wall[y, 236:])
        np.testing.assert_array_equal(a.features["open_graph"], b.features["open_graph"])

    @patch("spatial_exploration.barrier_deepkriging_study.render")
    def test_freeze_before_new_layout_fields_and_shared_acquisition_contract(self, _):
        config = NeuralMapperConfig(members=2, epochs=(1, 2), weight_decays=(0.01,), chunk_maps=4)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "study"
            original = study.generate_split

            def checked(designs, fields, seed, folder, phase):
                if phase == "test":
                    self.assertEqual(set(designs), set(study.TEST_DOORS))
                    self.assertFalse(set(designs).intersection(study.DOORS))
                    frozen = json.loads((folder / "frozen.json").read_text())
                    self.assertEqual(len(frozen["selected"]), 4)
                    self.assertEqual(len(frozen["std_multipliers"]), 6)
                return original(designs, fields, seed, folder, phase)

            with patch.object(study, "generate_split", side_effect=checked):
                summary = study.run(output, fields_per_layout=1, development_fields=1, config=config)
            self.assertEqual(len(summary), 12)
            self.assertTrue(all(row["fields"] == 2 and np.isfinite(row["rmse"]) for row in summary))
            with np.load(output / "test-observations.npz") as observations, \
                    np.load(output / "test-evaluation-only.npz") as evaluation:
                for door in study.TEST_DOORS:
                    for count, distance in study.BUDGETS:
                        prefix = f"d{door}_n{count}_"
                        indices = observations[prefix + "indices"]
                        self.assertEqual(indices.shape, (2, count))
                        self.assertTrue((observations[prefix + "travel"] <= distance + 1e-12).all())
                        self.assertFalse(evaluation[f"d{door}_heldout"][indices].any())
            with (output / "metrics.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            groups = {}
            for row in rows:
                key = tuple(row[k] for k in ("door", "seed", "regime", "measurements"))
                groups.setdefault(key, []).append((row["model"], row["start"], row["distance"]))
            for group in groups.values():
                self.assertEqual({r[0] for r in group}, set(study.MODELS))
                self.assertEqual(len({r[1:] for r in group}), 1)
