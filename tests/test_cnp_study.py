import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    from spatial_exploration.cnp_study import run_study
    from spatial_exploration.conditional_process import CNPConfig
    from spatial_exploration.deepkriging import NeuralMapperConfig
    from spatial_exploration.estimated_gp import EstimatorConfig
    from spatial_exploration.resource_replay import ResourceConfig, generate_fields


@unittest.skipUnless(HAS_TORCH, "Install PyTorch for offline CNP exploration")
class CNPWorkflowTests(unittest.TestCase):
    @patch("spatial_exploration.cnp_study.render_results")
    @patch("spatial_exploration.crossover_study.render_results")
    def test_two_seeds_freeze_before_new_maps_and_all_same_data_crossovers_match(self, *_):
        resources = ResourceConfig(grid_size=5, fields=2, calibration_fields=2,
            observation_caps=(2, 4), travel_budgets=(0.5, 1.0))
        cnp = CNPConfig(training_fields=16, validation_fields=2, checkpoints=(2, 4), width=8,
            representation=8, batch_size=4, targets=4, max_context=4)
        neural = NeuralMapperConfig(knot_grids=(2, 3), basis_width=8, members=2,
                                   epochs=(2,), weight_decays=(0.0001,))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "source"
            def checked_fields(config, fields, seed):
                if seed == 6300000:
                    self.assertTrue((output / "frozen_protocol.json").exists())
                    self.assertFalse((output / "benchmark_acquired.npz").exists())
                return generate_fields(config, fields, seed)
            with patch("spatial_exploration.cnp_study.generate_fields", side_effect=checked_fields):
                summaries, diagnostics = run_study(output, resources, cnp, neural,
                    EstimatorConfig(optimizer_restarts=0))
            self.assertEqual(len(summaries), 20)
            self.assertEqual(diagnostics["benchmark_episodes"], 40)
            for key in ("all_count_caps_respected", "all_distance_caps_respected",
                        "evaluation_sites_never_acquired", "cnp_weights_unchanged"):
                self.assertTrue(diagnostics[key])
            training = json.loads((output / "training_results.json").read_text())
            self.assertIn(training["selected_step"], (2, 4))
            self.assertEqual({r["seed"] for r in training["checkpoint_scores"]}, {0, 1})
            cross = output.with_name("source-crossover")
            checks = json.loads((cross / "diagnostics.json").read_text())
            self.assertEqual(checks["endpoint_evaluations"], 200)
            self.assertEqual(checks["max_native_diagonal_error"], 0)
            self.assertTrue(checks["source_unchanged"])
            episodes = json.loads((output / "benchmark_routes.json").read_text())
            for item in json.loads((cross / "prediction_index.json").read_text()):
                source = next(e for e in episodes if e["field"] == item["field"] and e["route"] == item["indices"])
                np.testing.assert_array_equal(item["values"], source["values"])


if __name__ == "__main__":
    unittest.main()
