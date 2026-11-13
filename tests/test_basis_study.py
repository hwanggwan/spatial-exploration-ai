import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    from spatial_exploration.basis_study import run_study
    from spatial_exploration.crossover_study import run_study as run_crossover
    from spatial_exploration.deepkriging import NeuralMapperConfig
    from spatial_exploration.estimated_gp import EstimatorConfig
    from spatial_exploration.resource_replay import ResourceConfig, generate_fields


@unittest.skipUnless(HAS_TORCH, "Install PyTorch for the matched neural control study")
class BasisWorkflowTests(unittest.TestCase):
    @patch("spatial_exploration.basis_study.render_results")
    @patch("spatial_exploration.crossover_study.render_results")
    def test_frozen_new_fields_resources_and_three_way_same_data_crossover(self, *_):
        config = ResourceConfig(grid_size=5, fields=2, calibration_fields=2,
                                observation_caps=(2, 4), travel_budgets=(0.5, 1.0))
        neural = NeuralMapperConfig(knot_grids=(2, 3), basis_width=8, members=2,
            epochs=(1, 2), weight_decays=(0.0001,), tuning_fields=2)
        with tempfile.TemporaryDirectory() as directory:
            output, cross = Path(directory) / "source", Path(directory) / "cross"
            def checked_fields(config, fields, seed):
                if seed == 5200000:
                    self.assertTrue((output / "frozen_protocol.json").exists())
                    self.assertTrue((output / "tuning_results.json").exists())
                    self.assertFalse((output / "benchmark_acquired.npz").exists())
                return generate_fields(config, fields, seed)
            with patch("spatial_exploration.basis_study.generate_fields", side_effect=checked_fields):
                summaries, diagnostics = run_study(output, config, neural,
                    EstimatorConfig(optimizer_restarts=0), prior_candidates=(0.2, 1.0))
            self.assertEqual(len(summaries), 12)
            self.assertEqual(diagnostics["benchmark_episodes"], 24)
            for key in ("all_count_caps_respected", "all_distance_caps_respected", "evaluation_sites_never_acquired"):
                self.assertTrue(diagnostics[key])
            summaries, decomposition, checks = run_crossover(output, cross)
            self.assertEqual(len(summaries), 36)
            self.assertEqual(checks["endpoint_evaluations"], 72)
            self.assertEqual(checks["max_native_diagonal_error"], 0)
            self.assertTrue(checks["source_unchanged"])
            self.assertEqual(len(decomposition), 8)
            episodes = json.loads((output / "benchmark_routes.json").read_text())
            for cap in config.observation_caps:
                for budget in config.travel_budgets:
                    routes = [e["route"] for e in episodes if e["model"] == "basis_bayes"
                              and e["observation_cap"] == cap and e["travel_budget"] == budget]
                    self.assertEqual(routes[0], routes[1])
            index = json.loads((cross / "prediction_index.json").read_text())
            for item in index:
                source = next(e for e in episodes if e["field"] == item["field"] and e["route"] == item["indices"])
                np.testing.assert_array_equal(item["values"], source["values"])
            for row in decomposition:
                self.assertAlmostEqual(row["sampling_design_mean_difference"] + row["reconstruction_mapper_mean_difference"],
                                       row["mean_total_rmse_difference"], places=12)


if __name__ == "__main__":
    unittest.main()
