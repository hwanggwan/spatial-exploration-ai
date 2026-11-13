import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    from spatial_exploration.crossover_study import run_study
    from spatial_exploration.deepkriging import NeuralMapperConfig
    from spatial_exploration.estimated_gp import EstimatorConfig
    from spatial_exploration.resource_replay import ResourceConfig
    from spatial_exploration.uncertainty_study import run_study as run_source


@unittest.skipUnless(HAS_TORCH, "Install PyTorch for frozen-route crossover diagnostics")
class CrossoverTests(unittest.TestCase):
    @patch("spatial_exploration.uncertainty_study.render_results")
    @patch("spatial_exploration.crossover_study.render_results")
    def test_native_results_reproduce_source_and_crossovers_preserve_observations(self, *_):
        config = ResourceConfig(grid_size=5, fields=2, calibration_fields=2,
                                observation_caps=(2, 4), travel_budgets=(0.5, 1.0))
        neural = NeuralMapperConfig(knot_grids=(2, 3), basis_width=8, members=2,
                                   epochs=(2,), weight_decays=(0.0001,))
        with tempfile.TemporaryDirectory() as directory:
            reference, output = Path(directory) / "source", Path(directory) / "cross"
            run_source(reference, config, neural, EstimatorConfig(optimizer_restarts=0))
            summary, decomposition, diagnostics = run_study(reference, output)
            self.assertEqual(len(summary), 16)
            self.assertEqual(diagnostics["endpoint_evaluations"], 32)
            self.assertEqual(diagnostics["max_native_diagonal_error"], 0)
            self.assertTrue(diagnostics["source_unchanged"])
            routes = json.loads((reference / "benchmark_routes.json").read_text())
            index = json.loads((output / "prediction_index.json").read_text())
            for item in index:
                matches = [episode for episode in routes if episode["field"] == item["field"]
                           and episode["route"] == item["indices"]]
                self.assertTrue(matches)
                self.assertEqual(item["values"], matches[0]["values"])
            for row in decomposition:
                self.assertAlmostEqual(row["sampling_design_mean_difference"] + row["reconstruction_mapper_mean_difference"],
                                       row["mean_total_rmse_difference"], places=12)


if __name__ == "__main__":
    unittest.main()
