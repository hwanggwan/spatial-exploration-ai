import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    from spatial_exploration.attention_study import run_study
    from spatial_exploration.cnp_study import run_study as run_source
    from spatial_exploration.conditional_process import CNPConfig
    from spatial_exploration.deepkriging import NeuralMapperConfig
    from spatial_exploration.estimated_gp import EstimatorConfig
    from spatial_exploration.resource_replay import ResourceConfig, generate_fields


@unittest.skipUnless(HAS_TORCH, "Install PyTorch for matched attention exploration")
class AttentionWorkflowTests(unittest.TestCase):
    @patch("spatial_exploration.cnp_study.render_results")
    @patch("spatial_exploration.attention_study.render_results")
    @patch("spatial_exploration.attention_study.render_attention")
    @patch("spatial_exploration.crossover_study.render_results")
    def test_reuses_frozen_development_data_and_preserves_all_native_crossovers(self, *_):
        resources = ResourceConfig(grid_size=5, fields=2, calibration_fields=2,
            observation_caps=(2, 4), travel_budgets=(0.5, 1.0))
        cnp = CNPConfig(training_fields=16, validation_fields=2, checkpoints=(2, 4), width=8,
            representation=8, batch_size=4, targets=4, max_context=4)
        neural = NeuralMapperConfig(knot_grids=(2, 3), basis_width=8, members=2,
                                   epochs=(2,), weight_decays=(0.0001,))
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "source", Path(directory) / "attention"
            run_source(source, resources, cnp, neural, EstimatorConfig(optimizer_restarts=0), crossover=False)
            def checked_fields(config, fields, seed):
                self.assertIn(seed, (7200000, 7300000))
                if seed == 7300000:
                    self.assertTrue((output / "frozen_protocol.json").exists())
                    self.assertFalse((output / "benchmark_acquired.npz").exists())
                return generate_fields(config, fields, seed)
            with patch("spatial_exploration.attention_study.generate_fields", side_effect=checked_fields):
                summaries, diagnostics = run_study(source, output)
            self.assertEqual(len(summaries), 28)
            self.assertEqual(diagnostics["benchmark_episodes"], 56)
            for key in ("all_count_caps_respected", "all_distance_caps_respected", "evaluation_sites_never_acquired",
                        "source_unchanged", "weights_and_prior_unchanged"):
                self.assertTrue(diagnostics[key])
            protocol = json.loads((output / "protocol.json").read_text())
            self.assertIn(protocol["selected_attention_step"], (2, 4))
            self.assertEqual(protocol["selected_cnp_step"], json.loads((source / "protocol.json").read_text())["selected_cnp_step"])
            fixed = json.loads((output / "fixed_step_comparison.json").read_text())
            self.assertEqual(len(fixed["summaries"]), 8)
            self.assertEqual({r["updates"] for r in fixed["summaries"]}, {4})
            self.assertEqual({r["training_seed"] for r in fixed["summaries"]}, {0, 1})
            self.assertTrue((output / "fixed_step_predictions.npz").exists())
            for name in ("offline_training_observations.npz", "validation_acquired.npz", "learned_prior_covariance.npy"):
                self.assertEqual((source / name).read_bytes(), (output / name).read_bytes())
            cross = json.loads((output.with_name("attention-crossover") / "diagnostics.json").read_text())
            self.assertEqual(cross["endpoint_evaluations"], 392)
            self.assertEqual(cross["max_native_diagonal_error"], 0)
            self.assertTrue(cross["source_unchanged"])


if __name__ == "__main__":
    unittest.main()
