import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from spatial_exploration.resource_replay import ResourceConfig, generate_fields, geometry
from spatial_exploration.uncertainty_planner import choose_uncertainty_action

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    from spatial_exploration.deepkriging import NeuralMapperConfig
    from spatial_exploration.estimated_gp import EstimatorConfig
    from spatial_exploration.uncertainty_study import collect_paths, run_study


class UncertaintyPlannerTests(unittest.TestCase):
    def test_variance_per_distance_and_budget_masks_determine_action(self):
        points = np.array([[0, 0], [0.1, 0], [0.2, 0], [1, 0]])
        variance = np.array([0, 1, 4, 6.0])
        visited = np.array([True, False, False, False])
        allowed = np.ones(4, dtype=bool)
        self.assertEqual(choose_uncertainty_action(points, variance, 0, visited, allowed, 1, 2), 2)
        self.assertEqual(choose_uncertainty_action(points, variance, 0, visited, allowed, 0.15, 2), 1)
        allowed[2] = False
        self.assertEqual(choose_uncertainty_action(points, variance, 0, visited, allowed, 1, 2), 1)
        self.assertIsNone(choose_uncertainty_action(points, variance, 0, visited, allowed, 0, 2))
        self.assertIsNone(choose_uncertainty_action(points, variance, 0, visited, allowed, 1, 0))

    def test_positive_scalar_calibration_preserves_ranking_and_invalid_variances_fail(self):
        points = np.array([[0, 0], [0.1, 0], [0.2, 0], [1, 0]])
        visited = np.array([True, False, False, False])
        allowed = np.ones(4, dtype=bool)
        variance = np.array([0, 1, 4, 6.0])
        for scale in (0.01, 1, 1000):
            self.assertEqual(choose_uncertainty_action(points, scale * variance, 0, visited, allowed, 1, 2), 2)
        for bad in (np.array([0, -1, 4, 6]), np.array([0, np.nan, 4, 6])):
            with self.assertRaises(ValueError):
                choose_uncertainty_action(points, bad, 0, visited, allowed, 1, 2)


@unittest.skipUnless(HAS_TORCH, "Install PyTorch for mapper-driven neural exploration")
class UncertaintyWorkflowTests(unittest.TestCase):
    def configs(self):
        config = ResourceConfig(grid_size=5, fields=2, calibration_fields=2,
                                observation_caps=(2, 4), travel_budgets=(0.5, 1.0))
        neural = NeuralMapperConfig(knot_grids=(2, 3), basis_width=8, members=2,
                                   epochs=(2,), weight_decays=(0.0001,))
        return config, neural, EstimatorConfig(optimizer_restarts=0)

    def test_collection_is_reproducible_and_never_reads_unvisited_or_heldout_values(self):
        config, neural, estimator = self.configs()
        evaluator = generate_fields(config, 2, 7)
        sensor = evaluator["potential_sensor_values"]
        calls = []
        def read(field, site):
            calls.append((field, site))
            return sensor[field, site]
        episodes, states, _ = collect_paths(config, 2, read, neural, estimator, 8)
        self.assertEqual(len(calls), sum(len(e["route"]) for e in episodes))
        mask = evaluator["evaluation_mask"]
        self.assertFalse(any(mask[site] for _, site in calls))
        changed = sensor.copy()
        for field in range(2):
            used = {site for f, site in calls if f == field}
            for site in set(range(len(mask))) - used:
                changed[field, site] += 1000
        repeated, repeated_states, _ = collect_paths(config, 2, lambda f, s: changed[f, s], neural, estimator, 8)
        self.assertEqual([e["route"] for e in episodes], [e["route"] for e in repeated])
        for a, b in zip(states, repeated_states):
            np.testing.assert_array_equal(a["mean"], b["mean"])
            np.testing.assert_array_equal(a["variance"], b["variance"])
        points, _ = geometry(config)
        for episode in episodes:
            self.assertEqual(len(episode["route"]), len(episode["state_ids"]))
            self.assertLessEqual(len(episode["route"]), episode["observation_cap"])
            self.assertLessEqual(episode["distances"][-1], episode["travel_budget"] + 1e-12)
            for step, state_id in enumerate(episode["state_ids"]):
                state = states[state_id]
                self.assertEqual(state["model"], episode["model"])
                self.assertEqual(state["indices"], episode["route"][:step + 1])
                np.testing.assert_array_equal(state["values"], sensor[episode["field"], state["indices"]])
                if step < len(episode["route"]) - 1:
                    visited = np.zeros(len(points), dtype=bool)
                    visited[state["indices"]] = True
                    action = choose_uncertainty_action(points, state["variance"], state["indices"][-1], visited,
                        ~mask, max(episode["travel_budget"] - episode["distances"][step], 0),
                        episode["observation_cap"] - len(state["indices"]))
                    self.assertEqual(action, episode["route"][step + 1])

    @patch("spatial_exploration.uncertainty_study.render_results")
    def test_calibration_freezes_before_new_benchmark_fields_and_summaries_are_paired(self, *_):
        config, neural, estimator = self.configs()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            def checked_fields(config, fields, seed):
                if seed == 4100000:
                    self.assertTrue((output / "frozen_protocol.json").exists())
                    self.assertFalse((output / "benchmark_acquired.npz").exists())
                return generate_fields(config, fields, seed)
            with patch("spatial_exploration.uncertainty_study.generate_fields", side_effect=checked_fields):
                summary, diagnostics = run_study(output, config, neural, estimator)
            self.assertEqual(len(summary), 8)
            self.assertEqual(diagnostics["benchmark_episodes"], 16)
            self.assertTrue(diagnostics["all_count_caps_respected"])
            self.assertTrue(diagnostics["all_distance_caps_respected"])
            self.assertTrue(diagnostics["evaluation_sites_never_acquired"])
            self.assertEqual(len(json.loads((output / "paired_comparisons.json").read_text())), 4)


if __name__ == "__main__":
    unittest.main()
