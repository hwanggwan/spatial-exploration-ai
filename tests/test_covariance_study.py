import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from spatial_exploration.baseline import Config
from spatial_exploration.covariance_study import MODELS, run_study
from spatial_exploration.estimated_gp import EstimatorConfig


class CovarianceStudyTests(unittest.TestCase):
    @patch("spatial_exploration.covariance_study.render_curves")
    @patch("spatial_exploration.covariance_study.render_replay")
    @patch("spatial_exploration.covariance_study.render")
    def test_replay_really_uses_identical_acquired_data(self, *_):
        with tempfile.TemporaryDirectory() as directory:
            summary = run_study(Config(grid_size=5, seeds=2),
                EstimatorConfig(optimizer_restarts=0), directory)
            with (Path(directory) / "trajectories.csv").open() as stream:
                trajectories = list(csv.DictReader(stream))
            for seed in range(2):
                sequences = []
                for name in MODELS:
                    sequences.append([(row["index"], row["observed_value"], row["cumulative_distance"])
                        for row in trajectories if row["protocol"] == "replay"
                        and int(row["seed"]) == seed and row["model"] == name])
                self.assertEqual(sequences[0], sequences[1])
                self.assertEqual(sequences[0], sequences[2])
                fixed_closed = [(row["index"], row["observed_value"], row["cumulative_distance"])
                    for row in trajectories if row["protocol"] == "closed_loop"
                    and int(row["seed"]) == seed and row["model"] == "fixed_initial"]
                self.assertEqual(sequences[1], fixed_closed)
            for protocol in ("replay", "closed_loop"):
                for name in MODELS:
                    self.assertEqual(summary[protocol][name]["seeds"], 2)


if __name__ == "__main__":
    unittest.main()
