import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

HAS_RL = all(importlib.util.find_spec(name) is not None for name in ("torch", "gymnasium", "sb3_contrib"))
if HAS_RL:
    import torch
    from spatial_exploration.exploration_env import ExplorationConfig, SpatialExplorationEnv
    from spatial_exploration.imitation import (ClonedPolicy, CloningConfig, collect_common_states,
        mask_logits, masked_cross_entropy, train_policy)
    from spatial_exploration.imitation_study import assert_disjoint_cases, make_cases


@unittest.skipUnless(HAS_RL, "Install the rl extra to test imitation learning")
class ImitationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_masked_loss_matches_valid_action_distribution_and_blocks_gradients(self):
        logits = torch.tensor([[1.0, 2.0, 1000.0]], requires_grad=True)
        masks = torch.tensor([[True, True, False]])
        loss = masked_cross_entropy(logits, masks, torch.tensor([1]))
        self.assertAlmostEqual(float(loss.detach()), float(np.log1p(np.exp(-1))), places=6)
        loss.backward()
        self.assertEqual(float(logits.grad[0, 2]), 0.0)
        self.assertEqual(int(mask_logits(logits, masks).argmax(1)), 1)
        with self.assertRaises(ValueError):
            masked_cross_entropy(logits, masks, torch.tensor([2]))
        with self.assertRaises(ValueError):
            mask_logits(logits, torch.zeros_like(masks))

    def test_episode_splits_are_reproducible_and_reject_overlap(self):
        config = ExplorationConfig(grid_size=3)
        train = make_cases(config, 2, 910000, 1100000)
        validation = make_cases(config, 2, 920000, 1200000)
        test = make_cases(config, 2, 930000, 1300000)
        self.assertEqual(train, make_cases(config, 2, 910000, 1100000))
        assert_disjoint_cases([train, validation, test])
        with self.assertRaises(ValueError):
            assert_disjoint_cases([train, train])

    def test_common_teacher_data_preserves_feasible_labels_and_episode_identities(self):
        config = ExplorationConfig(grid_size=3)
        cases = make_cases(config, 2, 910000, 1100000)
        with tempfile.TemporaryDirectory() as directory:
            data, details = collect_common_states(config, cases, Path(directory) / "states.npz")
            saved = np.load(Path(directory) / "states.npz")
            self.assertEqual(set(saved.files), {"observations", "masks", "greedy_actions", "rollout_actions", "episode_ids"})
            self.assertEqual(set(data["episode_ids"]), {0, 1})
            self.assertEqual(data["observations"].shape[1], 3**4 + 3**2 + 4)
            for teacher in ("greedy", "rollout"):
                self.assertTrue(data["masks"][np.arange(details["states"]), data[f"{teacher}_actions"]].all())
            for key in data:
                np.testing.assert_array_equal(data[key], saved[key])

    def test_training_improves_a_separable_problem_and_reload_preserves_actions(self):
        data = {"observations": np.array([[-1, 0], [-2, 0], [1, 0], [2, 0]], dtype=np.float32),
            "masks": np.array([[True, True, False]] * 4),
            "greedy_actions": np.array([0, 0, 1, 1])}
        config = CloningConfig(epochs=20, batch_size=4, learning_rate=0.02, hidden_sizes=(8,))
        with tempfile.TemporaryDirectory() as directory:
            selected, details = train_policy(data, data, "greedy", 0, config, directory)
            self.assertLess(details["selected_validation"]["cross_entropy"], details["initial_validation"]["cross_entropy"])
            self.assertEqual(details["selected_validation"]["action_accuracy"], 1.0)
            reloaded = ClonedPolicy.load(Path(directory) / "selected.pt")
            for observation, mask, label in zip(data["observations"], data["masks"], data["greedy_actions"]):
                self.assertEqual(selected.predict(observation, mask)[0], label)
                self.assertEqual(reloaded.predict(observation, mask)[0], label)

    def test_student_rollout_obeys_gym_masks_without_teacher_fallback(self):
        env = SpatialExplorationEnv(ExplorationConfig(grid_size=3))
        observation, _ = env.reset(seed=12)
        policy = ClonedPolicy(len(observation), env.count, hidden_sizes=(8,))
        while not env.finished:
            action, _ = policy.predict(observation, env.action_masks())
            observation, _, _, _, info = env.step(action)
            self.assertFalse(info["invalid_action"])
        self.assertLessEqual(env.travel_distance, env.budget + 1e-12)
        self.assertEqual(len(env.route), len(set(env.route)))
        self.assertTrue(np.all(env.allowed[env.route]))


if __name__ == "__main__":
    unittest.main()
