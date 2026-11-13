import importlib.util
import unittest

import numpy as np

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from spatial_exploration.baseline import make_grid
    from spatial_exploration.conditional_process import (CNPConfig, ConditionalNeuralProcess, FrozenCNP,
        learned_covariance, observation_nll, sample_training_batch)


@unittest.skipUnless(HAS_TORCH, "Install PyTorch for conditional neural processes")
class ConditionalProcessTests(unittest.TestCase):
    def setUp(self):
        self.config = CNPConfig(training_fields=8, validation_fields=2, checkpoints=(2, 4),
            width=8, representation=8, batch_size=4, targets=4, max_context=4)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(12)
            self.network = ConditionalNeuralProcess(self.config)

    def test_context_permutation_query_independence_and_value_conditioning(self):
        points = make_grid(5)
        predictor = FrozenCNP(self.network, points)
        before = {key: value.clone() for key, value in self.network.state_dict().items()}
        first = predictor([0, 7, 12], [0.2, -0.5, 1.0])
        reordered = predictor([12, 0, 7], [1.0, 0.2, -0.5])
        changed = predictor([0, 7, 12], [10.2, 9.5, 11.0])
        for key in first:
            np.testing.assert_allclose(first[key], reordered[key], rtol=1e-6, atol=1e-7)
        self.assertFalse(np.allclose(first["mean"], changed["mean"]))
        self.assertTrue(np.all(first["variance"] >= self.config.latent_std_floor**2 - 1e-9))
        with torch.inference_mode():
            context = torch.tensor(points[[0, 7, 12]][None], dtype=torch.float32)
            labels = torch.tensor([[0.2, -0.5, 1.0]])
            one, _ = self.network(context, labels, torch.tensor(points[[3]][None], dtype=torch.float32))
        self.assertAlmostEqual(first["mean"][3], float(one[0, 0]), places=6)
        for key, value in self.network.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        self.assertFalse(any(p.requires_grad for p in self.network.parameters()))
        for indices, values in (([0, 0], [1, 2]), ([25], [1]), ([0.1], [1]), ([0], [np.nan])):
            with self.assertRaises(ValueError):
                predictor(indices, values)

    def test_training_queries_are_disjoint_and_unavailable_sites_never_enter_context(self):
        points = make_grid(5)
        maps = np.random.default_rng(1).normal(size=(8, 25))
        allowed = np.arange(25) % 5 != 4
        rng = np.random.default_rng(2)
        counts = set()
        for _ in range(30):
            fields, context, query = sample_training_batch(points, maps, allowed, self.config, rng)
            counts.add(context.shape[1])
            self.assertTrue(allowed[context].all())
            np.testing.assert_array_equal(context[:, 0], 0)
            self.assertTrue(np.all(fields < len(maps)))
            for c, q in zip(context, query):
                self.assertFalse(set(c) & set(q))
                self.assertEqual(len(set(q)), len(q))
        self.assertEqual(counts, {1, 2, 3, 4})

    def test_known_noise_is_added_to_likelihood_not_latent_output(self):
        mean, std, values = torch.tensor([0.0]), torch.tensor([0.2]), torch.tensor([0.5])
        result = observation_nll(mean, std, values, 0.1)
        expected = -torch.distributions.Normal(0.0, np.sqrt(0.2**2 + 0.1**2)).log_prob(values)
        torch.testing.assert_close(result, expected)
        self.assertAlmostEqual(float(std.square()[0]), 0.04, places=7)

    def test_offline_covariance_uses_noisy_data_without_true_kernel(self):
        rng = np.random.default_rng(77)
        true = np.array([[1.0, 0.6], [0.6, 1.0]])
        sensor = rng.multivariate_normal([0, 0], true + 0.1**2 * np.eye(2), size=20000)
        estimate = learned_covariance(sensor, 0.1, shrinkage=0)
        np.testing.assert_allclose(estimate, true, atol=0.025)
        sparse = learned_covariance(sensor[:2], 0.1)
        self.assertGreater(np.linalg.eigvalsh(sparse).min(), 0)
        with self.assertRaises(ValueError):
            learned_covariance(sensor, 0)


if __name__ == "__main__":
    unittest.main()
