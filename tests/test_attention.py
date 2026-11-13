import importlib.util
import unittest

import numpy as np

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from spatial_exploration.baseline import make_grid
    from spatial_exploration.conditional_process import (CNPConfig, AttentiveConditionalNeuralProcess,
        ConditionalNeuralProcess, FrozenCNP)


@unittest.skipUnless(HAS_TORCH, "Install PyTorch for attention-CNP")
class AttentionTests(unittest.TestCase):
    def test_parameter_budget_matches_pooled_baseline(self):
        pooled = ConditionalNeuralProcess(CNPConfig())
        attention = AttentiveConditionalNeuralProcess(CNPConfig(width=60))
        self.assertEqual(sum(p.numel() for p in pooled.parameters()), 17218)
        self.assertEqual(sum(p.numel() for p in attention.parameters()), 17204)

    def test_attention_is_query_specific_normalized_and_context_permutation_invariant(self):
        config = CNPConfig(width=8, representation=8)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(12)
            network = AttentiveConditionalNeuralProcess(config)
        points = make_grid(5).astype(np.float32)
        context = torch.as_tensor(points[[0, 12, 24]])[None]
        query = torch.as_tensor(points[[3, 21]])[None]
        labels = torch.tensor([[0.2, -0.5, 1.0]])
        weights = network.attention_weights(context, query)
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones(1, 2))
        self.assertTrue(bool((weights >= 0).all()))
        self.assertFalse(torch.allclose(weights[0, 0], weights[0, 1], atol=1e-6))
        mean, std = network(context, labels, query)
        (mean.square().sum() + std.square().sum()).backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in network.position_encoder.parameters()), 1e-8)
        predictor = FrozenCNP(network, points)
        first = predictor([0, 12, 24], [0.2, -0.5, 1.0])
        permuted = predictor([24, 0, 12], [1.0, 0.2, -0.5])
        changed = predictor([0, 12, 24], [5.2, 4.5, 6.0])
        for key in first:
            np.testing.assert_allclose(first[key], permuted[key], rtol=1e-6, atol=1e-7)
        self.assertFalse(np.allclose(first["mean"], changed["mean"]))
        with torch.inference_mode():
            one, _ = network(context, labels, query[:, :1])
        self.assertAlmostEqual(float(one[0, 0]), first["mean"][3], places=6)

    def test_uniform_attention_reduces_to_same_width_pooled_model(self):
        config = CNPConfig(width=8, representation=8)
        pooled, attention = ConditionalNeuralProcess(config), AttentiveConditionalNeuralProcess(config)
        attention.encoder.load_state_dict(pooled.encoder.state_dict())
        attention.decoder.load_state_dict(pooled.decoder.state_dict())
        with torch.no_grad():
            for parameter in attention.position_encoder.parameters():
                parameter.zero_()
        context, values, query = torch.rand(2, 4, 2), torch.rand(2, 4), torch.rand(2, 7, 2)
        for first, second in zip(pooled(context, values, query), attention(context, values, query)):
            torch.testing.assert_close(first, second, rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
