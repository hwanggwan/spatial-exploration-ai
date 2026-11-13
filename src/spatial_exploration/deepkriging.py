"""Small-data DeepKriging-style mapper and capacity-matched coordinate control.

Core representation: multiresolution compact Wendland features, then a ReLU
MLP. This is a task adaptation, not an exact reproduction of the paper's
architecture, experiments, or density-estimation uncertainty method.
"""

from dataclasses import asdict, dataclass
import time

import numpy as np
import torch
from torch import nn

from .spatial_features import raw_features, standardized_features, wendland


@dataclass(frozen=True)
class NeuralMapperConfig:
    knot_grids: tuple = (4, 7, 13)
    support_multiplier: float = 2.5
    basis_width: int = 32
    members: int = 3
    learning_rate: float = 0.003
    epochs: tuple = (100, 300)
    weight_decays: tuple = (0.0001, 0.01)
    tuning_fields: int = 8
    calibration_fields: int = 8
    chunk_maps: int = 16
    variance_floor: float = 1e-6

    def validate(self):
        if not self.knot_grids or any(size < 2 for size in self.knot_grids):
            raise ValueError("Use knot grids of size at least two")
        if self.members < 2 or min(self.basis_width, self.tuning_fields, self.calibration_fields, self.chunk_maps) < 1:
            raise ValueError("Need at least two members and positive network/data sizes")
        if tuple(sorted(set(self.epochs))) != tuple(self.epochs) or self.epochs[0] < 1:
            raise ValueError("Epoch checkpoints must increase strictly")
        if not self.weight_decays or any(value < 0 or not np.isfinite(value) for value in self.weight_decays):
            raise ValueError("Weight decays must be finite and nonnegative")
        if any(not np.isfinite(value) or value <= 0 for value in
               (self.learning_rate, self.support_multiplier, self.variance_floor)):
            raise ValueError("Learning rate, support, and variance floor must be positive")


def parameter_count(features, width):
    return features * width + width + width**2 + width + width + 1


def matched_coordinate_width(config):
    target = parameter_count(2 + sum(size**2 for size in config.knot_grids), config.basis_width)
    return min(range(1, 1000), key=lambda width: abs(parameter_count(2, width) - target))


class BatchedSpatialMLP(nn.Module):
    """Independent per-field/member networks, vectorized without weight sharing.

    No batch normalization, cross-field pooling, or shared learned parameters.
    Summing per-instance mean losses gives the same gradients as separate fits.
    """

    def __init__(self, features, width, seeds):
        super().__init__()
        shapes = ((features, width), (width, width), (width, 1))
        weights, biases = [], []
        for layer, (input_size, output_size) in enumerate(shapes):
            generated = []
            for seed in seeds:
                generator = torch.Generator().manual_seed(int(seed) + 100003 * layer)
                bound = np.sqrt(6 / input_size) if layer < 2 else np.sqrt(1 / input_size)
                generated.append(torch.empty(input_size, output_size).uniform_(-bound, bound, generator=generator))
            weights.append(nn.Parameter(torch.stack(generated)))
            biases.append(nn.Parameter(torch.zeros(len(seeds), 1, output_size)))
        self.weights = nn.ParameterList(weights)
        self.biases = nn.ParameterList(biases)

    def forward(self, inputs):
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(0).expand(len(self.weights[0]), -1, -1)
        hidden = inputs
        for layer, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            hidden = torch.bmm(hidden, weight) + bias
            if layer < 2:
                hidden = torch.relu(hidden)
        return hidden.squeeze(-1)


def fit_ensemble_snapshots(features, observation_indices, observed_values, model, config,
                           seeds, weight_decay, checkpoints, save_path=None):
    """Fresh fits using only acquired labels; no evaluator targets accepted.

    Each row is one independent map with its own observation indices/values.
    Return ensemble predictions at fixed epoch counts. Caller handles selection
    and calibration using separate evaluator-owned development fields.
    """
    started = time.perf_counter()
    config.validate()
    features = np.asarray(features, dtype=np.float32)
    indices = np.asarray(observation_indices)
    values = np.asarray(observed_values, dtype=np.float32)
    if indices.ndim != 2 or values.shape != indices.shape or len(seeds) != len(indices) * config.members:
        raise ValueError("Observation rows and independent member seeds must agree")
    if not np.issubdtype(indices.dtype, np.integer) or not np.isfinite(values).all():
        raise ValueError("Use integer observation indices and finite acquired values")
    if np.any(indices < 0) or np.any(indices >= len(features)):
        raise ValueError("Observed site outside feature domain")
    if any(len(set(row)) != len(row) for row in indices):
        raise ValueError("Each map must have unique acquired sites")
    width = config.basis_width if model == "deepkriging" else matched_coordinate_width(config)
    network = BatchedSpatialMLP(features.shape[1], width, seeds)
    train_features = torch.as_tensor(np.repeat(features[indices], config.members, axis=0))
    targets = torch.as_tensor(np.repeat(values, config.members, axis=0))
    query = torch.as_tensor(features)
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate, weight_decay=weight_decay)
    results, timings, training_errors = {}, {}, {}
    for epoch in range(1, max(checkpoints) + 1):
        optimizer.zero_grad(set_to_none=True)
        errors = ((network(train_features) - targets)**2).mean(dim=1)
        loss = errors.sum()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Nonfinite neural training objective")
        loss.backward()
        optimizer.step()
        if epoch in checkpoints:
            with torch.inference_mode():
                prediction = network(query).numpy().reshape(len(indices), config.members, len(features))
                results[epoch] = {"mean": prediction.mean(axis=1).astype(float),
                    "raw_variance": np.maximum(prediction.var(axis=1, ddof=1), config.variance_floor).astype(float),
                    "members": prediction.astype(float)}
                training_errors[epoch] = float(((network(train_features) - targets)**2).mean())
            timings[epoch] = time.perf_counter() - started
    if save_path is not None:
        torch.save({"state_dict": network.state_dict(), "features": features.shape[1], "width": width,
            "seeds": [int(seed) for seed in seeds], "members_per_map": config.members,
            "epochs": max(checkpoints), "weight_decay": weight_decay, "config": asdict(config)}, save_path)
    return results, timings, training_errors


def calibration_multiplier(truth, mean, raw_variance, evaluation_mask):
    """Validation-only scalar interval scaling, without a conformal guarantee.

    Spatially correlated sites are pooled for this heuristic calibration. It is
    not a posterior covariance or a guarantee about new measurement conditioning.
    """
    residual = np.abs(truth[:, evaluation_mask] - mean[:, evaluation_mask])
    std = np.sqrt(raw_variance[:, evaluation_mask])
    ratio = residual / np.maximum(std, 1e-12)
    quantile = float(np.quantile(ratio, 0.95))
    return max((quantile / 1.96)**2, 1e-12)
