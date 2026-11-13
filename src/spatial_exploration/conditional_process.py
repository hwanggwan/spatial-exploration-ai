"""Task-adapted deterministic CNP with explicit known observation noise.

Encode acquired (x,y,value) pairs, mean-pool a permutation-invariant context,
and decode each query with that context. A count channel is an explicit task
adaptation. Outputs are factorized latent Gaussian marginals, not a coherent
GP posterior or a joint model for hypothetical information gain.
"""

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class CNPConfig:
    training_fields: int = 4096
    validation_fields: int = 32
    seeds: tuple = (0, 1)
    checkpoints: tuple = (1000, 3000, 6000)
    width: int = 64
    representation: int = 64
    batch_size: int = 32
    targets: int = 64
    max_context: int = 16
    learning_rate: float = 0.001
    latent_std_floor: float = 0.02

    def validate(self):
        integers = (self.training_fields, self.validation_fields, self.width, self.representation,
                    self.batch_size, self.targets, self.max_context)
        if any(not isinstance(x, (int, np.integer)) or x < 1 for x in integers):
            raise ValueError("CNP sizes must be positive integers")
        if min(self.training_fields, self.validation_fields) < 2:
            raise ValueError("Need at least two independent maps per development split")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds) or any(s < 0 for s in self.seeds):
            raise ValueError("Use distinct nonnegative training seeds")
        if not self.checkpoints or tuple(sorted(set(self.checkpoints))) != tuple(self.checkpoints) or self.checkpoints[0] < 1:
            raise ValueError("Training checkpoints must increase strictly")
        if any(not np.isfinite(x) or x <= 0 for x in (self.learning_rate, self.latent_std_floor)):
            raise ValueError("Learning rate and std floor must be positive and finite")


class ConditionalNeuralProcess(nn.Module):
    def __init__(self, config):
        super().__init__()
        config.validate()
        self.config = config
        self.encoder = nn.Sequential(nn.Linear(3, config.width), nn.ReLU(),
            nn.Linear(config.width, config.width), nn.ReLU(), nn.Linear(config.width, config.representation))
        self.decoder = nn.Sequential(nn.Linear(config.representation + 3, config.width), nn.ReLU(),
            nn.Linear(config.width, config.width), nn.ReLU(), nn.Linear(config.width, 2))

    def forward(self, context_points, context_values, query_points):
        representation = self.encoder(torch.cat((2 * context_points - 1, context_values.unsqueeze(-1)), dim=-1)).mean(dim=1)
        repeated = representation[:, None].expand(-1, query_points.shape[1], -1)
        # Mean pooling alone loses explicit cardinality. This additional scalar
        # is disclosed as an adaptation, not part of a claimed paper reproduction.
        count = torch.full_like(query_points[:, :, :1],
            math.log1p(context_points.shape[1]) / math.log1p(self.config.max_context))
        output = self.decoder(torch.cat((repeated, 2 * query_points - 1, count), dim=-1))
        return output[..., 0], self.config.latent_std_floor + F.softplus(output[..., 1])


class AttentiveConditionalNeuralProcess(ConditionalNeuralProcess):
    """Deterministic query-specific cross-attention, without a latent NP path.

    Shared learned coordinate embeddings supply queries/keys; encoded acquired
    pairs supply values. Only the context aggregation changes in forward().
    Default study widths are chosen separately to match the pooled CNP budget.
    """
    def __init__(self, config, position_width=22):
        super().__init__(config)
        if position_width < 1:
            raise ValueError("Position width must be positive")
        self.position_encoder = nn.Sequential(nn.Linear(2, position_width), nn.ReLU(),
                                              nn.Linear(position_width, config.representation))

    def attention_weights(self, context_points, query_points):
        keys = self.position_encoder(2 * context_points - 1)
        queries = self.position_encoder(2 * query_points - 1)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(self.config.representation)
        return torch.softmax(scores, dim=-1)

    def forward(self, context_points, context_values, query_points):
        encoded = self.encoder(torch.cat((2 * context_points - 1, context_values.unsqueeze(-1)), dim=-1))
        representation = torch.bmm(self.attention_weights(context_points, query_points), encoded)
        count = torch.full_like(query_points[:, :, :1],
            math.log1p(context_points.shape[1]) / math.log1p(self.config.max_context))
        output = self.decoder(torch.cat((representation, 2 * query_points - 1, count), dim=-1))
        return output[..., 0], self.config.latent_std_floor + F.softplus(output[..., 1])


def observation_nll(mean, latent_std, values, noise_std):
    variance = latent_std.square() + noise_std**2
    return 0.5 * (math.log(2 * math.pi) + variance.log() + (values - mean).square() / variance)


def sample_training_batch(points, sensor_maps, allowed, config, rng):
    """No hidden latent labels accepted; context and query targets are disjoint.

    Every context includes the deployment start. Half use globally scattered
    sites, half use a local neighborhood to expose clustered sensing patterns.
    Context count is uniform on 1..max_context, shared within a mini-batch.
    """
    points, sensor_maps = np.asarray(points), np.asarray(sensor_maps)
    count = int(rng.integers(1, config.max_context + 1))
    permitted = np.flatnonzero(allowed)
    rest = permitted[permitted != 0]
    if 0 not in permitted or config.max_context > len(permitted) or config.targets > len(points) - config.max_context:
        raise ValueError("Context/target sizes must fit the permitted sensing geometry")
    fields = rng.integers(0, len(sensor_maps), size=config.batch_size)
    context, targets = [], []
    for _ in fields:
        pool = rest
        if rng.random() < 0.5:
            anchor = points[0] if rng.random() < 0.5 else points[int(rng.choice(permitted))]
            ordered = np.argsort(np.linalg.norm(points[rest] - anchor, axis=1), kind="stable")
            pool = rest[ordered[:max(config.max_context - 1, min(48, len(rest)))]]
        sites = np.r_[0, rng.choice(pool, size=count - 1, replace=False)].astype(int)
        available = np.ones(len(points), dtype=bool); available[sites] = False
        target = rng.choice(np.flatnonzero(available), size=config.targets, replace=False)
        context.append(sites); targets.append(target)
    context, targets = np.asarray(context), np.asarray(targets)
    return fields, context, targets


class FrozenCNP:
    """Deployment accepts only actual context labels and query geometry."""
    def __init__(self, network, points):
        self.network = network.eval()
        for parameter in self.network.parameters():
            parameter.requires_grad_(False)
        self.points = np.asarray(points, dtype=np.float32)

    def predict_batch(self, indices, values):
        indices, values = np.asarray(indices), np.asarray(values, dtype=np.float32)
        if (indices.ndim != 2 or values.shape != indices.shape or indices.shape[1] < 1
                or not np.issubdtype(indices.dtype, np.integer) or not np.isfinite(values).all()
                or np.any(indices < 0) or np.any(indices >= len(self.points))
                or any(len(set(row)) != len(row) for row in indices)):
            raise ValueError("Supply finite acquired labels at unique integer sites")
        context = torch.as_tensor(self.points[indices])
        query = torch.as_tensor(self.points)[None].expand(len(indices), -1, -1)
        with torch.inference_mode():
            mean, std = self.network(context, torch.as_tensor(values), query)
        return {"mean": mean.numpy().astype(float), "variance": std.square().numpy().astype(float)}

    def __call__(self, indices, values):
        result = self.predict_batch(np.asarray(indices)[None], np.asarray(values)[None])
        return {key: value[0] for key, value in result.items()}


def learned_covariance(sensor_maps, noise_std, shrinkage=0.05):
    """Known-zero-mean empirical spatial prior using only noisy training maps.

    Remove known noise, project onto the PSD cone, and use a fixed diagonal
    shrinkage. No simulator covariance, latent fields, or new-map labels enter.
    """
    maps = np.asarray(sensor_maps, dtype=float)
    if (maps.ndim != 2 or min(maps.shape) < 2 or not np.isfinite(maps).all()
            or not np.isfinite(noise_std) or noise_std <= 0 or not 0 <= shrinkage <= 1):
        raise ValueError("Supply finite training maps, positive noise, and a valid shrinkage")
    covariance = maps.T @ maps / len(maps) - noise_std**2 * np.eye(maps.shape[1])
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    psd = (eigenvectors * np.maximum(eigenvalues, 0)) @ eigenvectors.T
    diagonal = max(float(np.trace(psd) / len(psd)), 1e-8)
    return (1 - shrinkage) * psd + shrinkage * diagonal * np.eye(len(psd))
