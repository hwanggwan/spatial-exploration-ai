"""Controlled reimplementation of the DeepKriging paper's 2D function task.

The supplement and public script disagree about network depth/regularization.
Expose both predeclared interpretations; neither reuses weights between folds.
No claim of exact numerical reproduction or the paper's density-based UQ.
"""

from dataclasses import dataclass
import hashlib
import time
import warnings

import numpy as np
import torch
from torch import nn
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from sklearn.model_selection import KFold

from .baseline import make_grid
from .spatial_features import wendland


@dataclass(frozen=True)
class PaperConfig:
    grid_size: int = 30
    folds: int = 10
    counts: tuple = (16, 32, 128, 810)
    neural_seeds: tuple = (0, 1, 2)
    split_seed: int = 123
    subset_seed: int = 8400000
    network_seed: int = 8500000
    gp_seed: int = 8600000
    knot_grids: tuple = (10, 19, 37)
    width: int = 100
    batch_size: int = 64
    paper_epochs: int = 200
    code_epochs: int = 400
    gp_restarts: int = 2

    def validate(self):
        if self.grid_size < 3 or not 2 <= self.folds <= self.grid_size**2:
            raise ValueError("Use a valid grid and at least two folds")
        largest_pool = self.grid_size**2 - int(np.ceil(self.grid_size**2 / self.folds))
        if (not self.counts or tuple(sorted(set(self.counts))) != tuple(self.counts)
                or any(not isinstance(n, (int, np.integer)) or n < 2 or n > largest_pool for n in self.counts)):
            raise ValueError("Use increasing measurement counts within every outer training pool")
        if not self.neural_seeds or len(set(self.neural_seeds)) != len(self.neural_seeds):
            raise ValueError("Use distinct neural initialization seeds")
        if min(self.width, self.paper_epochs, self.code_epochs) < 1 or self.batch_size < 2 or self.gp_restarts < 0:
            raise ValueError("Invalid fitting settings")
        if not self.knot_grids or any(n < 2 for n in self.knot_grids):
            raise ValueError("Basis grid sizes must be at least two")


NEURAL_MODELS = ("dk_paper", "dk_code_clean", "coordinate_paper")
GP_MODELS = ("gp_exponential", "gp_rbf")
MODELS = GP_MODELS + NEURAL_MODELS


def paper_signal(points):
    """Deterministic signal from supplement S3.2, with no added sensor noise."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Supply finite 2D positions")
    shifted = points.mean(axis=1) - 0.9
    return np.sin(30 * shifted**4) * np.cos(2 * shifted) + shifted / 2


def paper_features(points, knot_grids):
    """Raw basis-only inputs, support 2.5/m as in the public 2D script.

    No geometry standardization or extra coordinate columns. Earlier project
    features use 2.5/(m-1), so do not silently reuse that different transform.
    """
    features = []
    for size in knot_grids:
        knots = make_grid(size)
        distances = np.linalg.norm(np.asarray(points)[:, None] - knots[None], axis=-1) / (2.5 / size)
        features.append(wendland(distances))
    return np.column_stack(features).astype(np.float32)


def nested_design(config):
    config.validate()
    points = make_grid(config.grid_size)
    rows = []
    splitter = KFold(n_splits=config.folds, shuffle=True, random_state=config.split_seed)
    for fold, (pool, test) in enumerate(splitter.split(points)):
        nested = np.random.default_rng(config.subset_seed + fold).permutation(pool)
        for count in config.counts:
            # Sorted inputs remove arbitrary batching effects due only to a
            # subset's selection order; membership remains strictly nested.
            rows.append({"fold": fold, "count": count, "train": np.sort(nested[:count]), "test": test.copy()})
    return points, rows


def initialization_seed(config, model, fold, member):
    return int(np.random.SeedSequence([config.network_seed, NEURAL_MODELS.index(model), fold, member]).generate_state(
        1, dtype=np.uint64)[0]) % (2**63)


class PaperNetwork(nn.Module):
    def __init__(self, inputs, model, width=100):
        super().__init__()
        if model not in NEURAL_MODELS:
            raise ValueError("Unknown neural model")
        layers = 4 if model == "dk_paper" else 3
        regularized = model != "dk_code_clean"
        modules = []
        for layer in range(layers):
            dense = nn.Linear(inputs if layer == 0 else width, width)
            if layer == 0:
                nn.init.kaiming_uniform_(dense.weight, nonlinearity="relu")
            else:
                nn.init.xavier_uniform_(dense.weight)
            nn.init.zeros_(dense.bias)
            modules.extend((dense, nn.ReLU()))
            if regularized and layer in (0, 1):
                modules.append(nn.Dropout(0.5))
            if regularized and layer in (0, 2):
                # Match Keras epsilon and running-stat decay convention.
                modules.append(nn.BatchNorm1d(width, eps=1e-3, momentum=0.01))
        output = nn.Linear(width, 1)
        nn.init.xavier_uniform_(output.weight); nn.init.zeros_(output.bias)
        self.layers = nn.Sequential(*modules, output)

    def forward(self, features):
        return self.layers(features).squeeze(-1)


def weights_digest(network):
    digest = hashlib.sha256()
    for name, value in network.state_dict().items():
        digest.update(name.encode()); digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def fit_neural(features, train_indices, acquired_values, model, config, seed, checkpoint=None):
    """Only acquired labels are accepted; always construct a fresh network."""
    features = np.asarray(features, dtype=np.float32)
    indices, values = np.asarray(train_indices), np.asarray(acquired_values, dtype=np.float32)
    if (indices.ndim != 1 or len(indices) < 2 or values.shape != indices.shape
            or not np.issubdtype(indices.dtype, np.integer) or not np.isfinite(values).all()
            or np.any(indices < 0) or np.any(indices >= len(features)) or len(set(indices)) != len(indices)):
        raise ValueError("Supply finite acquired labels at unique integer locations")
    epochs = config.code_epochs if model == "dk_code_clean" else config.paper_epochs
    began = time.perf_counter()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        network = PaperNetwork(features.shape[1], model, config.width)
        initial_hash = weights_digest(network)
        optimizer = torch.optim.Adam(network.parameters(), lr=0.001, eps=1e-7)
        query = torch.as_tensor(features)
        x, y = query[indices], torch.as_tensor(values)
        generator = torch.Generator().manual_seed(int(seed) + 319)
        for epoch in range(epochs):
            network.train()
            batches = list(torch.randperm(len(indices), generator=generator).split(config.batch_size))
            if len(batches) > 1 and len(batches[-1]) == 1:
                batches[-2] = torch.cat((batches[-2], batches[-1])); batches.pop()
            for batch in batches:
                optimizer.zero_grad(set_to_none=True)
                loss = (network(x[batch]) - y[batch]).square().mean()
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("Nonfinite neural loss")
                loss.backward(); optimizer.step()
        fit_seconds = time.perf_counter() - began
        network.eval()
        queried = time.perf_counter()
        with torch.inference_mode():
            mean = network(query).numpy().astype(float)
        predict_seconds = time.perf_counter() - queried
        final_hash = weights_digest(network)
        if checkpoint is not None:
            torch.save({"state_dict": network.state_dict(), "inputs": features.shape[1], "model": model,
                        "width": config.width, "epochs": epochs, "seed": seed}, checkpoint)
    return {"mean": mean, "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
            "epochs": epochs, "initial_sha256": initial_hash, "final_sha256": final_hash,
            "parameters": sum(p.numel() for p in network.parameters())}


def fit_gp(points, train_indices, acquired_values, model, config):
    """Source-style zero-mean GP with fitted variance, length and nugget.

    Exponential is Matérn nu=1/2; RBF is a stronger smooth-function control.
    Bounded optimization and numerical jitter are disclosed deviations from GPy.
    """
    if model not in GP_MODELS:
        raise ValueError("Unknown GP model")
    spatial = Matern(1.0, (1e-3, 100.0), nu=0.5) if model == "gp_exponential" else RBF(1.0, (1e-3, 100.0))
    kernel = ConstantKernel(1.0, (1e-6, 1000.0)) * spatial + WhiteKernel(1.0, (1e-10, 1.0))
    network = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, normalize_y=False,
        n_restarts_optimizer=config.gp_restarts, random_state=config.gp_seed)
    began = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        network.fit(np.asarray(points)[train_indices], acquired_values)
    fit_seconds = time.perf_counter() - began
    began = time.perf_counter()
    mean, std = network.predict(points, return_std=True)
    noise = float(network.kernel_.k2.noise_level)
    return {"mean": mean, "variance": np.maximum(std**2 - noise, 0),
        "fit_seconds": fit_seconds, "predict_seconds": time.perf_counter() - began,
        "length_scale": float(network.kernel_.k1.k2.length_scale),
        "process_variance": float(network.kernel_.k1.k1.constant_value), "noise_variance": noise,
        "log_marginal_likelihood": float(network.log_marginal_likelihood_value_),
        "warnings": [str(w.message) for w in caught]}
