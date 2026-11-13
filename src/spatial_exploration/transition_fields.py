"""Paired changes in dependence-transition sharpness, not physical barriers."""

import numpy as np

from .baseline import make_grid, rbf_covariance


TRANSITIONS = ("gradual", "moderate", "sharp")
WIDTHS = (None, 0.1, 0.05)
TRANSITION_LABELS = {"gradual": "Gradual: q(y)=y", "moderate": "Moderate: logistic tau=0.10",
                     "sharp": "Sharp: logistic tau=0.05"}


def transition_weights(y, width):
    y = np.asarray(y, dtype=float)
    if not np.isfinite(y).all() or np.any((y < 0) | (y > 1)):
        raise ValueError("Coordinates must lie in the finite unit interval")
    if width is None:
        q = y
    else:
        if not np.isfinite(width) or width <= 0:
            raise ValueError("Logistic width must be finite and positive")
        t = (y - 0.5) / width
        small = np.exp(-np.abs(t))
        q = np.where(t >= 0, 1 / (1 + small), small / (1 + small))
    angle = np.pi * q / 2
    return np.cos(angle), np.sin(angle)


def transition_covariances(points, config):
    long = rbf_covariance(points, config.long_length)
    short = rbf_covariance(points, config.short_length)
    weights = [transition_weights(points[:, 1], width) for width in WIDTHS]
    return np.asarray([np.outer(a, a) * long + np.outer(b, b) * short for a, b in weights])


def generate_transition_dataset(config):
    """Same component realizations, sensor noise and design across conditions.

    Truth and generating covariances are returned in a separate evaluator-only
    object. Unlike the earlier covariance-Cholesky pairing, this explicitly
    holds the underlying long/short component fields fixed between conditions.
    """
    config.validate()
    points = make_grid(config.grid_size)
    mask = np.arange(len(points)) % 5 == 4
    allowed = np.flatnonzero(~mask)
    indices = np.asarray([np.random.default_rng(config.design_seed_start + field).permutation(allowed)[
        :config.sample_sizes[-1]] for field in range(config.fields)])
    jitter = 1e-10 * np.eye(len(points))
    long_factor = np.linalg.cholesky(rbf_covariance(points, config.long_length) + jitter)
    short_factor = np.linalg.cholesky(rbf_covariance(points, config.short_length) + jitter)
    truth = np.empty((len(TRANSITIONS), config.fields, len(points)))
    values = np.empty((len(TRANSITIONS), config.fields, config.sample_sizes[-1]))
    weights = [transition_weights(points[:, 1], width) for width in WIDTHS]
    for field in range(config.fields):
        zlong, zshort, noise = np.random.default_rng(config.field_seed_start + field).standard_normal((3, len(points)))
        long, short = long_factor @ zlong, short_factor @ zshort
        for scenario, (a, b) in enumerate(weights):
            truth[scenario, field] = a * long + b * short
            values[scenario, field] = (truth[scenario, field] + config.noise_std * noise)[indices[field]]
    observations = {"points": points, "scenario_names": np.asarray(TRANSITIONS),
        "field_ids": np.arange(config.fields), "sample_sizes": np.asarray(config.sample_sizes),
        "observation_indices": indices, "observed_values": values}
    evaluator = {"truth": truth, "evaluation_mask": mask,
                 "true_covariances": transition_covariances(points, config)}
    return observations, evaluator
