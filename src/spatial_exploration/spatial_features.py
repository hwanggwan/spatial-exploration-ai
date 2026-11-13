"""Geometry-only multiresolution spatial representation shared by all controls."""

import numpy as np


def wendland(distance):
    distance = np.asarray(distance, dtype=float)
    if np.any(distance < 0) or not np.isfinite(distance).all():
        raise ValueError("Distances must be finite and nonnegative")
    bounded = np.minimum(distance, 1)
    return (1 - bounded)**6 * (35 * bounded**2 + 18 * bounded + 3) / 3


def raw_features(points, model, config):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Supply finite two-dimensional coordinates")
    if model == "coordinate_dnn":
        return points.copy()
    if model != "deepkriging":
        raise ValueError("Unknown neural representation")
    features = [points]
    for size in config.knot_grids:
        xx, yy = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
        knots = np.column_stack((xx.ravel(), yy.ravel()))
        support = config.support_multiplier / (size - 1)
        distance = np.linalg.norm(points[:, None] - knots[None, :], axis=-1) / support
        features.append(wendland(distance))
    return np.column_stack(features)


def standardized_features(points, model, config):
    """Geometry-only transform; uses no measured values or hidden field targets."""
    raw = raw_features(points, model, config)
    center, scale = raw.mean(axis=0), raw.std(axis=0)
    scale = np.maximum(scale, 1e-6)
    return ((raw - center) / scale).astype(np.float32), center, scale
