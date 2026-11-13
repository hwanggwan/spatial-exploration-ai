"""A shared uncertainty-per-distance heuristic, not integrated information gain."""

import numpy as np


def choose_uncertainty_action(points, variance, current, visited, allowed,
                              remaining_distance, remaining_observations):
    points = np.asarray(points, dtype=float)
    variance = np.asarray(variance, dtype=float)
    visited, allowed = np.asarray(visited, dtype=bool), np.asarray(allowed, dtype=bool)
    size = len(points)
    if points.shape != (size, 2) or not np.isfinite(points).all():
        raise ValueError("Use finite two-dimensional coordinates")
    if variance.shape != (size,) or not np.isfinite(variance).all() or np.any(variance < 0):
        raise ValueError("Use finite nonnegative marginal variances")
    if visited.shape != (size,) or allowed.shape != (size,) or not 0 <= current < size:
        raise ValueError("Masks and current location must match the domain")
    if not np.isfinite(remaining_distance) or remaining_distance < 0:
        raise ValueError("Remaining travel distance must be nonnegative")
    if remaining_observations <= 0:
        return None
    costs = np.linalg.norm(points - points[current], axis=1)
    candidates = np.flatnonzero(allowed & ~visited & (costs > 1e-12)
                               & (costs <= remaining_distance + 1e-12))
    if not len(candidates):
        return None
    # Sorted candidate indices and first argmax give deterministic tie breaking.
    scores = variance[candidates] / costs[candidates]
    return int(candidates[np.argmax(scores)])
