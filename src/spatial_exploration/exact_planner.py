"""Exhaustively solve a small, discrete, known-covariance exploration problem."""

from dataclasses import dataclass
import time

import numpy as np


class SearchLimitExceeded(RuntimeError):
    """Enumeration was interrupted; no optimality claim may be made."""


@dataclass(frozen=True)
class ExactPlan:
    route: tuple
    final_mean_variance: float
    travel_distance: float
    prefixes_evaluated: int
    terminal_routes: int
    maximum_additional_measurements: int
    search_seconds: float


def exhaustive_plan(covariance, distances, current, visited, allowed,
                    remaining_budget, noise_variance, max_prefixes=1000000):
    """Enumerate every feasible non-repeating route, including every prefix.

    Minimize final mean posterior variance, not gain-per-distance. The covariance
    must already condition on all currently visited sites. Selected routes begin
    at `current`; their distance excludes travel before this call. Only budget
    infeasibility and revisits are pruned. If the explicit work limit is reached,
    raise instead of returning an approximate solution labelled optimal.
    """
    covariance = np.asarray(covariance, dtype=float)
    distances = np.asarray(distances, dtype=float)
    visited, allowed = np.asarray(visited, dtype=bool), np.asarray(allowed, dtype=bool)
    count = len(covariance)
    if covariance.shape != (count, count) or distances.shape != (count, count):
        raise ValueError("Covariance and distance matrices must be square and have matching shapes")
    if visited.shape != (count,) or allowed.shape != (count,) or not 0 <= current < count:
        raise ValueError("Invalid masks or current index")
    if not visited[current]:
        raise ValueError("Current position must already be marked visited")
    if not np.isfinite(covariance).all() or not np.isfinite(distances).all():
        raise ValueError("Matrices must be finite")
    off_diagonal = ~np.eye(count, dtype=bool)
    if np.any(distances[off_diagonal] <= 0) or np.any(np.diag(distances) != 0):
        raise ValueError("Distinct locations need positive distances and zero self-distance")
    if not np.isfinite(remaining_budget) or remaining_budget < 0:
        raise ValueError("Budget must be finite and nonnegative")
    if not np.isfinite(noise_variance) or noise_variance <= 0 or max_prefixes < 1:
        raise ValueError("Noise variance and search limit must be positive")
    best_route = (int(current),)
    best_variance = float(np.trace(covariance) / count)
    best_distance = 0.0
    prefixes, terminals, maximum_depth = 0, 0, 0
    started = time.perf_counter()

    def visit(posterior, position, available, traveled, route):
        nonlocal best_route, best_variance, best_distance, prefixes, terminals, maximum_depth
        prefixes += 1
        if prefixes > max_prefixes:
            raise SearchLimitExceeded(
                f"Exceeded {max_prefixes} prefixes; reduce the grid or budget. Search is not complete.")
        maximum_depth = max(maximum_depth, len(route) - 1)
        variance = float(np.trace(posterior) / count)
        # Treat objective differences below 1e-12 as numerical ties, preferring
        # the shorter route. This is a numerical, not symbolic, exact reference.
        if variance < best_variance - 1e-12 or (
            abs(variance - best_variance) <= 1e-12 and traveled < best_distance
        ):
            best_variance, best_distance, best_route = variance, traveled, route
        candidates = np.flatnonzero(available & (
            distances[position] <= remaining_budget - traveled + 1e-12))
        if not len(candidates):
            terminals += 1
        for index in candidates:
            index = int(index)
            vector = posterior[:, index]
            updated = posterior - np.outer(vector, vector) / (posterior[index, index] + noise_variance)
            updated = (updated + updated.T) / 2
            next_available = available.copy()
            next_available[index] = False
            visit(updated, index, next_available,
                  traveled + float(distances[position, index]), route + (index,))

    visit(covariance, int(current), allowed & ~visited, 0.0, (int(current),))
    return ExactPlan(best_route, best_variance, best_distance, prefixes,
                     terminals, maximum_depth, time.perf_counter() - started)
