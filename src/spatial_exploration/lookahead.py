"""Short-horizon variance-reduction planning with exact GP covariance updates."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class MeasurementPlan:
    first: int
    second: Optional[int]
    score: float
    expected_variance_reduction: float
    planned_distance: float
    sequences_evaluated: int


def mean_variance_reduction(covariance, noise_variance, candidates):
    """Mean reduction in grid variance after one noisy reading per candidate."""
    cross = covariance[:, candidates]
    denominators = np.diag(covariance)[candidates] + noise_variance
    return np.mean(cross * cross, axis=0) / denominators


def plan_measurements(covariance, distances, current, visited, allowed,
                      remaining_budget, noise_variance, depth=2):
    """Maximize whole-grid variance reduction per distance over one/two moves.

    At depth two, evaluate every affordable ordered pair. A one-move terminal
    sequence is considered only when no second measurement is feasible after
    that first move. Execute only the returned first move, then plan again.
    No field values are consumed and caller-owned state is never modified.
    """
    if depth not in (1, 2):
        raise ValueError("Supported planning depths are 1 and 2")
    if noise_variance <= 0 or not np.isfinite(noise_variance):
        raise ValueError("noise_variance must be finite and positive")
    available = allowed & ~visited
    first_candidates = np.flatnonzero(available & (
        distances[current] <= remaining_budget + 1e-12))
    if not len(first_candidates):
        raise RuntimeError("No feasible measurement remains")
    count = len(covariance)
    first_gains = mean_variance_reduction(covariance, noise_variance, first_candidates)
    best_plan, sequences = None, 0
    for offset, first in enumerate(first_candidates):
        first = int(first)
        first_cost = float(distances[current, first])
        if first_cost <= 0:
            raise ValueError("Current position must be marked visited; move costs must be positive")
        first_gain = float(first_gains[offset])
        second_candidates = np.flatnonzero(available & (np.arange(count) != first) & (
            distances[first] <= remaining_budget - first_cost + 1e-12)) if depth == 2 else np.array([], dtype=int)
        if len(second_candidates):
            vector = covariance[:, first]
            denominator = covariance[first, first] + noise_variance
            hypothetical = covariance - np.outer(vector, vector) / denominator
            second_gains = mean_variance_reduction(hypothetical, noise_variance, second_candidates)
            costs = first_cost + distances[first, second_candidates]
            gains = first_gain + second_gains
            scores = gains / costs
            winner = int(np.argmax(scores))
            second = int(second_candidates[winner])
            gain, cost, score = float(gains[winner]), float(costs[winner]), float(scores[winner])
            sequences += len(second_candidates)
        else:
            second = None
            gain, cost, score = first_gain, first_cost, first_gain / first_cost
            sequences += 1
        candidate = MeasurementPlan(first, second, score, gain, cost, 0)
        if best_plan is None or candidate.score > best_plan.score:
            best_plan = candidate
    return MeasurementPlan(best_plan.first, best_plan.second, best_plan.score,
        best_plan.expected_variance_reduction, best_plan.planned_distance, sequences)


def plan_for_environment(env, depth=2):
    return plan_measurements(env.belief.covariance, env.distances, env.current,
        env.visited, env.allowed, env.budget - env.travel_distance,
        env.config.noise_std**2, depth)
