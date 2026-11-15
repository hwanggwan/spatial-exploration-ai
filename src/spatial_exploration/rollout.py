"""One-step policy improvement using deterministic greedy completions."""

from dataclasses import dataclass

import numpy as np

from .lookahead import plan_measurements


@dataclass(frozen=True)
class RolloutPlan:
    first: int
    route: tuple
    final_mean_variance: float
    planned_distance: float
    greedy_route: tuple
    greedy_final_mean_variance: float
    candidates_evaluated: int
    simulated_measurements: int


def plan_rollout(covariance, distances, current, visited, allowed,
                 remaining_budget, noise_variance, *, remaining_observations=None,
                 terminal_variance=None, first_move_candidates=None):
    """Try feasible first moves, then run greedy to a terminal state.

    Minimize the resulting *final* whole-grid mean posterior variance, stopping
    each hypothetical route at either the travel or optional observation cap. Execute
    only the first move and call again at the next state. The base policy is the
    existing deterministic gain-per-distance greedy rule, including its index
    tie breaking. Its own first move is always included. Prefer that completion
    on objective ties within 1e-12; this avoids gratuitous numerical changes.

    Covariance is already conditioned on all visited sites. No future measured
    values or latent field are used. The optional first-move list limits only
    the first-move search. The baseline first move and full greedy suffix
    remain available. With a fixed covariance model and exact arithmetic,
    repeated rollout cannot do worse than the base policy
    under the default Gaussian terminal score. A custom terminal variance can
    rescore each greedy completion, excluding the already measured start.
    This is a policy-improvement property, not a global optimality claim.
    """
    covariance = np.asarray(covariance, dtype=float)
    distances = np.asarray(distances, dtype=float)
    visited, allowed = np.asarray(visited, dtype=bool), np.asarray(allowed, dtype=bool)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("Covariance must be square")
    count = len(covariance)
    if distances.shape != (count, count):
        raise ValueError("Distance and covariance matrices must have matching shapes")
    if visited.shape != (count,) or allowed.shape != (count,) or not 0 <= current < count:
        raise ValueError("Invalid masks or current index")
    if not visited[current]:
        raise ValueError("Current position must already be marked visited")
    if not np.isfinite(covariance).all() or not np.isfinite(distances).all():
        raise ValueError("Matrices must be finite")
    off_diagonal = ~np.eye(count, dtype=bool)
    if np.any(distances[off_diagonal] <= 0) or np.any(np.diag(distances) != 0):
        raise ValueError("Distinct locations need positive distances and zero self-distance")
    if not np.isfinite(remaining_budget) or remaining_budget < -1e-12:
        raise ValueError("Budget must be finite and nonnegative within budget tolerance")
    if not np.isfinite(noise_variance) or noise_variance <= 0:
        raise ValueError("Noise variance must be finite and positive")
    if remaining_observations is not None:
        if not isinstance(remaining_observations, (int, np.integer)) or isinstance(remaining_observations, (bool, np.bool_)):
            raise ValueError("Remaining observations must be an integer")
        if remaining_observations < 0:
            raise ValueError("Remaining observations must be nonnegative")
        if remaining_observations == 0:
            raise RuntimeError("No observations remain")
    candidates = np.flatnonzero(allowed & ~visited & (
        distances[current] <= remaining_budget + 1e-12))
    if not len(candidates):
        raise RuntimeError("No feasible measurement remains")

    base_first = plan_measurements(covariance, distances, current, visited, allowed,
                                   remaining_budget, noise_variance, depth=1).first
    if first_move_candidates is not None:
        candidates = candidates[np.isin(candidates, first_move_candidates)]
    # Start with the exact base-policy completion as the incumbent. With no
    # material improvement, preserve its first action and deterministic ties.
    ordered = [base_first] + [int(index) for index in candidates if index != base_first]
    best = None
    greedy_route, greedy_variance = None, None
    simulations = 0
    for first in ordered:
        posterior = covariance.copy()
        hypothetical_visited = visited.copy()
        route, travel = [int(current)], 0.0
        action = first
        while True:
            travel += float(distances[route[-1], action])
            route.append(action)
            hypothetical_visited[action] = True
            vector = posterior[:, action].copy()
            posterior -= np.outer(vector, vector) / (posterior[action, action] + noise_variance)
            posterior = (posterior + posterior.T) / 2
            simulations += 1
            if remaining_observations is not None and len(route) - 1 >= remaining_observations:
                break
            feasible = allowed & ~hypothetical_visited & (
                distances[action] <= remaining_budget - travel + 1e-12)
            if not feasible.any():
                break
            action = plan_measurements(posterior, distances, action, hypothetical_visited,
                allowed, remaining_budget - travel, noise_variance, depth=1).first
        if terminal_variance is None:
            variance = float(np.trace(posterior) / count)
        else:
            variance = terminal_variance(tuple(route[1:]))
        if first == base_first:
            greedy_route, greedy_variance = tuple(route), variance
        if best is None or variance < best[0] - 1e-12:
            best = (variance, tuple(route), travel)
    return RolloutPlan(best[1][1], best[1], best[0], best[2], greedy_route,
                       greedy_variance, len(ordered), simulations)


def rollout_for_environment(env):
    return plan_rollout(env.belief.covariance, env.distances, env.current,
        env.visited, env.allowed, env.budget - env.travel_distance,
        env.config.noise_std**2)
