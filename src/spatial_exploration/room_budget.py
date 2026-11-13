"""A room quota for greedy sensing on a known two-room floor plan."""

import numpy as np

from .lookahead import plan_measurements


def reserve_costs(distances, origins, eligible, count):
    """Construct feasible nearest-neighbor completions; not shortest-tour bounds."""
    if count == 0:
        return np.zeros(len(origins))
    if np.count_nonzero(eligible) < count:
        return np.full(len(origins), np.inf)
    # One hypothetical continuation per candidate; uses geometry alone.
    remaining = np.broadcast_to(eligible, (len(origins), len(eligible))).copy()
    rows = np.arange(len(origins))
    current = np.asarray(origins)
    remaining[rows, current] = False
    cost = np.zeros(len(origins))
    for _ in range(count):
        choices = np.where(remaining, distances[current], np.inf)
        sites = np.argmin(choices, axis=1)
        cost += choices[rows, sites]
        remaining[rows, sites] = False
        current = sites
    return cost


class RoomQuota:
    """Try to spend the first half in the starting room and reserve the other half."""

    def __init__(self, layout, allowed, start, cap, distance_cap):
        self.layout = layout
        self.allowed = allowed
        self.home = layout.room == layout.room[start]
        self.home_target = (cap + 1) // 2
        self.away_target = cap // 2
        self.active = self.away_target > 0 and bool(reserve_costs(
            layout.distances, [start], allowed & ~self.home, self.away_target)[0]
            <= distance_cap + 1e-12)

    def next(self, belief, current, visited, remaining_distance):
        distances = self.layout.distances
        available = self.allowed & ~visited
        feasible = available & (distances[current] <= remaining_distance + 1e-12)
        if not feasible.any():
            return None

        def reserving(candidates, count):
            sites = np.flatnonzero(candidates)
            if not len(sites):
                return candidates
            reserve = reserve_costs(distances, sites, available & ~self.home, count)
            ok = distances[current, sites] + reserve <= remaining_distance + 1e-12
            result = np.zeros_like(candidates)
            result[sites[ok]] = True
            return result

        candidates = feasible
        if self.active:
            candidates = np.zeros_like(feasible)
            if self.home[current] and np.count_nonzero(visited & self.home) < self.home_target:
                candidates = reserving(feasible & self.home, self.away_target)
            if not candidates.any():
                # If home samples would spend the reserve, switch rooms early.
                # Do not cross back. All choices still have a feasible quota witness.
                away_after_choice = max(0, self.away_target - np.count_nonzero(visited & ~self.home) - 1)
                candidates = reserving(feasible & ~self.home, away_after_choice)
            if not candidates.any():
                return None
        return plan_measurements(belief.covariance, distances, current, visited,
                                 candidates, remaining_distance, belief.noise_variance, depth=1).first
