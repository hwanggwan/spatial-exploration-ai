"""Feasible room allocations on a known grid, without reading sensor values."""

from itertools import permutations

import numpy as np
from scipy.sparse.csgraph import shortest_path

from .barrier_gp import RoomLayout
from .baseline import make_grid
from .lookahead import plan_measurements


SIZE = 12
STARTS = (13, 49, 85, 121)
TOL = 1e-12


def three_rooms(doors):
    """Three rooms in a chain, with widths 3, 6, and 3 grid columns."""
    if len(doors) != 2 or any(not 0 <= y < SIZE for y in doors):
        raise ValueError("Specify one valid door row per wall")
    row, col = np.divmod(np.arange(SIZE**2), SIZE)
    room = np.digitize(col, (3, 9))
    adjacent = (abs(row[:, None] - row) + abs(col[:, None] - col)) == 1
    across = room[:, None] != room
    door_edge = np.zeros_like(adjacent)
    for col, y in zip((2, 8), doors):
        left = y * SIZE + col
        door_edge[left, left + 1] = door_edge[left + 1, left] = True
    wall = adjacent & across & ~door_edge
    movement = adjacent & ~wall
    hops, predecessor = shortest_path(movement.astype(float), directed=False, unweighted=True,
                                     return_predecessors=True)
    return RoomLayout(SIZE, tuple(doors), make_grid(SIZE), room, wall, movement,
                      hops / (SIZE - 1), predecessor)


def completion_costs(layout, available, origins, order, quotas):
    """Construct one legal nearest-neighbor suffix per origin, in room order."""
    origins = np.asarray(origins, dtype=int)
    remaining = np.broadcast_to(available, (len(origins), len(available))).copy()
    rows = np.arange(len(origins))
    remaining[rows, origins] = False
    cost = np.zeros(len(origins))
    current = origins
    for room in order:
        in_room = layout.room == room
        for _ in range(quotas[room]):
            choices = np.where(remaining & in_room, layout.distances[current], np.inf)
            nearest = choices.argmin(axis=1)
            cost += choices[rows, nearest]
            remaining[rows, nearest] = False
            current = nearest
    return cost


def allocate(total, order, weights):
    """Give each selected room one reading, then apportion the rest."""
    counts = np.zeros(3, dtype=int)
    selected = np.asarray(order)
    exact = (total - len(order)) * weights[selected] / weights[selected].sum()
    integral = exact.astype(int)
    counts[selected] = 1 + integral
    remainder = total - counts.sum()
    counts[selected[np.argsort(-(exact - integral), kind="stable")[:remainder]]] += 1
    return counts


def choose_allocation(layout, allowed, start, cap, budget, covariance, method):
    if method not in ("equal", "equal_coverage", "priority"):
        raise ValueError("Unknown room allocator")
    home = int(layout.room[start])
    others = [r for r in range(3) if r != home]
    available = allowed.copy()
    available[start] = False
    mass = np.bincount(layout.room, weights=np.diag(covariance), minlength=3)
    entry = np.array([layout.distances[start, available & (layout.room == r)].min(initial=np.inf)
                      for r in range(3)])
    entry[home] = 0
    weights = mass / (1 + entry)
    best_key, best = None, None
    for length in range(3):
        for tail in permutations(others, length):
            order = (home, *tail)
            if np.any(weights[list(order)] <= 0):
                continue
            for n in range(cap, len(order) - 1, -1):
                quotas = allocate(n, order, weights if method == "priority" else np.ones(3))
                pending = quotas.copy()
                pending[home] -= 1
                cost = completion_costs(layout, available, [start], order, pending)[0]
                if cost > budget + TOL:
                    continue
                # A fixed diminishing-return proxy, not the rollout objective.
                utility = float(np.sum(weights * -np.expm1(-quotas / 4)))
                if method == "equal":
                    rank = (n, len(order), -cost)
                elif method == "equal_coverage":
                    rank = (len(order), n, -cost)
                else:
                    rank = (utility, n, -cost)
                if best_key is None or rank > best_key:
                    best_key, best = rank, (order, pending)
                break  # For a fixed order, additional allocated readings only increase utility.
    if best is None:
        raise RuntimeError("No feasible allocation, even with only the start")
    return best


def quota_next(layout, allowed, visited, current, remaining, order, pending, belief):
    """Choose greedily only among moves with a certified quota completion."""
    next_room = next(room for room in order if pending[room])
    sites = np.flatnonzero(allowed & ~visited & (layout.room == next_room)
                           & (layout.distances[current] <= remaining + TOL))
    after = pending.copy()
    after[next_room] -= 1
    witness = completion_costs(layout, allowed & ~visited, sites, order, after)
    legal = sites[layout.distances[current, sites] + witness <= remaining + TOL]
    if not len(legal):
        raise RuntimeError("Lost the allocated route's feasibility witness")
    candidates = np.zeros_like(allowed)
    candidates[legal] = True
    return plan_measurements(belief.covariance, layout.distances, current, visited, candidates,
                             remaining, belief.noise_variance, depth=1).first
