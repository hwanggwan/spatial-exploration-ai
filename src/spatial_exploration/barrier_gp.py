"""Known room geometry, graph-based covariance, and observation-only GP fitting."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_triangular
from scipy.sparse.csgraph import shortest_path

from .baseline import GPBelief, make_grid, rbf_covariance
from .lookahead import plan_measurements


LENGTHS = (0.09, 0.14, 0.22, 0.34, 0.50)
GRAPH_WEIGHTS = (0.0, 0.5, 1.0)


@dataclass(frozen=True)
class RoomLayout:
    size: int
    door_row: int
    points: np.ndarray
    room: np.ndarray
    wall_edges: np.ndarray
    movement: np.ndarray
    distances: np.ndarray
    predecessors: np.ndarray

    def walk(self, start, end):
        """Expand a measurement-to-measurement leg into legal adjacent moves."""
        path = [int(end)]
        while path[-1] != start:
            previous = int(self.predecessors[start, path[-1]])
            if previous < 0:
                raise ValueError("No movement path")
            path.append(previous)
        return path[::-1]


def room_layout(size, door_row):
    if size < 6 or size % 2 or not 0 <= door_row < size:
        raise ValueError("Use an even grid >= 6 with a door on a grid row")
    points = make_grid(size)
    row, col = np.divmod(np.arange(size**2), size)
    room = col >= size // 2
    adjacent = (np.abs(row[:, None] - row) + np.abs(col[:, None] - col)) == 1
    wall_edges = adjacent & (room[:, None] != room) & (row[:, None] != door_row)
    movement = adjacent & ~wall_edges
    distances, predecessors = shortest_path(movement.astype(float), directed=False,
                                             unweighted=True, return_predecessors=True)
    return RoomLayout(size, door_row, points, room, wall_edges, movement,
                      distances / (size - 1), predecessors)


def graph_spectrum(layout, permeability):
    if not 0 <= permeability <= 1:
        raise ValueError("Permeability must be in [0, 1]")
    edges = layout.movement.astype(float) + permeability * layout.wall_edges
    eigenvalues, vectors = np.linalg.eigh(np.diag(edges.sum(axis=1)) - edges)
    return np.maximum(eigenvalues, 0), vectors


def unit_covariance(vectors, spectrum):
    raw = (vectors * spectrum) @ vectors.T
    sd = np.sqrt(raw.diagonal())
    return raw / (sd[:, None] * sd)


def transport_covariance(layout, permeability, kappa):
    """Evaluator's resolvent-squared field; not one of the fitted kernels."""
    values, vectors = graph_spectrum(layout, permeability)
    return unit_covariance(vectors, (kappa**2 + values)**-2)


@dataclass(frozen=True)
class KernelBank:
    parameters: tuple
    covariances: np.ndarray
    initial: int

    def posterior(self, indices, observations, noise_std):
        """Select a candidate by marginal likelihood after four acquisitions."""
        idx = np.asarray(indices, dtype=int)
        y = np.asarray(observations, dtype=float)
        winner = self.initial
        if len(idx) >= 4:
            sample = self.covariances[:, idx[:, None], idx] + noise_std**2 * np.eye(len(idx))
            factors = np.linalg.cholesky(sample)
            rhs = np.broadcast_to(y, (len(factors), len(y)))[..., None]
            whitened = np.linalg.solve(factors, rhs)[..., 0]
            log_evidence = -0.5 * (whitened**2).sum(axis=1) - np.log(
                np.diagonal(factors, axis1=1, axis2=2)).sum(axis=1)
            winner = int(np.argmax(log_evidence))
        prior = self.covariances[winner]
        belief = GPBelief(prior, noise_std)
        if len(idx):
            factor = np.linalg.cholesky(prior[np.ix_(idx, idx)] + noise_std**2 * np.eye(len(idx)))
            cross = solve_triangular(factor, prior[idx], lower=True, check_finite=False)
            belief.mean = cross.T @ solve_triangular(factor, y, lower=True, check_finite=False)
            belief.covariance -= cross.T @ cross
        return belief, self.parameters[winner]


def kernel_banks(layout):
    values, vectors = graph_spectrum(layout, 0.0)
    stationary, aware = [], []
    for length in LENGTHS:
        rbf = rbf_covariance(layout.points, length)
        heat = unit_covariance(vectors, np.exp(-0.5 * (length * (layout.size - 1))**2 * values))
        stationary.append(((length, 0.0), rbf))
        aware.extend(((length, weight), (1 - weight) * rbf + weight * heat)
                     for weight in GRAPH_WEIGHTS)

    def bank(candidates, initial):
        parameters, matrices = zip(*candidates)
        return KernelBank(parameters, np.asarray(matrices), parameters.index(initial))

    return {"stationary": bank(stationary, (0.22, 0.0)),
            "geometry": bank(aware, (0.22, 0.5))}


def next_measurement(belief, layout, route, allowed, remaining_distance, remaining_count):
    """Planner sees modeled covariance and geometry, never unacquired values."""
    if remaining_count <= 0:
        return None
    visited = np.zeros(len(allowed), dtype=bool)
    visited[route] = True
    current = route[-1]
    if not np.any(allowed & ~visited & (layout.distances[current] <= remaining_distance + 1e-12)):
        return None
    return plan_measurements(belief.covariance, layout.distances, current, visited, allowed,
                             remaining_distance, belief.noise_variance, depth=1).first
