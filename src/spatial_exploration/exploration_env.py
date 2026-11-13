"""Budgeted information gathering with no hidden field in state or reward."""

from dataclasses import dataclass

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .baseline import GPBelief, make_grid, rbf_covariance


def encode_policy_observation(covariance, visited, current_xy, remaining_budget,
                              budget_scale, length_scale, length_scale_scale):
    """Shared frozen-policy input contract; preserve raw values without clipping.

    The scales are training-time constants, not the hidden generating parameters.
    In a plug-in deployment, pass the mapper's current estimated length scale.
    """
    return np.concatenate((covariance.ravel(), visited.astype(float), current_xy,
        [max(0.0, remaining_budget) / budget_scale,
         length_scale / length_scale_scale])).astype(np.float32)


@dataclass(frozen=True)
class ExplorationConfig:
    grid_size: int = 7
    length_scale_range: tuple = (0.12, 0.35)
    budget_range: tuple = (1.5, 2.5)
    noise_std: float = 0.1
    reward_scale: float = 100.0

    def validate(self):
        if self.grid_size < 3:
            raise ValueError("grid_size must be at least 3")
        for bounds in (self.length_scale_range, self.budget_range):
            if len(bounds) != 2 or not all(np.isfinite(bounds)) or not 0 < bounds[0] <= bounds[1]:
                raise ValueError("Ranges must contain finite positive ordered bounds")
        if self.budget_range[0] < 2 / (self.grid_size - 1):
            raise ValueError("Minimum budget must allow a move from every permitted start")
        if not np.isfinite(self.noise_std) or self.noise_std <= 0:
            raise ValueError("noise_std must be positive")
        if not np.isfinite(self.reward_scale) or self.reward_scale <= 0:
            raise ValueError("reward_scale must be positive")


class SpatialExplorationEnv(gym.Env):
    """Discrete next-measurement choices with exact GP covariance updates.

    For the known fixed kernel, the posterior covariance and variance-reduction
    objective are independent of measurement values. Training therefore needs no
    sampled field: zero pseudo-observations produce the exact covariance states.
    Actual latent-field prediction is evaluated separately on paired fields.
    """

    metadata = {"render_modes": []}

    def __init__(self, config=None):
        super().__init__()
        self.config = config or ExplorationConfig()
        self.config.validate()
        self.points = make_grid(self.config.grid_size)
        self.count = len(self.points)
        self.evaluation_mask = np.arange(self.count) % 5 == 4
        self.allowed = ~self.evaluation_mask
        self.distances = np.linalg.norm(self.points[:, None] - self.points[None, :], axis=-1)
        self.action_space = spaces.Discrete(self.count)
        # Covariance, visited flags, current xy, remaining budget, kernel length.
        self.observation_space = spaces.Box(-1.1, 1.1,
            shape=(self.count**2 + self.count + 4,), dtype=np.float32)
        self.belief = None
        self.finished = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.length_scale = float(options.get("length_scale",
            self.np_random.uniform(*self.config.length_scale_range)))
        self.budget = float(options.get("budget", self.np_random.uniform(*self.config.budget_range)))
        permitted = np.flatnonzero(self.allowed)
        self.current = int(options.get("start_index", self.np_random.choice(permitted)))
        for value, bounds in ((self.length_scale, self.config.length_scale_range),
                              (self.budget, self.config.budget_range)):
            if not np.isfinite(value) or not bounds[0] <= value <= bounds[1]:
                raise ValueError("Reset settings must lie in the configured ranges")
        if self.current not in permitted:
            raise ValueError("Starting index must be a permitted sensing location")
        self.belief = GPBelief(rbf_covariance(self.points, self.length_scale), self.config.noise_std)
        self.belief.observe(self.current, 0.0)
        self.visited = np.zeros(self.count, dtype=bool)
        self.visited[self.current] = True
        self.route = [self.current]
        self.travel_distance = 0.0
        self.cumulative_distances = [0.0]
        self.initial_variance = self.mean_variance
        self.finished = False
        return self._observation(), self._info()

    @property
    def mean_variance(self):
        return float(np.trace(self.belief.covariance) / self.count)

    def action_masks(self):
        if self.finished:
            return np.zeros(self.count, dtype=bool)
        return self.allowed & ~self.visited & (
            self.distances[self.current] <= self.budget - self.travel_distance + 1e-12
        )

    def greedy_action(self):
        candidates = np.flatnonzero(self.action_masks())
        if not len(candidates):
            raise RuntimeError("No feasible action remains")
        cross = self.belief.covariance[:, candidates]
        variances = np.diag(self.belief.covariance)[candidates] + self.config.noise_std**2
        reduction = np.mean(cross * cross, axis=0) / variances
        return int(candidates[np.argmax(reduction / self.distances[self.current, candidates])])

    def step(self, action):
        if self.finished:
            raise RuntimeError("Call reset before stepping a terminated episode")
        index = int(action)
        if not self.action_space.contains(index) or not self.action_masks()[index]:
            # Gym environment checkers sample without masks. A masked policy
            # should never take this branch; retain an explicit diagnostic.
            self.finished = True
            return self._observation(), -self.config.reward_scale, True, False, self._info(invalid=True)
        previous_variance = self.mean_variance
        self.travel_distance += float(self.distances[self.current, index])
        self.current = index
        self.visited[index] = True
        self.route.append(index)
        self.cumulative_distances.append(self.travel_distance)
        self.belief.observe(index, 0.0)
        reward = self.config.reward_scale * (previous_variance - self.mean_variance)
        self.finished = not bool(self.action_masks().any())
        return self._observation(), reward, self.finished, False, self._info()

    def _observation(self):
        return encode_policy_observation(self.belief.covariance, self.visited,
            self.points[self.current], self.budget - self.travel_distance,
            self.config.budget_range[1], self.length_scale, self.config.length_scale_range[1])

    def _info(self, invalid=False):
        return {"mean_variance": self.mean_variance,
                "information_gain": self.initial_variance - self.mean_variance,
                "travel_distance": self.travel_distance, "measurements": len(self.route),
                "invalid_action": invalid}
