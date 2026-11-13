"""Planner-facing exploration state receiving only acquired observations."""

import numpy as np

from .baseline import make_grid
from .exploration_env import encode_policy_observation
from .lookahead import plan_measurements
from .rollout import plan_rollout


class ExplorationBeliefState:
    """No latent field, potential sensor array, or hidden kernel in this interface.

    The supplied mapper already conditions on the free initial measurement.
    Fixed mapper parameters are explicit model assumptions. Estimated mappers
    supply their own current parameters. The evaluator keeps truth separately.
    """

    def __init__(self, config, belief, start, budget, fixed_length_scale, fixed_process_variance):
        config.validate()
        self.config, self.belief = config, belief
        self.points = make_grid(config.grid_size)
        self.count = len(self.points)
        self.distances = np.linalg.norm(self.points[:, None] - self.points[None, :], axis=-1)
        self.allowed = np.arange(self.count) % 5 != 4
        if not 0 <= start < self.count or not self.allowed[start]:
            raise ValueError("Start must be a permitted grid index")
        if not np.isfinite(budget) or budget < 0:
            raise ValueError("Budget must be finite and nonnegative")
        if belief.covariance.shape != (self.count, self.count):
            raise ValueError("Mapper covariance does not match the grid")
        self.current, self.budget = int(start), float(budget)
        self.fixed_length_scale = fixed_length_scale
        self.fixed_process_variance = fixed_process_variance
        self.visited = np.zeros(self.count, dtype=bool)
        self.visited[start] = True
        self.route, self.cumulative_distances = [int(start)], [0.0]
        self.travel_distance = 0.0

    @property
    def length_scale(self):
        return float(getattr(self.belief, "length_scale", self.fixed_length_scale))

    @property
    def process_variance(self):
        return float(getattr(self.belief, "process_variance", self.fixed_process_variance))

    def action_masks(self):
        return self.allowed & ~self.visited & (
            self.distances[self.current] <= self.budget - self.travel_distance + 1e-12)

    @property
    def finished(self):
        return not bool(self.action_masks().any())

    def observation(self):
        return encode_policy_observation(self.belief.covariance, self.visited,
            self.points[self.current], self.budget - self.travel_distance,
            self.config.budget_range[1], self.length_scale, self.config.length_scale_range[1])

    def input_diagnostics(self):
        observation = self.observation()
        return {"model_length_scale": self.length_scale, "model_process_variance": self.process_variance,
            "length_outside_training_range": not self.config.length_scale_range[0] <= self.length_scale <= self.config.length_scale_range[1],
            "posterior_diagonal_above_training_cap": bool(np.max(np.diag(self.belief.covariance)) > 1 + 1e-6),
            "observation_outside_original_box": bool(np.any(np.abs(observation) > 1.1 + 1e-6)),
            "maximum_absolute_covariance": float(np.max(np.abs(self.belief.covariance)))}

    def move_and_observe(self, action, value):
        if not 0 <= action < self.count or not self.action_masks()[action]:
            raise ValueError("Measurement must be permitted, unvisited, and affordable")
        self.travel_distance += float(self.distances[self.current, action])
        self.current = int(action)
        self.visited[action] = True
        self.route.append(int(action))
        self.cumulative_distances.append(self.travel_distance)
        self.belief.observe(action, value)


def select_action(method, policy, state):
    """Only modeled belief and physical state reach action selection."""
    if method in ("greedy", "rollout"):
        arguments = (state.belief.covariance, state.distances, state.current, state.visited,
            state.allowed, state.budget - state.travel_distance, state.config.noise_std**2)
        plan = plan_rollout(*arguments) if method == "rollout" else plan_measurements(*arguments, depth=1)
        return plan.first, plan
    if policy is None:
        raise ValueError("A frozen neural policy is required")
    action, _ = policy.predict(state.observation(), action_masks=state.action_masks(), deterministic=True)
    return int(action), None
