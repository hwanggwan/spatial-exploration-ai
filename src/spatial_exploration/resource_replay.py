"""Shared sequential sensing routes under observation and travel constraints."""

from dataclasses import dataclass

import numpy as np

from .baseline import Config, explore, make_grid, rbf_covariance
from .transition_fields import transition_weights


@dataclass(frozen=True)
class ResourceConfig:
    grid_size: int = 15
    fields: int = 64
    calibration_fields: int = 8
    observation_caps: tuple = (8, 16)
    travel_budgets: tuple = (1.0, 2.0)
    noise_std: float = 0.1
    long_length: float = 0.35
    short_length: float = 0.12
    design_length: float = 0.35
    design_variance: float = 1.5

    def validate(self):
        if min(self.grid_size, self.fields, self.calibration_fields) < 2 or self.grid_size < 3:
            raise ValueError("Need a grid of at least three and at least two fields per split")
        if not self.observation_caps or any(not isinstance(n, (int, np.integer)) or isinstance(n, (bool, np.bool_))
                or n < 1 for n in self.observation_caps):
            raise ValueError("Observation caps must be positive integers")
        if not self.travel_budgets or any(not np.isfinite(b) or b < 0 for b in self.travel_budgets):
            raise ValueError("Travel budgets must be finite and nonnegative")
        if len(set(self.observation_caps)) != len(self.observation_caps) or len(set(self.travel_budgets)) != len(self.travel_budgets):
            raise ValueError("Budget conditions must be unique")
        if any(not np.isfinite(x) or x <= 0 for x in (self.noise_std,self.long_length,self.short_length,
                                                      self.design_length,self.design_variance)):
            raise ValueError("Noise and spatial scales must be finite and positive")
        if self.short_length >= self.long_length:
            raise ValueError("Short length must be smaller than long length")


def geometry(config):
    points = make_grid(config.grid_size)
    evaluation_mask = np.arange(len(points)) % 5 == 4
    return points, evaluation_mask


def design_routes(config, fields, seed_base):
    """Plan without any field values or true covariance; start is site 0.

    Fixed-assumption greedy and random are reference observation designs, not
    policies driven by the alternative mapper. Canonical identical prefixes are
    stored once so changing a cap cannot change a prediction on identical data.
    """
    config.validate()
    points, mask = geometry(config)
    prior = config.design_variance * rbf_covariance(points, config.design_length)
    episodes, states, lookup = [], [], {}
    for field in range(fields):
        for cap in config.observation_caps:
            for budget in config.travel_budgets:
                env = Config(grid_size=config.grid_size, noise_std=config.noise_std,
                    length_scale=config.design_length, travel_budget=budget, seeds=1, max_observations=cap)
                for strategy in ("random", "greedy"):
                    _, route, distances = explore(points, prior, np.zeros(len(points)), ~mask, env, strategy,
                                                   np.random.default_rng(seed_base + field))
                    state_ids = []
                    for count in range(1, len(route)+1):
                        key = (field, tuple(route[:count]))
                        if key not in lookup:
                            lookup[key] = len(states)
                            states.append({"field":field, "indices": list(key[1])})
                        state_ids.append(lookup[key])
                    episodes.append({"episode":len(episodes),"field":field,"strategy":strategy,
                        "observation_cap":int(cap),"travel_budget":float(budget),"route":route,
                        "distances":distances,"state_ids":state_ids,
                        "stop_reason":"observation_cap" if len(route) == cap else "no_affordable_unvisited_site"})
    return episodes, states


def generate_fields(config, fields, seed_base):
    """Simulator-only latent truth and potential sensor values; gradual mixture."""
    points, mask = geometry(config)
    long = rbf_covariance(points,config.long_length)
    short = rbf_covariance(points,config.short_length)
    a,b = transition_weights(points[:,1],None)
    covariance = np.outer(a,a)*long + np.outer(b,b)*short
    jitter = 1e-10*np.eye(len(points))
    long_factor,short_factor = np.linalg.cholesky(long+jitter),np.linalg.cholesky(short+jitter)
    truth, sensor = [],[]
    for field in range(fields):
        zlong,zshort,noise = np.random.default_rng(seed_base+field).standard_normal((3,len(points)))
        latent = a*(long_factor@zlong) + b*(short_factor@zshort)
        truth.append(latent)
        sensor.append(latent+config.noise_std*noise)
    return {"truth":np.asarray(truth),"potential_sensor_values":np.asarray(sensor),
            "true_covariance":covariance,"evaluation_mask":mask}


def acquired_states(config, states, sensor_values):
    """Expose only values actually acquired on each recorded route prefix."""
    size = max(len(state["indices"]) for state in states)
    indices = np.full((len(states),size),-1,dtype=int)
    values = np.full(indices.shape,np.nan)
    for row,state in enumerate(states):
        count = len(state["indices"])
        indices[row,:count] = state["indices"]
        values[row,:count] = sensor_values[state["field"],state["indices"]]
    return {"points":geometry(config)[0],"indices":indices,"values":values,
            "counts":np.asarray([len(state["indices"]) for state in states]),
            "fields":np.asarray([state["field"] for state in states])}
