"""Plug-in RBF GP: fit covariance parameters using acquired observations only."""

import time
import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

from .baseline import GPBelief, rbf_covariance


@dataclass(frozen=True)
class EstimatorConfig:
    initial_length_scale: float = 0.35
    initial_process_variance: float = 1.5
    length_scale_bounds: tuple = (0.05, 1.0)
    process_variance_bounds: tuple = (0.05, 5.0)
    min_observations: int = 4
    refit_every: int = 3
    optimizer_restarts: int = 2
    optimizer_seed: int = 731

    def validate(self):
        for name, value, bounds in (
            ("length_scale", self.initial_length_scale, self.length_scale_bounds),
            ("process_variance", self.initial_process_variance, self.process_variance_bounds),
        ):
            low, high = bounds
            if not all(np.isfinite(v) for v in (value, low, high)):
                raise ValueError(name + " values must be finite")
            if not 0 < low <= value <= high or low == high:
                raise ValueError(name + " must have positive bounds containing its initial value")
        if self.min_observations < 2 or self.refit_every < 1 or self.optimizer_restarts < 0:
            raise ValueError("Invalid fitting schedule")


class EstimatedGPBelief(GPBelief):
    """Known zero mean and sensor noise; unknown variance and length scale.

    Between scheduled maximum marginal likelihood fits, apply exact sequential
    updates. After a fit, rebuild the posterior using all observations and the
    accepted kernel. Posterior uncertainty is conditional on plug-in parameters.
    """

    def __init__(self, points, noise_std, config=None):
        self.config = config or EstimatorConfig()
        self.config.validate()
        self.points = np.array(points, copy=True)
        self.noise_std = noise_std
        self.length_scale = self.config.initial_length_scale
        self.process_variance = self.config.initial_process_variance
        self.indices = []
        self.values = []
        self.fit_history = []
        self.total_update_seconds = 0.0
        self.total_fit_seconds = 0.0
        prior = self.process_variance * rbf_covariance(self.points, self.length_scale)
        super().__init__(prior, noise_std)

    def observe(self, index, value):
        started = time.perf_counter()
        self.indices.append(index)
        self.values.append(float(value))
        count = len(self.indices)
        fit_due = count >= self.config.min_observations and (
            (count - self.config.min_observations) % self.config.refit_every == 0
        )
        if fit_due and self._fit_parameters():
            self._rebuild_posterior()
        else:
            GPBelief.observe(self, index, value)
        self.total_update_seconds += time.perf_counter() - started

    def _rebuild_posterior(self):
        prior = self.process_variance * rbf_covariance(self.points, self.length_scale)
        GPBelief.__init__(self, prior, self.noise_std)
        for index, value in zip(self.indices, self.values):
            GPBelief.observe(self, index, value)

    @classmethod
    def from_observations(cls, points, noise_std, indices, values, config=None):
        """Fit one fresh snapshot from exactly the supplied acquired data.

        Restart from the configured initial parameters, independently of smaller
        sample-size fits. Fit once when min_observations is reached, then rebuild
        the posterior on all supplied measurements. The online refit schedule in
        observe() is unchanged; this is an explicit batch comparison interface.
        """
        indices, values = np.asarray(indices), np.asarray(values, dtype=float)
        if indices.ndim != 1 or values.shape != indices.shape or not len(indices):
            raise ValueError("Supply nonempty matching observation vectors")
        if not np.issubdtype(indices.dtype, np.integer) or not np.isfinite(values).all():
            raise ValueError("Indices must be integers and observed values finite")
        if np.any(indices < 0) or np.any(indices >= len(points)) or len(set(indices)) != len(indices):
            raise ValueError("Observed indices must be unique and inside the domain")
        if not np.isfinite(noise_std) or noise_std <= 0:
            raise ValueError("Noise standard deviation must be finite and positive")
        started = time.perf_counter()
        belief = cls(points, noise_std, config)
        belief.indices = [int(index) for index in indices]
        belief.values = [float(value) for value in values]
        if len(indices) >= belief.config.min_observations:
            belief._fit_parameters()
        belief._rebuild_posterior()
        belief.total_update_seconds = time.perf_counter() - started
        return belief

    def _fit_parameters(self):
        kernel = ConstantKernel(self.process_variance, self.config.process_variance_bounds) * RBF(
            self.length_scale, self.config.length_scale_bounds
        )
        model = GaussianProcessRegressor(
            kernel=kernel, alpha=self.noise_variance, normalize_y=False,
            n_restarts_optimizer=self.config.optimizer_restarts,
            random_state=self.config.optimizer_seed + len(self.indices),
        )
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(self.points[self.indices], np.asarray(self.values))
        elapsed = time.perf_counter() - started
        self.total_fit_seconds += elapsed
        initial_lml = float(model.log_marginal_likelihood(kernel.theta))
        optimized_lml = float(model.log_marginal_likelihood_value_)
        accepted = bool(np.isfinite(optimized_lml) and optimized_lml >= initial_lml - 1e-8)
        if accepted:
            self.process_variance = float(model.kernel_.k1.constant_value)
            self.length_scale = float(model.kernel_.k2.length_scale)
        boundary_hit = any(
            np.isclose(np.log(value), np.log(bound), atol=1e-4, rtol=0)
            for value, bounds in (
                (self.length_scale, self.config.length_scale_bounds),
                (self.process_variance, self.config.process_variance_bounds),
            )
            for bound in bounds
        )
        self.fit_history.append({
            "measurements": len(self.indices), "length_scale": self.length_scale,
            "process_variance": self.process_variance, "initial_lml": initial_lml,
            "optimized_lml": optimized_lml, "accepted": accepted,
            "boundary_hit": bool(boundary_hit), "fit_seconds": elapsed,
            "warnings": " | ".join(str(item.message) for item in caught),
        })
        return accepted
