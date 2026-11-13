"""Known-noise Bayesian linear control on the neural mapper's spatial basis.

The induced kernel is a finite-feature GP, not a covariance-free method.
Posterior covariance is exact conditional on this assumed, frozen prior; that
does not guarantee calibration under the simulator's different covariance.
"""

import numpy as np

from .baseline import GPBelief


class BayesianBasisModel:
    def __init__(self, features, noise_std, prior_variance, *, intercept_scale=1.0):
        features = np.asarray(features, dtype=float)
        if features.ndim != 2 or min(features.shape) < 1 or not np.isfinite(features).all():
            raise ValueError("Supply a finite nonempty feature matrix")
        if any(not np.isfinite(x) or x <= 0 for x in (noise_std, prior_variance)):
            raise ValueError("Noise and prior variance must be positive and finite")
        if not np.isfinite(intercept_scale) or intercept_scale < 0:
            raise ValueError("Intercept scale must be finite and nonnegative")
        # Standardized basis columns alone force a zero grid-average function.
        # Include a Bayesian intercept, analogous to the neural network biases.
        self.features = np.column_stack((features, np.full(len(features), intercept_scale))) / np.sqrt(features.shape[1] + 1)
        self.prior_variance = float(prior_variance)
        self.noise_std = float(noise_std)
        self.prior_covariance = self.prior_variance * (self.features @ self.features.T)

    def condition(self, indices, values):
        indices, values = np.asarray(indices), np.asarray(values, dtype=float)
        if indices.ndim != 1 or values.shape != indices.shape or not np.isfinite(values).all():
            raise ValueError("Supply matching observation vectors with finite values")
        if indices.size and (not np.issubdtype(indices.dtype, np.integer)
                or np.any(indices < 0) or np.any(indices >= len(self.features))
                or len(set(indices.tolist())) != len(indices)):
            raise ValueError("Use unique integer sites within the domain")
        belief = GPBelief(self.prior_covariance, self.noise_std)
        for index, value in zip(indices, values):
            belief.observe(int(index), float(value))
        return belief
