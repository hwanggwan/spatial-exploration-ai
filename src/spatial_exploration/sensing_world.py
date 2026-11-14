"""A learned, geometry-conditioned Gaussian mixture for active sensing."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import nnls
from scipy.special import logsumexp, ndtri
from scipy.stats import qmc

from .barrier_gp import graph_spectrum, unit_covariance
from .lookahead import mean_variance_reduction
from .rollout import plan_rollout


# Mix resolvable spatial scales on graphs with different wall conductance.
CONDUCTANCES = (0., .2, 1.)
HEAT_LENGTHS = (.08, .17, .32, .58)
NODES = np.array([-np.sqrt(3), 0., np.sqrt(3)])
NODE_WEIGHTS = np.array([1., 4., 1.]) / 6


def kernel_basis(layout):
    terms = [np.eye(len(layout.points))]
    for conductance in CONDUCTANCES:
        eigenvalues, eigenvectors = graph_spectrum(layout, conductance)
        for length in HEAT_LENGTHS:
            spectrum = np.exp(-.5 * (length * (layout.size - 1))**2 * eigenvalues)
            terms.append(unit_covariance(eigenvectors, spectrum))
    return np.stack(terms)


def fit_kernel_weights(training, noise_std):
    """Fit PSD kernels to simulated training observations grouped by field type."""
    designs, targets = [], []
    for basis, groups in training:
        n = len(basis[0])
        i, j = np.triu_indices(n)
        designs.append(basis[:, i, j].T)
        moments = np.stack([samples.T @ samples / len(samples) for samples in groups])
        moments -= noise_std**2 * np.eye(n)
        targets.append(moments[:, i, j].T)
    design = np.concatenate(designs)
    return np.stack([nnls(design, target)[0] for target in np.concatenate(targets).T])


@dataclass(frozen=True)
class Belief:
    """Posterior over a latent spatial field, with one Gaussian per hidden mode."""

    means: np.ndarray
    covariances: np.ndarray
    log_weights: np.ndarray
    noise_variance: float

    @classmethod
    def prior(cls, covariances, noise_std):
        modes, sites, _ = covariances.shape
        return cls(np.zeros((modes, sites)), covariances, np.full(modes, -np.log(modes)), noise_std**2)

    def moments(self):
        weights = np.exp(self.log_weights)
        mean = weights @ self.means
        centered = self.means - mean
        covariance = np.einsum("m,mij->ij", weights, self.covariances)
        covariance += (centered.T * weights) @ centered
        return mean, covariance

    def branches(self, site, readings):
        """Posterior mode means and weights for each possible reading at a site."""
        readings = np.atleast_1d(readings)
        column = self.covariances[:, :, site]
        variance = column[:, site] + self.noise_variance
        residual = readings[:, None] - self.means[:, site]
        log_joint = self.log_weights - .5 * (np.log(2 * np.pi * variance) + residual**2 / variance)
        log_p = logsumexp(log_joint, axis=1)
        means = self.means + (residual / variance)[:, :, None] * column
        covariances = self.covariances - np.einsum("mi,mj,m->mij", column, column, 1 / variance)
        return means, covariances, log_joint - log_p[:, None], log_p

    def observe(self, site, reading):
        means, covariances, weights, _ = self.branches(site, reading)
        return Belief(means[0], covariances, weights[0], self.noise_variance)

    def reading_moments(self, site):
        weights = np.exp(self.log_weights)
        mean = weights @ self.means[:, site]
        variance = weights @ (self.covariances[:, site, site] + (self.means[:, site] - mean)**2)
        return float(mean), float(np.sqrt(variance + self.noise_variance))

    def outcomes(self, site):
        if len(self.means) == 1:
            # Gaussian covariance updates are independent of the reading.
            return self.means[:, site], np.ones(1)
        means = self.means[:, site]
        std = np.sqrt(self.covariances[:, site, site] + self.noise_variance)
        return (means[:, None] + std[:, None] * NODES).ravel(), (
            np.exp(self.log_weights)[:, None] * NODE_WEIGHTS).ravel()


def choose_site(belief, hops, allowed, visited, remaining, remaining_readings, shortlist=5):
    """Plan two measurements, adapting the second stop to possible first readings."""
    current = visited[-1]
    available = allowed.copy()
    available[visited] = False
    candidates = np.flatnonzero(available & (hops[current] <= remaining))
    if not len(candidates) or not remaining_readings:
        return None
    _, covariance = belief.moments()
    prior = np.diag(covariance).mean()
    immediate = mean_variance_reduction(covariance, belief.noise_variance, candidates) / hops[current, candidates]
    finalists = candidates[np.argsort(-immediate, kind="stable")[:shortlist]]
    best_site, best_score = None, -np.inf
    for site in finalists:
        cost = hops[current, site]
        second = np.flatnonzero(available & (hops[site] <= remaining - cost) & (hops[site] > 0))
        outcomes, probabilities = belief.outcomes(site)
        means, covariances, weights, _ = belief.branches(site, outcomes)
        weights = np.exp(weights)
        centered = means - np.einsum("bm,bmn->bn", weights, means)[:, None, :]
        within = np.einsum("bm,mn->bn", weights, np.diagonal(covariances, axis1=1, axis2=2))
        variance = within + np.einsum("bm,bmn->bn", weights, centered**2)
        reduction = prior - variance.mean(axis=1)
        if remaining_readings > 1 and len(second):
            # Match the continuation covariance to its first two moments.
            cross = np.einsum("bm,mnc->bnc", weights, covariances[:, :, second])
            cross += np.einsum("bm,bmn,bmc->bnc", weights, centered, centered[:, :, second])
            gain = np.mean(cross**2, axis=1) / (variance[:, second] + belief.noise_variance)
            scores = ((reduction[:, None] + gain) / (cost + hops[site, second])).max(axis=1)
        else:
            scores = reduction / cost
        score = float(probabilities @ scores)
        if score > best_score:
            best_site, best_score = int(site), score
    return best_site


def rollout_samples(max_readings, *, seed=283001):
    """Common scrambled Sobol innovations for all feasible suffixes."""
    return ndtri(qmc.Sobol(d=max_readings, scramble=True, seed=seed).random_base2(5))


def expected_terminal_variance(belief, route, innovations):
    """Analytic within-mode and sampled between-mode variance after a suffix."""
    sites = np.asarray(route)
    means = belief.means[:, sites]
    cross = belief.covariances[:, sites, :]
    noisy = cross[:, :, sites] + belief.noise_variance * np.eye(len(sites))
    factors = np.linalg.cholesky(noisy)
    weights = np.exp(belief.log_weights)
    basis = np.stack([solve_triangular(factor, k, lower=True, check_finite=False)
                      for factor, k in zip(factors, cross)])
    within = np.diagonal(belief.covariances, axis1=1, axis2=2) - np.sum(basis**2, axis=1)
    analytic = weights @ within.mean(axis=1)
    if len(weights) == 1:
        return float(analytic)

    # Draw a block of correlated noisy readings from each mode of the belief.
    z = innovations[:, :len(sites)]
    draws = (means[:, None, :] + z @ factors.transpose(0, 2, 1)).reshape(-1, len(sites))
    whitened = np.stack([solve_triangular(factor, (draws - m).T, lower=True, check_finite=False).T
                         for factor, m in zip(factors, means)])
    log_joint = (belief.log_weights[:, None] - np.log(np.diagonal(factors, axis1=1, axis2=2)).sum(axis=1)[:, None]
                 - .5 * np.sum(whitened**2, axis=-1))
    posterior = np.exp(log_joint - logsumexp(log_joint, axis=0))
    updated = belief.means[:, None, :] + whitened @ basis
    centered = updated - np.einsum("mb,mbn->bn", posterior, updated)
    between = np.einsum("mb,mbn->b", posterior, centered**2) / updated.shape[-1]
    return float(analytic + weights @ between.reshape(len(weights), len(z)).mean(axis=1))


def choose_rollout_site(belief, hops, allowed, visited, remaining, remaining_readings, *, innovations):
    """Roll out the moment-matched GP, reranking suffixes under the mixture."""
    current = visited[-1]
    measured = np.zeros(len(allowed), dtype=bool)
    measured[visited] = True
    if not remaining_readings or not np.any(allowed & ~measured & (hops[current] <= remaining)):
        return None
    _, covariance = belief.moments()
    plan = plan_rollout(covariance, hops, current, measured, allowed, remaining, belief.noise_variance,
        remaining_observations=remaining_readings,
        terminal_variance=lambda route: expected_terminal_variance(belief, route, innovations))
    return plan.first
