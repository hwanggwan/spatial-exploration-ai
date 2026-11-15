from functools import partial
import importlib.util
import unittest

import numpy as np

from threadpoolctl import threadpool_limits

from spatial_exploration.baseline import GPBelief
from spatial_exploration.rollout import plan_rollout
from spatial_exploration.sensing_world import (
    Belief, choose_rollout_site, choose_site, expected_terminal_variance, fit_kernel_weights, rollout_samples,
)


class SensingWorldTests(unittest.TestCase):
    def test_single_mode_matches_gp_and_branches_leave_current_belief_unchanged(self):
        covariance = np.array([[1., .4, .1], [.4, 1., .3], [.1, .3, 1.]])
        prior = Belief.prior(covariance[None], .12)
        conditioned = prior.observe(2, 1.3).observe(0, -.4)
        baseline = GPBelief(covariance, .12)
        baseline.observe(2, 1.3)
        baseline.observe(0, -.4)
        mean, cov = conditioned.moments()
        np.testing.assert_allclose(mean, baseline.mean, atol=1e-12)
        np.testing.assert_allclose(cov, baseline.covariance, atol=1e-12)
        np.testing.assert_array_equal(prior.covariances[0], covariance)
        np.testing.assert_array_equal(prior.means, 0)

    def test_distinct_outcomes_change_mixture_predictions(self):
        covariance = np.array([[[1., .1], [.1, 1.]], [[4., 1.], [1., 1.]]])
        prior = Belief.prior(covariance, .12)
        low = prior.observe(0, 0.)
        high = prior.observe(0, 4.)
        self.assertLess(low.log_weights[1], high.log_weights[1])
        self.assertNotAlmostEqual(low.moments()[1][1, 1], high.moments()[1][1, 1])
        values, weights = prior.outcomes(0)
        mean, std = prior.reading_moments(0)
        self.assertAlmostEqual(weights @ values, mean)
        self.assertAlmostEqual(weights @ (values - mean)**2, std**2)

    def test_plan_rejects_heldout_and_infeasible_sites(self):
        prior = Belief.prior(np.eye(4)[None], .12)
        hops = np.abs(np.arange(4)[:, None] - np.arange(4))
        mask = np.array([True, False, True, True])
        self.assertIsNone(choose_site(prior, hops, mask, [0], 1, 2))
        self.assertIsNone(choose_site(prior, hops, mask, [0], 3, 0))
        self.assertEqual(choose_site(prior, hops, mask, [0], 2, 2), 2)

    def test_rollout_score_matches_exact_gaussian_and_reuses_the_gp_policy(self):
        sites = np.arange(6)
        hops = np.abs(sites[:, None] - sites)
        covariance = np.exp(-hops / 1.5)
        belief = Belief.prior(covariance[None], .12).observe(2, .6)
        innovations = rollout_samples(3)
        route = (3, 5)
        posterior = belief.observe(3, 2.).observe(5, -1.)
        np.testing.assert_allclose(expected_terminal_variance(belief, route, innovations),
                                   np.diag(posterior.moments()[1]).mean(), atol=1e-12)
        allowed = sites != 4
        gp = plan_rollout(belief.moments()[1], hops, 2, sites == 2, allowed, 4, .12**2,
                          remaining_observations=2)
        learned = choose_rollout_site(belief, hops, allowed, [2], 4, 2, innovations=innovations)
        self.assertEqual(learned, gp.first)
        self.assertIsNone(choose_rollout_site(belief, hops, allowed, [2], 0, 2, innovations=innovations))

    def test_rollout_coverage_breaks_equal_variance_ties_with_walking_distance(self):
        positions = np.arange(5)
        hops = np.abs(positions[:, None] - positions)
        belief = Belief.prior(np.eye(5)[None], .12).observe(0, 0.)
        allowed = np.ones(5, dtype=bool)
        innovations = rollout_samples(1)
        self.assertEqual(choose_rollout_site(belief, hops, allowed, [0], 4, 1, innovations=innovations), 1)
        self.assertEqual(choose_rollout_site(belief, hops, allowed, [0], 4, 1, innovations=innovations,
                                             coverage_weight=.025), 3)
        allowed[3] = False
        self.assertEqual(choose_rollout_site(belief, hops, allowed, [0], 2, 1, innovations=innovations,
                                             coverage_weight=.025), 2)

    def test_mixture_terminal_score_agrees_with_integrated_posterior_variance(self):
        covariance = np.array([[[1., .6], [.6, 1.]], [[1., .1], [.1, 1.]]])
        belief = Belief.prior(covariance, .12)
        innovations = rollout_samples(1)
        outcomes = np.sqrt(1 + .12**2) * innovations[:, 0]
        direct = np.mean([np.diag(belief.observe(0, y).moments()[1]).mean() for y in outcomes])
        terminal = expected_terminal_variance(belief, (0,), innovations)
        self.assertAlmostEqual(terminal, direct, places=12)

    def test_fit_recovers_nonnegative_noise_adjusted_kernel_weights(self):
        basis = np.array([np.eye(2), [[1., .5], [.5, 1.]]])
        weights = np.array([.2, .8])
        noise = .12
        covariance = np.einsum("k,kij->ij", weights, basis) + noise**2 * np.eye(2)
        samples = np.sqrt(2) * np.linalg.cholesky(covariance).T
        np.testing.assert_allclose(fit_kernel_weights([(basis, [samples])], noise), weights[None])


@unittest.skipUnless(importlib.util.find_spec("torch"), "Study fixture requires PyTorch")
class SensingWorldRouteTests(unittest.TestCase):
    def test_changing_unmeasured_values_cannot_change_closed_loop(self):
        from spatial_exploration.sensing_world_study import design_for, run_episode

        design = design_for((2, 9))
        prior = Belief.prior(np.stack((np.eye(144), .5 * np.eye(144))), .12)
        sensor = np.linspace(-1, 1, 144)
        calls = []

        def read(site):
            calls.append(site)
            return sensor[site]

        planners = (choose_site, partial(choose_rollout_site, innovations=rollout_samples(16)))
        with threadpool_limits(limits=1):
            for planner in planners:
                with self.subTest(planner=planner):
                    calls.clear()
                    result = run_episode(prior, design, 13, 6, read, planner=planner)
                    sites, walk, _, _, _, forecasts, _ = result
                    self.assertEqual(calls, sites)
                    self.assertEqual(len(forecasts), len(sites) - 1)
                    changed = sensor.copy()
                    changed[np.setdiff1d(np.arange(144), sites)] += 1000
                    other = run_episode(prior, design, 13, 6, changed.__getitem__, planner=planner)
                    self.assertEqual((sites, walk), other[:2])
                    np.testing.assert_allclose(forecasts, other[5])
