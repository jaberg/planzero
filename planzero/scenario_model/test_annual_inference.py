"""
Test that various synthetic timeseries that should be
well-approximated by low-order auto-regressive processes,
are, in fact, learned by auto-regressive processes implemented
using model_element_annual_step, annual variables etc. as implemented
in this scenario_model sub-package.
"""

import unittest

import pytest
from jax import numpy as jnp
from numpyro import distributions as dist

from . import (
    ModelElement,
    Phase,
    ScenarioModel,
    Workspace,
    access_annual,
    access_val,
    compute_model,
    define,
    define_annual,
    define_carry,
)
from .base import GroupedPosterior


class AR2_ModelElement(ModelElement):

    def __init__(self, data):
        super().__init__()
        self.data = data

    def _phi(self, ws):
        margin = 0.01
        high = 1 - margin
        low = -1 + margin
        # k: scaled, unconstrained
        k1 = (high - low) * ws.val.general['alpha_1'] + low
        k2 = (high - low) * ws.val.general['alpha_2'] + low
        # phi: constrained to non-divergent dynamics
        phi_2 = k2
        phi_1 = k1 * (1 - k2)
        return phi_1, phi_2

    @define('alpha_1', prior_shape=[])
    @define('alpha_2', prior_shape=[])
    @define('z_1', prior_shape=[])
    @define('z_2', prior_shape=[])
    @define('AR_sigma_raw', prior_shape=[])
    @define('AR_sigma', prior_shape=[])
    @define_carry('x_prev', shape=[])
    @define_carry('x_prev_prev', shape=[])
    def model_element_prepare(self, ws:Workspace):
        # these are broad, almost-uniform distributions that drop near the edges
        ws.dist.general['alpha_1'] = dist.Kumaraswamy(1.25, 1.25)
        ws.dist.general['alpha_2'] = dist.Kumaraswamy(1.25, 1.25)

        # this parameter may be problematic: a large pt_sigma
        # can cause AR_sigma's effect to collapse 
        ws.dist.general['AR_sigma_raw'] = dist.LogNormal(-5, 0.7)
        ws.val.general['AR_sigma'] = ws.val.general['AR_sigma_raw']

        phi_1, phi_2 = self._phi(ws)
        # initialize x_prev and x_prev_prev from their exact bivariate stationary
        # distribution induced by dynamics phi. The following formulas are
        # supposed to be doing ancestral sampling using the Cholesky decomposition
        # of the dynamics.
        rho1 = phi_1 / (1 - phi_2)
        gamma0_top = (1 - phi_2) * ws.val.general['AR_sigma'] ** 2
        gamma0_bot = (1 + phi_2) * ((1 - phi_2) ** 2 - phi_1 ** 2)
        gamma0 = gamma0_top / gamma0_bot

        ws.dist.general['z_1'] = dist.Normal(0, ws.val.general['AR_sigma'])
        z_1 = ws.val.general['z_1']
        ws.val.initial_carry['x_prev_prev'] = z_1 * jnp.sqrt(gamma0)

        ws.dist.general['z_2'] = dist.Normal(0, ws.val.general['AR_sigma'])
        z_2 = ws.val.general['z_2']
        ws.val.initial_carry['x_prev'] = jnp.sqrt(gamma0) * (
                rho1 * z_1 + jnp.sqrt(1 - rho1 ** 2) * z_2)

    @access_val('alpha_1')
    @access_val('alpha_2')
    @access_val('AR_sigma')
    @define_annual('z_t', annual_shape=[])
    @define_annual('AR_mean', annual_shape=[])
    def model_element_annual_step(self, ws:Workspace):
        phi_1, phi_2 = self._phi(ws)

        # z: zero-mean noise
        ws.dist.this_annual['z_t'] = dist.Normal(0, ws.val.general['AR_sigma'])

        # this val accesses MCMC samples and ignores dist
        # in Posterior phase until scan_step_ii >= len(self.data)
        z_t = ws.val.this_annual['z_t']

        # x: zero-mean dynamic recursion (aka AR_mean)
        ws.val.this_annual['AR_mean'] = (
                phi_1 * ws.val.this_carry['x_prev']
                + phi_2 * ws.val.this_carry['x_prev_prev']
                + z_t)
        ws.val.next_carry['x_prev'] = ws.val.this_annual['AR_mean']
        ws.val.next_carry['x_prev_prev'] = ws.val.this_carry['x_prev']

    @access_val('AR_sigma')
    @access_annual('AR_mean')
    @define('mu', prior_shape=[])
    #@define('sigma_pt', prior_shape=[])
    @define_annual('output', annual_shape=[])
    def model_element_postprocess(self, ws:Workspace):
        ws.dist.general['mu'] = dist.Normal()

        output_loc = (ws.val.general['mu'] + ws.val.annual['AR_mean'])

        if ws.phase == Phase.Prior:
            ws.obs_val.annual['output'] = self.data
        ws.dist.annual['output'] = dist.Normal(
                output_loc,
                (self.data.max() - self.data.min()) / 100)


class AR2_Alternating(unittest.TestCase):
    """
    Test that several alternating sequences of different means and amplitudes
    can be learned (correctly extended) with a common AR2 prior.
    """

    def assert_annual_post_matches_inf(self, comp, model, elem, key_name):

        post = comp.get_ndarray(
                elem.annual_key(key_name),
                sample_if_necessary=True)
        inf = comp.get_ndarray(
                GroupedPosterior(prior_var_key=elem.annual_key(key_name)),
                sample_if_necessary=True)

        (n_groups, n_saved_samples, n_prior_years, *annual_shape) \
                = inf.shape
        assert n_prior_years == model.mv.n_prior_years
        # reshape to combine groups and saved samples as if single chain mcmc
        reshaped = inf.reshape(
                [n_groups * n_saved_samples, n_prior_years]
                + annual_shape)
        # transpose mcmc dimension to the end
        transposed = jnp.moveaxis(reshaped, 0, -1)

        # post-conditions:
        assert transposed.shape[0] == n_prior_years
        assert transposed.shape[-1] == comp.n_mcmc
        assert transposed.shape[1:-1] == tuple(annual_shape)

        assert jnp.all(
                post[:n_prior_years]
                == transposed)

    def setUp(self):
        self.num_warmup = 200

    def tearDown(self):
        n_data_years = 16
        data = jnp.array(
                    [float(self.low if ii % 2 == 0 else self.high)
                     for ii in range(n_data_years)])
        model = ScenarioModel(
                first_year=2000,
                n_prior_years=n_data_years,
                n_posterior_years = n_data_years + 2,
                num_warmup=self.num_warmup,
                num_samples=100)
        elem = AR2_ModelElement(data=data)
        model.add_element(elem)
        comp = compute_model(model, seed_or_key=1)

        self.assert_annual_post_matches_inf(comp, model, elem, 'z_t')

        output = comp.get_ndarray(
                elem.annual_key('output'),
                sample_if_necessary=True)

        assert n_data_years % 2 == 0

        q95_low = jnp.quantile(abs(output[n_data_years] - self.low), .95)
        assert q95_low < self.eps

        q95_high = jnp.quantile(abs(output[n_data_years + 1] - self.high), .95)
        assert q95_high < self.eps

    @property
    def eps(self):
        try:
            return self._eps
        except AttributeError:
            return (self.high - self.low) / 10 * 1.5

    def test_neg(self):
        self.low = -5
        self.high = -4

    def test_pos(self):
        self.low = 4
        self.high = 5

    def test_big(self):
        self.low = -50
        self.high = 40

    def test_small(self):
        self.low = .4
        self.high = .5
        # model is robust, but not invariant to scale
        self._eps = .05
        self.num_warmup = 500

    def test_small_ish(self):
        self.low = .3
        self.high = .6
        self._eps = .12
        self.num_warmup = 500
