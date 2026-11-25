"""
Test that various synthetic timeseries that should be
well-approximated by low-order auto-regressive processes,
are, in fact, learned by auto-regressive processes implemented
using model_element_annual_step, annual variables etc. as implemented
in this scenario_model sub-package.
"""

import unittest

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


class AR2_ModelElement(ModelElement):

    def __init__(self, data):
        super().__init__()
        self.data = data

    def _phi(self, ws):
        # k: scaled, unconstrained
        k1 = 2 * ws.val.general['alpha_1'] - 1  # over covers range (-1, 1)
        k2 = 2 * ws.val.general['alpha_2'] - 1  # over covers range (-1, 1)
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
        ws.dist.general['AR_sigma_raw'] = dist.LogNormal(-1, 0.7)
        ws.val.general['AR_sigma'] = ws.val.general['AR_sigma_raw'] / 10

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
    @define('sigma_pt', prior_shape=[])
    @define_annual('output', annual_shape=[])
    def model_element_postprocess(self, ws:Workspace):
        ws.dist.general['mu'] = dist.Normal()
        ws.dist.general['sigma_pt'] = dist.LogNormal(-1, 0.7)

        output_loc = (ws.val.general['mu']
                      + ws.val.annual['AR_mean'])
        output_scale = jnp.sqrt(ws.val.general['sigma_pt'] ** 2
                                + ws.val.general['AR_sigma'] ** 2)

        if ws.phase == Phase.Prior:
            ws.obs_val.annual['output'] = self.data
        ws.dist.annual['output'] = dist.Normal(output_loc, output_scale)


class AR2_Alternating(unittest.TestCase):
    """
    Test that several alternating sequences of different means and amplitudes
    can be learned (correctly extended) with a common AR2 prior.
    """

    def tearDown(self):
        n_data_years = 100
        data = jnp.array(
                    [float(self.low if ii % 2 == 0 else self.high)
                     for ii in range(n_data_years)])
        model = ScenarioModel(
                first_year=2000,
                n_prior_years=n_data_years,
                n_posterior_years = n_data_years + 2,
                num_warmup=300,
                num_samples=100)
        elem = AR2_ModelElement(data=data)
        model.add_element(elem)
        comp = compute_model(model, seed_or_key=1)
        out = comp.get_ndarray(
                elem.annual_key('output'),
                sample_if_necessary=True)
        print(out)
        assert n_data_years % 2 == 0
        eps = (self.high - self.low) / 10 * 1.5

        q95_low = jnp.quantile(abs(out[n_data_years] - self.low), .95)
        assert q95_low < eps

        q95_high = jnp.quantile(abs(out[n_data_years + 1] - self.high), .95)
        assert q95_high < eps

    def test_neg(self):
        self.low = -5
        self.high = -4

    def test_pos(self):
        self.low = 5
        self.high = 4

    def test_big(self):
        self.low = -50
        self.high = 40

    def test_small(self):
        self.low = .5
        self.high = .4
