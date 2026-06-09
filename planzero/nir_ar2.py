
"""
Example inference that's fast and represents a lower bound on accuracy.
"""
import numpy as np
import jax.numpy as jnp
import jax.random as jrandom
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer import Predictive
from numpyro.contrib.control_flow import scan

from .my_functools import inference_cache
from . import nir2025
from .enums import IPCC_Sector


def ar2_scan_random_walk(scaled_ca,
                         scaled_pt,
                         sector_ghg_scale,
                         future_idx,
                         noise_ca,
                         observe_past=True,
                         n_future_timesteps=10):

    # rms for each region, nanmean over time
    pt_rms = jnp.sqrt(
        jnp.maximum(
            jnp.nanmean(scaled_pt[:, :future_idx] ** 2, axis=1),
            .05 ** 2))

    alpha_1 = .1 + 1.5 * numpyro.sample(
        "alpha_1",
        dist.Kumaraswamy(1.25, 1.25).expand((13,)))
    alpha_2 = -.5 + 1.0 * numpyro.sample(
        "alpha_2",
        dist.Kumaraswamy(1.25, 1.25).expand((13,)))
    const = 0

    # This gives the dynamic range to mu.
    # It should be smaller for some regions than others
    # Here we cheat a bit, and use data stats as a prior
    mu_sigma = .05 * pt_rms

    n_past_steps = len(scaled_ca[:future_idx])

    def transition(carry, _):
        y_prev, y_prev_prev = carry
        m_t = const + alpha_1 * y_prev + alpha_2 * y_prev_prev
        y_t = numpyro.sample("y", dist.Normal(m_t, mu_sigma))
        carry = (y_t, y_prev)
        return carry, m_t

    timesteps = jnp.arange(n_past_steps - 2 + n_future_timesteps)

    obs_valid = jnp.isfinite(scaled_pt[:, :future_idx])
    approx_obs = jnp.where(
        obs_valid,
        scaled_pt[:, :future_idx],
        jnp.nanmean(scaled_pt[:, :future_idx], axis=1, keepdims=True))

    mu_0 = numpyro.sample(
        "mu_0",
        dist.Normal(0, 1).expand((13,)),
        obs=approx_obs[:, 0])
    mu_1 = numpyro.sample(
        "mu_1",
        dist.Normal(0, 1).expand((13,)),
        obs=approx_obs[:, 1])

    init = (mu_1, mu_0)
    _, mu = scan(transition, init, timesteps)

    numpyro.deterministic("mu", mu)

    numpyro.sample("past-pt",
                   dist.Normal(mu[:n_past_steps - 2],
                               noise_ca * pt_rms).mask(obs_valid[:, 2:future_idx].T),
                   obs=approx_obs[:, 2:future_idx].T if observe_past else None)
    numpyro.sample("past-ca",
                   dist.Normal(jnp.sum(mu[:n_past_steps - 2], axis=1),
                               noise_ca),
                   obs=scaled_ca[2:future_idx] if observe_past else None)

    mu_1step = numpyro.sample(
        "mu_1step",
        dist.Normal(const
                    + alpha_1 * mu[1:future_idx - 1]
                    + alpha_2 * mu[:future_idx - 2],
                    mu_sigma))
    numpyro.sample("past-pt-1",
                   dist.Normal(mu_1step, noise_ca * pt_rms))
    numpyro.sample("past-ca-1",
                   dist.Normal(jnp.sum(mu_1step, axis=1), noise_ca))

    if n_future_timesteps:
        numpyro.sample("future-pt",
                       dist.Normal(mu[n_past_steps - 2:],
                                   noise_ca * pt_rms))
        numpyro.sample("future-ca",
                       dist.Normal(jnp.sum(mu, axis=1)[n_past_steps - 2:],
                                   noise_ca))


class NIR2025_AR2(object):

    def __init__(self, sector, ghg, seed=0):
        self.sector = sector
        self.ghg = ghg
        arr_pt, arr_ca = nir2025.ktCO2e_dense_w_nan()
        self.jnp_pt = jnp.array(arr_pt[nir2025.idx_of_sector[sector],
                                       nir2025.idx_of_ghg[ghg]],
                                dtype='float64')
        self.jnp_ca = jnp.array(arr_ca[nir2025.idx_of_sector[sector],
                                       nir2025.idx_of_ghg[ghg]],
                                dtype='float64')

        self.scale = max([
            np.sqrt(np.mean(self.jnp_ca ** 2)),
            np.sqrt(np.nanmean(np.nansum(self.jnp_pt ** 2, axis=0))),
            1.0,
        ])

        assert np.isfinite(self.scale)
        self.scaled_ca = self.jnp_ca / self.scale
        self.scaled_pt = self.jnp_pt / self.scale
        self.post_samples = None
        self.rng_key = jrandom.key(seed)

    @property
    def noise_ca(self):
        if self.sector == IPCC_Sector.Harvested_Wood_Products:
            return 0.05
        else:
            return 0.015

    @inference_cache()
    @staticmethod
    def posterior_inference(sector, ghg, version):
        self = NIR2025_AR2(sector, ghg)

        # Start from this source of randomness. We will split keys for subsequent operations.
        self.rng_key, rng_key_ = jrandom.split(self.rng_key)

        # Run NUTS.
        mcmc = MCMC(NUTS(ar2_scan_random_walk),
                    num_warmup=500,
                    num_samples=1000)
        mcmc.run(rng_key_,
                 scaled_ca=self.scaled_ca,
                 scaled_pt=self.scaled_pt,
                 sector_ghg_scale=self.scale,
                 future_idx=len(self.scaled_ca), # XXX
                 noise_ca=self.noise_ca,
                )
        #mcmc.print_summary()
        self.mcmc = mcmc
        self.post_samples = mcmc.get_samples()
        return self

    def reconstructed_past(self):
        predictive = Predictive(
            ar2_scan_random_walk,
            self.post_samples,
            return_sites=['past-pt-1', 'past-ca-1',
                          'past-pt-2', 'past-ca-2',],
            )
        self.rng_key, rng_key_ = jrandom.split(self.rng_key)
        predictions = predictive(
            rng_key_,
            scaled_ca=self.scaled_ca,
            scaled_pt=self.scaled_pt,
            sector_ghg_scale=self.scale,
            future_idx=len(self.scaled_ca),
            noise_ca=self.noise_ca,
            observe_past=False,
            )
        return predictions

    def future_predictions(self):
        predictive = Predictive(
            ar2_scan_random_walk,
            self.post_samples,
            return_sites=['obs_pt', 'obs_ca'],
            )
        self.rng_key, rng_key_ = jrandom.split(self.rng_key)
        predictions = predictive(
            rng_key_,
            scaled_ca=self.scaled_ca,
            scaled_pt=self.scaled_pt,
            sector_ghg_scale=self.scale,
            future_idx=len(self.scaled_ca),
            noise_ca=self.noise_ca,
            )
        return predictions

# XXX test rhat distribution
