
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
                         obs_sigma_sq,
                         observe_past=True,
                         n_future_timesteps=10):
    alpha_1 = numpyro.sample("alpha_1", dist.Normal(1, 1).expand((13,)))
    alpha_2 = numpyro.sample("alpha_2", dist.Normal(0, 1).expand((13,)))
    const = numpyro.sample("const", dist.Normal(0, 1).expand((13,)))
    const = 0
    sigma_sq = numpyro.sample("sigma_sq", dist.LogNormal(-1, 1))
    sigma_sq = .1 ** 2

    n_past_steps = len(scaled_ca[:future_idx])

    # rms for each province, nanmean over time
    pt_rms = jnp.maximum(
        jnp.nanmean(scaled_pt[:, :future_idx] ** 2, axis=1),
        .1 ** 2)

    def transition(carry, _):
        y_prev, y_prev_prev = carry
        m_t = const + alpha_1 * y_prev + alpha_2 * y_prev_prev
        y_t = numpyro.sample("y", dist.Normal(m_t, sigma_sq))
        carry = (y_t, y_prev)
        return carry, m_t

    timesteps = jnp.arange(n_past_steps - 2 + n_future_timesteps)

    obs_valid = jnp.isfinite(scaled_pt[:, :future_idx])
    #obs_sigma_sq = 0.1 ** 2
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

    #pt_sigma = jnp.where(
    #    jnp.isfinite(scaled_pt[:, :]), obs_sigma_sq,
    #    jnp.maximum(pt_scale[:, None], 0.01))
    pt_sigma = jnp.zeros_like(scaled_pt) + obs_sigma_sq

    numpyro.sample("past-pt",
                   dist.Normal(mu[:n_past_steps - 2],
                               pt_sigma[:, 2:future_idx].T).mask(obs_valid[:, 2:future_idx].T),
                   obs=approx_obs[:, 2:future_idx].T if observe_past else None)
    numpyro.sample("past-ca",
                   dist.Normal(jnp.sum(mu[:n_past_steps - 2], axis=1),
                               obs_sigma_sq),
                   obs=scaled_ca[2:future_idx] if observe_past else None)

    mu_1step = numpyro.sample(
        "mu_1step",
        dist.Normal(const
                    + alpha_1 * mu[1:future_idx - 1]
                    + alpha_2 * mu[:future_idx - 2],
                    sigma_sq))
    numpyro.sample("past-pt-1",
                   dist.Normal(mu_1step, obs_sigma_sq))
    numpyro.sample("past-ca-1",
                   dist.Normal(jnp.sum(mu_1step, axis=1), obs_sigma_sq))

    mu_2step = numpyro.sample(
        "mu_2step",
        dist.Normal(const
                    + alpha_1 * mu_1step
                    + alpha_2 * mu[1:future_idx - 1],
                    sigma_sq))
    numpyro.sample("past-pt-2",
                   dist.Normal(mu_2step, obs_sigma_sq))
    numpyro.sample("past-ca-2",
                   dist.Normal(jnp.sum(mu_2step, axis=1), obs_sigma_sq))

    if n_future_timesteps:
        numpyro.sample("future-pt",
                       dist.Normal(mu[n_past_steps - 2:],
                                   obs_sigma_sq))
        numpyro.sample("future-ca",
                       dist.Normal(jnp.sum(mu, axis=1)[n_past_steps - 2:],
                                   obs_sigma_sq))

    #numpyro.sample("forecast", dist.Normal(mu, sigma), sample_shape=(n_forecast,))
#ar2_scan_random_walk(do_sample=False)

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
    def obs_sigma_sq(self):
        if self.sector == IPCC_Sector.Harvested_Wood_Products:
            return 0.25 ** 2
        else:
            return 0.1 ** 2

    @inference_cache(recompute=True)
    @staticmethod
    def posterior_inference(sector, ghg):
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
                 obs_sigma_sq=self.obs_sigma_sq,
                )
        #mcmc.print_summary()
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
            obs_sigma_sq=self.obs_sigma_sq,
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
            obs_sigma_sq=self.obs_sigma_sq,
            )
        return predictions

# XXX test rhat distribution
