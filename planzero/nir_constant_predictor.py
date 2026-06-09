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

from .my_functools import inference_cache
from . import nir2025


def constant_model(scaled_pt=None, scaled_ca=None):
    n_regions = 13
    mu = numpyro.sample("mu", dist.Normal(0.0, 1.0).expand((n_regions,)))
    sigma_pt = numpyro.sample("sigma_pt",
                              dist.LogNormal(-1.0, 0.7).expand((n_regions,)))
    sigma_ca = numpyro.sample("sigma_ca",
                              dist.LogNormal(-1.0, 0.7))
    if scaled_pt is None:
        dist_pt = dist.Normal(mu, sigma_pt)
        obs_pt = None
    else:
        valid_pt = jnp.isfinite(scaled_pt)
        dist_pt = dist.Normal(mu, sigma_pt).mask(valid_pt)
        obs_pt = jnp.where(valid_pt, scaled_pt, 0)
    numpyro.sample("obs_pt", dist_pt, obs=obs_pt)

    numpyro.sample("obs_ca",
                   dist.Normal(jnp.sum(mu), sigma_ca),
                   obs=scaled_ca)
    #numpyro.sample("forecast", dist.Normal(mu, sigma), sample_shape=(n_forecast,))


class NIR2025_Model(object):

    def __init__(self, sector, ghg, seed=0):
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
        self.scaled_ca = jnp.array(self.jnp_ca / self.scale)
        self.scaled_pt = jnp.array(self.jnp_pt / self.scale)
        self.post_samples = None
        self.rng_key = jrandom.key(seed)

    @inference_cache
    @staticmethod
    def posterior_inference(sector, ghg, seed=0,
                            num_warmup=250,
                            num_samples=1000,
                            print_summary=False):

        self = NIR2025_Model(sector, ghg, seed=seed)

        mcmc = MCMC(NUTS(constant_model),
                    num_warmup=250,
                    num_samples=num_samples)

        self.rng_key, rng_key_ = jrandom.split(self.rng_key)
        mcmc.run(rng_key_,
                 scaled_pt=self.scaled_pt.T,
                 scaled_ca=self.scaled_ca)
        if print_summary:
            mcmc.print_summary()
        self.post_samples = mcmc.get_samples()
        return self

    def predictions(self):
        predictive = Predictive(
            constant_model,
            self.post_samples,
            return_sites=['obs_pt', 'obs_ca'],
            )
        self.rng_key, rng_key_ = jrandom.split(self.rng_key)
        predictions = predictive(rng_key_)
        return predictions


def plot_regression(x, y_mean, y_hpdi):
    # Sort values for plotting by x axis
    #idx = jnp.argsort(x)
    #marriage = x[idx]
    #mean = y_mean[idx]
    #hpdi = y_hpdi[:, idx]
    #divorce = dset.DivorceScaled.values[idx]

    # Plot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    ax.plot(x, uncenter(np.ones(len(x)) * y_mean))
    ax.plot(x, uncenter(centred_ca), "o")
    ax.fill_between(x, uncenter(np.ones(len(x)) * y_hpdi[0]), uncenter(np.ones(len(x)) * y_hpdi[1]), alpha=0.3, interpolate=True)
    return ax


def foo():
    # Compute empirical posterior distribution over mu
    posterior_mu = (
        jnp.expand_dims(samples_1["mu"], - 1)
    )

    mean_mu = jnp.mean(posterior_mu, axis=0)
    hpdi_mu = hpdi(posterior_mu, 0.95)
    ax = plot_regression(nir2025_year_ints, mean_mu, hpdi_mu)
    ax.set(
        xlabel="Time", ylabel="Emissions (kt CO2e)", title="Plausible Values of a Constant-Predictor"
    );
    ax.axhline(uncenter(1.96 * np.std(centred_ca) / np.sqrt(len(centred_ca))), ls='--')
