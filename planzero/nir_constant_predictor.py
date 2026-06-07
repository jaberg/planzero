"""
Example inference that's fast and represents a lower bound on accuracy.
"""
import numpy as np
import jax.numpy as jnp
import jax.random as jrandom
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from . import nir2025
from .johnsonsu import JohnsonSU


def constant_model(scaled_pt=None, scaled_ca=None, future_idx=1):
    n_regions, n_timesteps = scaled_pt.shape
    mu = numpyro.sample("mu", dist.Normal(0.0, 5.0).expand((n_regions,)))
    sigma = numpyro.sample("sigma", dist.Exponential(1.0))
    valid_pt = jnp.isfinite(scaled_pt)
    numpyro.sample("obs_pt", dist.Normal(mu, sigma).mask(valid_pt.T), obs=scaled_pt.T).T
    numpyro.sample("obs_ca", dist.Normal(jnp.sum(mu), sigma), obs=scaled_ca)
    #numpyro.sample("forecast", dist.Normal(mu, sigma), sample_shape=(n_forecast,))


class NIR2025_Model(object):

    def __init__(self, sector, ghg):
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
        ])

        assert np.isfinite(self.scale)
        self.scaled_ca = jnp.array(self.jnp_ca / self.scale)
        self.scaled_pt = jnp.array(self.jnp_pt / self.scale)
        self.post_samples = None

    def posterior_inference(self, seed=0, num_warmup=250, num_samples=1000):

        # Start from this source of randomness. We will split keys for
        # subsequent operations.
        rng_key = jrandom.key(seed)
        rng_key, rng_key_ = jrandom.split(rng_key)

        # Run NUTS.
        mcmc = MCMC(NUTS(constant_model),
                    num_warmup=250,
                    num_samples=num_samples)
        mcmc.run(rng_key_,
                 scaled_pt=self.scaled_pt,
                 scaled_ca=self.scaled_ca)
        mcmc.print_summary()
        self.post_samples = mcmc.get_samples()


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
