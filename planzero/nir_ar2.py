
"""
Example inference that's fast and represents a lower bound on accuracy.
"""
try:
    import jax.numpy as jnp
    import jax.random as jrandom
    import numpy as np
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS
    from numpyro.infer import Predictive
    from numpyro.infer import init_to_median, init_to_feasible, init_to_value

    from numpyro.contrib.control_flow import scan
except ImportError:
    pass
import os
from pathlib import Path
import yaml

from .my_functools import inference_cache
from . import nir2025
from .enums import IPCC_Sector, GHG
from .symmetric_blended_lognormal import SymmetricBlendedLogNormal

version = 0.1

rolloff = 0.1


def ar2_scan_random_walk(scaled_ca,
                         scaled_pt,
                         sector_ghg_scale,
                         future_idx,
                         noise_ca,
                         pt_rms,
                         relerr,
                         observe_past=True,
                         n_future_timesteps=10):

    if 0:
        alpha_1 = .1 + 1.5 * numpyro.sample(
            "alpha_1",
            dist.Kumaraswamy(1.25, 1.25).expand((13,)))
        alpha_2 = -.5 + 1.0 * numpyro.sample(
            "alpha_2",
            dist.Kumaraswamy(1.25, 1.25).expand((13,)))
    else:
        # much bigger range
        alpha_1 = -.5 + 3 * numpyro.sample(
            "alpha_1",
            dist.Kumaraswamy(1.25, 1.25).expand((13,)))
        alpha_2 = -1.5 + 3 * numpyro.sample(
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
        y_t = numpyro.sample("mu", dist.Normal(m_t, mu_sigma))
        carry = (y_t, y_prev)
        return carry, m_t

    timesteps = jnp.arange(n_past_steps - 2 + n_future_timesteps)

    obs_valid = jnp.isfinite(scaled_pt[:, :future_idx])
    approx_obs = jnp.where(
        obs_valid,
        scaled_pt[:, :future_idx],
        jnp.nanmean(scaled_pt[:, :future_idx], axis=1, keepdims=True))

    mu_0 = numpyro.sample("mu_0", dist.Normal(0, mu_sigma))
    mu_1 = numpyro.sample("mu_1", dist.Normal(0, mu_sigma))

    init = (mu_1, mu_0)
    _, mu = scan(transition, init, timesteps)

    if 0:

        numpyro.sample("past-pt",
                       dist.Normal(mu[:n_past_steps - 2],
                                   noise_ca * pt_rms).mask(obs_valid[:, 2:future_idx].T),
                       obs=approx_obs[:, 2:future_idx].T if observe_past else None)
        numpyro.sample("past-ca",
                       dist.Normal(jnp.sum(mu[:n_past_steps - 2], axis=1),
                                   noise_ca),
                       obs=scaled_ca[2:future_idx] if observe_past else None)
    else:
        past_pt_dist0 = SymmetricBlendedLogNormal.rolloff_relerr(
                           mu_0,
                           rolloff=rolloff,
                           relerr=relerr)
        numpyro.sample("past-pt0",
                       past_pt_dist0.mask(obs_valid[:, 0]),
                       obs=approx_obs[:, 0] if observe_past else None)

        past_pt_dist1 = SymmetricBlendedLogNormal.rolloff_relerr(
                           mu_1,
                           rolloff=rolloff,
                           relerr=relerr)
        numpyro.sample("past-pt1",
                       past_pt_dist1.mask(obs_valid[:, 1]),
                       obs=approx_obs[:, 1] if observe_past else None)

        past_pt_dist = SymmetricBlendedLogNormal.rolloff_relerr(
                           mu[:n_past_steps - 2],
                           rolloff=rolloff,
                           relerr=relerr)
        numpyro.sample("past-pt",
                       past_pt_dist.mask(obs_valid[:, 2:future_idx].T),
                       obs=approx_obs[:, 2:future_idx].T if observe_past else None)
        if 1:
            target_ca_mean = jnp.sum(mu[:n_past_steps - 2], axis=1)
            past_ca_dist = SymmetricBlendedLogNormal.rolloff_relerr(
                               target_ca_mean,
                               rolloff=rolloff,
                               relerr=relerr)
            numpyro.sample("past-ca",
                           past_ca_dist,
                           obs=scaled_ca[2:future_idx] if observe_past else None)

    if 0:
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


def complete_mu(samples):
    mu = np.empty(
        (samples['mu'].shape[0],
         samples['mu'].shape[1] + 2,
         samples['mu'].shape[2]),
        dtype=str(samples['mu'].dtype)
        )
    if 'mu_0' in samples:
        mu[:, 0] = samples['mu_0']
        mu[:, 1] = samples['mu_1']
    else:
        mu[:, 0] = samples['mu'][: , 0]
        mu[:, 1] = samples['mu'][: , 0]
    mu[:, 2:] = samples['mu']
    return mu


def sample_past_given_mu(samples, key, relerr, pad_mu0_m1=False):
    rval = dict(samples)
    key_a, key_b = jrandom.split(key, 2)
    if pad_mu0_m1:
        mu = complete_mu(samples)
    else:
        mu = samples['mu']
    past_pt_dist = SymmetricBlendedLogNormal.rolloff_relerr(
                       mu,
                       rolloff=rolloff,
                       relerr=relerr)
    rval['past-pt'] = past_pt_dist.sample(key=key_a)

    # relative to code in ar2_scan_random_walk
    # a samples dimension has been added up front, so the axis is 2 here
    # (TODO: figure out how to use numpyro effect handlers)
    mean_ca = jnp.sum(mu, axis=2)
    past_ca_dist = SymmetricBlendedLogNormal.rolloff_relerr(
                       mean_ca,
                       rolloff=rolloff,
                       relerr=relerr)
    rval['past-ca'] = past_ca_dist.sample(key=key_b)
    return rval


def samples_with_extended_mu(samples, n_steps, pt_rms, np_rng):
    mu = samples['mu']
    alpha_1 = samples['alpha_1']
    alpha_2 = samples['alpha_2']
    n_samples, n_timesteps, n_regions = mu.shape
    new_mu = np.empty(
        (n_samples, n_timesteps + n_steps, n_regions),
        dtype=mu.dtype)
    new_mu[:, :n_timesteps, :] = mu

    const = 0
    mu_sigma = .05 * pt_rms # match above

    for ii in range(n_timesteps, new_mu.shape[1]):
        new_mu[:, ii, :] = (
            (const
             + alpha_1 * new_mu[:, ii - 1, :]
             + alpha_2 * new_mu[:, ii - 2, :])
            + np_rng.standard_normal((n_samples, n_regions)) * mu_sigma)
    return {
        key: new_mu if key == 'mu' else val
        for key, val in samples.items()}



class NIR2025_AR2(object):

    def __init__(self, sector, ghg, future_idx, seed=0):
        self.sector = sector
        self.ghg = ghg
        self.future_idx = future_idx
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

        # rms for each region, nanmean over time
        self.pt_rms = jnp.sqrt(
            jnp.maximum(
                jnp.nanmean(self.scaled_pt[:, :self.future_idx] ** 2,
                            axis=1),
                0.05 ** 2))

        self.post_samples = None
        self.rng_key = jrandom.key(seed)
        self.n_past_steps = len(self.scaled_ca[:self.future_idx])
        self.obs_valid = jnp.isfinite(self.scaled_pt[:, :self.future_idx])
        self.approx_obs = jnp.where(
            self.obs_valid,
            self.scaled_pt[:, :self.future_idx],
            jnp.nanmean(self.scaled_pt[:, :self.future_idx], axis=1, keepdims=True))

    @property
    def noise_ca(self):
        # delete?
        if self.sector == IPCC_Sector.Harvested_Wood_Products:
            return 0.05
        else:
            return 0.015

    @property
    def relerr(self):
        # delete?
        if self.sector == IPCC_Sector.Cropland:
            return 0.5
        else:
            return 0.05

    @staticmethod
    def idx_of_last_train_year(last_train_year):
        # Start from this source of randomness. We will split keys for subsequent operations.
        for ii, year in enumerate(nir2025.nir2025_year_ints):
            if year == last_train_year:
                future_idx = ii + 1
                break
        else:
            assert last_train_year >= 2024
            future_idx = len(nir2025.nir2025_year_ints)
        return future_idx

    @classmethod
    def posterior_inference(cls, sector, ghg, last_train_year, version=version,
                            thinning=1,
                            num_warmup=1000,
                            num_samples=2000,
                            n_future_timesteps=0):
        future_idx = cls.idx_of_last_train_year(last_train_year)
        self = cls(sector, ghg, future_idx=future_idx)
        self.rng_key, rng_key_ = jrandom.split(self.rng_key)

        mcmc = MCMC(
            #NUTS(ar2_scan_random_walk, init_strategy=init_to_median),
            NUTS(ar2_scan_random_walk, #),
                 init_strategy=init_to_value(
                     values={"alpha_1": jnp.zeros(13) + .5,
                             "alpha_2": jnp.zeros(13) + .5,
                             "mu": self.approx_obs[:, 2:].T,
                             "mu_0": self.approx_obs[:, 0],
                             "mu_1": self.approx_obs[:, 1],
                            }
                     )),
            num_warmup=num_warmup,
            num_samples=num_samples,
            thinning=thinning)
        with numpyro.validation_enabled():
            mcmc.run(rng_key_,
                     scaled_ca=self.scaled_ca,
                     scaled_pt=self.scaled_pt,
                     sector_ghg_scale=self.scale,
                     future_idx=future_idx,
                     noise_ca=self.noise_ca,
                     pt_rms=self.pt_rms,
                     relerr=self.relerr,
                     n_future_timesteps=n_future_timesteps,
                    )
        #mcmc.print_summary()
        self.mcmc = mcmc
        self.post_samples = mcmc.get_samples()
        return self

    def reconstructed_past(self):
        predictive = Predictive(
            ar2_scan_random_walk,
            self.post_samples,
            return_sites=['past-pt-1', 'past-ca-1'],
            )
        self.rng_key, rng_key_ = jrandom.split(self.rng_key)
        predictions = predictive(
            rng_key_,
            scaled_ca=self.scaled_ca,
            scaled_pt=self.scaled_pt,
            sector_ghg_scale=self.scale,
            future_idx=self.future_idx,
            noise_ca=self.noise_ca,
            relerr=self.relerr,
            pt_rms=self.pt_rms,
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
            future_idx=self.future_idx,
            relerr=self.relerr,
            noise_ca=self.noise_ca,
            pt_rms=self.pt_rms,
            )
        return predictions

# XXX test rhat distribution

def model_root():
    return f'./cache/inference/AR2'


def sector_ghg_root(sector, ghg):
    return f'{model_root()}/{str(ghg)}-{str(sector)}'


def sector_ghg_config_path(sector, ghg):
    return f'{sector_ghg_root(sector, ghg)}/config.yaml'


class VersionMismatch(RuntimeError):
    pass


# TODO: refactor with nir_constant_predictor

def load_config(allow_version_mismatch):
    with open(f'{model_root()}/config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file) or {}
    versions_match = config.get('version', 0.1) == version
    if allow_version_mismatch or versions_match:
        return config
    raise VersionMismatch()


def sector_ghg_load_config(sector, ghg, allow_version_mismatch):
    with open(sector_ghg_config_path(sector, ghg), 'r') as config_file:
        config = yaml.safe_load(config_file) or {}
    versions_match = config.get('version', 0.1) == version
    if allow_version_mismatch or versions_match:
        return config
    raise VersionMismatch()


def load_config_samples(sector, ghg):
    config = sector_ghg_load_config(sector, ghg, allow_version_mismatch=False)
    rootdir = Path(sector_ghg_root(sector, ghg))

    # Dictionary to store the memory-mapped arrays
    mmap_dict = {}

    # Ensure the directory exists before iterating
    if rootdir.is_dir():
        # Loop over all files ending with .npy in the directory
        for file_path in rootdir.glob("*.npy"):
            # Extract the 'key' (the filename without the extension)
            key = file_path.stem
            try:
                # Load the file as a read-only memory-mapped array
                mmap_dict[key] = np.load(file_path, mmap_mode='r')
            except Exception as e:
                raise Exception(f"Error loading {file_path.name}: {e}") from e
    else:
        raise Exception(f"Directory not found: {rootdir}")

    return config, mmap_dict


def main():
    """Populate cache/inference/AR2
    """

    arr_pt, arr_ca = nir2025.ktCO2e_dense_w_nan()
    from .enums import IPCC_Sector, GHG

    config_data = {
        'near_zero_sector_ghgs': [],
        'version': version,
        'last_train_year': 2022,
        'thinning': 10,
        'num_warmup': 500,
        'num_samples': 2000,
        'save_params': ['alpha_1', 'alpha_2', 'mu_0', 'mu_1', 'mu'],
        'n_future_timesteps': 0,
    }
    nontrivial = []
    for sector in IPCC_Sector:
        for ghg in GHG:
            jnp_pt = jnp.array(
                arr_pt[nir2025.idx_of_sector[sector],
                       nir2025.idx_of_ghg[ghg]],
                dtype='float64')
            jnp_ca = jnp.array(
                arr_ca[nir2025.idx_of_sector[sector],
                       nir2025.idx_of_ghg[ghg]],
                dtype='float64')
            if jnp.nansum(abs(jnp_ca)) <= 1: # 1kt, aka very small
                config_data['near_zero_sector_ghgs'].append(
                    [str(sector), str(ghg)])
            else:
                nontrivial.append((sector, ghg))

    # Write the top-level config file
    os.makedirs(model_root(), exist_ok=True)
    with open(f'{model_root()}/config.yaml', 'w') as file:
        yaml.safe_dump(config_data, file, default_flow_style=False)

    for ii, (sector, ghg) in enumerate(nontrivial):
        os.makedirs(sector_ghg_root(sector, ghg), exist_ok=True)
        try:
            with open(sector_ghg_config_path(sector, ghg), 'r') as prev_config_file:
                prev_config = yaml.safe_load(prev_config_file) or {}
            prev_version = prev_config.get('version', 0)
            if prev_version != version:
                assert prev_version < version
                print ('updating old version', prev_version, 'to', version)
                raise RuntimeError('saved version is old')
            print(ii, '/', len(nontrivial), sector, ghg, 'done')
            continue
        except IOError:
            pass
        except RuntimeError:
            pass

        print(ii, '/', len(nontrivial), sector, ghg)
        self = NIR2025_AR2.posterior_inference(
            sector=sector,
            ghg=ghg,
            last_train_year=config_data['last_train_year'],
            version=None,
            thinning=config_data['thinning'],
            num_warmup=config_data['num_warmup'],
            num_samples=config_data['num_samples'],
            n_future_timesteps=config_data['n_future_timesteps'],
            )
        for key, arr in self.post_samples.items():
            print(key, arr.shape, arr.size, 'elems', arr.size * 4, 'float32')

        # Save samples
        # TODO: sub-sample them
        # TODO: only save alpha_1, alpha_2, mu_0, mu_1 and y
        os.makedirs(sector_ghg_root(sector, ghg), exist_ok=True)
        grouped_samples = self.mcmc.get_samples(group_by_chain=True)
        for key, val in grouped_samples.items():
            fp = np.lib.format.open_memmap(
                f'{sector_ghg_root(sector, ghg)}/{key}.npy',
                mode='w+',
                dtype='float32', # save space
                shape=val.shape)
            fp[:] = val
            fp.flush()

        # Write the per-sector-gas config file
        config = {
            'version': version,
            'scale': float(self.scale),
            'sector': sector.value,
            'ghg': ghg.value,
        }
        with open(sector_ghg_config_path(sector, ghg), 'w') as file:
            yaml.safe_dump(config, file, default_flow_style=False)


if __name__ == '__main__':
    import sys
    sys.exit(main())
