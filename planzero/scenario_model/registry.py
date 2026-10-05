import hashlib
import os
import pickle

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from jax.typing import ArrayLike

from .base import GroupedPosterior, VarKey
from .computation import Computation, Model, run_mcmc, hash_version_id

MODEL_REGISTRY_ROOT = os.environ.get('PLANZERO_MODEL_REGISTRY_ROOT')

# filenames shouldn't be ridiculously long
MAX_NAME_LEN = 100


def var_key_hash(var_key: VarKey, n_chars=16) -> str:
    """Returns a stable, cross-session SHA-256 hex string."""
    # 1. Encode the string to bytes
    encoded_data = str(var_key).encode('utf-8')

    # 2. Generate and return the hexadecimal digest
    return hashlib.shake_128(encoded_data).hexdigest(n_chars)


def model_root(model_name:str):
    assert len(model_name) < MAX_NAME_LEN, ("catch accidentally long strings", model_name)
    assert MODEL_REGISTRY_ROOT, "env var PLANZERO_MODEL_REGISTRY_ROOT not set"
    return f'{MODEL_REGISTRY_ROOT}/model_{model_name}'


def var_key_path(model_name:str, var_key:VarKey):
    return f'{model_root(model_name)}/var_key_np_{var_key_hash(var_key)}.npy'


def save_var(model_name:str, var_key:VarKey, val:np.ndarray):
    path = var_key_path(model_name, var_key)
    if val.dtype == np.float64:
        dtype = np.float32  # save space, TODO: check for underflow, overflow
    else:
        raise NotImplementedError(val.dtype)

    fp = np.lib.format.open_memmap(
        path,
        mode='w+',
        dtype=dtype,
        shape=val.shape)
    fp[:] = val
    fp.flush()


def load_var(model_name:str, var_key:VarKey):
    # TODO: consider putting back to original dtype
    path = var_key_path(model_name, var_key)
    return np.load(path, mmap_mode='r')


def sentinel_path(model_name:str, sentinel:str, submodel_name:str, version_hash:str):
    assert len(submodel_name) < MAX_NAME_LEN, ("catch accidentally long strings", submodel_name)
    return (f'{model_root(model_name)}'
            f'/sentinel_{sentinel}_{submodel_name}_{version_hash}')


def sentinel_present(model_name:str, sentinel:str, submodel_name:str, version_hash:str):
    try:
        with open(sentinel_path(model_name, sentinel, submodel_name, version_hash)):
            return True
    except OSError:
        return False


def sentinel_save(model_name:str, sentinel:str, submodel_name:str, version_hash:str, sample_sites:dict[VarKey, str]):
    with open(sentinel_path(model_name, sentinel, submodel_name, version_hash), 'wb') as ofile:
        pickle.dump(sample_sites, ofile)
        return True


def sentinel_load(model_name:str, sentinel:str, submodel_name:str, version_hash:str):
    with open(sentinel_path(model_name, sentinel, submodel_name, version_hash), 'rb') as ifile:
        return pickle.load(ifile)


def compute_mcmc_submodel(
        model_name:str,
        mcmc_group:str,
        submodel:Model,
        rng_key:ArrayLike,
        ):
    submodel_version_id = tuple(
            [(elem_id, elem.version_id)
             for elem_id, elem in sorted(submodel.model_elements.items())])
    version_hash:str = hash_version_id(submodel_version_id)
    if not sentinel_present(model_name, 'mcmc', mcmc_group, version_hash):
        print('collecting sample sites for', model_name, mcmc_group, version_hash)
        comp = Computation(model=submodel, rng_key=jrandom.key(1))
        comp.run()
        print('running mcmc for', model_name, mcmc_group, version_hash)
        mcmc = run_mcmc(submodel, seed=rng_key)
        grouped_samples = mcmc.get_samples(group_by_chain=True)
        for var_key, sample_site in comp.sample_sites.items():
            if sample_site in grouped_samples:
                save_var(
                        model_name,
                        GroupedPosterior(prior_var_key=var_key),
                        grouped_samples[sample_site])
        sentinel_save(model_name, 'mcmc', mcmc_group, version_hash, comp.sample_sites)
        sample_sites = comp.sample_sites
    else:
        sample_sites = sentinel_load(model_name, 'mcmc', mcmc_group, version_hash)
    return sample_sites


def compute_posterior(
        model_name:str,
        model:Model,
        rng_key:ArrayLike,
        sample_sites:dict[VarKey, str],
        ) -> Computation:
    grouped_samples = {}
    for var_key, sample_site in sample_sites.items():
        try:
            grouped_samples[sample_site] = load_var(
                    model_name,
                    GroupedPosterior(prior_var_key=var_key))
        except OSError as err:
            # TODO: it's okay if it was fully-observed
            err.add_note(f'var_key={var_key}')
            err.add_note(f'sample_site={sample_site}')
            raise
    comp = Computation(model=model, rng_key=rng_key)
    comp.set_phase_posterior(
            n_mcmc=model.num_samples,
            sample_sites=sample_sites,
            grouped_samples=grouped_samples)
    comp.run()
    return comp


_registry_mem_cache:dict[str, Computation] = {}


def registry_compute_model(
        model:Model,
        model_name:str,
        cache_posterior:bool,  # True will save it in memory by model_name, and return from memory if possible
        seed_or_key:int|ArrayLike,
        run_mcmc_kwargs:None|dict[str,int]=None,
        ) -> Computation:
    if cache_posterior and model_name in _registry_mem_cache:
        return _registry_mem_cache[model_name]

    if isinstance(seed_or_key, int):
        rng_key = jrandom.key(seed_or_key)
    else:
        rng_key = seed_or_key

    submodels = model.mcmc_submodels()

    os.makedirs(model_root(model_name), exist_ok=True)

    # It should be possible to run this loop concurrently e.g. vmap
    sample_sites = {}
    for mcmc_group, submodel in submodels.items():
        # for saving MCMC re-sampling of sample sites
        rng_key, tmp_key = jrandom.split(rng_key)
        submodel_sample_sites = compute_mcmc_submodel(
                model_name, mcmc_group, submodel,
                rng_key=tmp_key)
        for key, site in submodel_sample_sites.items():
            assert key not in sample_sites
            sample_sites[key] = site

    comp = compute_posterior(model_name, model, rng_key=rng_key, sample_sites=sample_sites)

    if cache_posterior:
        _registry_mem_cache[model_name] = comp
    return comp
