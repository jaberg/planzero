import filelock
import hashlib
import os
import pickle

import jax.random as jrandom
import numpy as np
from jax.typing import ArrayLike

from .base import GroupedPosterior, VarKey
from .computation import Computation, ScenarioModel, hash_version_id, run_mcmc

MODEL_REGISTRY_ROOT = os.environ.get('PLANZERO_MODEL_REGISTRY_ROOT')

# filenames shouldn't be ridiculously long
MAX_NAME_LEN = 100


def var_key_hash(var_key: VarKey, n_chars=16) -> str:
    """Returns a stable, cross-session SHA-256 hex string."""
    # 1. Encode the string to bytes
    encoded_data = str(var_key).encode('utf-8')

    # 2. Generate and return the hexadecimal digest
    rval = hashlib.shake_128(encoded_data).hexdigest(n_chars)
    return rval


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
        submodel:ScenarioModel,
        rng_key:ArrayLike,
        msg_prefix:str="",
        ):
    this_subsystem_version = 1.1
    submodel_version_id = (
            this_subsystem_version,
            tuple(
                [(elem_id, elem.version_id)
                 for elem_id, elem in sorted(submodel.model_elements.items())]),
                )
    version_hash:str = hash_version_id(submodel_version_id)
    do_mcmc = (
        not sentinel_present(model_name, 'mcmc', mcmc_group, version_hash)
        #or ('Fishing' in mcmc_group)
        #or ('Black' in mcmc_group)
        #or ('Ammonia' in mcmc_group)
        #or ('Nitric' in mcmc_group)
        )
    if do_mcmc:
        comp = Computation(model=submodel, rng_key=jrandom.key(1))
        comp.run()
        print(msg_prefix, 'running mcmc for', model_name, mcmc_group, version_hash)
        try:
            mcmc = run_mcmc(submodel,
                            seed=rng_key,
                            progress_bar=True)
        except Exception as err:
            err.add_note(f'model_name {model_name}')
            err.add_note(f'mcmc_group {mcmc_group}')
            raise
        grouped_samples = mcmc.get_samples(group_by_chain=True)
        saved_sample_sites = {}
        for var_key, sample_site in comp.sample_sites.items():
            if sample_site in grouped_samples:
                save_var(
                        model_name,
                        GroupedPosterior(prior_var_key=var_key),
                        grouped_samples[sample_site])
                saved_sample_sites[var_key] = sample_site
        assert len(saved_sample_sites) == len(grouped_samples)
        sentinel_save(model_name, 'mcmc', mcmc_group, version_hash, saved_sample_sites)
    else:
        saved_sample_sites = sentinel_load(model_name, 'mcmc', mcmc_group, version_hash)
    return saved_sample_sites


def maybe_compute_mcmc_submodel(
        model_name:str,
        mcmc_group:str,
        submodel:ScenarioModel,
        rng_key:ArrayLike,
        timeout:float,
        msg_prefix:str,
        ):
    this_subsystem_version = 1.1
    submodel_version_id = (
            this_subsystem_version,
            tuple(
                [(elem_id, elem.version_id)
                 for elem_id, elem in sorted(submodel.model_elements.items())]),
                )
    version_hash:str = hash_version_id(submodel_version_id)
    lockpath = sentinel_path(model_name, 'mcmc', mcmc_group, version_hash)
    lock = filelock.FileLock(lockpath + ".lock", timeout=timeout)
    with lock:
        return compute_mcmc_submodel(model_name=model_name,
                                     mcmc_group=mcmc_group,
                                     submodel=submodel,
                                     rng_key=rng_key,
                                     msg_prefix=msg_prefix)


def compute_posterior(
        model_name:str,
        model:ScenarioModel,
        rng_key:ArrayLike,
        saved_sample_sites:dict[VarKey, str],
        run_computation:bool=True,
        ) -> Computation:
    grouped_samples = {}
    for var_key, sample_site in saved_sample_sites.items():
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
            sample_sites=saved_sample_sites,
            grouped_samples=grouped_samples)
    if run_computation:
        comp.run()
    return comp


_registry_mem_cache:dict[str, Computation] = {}

def registry_compute_model(
        model:ScenarioModel,
        model_name:str,
        cache_posterior:bool,  # True will save it in memory by model_name, and return from memory if possible
        seed_or_key:int|ArrayLike,
        run_compute_posterior:bool=True,
        ) -> Computation:
    if cache_posterior and model_name in _registry_mem_cache:
        return _registry_mem_cache[model_name]

    if isinstance(seed_or_key, int):
        rng_key = jrandom.key(seed_or_key)
    else:
        rng_key = seed_or_key

    submodels = model.mcmc_submodels()

    os.makedirs(model_root(model_name), exist_ok=True)

    args_by_mcmc_group = {}
    for mcmc_group, submodel in submodels.items():
        # for saving MCMC re-sampling of sample sites
        rng_key, tmp_key = jrandom.split(rng_key)
        args_by_mcmc_group[mcmc_group] = (model_name, mcmc_group, submodel, tmp_key)

    saved_sites_d = {}
    n_groups_total = len(args_by_mcmc_group)
    n_groups_done = 0
    for passnum in range(10):
        n_timeouts = 0
        for mcmc_group, args in args_by_mcmc_group.items():
            if mcmc_group not in saved_sites_d:
                try:
                    submodel_saved_sample_sites = maybe_compute_mcmc_submodel(
                            *args,
                            msg_prefix=f'{n_groups_done}/{n_groups_total}',
                            timeout=0)
                    saved_sites_d[mcmc_group] = submodel_saved_sample_sites
                    n_groups_done += 1
                except filelock.Timeout:
                    n_timeouts += 1
        if n_timeouts == 0:
            # saved_sites_d is complete
            break

    # final pass where we wait for completion
    # and timeouts are errors
    for mcmc_group, args in args_by_mcmc_group.items():
        if mcmc_group not in saved_sites_d:
            submodel_saved_sample_sites = maybe_compute_mcmc_submodel(
                    *args,
                    msg_prefix=f'{n_groups_done}/{n_groups_total}',
                    timeout=-1)
            saved_sites_d[mcmc_group] = submodel_saved_sample_sites
            n_groups_done += 1

    assert n_groups_done == n_groups_total
    saved_sample_sites = {}
    for submodel_saved_sample_sites in saved_sites_d.values():
        for key, site in submodel_saved_sample_sites.items():
            assert key not in saved_sample_sites, key
            saved_sample_sites[key] = site

    comp = compute_posterior(
            model_name,
            model,
            rng_key=rng_key,
            saved_sample_sites=saved_sample_sites,
            run_computation=run_compute_posterior,
            )

    if cache_posterior:
        _registry_mem_cache[model_name] = comp
    return comp
