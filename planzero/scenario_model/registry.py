import os
from collections.abc import Iterable

import jax.random as jrandom
from jax.typing import ArrayLike

from .base import GroupedPosterior, VarKey
from .computation import Computation, Model, run_mcmc

MODEL_REGISTRY_ROOT = os.environ.get('PLANZERO_MODEL_REGISTRY_ROOT')


def var_keys_to_save_from_prior(comp: Computation) -> Iterable[VarKey]:
    yield from comp.storage_nd


def save_var(model_name:str, var_key:VarKey, val:ArrayLike):
    raise NotImplementedError()


def load_var(model_name:str, var_key:VarKey):
    raise NotImplementedError()


def sentinel_present(model_name:str, sentinel:str, submodel_name:str=""):
    raise NotImplementedError()


def sentinel_touch(model_name:str, sentinel:str, submodel_name:str=""):
    raise NotImplementedError()


def compute_mcmc_submodel(model_name, submodel_name, submodel, rng_key:ArrayLike, sample_sites:dict[VarKey, str]):
    if not sentinel_present(model_name, 'mcmc', submodel_name):
        mcmc = run_mcmc(submodel, seed=rng_key)
        grouped_samples = mcmc.get_samples(group_by_chain=True)
        for var_key, sample_site in sample_sites.items():
            if sample_site in grouped_samples:
                save_var(
                        model_name,
                        GroupedPosterior(prior_var_key=var_key),
                        grouped_samples[sample_site])
        sentinel_touch(model_name, 'mcmc', submodel_name)


def compute_posterior(
        model_name:str,
        model:Model,
        rng_key:ArrayLike,
        ) -> Computation:
    sample_sites = model.mv.sample_sites
    grouped_samples = {}
    for var_key, sample_site in sample_sites.items():
        try:
            grouped_samples[sample_site] = load_var(
                    model_name,
                    GroupedPosterior(prior_var_key=var_key))
        except OSError as err:
            # TODO: it's okay if it was fully-observed
            err.add_note(f'var_key={var_key:s}')
            err.add_note(f'sample_site={sample_site}')
            raise
    comp = Computation(
            model=model,
            rng_key=rng_key,
            grouped_samples=grouped_samples)
    comp.run()
    return comp


_registry_mem_cache = {}

def registry_compute_model(
        model:Model,
        model_name:str,
        cache:bool,  # True will save it in memory by model_name, and return from memory if possible
        seed_or_key:int|ArrayLike,
        submodels: dict[str, Model]|None=None,
        run_mcmc_kwargs:None|dict[str,int]=None,
        ) -> Computation:
    if cache and model_name in _registry_mem_cache:
        return _registry_mem_cache[model_name]

    if isinstance(seed_or_key, int):
        rng_key = jrandom.key(seed_or_key)
    else:
        rng_key = seed_or_key

    if not submodels:
        submodels = {model_name: model}

    # It should be possible to run this loop concurrently e.g. vmap

    for submodel_name, submodel in submodels.items():
        # for saving MCMC re-sampling of sample sites
        rng_key, tmp_key = jrandom.split(rng_key)
        compute_mcmc_submodel(model_name, submodel_name, submodel,
                              rng_key=tmp_key,
                              sample_sites=model.mv.sample_sites)

    comp = compute_posterior(model_name, model, rng_key=rng_key)

    if cache:
        _registry_mem_cache[model_name] = comp
    return comp
