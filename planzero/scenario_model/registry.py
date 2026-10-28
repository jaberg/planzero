"""
The design of the registry:

Every model for which registry_compute_model is called gets a folder
{MODEL_REGISTRY_ROOT}/model_{model_name}_{version_hash}_{seed_hash}.

In that [per-model] folder, there is at least "sentinel file" for each mcmc_group,
with which is associated a submodel whose mcmc can be run independently.

There is also a ".locks" subfolder with lock files that prevent multiple
processes from working on the same submodel at the same time.

In the sentinel file there is data saved with pickle:
* (model_name, version_hash, seed_hash) of model where these samples can be loaded
* saved_sample_sites

saved_sample_sites is a dictionary mapping model keys (type VarKey) to the string
names assigned by the numpyro MCMC sampler.
"""

import hashlib
import os
import pickle
import threading
import warnings
from dataclasses import dataclass

import filelock
import jax.random as jrandom
import numpy as np
from jax.typing import ArrayLike

from .base import GroupedPosterior, VarKey
from .computation import Computation, run_mcmc
from .scenario_model import ScenarioModel

MODEL_REGISTRY_ROOT = os.environ.get('PLANZERO_MODEL_REGISTRY_ROOT')

MAX_NAME_LEN = 100


def var_key_hash(var_key: VarKey, n_chars=32) -> str:
    assert not n_chars % 2
    encoded_data = str(var_key).encode('utf-8')
    rval = hashlib.shake_128(encoded_data).hexdigest(n_chars // 2)
    assert len(rval) == n_chars
    return rval


def infer_model_root(model_name: str, version_hash:str, seed_hash:str):
    assert len(model_name) < MAX_NAME_LEN, ("catch accidentally long strings", model_name)
    assert MODEL_REGISTRY_ROOT, "env var PLANZERO_MODEL_REGISTRY_ROOT not set"
    assert len(version_hash) == 32
    assert len(seed_hash) == 32
    return f'{MODEL_REGISTRY_ROOT}/model_{model_name}_{version_hash}_{seed_hash}'


def var_key_path(model_name: str, version_hash:str, seed_hash:str, var_key: VarKey):
    """Path where samples would be saved for var_key, if they were saved by model_name"""
    model_root = infer_model_root(model_name, version_hash, seed_hash=seed_hash)
    return f'{model_root}/var_key_np_{var_key_hash(var_key)}.npy'


def _iter_model_roots():
    """Iterate over model roots in the registry"""
    if not MODEL_REGISTRY_ROOT:
        return
    try:
        for entry in os.scandir(MODEL_REGISTRY_ROOT):
            if entry.is_dir() and entry.name.startswith('model_'):
                yield entry.path
    except OSError:
        return


def _parse_model_info_from_root(model_root_path: str) -> tuple[str, str, str]:
    name = os.path.basename(model_root_path)
    tokens = name.split('_')
    model_name = '_'.join(tokens[1:-2]) # model_..._versionhash_seedhash
    version_hash = tokens[-2]
    seed_hash = tokens[-1]
    assert model_name
    return model_name, version_hash, seed_hash


def _rng_key_hash(rng_key: ArrayLike, n_chars=32) -> str:
    arr = jrandom.key_data(rng_key)
    encoded = arr.tobytes()
    assert not n_chars % 2
    rval = hashlib.shake_128(encoded).hexdigest(n_chars // 2)
    assert len(rval) == n_chars
    return rval


@dataclass(frozen=True)
class AvailableInference:
    infer_key: tuple[str, str] # version_hash of submodel, mcmc group seed hash
    model_name: str
    model_root_path: str
    model_version_hash: str
    model_seed_hash:str
    sentinel_path: str
    saved_sample_sites: dict[VarKey, str] # mcmc group saved sites
    mtime_ns: int  # modification-time of inference sentinel files


_infer_registry_lock = threading.RLock()
_infer_registry: dict[tuple[str, str], AvailableInference] = {}
_registry_scanned_mtime: float | None = None


def _scan_available_inferences(force: bool = False) -> dict[tuple[str, str], AvailableInference]:
    global _registry_scanned_mtime

    if not MODEL_REGISTRY_ROOT:
        warnings.warn('MODEL_REGISTRY_ROOT not set')
        return _infer_registry

    try:
        root_stat = os.stat(MODEL_REGISTRY_ROOT)
        root_mtime = root_stat.st_mtime_ns
    except OSError:
        root_mtime = None

    # return cached if not force
    if (not force and _registry_scanned_mtime is not None
            and root_mtime == _registry_scanned_mtime
            and _infer_registry):
        return _infer_registry

    discovered: dict[tuple[str, str], AvailableInference] = {}

    for model_dir in _iter_model_roots():
        model_name, model_version_hash, model_seed_hash \
                = _parse_model_info_from_root(model_dir)
        try:
            for entry in os.scandir(model_dir):
                if not entry.is_file():
                    continue
                name = entry.name
                if (name.endswith('.tmp')
                    or not name.startswith('sentinel_mcmc_')):
                    continue
                parts = name.split('_')
                if len(parts) < 5:
                    continue
                submodel_version_hash = parts[-2]
                submodel_seed_hash = parts[-1]
                # mcmc_group = '_'.join(parts[3:-2])
                sentinel_path = entry.path
                assert len(submodel_version_hash) == 32, submodel_version_hash
                assert len(submodel_seed_hash) == 32, submodel_seed_hash
                try:
                    with open(sentinel_path, 'rb') as f:
                        (computing_model_name,
                         computing_model_version_hash,
                         computing_model_seed_hash) = pickle.load(f)
                        if computing_model_name != model_name:
                            # skip sentinel files that are themselves
                            # re-directs
                            continue
                        if computing_model_version_hash != model_version_hash:
                            continue
                        if computing_model_seed_hash != model_seed_hash:
                            continue
                        saved_sample_sites = pickle.load(f)
                except (OSError, pickle.PickleError, EOFError):
                    warnings.warn(
                            f"failed to unpickle saved_sample_sites from sentinel"
                            f" {sentinel_path}")
                    continue
                if not isinstance(saved_sample_sites, dict):
                    warnings.warn(
                            f"unpickled saved_sample_sites was not a dict"
                            f" {saved_sample_sites}")
                    continue
                for var_key in saved_sample_sites:
                    try:
                        var_key_path(
                                model_name=model_name,
                                version_hash=model_version_hash,
                                seed_hash=model_seed_hash,
                                var_key=var_key)
                    except Exception as err:
                        err.add_note(f'var_key={var_key}')
                        err.add_note(f'model_name={model_name}')
                        err.add_note(f'version_hash={model_version_hash}')
                        err.add_note(f'seed_hash={model_seed_hash}')
                        raise
                try:
                    mtime_ns = entry.stat().st_mtime_ns
                except OSError:
                    mtime_ns = 0
                infer_key = (submodel_version_hash, submodel_seed_hash)
                assert infer_key not in discovered
                discovered[infer_key] = AvailableInference(
                    infer_key=infer_key,
                    model_name=model_name,
                    model_version_hash=model_version_hash,
                    model_seed_hash=model_seed_hash,
                    model_root_path=model_dir,
                    sentinel_path=sentinel_path,
                    saved_sample_sites=saved_sample_sites,
                    mtime_ns=mtime_ns,
                )
        except OSError:
            continue

    # AI put this in in case there are threads, which there are not
    # but it's okay as is
    with _infer_registry_lock:
        _infer_registry.clear()
        _infer_registry.update(discovered)
        _registry_scanned_mtime = root_mtime

    return _infer_registry


def _get_available_inference(
        infer_key: tuple[str, str]) -> AvailableInference | None:
    with _infer_registry_lock:
        avail = _infer_registry.get(infer_key)
    if avail is None:
        _scan_available_inferences(force=True)
        with _infer_registry_lock:
            avail = _infer_registry.get(infer_key)
    return avail


def _register_inference(
        model_name: str,
        model_version_hash:str,
        model_seed_hash:str,
        infer_key: tuple[str, str],
        sentinel_path: str,
        saved_sample_sites: dict[VarKey, str]):
    try:
        mtime_ns = os.stat(sentinel_path).st_mtime_ns
    except OSError:
        mtime_ns = 0
    avail = AvailableInference(
        infer_key=infer_key,
        model_name=model_name,
        model_version_hash=model_version_hash,
        model_seed_hash=model_seed_hash,
        model_root_path=infer_model_root(
            model_name,
            version_hash=model_version_hash,
            seed_hash=model_seed_hash),
        sentinel_path=sentinel_path,
        saved_sample_sites=saved_sample_sites,
        mtime_ns=mtime_ns,
    )
    with _infer_registry_lock:
        _infer_registry[infer_key] = avail


def save_var(model_name: str,
             model_version_hash:str,
             model_seed_hash:str,
             var_key: VarKey,
             val: np.ndarray):
    path = var_key_path(
            model_name=model_name,
            version_hash=model_version_hash,
            seed_hash=model_seed_hash,
            var_key=var_key)
    if val.dtype == np.float64:
        dtype = np.float32
    else:
        raise NotImplementedError(val.dtype)

    fp = np.lib.format.open_memmap(
        path,
        mode='w+',
        dtype=dtype,
        shape=val.shape)
    fp[:] = val
    fp.flush()


def load_var(model_name: str, model_version_hash:str, model_seed_hash:str, var_key: VarKey):
    path = var_key_path(
            model_name=model_name,
            version_hash=model_version_hash,
            seed_hash=model_seed_hash,
            var_key=var_key)
    return np.load(path, mmap_mode='r')


def infer_sentinel_name(mcmc_group: str, version_hash: str, seed_hash: str):
    assert len(mcmc_group) < MAX_NAME_LEN, (
            "catch accidentally long strings", mcmc_group)
    assert len(version_hash) < MAX_NAME_LEN, (
            "catch accidentally long strings", version_hash)
    assert len(seed_hash) < MAX_NAME_LEN, (
            "catch accidentally long strings", seed_hash)
    assert '_' not in version_hash  # messes up parsing
    assert '_' not in seed_hash     # messes up parsing
    return f'sentinel_mcmc_{mcmc_group}_{version_hash}_{seed_hash}'


def infer_sentinel_path(model_name: str,
                        model_version_hash:str,
                        model_seed_hash:str,
                        mcmc_group: str, version_hash: str, seed_hash: str) -> str:
    return os.path.join(
            infer_model_root(model_name, model_version_hash, seed_hash=model_seed_hash),
            infer_sentinel_name(mcmc_group, version_hash, seed_hash))


def infer_lock_path(
        model_name: str,
        model_version_hash:str,
        model_seed_hash:str,
        mcmc_group: str, version_hash: str, seed_hash: str):
    locks_dir = os.path.join(
            infer_model_root(model_name, model_version_hash, model_seed_hash),
            '.locks')
    os.makedirs(locks_dir, exist_ok=True)
    return os.path.join(
            locks_dir,
            f'{infer_sentinel_name(mcmc_group, version_hash, seed_hash)}.lock')



class NoAvailableSentinel(Exception): pass

def _compute_mcmc_submodel_reuse(
        model_name, model_version_hash, model_seed_hash,
        mcmc_group, version_hash, seed_hash, msg_prefix,
        ) -> tuple[tuple[str, str, str], dict[VarKey, str]]:
    """
    Return (other_model_name, saved_sample_sites) if the required saved files
    exist within MODEL_REGISTRY_ROOT/model_{other_model_name}
    """
    infer_key = (version_hash, seed_hash)
    avail = _get_available_inference(infer_key)
    if avail is None:
        raise NoAvailableSentinel()
    else:
        saved_sample_sites = avail.saved_sample_sites
        for var_key in saved_sample_sites:
            vkpath = var_key_path(
                    model_name,
                    version_hash=model_version_hash,
                    seed_hash=model_seed_hash,
                    var_key=GroupedPosterior(prior_var_key=var_key))
            try:
                assert os.path.exists(vkpath)
            except Exception as err:
                err.add_note(f'model_name={model_name}')
                err.add_note(f'version_hash={model_version_hash}')
                err.add_note(f'seed_hash={model_seed_hash}')
                err.add_note(f'var_key={var_key}')
                err.add_note(f'vk_path={vkpath}')
                raise
        if msg_prefix:
            print(msg_prefix, 'reusing mcmc for', model_name, mcmc_group, version_hash, seed_hash)
        return ((avail.model_name,
                 avail.model_version_hash,
                 avail.model_seed_hash),
                saved_sample_sites)



def _compute_mcmc_submodel(
        model_name: str,
        model_version_hash:str,
        model_seed_hash:str,
        mcmc_group: str,
        submodel: ScenarioModel,
        rng_key: ArrayLike,
        msg_prefix: str = "",
) -> tuple[tuple[str, str, str], dict[VarKey, str]]:
    # This function may or may not actually do the mcmc computations,
    # depending on the determination of `do_mcmc` later in the function body.
    #
    # The post-condition of the function is that
    # 1. the samples are available somewhere within MODEL_REGISTRY_ROOT
    # 2. the return value of this function is determined
    # 3. the string keys in this function's return value
    #    can be used to load the samples.
    
    version_hash = submodel.version_hash()
    seed_hash = _rng_key_hash(rng_key)

    try:
        return _compute_mcmc_submodel_reuse(
            model_name, model_version_hash, model_seed_hash,
            mcmc_group, version_hash, seed_hash, msg_prefix)
    except NoAvailableSentinel:
        pass

    local_sentinel = infer_sentinel_path(
            model_name, model_version_hash, model_seed_hash,
            mcmc_group, version_hash, seed_hash)

    comp = Computation(model=submodel, rng_key=jrandom.key(1))
    comp.run()  # run once so the error is intelligible if it fails
    print(msg_prefix, 'running mcmc for', model_name, mcmc_group, version_hash)
    try:
        # this will run a JIT-compiled model whose error traceback
        # is less useful than comp.run() above
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
                model_name, model_version_hash, model_seed_hash,
                GroupedPosterior(prior_var_key=var_key),
                grouped_samples[sample_site])
            saved_sample_sites[var_key] = sample_site
    assert len(saved_sample_sites) == len(grouped_samples)
    sentinel_tmp = local_sentinel + '.tmp'
    with open(sentinel_tmp, 'wb') as ofile:
        pickle.dump((model_name, model_version_hash, model_seed_hash), ofile)
        pickle.dump(saved_sample_sites, ofile)
    os.replace(sentinel_tmp, local_sentinel)
    infer_key = (version_hash, seed_hash)
    _register_inference(
            model_name, model_version_hash, model_seed_hash,
            infer_key, local_sentinel, saved_sample_sites)

    return ((model_name, model_version_hash, model_seed_hash),
            saved_sample_sites)


def _compute_mcmc_submodel_with_lock(
        model_name: str,
        model_version_hash:str,
        model_seed_hash:str,
        mcmc_group: str,
        submodel: ScenarioModel,
        rng_key: ArrayLike,
        timeout: float,
        msg_prefix: str,
) -> tuple[tuple[str, str, str], dict[VarKey, str]]:
    version_hash = submodel.version_hash()
    seed_hash = _rng_key_hash(rng_key)
    lockpath = infer_lock_path(
            model_name, model_version_hash, model_seed_hash,
            mcmc_group, version_hash, seed_hash)
    lock = filelock.FileLock(lockpath, timeout=timeout)
    with lock:
        return _compute_mcmc_submodel(
                model_name=model_name,
                model_version_hash=model_version_hash,
                model_seed_hash=model_seed_hash,
                mcmc_group=mcmc_group,
                submodel=submodel,
                rng_key=rng_key,
                msg_prefix=msg_prefix)


def _compute_posterior(
        model: ScenarioModel,
        rng_key: ArrayLike,
        saved_sample_sites: dict[VarKey, str],
        saved_sample_models: dict[VarKey, tuple[str, str, str]],
        run_computation: bool = True,
) -> Computation:
    grouped_samples = {}
    for var_key, sample_site in saved_sample_sites.items():
        model_name, model_version_hash, model_seed_hash \
                = saved_sample_models[var_key]
        try:
            grouped_samples[sample_site] = load_var(
                    model_name=model_name,
                    model_version_hash=model_version_hash,
                    model_seed_hash=model_seed_hash,
                    var_key=GroupedPosterior(prior_var_key=var_key))
        except OSError as err:
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


_registry_mem_cache: dict[str, Computation] = {}


def registry_compute_model(
        model: ScenarioModel,
        model_name: str,
        cache_posterior: bool,
        seed: int,
        run_compute_posterior: bool = True,
        quiet: bool = True,
) -> Computation:
    if cache_posterior and model_name in _registry_mem_cache:
        return _registry_mem_cache[model_name]

    rng_key = jrandom.key(seed)

    model_version_hash = model.version_hash()
    model_seed_hash = _rng_key_hash(rng_key)

    model_root = infer_model_root(
            model_name=model_name,
            version_hash=model_version_hash,
            seed_hash=model_seed_hash)

    os.makedirs(model_root, exist_ok=True)
    os.makedirs(os.path.join(model_root, '.locks'), exist_ok=True)

    _scan_available_inferences(force=False)

    args_by_mcmc_group = {}
    submodels = model.mcmc_submodels()
    for mcmc_group, submodel in submodels.items():
        hasher = hashlib.sha256(f'{mcmc_group}_{seed}'.encode())
        mcmc_group_seed = int.from_bytes(hasher.digest()[:7], byteorder='big')
        args_by_mcmc_group[mcmc_group] = (
                model_name,
                model_version_hash,
                model_seed_hash,
                mcmc_group,
                submodel,
                jrandom.key(mcmc_group_seed))

    # mcmc_group -> VarKey -> saved_site
    saved_sites_d:dict[str, dict[VarKey, str]] = {}

    # mcmc_group -> (model_name, version_hash, seed_hash)
    saved_models_d:dict[str, tuple[str, str, str]] = {}

    n_groups_total = len(args_by_mcmc_group)
    n_groups_done = 0
    for passnum in range(10):
        n_timeouts = 0
        for mcmc_group, args in args_by_mcmc_group.items():
            if mcmc_group not in saved_sites_d:
                try:
                    if quiet:
                        msg_prefix = ''
                    else:
                        msg_prefix = f'{n_groups_done}/{n_groups_total}'
                    computing_model_info, submodel_saved_sample_sites \
                            = _compute_mcmc_submodel_with_lock(
                                *args,
                                msg_prefix=msg_prefix,
                                timeout=0)
                    saved_sites_d[mcmc_group] = submodel_saved_sample_sites
                    saved_models_d[mcmc_group] = computing_model_info
                    n_groups_done += 1
                except filelock.Timeout:
                    n_timeouts += 1
        if n_timeouts == 0:
            break

    for mcmc_group, args in args_by_mcmc_group.items():
        if mcmc_group not in saved_sites_d:
            computing_model_info, submodel_saved_sample_sites \
                    = _compute_mcmc_submodel_with_lock(
                        *args,
                        msg_prefix=f'{n_groups_done}/{n_groups_total}',
                        timeout=-1)
            saved_sites_d[mcmc_group] = submodel_saved_sample_sites
            saved_models_d[mcmc_group] = computing_model_info
            n_groups_done += 1

    assert n_groups_done == n_groups_total

    # key -> numpyro sample site name
    saved_sample_sites:dict[VarKey, str] = {}

    # key -> (model_name, version_hash, seed_hash)
    saved_sample_models:dict[VarKey, tuple[str, str, str]] = {}
    for mcmc_group, submodel_saved_sample_sites in saved_sites_d.items():
        for key, site in submodel_saved_sample_sites.items():
            assert key not in saved_sample_sites, key
            saved_sample_sites[key] = site
            saved_sample_models[key] = saved_models_d[mcmc_group]

    comp = _compute_posterior(
        model,
        rng_key=rng_key,
        saved_sample_sites=saved_sample_sites,
        saved_sample_models=saved_sample_models,
        run_computation=run_compute_posterior,
    )

    if cache_posterior:
        _registry_mem_cache[model_name] = comp
    return comp
