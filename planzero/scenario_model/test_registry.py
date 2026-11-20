import os
import pickle

import jax.random as jrandom

from . import registry


def test_family_basename_and_model_root(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, 'MODEL_REGISTRY_ROOT', str(tmp_path))
    version_hash = 'a' * 32
    seed_hash = 'b' * 32

    family_basename = registry.infer_model_family_basename(
            'NIR2025', version_hash, seed_hash)
    assert family_basename == f'model_family_NIR2025_{version_hash}_{seed_hash}'

    base_root = registry.infer_model_root('NIR2025', version_hash, seed_hash)
    assert base_root == (
            f'{tmp_path}/{family_basename}'
            f'/model_NIR2025_{version_hash}_{seed_hash}')

    ablation_version_hash = 'c' * 32
    ablation_root = registry.infer_model_root(
            'NIR2025_Some_Strategy',
            ablation_version_hash,
            seed_hash,
            family_basename=family_basename)
    assert os.path.dirname(ablation_root) == os.path.dirname(base_root)
    assert os.path.basename(ablation_root) == (
            f'model_NIR2025_Some_Strategy_'
            f'{ablation_version_hash}_{seed_hash}')


def test_iter_model_roots_scans_family_subdirectories(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, 'MODEL_REGISTRY_ROOT', str(tmp_path))
    version_hash = 'a' * 32
    seed_hash = 'b' * 32

    base_root = registry.infer_model_root('NIR2025', version_hash, seed_hash)
    os.makedirs(base_root)

    # A flat model directory is not part of the current layout.
    legacy_root = tmp_path / f'model_legacy_{version_hash}_{seed_hash}'
    os.makedirs(legacy_root)

    assert list(registry._iter_model_roots()) == [base_root]


def test_scan_finds_nested_sentinels(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, 'MODEL_REGISTRY_ROOT', str(tmp_path))
    monkeypatch.setattr(registry, '_infer_registry', {})
    monkeypatch.setattr(registry, '_registry_scanned_mtime', None)

    version_hash = 'a' * 32
    seed_hash = 'b' * 32
    model_root = registry.infer_model_root('NIR2025', version_hash, seed_hash)
    os.makedirs(model_root)

    submodel_version_hash = 'c' * 32
    submodel_seed_hash = 'd' * 32
    sentinel_path = registry.infer_sentinel_path(
            model_root, 'mcmc_group', submodel_version_hash, submodel_seed_hash)
    with open(sentinel_path, 'wb') as f:
        pickle.dump(('NIR2025', version_hash, seed_hash), f)
        pickle.dump({}, f)

    found = registry._scan_available_inferences(force=True)
    avail = found[(submodel_version_hash, submodel_seed_hash)]
    assert avail.model_root_path == model_root
    assert avail.saved_sample_sites == {}


def test_registry_basename_is_family_basename(monkeypatch):
    monkeypatch.setattr(registry, 'MODEL_REGISTRY_ROOT', '/tmp/registry')
    version_hash = 'a' * 32

    class Model:
        def version_hash(self):
            return version_hash

    result = registry.registry_basename(Model(), 'NIR2025', seed=42)
    seed_hash = registry._rng_key_hash(jrandom.key(42))
    assert result == f'model_family_NIR2025_{version_hash}_{seed_hash}'
