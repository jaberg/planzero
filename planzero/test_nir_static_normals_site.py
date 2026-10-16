import jax
import numpy as np

from .enums import GHG, IPCC_Sector
from .nir2025 import near_zero_sector_ghgs
from .nir_static_normals_site import Static_Normals_2024_12_31


def test_weighted_divergence():
    site_model = Static_Normals_2024_12_31()
    prenir = site_model.scenario_model.prediction_scores_prenir_2025_m04(
            comp=site_model.comp)
    print(prenir)
    near_zero = near_zero_sector_ghgs()
    assert np.isfinite(prenir['weighted_divergence'])
    wd = float(prenir['weighted_divergence'])
    assert 4.1 < wd < 4.3, wd
    for sector in IPCC_Sector:
        for ghg in GHG:
            if (sector, ghg) not in near_zero:
                assert 0 <= float(prenir['scores_ca'][sector][ghg]) <= 15, (
                        sector, ghg)


def test_memory_usage():
    site_model = Static_Normals_2024_12_31()

    total_bytes = 0
    bytes_used = []
    for key, nd in site_model.comp.ndarray_d.items_no_sample_if_necessary():
        total_bytes += nd.nbytes
        bytes_used.append((nd.nbytes, 'ndarray', key))

    for key, dist in site_model.comp.dist_d.items():
        leaves = jax.tree_util.tree_leaves(dist)
        leaves_bytes = sum(getattr(leaf, "nbytes", 0) for leaf in leaves)
        total_bytes += leaves_bytes
        bytes_used.append((leaves_bytes, 'dist', key))

    print('MB', total_bytes / 1_000_000)
    for ii, nbytes_key in enumerate(sorted(bytes_used, reverse=True)):
        print(ii, nbytes_key)
        if ii > 10:
            break

    # this limit is not specifically important, but my laptop may crash
    # if its' much bigger and I'm trying to run unit tests with web server loaded.
    assert total_bytes < 2500 * 1_000_000


def test_p_net_zero():
    site_model = Static_Normals_2024_12_31()
    p_net_zero = site_model.scenario_model.p_net_zero_2050(site_model.comp)

    assert 1e-24 < p_net_zero < 1e-22
