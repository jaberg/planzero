import unittest

import jax

from .scaling_site import ScalingSiteInference
from .test_site_inference import SiteInferenceSmokeTester


def test_memory_usage_baseline():
    site_model = ScalingSiteInference()

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

    assert total_bytes < 2000 * 1_000_000


class TestScalingSiteInference(unittest.TestCase, SiteInferenceSmokeTester):
    def setUp(self):
        self.siteinf = ScalingSiteInference()
