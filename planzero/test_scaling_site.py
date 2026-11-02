import datetime
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

    def test_predicted_emissions(self):
        low, high = self.siteinf.scenario_model.predicted_emissions_2050_MtCO2e_bounds_95(
                self.siteinf.comp)
        assert 600 < low < high < 1000

    def test_bovaer_cost_per_ton(self):
        ll, uu = self.sitinf.cost_per_tCO2e('Scale_Bovaer_2030')
        assert 185 < ll < uu < 220

    def test_prediction_scores_prenir_2025_m04():
        reg = completed_prob_registry()
        scaling = reg['ScalingStudy_All_Strategies']
        prenir_results = scaling.prediction_scores_prenir_2025_m04()
        weighted_divergence = prenir_results['weighted_divergence']
        print(weighted_divergence)
        assert 3.3 < weighted_divergence < 3.4, weighted_divergence

        data_cutoff = datetime.date(year=2024, month=12, day=31)
        model_id = model_id_from_data_cutoff(data_cutoff)
        t0 = time.time()
        weighted_divergence_, _, _ = weighted_KL_score(
                year=2023,
                model_id=model_id,
                )
        t1 = time.time()
        print(weighted_divergence_, (t1 - t0))
        assert abs(weighted_divergence - weighted_divergence_) < .1


    def test_netzero_2050():
        reg = completed_prob_registry()

        scaling = reg['ScalingStudy_minus_Scale_Bovaer']
        p_lt_0 = scaling.challenge_netzero_2050()
        print(p_lt_0)

        static_est = p_emissions_below_thresh_ex_LULUCF(
            model_id=model_id_from_data_cutoff(
                data_cutoff=datetime.date(year=2024, month=12, day=31)),
            thresh_ktCO2e=0)
        print(static_est)

        assert p_lt_0 * .9 < static_est < 1e-48
        assert p_lt_0 * 1.1 > static_est
