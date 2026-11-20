import unittest

from .scaling_site import ScalingSiteInference
from .test_site_inference import SiteInferenceSmokeTester


class TestScalingSiteInference(unittest.TestCase, SiteInferenceSmokeTester):
    def setUp(self):
        self.siteinf = ScalingSiteInference()
        self.summary_stats = self.siteinf.site_model_summary_stats

    def test_predicted_emissions(self):
        stats = self.summary_stats.predicted_emissions_2050_ktCO2e
        assert 600_000 < stats.quantiles[0.025] < stats.quantiles[0.975] < 1000_000

    def test_bovaer_cost_per_ton(self):
        ll, uu = self.siteinf.cost_per_tCO2e('Scale_Bovaer_2030')
        assert 185 < ll < uu < 220

    def test_prediction_scores_prenir_2025_m04(self):
        weighted_divergence = self.summary_stats.prenir_2025_m04.weighted_divergence
        print(weighted_divergence)
        assert (3.2 < weighted_divergence < 3.4), weighted_divergence

    def test_netzero_2050(self):
        p_lt_0 = self.summary_stats.p_net_zero_2050
        print(p_lt_0)
        assert p_lt_0 == 0
