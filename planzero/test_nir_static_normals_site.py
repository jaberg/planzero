import unittest

import numpy as np

from .enums import GHG, IPCC_Sector
from .nir2025 import near_zero_sector_ghgs
from .nir_static_normals_site import Static_Normals_2024_12_31
from .test_nir2025_site import SiteInferenceSmokeTester


class TestSiteInference(unittest.TestCase, SiteInferenceSmokeTester):
    def setUp(self):
        self.siteinf = Static_Normals_2024_12_31()
        self.summary_stats = self.siteinf.site_model_summary_stats

    def test_p_net_zero(self):
        assert self.summary_stats.p_net_zero_2050 == 0

    def test_predicted_emissions(self):
        stats = self.summary_stats.predicted_emissions_2050_ktCO2e
        assert 600_000 < stats.quantiles[.025] < stats.quantiles[0.975] < 1000_0000

    def test_weighted_divergence(self):
        prenir_2025_m04 = self.summary_stats.prenir_2025_m04
        near_zero = near_zero_sector_ghgs()
        assert np.isfinite(prenir_2025_m04.weighted_divergence)
        wd = float(prenir_2025_m04.weighted_divergence)
        assert 3.2 < wd < 3.4, wd
        for sector in IPCC_Sector:
            for ghg in GHG:
                if (sector, ghg) not in near_zero:
                    score = float(prenir_2025_m04.scores_ca[sector][ghg])
                    assert 0 <= score <= 15, (sector, ghg)
