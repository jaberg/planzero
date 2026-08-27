try:
    import jax.random as jrandom
except ImportError:
    pass
import numpy as np
from pydantic import computed_field

from . import nir2025  # as the model being rendered by this code
from . import nir2025 as site_nir  # as the reference model for the site
from .enums import GHG, PT, IPCC_Sector, LULUCF_Sectors
from .nir_ar2_site import (
    RegionalSparklineEChartHelper,  # TODO: rename e.g. TimeVaryingSparklineEChartHelper
    SparklineEChartHelper,  # TODO: rename e.g. TimeVaryingSparklineEChartHelper
)
from .prob import ClassVar, SiteInference

n_samples = 250 # enough to do the job, not too slowly
n_years = 34
n_regions = 13


class NIR2025_SparklineEChartHelper(SparklineEChartHelper):

    def load_data(self):
        self.years = np.arange(1990, 2023 + 1)

        mean_with_lulucf = 0
        estimates_with_lulucf = np.zeros(
            (n_samples, n_years))

        mean_without_lulucf = 0
        estimates_without_lulucf = np.zeros(
            (n_samples, n_years))

        rng_key = jrandom.key(1234)

        near_zeros = site_nir.near_zero_sector_ghgs()

        for sector in IPCC_Sector:

            estimated_sector_total_ca = np.zeros(
                (n_samples, n_years,))

            for ii, ghg in enumerate(GHG):
                if (sector, ghg) in near_zeros:
                    continue

                for jj, year in enumerate(self.years):
                    ca_dist, _ = nir2025.ktCO2e_numpyro_dist_pt_ca(
                        sector, ghg, year)

                    rng_key, rng_key_ = jrandom.split(rng_key)
                    estimated_sector_ghg_ca_year = ca_dist.sample(rng_key_, (n_samples,))
                    estimated_sector_total_ca[:, jj] += estimated_sector_ghg_ca_year * self.v_unit_scale

            mean_sector_total = self.compute_stats_and_add_data_for_sector(
                sector,
                estimated_sector_total_ca)

            if sector not in LULUCF_Sectors:
                estimates_without_lulucf += estimated_sector_total_ca
                mean_without_lulucf += mean_sector_total
            estimates_with_lulucf += estimated_sector_total_ca
            mean_with_lulucf += mean_sector_total

        self.add_data_for_LULUCF_totals(
            estimates_with_lulucf,
            mean_with_lulucf,
            estimates_without_lulucf,
            mean_without_lulucf)


class NIR2025_RegionalSparklineEChartHelper(RegionalSparklineEChartHelper):

    def load_data(self):
        self.years = np.arange(1990, 2023 + 1)

        estimates_ghg_pt = np.zeros(
            (n_samples, len(GHG), n_years, n_regions))

        estimates_ghg_ca = np.zeros(
            (n_samples, len(GHG), n_years))

        rng_key = jrandom.key(1234)

        near_zeros = site_nir.near_zero_sector_ghgs()

        for ii, ghg in enumerate(GHG):
            if (self.sector, ghg) in near_zeros:
                estimates_ghg_pt[:, ii] = 0
                estimates_ghg_ca[:, ii] = 0
            elif self.ghg is not None and self.ghg != ghg:
                estimates_ghg_pt[:, ii] = 0
                estimates_ghg_ca[:, ii] = 0
            else:
                for jj, year in enumerate(self.years):
                    ca_dist, pt_dists = nir2025.ktCO2e_numpyro_dist_pt_ca(
                        self.sector, ghg, year)

                    rng_key, rng_key_ = jrandom.split(rng_key)
                    ca_sample = ca_dist.sample(rng_key_, (n_samples,))
                    estimates_ghg_ca[:, ii, jj] = ca_sample * self.v_unit_scale
                    for kk, pt in enumerate(PT):
                        if pt == PT.XX:
                            continue
                        rng_key, rng_key_ = jrandom.split(rng_key)
                        pt_sample = pt_dists[kk].sample(rng_key_, (n_samples,))
                        estimates_ghg_pt[:, ii, jj, kk] = pt_sample * self.v_unit_scale

        # sum across GHGs
        estimates_pt = estimates_ghg_pt.sum(axis=1)
        estimates_ca = estimates_ghg_ca.sum(axis=1)

        self.add_data_from_estimates(
            estimates_pt=estimates_pt,
            estimates_ca=estimates_ca)


class NIR2025(SiteInference):
    """Emissions per province and territory,
    and per greenhouse gas, are taken from the 2025 National Inventory Report data,
    incorporating uncertainty estimates from Annex 2 of the report.
    """

    include_in_registry: ClassVar[bool] = True

    def one_line_description(self):
        return "Data with uncertainty from NIR-2025"

    @computed_field
    def predicted_emissions_2050_MtCO2e_bounds_ul(self) -> tuple[float, float]:
        return (float('nan'), float('nan'))

    def uncertain_sparkline_matrix_echart(self, div_id, v_unit):
        helper = NIR2025_SparklineEChartHelper(div_id, v_unit, model_name=self.__class__.__name__)
        helper.load_data()
        helper.order_sectors()
        helper.add_total_cells()
        helper.add_non_lulucf_cells()
        helper.add_lulucf_cells()
        return helper.make_echart()

    def GHGs_for_sector(self, sector):
        near_zeros = site_nir.near_zero_sector_ghgs()
        rval = [ghg for ghg in GHG if (sector, ghg) not in near_zeros]
        return rval

    def sector_echart(self, sector, ghg, v_unit):
        helper = NIR2025_RegionalSparklineEChartHelper(
            sector=sector,
            ghg=ghg,
            div_id=f'{self.__class__.__name__}_regional_sparkline_echart_{ghg.value if ghg else "all"}',
            v_unit=v_unit)
        helper.load_data()
        helper.order_regions()
        helper.add_regional_cells()
        return helper.make_echart()
