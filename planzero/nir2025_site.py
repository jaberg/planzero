try:
    import jax.random as jrandom
except ImportError:
    pass
import numpy as np

from . import nir2025  # as the model being rendered by this code
from . import nir2025 as site_nir  # as the reference model for the site
from .enums import GHG, PT, IPCC_Sector, LULUCF_Sectors
from .nir2025_model import NIR2025_ScenarioModel
from .nir_ar2_site import (
    RegionalSparklineEChartHelper,  # TODO: rename e.g. TimeVaryingSparklineEChartHelper
)
from .prob import ClassVar, SiteInference
from .scenario_model import ScenarioModel, registry_compute_model
from .sector_total_emissions_element import PseudoSector, SectorTotalEmissionsVarKey
from .sparkline_echart_helper import (
    national_emissions_by_sector_echart_from_sample,
)

n_samples = 250 # enough to do the job, not too slowly
n_years = 34
n_regions = 13



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


# used in sparkline_echart_helper
def nir2025_ca_quantile_bounds(sector:IPCC_Sector|PseudoSector, q):
    comp = NIR2025().comp
    var_key = SectorTotalEmissionsVarKey(sector=sector)
    sample = comp.storage_nd[var_key]
    qvals = np.quantile(sample, q=q, axis=0)
    rval = {q_ii: qvals_ii for q_ii, qvals_ii in zip(q, qvals)}
    return rval


class NIR2025(SiteInference):
    """Emissions per province and territory,
    and per greenhouse gas, are taken from the 2025 National Inventory Report data,
    incorporating uncertainty estimates from Annex 2 of the report.
    """

    include_in_registry: ClassVar[bool] = True

    def one_line_description(self):
        return "Emissions with uncertainty from NIR-2025"

    @property
    def scenario_model(self) -> ScenarioModel:
        return NIR2025_ScenarioModel(last_observed_year=2023)

    @property
    def comp(self):
        return registry_compute_model(
                model=self.scenario_model,
                model_name="NIR2025",
                cache_posterior=True,
                seed_or_key=42)

    def uncertain_sparkline_matrix_echart(self, div_id, v_unit):
        # TODO: make it work, then move to base class
        comp = self.comp
        ktCO2e_sample = {sector: comp.storage_nd[
            SectorTotalEmissionsVarKey(sector=sector)]
                         for sector in IPCC_Sector}
        rval = national_emissions_by_sector_echart_from_sample(
                ktCO2e_sample=ktCO2e_sample,
                year_0=comp.model.mv.year_0,
                n_years=comp.n_years,
                n_samples=comp.n_mcmc,
                div_id=div_id,
                v_unit=v_unit,
                model_name=self.__class__.__name__,
                )
        return rval

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

    @property
    def predicted_emissions_2050_MtCO2e_bounds_str(self) -> str:
        return "N/A"
