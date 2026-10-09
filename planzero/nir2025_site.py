import numpy as np

from . import nir2025
from .enums import GHG, PT, IPCC_Sector
from .nir2025_model import NIR2025_ModelElement, NIR2025_ScenarioModel
from .prob import ClassVar, SiteInference
from .regional_total_emissions_element import PseudoRegion, RegionalTotalEmissionsVarKey
from .scenario_model import ScenarioModel, registry_compute_model
from .sector_total_emissions_element import PseudoSector, SectorTotalEmissionsVarKey
from .sparkline_echart_helper import (
    national_emissions_by_sector_echart_from_sample,
    sectoral_emissions_by_region_echart_from_sample,
)


def sector_total_emissions_var_key(sector:IPCC_Sector|PseudoSector):
    return SectorTotalEmissionsVarKey(
            sector=sector,
            elem_id="SectorTotalEmissionsElement",
            name="")


def regional_total_emissions_var_key(
        sector:IPCC_Sector,
        pt):
    return RegionalTotalEmissionsVarKey(
            sector=sector,
            pt=pt,
            elem_id="RegionalTotalEmissionsElement",
            name="")


# used in sparkline_echart_helper
def nir2025_ca_quantile_bounds(sector:IPCC_Sector|PseudoSector, q):
    comp = NIR2025().comp
    var_key = sector_total_emissions_var_key(sector)
    sample = comp.ndarray_d[var_key]
    qvals = np.quantile(sample, q=q, axis=1)
    rval = {q_ii: qvals_ii for q_ii, qvals_ii in zip(q, qvals)}
    return rval


def nir2025_sample_by_pt(sector:IPCC_Sector, ghg:GHG|None):
    comp = NIR2025().comp
    if ghg is None:
        ktCO2e_sample = {pt: comp.ndarray_d[
            regional_total_emissions_var_key(sector, pt)]
                         for pt in PT if pt != PT.XX}
        ktCO2e_sample[PseudoRegion.NationalTotal] = comp.ndarray_d[
                regional_total_emissions_var_key(
                    sector,
                    pt=PseudoRegion.NationalTotal)]
    else:
        elem_id = NIR2025_ModelElement.element_identifier(sector, ghg, n_years=34)
        ktCO2e_sample = {
                pt: comp.ndarray_d[var_key]
                for pt, var_key in comp.model.model_elements[elem_id].pt_keys.items()}
        ktCO2e_sample[PseudoRegion.NationalTotal] = comp.ndarray_d[
                comp.model.model_elements[elem_id].ca_key]
    return ktCO2e_sample


def nir2025_pt_quantile_bounds(sector:IPCC_Sector, ghg:GHG|None, q) -> dict[PT|PseudoRegion, dict[float, np.ndarray]]:
    ktCO2e_sample = nir2025_sample_by_pt(sector, ghg)
    regions = [pt for pt in PT if pt != PT.XX] + [PseudoRegion.NationalTotal]
    rval = {pt: {
        qi: qval for qi, qval in zip(q, np.quantile(ktCO2e_sample[pt], q=q, axis=1))}
            for pt in regions}
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
        ktCO2e_sample = {sector: comp.ndarray_d[
            sector_total_emissions_var_key(sector)]
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
        near_zeros = nir2025.near_zero_sector_ghgs()
        rval = [ghg for ghg in GHG if (sector, ghg) not in near_zeros]
        return rval

    def sector_echart(self, sector, ghg:GHG|None, v_unit):
        ktCO2e_sample = nir2025_sample_by_pt(sector, ghg)
        return sectoral_emissions_by_region_echart_from_sample(
                ktCO2e_sample=ktCO2e_sample,
                year_0=1990,
                n_years=34,
                n_samples=self.comp.n_mcmc,
                sector=sector,
                ghg=ghg,
                div_id=f'{self.__class__.__name__}_regional_sparkline_echart_{ghg.value if ghg else "all"}',
                v_unit=v_unit)

    @property
    def predicted_emissions_2050_MtCO2e_bounds_str(self) -> str:
        return "N/A"
