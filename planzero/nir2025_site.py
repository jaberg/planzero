import numpy as np

from .enums import GHG, PT, IPCC_Sector
from .nir2025_model import NIR2025_ScenarioModel
from .prob import SiteInference as SiteModel
from .regional_total_emissions_element import PseudoRegion
from .sector_total_emissions_element import PseudoSector


# used in sparkline_echart_helper
def nir2025_ca_quantile_bounds(sector:IPCC_Sector|PseudoSector, q):
    site_model = NIR2025()
    comp = site_model.comp
    var_key = site_model.scenario_model.ktCO2e_var_key_sector(sector)
    sample = comp.ndarray_d[var_key]
    qvals = np.quantile(sample, q=q, axis=1)
    rval = {q_ii: qvals_ii for q_ii, qvals_ii in zip(q, qvals)}
    return rval


# used in sparkline_echart_helper
def nir2025_pt_quantile_bounds(sector:IPCC_Sector, ghg:GHG|None, q) -> dict[PT|PseudoRegion, dict[float, np.ndarray]]:
    site_model = NIR2025()
    comp = site_model.comp
    regions = [pt for pt in PT if pt != PT.XX] + [PseudoRegion.NationalTotal]
    rval = {}
    for region in regions:
        if ghg is None:
            var_key = site_model.scenario_model.ktCO2e_var_key_sector_region(
                    sector, region)
        else:
            var_key = site_model.scenario_model.ktCO2e_var_key_sector_ghg_region(
                    sector, ghg, region)
        ktCO2e = comp.ndarray_d[var_key]
        rval[region] = dict(zip(q, np.quantile(ktCO2e, q=q, axis=1)))
    return rval


class NIR2025(SiteModel):
    """Emissions per province and territory,
    and per greenhouse gas, are taken from the 2025 National Inventory Report data,
    incorporating uncertainty estimates from Annex 2 of the report.
    """

    include_in_registry:bool = True

    def one_line_description(self) -> str:
        return "Emissions with uncertainty from NIR-2025"

    def __init__(self):
        super().__init__(
                scenario_model_registry_name='NIR2025',
                scenario_model=NIR2025_ScenarioModel(last_observed_year=2023),
                )

    @property
    def predicted_emissions_2050_MtCO2e_bounds_str(self) -> str:
        return "N/A"
