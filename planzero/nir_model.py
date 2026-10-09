import jax.numpy as jnp

from .enums import GHG, PT, IPCC_Sector
from .regional_total_emissions_element import PseudoRegion, RegionalTotalEmissionsVarKey
from .scenario_model import Computation, GeneralKey, ScenarioModel
from .sector_total_emissions_element import PseudoSector, SectorTotalEmissionsVarKey


def sector_total_emissions_var_key(
        sector:IPCC_Sector|PseudoSector,
        ) -> SectorTotalEmissionsVarKey:
    return SectorTotalEmissionsVarKey(
            sector=sector,
            elem_id="SectorTotalEmissionsElement",
            name="")


def regional_total_emissions_var_key(
        sector:IPCC_Sector,
        region:PT|PseudoRegion,
        ) -> RegionalTotalEmissionsVarKey:
    return RegionalTotalEmissionsVarKey(
            sector=sector,
            pt=region,
            elem_id="RegionalTotalEmissionsElement",
            name="")


class NIR_ScenarioModel(ScenarioModel):

    def ktCO2e_var_key_sector_ghg_region(
            self,
            sector:IPCC_Sector,
            ghg:GHG,
            pt:PT|PseudoRegion,
            ) -> GeneralKey:
        raise NotImplementedError()

    def ktCO2e_var_key_sector_region(
            self,
            sector:IPCC_Sector,
            region:PT|PseudoRegion,
            ) -> GeneralKey:
        return regional_total_emissions_var_key(sector, region)

    def ktCO2e_var_key_sector(
            self,
            sector:IPCC_Sector|PseudoSector,
            ) -> GeneralKey:
        return sector_total_emissions_var_key(sector)

    def all_sector_echart_ktCO2e_sample(
            self,
            comp:Computation,
            ) -> dict[IPCC_Sector|PseudoSector, jnp.ndarray]:
        ktCO2e_sample = {}
        for sector in IPCC_Sector:
            ktCO2e_sample[sector] = comp.ndarray_d[
                    sector_total_emissions_var_key(sector)]
        for sector in PseudoSector:
            ktCO2e_sample[sector] = comp.ndarray_d[
                    sector_total_emissions_var_key(sector)]
        return ktCO2e_sample

    def per_sector_echart_ktCO2e_sample(
            self,
            comp:Computation,
            sector, ghg:GHG|None
            ) -> dict[PT|PseudoRegion, jnp.ndarray]:
        ktCO2e_sample:dict[PT|PseudoRegion, jnp.ndarray] = {}
        regions = [pt for pt in PT if pt != PT.XX] + [PseudoRegion.NationalTotal]
        if ghg is None:
            ktCO2e_sample.update({
                    region: comp.ndarray_d[
                        self.ktCO2e_var_key_sector_region(sector, region)]
                    for region in regions})
        else:
            ktCO2e_sample.update({
                    region: comp.ndarray_d[
                        self.ktCO2e_var_key_sector_ghg_region(sector, ghg, region)]
                    for region in regions})
        return ktCO2e_sample
