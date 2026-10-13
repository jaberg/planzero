import jax.numpy as jnp
import numpy as np

from .nir2025 import near_zero_sector_ghgs
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

    # PRE-NIR

    def prenir_2025_m04_total_key(self) -> GeneralKey:
        raise NotImplementedError()

    def prenir_2025_m04_sector_ghg_key(self, sector:IPCC_Sector, ghg:GHG) -> GeneralKey:
        raise NotImplementedError()

    def prediction_scores_prenir_2025_m04(self, comp:Computation):
        scores_ca = {}
        near_zero = near_zero_sector_ghgs()
        for sector in IPCC_Sector:
            scores_ca[sector] = {}
            for ghg in GHG:
                if (sector, ghg) not in near_zero:
                    sector_ghg_key = self.prenir_2025_m04_sector_ghg_key(sector, ghg)
                    sector_ghg_val = np.asarray(comp.ndarray_d[sector_ghg_key])
                    # remove the mcmc dim
                    assert sector_ghg_val.shape == (1,)
                    scores_ca[sector][ghg] = float(sector_ghg_val[0])
        total_key = self.prenir_2025_m04_total_key()
        total_val = comp.ndarray_d[total_key]
        assert total_val.shape == (1,)
        rval = {
                'weighted_divergence': float(total_val[0]),
                'scores_ca': scores_ca,
                }
        return rval
