import jax.numpy as jnp
import numpy as np

from .enums import GHG, PT, IPCC_Sector
from .nir2025 import near_zero_sector_ghgs
from .p_emissions_below_thresh import PEmissionsBelowThresh
from .prenir_element import PreNIR_ModelElement
from .regional_total_emissions_element import (
    PseudoRegion,
    RegionalTotalEmissionsElement,  # noqa: F401
    RegionalTotalEmissionsVarKey,
)
from .scenario_model import AnnualKey, Computation, GeneralKey, ScenarioModel, years_key
from .sector_total_emissions_element import (
    PseudoSector,
    SectorTotalEmissionsElement,  # noqa: F401
    SectorTotalEmissionsVarKey,
)


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
            ) -> AnnualKey|GeneralKey:
        raise NotImplementedError()

    def ktCO2e_var_key_sector_region(
            self,
            sector:IPCC_Sector,
            region:PT|PseudoRegion,
            ) -> AnnualKey|GeneralKey:
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

    def predicted_emissions_2050_MtCO2e_bounds_95(
            self,
            comp:Computation,
            ) -> tuple[float, float]:
        ktCO2e_sample = comp.ndarray_d[
                sector_total_emissions_var_key(PseudoSector.Total_with_LULUCF)]
        model_years = np.asarray(comp.ndarray_d[years_key])
        match_mask = (model_years == 2050)
        if np.any(match_mask):
            model_idx_of_prediction_year = np.argmax(match_mask)
        else:
            raise NotImplementedError('no prediction for 2050')
        ktCO2e_2050 = ktCO2e_sample[model_idx_of_prediction_year]
        n_mcmc, = ktCO2e_2050.shape
        assert n_mcmc > 100
        unit_conversion = 1 / 1000
        lower, upper = np.quantile(ktCO2e_2050, [0.025, 0.975]) * unit_conversion
        return lower, upper


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

    def add_element_prenir_2025_m04(self, **kwargs):
        self.add_element(
                PreNIR_ModelElement(
                    nir_year=2025,
                    months_ahead=4,
                    **kwargs),
                mcmc_group=None)


    # P-Emissions-Below-Thresh
    def p_net_zero_2050(self, comp:Computation) -> float:
        elem = PEmissionsBelowThresh(
            prediction_year=2050,
            thresh_ktCO2e=0.0,
            ktCO2e_normal_keys_by_sector_ghg={})
        pnz = comp.ndarray_d[elem.prob_key]
        assert pnz.shape == (1,)
        return float(pnz[0])

    def add_element_p_net_zero_2050(self, **kwargs):
        self.add_element(
                PEmissionsBelowThresh(
                    prediction_year=2050,
                    thresh_ktCO2e=0.0,
                    **kwargs),
                mcmc_group=None)
