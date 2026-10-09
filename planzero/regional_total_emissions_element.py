import enum
import warnings

import jax.numpy as jnp

from .enums import GHG, PT, IPCC_Sector
from .scenario_model import (
    AnnualKey,
    GeneralKey,
    ModelElement,
    Phase,
    Workspace,
    define,
    years_dim,
)


# TODO: if enums.Regions becomes a thing,
# then no need for this anymore
class PseudoRegion(str, enum.Enum):
    NationalTotal = 'National Total'


class RegionalTotalEmissionsVarKey(GeneralKey, frozen=True):

    sector: IPCC_Sector
    pt: PT | PseudoRegion


class RegionalTotalEmissionsElement(ModelElement):
    """
    Sum over GHG for each region (for a given sector),
    """

    sector_region_ghg_subtotal_keys: dict[
            tuple[IPCC_Sector, PT],
            dict[GHG, GeneralKey|AnnualKey]]

    def __init__(
            self,
            sector_region_ghg_subtotal_keys: dict[
                tuple[IPCC_Sector, PT],
                dict[GHG, GeneralKey|AnnualKey]],
            ):
        self.sector_region_ghg_subtotal_keys = sector_region_ghg_subtotal_keys

    @property
    def sector_region_keys(self) -> dict[
            tuple[IPCC_Sector, PT|PseudoRegion], RegionalTotalEmissionsVarKey]:
        rval:dict[tuple[IPCC_Sector, PT|PseudoRegion], RegionalTotalEmissionsVarKey] = {}
        rval.update({(sector, pt): RegionalTotalEmissionsVarKey(
                    elem_id=self.identifier, name='',
                    sector=sector, pt=pt)
                for sector in IPCC_Sector
                for pt in PT if pt != PT.XX})
        rval.update({
            (sector, PseudoRegion.NationalTotal): RegionalTotalEmissionsVarKey(
                elem_id=self.identifier, name='',
                sector=sector, pt=PseudoRegion.NationalTotal)
            for sector in IPCC_Sector})
        return rval


    @define(sector_region_keys, prior_shape=[years_dim])
    def model_element_postprocess(self, ws:Workspace):
        if ws.phase == Phase.Prior:
            # don't access the values to avoid sampling and slowing down MCMC
            pass
        elif ws.phase == Phase.Posterior:
            n_skipped = 0
            for sector in IPCC_Sector:
                national_total = jnp.zeros((ws.n_years,) + ws.mcmc_shape)
                for pt in PT:
                    if pt == PT.XX:
                        continue

                    try:
                        key_by_ghg = self.sector_region_ghg_subtotal_keys[sector, pt]
                    except KeyError:
                        n_skipped += 1
                        continue
                    kt_CO2e = jnp.zeros((ws.n_years,) + ws.mcmc_shape)
                    for var_key in key_by_ghg.values():
                        kt_CO2e += ws.val.general[var_key]

                    out_key = self.sector_region_keys[sector, pt]
                    ws.val.general[out_key] = kt_CO2e
                    national_total += kt_CO2e

                out_key = self.sector_region_keys[sector, PseudoRegion.NationalTotal]
                ws.val.general[out_key] = national_total
            if n_skipped:
                warnings.warn(
                        'regional_total_emissions_element skipped'
                        f' {n_skipped} sector-pt combinations')
        else:
            raise NotImplementedError(ws.phase)
