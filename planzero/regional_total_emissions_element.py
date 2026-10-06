import enum

from .enums import GHG, PT, IPCC_Sector
from .scenario_model import (
    ModelElement,
    VarKey,
    VarKeyBase,
    WorkSpace_Proc,
    define,
    years_dim,
)


# TODO: if enums.Regions becomes a thing,
# then no need for this anymore
class PseudoRegion(str, enum.Enum):
    NationalTotal = 'National Total'


class RegionalTotalEmissionsVarKey(VarKeyBase, frozen=True):

    sector: IPCC_Sector
    pt: PT | PseudoRegion


class RegionalTotalEmissionsElement(ModelElement):
    """
    Sum over GHG for each region (for a given sector),
    """

    sector_region_ghg_subtotal_keys: dict[
            tuple[IPCC_Sector, PT],
            dict[GHG, VarKey]]

    def __init__(
            self,
            sector_region_ghg_subtotal_keys: dict[
                tuple[IPCC_Sector, PT],
                dict[GHG, VarKey]],
            ):
        self.sector_region_ghg_subtotal_keys = sector_region_ghg_subtotal_keys

    @property
    def sector_region_keys(self) -> dict[
            tuple[IPCC_Sector, PT|PseudoRegion], RegionalTotalEmissionsVarKey]:
        rval:dict[tuple[IPCC_Sector, PT|PseudoRegion], RegionalTotalEmissionsVarKey] = {}
        rval.update({(sector, pt): RegionalTotalEmissionsVarKey(
                    sector=sector, pt=pt)
                for sector in IPCC_Sector
                for pt in PT if pt != PT.XX})
        rval.update({
            (sector, PseudoRegion.NationalTotal): RegionalTotalEmissionsVarKey(
                sector=sector, pt=PseudoRegion.NationalTotal)
            for sector in IPCC_Sector})
        return rval


    @define(sector_region_keys, prior_shape=[years_dim])
    def model_element_postprocess_posterior_only(self, ws:WorkSpace_Proc):
        for sector in IPCC_Sector:
            national_total = 0
            for pt in PT:
                if pt == PT.XX: continue

                key_by_ghg = self.sector_region_ghg_subtotal_keys[sector, pt]
                kt_CO2e = 0
                for var_key in key_by_ghg.values():
                    kt_CO2e += ws.val.general[var_key]

                out_key = self.sector_region_keys[sector, pt]
                ws.val.general[out_key] = kt_CO2e
                national_total += kt_CO2e

            out_key = self.sector_region_keys[sector, PseudoRegion.NationalTotal]
            ws.val.general[out_key] = national_total
