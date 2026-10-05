import enum

from .enums import GHG, IPCC_Sector, LULUCF_Sectors
from .scenario_model import (
    ModelElement,
    VarKey,
    VarKeyBase,
    WorkSpace_Proc,
    define,
    years_dim,
)


class PseudoSector(str, enum.Enum):

    Total_with_LULUCF = 'Total with LULUCF'
    Total_without_LULUCF = 'Total without LULUCF'


class SectorTotalEmissionsVarKey(VarKeyBase, frozen=True):

    sector: IPCC_Sector | PseudoSector


class SectorTotalEmissionsElement(ModelElement):
    """
    Sum over GHG for each sector,
    and define national totals with and without LULUCF sectors
    """

    sector_ghg_subtotal_keys: dict[IPCC_Sector, dict[GHG, VarKey]]

    def __init__(
            self,
            sector_ghg_subtotal_keys: dict[IPCC_Sector, dict[GHG, VarKey]],
            ):
        self.sector_ghg_subtotal_keys = sector_ghg_subtotal_keys

    @property
    def sector_keys(self) -> dict[IPCC_Sector, SectorTotalEmissionsVarKey]:
        return {sector: SectorTotalEmissionsVarKey(sector=sector)
                for sector in IPCC_Sector}

    @property
    def total_with_LULUCF_key(self) -> SectorTotalEmissionsVarKey:
        return SectorTotalEmissionsVarKey(sector=PseudoSector.Total_with_LULUCF)

    @property
    def total_without_LULUCF_key(self) -> SectorTotalEmissionsVarKey:
        return SectorTotalEmissionsVarKey(sector=PseudoSector.Total_without_LULUCF)

    @define(total_with_LULUCF_key, prior_shape=[years_dim])
    @define(total_without_LULUCF_key, prior_shape=[years_dim])
    @define(sector_keys, prior_shape=[years_dim])
    def model_element_postprocess_posterior_only(self, ws:WorkSpace_Proc):
        total_w_lulucf = 0
        total_wo_lulucf = 0

        for sector, var_key in self.sector_keys.items():
            sector_total = 0

            for sub_key in self.sector_ghg_subtotal_keys[sector].values():
                sector_total += ws.val.general[sub_key]

            ws.val.general[var_key] = sector_total

            if sector not in LULUCF_Sectors:
                total_wo_lulucf += sector_total

            total_w_lulucf += sector_total

        ws.val.general[self.total_with_LULUCF_key] = total_w_lulucf
        ws.val.general[self.total_without_LULUCF_key] = total_wo_lulucf
