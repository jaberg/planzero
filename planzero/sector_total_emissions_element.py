import enum
import warnings

import jax.numpy as jnp

from .enums import GHG, IPCC_Sector, LULUCF_Sectors
from .scenario_model import (
    AnnualKey,
    GeneralKey,
    ModelElement,
    Phase,
    Workspace,
    define,
    years_dim,
)


class PseudoSector(str, enum.Enum):

    Total_with_LULUCF = 'Total with LULUCF'
    Total_without_LULUCF = 'Total without LULUCF'


class SectorTotalEmissionsVarKey(GeneralKey, frozen=True):

    sector: IPCC_Sector | PseudoSector


class SectorTotalEmissionsElement(ModelElement):
    """
    Sum over GHG for each sector,
    and define national totals with and without LULUCF sectors
    """

    sector_ghg_subtotal_keys: dict[IPCC_Sector, dict[GHG, GeneralKey|AnnualKey]]
    expect_all_sectors:bool

    def __init__(
            self,
            sector_ghg_subtotal_keys: dict[IPCC_Sector, dict[GHG, GeneralKey|AnnualKey]],
            expect_all_sectors=True,
            ):
        self.sector_ghg_subtotal_keys = sector_ghg_subtotal_keys
        self.expect_all_sectors = expect_all_sectors

    @property
    def sector_keys(self) -> dict[IPCC_Sector, SectorTotalEmissionsVarKey]:
        return {sector: SectorTotalEmissionsVarKey(elem_id=self.identifier, name="", sector=sector)
                for sector in IPCC_Sector}

    @property
    def total_with_LULUCF_key(self) -> SectorTotalEmissionsVarKey:
        return SectorTotalEmissionsVarKey(elem_id=self.identifier, name="", sector=PseudoSector.Total_with_LULUCF)

    @property
    def total_without_LULUCF_key(self) -> SectorTotalEmissionsVarKey:
        return SectorTotalEmissionsVarKey(elem_id=self.identifier, name="", sector=PseudoSector.Total_without_LULUCF)

    @define(total_with_LULUCF_key, prior_shape=[years_dim])
    @define(total_without_LULUCF_key, prior_shape=[years_dim])
    @define(sector_keys, prior_shape=[years_dim])
    def model_element_postprocess(self, ws:Workspace):
        if ws.phase == Phase.Prior:
            # don't access the values to avoid sampling and slowing down MCMC
            pass
        elif ws.phase == Phase.Posterior:
            total_w_lulucf = jnp.zeros(())
            total_wo_lulucf = jnp.zeros(())
            n_skipped = 0

            for sector, var_key in self.sector_keys.items():
                sector_total = jnp.zeros(())

                try:
                    subkeys = self.sector_ghg_subtotal_keys[sector].values()
                except KeyError:
                    n_skipped += 1
                    continue
                for sub_key in subkeys:
                    sector_total += ws.val.general[sub_key]

                ws.val.general[var_key] = sector_total

                if sector not in LULUCF_Sectors:
                    total_wo_lulucf += sector_total

                total_w_lulucf += sector_total

            if n_skipped and self.expect_all_sectors:
                warnings.warn(
                        'sector_total_emissions_element skipped'
                        f' {n_skipped} sectors')

            ws.val.general[self.total_with_LULUCF_key] = total_w_lulucf
            ws.val.general[self.total_without_LULUCF_key] = total_wo_lulucf
        else:
            raise NotImplementedError(ws.phase)
