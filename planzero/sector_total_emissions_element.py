import jax.numpy as jnp

from .enums import GHG, IPCC_Sector
from .scenario_model import (
    ModelElement,
    VarKey,
    VarKeyBase,
    WorkSpace_Proc,
    define,
    years_dim,
)


class SectorTotalEmissionsVarKey(VarKeyBase, frozen=True):

    sector: IPCC_Sector


class SectorTotalEmissionsElement(ModelElement):

    sector_ghg_subtotal_keys: dict[IPCC_Sector, dict[GHG, VarKey]]

    def __init__(
            self,
            sector_ghg_subtotal_keys: dict[IPCC_Sector, dict[GHG, VarKey]],
            ):
        self.sector_ghg_subtotal_keys = sector_ghg_subtotal_keys

    @property
    def sector_keys(self):
        return {sector: SectorTotalEmissionsVarKey(sector=sector)
                for sector in IPCC_Sector}

    @define(sector_keys, prior_shape=[years_dim])
    def model_element_postprocess_posterior_only(self, ws:WorkSpace_Proc):

        for sector, var_key in self.sector_keys.items():
            sector_total = 0

            for sub_key in self.sector_ghg_subtotal_keys[sector].values():
                sector_total += ws.val.general[sub_key]

            ws.val.general[var_key] = sector_total
