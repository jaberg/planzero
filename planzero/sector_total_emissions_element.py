import jax.numpy as jnp

from .enums import GHG, IPCC_Sector
from .scenario_model.computation import ModelElement, VarKey, WorkSpace_Proc, define


class SectorTotalEmissionsVarKey(VarKey, frozen=True):

    sector: IPCC_Sector


class SectorTotalEmissionsElement(ModelElement):

    sector_ghg_subtotal_keys: dict[IPCC_Sector, dict[GHG, VarKey]]

    @property
    def sector_keys(self):
        return {sector: SectorTotalEmissionsVarKey(sector=sector)
                for sector in IPCC_Sector}

    @define(sector_keys, sampled=False, prior_shape=[years_dim])
    def model_element_postprocess(self, ws:WorkSpace_Proc):

        for sector, var_key in self.sector_keys.items():
            sector_total = jnp.zeros(ws.shape[var_key])

            for ghg, sub_key in self.sector_ghg_subtotal_keys[sector].items():
                sector_total += ws.val.general[sub_key]

            ws.val.general[var_key] = sector_total
