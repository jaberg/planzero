"""
Placeholder 2024 ModelElement based on sliced 2025 NIR data
"""

import jax.numpy as jnp

from .enums import GHG, PT, IPCC_Sector
from .nir2025 import (
    ktCO2e_numpyro_dist_pt_ca_years,
)
from .nir_model import NIR_ModelElement
from .scenario_model import (
    AnnualKey,
    GeneralKey,
    Workspace,
    define,
)


class NIR2024_ModelElement(NIR_ModelElement):
    sector: IPCC_Sector
    ghg: GHG
    _scale_key: GeneralKey
    _ca_key: GeneralKey
    _pt_keys: dict[PT, GeneralKey]
    report_emissions:bool

    def __init__(self, sector:IPCC_Sector, ghg:GHG, report_emissions:bool):
        super().__init__()
        self.sector = sector
        self.ghg = ghg
        self._ca_key = self.general_key(f'{self.sector.value}_{self.ghg.value}_ca')
        self._scale_key = self.general_key(f'{self.sector.value}_{self.ghg.value}_scale')
        self._pt_keys = {
                pt: self.general_key(f'{self.sector.value}_{self.ghg.value}_{pt.value}')
                for pt in PT if pt != PT.XX}
        self.report_emissions = report_emissions

    @classmethod
    def element_identifier(cls, sector, ghg):
        return f'NIR2024_ModelElement_{sector.value}_{ghg.value}'

    @property
    def identifier(self):
        return self.element_identifier(self.sector, self.ghg)

    @property
    def version_id(self):
        # there's no inference in this model element, but the values
        # are used to train other model elements, so this version
        # should be changed when the values change.
        return (#2,
                3, # new def of ktCO2e_numpyro_dist_pt_ca_years fn
                )

    @property
    def ca_key(self) -> GeneralKey:
        return self._ca_key

    @property
    def scale_key(self) -> GeneralKey:
        return self._scale_key

    @property
    def pt_keys(self) -> dict[PT, GeneralKey]:
        return self._pt_keys

    @property
    def years_key(self) -> GeneralKey:
        return self.general_key('defined_years')

    @define(years_key, prior_shape=[33])
    @define(ca_key, prior_shape=[33])
    @define(pt_keys, prior_shape=[33])
    def model_element_postprocess(self, ws:Workspace):
        ca_dist, pt_dists_by_pt = ktCO2e_numpyro_dist_pt_ca_years(
                self.sector, self.ghg,
                year_0=ws.year_0,
                n_years=33)
        ws.dist.general[self.ca_key] = ws.mcmc_expand(ca_dist)
        for pt, key in self.pt_keys.items():
            ws.dist.general[key] = ws.mcmc_expand(pt_dists_by_pt[pt])

        ws.val.general[self.years_key] = jnp.arange(
                1990, 1990 + 33) # 1990-2022 inclusive

    @property
    def ktCO2e_emissions_ca(
            self,
            ) -> dict[tuple[IPCC_Sector, GHG], AnnualKey]:
        rval = {}
        if self.report_emissions:
            rval[self.sector, self.ghg] = self.ca_key
        return rval

    @property
    def ktCO2e_emissions_pt(
            self,
            ) -> dict[tuple[IPCC_Sector, GHG], dict[PT, AnnualKey]]:
        rval = {}
        if self.report_emissions:
            rval[self.sector, self.ghg] = self.pt_keys
        return rval
