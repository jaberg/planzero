import jax.numpy as jnp

from .enums import GHG, PT, IPCC_Sector
from .nir2025 import (
    ktCO2e_numpyro_dist_pt_ca_years,
    near_zero_sector_ghgs,
)
from .nir_model import NIR_ModelElement, NIR_ScenarioModel, Sector_GHG_Observation
from .scenario_model import (
    AnnualKey,
    GeneralKey,
    ModelElement,
    Phase,
    VersionID,
    Workspace,
    access_annual,
    define,
    define_annual,
)


class NIR2025_ModelElement(NIR_ModelElement):
    sector: IPCC_Sector
    ghg: GHG
    _ca_key: GeneralKey
    _pt_keys: dict[PT, GeneralKey]
    _annual_ca_key: AnnualKey
    _annual_pt_keys: dict[PT, AnnualKey]
    _n_years: int
    report_emissions:bool

    def __init__(self, sector:IPCC_Sector, ghg:GHG, report_emissions:bool):
        super().__init__()
        self.sector = sector
        self.ghg = ghg
        self._ca_key = self.general_key(f'{self.sector.value}_{self.ghg.value}_ca')
        self._annual_ca_key = self.annual_key(f'{self.sector.value}_{self.ghg.value}_ca')
        self._pt_keys = {
                pt: self.general_key(f'{self.sector.value}_{self.ghg.value}_{pt.value}')
                for pt in PT if pt != PT.XX}
        self._annual_pt_keys = {
                pt: self.annual_key(f'{self.sector.value}_{self.ghg.value}_{pt.value}')
                for pt in PT if pt != PT.XX}
        self.report_emissions = report_emissions

    @classmethod
    def element_identifier(cls, sector, ghg):
        return f'NIR2025_ModelElement_{sector.value}_{ghg.value}'

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
    def pt_keys(self) -> dict[PT, GeneralKey]:
        return self._pt_keys

    @property
    def annual_ca_key(self) -> AnnualKey:
        return self._annual_ca_key

    @property
    def annual_pt_keys(self) -> dict[PT, AnnualKey]:
        return self._annual_pt_keys

    @property
    def years_key(self) -> GeneralKey:
        return self.general_key('defined_years')

    @define(years_key, prior_shape=[34])
    @define(ca_key, prior_shape=[34])
    @define(pt_keys, prior_shape=[34])
    @define_annual(annual_ca_key, annual_shape=[])
    @define_annual(annual_pt_keys, annual_shape=[])
    def model_element_postprocess(self, ws:Workspace):
        ca_dist, pt_dists_by_pt = ktCO2e_numpyro_dist_pt_ca_years(
                self.sector, self.ghg,
                year_0=ws.year_0,
                n_years=34)
        ws.dist.general[self.ca_key] = ws.mcmc_expand(ca_dist)
        for pt, key in self.pt_keys.items():
            ws.dist.general[key] = ws.mcmc_expand(pt_dists_by_pt[pt])

        ws.val.general[self.years_key] = jnp.arange(
                1990, 1990 + 34) # 1990-2023 inclusive

        if self.report_emissions:
            assert ws.n_years <= 34
            ca_dist, pt_dists_by_pt = ktCO2e_numpyro_dist_pt_ca_years(
                    self.sector, self.ghg,
                    year_0=ws.year_0,
                    n_years=ws.n_years)
            ws.dist.annual[self.annual_ca_key] = ws.mcmc_expand(ca_dist)
            for pt, key in self.annual_pt_keys.items():
                ws.dist.annual[key] = ws.mcmc_expand(pt_dists_by_pt[pt])
        else:
            # satisfy shape requirements
            annual_zeros = jnp.zeros(
                    (ws.n_years,) + ws.mcmc_broadcast_shape)
            ws.val.annual[self.annual_ca_key] = annual_zeros
            for pt, key in self.annual_pt_keys.items():
                ws.val.annual[key] = annual_zeros

    @property
    def ktCO2e_emissions_ca(
            self,
            ) -> dict[tuple[IPCC_Sector, GHG], AnnualKey]:
        rval = {}
        if self.report_emissions:
            rval[self.sector, self.ghg] = self.annual_ca_key
        return rval

    @property
    def ktCO2e_emissions_pt(
            self,
            ) -> dict[tuple[IPCC_Sector, GHG], dict[PT, AnnualKey]]:
        rval = {}
        if self.report_emissions:
            rval[self.sector, self.ghg] = self.annual_pt_keys
        return rval


class Bypass_Sector_GHG_Observation(Sector_GHG_Observation):

    @property
    def identifier(self) -> str:
        return f'Bypass_Sector_GHG_Observation_{self.sector.value}_{self.ghg.value}'

    @property
    def version_id(self) -> VersionID:
        return (1,)

    @access_annual(Sector_GHG_Observation._contributors, annual_shape=[])
    @define_annual(Sector_GHG_Observation.ktCO2e_ca, annual_shape=[], posterior_only=True)
    @define_annual(Sector_GHG_Observation.ktCO2e_pt, annual_shape=[], posterior_only=True)
    def model_element_postprocess(self, ws:Workspace):
        if ws.phase != Phase.Posterior:
            return

        for ii, (pt, keys) in enumerate(self.ktCO2e_pt_contributors.items()):
            key, = keys
            ws.dist.annual[self.ktCO2e_pt[pt]] = ws.dist.annual[key]

            # ensure that we alias values to match the aliased distributions
            ws.comp.set_ndarray(self.ktCO2e_pt[pt], ws.val.annual[key])

        ws.val.annual[self.ktCO2e_ca] = sum(
                ws.val.annual[keys[0]]
                for keys in self.ktCO2e_pt_contributors.values())


class NIR2025_ScenarioModel(NIR_ScenarioModel):

    def __init__(self):
        super().__init__(
                last_forecast_year=2023,
                num_warmup=0,
                num_samples=250,
                )
        for sector in IPCC_Sector:
            for ghg in GHG:
                if (sector, ghg) not in near_zero_sector_ghgs():
                    elem = NIR2025_ModelElement(
                                sector=sector,
                                ghg=ghg,
                                report_emissions=True)
                    self.add_element(elem, mcmc_group=str((sector.value, ghg.value)))
        self.add_emissions_sums(fit_to_data=False)

    def _add_Sector_GHG_Observation(self, sector, ghg, mcmc_group, fit_to_data:bool, nir_elem):
        sgo = Bypass_Sector_GHG_Observation(
                sector=sector,
                ghg=ghg,
                nir_ca_key=nir_elem.ca_key,
                nir_pt_keys=nir_elem.pt_keys,
                )
        self.add_element(sgo, mcmc_group=None)
        return sgo
