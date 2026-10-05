from .enums import GHG, PT, IPCC_Sector
from .nir2025 import (
    ktCO2e_numpyro_dist_pt_ca_years,
    near_zero_sector_ghgs,
)
from .regional_total_emissions_element import RegionalTotalEmissionsElement
from .scenario_model import (
    ElementKey,
    ModelElement,
    ScenarioModel,
    WorkSpace_Proc,
    define,
)
from .sector_total_emissions_element import SectorTotalEmissionsElement


class NIR2025_ModelElement(ModelElement):
    sector: IPCC_Sector
    ghg: GHG
    _ca_key: ElementKey
    _pt_keys: dict[PT, ElementKey]
    _n_years: int

    def __init__(self, sector:IPCC_Sector, ghg:GHG, n_years:int):
        super().__init__()
        self.sector = sector
        self.ghg = ghg
        self._n_years = n_years
        self._ca_key = self.element_key(f'{self.sector.value}_{self.ghg.value}_ca')
        self._pt_keys = {
                pt: self.element_key(f'{self.sector.value}_{self.ghg.value}_{pt.value}')
                for pt in PT if pt != PT.XX}

    @classmethod
    def element_identifier(cls, sector, ghg, n_years):
        return f'NIR2025_ModelElement_{sector.value}_{ghg.value}_{n_years}'

    @property
    def identifier(self):
        return self.element_identifier(self.sector, self.ghg, self.n_years)

    @property
    def version_id(self):
        return (1,)

    @property
    def ca_key(self) -> ElementKey:
        return self._ca_key

    @property
    def pt_keys(self) -> dict[PT, ElementKey]:
        return self._pt_keys

    @property
    def n_years(self) -> int:
        return self._n_years

    @define(ca_key, prior_shape=[n_years])
    @define(pt_keys, prior_shape=[n_years])
    def model_element_postprocess(self, ws:WorkSpace_Proc):
        ca_dist, pt_dists_by_pt = ktCO2e_numpyro_dist_pt_ca_years(
                self.sector, self.ghg,
                year_0=ws.year_0,
                n_years=self.n_years)
        ws.dist.general[self.ca_key] = ca_dist
        for pt, key in self.pt_keys.items():
            ws.dist.general[key] = pt_dists_by_pt[pt]


class NIR2025_ScenarioModel(ScenarioModel):

    def __init__(self, last_observed_year:int):
        assert last_observed_year >= 1990
        last_observed_year = min(last_observed_year, 2023)
        n_years = last_observed_year + 1 - 1990
        super().__init__(
                first_year=1990,
                n_prior_years=n_years,
                n_posterior_years=n_years,
                num_warmup=0,
                num_samples=250,
                )
        sector_ghg_subtotal_keys = {sector: {} for sector in IPCC_Sector}
        sector_region_ghg_subtotal_keys = {
                (sector, pt): {}
                for sector in IPCC_Sector for pt in PT if PT != PT.XX}
        for sector in IPCC_Sector:
            for ghg in GHG:
                if (sector, ghg) not in near_zero_sector_ghgs():
                    elem = NIR2025_ModelElement(
                                sector=sector,
                                ghg=ghg,
                                n_years=n_years)
                    self.add_element(elem, mcmc_group=str(sector))
                    sector_ghg_subtotal_keys[sector][ghg] = elem.ca_key
                    for pt in PT:
                        if pt == PT.XX:
                            continue
                        dd = sector_region_ghg_subtotal_keys.setdefault((sector, pt), {})
                        dd[ghg] = elem.pt_keys[pt]
        self.add_element(
                SectorTotalEmissionsElement(
                    sector_ghg_subtotal_keys=sector_ghg_subtotal_keys,
                    ),
                mcmc_group=None)
        self.add_element(
                RegionalTotalEmissionsElement(
                    sector_region_ghg_subtotal_keys=sector_region_ghg_subtotal_keys,
                    ),
                mcmc_group=None)
