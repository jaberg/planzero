from numpyro.distributions import Normal

from .enums import GHG, PT, IPCC_Sector
from .nir2025 import (
    ktCO2e_numpyro_dist_pt_ca,
    ktCO2e_numpyro_dist_pt_ca_years,
    near_zero_sector_ghgs,
)
from .numpyro_utils import stack_distributions
from .scenario_model.base import NamedKey
from .scenario_model.computation import (
    Model,
    ModelElement,
    WorkSpace_Prep,
    define,
    new_named_key,
)
from .symmetric_blended_lognormal import SymmetricBlendedLogNormal


def _sbln_from_normal(dist):
    if isinstance(dist, Normal):
        assert dist.loc == 0
        return SymmetricBlendedLogNormal.rolloff_relerr(
                mu=0,
                rolloff=dist.scale * 10,
                relerr=0.1)
    return dist


def _upgrade_to_sbln_if_necessary(dists):
    if any(isinstance(dist, SymmetricBlendedLogNormal) for dist in dists):
        return [_sbln_from_normal(dist) for dist in dists]
    else:
        return dists


class NIR2025_ModelElement(ModelElement):
    sector: IPCC_Sector
    ghg: GHG
    _ca_key: NamedKey
    _pt_keys: dict[PT, NamedKey]
    _n_samples_per_year: int
    _n_years: int

    def __init__(self, sector:IPCC_Sector, ghg:GHG, n_samples_per_year:int, n_years:int):
        self.sector = sector
        self.ghg = ghg
        self._n_samples_per_year = n_samples_per_year
        self._n_years = n_years
        self._ca_key = new_named_key(
                f'NIR2025_{self.sector}_{self.ghg}_ca')
        self._pt_keys = {
                pt: new_named_key(
                    f'NIR2025_{self.sector}_{self.ghg}_{pt}')
                for pt in PT if pt != PT.XX}

    @classmethod
    def element_identifier(cls, sector, ghg):
        return f'NIR2025_ModelElement_{sector.value}_{ghg.value}'

    @property
    def identifier(self):
        return self.element_identifier(self.sector, self.ghg)

    @property
    def ca_key(self) -> NamedKey:
        return self._ca_key

    @property
    def pt_keys(self) -> dict[PT, NamedKey]:
        return self._pt_keys

    @property
    def n_samples_per_year(self) -> int:
        return self._n_samples_per_year

    @property
    def n_years(self) -> int:
        return self._n_years

    @define(ca_key, sampled=True, shape=[n_samples_per_year, n_years])
    @define(pt_keys, sampled=True, shape=[n_samples_per_year, n_years])
    def model_element_prepare(self, ws:WorkSpace_Prep):
        if 0:
            yearly_ca_dists = []
            yearly_pt_dists_by_pt = {pt: [] for pt in PT if pt != PT.XX}
            for year in range(ws.year_0, ws.year_0 + self.n_years):
                ca_dist, pt_dists = ktCO2e_numpyro_dist_pt_ca(
                        self.sector, self.ghg, year)
                yearly_ca_dists.append(ca_dist)
                assert len(pt_dists) == 13
                for pt, pt_dist in zip(PT, pt_dists):
                    yearly_pt_dists_by_pt[pt].append(pt_dist)

            ws.dist.general[self.ca_key] = stack_distributions(
                    _upgrade_to_sbln_if_necessary(yearly_ca_dists)
                    ).expand_by((self.n_samples_per_year,))
            for pt, key in self.pt_keys.items():
                ws.dist.general[key] = stack_distributions(
                        _upgrade_to_sbln_if_necessary(yearly_pt_dists_by_pt[pt])
                        ).expand_by((self.n_samples_per_year,))
        else:
            ca_dist, pt_dists_by_pt = ktCO2e_numpyro_dist_pt_ca_years(
                    self.sector, self.ghg,
                    year_0=ws.year_0,
                    n_years=self.n_years)
            ws.dist.general[self.ca_key] = ca_dist.expand_by(
                    (self.n_samples_per_year,))
            for pt, key in self.pt_keys.items():
                ws.dist.general[key] = pt_dists_by_pt[pt].expand_by(
                        (self.n_samples_per_year,))


def NIR2025_Model(
        last_observed_year:int,
        n_samples_per_year:int,
        ) -> Model:
    assert last_observed_year >= 1990
    last_observed_year = min(last_observed_year, 2023)
    n_years = last_observed_year + 1 - 1990
    model = Model(
            first_year=1990,
            n_prior_years=n_years,
            n_posterior_years=n_years,
            )
    for sector in IPCC_Sector:
        for ghg in GHG:
            if (sector, ghg) not in near_zero_sector_ghgs():
                model.add_element(
                        NIR2025_ModelElement(
                            sector=sector,
                            ghg=ghg,
                            n_samples_per_year=n_samples_per_year,
                            n_years=n_years))
    return model
