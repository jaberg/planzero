import jax.numpy as jnp
import numpyro.distributions as dist

from .enums import GHG, PT, IPCC_Sector
from .nir2025 import (
    near_zero_sector_ghgs,
)
from .nir2025_model import NIR2025_ModelElement
from .regional_total_emissions_element import RegionalTotalEmissionsElement
from .scenario_model import (
    GeneralKey,
    ModelElement,
    Phase,
    ScenarioModel,
    Workspace,
    define,
    years_dim,
)
from .sector_total_emissions_element import SectorTotalEmissionsElement


class StaticNormals_ModelElement(ModelElement):
    sector: IPCC_Sector
    ghg: GHG
    _unit_ca_key: GeneralKey
    _unit_pt_keys: dict[PT, GeneralKey]
    _ktCO2e_ca_key: GeneralKey
    _ktCO2e_pt_keys: dict[PT, GeneralKey]

    def __init__(self, sector:IPCC_Sector, ghg:GHG, nir_ca_key, nir_pt_keys, nir_scale_key):
        self.sector = sector
        self.ghg = ghg
        self.nir_ca_key = nir_ca_key
        self.nir_pt_keys = nir_pt_keys
        self.nir_scale_key = nir_scale_key

        self._unit_ca_key = self.general_key(
                f'unit_{self.sector.value}_{self.ghg.value}_ca')
        self._unit_pt_keys = {
                pt: self.general_key(
                    f'unit_{self.sector.value}_{self.ghg.value}_{pt.value}')
                for pt in PT if pt != PT.XX}

        self._ktCO2e_ca_key = self.general_key(
                f'ktCO2e_{self.sector.value}_{self.ghg.value}_ca')
        self._ktCO2e_pt_keys = {
                pt: self.general_key(
                    f'ktCO2e_{self.sector.value}_{self.ghg.value}_{pt.value}')
                for pt in PT if pt != PT.XX}

    @classmethod
    def element_identifier(cls, sector, ghg):
        return f'StaticNormals_ModelElement_{sector.value}_{ghg.value}'

    @property
    def identifier(self):
        return self.element_identifier(self.sector, self.ghg)

    @property
    def version_id(self):
        return (1,)

    @property
    def unit_ca_key(self) -> GeneralKey:
        return self._unit_ca_key

    @property
    def unit_pt_keys(self) -> dict[PT, GeneralKey]:
        return self._unit_pt_keys

    @property
    def ktCO2e_ca_key(self) -> GeneralKey:
        return self._ktCO2e_ca_key

    @property
    def ktCO2e_pt_keys(self) -> dict[PT, GeneralKey]:
        return self._ktCO2e_pt_keys

    @define('mu', prior_shape=[13])
    @define('sigma_pt', prior_shape=[13])
    @define('sigma_ca', prior_shape=[])
    @define(unit_ca_key, prior_shape=[])
    @define(unit_pt_keys, prior_shape=[])
    @define(ktCO2e_ca_key, prior_shape=[years_dim])
    @define(ktCO2e_pt_keys, prior_shape=[years_dim])
    def model_element_postprocess(self, ws:Workspace):
        n_regions = 13
        assert n_regions == len(self.unit_pt_keys)

        ws.dist.general['mu'] = dist.Normal(0.0, 1.0).expand((n_regions,))
        ws.dist.general['sigma_pt'] = dist.LogNormal(-1.0, 0.7).expand((n_regions,))
        ws.dist.general['sigma_ca'] = dist.LogNormal(-1.0, 0.7)

        mu_ca = ws.val.general['mu'].sum(axis=0) # over regions

        emission_scale = ws.val.general[self.nir_scale_key]

        squish = dist.transforms.AffineTransform(loc=0, scale=1/emission_scale)
        as_years = jnp.ones((ws.n_years,) + ws.mcmc_broadcast_shape)

        # TODO: don't hard-code this, have the nir model element export it
        hardcoded_weight = jnp.array(33.0)

        for ii, (pt, unit_key) in enumerate(self.unit_pt_keys.items()):

            ws.obs_weight.general[unit_key] = hardcoded_weight
            ws.obs_dist.general[unit_key] = dist.TransformedDistribution(
                    ws.dist.general[self.nir_pt_keys[pt]],
                    squish)

            ws.dist.general[unit_key] = dist.Normal(
                    ws.val.general['mu'][ii],
                    ws.val.general['sigma_pt'][ii])

            if ws.phase == Phase.Posterior:
                # predictions for future years
                ws.dist.general[self.ktCO2e_pt_keys[pt]] = dist.Normal(
                        as_years * emission_scale * ws.val.general['mu'][ii],
                        as_years * emission_scale * ws.val.general['sigma_pt'][ii])


        ws.obs_weight.general[self.unit_ca_key] = hardcoded_weight
        ws.obs_dist.general[self.unit_ca_key] = dist.TransformedDistribution(
                ws.dist.general[self.nir_ca_key],
                squish)
        ws.dist.general[self.unit_ca_key] = dist.Normal(
                mu_ca,
                ws.val.general['sigma_ca'])

        if ws.phase == Phase.Posterior:
            # predictions for future years
            # which are distinct rather than scaled random variables
            # relative to the unit_ca_key, because unit_ca_key
            # refers to previous, observed years' emissions
            ws.dist.general[self.ktCO2e_ca_key] = dist.Normal(
                    as_years * emission_scale * mu_ca,
                    as_years * emission_scale * ws.val.general['sigma_ca'])


class StaticNormals_ScenarioModel(ScenarioModel):

    def __init__(self,
                 last_observed_year:int,
                 last_forecast_year=2050,
                 sectors:type[IPCC_Sector]|list[IPCC_Sector]=IPCC_Sector,
                 ):
        """
        sectors: default to all sectors (site models require all sectors), but subsets can be used in testing
        """
        assert 1990 <= last_observed_year <= 2023
        super().__init__(
                first_year=1990,
                n_prior_years=last_observed_year + 1 - 1990,
                n_posterior_years=last_forecast_year + 1 - 1990,
                num_warmup=250,
                num_samples=500,
                )
        sector_ghg_subtotal_keys = {sector: {} for sector in sectors}
        sector_region_ghg_subtotal_keys = {
                (sector, pt): {}
                for sector in sectors for pt in PT if PT != PT.XX}
        for sector in sectors:
            for ghg in GHG:
                if (sector, ghg) not in near_zero_sector_ghgs():
                    mcmc_group = f'{sector.value}_{ghg.value}'
                    nir_elem = NIR2025_ModelElement(
                            sector=sector,
                            ghg=ghg,
                            n_years=self.n_prior_years)
                    self.add_element(nir_elem, mcmc_group=mcmc_group)
                    snorm_elem = StaticNormals_ModelElement(
                                sector=sector,
                                ghg=ghg,
                                nir_scale_key=nir_elem.scale_key,
                                nir_ca_key=nir_elem.ca_key,
                                nir_pt_keys=nir_elem.pt_keys)
                    self.add_element(snorm_elem, mcmc_group=mcmc_group)
                    sector_ghg_subtotal_keys[sector][ghg] = snorm_elem.ktCO2e_ca_key
                    for pt in PT:
                        if pt == PT.XX:
                            continue
                        dd = sector_region_ghg_subtotal_keys.setdefault((sector, pt), {})
                        dd[ghg] = snorm_elem.ktCO2e_pt_keys[pt]
        self.add_element(
                SectorTotalEmissionsElement(
                    sector_ghg_subtotal_keys=sector_ghg_subtotal_keys,
                    expect_all_sectors=(sectors=IPCC_Sector),
                    ),
                mcmc_group=None)
        self.add_element(
                RegionalTotalEmissionsElement(
                    sector_region_ghg_subtotal_keys=sector_region_ghg_subtotal_keys,
                    expect_all_sectors=(sectors=IPCC_Sector),
                    ),
                mcmc_group=None)


def loglik_NIR_Normal(
    component_id,
    NIR_year,
    emission_year,
    ):
    raise NotImplementedError()

def loglik_NIR_BayesianNormal(
    component_id,
    NIR_year,
    emission_year,
    ):
    raise NotImplementedError()

def weighted_KL_score(year, model_id):
    raise NotImplementedError()

def p_emissions_below_thresh_ex_LULUCF(
        model_id:str,
        thresh_ktCO2e:float,
        ) -> float:
    raise NotImplementedError()
