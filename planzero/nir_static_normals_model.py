import datetime

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

from .enums import GHG, PT, IPCC_Sector
from .nir2025 import (
    idx_of_ghg,
    idx_of_sector,
    ktCO2e_dense_w_nan,
    near_zero_sector_ghgs,
)
from .nir2025_model import NIR2025_ModelElement
from .nir_model import NIR_ScenarioModel, PseudoRegion
from .prenir_element import PreNIR_ModelElement
from .regional_total_emissions_element import RegionalTotalEmissionsElement
from .scenario_model import (
    GeneralKey,
    ModelElement,
    Phase,
    Workspace,
    define,
    years_dim,
)
from .sector_total_emissions_element import PseudoSector, SectorTotalEmissionsElement

n_regions = 13

class StaticNormals_ModelElement(ModelElement):
    sector: IPCC_Sector
    ghg: GHG
    _unit_ca_key: GeneralKey
    _unit_pt_keys: dict[PT, GeneralKey]
    _ktCO2e_ca_key: GeneralKey
    _ktCO2e_pt_keys: dict[PT, GeneralKey]
    _fit_to_data:bool

    def __init__(self,
                 sector:IPCC_Sector,
                 ghg:GHG,
                 nir_ca_key,
                 nir_pt_keys,
                 nir_scale_key,
                 fit_to_data:bool,
                 ):
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

        self._fit_to_data = fit_to_data

    @classmethod
    def element_identifier(cls, sector, ghg):
        return f'StaticNormals_ModelElement_{sector.value}_{ghg.value}'

    @property
    def identifier(self):
        return self.element_identifier(self.sector, self.ghg)

    @property
    def version_id(self):
        if self._fit_to_data:
            return (1,
                    self._fit_to_data,
                    # 1.2, accidentally using whole arr_ca rms not just sector/ghg
                    1.3,
                    )
        else:
            return (1, self._fit_to_data)

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

    @define('mu', prior_shape=[n_regions])
    @define('sigma_pt', prior_shape=[n_regions])
    @define('sigma_ca', prior_shape=[])
    @define(unit_ca_key, prior_shape=[])
    @define(unit_pt_keys, prior_shape=[])
    @define(ktCO2e_ca_key, prior_shape=[years_dim])
    @define(ktCO2e_pt_keys, prior_shape=[years_dim])
    def model_element_postprocess(self, ws:Workspace):
        assert n_regions == len(self.unit_pt_keys)

        _, arr_ca = ktCO2e_dense_w_nan()
        emission_scale = np.sqrt(
                np.mean(
                    arr_ca[idx_of_sector[self.sector],
                           idx_of_ghg[self.ghg],
                           ] ** 2))

        ws.dist.general['mu'] = dist.Normal(0.0, 1.0).expand((n_regions,))
        ws.dist.general['sigma_pt'] = dist.LogNormal(-1.0, 0.7).expand((n_regions,))
        ws.dist.general['sigma_ca'] = dist.LogNormal(-1.0, 0.7)

        mu_ca = ws.val.general['mu'].sum(axis=0) # over regions

        squish = dist.transforms.AffineTransform(loc=0, scale=1/emission_scale)
        as_years = jnp.ones((ws.n_years,) + ws.mcmc_broadcast_shape)

        obs_weight = jnp.array(1.0)

        for ii, (pt, unit_key) in enumerate(self.unit_pt_keys.items()):

            if self._fit_to_data:
                ws.obs_weight.general[unit_key] = obs_weight
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

        if self._fit_to_data:
            ws.obs_weight.general[self.unit_ca_key] = obs_weight
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


class StaticNormals_ScenarioModel(NIR_ScenarioModel):

    def __init__(self,
                 data_cutoff:datetime.date,
                 last_forecast_year=2050,
                 sectors:type[IPCC_Sector]|list[IPCC_Sector]=IPCC_Sector,
                 ):
        """
        sectors: default to all sectors (site models require all sectors), but subsets can be used in testing
        """
        if data_cutoff == datetime.date(year=2024, month=12, day=31):
            last_observed_nir_year = 2022
        else:
            raise NotImplementedError(data_cutoff)
        assert 1990 <= last_observed_nir_year <= 2023
        super().__init__(
                first_year=1990,
                n_prior_years=last_observed_nir_year + 1 - 1990,
                n_posterior_years=last_forecast_year + 1 - 1990,
                num_warmup=250,
                num_samples=250,
                )
        self.data_cutoff = data_cutoff
        sector_ghg_subtotal_keys = {sector: {} for sector in IPCC_Sector}
        sector_region_ghg_subtotal_keys = {
                (sector, pt): {}
                for sector in IPCC_Sector for pt in PT if PT != PT.XX}
        sector_ghg_nir_keys:dict = {}
        nir_years_key = None
        for sector in IPCC_Sector:
            for ghg in GHG:
                if (sector, ghg) not in near_zero_sector_ghgs():
                    mcmc_group = f'{sector.value}_{ghg.value}'
                    nir_elem = NIR2025_ModelElement(
                            sector=sector,
                            ghg=ghg,
                            n_years=self.n_prior_years)
                    nir_years_key = nir_elem.years_key
                    self.add_element(nir_elem, mcmc_group=mcmc_group)
                    if isinstance(sectors, list) and sector not in sectors:
                        fit_to_data = False
                    else:
                        fit_to_data = True
                    snorm_elem = StaticNormals_ModelElement(
                                sector=sector,
                                ghg=ghg,
                                nir_scale_key=nir_elem.scale_key,
                                nir_ca_key=nir_elem.ca_key,
                                nir_pt_keys=nir_elem.pt_keys,
                                fit_to_data=fit_to_data)
                    self.add_element(snorm_elem, mcmc_group=mcmc_group)
                    sector_ghg_subtotal_keys[sector][ghg] = snorm_elem.ktCO2e_ca_key

                    nir_d = sector_ghg_nir_keys[sector, ghg] = {}
                    nir_d[PseudoRegion.NationalTotal] = nir_elem.ca_key
                    for pt in PT:
                        if pt == PT.XX:
                            continue
                        dd = sector_region_ghg_subtotal_keys.setdefault((sector, pt), {})
                        dd[ghg] = snorm_elem.ktCO2e_pt_keys[pt]
                        nir_d[pt] = nir_elem.pt_keys[pt]

        self.add_element(
                SectorTotalEmissionsElement(
                    sector_ghg_subtotal_keys=sector_ghg_subtotal_keys,
                    expect_all_sectors=(sectors==IPCC_Sector),
                    ),
                mcmc_group=None)

        self.add_element(
                RegionalTotalEmissionsElement(
                    sector_region_ghg_subtotal_keys=sector_region_ghg_subtotal_keys,
                    expect_all_sectors=(sectors==IPCC_Sector),
                    ),
                mcmc_group=None)

        self.add_element(
                PreNIR_ModelElement(
                    nir_year=2025,
                    months_ahead=4,
                    nir_years_key=nir_years_key,
                    sector_ghg_nir_keys=sector_ghg_nir_keys,
                    sector_ghg_subtotal_keys=sector_ghg_subtotal_keys,
                    sector_region_ghg_subtotal_keys=sector_region_ghg_subtotal_keys,
                    expect_all_sectors=(sectors==IPCC_Sector),
                    ),
                mcmc_group=None)


    # NIR_ScenarioModel requirement
    def ktCO2e_var_key_sector_ghg_region(self, sector, ghg, pt) -> GeneralKey:
        elem_id = StaticNormals_ModelElement.element_identifier(sector, ghg)
        model_elem = self.model_elements[elem_id]
        if isinstance(model_elem, StaticNormals_ModelElement):
            if isinstance(pt, PT):
                return model_elem.ktCO2e_pt_keys[pt]
            elif isinstance(pt, PseudoRegion) and pt == PseudoRegion.NationalTotal:
                return model_elem.ktCO2e_ca_key
            else:
                raise NotImplementedError(pt)
        else:
            raise NotImplementedError()

    def prenir_2025_m04_total_key(self) -> GeneralKey:
        return PreNIR_ModelElement.SubtotalVarKey(
                elem_id=PreNIR_ModelElement.element_identifier(2025, 4),
                name="subtotal",
                sector=PseudoSector.Total_with_LULUCF,
                ghg=None,
                region=PseudoRegion.NationalTotal,
                nir_year=2025,
                months_ahead=4)

    def prenir_2025_m04_sector_ghg_key(self, sector:IPCC_Sector, ghg:GHG) -> GeneralKey:
        return PreNIR_ModelElement.SubtotalVarKey(
                elem_id=PreNIR_ModelElement.element_identifier(2025, 4),
                name="subtotal",
                sector=sector,
                ghg=ghg,
                region=PseudoRegion.NationalTotal,
                nir_year=2025,
                months_ahead=4)
