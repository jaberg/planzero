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
    AnnualKey,
    GeneralKey,
    ModelElement,
    Phase,
    Workspace,
    define,
    define_annual,
)
from .sector_total_emissions_element import PseudoSector, SectorTotalEmissionsElement

n_regions = 13

class StaticNormals_ModelElement(ModelElement):
    sector: IPCC_Sector
    ghg: GHG
    _unit_ca_key: GeneralKey
    _unit_pt_keys: dict[PT, GeneralKey]
    _ktCO2e_ca_key: AnnualKey
    _ktCO2e_pt_keys: dict[PT, AnnualKey]
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

        self._ktCO2e_ca_key = self.annual_key(
                f'ktCO2e_{self.sector.value}_{self.ghg.value}_ca')
        self._ktCO2e_pt_keys = {
                pt: self.annual_key(
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
                    #1.3, # mcmc works using larger 10ktCO2e "near zero" nir2025
                    #1.4, # no more sigma_ca, relate it to sigma_pt
                    # 1.5, # n_quad=16 with quad_p and quad_q blend
                    # 1.6, # n_quad=16 with just quad_p
                    #1.8, # centered priors, 1000 warmup steps
                    #1.9, # tweaked logic here, new ktCO2e dists, init_uniform, target_accept_prob way up at .995
                    1.91, # <-- tweaked epsilon for fishing emissions
                    # and also empirical_prior for nationally reported sectors

                    # Idea: if some PT contribs are too tiny, then just
                    #       cut their contribution to the national total
                    #       which may help remove divergences at default
                    #       target_accept_prob?
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
    def ktCO2e_ca_key(self) -> AnnualKey:
        return self._ktCO2e_ca_key

    @property
    def ktCO2e_pt_keys(self) -> dict[PT, AnnualKey]:
        return self._ktCO2e_pt_keys

    def _empirical_prior_for_only_nationally_reported(self, ws):
        # This different algorithm is used to get priors for
        # sectors that are evidently not reported provincially.
        # The provincial reports are all zero, no attempt to add up to
        # national total.
        _, arr_ca = ktCO2e_dense_w_nan()
        ca_totals = arr_ca[idx_of_sector[self.sector], idx_of_ghg[self.ghg]]

        ca_mean = np.mean(ca_totals)

        # yes, return location and scale as same number
        return (jnp.array(ca_mean / 13 / 2),
                jnp.array(ca_mean / 13 / 2))

    def _default_empirical_prior(self, ws):
        # it's cheating to use data here, but
        # the scales are so wildly different it's hard to avoid
        # sharp curvatures without putting each PT on its own scale
        arr_pt, arr_ca = ktCO2e_dense_w_nan()

        # TODO: we don't even use emission_scale anymore
        debug = False # and ws.phase == Phase.Prior

        if debug:
            print('arr_ca', arr_ca[idx_of_sector[self.sector],
                               idx_of_ghg[self.ghg],
                               ])

            from .nir2025 import (
                ktCO2e_numpyro_dist_pt_ca_years,
            )
            ca_dist, _ = ktCO2e_numpyro_dist_pt_ca_years(
                    self.sector, self.ghg,
                    year_0=ws.year_0,
                    n_years=34)
            print('mu', ca_dist.mu)

            # print('emission scale', emission_scale)
        pt_mean = np.nanmean(
                arr_pt[idx_of_sector[self.sector], idx_of_ghg[self.ghg]],
                dtype=np.float64,
                axis=1)
        pt_var = np.nanvar(
                arr_pt[idx_of_sector[self.sector], idx_of_ghg[self.ghg]],
                dtype=np.float64,
                axis=1)
        assert pt_var.shape == (n_regions,)
        relative_pt_mean = []
        relative_pt_scale = []
        for pt_var_ii, pt_mean_ii in zip(pt_var, pt_mean):
            if np.isfinite(pt_mean_ii):
                relative_pt_mean.append(pt_mean_ii)
                variance_baseline = (relative_pt_mean[-1] * .1) ** 2
            else:
                relative_pt_mean.append(10) # a small-ish amount?
                variance_baseline = (relative_pt_mean[-1] * .8) ** 2

            if np.isfinite(pt_var_ii):
                relative_pt_scale.append(
                        np.sqrt(1e-12 + pt_var_ii + variance_baseline))
            else:
                relative_pt_scale.append(1e-12 + variance_baseline)

        jnp_rel_pt_loc = jnp.array(relative_pt_mean, dtype=jnp.float64).reshape(
                (n_regions,) + ws.mcmc_broadcast_shape)
        jnp_rel_pt_scale = jnp.array(relative_pt_scale, dtype=jnp.float64).reshape(
                (n_regions,) + ws.mcmc_broadcast_shape)

        if debug:
            print('pt_var', pt_var)
            print('pt_mean', pt_mean)
            print('rel_loc', jnp_rel_pt_loc)
            print('rel_scale', jnp_rel_pt_scale)

        return jnp_rel_pt_loc, jnp_rel_pt_scale

    def _empirical_prior(self, ws):
        if (self.sector in (
                IPCC_Sector.Ammonia_Production,
                IPCC_Sector.Nitric_Acid_Production,
                IPCC_Sector.Petrochemical_and_Carbon_Black_Production,
                )
            ):
            return self._empirical_prior_for_only_nationally_reported(ws)
        else:
            return self._default_empirical_prior(ws)


    @define('mu', prior_shape=[n_regions])
    @define('sigma_pt', prior_shape=[n_regions])
    @define(unit_ca_key, prior_shape=[]) # unit is mis-nomer, now means "prior years"
    @define(unit_pt_keys, prior_shape=[])
    @define_annual(ktCO2e_ca_key, annual_shape=[])
    @define_annual(ktCO2e_pt_keys, annual_shape=[])
    def model_element_postprocess(self, ws:Workspace):
        assert n_regions == len(self.unit_pt_keys)

        ws.dist.general['mu'] = dist.Normal().expand_by((n_regions,))
        ws.dist.general['sigma_pt'] = dist.LogNormal(-1, 0.7).expand_by((n_regions,))

        pt_loc, pt_scale = self._empirical_prior(ws)

        ktCO2e_mu = ws.val.general['mu'] * pt_scale + pt_loc
        ktCO2e_sigma = ws.val.general['sigma_pt'] * pt_scale

        ktCO2e_sigma_ca = jnp.sqrt((ktCO2e_sigma ** 2).sum(axis=0))
        ktCO2e_mu_ca = ktCO2e_mu.sum(axis=0)

        as_years = jnp.ones((ws.n_years,) + ws.mcmc_broadcast_shape)

        obs_weight = jnp.array(1.0)

        for ii, (pt, unit_key) in enumerate(self.unit_pt_keys.items()):

            if self._fit_to_data:
                ws.obs_weight.general[unit_key] = obs_weight
                ws.obs_dist.general[unit_key] = ws.dist.general[self.nir_pt_keys[pt]]

            ws.dist.general[unit_key] = dist.Normal(ktCO2e_mu[ii], ktCO2e_sigma[ii])

            if ws.phase == Phase.Posterior:
                # predictions for future years
                ws.dist.annual[self.ktCO2e_pt_keys[pt]] = dist.Normal(
                        as_years * ktCO2e_mu[ii],
                        as_years * ktCO2e_sigma[ii])

        if self._fit_to_data:
            ws.obs_weight.general[self.unit_ca_key] = obs_weight
            ws.obs_dist.general[self.unit_ca_key] = ws.dist.general[self.nir_ca_key]
        ws.dist.general[self.unit_ca_key] = dist.Normal(ktCO2e_mu_ca, ktCO2e_sigma_ca)

        if ws.phase == Phase.Posterior:
            # predictions for future years
            # which are distinct rather than scaled random variables
            # relative to the unit_ca_key, because unit_ca_key
            # refers to previous, observed years' emissions
            ws.dist.annual[self.ktCO2e_ca_key] = dist.Normal(
                    as_years * ktCO2e_mu_ca,
                    as_years * ktCO2e_sigma_ca)


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
                num_warmup=1500,
                num_samples=500,
                )
        # num samples can be larger if scenario_model/computation.py
        # can free some memory as it goes... as of writing, it does not.
        self.data_cutoff = data_cutoff
        sector_ghg_subtotal_keys = {sector: {} for sector in IPCC_Sector}
        sector_region_ghg_subtotal_keys = {
                (sector, pt): {}
                for sector in IPCC_Sector for pt in PT if PT != PT.XX}
        sector_ghg_nir_keys:dict = {}
        pbt_keys:dict = {}
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
                    pbt_keys[sector, ghg] = snorm_elem.ktCO2e_ca_key

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

        self.add_element_prenir_2025_m04(
                nir_years_key=nir_years_key,
                sector_ghg_nir_keys=sector_ghg_nir_keys,
                sector_ghg_subtotal_keys=sector_ghg_subtotal_keys,
                sector_region_ghg_subtotal_keys=sector_region_ghg_subtotal_keys,
                expect_all_sectors=(sectors==IPCC_Sector),
                )

        self.add_element_p_net_zero_2050(
                ktCO2e_normal_keys_by_sector_ghg=pbt_keys,
                )


    # NIR_ScenarioModel requirement
    def ktCO2e_var_key_sector_ghg_region(self, sector, ghg, pt) -> AnnualKey:
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
