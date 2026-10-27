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
from .nir_model import (
    NIR_ModelElement,
    NIR_ScenarioModel,
)
from .scenario_model import (
    AnnualKey,
    Workspace,
    define,
    define_annual,
)

n_regions = 13

class StaticNormals_ModelElement(NIR_ModelElement):
    sector: IPCC_Sector
    ghg: GHG
    _ktCO2e_ca_key: AnnualKey
    _ktCO2e_pt_keys: dict[PT, AnnualKey]

    def __init__(self,
                 sector:IPCC_Sector,
                 ghg:GHG,
                 ):
        self.sector = sector
        self.ghg = ghg

        self._ktCO2e_ca_key = self.annual_key(
                f'ktCO2e_{self.sector.value}_{self.ghg.value}_ca')
        self._ktCO2e_pt_keys = {
                pt: self.annual_key(
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
        return (1,
                # 1.2, accidentally using whole arr_ca rms not just sector/ghg
                #1.3, # mcmc works using larger 10ktCO2e "near zero" nir2025
                #1.4, # no more sigma_ca, relate it to sigma_pt
                # 1.5, # n_quad=16 with quad_p and quad_q blend
                # 1.6, # n_quad=16 with just quad_p
                #1.8, # centered priors, 1000 warmup steps
                #1.9, # tweaked logic here, new ktCO2e dists, init_uniform, target_accept_prob way up at .995
                # 1.91, # <-- tweaked epsilon for fishing emissions
                # and also empirical_prior for nationally reported sectors
                1.92, # new KL quadrature implementation

                # Idea: if some PT contribs are too tiny, then just
                #       cut their contribution to the national total
                #       which may help remove divergences at default
                #       target_accept_prob?
                )

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

        # without the floor (element 3)
        # there is a problem in the prior specification, which
        # allows strongly multimodal solution of a single random region
        # being solely reponsible for national total, and the rest confidently zero
        return (jnp.array(ca_mean / 13),
                jnp.array(ca_mean * 2 / 13),
                (jnp.array(ca_mean * 0.2)
                 if self.sector == IPCC_Sector.Nitric_Acid_Production
                 else jnp.array(ca_mean / 13))
                )

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
        assert isinstance(pt_var, np.ndarray)
        assert isinstance(pt_mean, np.ndarray)
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

        return jnp_rel_pt_loc, jnp_rel_pt_scale, 0

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
    @define_annual(ktCO2e_ca_key, annual_shape=[], posterior_only=True)
    @define_annual(ktCO2e_pt_keys, annual_shape=[], posterior_only=True)
    def model_element_postprocess(self, ws:Workspace):
        assert n_regions == len(self.ktCO2e_pt_keys)

        ws.dist.general['mu'] = dist.Normal().expand_by((n_regions,))
        ws.dist.general['sigma_pt'] = dist.LogNormal(-1, 0.7).expand_by((n_regions,))

        pt_loc, pt_scale, pt_scale_floor = self._empirical_prior(ws)

        ktCO2e_mu = ws.val.general['mu'] * pt_scale + pt_loc
        ktCO2e_sigma = ws.val.general['sigma_pt'] * pt_scale + pt_scale_floor

        ktCO2e_sigma_ca = jnp.sqrt((ktCO2e_sigma ** 2).sum(axis=0))
        ktCO2e_mu_ca = ktCO2e_mu.sum(axis=0)

        as_years = jnp.ones((ws.n_years,) + ws.mcmc_broadcast_shape)

        for ii, (pt, ktCO2e_key) in enumerate(self.ktCO2e_pt_keys.items()):
            ws.dist.annual[self.ktCO2e_pt_keys[pt]] = dist.Normal(
                    as_years * ktCO2e_mu[ii],
                    as_years * ktCO2e_sigma[ii])

        ws.dist.annual[self.ktCO2e_ca_key] = dist.Normal(
                as_years * ktCO2e_mu_ca,
                as_years * ktCO2e_sigma_ca)

    @property
    def ktCO2e_emissions_ca(
            self,
            ) -> dict[tuple[IPCC_Sector, GHG], AnnualKey]:
        rval = {}
        rval[self.sector, self.ghg] = self.ktCO2e_ca_key
        return rval

    @property
    def ktCO2e_emissions_pt(
            self,
            ) -> dict[tuple[IPCC_Sector, GHG], dict[PT, AnnualKey]]:
        rval = {}
        rval[self.sector, self.ghg] = self.ktCO2e_pt_keys
        return rval


class StaticNormals_ScenarioModel(NIR_ScenarioModel):

    def __init__(self,
                 data_cutoff:datetime.date,
                 last_forecast_year=2050,
                 sectors:type[IPCC_Sector]|list[IPCC_Sector]=IPCC_Sector,
                 ):
        """
        sectors: default to all sectors (site models require all sectors), but subsets can be used in testing
        """
        super().__init__(
                last_forecast_year=last_forecast_year,
                data_cutoff=data_cutoff,
                num_warmup=1500,
                num_samples=500,
                expect_all_sectors=(sectors==IPCC_Sector),
                )
        # num samples can be larger if scenario_model/computation.py
        # can free some memory as it goes... as of writing, it does not.
        for sector in IPCC_Sector:
            for ghg in GHG:
                if (sector, ghg) not in near_zero_sector_ghgs():
                    if isinstance(sectors, list) and sector not in sectors:
                        pass
                    else:
                        mcmc_group = f'{sector.value}_{ghg.value}'
                        snorm_elem = StaticNormals_ModelElement(
                                    sector=sector,
                                    ghg=ghg)
                        self.add_element(snorm_elem, mcmc_group=mcmc_group)

        self.add_emissions_sums()
        self.add_element_prenir_2025_m04()
