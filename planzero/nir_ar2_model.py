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
    access_annual,
    access_val,
    define,
    define_annual,
    define_carry,
    varkey_property,
)

n_regions = 13

class AR2_ModelElement(NIR_ModelElement):
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
        return f'AR2_ModelElement_{sector.value}_{ghg.value}'

    @property
    def identifier(self):
        return self.element_identifier(self.sector, self.ghg)

    @property
    def version_id(self):
        return (1,
                # more versions coming soon no doubt...
                )

    @varkey_property()
    def ktCO2e_ca_key(self) -> AnnualKey:
        return self._ktCO2e_ca_key

    @varkey_property()
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

    @define('phi_g_raw', prior_shape=[n_regions])
    @define('phi_g', prior_shape=[n_regions])
    @define('sigma_g_raw', prior_shape=[n_regions])
    @define('sigma_g', prior_shape=[n_regions])
    #@define('sigma_z_raw', prior_shape=[n_regions])
    #@define('sigma_z', prior_shape=[n_regions])
    @define('sigma_proc_init_raw', prior_shape=[n_regions])
    @define('sigma_proc_init', prior_shape=[n_regions])
    @define_carry('g_t', shape=[n_regions])
    @define_carry('z_t', shape=[n_regions])
    def model_element_prepare(self, ws:Workspace):
        # phi_g: damping the growth process
        # these are broad, almost-uniform distributions that drop near the edges
        ws.dist.general['phi_g_raw'] = dist.Kumaraswamy(1.25, 1.25).expand_by((n_regions,))
        ws.val.general['phi_g'] = ws.val.general['phi_g_raw'] * .1 + .9

        # how much innovation do we expect in growth from year to year? quite little.
        # growth itself should usually be a number well within (-.3, .3)
        # and it should take 10 years to cross that range
        ws.dist.general['sigma_g_raw'] = dist.HalfNormal().expand_by((n_regions,))
        ws.val.general['sigma_g'] = ws.val.general['sigma_g_raw'] / 10

        if 0:
            ws.dist.general['sigma_z_raw'] = dist.LogNormal(-3, 1).expand_by((n_regions,))
            ws.val.general['sigma_z'] = ws.val.general['sigma_z_raw'] / 30

        ws.dist.initial_carry['g_t'] = dist.Normal(0, 0.05).expand_by((n_regions,))
        ws.dist.initial_carry['z_t'] = dist.Normal(0, 1).expand_by((n_regions,))

        ws.dist.general['sigma_proc_init_raw'] = dist.HalfNormal().expand_by((n_regions,))
        ws.val.general['sigma_proc_init'] = ws.val.general['sigma_proc_init_raw']

    @access_val('phi_g') # damping on growth
    @access_val('sigma_g') # growth innovation scale
    #@access_val('sigma_z') # emission innovation scale
    @access_val('sigma_proc_init') # emission innovation scale
    @define_annual('omega_g_t', annual_shape=[n_regions]) # innovation scale on growth
    #@define_annual('omega_z_t', annual_shape=[n_regions]) # innovation scale on emission means
    @define_annual('process_loc', annual_shape=[n_regions])
    @define_annual('process_scale', annual_shape=[n_regions])
    def model_element_annual_step(self, ws:Workspace):

        ws.dist.this_annual['omega_g_t'] = dist.Normal(0, ws.val.general['sigma_g'])
        #ws.dist.this_annual['omega_z_t'] = dist.Normal(0, ws.val.general['sigma_z'])

        ws.val.next_carry['g_t'] = (
                ws.val.general['phi_g'] * ws.val.this_carry['g_t']
                + ws.val.this_annual['omega_g_t'])

        ws.val.next_carry['z_t'] = (
                ws.val.this_carry['z_t']
                + ws.val.this_carry['g_t']
                #+ ws.val.this_annual['omega_z_t']
                )

        ws.val.this_annual['process_loc'] = ws.val.this_carry['z_t']
        # TODO: allow this scale to decay exponentially
        ws.val.this_annual['process_scale'] = ws.val.general['sigma_proc_init']
    
    @access_annual('process_loc')
    @access_annual('process_scale')
    @define_annual(ktCO2e_ca_key, annual_shape=[])
    @define_annual(ktCO2e_pt_keys, annual_shape=[])
    def model_element_postprocess(self, ws:Workspace):
        assert n_regions == len(self.ktCO2e_pt_keys)

        pt_loc, pt_scale, pt_scale_floor = self._empirical_prior(ws)

        ktCO2e_mu = ws.val.annual['process_loc'] * pt_scale + pt_loc
        ktCO2e_sigma = ws.val.annual['process_scale'] * pt_scale + pt_scale_floor

        # axis=0 is years
        ktCO2e_sigma_ca = jnp.sqrt((ktCO2e_sigma ** 2).sum(axis=1))
        ktCO2e_mu_ca = ktCO2e_mu.sum(axis=1)

        for ii, (pt, ktCO2e_key) in enumerate(self.ktCO2e_pt_keys.items()):
            ws.dist.annual[self.ktCO2e_pt_keys[pt]] = dist.Normal(
                    ktCO2e_mu[:, ii],
                    ktCO2e_sigma[:, ii])

        ws.dist.annual[self.ktCO2e_ca_key] = dist.Normal(
                ktCO2e_mu_ca,
                ktCO2e_sigma_ca)

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
