import jax.numpy as jnp
import numpyro.distributions as dist

from . import cattle
from .enums import GHG, PT, IPCC_Sector
from .ghgvalues import GWP_100
from .nir_model import NIR_ModelElement
from .sc_3210013001 import (
    FarmType,
    Livestock,
    Livestock_nonsums,
)
from .scenario_model import (
    AnnualKey,
    CarryKey,
    GeneralKey,
    Phase,
    Workspace,
    access_annual,
    define,
    define_annual,
)

n_regions = 13
n_livestock_types = len(Livestock_nonsums)


class Cattle_StaticNormals_ModelElement(NIR_ModelElement):
    """This model element makes static normal estimates of cattle populations
    and enteric fermentation emission factors to explain three
    things:
    <ol>
    <li>cattle counts</li>
    <li>NIR2025 enteric fermentation emission factors</li>
    <li>NIR2025 overall enteric fermentation emissions (including all livestock)</li>
    </ol>
    Estimates are made for each of 8 cattle populations (beef cows, dairy cows, beef heifers for replacement, beef heifers for slaughter, dairy heifers, calves, steers, and bulls),
    and each province and territory.
    """
    _fit_to_data:bool
    farm_type = FarmType.AllCattle
    model_headcounts:bool
    model_emission_factors:bool
    model_emissions:bool

    def __init__(
            self,
            model_headcounts:bool=True,
            model_emission_factors:bool=True,
            model_emissions:bool=True,
            ):
        self.model_headcounts = model_headcounts
        self.model_emission_factors = model_emission_factors
        self.model_emissions = model_emissions

        self._ktCO2e_ca_key = self.annual_key('ktCO2e_ca')
        self._ktCO2e_pt_keys = {
                pt: self.annual_key(f'ktCO2e_{pt.value}')
                for pt in PT if pt != PT.XX}

        self._hist_ktCO2e_ca_key = self.general_key('hist_ktCO2e_ca')
        self._hist_ktCO2e_pt_keys = {
                pt: self.general_key(f'hist_ktCO2e_{pt.value}')
                for pt in PT if pt != PT.XX}

    @property
    def bovaer_actual_vs_nominal(self) -> float:
        """What fraction of cattle nominally on bovaer actually eat it properly?"""
        return .95

    @property
    def bovaer_methane_reduction(self) -> dict[object, object]:
        guess = .4
        return {
            Livestock.Bulls: guess,
            Livestock.DairyCows: 0.30, # https://www.dsm-firmenich.com/anh/products-and-services/products/methane-inhibitors/bovaer.html
            Livestock.BeefCows: .45, # https://www.dsm-firmenich.com/anh/news/press-releases/2024/2024-01-31-canada-approves-bovaer-as-first-feed-ingredient-to-reduce-methane-emissions-from-cattle.html
            Livestock.DairyHeifers: guess,
            Livestock.BeefHeifers: guess,
            Livestock.SlaughterHeifers: guess,
            Livestock.Steers: guess,
            Livestock.Calves: guess,
        }


    @property
    def version_id(self):
        if self._fit_to_data:
            return (1, self._fit_to_data)
        else:
            return (1, self._fit_to_data)

    @property
    def static_emission_factors(self) -> GeneralKey:
        return self.general_key('static_emission_factors')

    @property
    def ktCO2e_ca_key(self) -> AnnualKey:
        return self._ktCO2e_ca_key

    @property
    def ktCO2e_pt_keys(self) -> dict[PT, AnnualKey]:
        return self._ktCO2e_pt_keys

    @property
    def bovine_fraction_on_bovaer(self) -> CarryKey:
        return self.carry_key('bovine_fraction_on_bovaer')

    @property
    def annual_headcounts(self) -> AnnualKey:
        return self.annual_key('annual_headcounts')

    @property
    def annual_bovaer_fraction(self) -> AnnualKey:
        return self.annual_key('annual_bovaer_fraction')

    @define_annual(ktCO2e_ca_key, annual_shape=[], posterior_only=True)
    @define_annual(ktCO2e_pt_keys, annual_shape=[], posterior_only=True)
    @define(static_emission_factors, prior_shape=[n_livestock_types])
    @define('headcounts', prior_shape=[n_livestock_types, n_regions])
    @define_annual(annual_headcounts, annual_shape=[n_livestock_types, n_regions])
    @define('hist_emission_factors', prior_shape=[n_livestock_types])
    @define('mu_livestock_counts', prior_shape=[n_livestock_types, n_regions])
    @define('sigma_livestock_counts', prior_shape=[n_livestock_types, n_regions])
    @define('emissions_obs_noise_ca', prior_shape=[])
    @define('emissions_obs_noise_pt', prior_shape=[n_regions])
    @access_annual(annual_bovaer_fraction, annual_shape=[], default=0)
    def model_element_postprocess(self, ws:Workspace):
        ws.dist.general['mu_livestock_counts'] = dist.LogNormal(-1, 2).expand(
                (n_livestock_types, n_regions,))

        ws.dist.general['sigma_livestock_counts'] = dist.LogNormal(-1, 1).expand(
                (n_livestock_types, n_regions,))

        ws.dist.general[self.static_emission_factors] = dist.Normal(
                    jnp.array([130, 77, 120.0, 120.0, 90, 52, 45, 44.0]),
                    jnp.array([ 10,  2,   5,     5.0,  4,  5,  5,  1.0]))

        # (n_years, n_livestock_types, n_PTs=14)
        livestock_counts = jnp.array(cattle.np_data(self.farm_type))
        livestock_scale = jnp.nanmax(livestock_counts)

        if self.model_headcounts:
            self._model_headcounts(ws, livestock_counts, livestock_scale)
        if self.model_emission_factors:
            self._model_emission_factors(ws)
        if self.model_emissions:
            self._model_emissions(ws, livestock_scale)

    def _model_headcounts(self, ws, livestock_counts, livestock_scale):

        # cattle data goes way back before 1990.
        # it would be okay to use an earlier initial year.
        data_years = jnp.array(cattle.sorted_years(self.farm_type))
        valid_year_mask = (
                (ws.year_0 <= data_years)
                & (data_years < ws.year_0 + ws.n_years))

        ws.obs_valid.general['headcounts'] = ws.mcmc_broadcast_ndarray(
                valid_year_mask[:, None, None])
        ws.obs_val.general['headcounts'] = ws.mcmc_broadcast_ndarray(
                jnp.where(
                    valid_year_mask[:, None, None],
                    livestock_counts[:, :, :n_regions],
                    0))
        ws.dist.general['headcounts'] = dist.Normal(
                livestock_scale * ws.val.general['mu_livestock_counts'],
                livestock_scale * ws.val.general['sigma_livestock_counts'])

        ws.val.general[self.annual_headcounts] = (
                jnp.zeros((ws.n_years,) + ws.val.general['mu_livestock_counts'].shape)
                + livestock_scale * ws.val.general['mu_livestock_counts'])

    def _model_emission_factors(self, ws):

        # cattle data goes way back before 1990.
        # it would be okay to use an earlier initial year, but
        # emission factor estimation only really start around 1990 or later
        data_years = jnp.array(cattle.sorted_years(self.farm_type))
        valid_year_mask = (
                (ws.year_0 <= data_years)
                & (data_years < ws.year_0 + ws.n_years))
        # (years, livestock_types)
        nir_emission_factors = cattle.np_emission_factors(self.farm_type)

        # deliberately assigning observations with extra leading shape dimension
        # as this will be interpreted as multiple observations (over time)
        ws.obs_valid.general['hist_emission_factors'] = ws.mcmc_broadcast_ndarray(
                valid_year_mask[:, None])
        ws.obs_val.general['hist_emission_factors'] = ws.mcmc_broadcast_ndarray(
                jnp.where(
                    valid_year_mask[:, None],
                    nir_emission_factors,
                    0))
        ws.dist.general['hist_emission_factors'] = dist.Normal(
                ws.val.general[self.static_emission_factors],
                20)

    def _model_emissions(self, ws, livestock_scale):

        bovine_fraction_on_bovaer = ws.val.annual[self.annual_bovaer_fraction]
        # broadcast over PTs and MCMC dimension
        actual_methane_reduction = self.bovaer_actual_vs_nominal * jnp.array([
            self.bovaer_methane_reduction[lt]
            for lt in Livestock_nonsums]).reshape(
                    (-1,) + (1,) + ws.mcmc_broadcast_shape)

        # (years, n_PTs,)
        annual_PT_emissions_kg_CH4_from_cattle_on_bovaer = (
                ws.val.general[self.static_emission_factors][:, None]
                * ws.val.general['mu_livestock_counts']
                * livestock_scale
                * bovine_fraction_on_bovaer[:, None, None]
                * (1 - actual_methane_reduction)
                ).sum(axis=1) # over livestock types

        # (years, n_PTs,)
        annual_PT_emissions_kg_CH4_from_cattle_not_on_bovaer = (
                ws.val.general[self.static_emission_factors][:, None]
                * ws.val.general['mu_livestock_counts']
                * livestock_scale
                * (1 - bovine_fraction_on_bovaer)[:, None, None]
                ).sum(axis=1) # over livestock types

        # (years, n_PTs,)
        annual_PT_emissions_ktCO2e = (
                (annual_PT_emissions_kg_CH4_from_cattle_on_bovaer
                 + annual_PT_emissions_kg_CH4_from_cattle_not_on_bovaer)
                * (
                    GWP_100[GHG.CH4].magnitude # CH4 -> CO2e
                    / 1_000_000 # kg -> kt
                    ))

        ws.dist.general['emissions_obs_noise_pt'] = dist.HalfNormal(1.0).expand((n_regions,))
        ws.dist.general['emissions_obs_noise_ca'] = dist.HalfNormal(1.0)

        for ii, (pt, key) in enumerate(self.ktCO2e_pt_keys.items()):
            ws.dist.annual[key] = dist.Normal(
                    annual_PT_emissions_ktCO2e[:, ii],
                    ws.val.general['emissions_obs_noise_pt'][ii])
        ws.dist.annual[self.ktCO2e_ca_key] = dist.Normal(
                annual_PT_emissions_ktCO2e.sum(axis=1),
                ws.val.general['emissions_obs_noise_ca'])

    @property
    def ktCO2e_emissions_ca(
            self,
            ) -> dict[tuple[IPCC_Sector, GHG], AnnualKey]:
        rval = {}
        rval[IPCC_Sector.Enteric_Fermentation, GHG.CH4] = self.ktCO2e_ca_key
        return rval

    @property
    def ktCO2e_emissions_pt(
            self,
            ) -> dict[tuple[IPCC_Sector, GHG], dict[PT, AnnualKey]]:
        rval = {}
        rval[IPCC_Sector.Enteric_Fermentation, GHG.CH4] = self.ktCO2e_pt_keys
        return rval
