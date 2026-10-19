import jax.numpy as jnp
import numpyro.distributions as dist

from . import cattle
from .enums import GHG, PT
from .ghgvalues import GWP_100
from .sc_3210013001 import (
    FarmType,
    Livestock,
    Livestock_nonsums,
    number_of_cattle_by_class_and_farm_type_combined_surveys,
)
from .scenario_model import (
    AnnualKey,
    GeneralKey,
    ModelElement,
    Phase,
    Workspace,
    define,
    define_annual,
    years_dim,
)

n_regions = 13
n_livestock_types = len(Livestock_nonsums)

class Cattle_StaticNormals_ModelElement(ModelElement):
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
    nir_ca_key: GeneralKey
    nir_pt_keys: dict[PT, GeneralKey]
    farm_type = FarmType.AllCattle
    model_headcounts:bool
    model_emission_factors:bool
    model_emissions:bool

    def __init__(
            self,
            nir_ca_key:GeneralKey,
            nir_pt_keys:dict[PT, GeneralKey],
            fit_to_data:bool=True,
            model_headcounts:bool=True,
            model_emission_factors:bool=True,
            model_emissions:bool=True,
            ):
        self.nir_ca_key = nir_ca_key
        self.nir_pt_keys = nir_pt_keys
        self._fit_to_data = fit_to_data
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
    def static_livestock_counts(self) -> GeneralKey:
        return self.general_key('static_livestock_counts')

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
    def hist_ktCO2e_ca_key(self) -> GeneralKey:
        return self._hist_ktCO2e_ca_key

    @property
    def hist_ktCO2e_pt_keys(self) -> dict[PT, GeneralKey]:
        return self._hist_ktCO2e_pt_keys

    @property
    def bovine_fraction_on_bovaer(self) -> AnnualKey:
        return self.annual_key('bovine_fraction_on_bovaer')

    @property
    def annual_headcounts(self) -> AnnualKey:
        return self.annual_key('annual_headcounts')

    @access_annual(bovine_fraction_on_bovaer, annual_shape=[], default=0)
    @define_annual(ktCO2e_ca_key, annual_shape=[])
    @define_annual(ktCO2e_pt_keys, annual_shape=[])
    @define(hist_ktCO2e_ca_key, prior_shape=[])
    @define(hist_ktCO2e_pt_keys, prior_shape=[])
    @define(static_livestock_counts, prior_shape=[n_livestock_types, n_regions])
    @define(static_emission_factors, prior_shape=[n_livestock_types])
    @define('headcounts', prior_shape=[n_livestock_types, n_regions])
    @define(annual_headcounts, annual_shape=[n_livestock_types, n_regions])
    @define('hist_emission_factors', prior_shape=[n_livestock_types])
    @define('mu_livestock_counts', prior_shape=[n_livestock_types, n_regions])
    @define('sigma_livestock_counts', prior_shape=[n_livestock_types, n_regions])
    @define('emissions_obs_noise_ca', prior_shape=[])
    @define('emissions_obs_noise_pt', prior_shape=[n_regions])
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

        bovine_fraction_on_bovaer = ws.val.annual[self.bovine_fraction_on_bovaer]
        # broadcast over PTs and MCMC dimension
        actual_methane_reduction = self.bovaer_actual_vs_nominal * jnp.array([
            self.bovaer_methane_reduction[lt]
            for lt in Livestock_nonsums]).reshape(
                    (-1,) + (1,) + ws.mcmc_broadcast_shape)

        # (n_PTs,)
        PT_emissions_kg_CH4_from_cattle_on_bovaer = (
                ws.val.general[self.static_emission_factors][:, None]
                * ws.val.general['mu_livestock_counts']
                * livestock_scale
                * bovine_fraction_on_bovaer
                * (1 - actual_methane_reduction)
                ).sum(axis=0) # over livestock types

        # (n_PTs,)
        PT_emissions_kg_CH4_from_cattle_not_on_bovaer = (
                ws.val.general[self.static_emission_factors][:, None]
                * ws.val.general['mu_livestock_counts']
                * livestock_scale
                * (1 - bovine_fraction_on_bovaer)
                ).sum(axis=0) # over livestock types

        # (n_PTs,)
        PT_emissions_ktCO2e = (
                (PT_emissions_kg_CH4_from_cattle_on_bovaer
                 + PT_emissions_kg_CH4_from_cattle_not_on_bovaer)
                * (
                    GWP_100[GHG.CH4].magnitude # CH4 -> CO2e
                    / 1_000_000 # kg -> kt
                    ))

        ws.dist.general['emissions_obs_noise_pt'] = dist.HalfNormal(1.0).expand((n_regions,))
        ws.dist.general['emissions_obs_noise_ca'] = dist.HalfNormal(1.0)

        pt_obs_weight = jnp.array(1.0)
        ca_obs_weight = jnp.array(n_regions)
        as_years = jnp.ones((ws.n_years,) + ws.mcmc_broadcast_shape)

        for ii, (pt, hist_key) in enumerate(self.hist_ktCO2e_pt_keys.items()):
            if self._fit_to_data:
                ws.obs_weight.general[hist_key] = pt_obs_weight
                ws.obs_dist.general[hist_key] = ws.dist.general[self.nir_pt_keys[pt]]

            ws.dist.general[hist_key] = dist.Normal(
                    PT_emissions_ktCO2e[ii],
                    ws.val.general['emissions_obs_noise_pt'][ii])

        if self._fit_to_data:
            ws.obs_weight.general[self.hist_ktCO2e_ca_key] = ca_obs_weight
            ws.obs_dist.general[self.hist_ktCO2e_ca_key] = ws.dist.general[self.nir_ca_key]
        ws.dist.general[self.hist_ktCO2e_ca_key] = dist.Normal(
                PT_emissions_ktCO2e.sum(axis=0),
                ws.val.general['emissions_obs_noise_ca'])

        # predictions for future years
        # which are distinct rather than scaled random variables
        # relative to the unit_ca_key, because unit_ca_key
        # refers to previous, observed years' emissions
        if ws.phase == Phase.Posterior:
            for ii, (pt, hist_key) in enumerate(self.hist_ktCO2e_pt_keys.items()):
                ws.dist.annual[self.ktCO2e_pt_keys[pt]] = dist.Normal(
                        as_years * PT_emissions_ktCO2e[ii],
                        ws.val.general['emissions_obs_noise_pt'][ii])
            ws.dist.annual[self.ktCO2e_ca_key] = dist.Normal(
                    as_years * PT_emissions_ktCO2e.sum(axis=0),
                    ws.val.general['emissions_obs_noise_ca'])
