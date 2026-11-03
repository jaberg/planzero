"""SiteModel is a data class that contains sufficient stats to render
the /model/{site}/* pages of the site.
"""

from dataclasses import dataclass

import numpy as np
from jax import random as jrandom

from .enums import GHG, PT, SPE, IPCC_Sector
from .nir2025 import near_zero_sector_ghgs
from .nir_model import NIR_ScenarioModel
from .regional_total_emissions_element import PseudoRegion
from .scenario_model import Computation, years_key
from .scenario_model.registry import _rng_key_hash, registry_compute_model
from .sector_total_emissions_element import (
    PseudoSector,
)


@dataclass
class PreNIR_2025_m04_Stats:
    scores_ca: dict[IPCC_Sector, dict[GHG, float]]
    weighted_divergence: float


@dataclass
class AnnualTimeSeriesStats:
    # Invariant: the lists should all be the same length
    years: list[int]
    quantiles: dict[float, list[float]]
    mean: list[float]

@dataclass
class ScalarStats:
    # quantile dict y -> x such that p(X < x) = y
    quantiles: dict[float, float]
    mean: float


@dataclass
class SiteModelSummaryStats:
    scenario_model:NIR_ScenarioModel
    seed: int

    # used by e.g. models/prob/foo/sectors/bar
    # to determine which GHGs to show
    GHGs_for_sector: dict[IPCC_Sector, list[GHG]]

    all_sector_echart_ktCO2e: dict[
            IPCC_Sector|PseudoSector, AnnualTimeSeriesStats]

    per_sector_echart_ktCO2e: dict[
            PT|PseudoRegion, AnnualTimeSeriesStats]

    prenir_2025_m04: None|PreNIR_2025_m04_Stats

    p_net_zero_2050: None|float

    cost_per_tCO2e_quantiles: dict[
            str, # strategy_element_id
            ScalarStats]

    national_emission_impact: dict[
            str, # strategy element id
            dict[IPCC_Sector|PseudoSector, AnnualTimeSeriesStats]]

    national_cost_impact: dict[
            str, # strategy element id
            dict[SPE, AnnualTimeSeriesStats]]

    @property
    def predicted_emissions_2050_ktCO2e(self) -> None|ScalarStats:
        # TODO: look these up from self.all_sector_echart_ktCO2e
        # if the years include 2050
        raise NotImplementedError()


def GHGs_for_sector() -> dict[IPCC_Sector, list[GHG]]:
    rval = {}
    near_zeros = near_zero_sector_ghgs()
    for (sector, ghg) in near_zeros:
        rval.setdefault(sector, []).append(ghg)
    return rval


def predicted_emissions_2050_MtCO2e_bounds_95(
        model:NIR_ScenarioModel,
        comp:Computation,
        ) -> tuple[float, float]:
    ktCO2e_sample = comp.ndarray_d[
            model.sum_over_sectors[False].ktCO2e_ca]
    model_years = np.asarray(comp.ndarray_d[years_key])
    match_mask = (model_years == 2050)
    if np.any(match_mask):
        model_idx_of_prediction_year = np.argmax(match_mask)
    else:
        raise NotImplementedError('no prediction for 2050')
    ktCO2e_2050 = ktCO2e_sample[model_idx_of_prediction_year]
    n_mcmc, = ktCO2e_2050.shape
    assert n_mcmc > 100
    unit_conversion = 1 / 1000 # ktCO2e -> MtCO2e
    lower, upper = np.quantile(ktCO2e_2050, [0.025, 0.975]) * unit_conversion
    return lower, upper


def all_sector_echart_ktCO2e_sample(
        self,
        comp:Computation,
        ) -> dict[IPCC_Sector|PseudoSector, jnp.ndarray]:
    ktCO2e_sample = {}
    for sector in IPCC_Sector:
        ktCO2e_sample[sector] = comp.ndarray_d[
                self.sector_sum_over_ghg[sector].ktCO2e_ca]
    ktCO2e_sample[PseudoSector.Total_with_LULUCF] = comp.ndarray_d[
            self.sum_over_sectors[True].ktCO2e_ca]
    ktCO2e_sample[PseudoSector.Total_without_LULUCF] = comp.ndarray_d[
            self.sum_over_sectors[False].ktCO2e_ca]
    return ktCO2e_sample


def per_sector_echart_ktCO2e_sample(
        self,
        comp:Computation,
        sector:IPCC_Sector,
        ghg:GHG|None
        ) -> dict[PT|PseudoRegion, jnp.ndarray]:

    ktCO2e_sample:dict[PT|PseudoRegion, jnp.ndarray] = {}
    regions = [pt for pt in PT if pt != PT.XX] + [PseudoRegion.NationalTotal]
    if ghg is None:
        for region in regions:
            if isinstance(region, PT):
                ktCO2e_sample[region] = comp.ndarray_d[
                        self.sector_sum_over_ghg[sector].ktCO2e_pt[region]]
            else:
                ktCO2e_sample[region] = comp.ndarray_d[
                        self.sector_sum_over_ghg[sector].ktCO2e_ca]
    else:
        for region in regions:
            if isinstance(region, PT):
                ktCO2e_sample[region] = comp.ndarray_d[
                        self.sector_ghg_observations[sector, ghg].ktCO2e_pt[region]]
            else:
                ktCO2e_sample[region] = comp.ndarray_d[
                        self.sector_ghg_observations[sector, ghg].ktCO2e_ca]
    return ktCO2e_sample

def prediction_scores_prenir_2025_m04(self, comp:Computation):
    scores_ca = {}
    near_zero = near_zero_sector_ghgs()
    for sector in IPCC_Sector:
        scores_ca[sector] = {}
        for ghg in GHG:
            if (sector, ghg) not in near_zero:
                sector_ghg_key = self.prenir_2025_m04_sector_ghg_key(sector, ghg)
                sector_ghg_val = np.asarray(comp.ndarray_d[sector_ghg_key])
                # remove the mcmc dim
                assert sector_ghg_val.shape == (1,)
                scores_ca[sector][ghg] = float(sector_ghg_val[0])
    total_key = self.prenir_2025_m04_total_key()
    total_val = comp.ndarray_d[total_key]
    assert total_val.shape == (1,)
    rval = {
            'weighted_divergence': float(total_val[0]),
            'scores_ca': scores_ca,
            }
    return rval


# P-Emissions-Below-Thresh
def p_net_zero_2050(self, comp:Computation) -> float:
    # TODO: assert years end at 2050
    idx_2050 = -1
    total_ems = comp.get_ndarray(
            self.sum_over_sectors[False].ktCO2e_ca,
            sample_if_necessary=True)
    rval = float(jnp.mean(total_ems[idx_2050] < 0))
    return rval

def national_emissions_impact_quantiles(
        self,
        strategy_elem_id:str,
        div_id:str,
        v_unit:str):
    sample_with = self.scenario_model.all_sector_echart_ktCO2e_sample(
            self.comp)
    ablated = self._ablations[strategy_elem_id]
    ablated_comp = self.registry_compute_model(
            ablate_strategy_elem_id=strategy_elem_id)
    sample_without = ablated.all_sector_echart_ktCO2e_sample(ablated_comp)
    rval = national_emissions_impact_echart(
            with_strategy_ktCO2e_sample=sample_with,
            without_strategy_ktCO2e_sample=sample_without,
            year_0=self.comp.model.mv.year_0,
            n_years=self.comp.n_years,
            n_samples=self.comp.n_mcmc,
            div_id=div_id,
            v_unit=v_unit,
            model_name=f'{self.url_name}',
            )
    return rval

def national_cost_impact_quantiles(
        self,
        strategy_elem_id:str,
        div_id:str,
        v_unit:str):
    sample_with = self.scenario_model.all_SPE_echart_cost_sample(
            self.comp)
    ablated = self._ablations[strategy_elem_id]
    ablated_comp = self.registry_compute_model(
            ablate_strategy_elem_id=strategy_elem_id)
    sample_without = ablated.all_SPE_echart_cost_sample(ablated_comp)
    rval = national_cost_impact_echart(
            costs_with_strategy=sample_with,
            costs_without_strategy=sample_without,
            year_0=self.comp.model.mv.year_0,
            n_years=self.comp.n_years,
            div_id=div_id,
            v_unit='Billions CAD',
            model_name=self.url_name,
            )
    return rval

def cost_per_tCO2e(self, strategy_elem_id) -> tuple[float, float]:
    """Return 95% CI for strategy's cost
    """
    strategy = self.strategies[strategy_elem_id]
    CAD_with = self.scenario_model.all_SPE_echart_cost_sample(
            self.comp)
    ktCO2e_with = self.scenario_model.all_sector_echart_ktCO2e_sample(
            self.comp)
    ablated_model = self._ablations[strategy_elem_id]
    ablated_comp = self.registry_compute_model(
            ablate_strategy_elem_id=strategy_elem_id)
    CAD_without = ablated_model.all_SPE_echart_cost_sample(ablated_comp)
    ktCO2e_without = ablated_model.all_sector_echart_ktCO2e_sample(ablated_comp)

    CAD_delta = CAD_with[strategy.cost_centre] - CAD_without[strategy.cost_centre]
    ktCO2e_delta = (ktCO2e_with[PseudoSector.Total_with_LULUCF]
                    - ktCO2e_without[PseudoSector.Total_with_LULUCF])
    tCO2e_delta = ktCO2e_delta * 1000
    # tCO2e_delta has shape (n_years, n_samples)
    estimates = (
            CAD_delta.sum(axis=0)
            / tCO2e_delta.sum(axis=0))
    lower, upper = np.quantile(
            estimates,
            q=[0.025, 0.975])
    return lower, upper


def _site_model_summary_stats_no_cache(
        scenario_model_registry_name:str,
        scenario_model:NIR_ScenarioModel,
        ablations:dict[str, NIR_ScenarioModel],
        seed: int,
        ) -> SiteModelSummaryStats:

    base_comp = registry_compute_model(
            model=scenario_model,
            model_name=scenario_model_registry_name,
            run_compute_posterior=True,
            quiet=False,
            seed=seed)

    rval = SiteModelSummaryStats(
            scenario_model=scenario_model,
            seed=seed,
            GHGs_for_sector=GHGs_for_sector(),
            predicted_emissions_2050_ktCO2e_quantiles=predicted_emissions_2050_ktCO2e_quantiles(base_comp),
            all_sector_echart_ktCO2e_quantiles=all_sector_echart_ktCO2e_quantiles(
                base_comp),
            per_sector_echart_ktCO2e_quantiles=per_sector_echart_ktCO2e_quantiles(
                base_comp),
            prenir_2025_m04_stats=prenir_2025_m04_stats(base_comp),
            p_net_zero_2050=p_net_zero_2050(base_comp),
            cost_per_tCO2e_quantiles={},
            national_emission_impact_quantiles={},
            national_cost_impact_quantiles={},
            )

    for strategy_id, ablated_model in ablations.items():
        model_name = f'{scenario_model_registry_name}_{strategy_id}'
        ablated_comp = registry_compute_model(
                model=ablated_model,
                model_name=model_name,
                run_compute_posterior=True,
                quiet=False,
                seed=seed)

        rval.cost_per_tCO2e_quantiles[strategy_id] = cost_per_tCO2e_quantiles(
                base_comp, ablated_comp)
        rval.national_emission_impact_quantiles[strategy_id] = national_emission_impact_quantiles(
                base_comp, ablated_comp)
        rval.national_cost_impact_quantiles[strategy_id] = national_cost_impact_quantiles(
                base_comp, ablated_comp)

    return rval



_site_model_cache = {}

def site_model_summary_stats(
        scenario_model_registry_name:str,
        scenario_model:NIR_ScenarioModel,
        ablations:dict[str, NIR_ScenarioModel],
        seed: int,
        ) -> SiteModelSummaryStats:

    rng_key = jrandom.key(seed)

    model_version_hash = scenario_model.version_hash()
    model_seed_hash = _rng_key_hash(rng_key)

    cache_key = (model_version_hash, model_seed_hash)

    if cache_key in _site_model_cache:
        return _site_model_cache[cache_key]
    else:
        rval = _site_model_summary_stats_no_cache(
            scenario_model_registry_name,
            scenario_model,
            ablations,
            seed)
        _site_model_cache[cache_key] = rval
        return rval
