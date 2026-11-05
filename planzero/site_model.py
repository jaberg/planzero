"""SiteModel is a data class that contains sufficient stats to render
the /model/{site}/* pages of the site.
"""

import os
from dataclasses import dataclass

import diskcache
import numpy as np
from jax import random as jrandom

from .enums import GHG, PT, SPE, IPCC_Sector
from .nir2025 import near_zero_sector_ghgs
from .nir_model import NIR_ScenarioModel
from .regional_total_emissions_element import PseudoRegion
from .scenario_model import Computation, VarKey, years_key
from .scenario_model.registry import _rng_key_hash, registry_compute_model
from .sector_total_emissions_element import (
    PseudoSector,
)

ROOT = os.environ['PLANZERO_SITE_MODEL_SUMMARY_STATS_ROOT']

# The quantiles that are pre-computed for every time series and scalar.
QUANTILE_LEVELS = (0.025, 0.5, 0.975)


@dataclass
class PreNIR_2025_m04_Stats:
    scores_ca: dict[IPCC_Sector, dict[GHG, float]]
    weighted_divergence: float


@dataclass
class AnnualTimeSeriesStats:
    # Invariant: the arrays should all be the same length
    years: np.ndarray # dtype int32
    quantiles: dict[float, np.ndarray] # each dtype float32
    mean: np.ndarray # dtype float32

@dataclass
class ScalarStats:
    # quantile dict y -> x such that p(X < x) = y
    quantiles: dict[float, float]
    mean: float


@dataclass
class SiteModelSummaryStats:
    last_forecast_year:int

    shapes: dict[VarKey, list[int]]

    # used by e.g. models/prob/foo/sectors/bar
    # to determine which GHGs to show
    GHGs_for_sector: dict[IPCC_Sector, list[GHG]]

    all_sector_echart_ktCO2e: dict[
            IPCC_Sector|PseudoSector, AnnualTimeSeriesStats]

    # sector -> ghg (None means all GHGs) -> region
    per_sector_echart_ktCO2e: dict[
            IPCC_Sector,
            dict[GHG|None, dict[PT|PseudoRegion, AnnualTimeSeriesStats]]]

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
        total = self.all_sector_echart_ktCO2e.get(
                PseudoSector.Total_without_LULUCF)
        if total is None:
            return None
        idx = np.flatnonzero(total.years == 2050)
        if idx.size == 0:
            return None
        idx_2050 = int(idx[0])
        return ScalarStats(
                quantiles={
                    q: values[idx_2050]
                    for q, values in total.quantiles.items()},
                mean=total.mean[idx_2050],
                )


def GHGs_for_sector(
        scenario_model:NIR_ScenarioModel,
        ) -> dict[IPCC_Sector, list[GHG]]:
    """Return the GHGs that this model actually represents, by sector.

    The per-sector charts are only pre-computed for these (sector, ghg)
    pairs, so the site's sector pages must only ask for these.
    """
    rval:dict[IPCC_Sector, set[GHG]] = {}
    for sector, ghg in scenario_model.sector_ghg_observations:
        rval.setdefault(sector, set()).add(ghg)
    for sector in scenario_model.sector_sum_over_ghg:
        rval.setdefault(sector, set())
    return {
            sector: [ghg for ghg in GHG if ghg in ghgs]
            for sector, ghgs in rval.items()}


def _annual_stats_from_sample(
        sample,
        years:np.ndarray,
        ) -> AnnualTimeSeriesStats:
    """Summarize an array of shape (n_years, n_samples)."""
    arr = np.asarray(sample)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f'expected a 2-d sample, got shape {arr.shape}')
    if arr.shape[0] != len(years):
        raise ValueError(
                f'sample has {arr.shape[0]} years, expected {len(years)}')

    quantile_values = np.quantile(arr, q=QUANTILE_LEVELS, axis=1)
    mean_values = np.mean(arr, axis=1)
    return AnnualTimeSeriesStats(
            years=np.asarray(years, dtype=np.int32),
            quantiles={
                float(q): np.asarray(values, dtype=np.float32)
                for q, values in zip(QUANTILE_LEVELS, quantile_values)},
            mean=np.asarray(mean_values, dtype=np.float32),
            )


def _scalar_stats_from_sample(sample) -> ScalarStats:
    arr = np.asarray(sample).reshape(-1)
    quantile_values = np.quantile(arr, q=QUANTILE_LEVELS)
    return ScalarStats(
            quantiles={
                float(q): float(value)
                for q, value in zip(QUANTILE_LEVELS, quantile_values)},
            mean=float(np.mean(arr)),
            )


def _years_from_comp(comp:Computation) -> np.ndarray:
    years = np.asarray(comp.ndarray_d[years_key])
    if years.ndim > 1:
        years = years[:, 0]
    return np.asarray(years, dtype=np.int32)


def _shapes_from_comp(comp:Computation) -> dict[VarKey, list[int]]:
    rval = {}
    for vk, nda in comp._ndarray_d.items():
        rval[vk] = nda.shape

    for vk, dist in comp._dist_d.items():
        dshape = dist.shape()
        rval.setdefault(vk, dshape)
        assert rval[vk] == dshape
    return rval


def _pre_nir_2025_m04_stats(
        scenario_model:NIR_ScenarioModel,
        comp:Computation,
        ) -> None|PreNIR_2025_m04_Stats:
    total_key = scenario_model.prenir_2025_m04_total_key()
    try:
        total_val = comp.ndarray_d[total_key]
    except KeyError:
        # This model does not include the Pre-NIR element.
        return None

    near_zero = near_zero_sector_ghgs()
    scores_ca = {}
    for sector in IPCC_Sector:
        scores_ca[sector] = {}
        for ghg in GHG:
            if (sector, ghg) in near_zero:
                continue
            sector_ghg_key = scenario_model.prenir_2025_m04_sector_ghg_key(
                    sector, ghg)
            sector_ghg_val = np.asarray(comp.ndarray_d[sector_ghg_key])
            assert sector_ghg_val.shape == (1,), sector_ghg_val.shape
            scores_ca[sector][ghg] = float(sector_ghg_val[0])

    total_val = np.asarray(total_val)
    assert total_val.shape == (1,), total_val.shape
    return PreNIR_2025_m04_Stats(
            scores_ca=scores_ca,
            weighted_divergence=float(total_val[0]),
            )


def _p_net_zero_2050(
        scenario_model:NIR_ScenarioModel,
        comp:Computation,
        years:np.ndarray,
        ) -> None|float:
    idx = np.flatnonzero(years == 2050)
    if idx.size == 0:
        return None
    idx_2050 = int(idx[0])
    total_ems = comp.ndarray_d[
            scenario_model.sum_over_sectors[False].ktCO2e_ca]
    return float(np.mean(np.asarray(total_ems)[idx_2050] < 0))


def _all_sector_echart_stats(
        scenario_model:NIR_ScenarioModel,
        comp:Computation,
        years:np.ndarray,
        ) -> dict[IPCC_Sector|PseudoSector, AnnualTimeSeriesStats]:
    samples = scenario_model.all_sector_echart_ktCO2e_sample(comp)
    return {
            sector: _annual_stats_from_sample(sample, years)
            for sector, sample in samples.items()}


def _per_sector_echart_stats(
        scenario_model:NIR_ScenarioModel,
        comp:Computation,
        years:np.ndarray,
        ) -> dict[
            IPCC_Sector,
            dict[GHG|None, dict[PT|PseudoRegion, AnnualTimeSeriesStats]]]:
    rval = {}
    ghgs_for_sector = GHGs_for_sector(scenario_model)
    for sector in scenario_model.sector_sum_over_ghg:
        rval[sector] = {}
        for ghg in [None] + list(ghgs_for_sector[sector]):
            samples = scenario_model.per_sector_echart_ktCO2e_sample(
                    comp=comp,
                    sector=sector,
                    ghg=ghg)
            rval[sector][ghg] = {
                    region: _annual_stats_from_sample(sample, years)
                    for region, sample in samples.items()}
    return rval


def _cost_per_tCO2e_stats(
        scenario_model:NIR_ScenarioModel,
        comp:Computation,
        ablated_model:NIR_ScenarioModel,
        ablated_comp:Computation,
        strategy_elem_id:str,
        ) -> ScalarStats:
    strategy = scenario_model.model_elements[strategy_elem_id]
    CAD_with = scenario_model.all_SPE_echart_cost_sample(comp)
    ktCO2e_with = scenario_model.all_sector_echart_ktCO2e_sample(comp)
    CAD_without = ablated_model.all_SPE_echart_cost_sample(ablated_comp)
    ktCO2e_without = ablated_model.all_sector_echart_ktCO2e_sample(
            ablated_comp)

    CAD_delta = CAD_with[strategy.cost_centre] - CAD_without[strategy.cost_centre]
    ktCO2e_delta = (
            ktCO2e_with[PseudoSector.Total_with_LULUCF]
            - ktCO2e_without[PseudoSector.Total_with_LULUCF])
    tCO2e_delta = np.asarray(ktCO2e_delta) * 1000
    # tCO2e_delta has shape (n_years, n_samples)
    estimates = (
            np.asarray(CAD_delta).sum(axis=0)
            / tCO2e_delta.sum(axis=0))
    return _scalar_stats_from_sample(estimates)


def _national_emission_impact_stats(
        scenario_model:NIR_ScenarioModel,
        comp:Computation,
        ablated_model:NIR_ScenarioModel,
        ablated_comp:Computation,
        years:np.ndarray,
        ) -> dict[IPCC_Sector|PseudoSector, AnnualTimeSeriesStats]:
    with_strategy = scenario_model.all_sector_echart_ktCO2e_sample(comp)
    without_strategy = ablated_model.all_sector_echart_ktCO2e_sample(
            ablated_comp)
    return {
            sector: _annual_stats_from_sample(
                np.asarray(with_strategy[sector])
                - np.asarray(without_strategy[sector]),
                years)
            for sector in with_strategy}


def _national_cost_impact_stats(
        scenario_model:NIR_ScenarioModel,
        comp:Computation,
        ablated_model:NIR_ScenarioModel,
        ablated_comp:Computation,
        years:np.ndarray,
        ) -> dict[SPE, AnnualTimeSeriesStats]:
    with_strategy = scenario_model.all_SPE_echart_cost_sample(comp)
    without_strategy = ablated_model.all_SPE_echart_cost_sample(ablated_comp)
    return {
            spe: _annual_stats_from_sample(
                np.asarray(with_strategy[spe])
                - np.asarray(without_strategy[spe]),
                years)
            for spe in with_strategy}


def _site_model_summary_stats_no_cache(
        scenario_model_registry_name:str,
        scenario_model:NIR_ScenarioModel,
        ablations:dict[str, NIR_ScenarioModel],
        seed: int,
        quiet:bool,
        ) -> SiteModelSummaryStats:

    base_comp = registry_compute_model(
            model=scenario_model,
            model_name=scenario_model_registry_name,
            run_compute_posterior=True,
            quiet=quiet,
            seed=seed)

    years = _years_from_comp(base_comp)

    rval = SiteModelSummaryStats(
            last_forecast_year=years[-1],
            shapes=_shapes_from_comp(base_comp),
            GHGs_for_sector=GHGs_for_sector(scenario_model),
            all_sector_echart_ktCO2e=_all_sector_echart_stats(
                scenario_model, base_comp, years),
            per_sector_echart_ktCO2e=_per_sector_echart_stats(
                scenario_model, base_comp, years),
            prenir_2025_m04=_pre_nir_2025_m04_stats(
                scenario_model, base_comp),
            p_net_zero_2050=_p_net_zero_2050(
                scenario_model, base_comp, years),
            cost_per_tCO2e_quantiles={},
            national_emission_impact={},
            national_cost_impact={},
            )

    for strategy_id, ablated_model in ablations.items():
        model_name = f'{scenario_model_registry_name}_{strategy_id}'
        ablated_comp = registry_compute_model(
                model=ablated_model,
                model_name=model_name,
                run_compute_posterior=True,
                quiet=False,
                seed=seed)

        rval.cost_per_tCO2e_quantiles[strategy_id] = _cost_per_tCO2e_stats(
                scenario_model,
                base_comp,
                ablated_model,
                ablated_comp,
                strategy_id)
        rval.national_emission_impact[strategy_id] = \
                _national_emission_impact_stats(
                    scenario_model,
                    base_comp,
                    ablated_model,
                    ablated_comp,
                    years)
        rval.national_cost_impact[strategy_id] = _national_cost_impact_stats(
                scenario_model,
                base_comp,
                ablated_model,
                ablated_comp,
                years)

    return rval


# saves having to recompute after each file change
# during e.g. site edits or between unit test attempts
_disk_cache = diskcache.Cache(ROOT)
_mem_cache = {}

def site_model_summary_stats(
        scenario_model_registry_name:str,
        scenario_model:NIR_ScenarioModel,
        ablations:dict[str, NIR_ScenarioModel],
        seed: int,
        quiet:bool=False,
        ) -> SiteModelSummaryStats:

    rng_key = jrandom.key(seed)

    model_version_hash = scenario_model.version_hash()
    model_seed_hash = _rng_key_hash(rng_key)

    ablation_hash = tuple(sorted(
        (strategy_id, model.version_hash())
        for strategy_id, model in ablations.items()))

    cache_key = (
            scenario_model_registry_name,
            model_version_hash,
            model_seed_hash,
            ablation_hash,
            )

    if cache_key in _mem_cache:
        rval = _mem_cache[cache_key]
    elif cache_key in _disk_cache:
        rval = _disk_cache[cache_key]
        _mem_cache[cache_key] = rval
    else:
        rval = _site_model_summary_stats_no_cache(
            scenario_model_registry_name,
            scenario_model,
            ablations,
            seed=seed,
            quiet=quiet)
        _mem_cache[cache_key] = rval
        _disk_cache[cache_key] = rval
    assert isinstance(rval, SiteModelSummaryStats)
    return rval
