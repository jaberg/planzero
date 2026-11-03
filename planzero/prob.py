"""
BaseModels for Probabilistic Models
"""
from __future__ import annotations

from .enums import GHG, SPE, IPCC_Sector
from .html import html_by_ghg
from .nir_model import NIR_ScenarioModel
from .scenario_model import (
    Computation,
    ModelElement,
    registry_basename,
    registry_compute_model,
)
from .singleton_registry import SingletonRegistry
from .site_model import SiteModelSummaryStats, site_model_summary_stats
from .sparkline_echart_helper import (
    national_cost_impact_echart_from_stats,
    national_emissions_by_sector_echart_from_stats,
    national_emissions_impact_echart_from_stats,
    sectoral_emissions_by_region_echart_from_stats,
)

registry = SingletonRegistry()


class SiteInference:
    """A specific simulation (no caller configuration, all pre-loaded)
    to be listed on the site's simulation page.

    It may correspond to a single model, or to multiple models that comprise
    an ablation study.
    """

    # include this object in the registry of models to show on the site
    # e.g. on the models/ page
    # This should be over-ridden with a derived class class variable
    include_in_registry: bool

    # the "model_name" of the model (or baseline model without any ablation)
    _scenario_model_registry_name: str

    # the baseline model without any ablation
    _scenario_model: NIR_ScenarioModel

    # strategies are site-visible model elements that can be omitted
    # for the purpose of ablative analysis
    _strategies: dict[str, ModelElement]

    # ablations are the ScenarioModels that result from removing each strategy
    _ablations: dict[str, NIR_ScenarioModel]

    # barriers are site-visible model elements that implement the mathematical
    # relationship between the things we control and the results we seek to influence.
    _barriers: dict[str, ModelElement]

    site_model_summary_stats:SiteModelSummaryStats

    registry_computation_seed:int = 42

    def __init__(self, *,
                 scenario_model_registry_name:str,
                 scenario_model:NIR_ScenarioModel,
                 ):
        self._scenario_model_registry_name = scenario_model_registry_name
        self._scenario_model = scenario_model
        self._strategies = {
                elem_id: elem
                for elem_id, elem in scenario_model.model_elements.items()
                if getattr(elem, 'is_strategy', False)}
        self._barriers = {
                elem_id: elem
                for elem_id, elem in scenario_model.model_elements.items()
                if getattr(elem, 'is_barrier', False)}
        self._ablations = {}
        for strat_id in self._strategies:
            self._add_ablation(strat_id)
        self.site_summary_stats = site_model_summary_stats(
                self._scenario_model_registry_name,
                scenario_model=self._scenario_model,
                ablations=self._ablations,
                seed=self.registry_computation_seed,
                )

    def _add_ablation(self, strategy_elem_id):
        assert strategy_elem_id not in self._ablations
        base = self._scenario_model
        # It might take a few iterations to get the nuances of how to
        # clone the base model while omitting one of the elements.
        #
        # For `ablation` to do the right thing a strategy element must only
        # @define keys that are always accessed with a default value.
        # The typical pattern is
        # * element A creates a key
        # * elements A, B, C etc. @access the key with a default value
        # * strategy element S includes a method that @defines the key
        ablation = NIR_ScenarioModel(
                last_forecast_year=base.mv.year_0 + base.mv.n_posterior_years - 1,
                data_cutoff=base.data_cutoff,
                expect_all_sectors=base.expect_all_sectors,
                num_warmup=base.num_warmup,
                num_samples=base.num_samples,
                thinning=base.thinning,
                add_builtins=False,
                )
        for elem_id, elem in base.model_elements.items():
            if elem_id != strategy_elem_id:
                ablation.add_element(
                        elem,
                        mcmc_group=base.mcmc_group_by_elemid[elem_id])
        ablation.mv.forgettable_dist_keys.update(
                base.mv.forgettable_dist_keys)
        ablation.mv.forgettable_val_keys.update(
                base.mv.forgettable_val_keys)
        self._ablations[strategy_elem_id] = ablation

    @property
    def scenario_model_registry_name(self) -> str:
        # this is the first part of the 3-part registry_basename
        # {model_name}_{version_hash}_{seed_hash}
        return self._scenario_model_registry_name

    @property
    def scenario_model(self) -> NIR_ScenarioModel:
        return self._scenario_model

    def registry_basename(self) -> str:
        # called from __main__
        # {model_name}_{version_hash}_{seed_hash}
        return registry_basename(
                model=self._scenario_model,
                model_name=self._scenario_model_registry_name,
                seed=self.registry_computation_seed)

    def registry_computation(self, quiet:bool=False) -> None:
        # called from __main__
        def _registry_compute_model(
                ablate_strategy_elem_id:str|None=None,
                ):
            if ablate_strategy_elem_id is None:
                model = self._scenario_model
                model_name = self._scenario_model_registry_name
            else:
                model = self._ablations[ablate_strategy_elem_id]
                model_name = f'{self.scenario_model_registry_name}_{ablate_strategy_elem_id}'
            registry_compute_model(
                    model=model,
                    model_name=model_name,
                    run_compute_posterior=False,
                    quiet=quiet,
                    seed=self.registry_computation_seed)

        _registry_compute_model()
        for strategy_elem_id in self._ablations:
            _registry_compute_model(ablate_strategy_elem_id=strategy_elem_id)

    def registry_compute_model(
            self,
            compute_posterior:bool=True,
            quiet:bool=True,
            ablate_strategy_elem_id:str|None=None,
            ) -> Computation:
        if ablate_strategy_elem_id is None:
            model = self._scenario_model
            model_name = self._scenario_model_registry_name
        else:
            model = self._ablations[ablate_strategy_elem_id]
            model_name = f'{self.scenario_model_registry_name}_{ablate_strategy_elem_id}'
        return registry_compute_model(
                model=model,
                model_name=model_name,
                run_compute_posterior=compute_posterior,
                quiet=quiet,
                seed=self.registry_computation_seed)

    @property
    def comp(self) -> Computation:
        # Compatibility accessor for callers that still expect a Computation.
        return self.registry_compute_model()

    def version_hash(self) -> str:
        """Return a hash string that uniquely identifies
        this model, including all of its elements and their versions.

        It is used to save & restore cached inferences on GitHub.
        """
        return self._scenario_model.version_hash()

    @classmethod
    def __init_subclass__(cls):
        super().__init_subclass__()
        if getattr(cls, 'include_in_registry', False):
            registry.add_class(cls)

    @property
    def show_on_models_page(self) -> bool:
        return True

    def one_line_description(self) -> str:
        raise NotImplementedError()

    @property
    def short_description(self) -> str:
        rval = self.__class__.__doc__
        if rval is None:
            return ''
        else:
            return rval

    @property
    def predicted_emissions_2050_MtCO2e_bounds_str(self) -> str:
        stats = self.site_summary_stats.predicted_emissions_2050_ktCO2e
        if stats is None:
            return 'N/A'
        unit_conversion = 1 / 1000 # ktCO2e -> MtCO2e
        low = stats.quantiles[0.025] * unit_conversion
        high = stats.quantiles[0.975] * unit_conversion
        rval = f"{low:.1f} - {high:.1f} Mt{html_by_ghg[GHG.CO2]}e"
        return rval

    def prediction_scores_prenir_2025_m04(self) -> dict:
        stats = self.site_summary_stats.prenir_2025_m04
        if stats is None:
            return {
                    'weighted_divergence': float('nan'),
                    'scores_ca': {},
                    }
        return {
                'weighted_divergence': stats.weighted_divergence,
                'scores_ca': stats.scores_ca,
                }

    def challenge_netzero_2050(self) -> float:
        rval = self.site_summary_stats.p_net_zero_2050
        if rval is None:
            raise NotImplementedError(
                    'this model does not include a prediction for 2050')
        return rval

    def challenge_result_url(self, challenge_name) -> str:
        return ''

    @property
    def show_prediction_quality(self) -> bool:
        return False

    @property
    def pretty_name(self) -> str:
        """Standard short name for this model for page content"""
        return self.__class__.__name__

    @property
    def url_name(self) -> str:
        """the name for this model in urls e.g. /model/{url_name}/strategy/{foo}"""
        return self.__class__.__name__

    @property
    def posts_developing_this_page(self) -> list[str]:
        return []

    def affected_sectors_by_strategy(self) -> dict:
        # This could be determined empirically
        raise NotImplementedError()

    @property
    def strategies(self) -> dict[str, SiteModelElement]:
        rval = {}
        for key, val in self._strategies.items():
            assert isinstance(val, SiteModelElement)
            rval[key] = val
        return rval

    @property
    def barriers(self) -> dict[str, SiteModelElement]:
        rval = {}
        for key, val in self._barriers.items():
            assert isinstance(val, SiteModelElement)
            rval[key] = val
        return rval

    def GHGs_for_sector(self, sector:IPCC_Sector) -> list[GHG]:
        # called by e.g. models/prob/foo/sectors/bar
        return self.site_summary_stats.GHGs_for_sector.get(sector, [])

    def uncertain_sparkline_matrix_echart(self, div_id, v_unit):
        return national_emissions_by_sector_echart_from_stats(
                ktCO2e_stats=self.site_summary_stats.all_sector_echart_ktCO2e,
                div_id=div_id,
                v_unit=v_unit,
                model_name=self.url_name,
                )

    def sector_echart(self, sector, ghg:GHG|None, v_unit):
        ktCO2e_stats = \
                self.site_summary_stats.per_sector_echart_ktCO2e[sector][ghg]
        return sectoral_emissions_by_region_echart_from_stats(
                ktCO2e_stats=ktCO2e_stats,
                sector=sector,
                ghg=ghg,
                div_id=(f'{self.__class__.__name__}'
                        f'_regional_sparkline_echart'
                        f'_{ghg.value if ghg else "all"}'),
                v_unit=v_unit)

    def national_emissions_impact_echart(
            self,
            strategy_elem_id:str,
            div_id:str,
            v_unit:str):
        return national_emissions_impact_echart_from_stats(
                impact_stats=(
                    self.site_summary_stats.national_emission_impact[
                        strategy_elem_id]),
                div_id=div_id,
                v_unit=v_unit,
                model_name=f'{self.url_name}',
                )

    def national_cost_impact_echart(
            self,
            strategy_elem_id:str,
            div_id:str,
            v_unit:str):
        return national_cost_impact_echart_from_stats(
                cost_impact_stats=(
                    self.site_summary_stats.national_cost_impact[
                        strategy_elem_id]),
                div_id=div_id,
                v_unit=v_unit,
                model_name=self.url_name,
                )

    def cost_per_tCO2e(self, strategy_elem_id) -> tuple[float, float]:
        """Return 95% CI for strategy's cost
        """
        stats = self.site_summary_stats.cost_per_tCO2e_quantiles[
                strategy_elem_id]
        return stats.quantiles[0.025], stats.quantiles[0.975]



class SiteModelElement(ModelElement):
    """Contract for attributes used to render barriers and strategies.
    """

    is_strategy: bool = False
    is_barrier: bool = False

    @property
    def pretty_name(self) -> str:
        """Name used in page content, to be read naturally."""
        return self.__class__.__name__.replace('_', ' ')

    @property
    def extra_ipcc_sectors(self) -> list[IPCC_Sector]:
        """Sectors related to this modelling element.
        To be shown in e.g. the table on the /models/prob/{model} page.
        """
        return []

    def description_html(self, context:None|dict=None) -> str:
        raise NotImplementedError()

    def see_also_html(self, context:None|dict=None) -> list[str]:
        return []

    @property
    def cost_centre(self) -> SPE:
        # for Strategies, what SPE balance over time should be used to compute
        # cost_per_tCO2e ?
        raise NotImplementedError()
