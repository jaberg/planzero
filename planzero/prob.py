"""
BaseModels for Probabilistic Models
"""

from . import ablation
from .annual_emission_results import AnnualEmissionResults
from .enums import GHG
from .html import html_by_ghg
from .nir2025 import near_zero_sector_ghgs
from .nir_model import NIR_ScenarioModel
from .scenario_model import Computation, registry_compute_model
from .singleton_registry import SingletonRegistry
from .sparkline_echart_helper import (
    national_emissions_by_sector_echart_from_sample,
    sectoral_emissions_by_region_echart_from_sample,
)

registry = SingletonRegistry()


class SiteInference:
    """A specific simulation (no caller configuration, all pre-loaded)
    to be listed on the site's simulation page.
    """

    # include this object in the registry of models to show on the site
    # e.g. on the models/ page
    # This should be over-ridden with a derived class class variable
    include_in_registry: bool

    # str if if this SiteInference is an ablated model
    # None if this SiteInference is not an ablated model
    strategy_id: str|None

    _scenario_model_registry_name: str
    _scenario_model: NIR_ScenarioModel

    def __init__(self, *,
                 scenario_model_registry_name:str,
                 scenario_model:NIR_ScenarioModel,
                 strategy_id:str|None=None,
                 ):
        self._scenario_model_registry_name = scenario_model_registry_name
        self._scenario_model = scenario_model
        self.strategy_id = strategy_id

    @property
    def scenario_model_registry_name(self) -> str:
        # this is attribute is used by GitHub workflow via
        # __main__.py to manage the artifact cache
        raise NotImplementedError()

    @property
    def scenario_model(self) -> NIR_ScenarioModel:
        return self._scenario_model

    def registry_compute_model(
            self,
            compute_posterior=True,
            quiet=True,
            ) -> Computation:
        return registry_compute_model(
                model=self._scenario_model,
                model_name=self._scenario_model_registry_name,
                cache_posterior=compute_posterior,
                run_compute_posterior=compute_posterior,
                quiet=quiet,
                seed=42)

    @property
    def comp(self) -> Computation:
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
        low, high = self.scenario_model.predicted_emissions_2050_MtCO2e_bounds_95(self.comp)
        rval = f"{low:.1f} - {high:.1f} Mt{html_by_ghg[GHG.CO2]}e"
        return rval

    def prediction_scores_prenir_2025_06(self) -> dict:
        raise NotImplementedError()

    def challenge_result_url(self, challenge_name) -> str:
        return ''

    @property
    def show_prediction_quality(self) -> bool:
        return False

    @property
    def pretty_name(self) -> str:
        return self.__class__.__name__.replace(' ', '_')

    @property
    def posts_developing_this_page(self) -> list[str]:
        return []

    @property
    def strategy_ids(self) -> list[str]:
        as_id = self.ablation_study_id
        if as_id is None:
            return []
        else:
            return ablation.registry[as_id].strategy_ids

    @property
    def affected_sectors_by_strategy(self) -> dict:
        raise NotImplementedError()

    @property
    def strategies(self) -> dict:
        if self.ablation_study is None:
            rval = {}
        else:
            assert None not in self.ablation_study.strategies
            rval = {
                    key: val
                    for key, val in self.ablation_study.strategies.items()
                    if key != self.strategy_id
                    }
        return rval

    @property
    def barriers(self) -> dict:
        if self.ablation_study is None:
            return {}
        else:
            return self.ablation_study.barriers

    @property
    def ablation_study_id(self) -> str|None:
        return None

    @property
    def ablation_study(self) -> ablation.AblationStudy | None:
        as_id = self.ablation_study_id
        if as_id is None:
            return None
        else:
            return ablation.registry[as_id]

    def emission_results(self) -> AnnualEmissionResults:
        raise NotImplementedError()

    def uncertain_sparkline_matrix_echart(self, div_id, v_unit):
        # TODO: make it work, then move to base class
        ktCO2e_sample = self.scenario_model.all_sector_echart_ktCO2e_sample(self.comp)

        rval = national_emissions_by_sector_echart_from_sample(
                ktCO2e_sample=ktCO2e_sample,
                year_0=self.comp.model.mv.year_0,
                n_years=self.comp.n_years,
                n_samples=self.comp.n_mcmc,
                div_id=div_id,
                v_unit=v_unit,
                model_name=self.__class__.__name__,
                )
        return rval

    def sector_echart(self, sector, ghg:GHG|None, v_unit):
        ktCO2e_sample = self.scenario_model.per_sector_echart_ktCO2e_sample(
                comp=self.comp, sector=sector, ghg=ghg)
        return sectoral_emissions_by_region_echart_from_sample(
                ktCO2e_sample=ktCO2e_sample,
                year_0=1990,
                n_years=34,
                n_samples=self.comp.n_mcmc,
                sector=sector,
                ghg=ghg,
                div_id=(f'{self.__class__.__name__}'
                        f'_regional_sparkline_echart'
                        f'_{ghg.value if ghg else "all"}'),
                v_unit=v_unit)

    # called by e.g. models/prob/foo/sectors/bar
    # to determine which GHGs to show
    def GHGs_for_sector(self, sector):
        near_zeros = near_zero_sector_ghgs()
        rval = [ghg for ghg in GHG if (sector, ghg) not in near_zeros]
        return rval
