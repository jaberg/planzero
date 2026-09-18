"""
BaseModels for Probabilistic Models
"""

from typing import ClassVar

from pydantic import BaseModel, computed_field

from . import ablation
from .annual_emission_results import AnnualEmissionResults
from .singleton_registry import SingletonRegistry

registry = SingletonRegistry()


class SiteInference(BaseModel):
    """A specific simulation (no caller configuration, all pre-loaded)
    to be listed on the site's simulation page.
    """

    include_in_registry: ClassVar[bool] = False
    strategy_id: str|None = None

    def main_model_id(self) -> int:
        # if model corresponds to a model in model_db, print model_id to
        # stdout and return 0
        print()
        return 1

    def main_inference_prep(self):
        # called via e.g. __main__.py
        # The required samples files may be cached, but the model_db is
        # under construction. It should be built as if from scratch.
        pass

    def main_inference_work(self):
        # called via e.g. __main__.py
        # post-condition: all inferences required for the model have been
        # computed and cached in local file system
        #
        # The required files may be cached, workers should check for completed
        # files and assume they are correct if they are present.
        pass

    @classmethod
    def __init_subclass__(cls):
        super().__init_subclass__()
        if getattr(cls, 'include_in_registry', False):
            registry.add_class(cls)

    @computed_field
    def show_on_models_page(self) -> bool:
        return True

    @computed_field
    def short_description(self) -> str:
        rval = self.__class__.__doc__
        if rval is None:
            return ''
        else:
            return rval

    @computed_field
    def predicted_emissions_2050_MtCO2e_bounds_ul(self) -> tuple[float, float]:
        return (float('nan'), float('nan'))

    def prediction_scores_prenir_2025_06(self) -> dict:
        raise NotImplementedError()

    def challenge_result_url(self, challenge_name) -> str:
        return ''

    @computed_field
    def show_prediction_quality(self) -> bool:
        return False

    @computed_field
    def pretty_name(self) -> str:
        return self.__class__.__name__.replace(' ', '_')

    @property
    def posts_developing_this_page(self) -> list[str]:
        return []

    @computed_field
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
        if self.ablation_study_id is None:
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
        if self.ablation_study_id is None:
            return {}
        else:
            return self.ablation_study.barriers

    @computed_field
    def ablation_study_id(self) -> str|None:
        return None

    @property
    def ablation_study(self) -> ablation.AblationStudy:
        as_id = self.ablation_study_id
        if as_id is None:
            return None
        else:
            return ablation.registry[as_id]

    def emission_results(self) -> AnnualEmissionResults:
        raise NotImplementedError()


# TODO: move to registry-building file
# TODO: rename these site files to make decl/impl pairs of files
from . import nir2025_site
from . import nir_static_normals_site
from . import nir_ar2_site
