"""
BaseModels for Probabilistic Models

Analog to sim.py for Simulation-based models.
"""

from typing import ClassVar
from pydantic import BaseModel, computed_field

from .singleton_registry import SingletonRegistry

registry = SingletonRegistry()


# TODO Is this class used?
class InferenceResult(BaseModel):

    inference_name: str

    # these can be large, should be str->memory-mapped file
    post_samples: dict[str, object]


site_inferences = registry # TODO: deprecate alais

class SiteInference(BaseModel):
    """A specific simulation (no caller configuration, all pre-loaded)
    to be listed on the site's simulation page.
    """

    include_in_registry: ClassVar[bool] = False

    def main_model_id(self) -> int:
        # if model corresponds to a model in model_db, print model_id to
        # stdout and return 0
        print()
        return 1

    def main_inference_cache_key(self):
        # called via e.g. __main__.py
        # e.g. GH Actions will be used to cache a directory based on this key
        pass

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
        return self.__class__.__doc__

    @computed_field
    def predicted_emissions_2050_MtCO2e_bounds_ul(self) -> tuple[float, float]:
        return (float('nan'), float('nan'))

    def challenge_scores(self, challenge_name):
        return dict(
            total=float('nan'),
            )


from . import nir2025_site
from . import nir_static_normals_site
from . import nir_ar2_site
