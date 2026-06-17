"""
BaseModels for Probabilistic Models

Analog to sim.py for Simulation-based models.
"""

from pydantic import BaseModel, computed_field

class InferenceResult(BaseModel):

    inference_name: str

    # these can be large, should be str->memory-mapped file
    post_samples: dict[str, object]


site_inferences = {}

class SiteInference(BaseModel):
    """A specific simulation (no caller configuration, all pre-loaded)
    to be listed on the site's simulation page.
    """

    @classmethod
    def __init_subclass__(cls):
        super().__init_subclass__()
        site_inferences[cls.__name__] = cls()

    @computed_field
    def show_on_models_page(self) -> bool:
        return True

    @computed_field
    def short_description(self) -> str:
        return self.__class__.__doc__

    @computed_field
    def predicted_emissions_2050_MtCO2e_bounds_ul(self) -> tuple[float, float]:
        return (float('nan'), float('nan'))


if 0:
    class AR2(SiteInference):
        """Emissions per province and territory,
        and per greenhouse gas, evolve according
        to a 2-step autoregressive process.
        """

from . import nir_constant_predictor
