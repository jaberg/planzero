"""
Prediction Challenges
"""

from typing import ClassVar
from pydantic import BaseModel, computed_field

from .singleton_registry import SingletonRegistry

registry = SingletonRegistry()


class Challenge(BaseModel):
    """A prediction challenge, typically featured on the Predictions page.
    """

    include_in_registry: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if getattr(cls, 'include_in_registry', False):
            registry.add_class(cls)

    @computed_field
    def show_on_predictions_page(self) -> bool:
        return True

    @computed_field
    def pretty_name(self) -> str:
        return self.__class__.__name__

    @computed_field
    def short_description(self) -> str:
        return self.__class__.__doc__

    @computed_field
    def overall_loss_column_title(self) -> str:
        return 'overall loss'

    def overall_loss(self, model) -> float:
        raise NotImplementedError()


class PreNIR(Challenge):

    @computed_field
    def pretty_name(self) -> str:
        return f'Pre-NIR-{self.nir_year}-m{self.months_ahead:02d}'

    @computed_field
    def overall_loss_column_title(self) -> str:
        return 'weighted KL-divergence (lower is better)'

    @computed_field
    def short_description(self) -> str:
        return f"""
        {self.pretty_name} is about predicting the last year of NIR-{self.nir_year} data
        (calendar year {self.nir_year - 2}), {self.months_ahead}
        months ahead of publication."""


class PreNIR_2025_m06(PreNIR):
    include_in_registry: ClassVar[bool] = True

    nir_year:int = 2025
    months_ahead:int = 6
    results_available:bool = True

    def overall_loss(self, model) -> float:
        return model.prediction_scores_prenir_2025_m06()['weighted_divergence']


class PreNIR_2026_m06(PreNIR):

    include_in_registry: ClassVar[bool] = True

    nir_year:int = 2026
    months_ahead:int = 6
    results_available:bool = True

    @computed_field
    def show_on_predictions_page(self) -> bool:
        return False


class PreNIR_2027_m06(PreNIR):
    include_in_registry: ClassVar[bool] = True

    nir_year:int = 2027
    months_ahead:int = 6
    results_available:bool = False

    @computed_field
    def show_on_predictions_page(self) -> bool:
        return False
