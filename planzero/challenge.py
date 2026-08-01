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


class PreNIR(Challenge):

    @computed_field
    def pretty_name(self) -> str:
        return f'Pre-NIR-{self.nir_year}-{self.months_ahead:02d}'


class PreNIR_2025_06(PreNIR):
    """
    Pre-NIR-2025-06 is about predicting the last year of NIR-2025 data
    (calendar year 2023).
    """
    include_in_registry: ClassVar[bool] = True

    nir_year:int = 2025
    months_ahead:int = 6
    results_available:bool = True


class PreNIR_2026_06(PreNIR):
    """
    Pre-NIR-2026-06 is about predicting the last year of NIR-2026 data
    (calendar year 2024).
    """

    include_in_registry: ClassVar[bool] = True

    nir_year:int = 2026
    months_ahead:int = 6
    results_available:bool = True


class PreNIR_2027_06(PreNIR):
    """
    Pre-NIR-2027-06 is about predicting the last year of NIR-2027 data
    (calendar year 2025).
    """
    include_in_registry: ClassVar[bool] = True

    nir_year:int = 2027
    months_ahead:int = 6
    results_available:bool = False
