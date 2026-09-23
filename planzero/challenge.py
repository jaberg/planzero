"""
Prediction Challenges
"""

from typing import ClassVar

import numpy as np
from pydantic import BaseModel, computed_field

from . import nir2025
from .enums import GHG, PT, IPCC_Sector
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
        return 'weighted KL-divergence (lower is better, zero is perfect)'

    @computed_field
    def short_description(self) -> str:
        return f"""
        {self.pretty_name} is about predicting the last year of NIR-{self.nir_year} data
        (calendar year {self.nir_year - 2}), {self.months_ahead}
        months ahead of publication."""


class PreNIR_2025_m04(PreNIR):
    include_in_registry: ClassVar[bool] = True

    nir_year:int = 2025
    months_ahead:int = 4
    results_available:bool = True

    def overall_loss(self, model) -> float:
        return model.prediction_scores_prenir_2025_m04()['weighted_divergence']

    def key_sector_ghg_ca(self, sector, ghg) -> tuple[str, IPCC_Sector, GHG]:
        return ('PreNIR_2025_m04_ca', sector, ghg)

    def key_sector_ghg_pt(self, sector, ghg) -> tuple[str, IPCC_Sector, GHG]:
        return ('PreNIR_2025_m04_pt', sector, ghg)

    def prediction_scores_from_batch_rollout(self, results):
        constants = results['constants']
        post_vals = results['post_vals']

        abs_ktCO2e = np.zeros((len(IPCC_Sector), len(GHG), len(PT)))
        # weights are abs( expected true value)
        # aka abs NIR2025 estimated national means

        _, arr_ca = nir2025.ktCO2e_dense_w_nan()
        arr_idx_of_year = nir2025.idx_of_year[2023]
        abs_m_sgt = abs(arr_ca[:, :, arr_idx_of_year])
        abs_ktCO2e[:] = abs_m_sgt[:, :, None]

        # zero-out the tiny sector-gas combinations
        abs_ktCO2e[ abs_ktCO2e < 1 ] = 0

        scores_pt = {}
        scores_ca = {}
        KL_values = np.zeros_like(abs_ktCO2e)
        for ii, sector in enumerate(IPCC_Sector):
            scores_pt[sector] = {}
            scores_ca[sector] = {}
            for jj, ghg in enumerate(GHG):
                key_pt = self.key_sector_ghg_pt(sector, ghg)
                key_ca = self.key_sector_ghg_ca(sector, ghg)
                try:
                    scores_pt[sector][ghg] = constants._dict[key_pt]
                    scores_ca[sector][ghg] = constants._dict[key_ca]
                except KeyError:
                    scores_pt[sector][ghg] = post_vals[key_pt]
                    scores_ca[sector][ghg] = post_vals[key_ca]

                KL_values[ii, jj][:13] = scores_pt[sector][ghg]
                KL_values[ii, jj][13] = scores_ca[sector][ghg]


        weighted_divergence = (abs_ktCO2e * KL_values).sum() / abs_ktCO2e.sum()

        return {
                'weighted_divergence': weighted_divergence,
                'scores_ca': scores_ca,
                'scores_pt': scores_pt,
                }


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
