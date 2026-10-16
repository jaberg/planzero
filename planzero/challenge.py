"""
Prediction Challenges
"""

from typing import ClassVar

import jax.numpy as jnp
import numpy as np
from numpyro import distributions as dist
from pydantic import BaseModel, computed_field

from . import nir2025
from .annual_emission_results import aer_key_normal_sigma_ca
from .enums import GHG, PT, IPCC_Sector, LULUCF_Sectors
from .singleton_registry import SingletonRegistry

registry = SingletonRegistry()


class Challenge(BaseModel):
    """A prediction challenge, typically featured on the Predictions page.
    """

    include_in_registry: ClassVar[bool] = False

    results_available:bool  # can models be expected to have scores?

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
    def description(self) -> str:
        return self.__class__.__doc__ or ""

    @computed_field
    def short_description(self) -> str:
        # one-liner
        return self.__class__.__doc__ or ""

    @computed_field
    def overall_loss_column_title(self) -> str:
        return 'overall loss'

    def overall_loss(self, model) -> float:
        raise NotImplementedError()

    def overall_loss_str(self, model) -> str:
        return f'{self.overall_loss(model):.3f}'

    @property
    def overall_loss_initial_sort_class(self) -> str:
        return 'gemini-sort-initial-asc'


class PreNIR(Challenge):

    nir_year:int
    months_ahead:int

    @computed_field
    def pretty_name(self) -> str:
        return f'Pre-NIR-{self.nir_year}-m{self.months_ahead:02d}'

    @computed_field
    def overall_loss_column_title(self) -> str:
        return 'weighted KL-divergence (lower is better, zero is perfect)'

    @computed_field
    def short_description(self) -> str:
        return f"""How good are models at NIR{self.nir_year} prediction?"""


class PreNIR_2025_m04(PreNIR):
    include_in_registry: ClassVar[bool] = True

    nir_year:int = 2025
    months_ahead:int = 4
    results_available:bool = True

    @computed_field
    def description(self) -> str:
        return f"""
        {self.pretty_name} is about predicting the last year of NIR-{self.nir_year} data
        (calendar year {self.nir_year - 2}), {self.months_ahead}
        months ahead of publication, that is, based on information that was available by Dec 31, 2024."""

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


class NetZero_2050(Challenge):

    include_in_registry: ClassVar[bool] = True
    results_available:bool = True

    @computed_field
    def pretty_name(self) -> str:
        return "Net-Zero-2050"

    @computed_field
    def overall_loss_column_title(self) -> str:
        return 'probability of annual net emissions <= 0 (higher is better, 1 is perfect)'

    @computed_field
    def short_description(self) -> str:
        return "How probable is net-zero by 2050?"

    @computed_field
    def description(self) -> str:
        return f"""
        {self.pretty_name} ranks models by how probable it is that net emissions (exclusive of land-use, land-use change and forestry, LULUCF) are less than zero by 2050."""

    # TODO: call this a metric, not a loss [function]
    def overall_loss(self, model) -> float:
        return model.challenge_netzero_2050()

    def overall_loss_str(self, model) -> str:
        return f'{self.overall_loss(model):.2e}'

    @computed_field
    def show_on_predictions_page(self) -> bool:
        return True

    def prediction_scores_from_batch_rollout(self, results):
        constants = results['constants']
        # post_vals = results['post_vals']

        mu_ca_by_sector_ghg = {}
        sigma_ca_by_sector_ghg = {}

        for key, val in constants.items():
            if isinstance(key, tuple) and key[0] == 'AER_Normal_mu_ca':
                _, sector, ghg, activity = key
                mu_ca_by_sector_ghg.setdefault((sector, ghg), {})[activity] = val
                sigma_ca_by_sector_ghg.setdefault((sector, ghg), {})[activity] \
                        = constants[aer_key_normal_sigma_ca(sector, ghg, activity)]

        mu_ca = 0
        sigma_squared_ca = 0
        n_contribs = 0
        for sector in IPCC_Sector:
            if sector in LULUCF_Sectors:
                continue
            for ghg in GHG:
                mu_inc = sum(mu_ca_by_sector_ghg[sector, ghg].values())
                if abs(np.sum(mu_inc)) > .0001:
                    mu_ca += mu_inc
                    sigma_squared_ca += sum(
                            [sigma ** 2
                             for sigma in sigma_ca_by_sector_ghg[sector, ghg].values()])
                    n_contribs += 1

        n_samples, = mu_ca.shape
        n_samples_, = sigma_squared_ca.shape
        assert n_samples == n_samples_

        normal = dist.Normal(loc=mu_ca, scale=jnp.sqrt(sigma_squared_ca))
        p_lt_0 = normal.cdf(0).mean()
        return p_lt_0

    @property
    def overall_loss_initial_sort_class(self) -> str:
        return 'gemini-sort-initial-desc'
