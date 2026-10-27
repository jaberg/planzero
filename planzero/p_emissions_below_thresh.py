import jax.numpy as jnp
import numpyro.distributions as dists

from .enums import GHG, IPCC_Sector
from .scenario_model import (
    AnnualKey,
    GeneralKey,
    ModelElement,
    Phase,
    Workspace,
    define,
    years_key,
)


class PEmissionsBelowThresh(ModelElement):
    """
    Calculate a Pre-NIR-{year}-m{months_ahead} model metric. 
    """

    prediction_year:int # e.g. 2025
    thresh_ktCO2e:float
    ktCO2e_keys_by_sector_ghg: dict[
            IPCC_Sector, dict[GHG, GeneralKey|AnnualKey]]

    def __init__(
            self,
            prediction_year:int,
            thresh_ktCO2e:float,
            ktCO2e_keys_by_sector_ghg: dict[
                    IPCC_Sector, dict[GHG, GeneralKey|AnnualKey]],
            ):
        super().__init__()
        self.prediction_year = prediction_year
        self.thresh_ktCO2e = thresh_ktCO2e
        self.ktCO2e_keys_by_sector_ghg = ktCO2e_keys_by_sector_ghg

    @staticmethod
    def element_identifier(pred_year, thresh_ktCO2e) -> str:
        return f'PEmissionsBelowThresh-{pred_year}-{thresh_ktCO2e}-ktCO2e'

    @property
    def identifier(self) -> str:
        return self.element_identifier(self.prediction_year,
                                       self.thresh_ktCO2e)

    @property
    def prob_key(self) -> GeneralKey:
        return self.general_key('prob_key')

    @define(prob_key, prior_shape=[], posterior_only=True)
    def model_element_postprocess(self, ws:Workspace):
        if ws.phase == Phase.Prior:
            # don't access the values to avoid sampling and slowing down MCMC
            pass
        elif ws.phase == Phase.Posterior:
            ws.val.general[self.prob_key] = jnp.zeros((1,))
        else:
            raise NotImplementedError(ws.phase)
