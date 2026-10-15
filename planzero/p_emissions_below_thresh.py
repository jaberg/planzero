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
    ktCO2e_normal_keys_by_sector_ghg: dict[
            tuple[IPCC_Sector, GHG],
            GeneralKey|AnnualKey]

    def __init__(
            self,
            prediction_year:int,
            thresh_ktCO2e:float,
            ktCO2e_normal_keys_by_sector_ghg: dict[
                    tuple[IPCC_Sector, GHG],
                    GeneralKey|AnnualKey],
            ):
        super().__init__()
        self.prediction_year = prediction_year
        self.thresh_ktCO2e = thresh_ktCO2e
        self.ktCO2e_normal_keys_by_sector_ghg = ktCO2e_normal_keys_by_sector_ghg

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

    @define(prob_key, prior_shape=[])
    def model_element_postprocess(self, ws:Workspace):
        if ws.phase == Phase.Prior:
            # don't access the values to avoid sampling and slowing down MCMC
            pass
        elif ws.phase == Phase.Posterior:

            model_years = ws.val.general[years_key]
            for ii, model_year in enumerate(list(model_years)):
                if model_year == self.prediction_year:
                    model_idx_of_prediction_year = ii
                    break
            else:
                raise NotImplementedError(self.prediction_year, model_years)

            total_mu = jnp.zeros(())
            total_sigma_sq = jnp.zeros(())

            for (sector, ghg), ws_key in self.ktCO2e_normal_keys_by_sector_ghg.items():
                if isinstance(ws_key, GeneralKey):
                    raise NotImplementedError(sector, ghg)
                elif isinstance(ws_key, AnnualKey):
                    pred_dist = ws.dist.annual[ws_key]
                    if isinstance(pred_dist, dists.Normal):
                        pred_mu = pred_dist.loc
                        pred_sigma_pt = pred_dist.scale
                        mu_pred_year = pred_mu[model_idx_of_prediction_year]
                        sigma_pred_year = pred_sigma_pt[model_idx_of_prediction_year]
                        assert mu_pred_year.shape == (ws.n_mcmc,)
                        assert sigma_pred_year.shape == (ws.n_mcmc,)
                        total_mu += mu_pred_year
                        total_sigma_sq += sigma_pred_year ** 2
                    else:
                        raise NotImplementedError(pred_dist)
                else:
                    raise NotImplementedError(ws_key)

            total_normal = dists.Normal(total_mu, jnp.sqrt(total_sigma_sq))
            p_below_thresh = total_normal.cdf(self.thresh_ktCO2e)
            assert p_below_thresh.shape == (ws.n_mcmc,)
            rval = p_below_thresh.mean()
            return rval

            ws.val.general[self.prob_key] = rval
        else:
            raise NotImplementedError(ws.phase)
