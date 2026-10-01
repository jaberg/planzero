import datetime

import jax.numpy as jnp
import numpyro.distributions as dist

from .enums import IPCC_Sector
from .scenario_model.base import (
    ndarray_dim,
    observation,
    observation_valid,
    years_dim,
)
from .scenario_model.computation import (
    ModelElement,
    WorkSpace_Prep,
    WorkSpace_Proc,
    WorkSpace_Step,
    define,
    define_annual,
    define_carry,
    years_key,
)
from .scenario_model.computation import Computation, Model, run_mcmc


class NIR_Sector_Static_Normal_BarrierElement(ModelElement):
    sector: IPCC_Sector
    data_cutoff: datetime.date


def static_normals_model(
        last_observed_year:int,
        last_forecast_year:int,
        ) -> Model:
    data_cutoff = datetime.date(year=last_observed_year, month=12, day=31)
    model = Model(
            first_year=1990,
            n_prior_years=last_observed_year + 1 - 1990,
            n_posterior_years=last_forecast_year + 1 - 1990,
            )
    for sector in IPCC_Sector:
        model.add_element(
                NIR_Sector_Static_Normal_BarrierElement(
                    sector=sector,
                    data_cutoff=data_cutoff,
                    ))
    return model


def entrypoint_static_normals_inference(payload, model_db=model_db):
    raise NotImplementedError()

def inference_work_loop(data_cutoff):
    for payload in model_db.task_completion_iter():
        assert payload['entrypoint'] == 'static_normals_inference'
        entrypoint_static_normals_inference(payload)

def loglik_NIR_Normal(
    component_id,
    NIR_year,
    emission_year,
    ):
    raise NotImplementedError()

def loglik_NIR_BayesianNormal(
    component_id,
    NIR_year,
    emission_year,
    ):
    raise NotImplementedError()

def weighted_KL_score(year, model_id):
    raise NotImplementedError()

def p_emissions_below_thresh_ex_LULUCF(
        model_id:str,
        thresh_ktCO2e:float,
        ) -> float:
    raise NotImplementedError()
