from .cattle_model import Cattle_StaticNormals_ModelElement
from .scenario_model import (
    ScenarioModel,
    compute_model,
)

def test_smoke_emissions():

    sm = ScenarioModel(
            first_year=1990,
            n_prior_years=10,
            n_posterior_years=11,
            num_warmup=500,
            num_samples=50,
            )
    sm.add_element(Cattle_StaticNormals_ModelElement())
    comp = compute_model(sm, seed_or_key=1)
    del comp
