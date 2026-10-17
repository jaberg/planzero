from .cattle_model import Cattle_StaticNormals_ModelElement
from .enums import IPCC_Sector, GHG
from .nir2025_model import NIR2025_ModelElement
from .scenario_model import (
    AnnualKey,
    GeneralKey,
    ModelElement,
    Phase,
    ScenarioModel,
    Workspace,
    compute_model,
    define,
    define_annual,
)

def test_smoke_headcounts():
    sm = ScenarioModel(
            first_year=1990,
            n_prior_years=10,
            n_posterior_years=11,
            num_warmup=7,
            num_samples=9,
            )
    sm.add_element(Cattle_StaticNormals_ModelElement(
        nir_ca_key=GeneralKey(elem_id='foo', name='bar'),
        nir_pt_keys={},
        model_headcounts=True,
        model_emission_factors=False,
        model_emissions=False,
        ))
    comp = compute_model(sm, seed_or_key=1)


def test_smoke_emission_factors():
    sm = ScenarioModel(
            first_year=1990,
            n_prior_years=10,
            n_posterior_years=11,
            num_warmup=7,
            num_samples=9,
            )
    sm.add_element(Cattle_StaticNormals_ModelElement(
        nir_ca_key=GeneralKey(elem_id='foo', name='bar'),
        nir_pt_keys={},
        model_headcounts=False,
        model_emission_factors=True,
        model_emissions=False,
        ))
    comp = compute_model(sm, seed_or_key=1)


def test_smoke_emissions():

    sm = ScenarioModel(
            first_year=1990,
            n_prior_years=10,
            n_posterior_years=11,
            num_warmup=500,
            num_samples=500,
            )
    nir_elem = NIR2025_ModelElement(
            sector=IPCC_Sector.Enteric_Fermentation,
            ghg=GHG.CH4,
            n_years=sm.n_prior_years)
    sm.add_element(nir_elem)
    sm.add_element(Cattle_StaticNormals_ModelElement(
        nir_ca_key=nir_elem.ca_key,
        nir_pt_keys=nir_elem.pt_keys,
        model_headcounts=True,
        model_emission_factors=True,
        model_emissions=True,
        ))
    comp = compute_model(sm, seed_or_key=1)
