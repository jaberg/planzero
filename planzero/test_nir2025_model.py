from .nir2025_model import NIR2025_ScenarioModel
from .scenario_model import registry_compute_model


def test_registry_computation():
    model = NIR2025_ScenarioModel(last_observed_year=2023)
    comp = registry_compute_model(
            model,
            model_name='NIR2025',
            cache_posterior=False,
            seed_or_key=1)
