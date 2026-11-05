from .nir2025_model import NIR2025_ScenarioModel


def test_constructor_smoke():
    model = NIR2025_ScenarioModel()
    del model
