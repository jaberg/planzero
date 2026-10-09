from .enums import IPCC_Sector
from .nir_static_normals_model import StaticNormals_ScenarioModel
from .scenario_model import registry_compute_model

variety_of_sectors = [
                #IPCC_Sector.Forest_Land, # large, pos and neg
                IPCC_Sector.Municipal_Solid_Waste_Landfills, # large, high var, just CH4
                #IPCC_Sector.Transport__Road__Light_Duty_Gasoline_Vehicles, # large low var
                #IPCC_Sector.SCS__Construction, # small low var
                ]

def test_smoke():
    model = StaticNormals_ScenarioModel(
            last_observed_year=2022,
            last_forecast_year=2050,
            sectors=variety_of_sectors,
            )

    registry_compute_model(
            model,
            'StaticNormals_loy2022_lfy2050',
            cache_posterior=True,
            seed_or_key=1)
