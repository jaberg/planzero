import datetime

from .enums import GHG, IPCC_Sector
from .nir_model import NIR_ScenarioModel
from .nir_static_normals_model import (
    StaticNormals_ModelElement,
)

variety_of_sectors = [
                IPCC_Sector.Forest_Land, # large, pos and neg
                IPCC_Sector.Municipal_Solid_Waste_Landfills, # large, high var, just CH4
                IPCC_Sector.Transport__Road__Light_Duty_Gasoline_Vehicles, # large low var
                IPCC_Sector.SCS__Construction, # small low var
                ]

def test_smoke_sector_ghg(
    sector=IPCC_Sector.SCS__Public_Electricity_and_Heat,
    ghg=GHG.CO2,
    num_warmup=1500,
    num_samples=100,
    ):

    model = NIR_ScenarioModel(
            data_cutoff=datetime.date(2024, 12, 31),
            last_forecast_year=2030,
            num_warmup=num_warmup,
            num_samples=num_samples,
            expect_all_sectors=False)

    snorm_elem = StaticNormals_ModelElement(
                sector=sector,
                ghg=ghg)
    model.add_element(snorm_elem)
            #mcmc_group=f'{sector.value}_{ghg.value}')

    model.add_emissions_sums()
    model.add_element_prenir_2025_m04()

    # this test used to compute the model, but it has been cut
    # to attempt to save memory


def test_smoke_mining_n2o():
    test_smoke_sector_ghg(
            sector=IPCC_Sector.SCS__Mining,
            ghg=GHG.N2O)


def test_smoke_pulp_paper_ch4():
    test_smoke_sector_ghg(
            sector=IPCC_Sector.SCS__Manufacturing__Pulp_and_Paper,
            ghg=GHG.CH4)


def test_smoke_fishing_CO2():
    test_smoke_sector_ghg(
            sector=IPCC_Sector.Transport__Marine__Fishing,
            ghg=GHG.CO2)
