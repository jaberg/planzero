import datetime

from .enums import GHG, IPCC_Sector
from .nir_ar2_model import (
    AR2_ModelElement,
)
from .nir_model import NIR_ScenarioModel
from .scenario_model import compute_model


def test_smoke_sector_ghg(
    sector=IPCC_Sector.SCS__Public_Electricity_and_Heat,
    ghg=GHG.CO2,
    num_warmup=500,
    num_samples=250,
    ):

    model = NIR_ScenarioModel(
            data_cutoff=datetime.date(2024, 12, 31),
            last_forecast_year=2030,
            num_warmup=num_warmup,
            num_samples=num_samples,
            expect_all_sectors=False)

    snorm_elem = AR2_ModelElement(
                sector=sector,
                ghg=ghg)
    model.add_element(snorm_elem)

    model.add_emissions_sums()
    model.add_element_prenir_2025_m04()

    # this test used to compute the model, but it has been cut
    # to attempt to save memory
    compute_model(model, seed_or_key=1)


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
