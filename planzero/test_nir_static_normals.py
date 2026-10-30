import datetime

from .enums import GHG, IPCC_Sector
from .nir_model import NIR_ScenarioModel
from .nir_static_normals_model import (
    StaticNormals_ModelElement,
    StaticNormals_ScenarioModel,
)
from .scenario_model import (
    compute_model,
    registry_compute_model,
)
from .scenario_model.base import GroupedPosterior

variety_of_sectors = [
                IPCC_Sector.Forest_Land, # large, pos and neg
                IPCC_Sector.Municipal_Solid_Waste_Landfills, # large, high var, just CH4
                IPCC_Sector.Transport__Road__Light_Duty_Gasoline_Vehicles, # large low var
                IPCC_Sector.SCS__Construction, # small low var
                ]

def test_smoke_registry():
    model = StaticNormals_ScenarioModel(
            data_cutoff=datetime.date(year=2024, month=12, day=31),
            last_forecast_year=2050,
            sectors=variety_of_sectors,
            )

    registry_compute_model(
            model,
            'StaticNormals_loy2022_lfy2050',
            cache_posterior=True,
            seed=42)


def test_smoke_sector_ghg(
    sector=IPCC_Sector.SCS__Public_Electricity_and_Heat,
    ghg=GHG.CO2,
    num_warmup=1500,
    num_samples=1000,
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

    comp = compute_model(model, seed_or_key=1)

    if 0:
        print("sigma pt")
        sample = comp.ndarray_d[
                GroupedPosterior(
                    prior_var_key=snorm_elem.general_key('sigma_pt'))]
        print(sample[0, :, 0])

        print("mu")
        sample = comp.ndarray_d[
                GroupedPosterior(
                    prior_var_key=snorm_elem.general_key('mu'))]
        print(sample[0, :, 0])
    #assert abs(jnp.max(sample[0, :, 0])) < 0.1


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
