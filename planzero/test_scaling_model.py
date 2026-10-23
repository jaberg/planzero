import datetime

from .enums import IPCC_Sector
from .scaling_model import Scaling_ScenarioModel
from .scenario_model import Computation, compute_model


def test_constructor():
    Scaling_ScenarioModel(
            data_cutoff=datetime.date(year=2024, month=12, day=31)
            )

def test_compute_prior_ef_man_other():
    sm = Scaling_ScenarioModel(
            data_cutoff=datetime.date(year=2024, month=12, day=31),
            sectors=[IPCC_Sector.Enteric_Fermentation,
                     IPCC_Sector.SCS__Manufacturing__Other]
            )

    comp = Computation(sm, rng_key=1)
    comp.run()


def test_compute_model_ef_man_other():
    sm = Scaling_ScenarioModel(
            data_cutoff=datetime.date(year=2024, month=12, day=31),
            sectors=[IPCC_Sector.Enteric_Fermentation,
                     IPCC_Sector.SCS__Manufacturing__Other],
            num_warmup=90,
            num_samples=11,
            )

    comp = compute_model(sm, seed_or_key=1)
