import jax.random as jrandom

from .enums import IPCC_Sector, GHG
from .nir2025 import *
from .nir2025_model import NIR2025_Model, NIR2025_ModelElement
from .scenario_model.computation import Computation


def test_co2e_objtensors():
    rval_pt, rval_ca = co2e_objtensors()

    for total in [rval_ca.sum(), rval_pt.sum()]:
        assert total.times[0] == 1990
        # Reported Total + Land Use Total
        assert abs(total.values[1] - (606392 + 49847)) < 10

        assert total.times[-1] == 2023
        # Reported Total + Land Use Total
        assert abs(total.values[-1] -
                   (693912 + 4169)) < 10


def test_no_off_by_one():
    er = NIR2025_EmissionsResults()
    total = er.total()

    assert total.times[0] == 1990
    # Reported Total + Land Use Total
    assert abs(total.values[1] - (606392 + 49847)) < 10

    assert total.times[-1] == 2023
    assert abs(total.values[-1] -
               (693912 + 4169)) < 10


def test_model_2023(last_observed_year=2023, n_years=34):
    model = NIR2025_Model(last_observed_year, 2)
    comp = Computation(
            model=model,
            grouped_samples=None,
            rng_key=jrandom.key(0))
    comp.run()
    for sector in IPCC_Sector:
        for ghg in GHG:
            elem_id = NIR2025_ModelElement.element_identifier(sector, ghg)
            elem = model.model_elements[elem_id]
            assert comp.storage_nd[elem.ca_key].shape == (2, n_years)
            for key in elem.pt_keys.values():
                assert comp.storage_nd[key].shape == (2, n_years)


def test_model_2022():
    test_model_2023(last_observed_year=2022, n_years=33)
