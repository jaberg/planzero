from .nir2025 import *
from . import ipcc_canada

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
