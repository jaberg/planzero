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


def test_averaging_window_1():
    n_year_filter = 1
    ts = STS(
        times=array.array('d', [1, 2, 3]),
        t_unit=u.years,
        values=array.array('d', [float('nan'), 3, 4, 5]),
        v_unit=kt_by_ghg[GHG.CO2] / u.year,
        interpolation='current')
    obj = NIR2025(
        n_year_filter=n_year_filter,
        filter_algo='average')
    ef = obj.filter_ts_average(ts, GHG.CO2)
    assert ef.times == array.array('d', [1, 2, 3])
    assert ef.values[0] == 0
    assert ef.values[1] == 3
    assert ef.values[2] == 4
    assert ef.values[3] == 5


def test_averaging_window_2():
    n_year_filter = 2
    ts = STS(
        times=array.array('d', [1, 2, 3]),
        t_unit=u.years,
        values=array.array('d', [float('nan'), 3, 4, 5]),
        v_unit=kt_by_ghg[GHG.CO2] / u.year,
        interpolation='current')
    obj = NIR2025(
        n_year_filter=n_year_filter,
        filter_algo='average')
    ef = obj.filter_ts_average(ts, GHG.CO2)
    assert ef.times == array.array('d', [1, 2, 3])
    assert ef.values[0] == 0
    assert ef.values[1] == 3
    assert ef.values[2] == 3.5
    assert ef.values[3] == 4.5


def test_SoftmaxGP_MAP_smoke():
    ideal_pt = SoftmaxGP_MAP_NIR2025()
    ideal_ca = ideal_pt.sum(axis=2)
    actual_pt, actual_ca = ktCO2e_dense_w_nan()
    assert np.allclose(ideal_ca, actual_ca, atol=1, rtol=1e-3)
