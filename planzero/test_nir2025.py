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


def test_idealized_NIR2025_smoke():
    ideal_pt = idealized_NIR2025(length_scale=1.0)
    ideal_ca = ideal_pt.sum(axis=2)
    actual_pt, actual_ca = ktCO2e_dense_w_nan()
    diff = ideal_ca - actual_ca
    idx_of_sector = {sector: ii for ii, sector in enumerate(IPCC_Sector)}
    idx_of_ghg = {ghg: ii for ii, ghg in enumerate(GHG)}
    idx_of_pt = {pt: ii for ii, pt in enumerate(PT)}
    idx_of_yr = {yr: ii for ii, yr in enumerate(nir2025_year_ints)}
    for sector, ii in idx_of_sector.items():
        for ghg, jj in idx_of_ghg.items():
            if sector == IPCC_Sector.Harvested_Wood_Products:
                tol = 200
            else:
                tol = 100
            assert abs(diff[ii, jj]).max() < tol, (sector, ghg)
