
"""
Blended prediction error is

1/2 1-year prediction
1/4 2-year prediction (if applicable)
1/4 3-year prediction (if applicable)

1/2 all-of-Canada
1/2 * PER(t) for the actual fraction of provincial emissions from each year


Test 2021-2023 inclusive: trailing 3 years, rolling (3 data points)
These years aren't used at all for training and validation.

Training and validation: 1990-2020 inclusive
validation: 2018-2020 inclusive, trailing 3 years rolling (3 data points)
Training: 1990-2017 inclusive.
"""

import numpy as np
from .enums import IPCC_Sector, PT, GHG
from .ureg import u

def NIR_ref_np(ref_er, int_year_range):
    """Return a 4D numpy array of co2e amounts for the requested years
    """
    start, stop = int_year_range
    rval = np.zeros(
        (len(IPCC_Sector), len(PT), len(GHG), stop - start),
        dtype=np.float32)
    ref_co2e = ref_er.co2e_sum_over_drivers()
    year_ints = np.arange(start, stop)
    for ii, sector in enumerate(IPCC_Sector):
        for jj, pt in enumerate(PT):
            for kk, ghg in enumerate(GHG):
                key = (sector, ghg, pt)
                if key in ref_co2e:
                    ts = ref_co2e[key].to(u.kt_CO2e)
                    dct = {tt:vv for tt, vv in zip(ts.times, ts.values[1:])}
                    assert ts.t_unit == u.years
                    for ll, yr in enumerate(range(start, stop)):
                        assert np.isfinite(dct[yr])
                        rval[ii, jj, kk, ll] = dct[yr]
    return rval


def NIR_other_np(other_er, int_year_range):
    """Return a 4D numpy array of co2e amounts for the requested years
    """
    start, stop = int_year_range
    rval = np.zeros(
        (len(IPCC_Sector), len(PT), len(GHG), stop - start),
        dtype=np.float32)
    other_co2e = other_er.co2e_sum_over_drivers()
    year_ints = np.arange(start, stop)
    for ii, sector in enumerate(IPCC_Sector):
        for jj, pt in enumerate(PT):
            for kk, ghg in enumerate(GHG):
                key = (sector, ghg, pt)
                if key in other_co2e:
                    vals = other_co2e[key].query(year_ints * u.years).to(u.kt_CO2e).magnitude
                    assert np.isfinite(vals).all(), (vals, year_ints, sector, pt, ghg, other_co2e[key])
                    rval[ii, jj, kk, :] = vals
    return rval


def blended_RMSE_year_weights(n_years):
    year_weights = [1 / (1 << ii) for ii in range(n_years)]
    year_weights[-1] += (1.0 - sum(year_weights))
    return np.asarray(year_weights)


def blended_RMSE(ref_er, other_er, int_year_range):
    ref_np = NIR_ref_np(ref_er, int_year_range)
    other_np = NIR_other_np(other_er, int_year_range)

    diff = ref_np - other_np

    n_sectors, n_pt, n_ghg, n_yrs = diff.shape
    year_weights = blended_RMSE_year_weights(n_yrs)

    diff_sq = diff * diff
    diff_mse_by_yr = np.sqrt(diff_sq.mean(axis=(0, 1, 2)))
    rval = np.dot(diff_mse_by_yr, year_weights)

    return rval


def blended_RMSE_algo(NIR_prediction_algo, ref_er, last_training_year, horizon):
    state = NIR_prediction_algo(last_training_year, horizon)
    int_year_range = (last_training_year + 1, horizon)
    er = state.compute_annual_emissions(
        int_year_range=int_year_range)
    rval = blended_RMSE(ref_er, er, int_year_range)
    return float(rval)
