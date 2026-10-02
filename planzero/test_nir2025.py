import jax.random as jrandom
import numpy as np
from numpyro.distributions import Normal

from .enums import PT, IPCC_Sector, GHG
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
            if (sector, ghg) in near_zero_sector_ghgs():
                continue
            elem = model.model_elements[elem_id]
            assert comp.storage_nd[elem.ca_key].shape == (2, n_years)
            for key in elem.pt_keys.values():
                assert comp.storage_nd[key].shape == (2, n_years)


def test_model_2022():
    test_model_2023(last_observed_year=2022, n_years=33)


def _pparams(dist):
    return (dist.mu, dist.rolloff, dist.relerr)


def test_ktCO2e_numpyro_dist_pt_ca_years():
    real_pts = [pt for pt in PT if pt != PT.XX]

    # shapes: one batched SBLN per key, batch dim == years
    ca_dist, pt_dists = ktCO2e_numpyro_dist_pt_ca_years(
        IPCC_Sector.SCS__Public_Electricity_and_Heat,
        GHG.CO2,
        1990,
        34)
    assert ca_dist.batch_shape == (34,)
    assert set(pt_dists) == set(real_pts)
    for pt in real_pts:
        assert pt_dists[pt].batch_shape == (34,)

    # case 1: near-zero sectors use the positive near-zero SBLN,
    # where the single-year function returned a unit Normal.
    (sector1, ghg1), _ = next(iter(near_zero_sector_ghgs().items()))
    c1_ca, c1_pts = ktCO2e_numpyro_dist_pt_ca_years(
            sector1, ghg1, 1990, 34)
    for vp, exp in zip(_pparams(c1_ca), (.5, .01, .5)):
        assert np.all(vp == exp)
    for pt in real_pts:
        for vp, exp in zip(_pparams(c1_pts[pt]), (.5, .01, .5)):
            assert np.all(vp == exp), pt
    orig_ca, _ = ktCO2e_numpyro_dist_pt_ca(sector1, ghg1, 2000)
    assert isinstance(orig_ca, Normal) # the differing logic, pinned

    # Nunavut before 1999: positive near-zero.
    nu = pt_dists[PT.NU]
    assert np.all(nu.mu[:9] == .5)
    assert np.all(nu.rolloff[:9] == .01)
    assert np.all(nu.relerr[:9] == .5)

    # Oil & Gas, NT before 1999 (never reported): positive near-zero.
    _, og_pts = ktCO2e_numpyro_dist_pt_ca_years(
        IPCC_Sector.SCS__Oil_and_Gas_Extraction, GHG.CO2, 1990, 34)
    nt = og_pts[PT.NT]
    assert np.all(nt.mu[:9] == .5)
    assert np.all(nt.rolloff[:9] == .01)
    assert np.all(nt.relerr[:9] == .5)

    # parity with the single-year function over a variety of cases
    cases = [
        (IPCC_Sector.Harvested_Wood_Products, GHG.CO2),     # can be negative
        (IPCC_Sector.Settlements, GHG.CO2),                 # can be negative
        (IPCC_Sector.Forest_Land, GHG.CO2),                 # can be negative; NT has NaNs
        (IPCC_Sector.SCS__Mining, GHG.CO2),                 # sometimes-reported NaNs
        (IPCC_Sector.SCS__Public_Electricity_and_Heat, GHG.CO2),  # non-negative
        (IPCC_Sector.SCS__Oil_and_Gas_Extraction, GHG.CO2), # NT pre-1999 special case
    ]
    for sector, ghg in cases:
        vca, vpis = ktCO2e_numpyro_dist_pt_ca_years(sector, ghg, 1990, 34)
        for jj in range(34):
            year = 1990 + jj
            oca, opsi = ktCO2e_numpyro_dist_pt_ca(sector, ghg, year)
            if isinstance(oca, Normal):
                assert _pparams(vca)[0][jj] == .5
                assert _pparams(vca)[1][jj] == .01
                assert _pparams(vca)[2][jj] == .5
            else:
                for vp, op in zip(_pparams(vca), _pparams(oca)):
                    assert vp[jj] == op, (sector, ghg, year)
            for kk, pt in enumerate(real_pts):
                od = opsi[kk]
                vd = vpis[pt]
                if isinstance(od, Normal):
                    assert vd.mu[jj] == .5 and vd.rolloff[jj] == .01 and vd.relerr[jj] == .5
                else:
                    for vp, op in zip(_pparams(vd), _pparams(od)):
                        assert vp[jj] == op, (sector, ghg, pt, year)

    # present-but-small (< 1 kt): positive near-zero (Public Electricity and Heat)
    arr_pt, _ = ktCO2e_dense_w_nan()
    small_sector = IPCC_Sector.SCS__Public_Electricity_and_Heat
    vca, vps = ktCO2e_numpyro_dist_pt_ca_years(small_sector, GHG.CH4, 1990, 34)
    found_small = None
    for kk, pt in enumerate(real_pts):
        col = arr_pt[idx_of_sector[small_sector],
                     idx_of_ghg[GHG.CH4],
                     idx_of_pt[pt], :]
        for jj in range(34):
            if not np.isnan(col[jj]) and 0 < col[jj] < 1:
                d = vps[pt]
                assert d.mu[jj] == .5 and d.rolloff[jj] == .01 and d.relerr[jj] == .5
                if found_small is None:
                    found_small = (pt, 1990 + jj)
    assert found_small is not None, 'expected a small-but-present region value'

    # sometimes-reported NaN: SBLN(mu=max(time-mean,1), rolloff=.1 mu, relerr=.7)
    nan_sector = IPCC_Sector.SCS__Mining
    vca, vps = ktCO2e_numpyro_dist_pt_ca_years(nan_sector, GHG.CO2, 1990, 34)
    found_d = None
    for kk, pt in enumerate(real_pts):
        col = arr_pt[idx_of_sector[nan_sector],
                     idx_of_ghg[GHG.CO2],
                     idx_of_pt[pt], :]
        if not np.isfinite(np.nanmean(col)):
            continue
        for jj in range(34):
            if not np.isnan(col[jj]):
                continue
            oca, opsi = ktCO2e_numpyro_dist_pt_ca(nan_sector, GHG.CO2, 1990 + jj)
            od = opsi[kk]
            if od.relerr != np.float32(.7):
                continue # not the D branch, e.g. Nunavut before 1999
            d = vps[pt]
            for vp, op in zip(_pparams(d), _pparams(od)):
                assert vp[jj] == op, (pt, 1990 + jj)
            if found_d is None:
                found_d = (pt, 1990 + jj)
    assert found_d is not None, 'expected a sometimes-reported region value'
