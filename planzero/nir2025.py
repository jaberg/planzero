import array
import os

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from scipy.optimize import minimize

# Enable Float64 for Gaussian Process numerical stability
from jax import config
config.update("jax_enable_x64", True)


from . import ipcc_canada
from . import ghgvalues
from .base import State, DynamicElement
from .enums import (
    IPCC_Sector,
    IPCC_Sector_from_catpath_with_whitespace,
    PT,
    GHG)
from .objtensor import ObjectTensor
from .ureg import u, kt_by_ghg
from .sts import SparseTimeSeries, STS
from .my_functools import cache



nir2025_year_ints = np.arange(1990, 2024)


def load_inventory():
    """Return inventory as DataFrame with standard pre-processing
    """
    inv = pd.read_csv(os.path.join(os.environ['PLANZERO_DATA'], 'EN_GHG_IPCC_Can_Prov_Terr.csv'))
    assert ('kt',) == inv['Unit'].unique()

    inv['CategoryPathWithWhitespace'] = (
        inv['Category'].fillna('').astype(str) + '/' +
        inv['Sub-category'].fillna('').astype(str) + '/' +
        inv['Sub-sub-category'].fillna('').astype(str)
    ).str.rstrip('/')
    assert not inv['CategoryPathWithWhitespace'].isna().any()

    # ensure that 'x' means NaN
    non_strs = set()
    for key in ['CO2', 'CH4', 'CH4 (CO2eq)', 'N2O',
                 'N2O (CO2eq)', 'HFCs', 'PFCs', 'SF6', 'NF3', 'CO2eq']:
        for strval in inv[key].values:
            try:
                assert np.isfinite(float(strval))
            except:
                non_strs.add(strval)
    assert non_strs == {'x'}

    for key in ['CO2', 'CH4', 'CH4 (CO2eq)', 'N2O',
                 'N2O (CO2eq)', 'HFCs', 'PFCs', 'SF6', 'NF3', 'CO2eq']:
        inv[key] = inv[key].replace('x', float('nan')).astype(float)
    del key

    # Year could have been cast to int, they're all whole numbers.
    inv['Year'] = inv['Year'].astype(float)
    return inv


idx_of_sector = {sector: ii for ii, sector in enumerate(IPCC_Sector)}
idx_of_ghg = {ghg: ii for ii, ghg in enumerate(GHG)}
idx_of_pt = {pt: ii for ii, pt in enumerate(PT)}
idx_of_yr = {yr: ii for ii, yr in enumerate(nir2025_year_ints)}


@cache
def ktCO2e_dense_w_nan():
    """Return the NIR-2025 in the form of two numpy arrays:
    * arr_pt (IPCC_Sector idx, GHG idx, PT idx, year since 1990)
    * arr_ca (IPCC_Sector idx, GHG idx, year since 1990)

    In arr_pt, censored data is represented with NaN.
    In arr_ca, there is no censored data.
    """
    inv = load_inventory()

    arr_pt = np.zeros((len(IPCC_Sector),
                       len(GHG),
                       len(PT) - 1,
                       len(nir2025_year_ints)),
                      dtype=np.float32)
    arr_ca = np.zeros((len(IPCC_Sector), len(GHG), len(nir2025_year_ints)), dtype=np.float32)

    non_agg = inv[inv['Total'] != 'y']
    for catpathww, nonagg_catpath in non_agg.groupby('CategoryPathWithWhitespace'):
        ipcc_sector = IPCC_Sector_from_catpath_with_whitespace[catpathww]

        for region, region_df in nonagg_catpath.groupby('Region'):
            if region.lower() == 'canada':
                pt = None
            elif region == 'Northwest Territories and Nunavut':
                pt = PT.XX
            else:
                pt = PT(region)

            for ghg in GHG:
                if ghg in [GHG.CH4, GHG.N2O]:
                    values = region_df[f'{ghg.value} (CO2eq)'].values
                else:
                    values = region_df[ghg.value].values
                years = region_df['Year'].values

                for year, val in zip(years, values):
                    if np.isfinite(val):
                        if pt is None:
                            arr_ca[idx_of_sector[ipcc_sector],
                                   idx_of_ghg[ghg],
                                   idx_of_yr[year]] = val
                        elif pt is PT.XX:
                            arr_pt[idx_of_sector[ipcc_sector],
                                   idx_of_ghg[ghg],
                                   idx_of_pt[PT.NT],
                                   idx_of_yr[year]] = float('nan')
                            arr_pt[idx_of_sector[ipcc_sector],
                                   idx_of_ghg[ghg],
                                   idx_of_pt[PT.NU],
                                   idx_of_yr[year]] = float('nan')
                        else:
                            arr_pt[idx_of_sector[ipcc_sector],
                                   idx_of_ghg[ghg],
                                   idx_of_pt[pt],
                                   idx_of_yr[year]] = val
                    else:
                        assert pt is not None
                        arr_pt[idx_of_sector[ipcc_sector],
                               idx_of_ghg[ghg],
                               idx_of_pt[pt],
                               idx_of_yr[year]] = float('nan')

    return arr_pt, arr_ca


@cache
def co2e_objtensors():
    """Return the NIR-2025 as a pair of ObjectTensors (rval_pt, rval_ca)

    The return values contain the mass of CO2e emitted in each sector, for
    each GHG, and for rval_pt, for each region.

    rval_pt's PT.XX region contains a differencing amount, where necessary so that
    the regional totals match the national total.

    The elements of the returned values are either 0 kt_CO2e constants, or else
    no-interpolation timeseries for year 1990 to 2023 inclusive.
    """

    inv = load_inventory()
    non_agg = inv[inv['Total'] != 'y']

    rval_pt = ObjectTensor.empty(IPCC_Sector, GHG, PT)
    rval_ca = ObjectTensor.empty(IPCC_Sector, GHG)

    rval_pt[:] = 0 * u.kt_CO2e
    rval_ca[:] = 0 * u.kt_CO2e

    for catpathww, nonagg_catpath in non_agg.groupby('CategoryPathWithWhitespace'):
        ipcc_sector = IPCC_Sector_from_catpath_with_whitespace[catpathww]

        for region, region_df in nonagg_catpath.groupby('Region'):
            if region.lower() == 'canada':
                pt = None
            elif region == 'Northwest Territories and Nunavut':
                pt = PT.XX
            else:
                pt = PT(region)

            for ghg in GHG:
                if ghg in [GHG.CH4, GHG.N2O]:
                    values = region_df[f'{ghg.value} (CO2eq)'].values
                else:
                    values = region_df[ghg.value].values
                years = region_df['Year'].values

                kt_by_yr = {int(year): float(val)
                            for year, val in zip(years, values)
                            if np.isfinite(val)}
                if kt_by_yr:
                    ts_zero = STS(
                        times=array.array('d', nir2025_year_ints),
                        t_unit=u.years,
                        values=array.array('d', [float('nan')] + [
                            kt_by_yr.get(yr, 0.0)
                            for yr in nir2025_year_ints]),
                        v_unit=u.kt_CO2e,
                        interpolation='no_interpolation')
                    if any(vv != 0 for vv in ts_zero.values[1:]):
                        if pt is None:
                            rval_ca[ipcc_sector, ghg] = ts_zero
                        else:
                            rval_pt[ipcc_sector, ghg, pt] = ts_zero
                else:
                    # we already set it to zero in initialization above
                    pass

    # now rectify the regional totals to make them match the national ones
    # by adding unassigned CO2e to the PT.XX region.
    # typically this is due to censored values in the provincial reports,
    # but it can be for other reasons too, such as differences in measurement
    # protocols or data sources.
    for sector in IPCC_Sector:
        for ghg in GHG:
            if isinstance(rval_ca[sector, ghg], STS):
                assert np.isfinite(rval_ca[sector, ghg].values[1:]).all()
            diff = rval_ca[sector, ghg] - rval_pt[sector, ghg].sum()
            if not isinstance(diff, STS):
                continue
            if abs(np.asarray(diff.values[1:])).max() < 5:  # kt_CO2e
                continue
            if sector in [IPCC_Sector.Non_Energy_Products_from_Fuels_and_Solvent_Use]:
                # Something's up with this one.
                # Look at e.g.
                #
                # inv[
                #    (inv['Total'] != 'y')
                #    & (inv['Year'] == 1990)
                #    & (inv['CategoryPathWithWhitespace'] == 'Non-Energy Products from Fuels and Solvent Use')
                #    ]
                #
                # Every province's emissions add up to
                # 12_000 kt, while the national total is listed as 5900.
                # It is the only IPCC sector whose provincial estimates exceed
                # the national one.
                #
                # Google Gemini suggested that the top-down national estimate
                # is more accurate because the provinces aren't positioned to
                # estimate this sector very well without double counting
                #
                # In this function, below, we set the PT.XX emissions as
                # negative.
                eps = None
            elif sector in [IPCC_Sector.Incineration_and_Open_Burning_Waste,
                            IPCC_Sector.Industrial_Wastewater_Treatment_and_Discharge]:
                eps = 1.1
                assert (np.asarray(diff.values[1:])
                        >= -eps).all(), (sector, ghg, diff)
            else:
                eps = 1e-3
                assert (np.asarray(diff.values[1:])
                        >= -eps).all(), (sector, ghg, diff)

            rval_pt[sector, ghg, PT.XX] += diff

    return rval_pt, rval_ca


class NIR2025(DynamicElement):

    # this year (inclusive) and previous ones are used by the class
    # set to e.g. 2020 to ignore data from years 2021 and beyond
    last_training_year: int = 3000

    skip_registered_sectors: bool = False


    def filter_ts(self, ts, ghg):
        scale = 1.0 / ghgvalues.GWP_100[ghg].magnitude
        rval = STS(
            times=array.array('d', [
                yr for yr in ts.times
                if yr <= self.last_training_year]),
            t_unit=u.years,
            values=array.array('d', [0] + [
                scale * val
                for yr, val in zip(ts.times, ts.values[1:])
                if yr <= self.last_training_year]),
            v_unit=kt_by_ghg[ghg] / u.year,
            interpolation='current')
        return rval

    def on_add_project(self, state):

        driver_name = f'NIR-2025 Emissions Placeholder'
        state.declare_sts(
            self,
            sts=SparseTimeSeries(
                identifier=driver_name,
                default_value=1.0 * u.dimensionless,
                t_unit=u.year),
            write=True)
        for pt in PT:
            state.register_driver(
                pt=pt,
                driver=driver_name,
                sts_key=driver_name)

        if self.skip_registered_sectors:
            skip_sectors = {
                ipcc_sector_key
                for by_pt in state.registries['emission_factor'].values()
                for by_ipcc_sector in by_pt.values()
                for ipcc_sector_key in by_ipcc_sector}
        else:
            skip_sectors = {}

        co2e_pt, co2e_ca = co2e_objtensors()

        for (sector, ghg, pt), offset in co2e_pt.ravel_keys_offsets():
            obj = co2e_pt.buf[offset]
            if not isinstance(obj, STS):
                continue
            assert obj.t_unit == u.year
            assert obj.v_unit == u.kt_CO2e

            if sector in skip_sectors:
                continue

            name = f'{self.__class__.__name__} {ghg.value} from {sector.value} in {pt.value}'
            state.declare_sts(
                project=self,
                sts=self.filter_ts(obj, ghg),
                name=name,
                write=True)
            state.register_emission_factor(
                pt=pt,
                driver=driver_name,
                sts_key=name,
                ipcc_sector=sector,
                ghg=ghg)


def NIR2025_EmissionsResults():
    state = State(
        name=f'foo',
        t_start=1990 * u.years)
    state.add_project(NIR2025())
    state.run_until(2024 * u.years)
    rval = state.compute_annual_emissions()
    return rval
