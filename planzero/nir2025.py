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
import numpyro.distributions as dist


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
from .symmetric_blended_lognormal import SymmetricBlendedLogNormal



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
idx_of_yr = {yr: ii for ii, yr in enumerate(nir2025_year_ints)} # deprecate name
idx_of_year = idx_of_yr


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


def load_uncertainty():
    """Return uncertainty as DataFrame with standard pre-processing
    """
    str_columns = [
        'IPCC_Source_Category_Code',
        'IPCC_Source_Category',
        'Gas',
    ]
    float_columns = [
        'Emissions (ktCO2eq) - Base Year',
        'Emissions (ktCO2eq) - 2023',
    ]
    percent_columns = [
        'Activity Data Uncertainty (%) - Base Year',
        'Activity Data Uncertainty (%) - 2023',
        'Emission Factor Uncertainty (%) - Base Year',
        'Emission Factor Uncertainty (%) - 2023',
        'Combined Uncertainty (%) - Base Year',
        'Combined Uncertainty (%) - 2023',
        'Combined uncertainty as % of TOTAL (%) - Base Year',
        'Combined uncertainty as % of TOTAL (%) - 2023',
    ]
    df = pd.read_excel(
        os.path.join(os.environ['PLANZERO_DATA'], 'EN_Annex2_Uncertainty.xlsx'),
        sheet_name='Table A2-2',
        usecols='A:M',
        skiprows=4,
        nrows=129 - 4,
        names=str_columns + float_columns + percent_columns,
        )
    for col in float_columns:
        df[col] = df[col].astype(float)
    for col in percent_columns:
        df[col] = df[col].replace('-', float('nan')).astype(float)
    return df


@cache
def near_zero_sector_ghgs():
    """Return a dictionary of (sector, ghg) pairs that have historically been
    negligible.
    The values in the dictionary are the total of absolute-values of ktCO2e associated.

    This function serves to standardize the definition of small, irrelevant
    sector-gas combinations across various models and visualizations.
    """
    _, arr_ca = ktCO2e_dense_w_nan()
    rval = {}
    for sector in IPCC_Sector:
        for ghg in GHG:
            total_abs = np.nansum(abs(arr_ca[idx_of_sector[sector], idx_of_ghg[ghg]]))
            if total_abs < 1: # ktCO2e
                rval[sector, ghg] = total_abs
    return rval


def ktCO2e_numpyro_dist_pt_ca(sector, ghg, year):
    """Returns  ca_dist, pt_dists""" # XXX backward rel to fn name
    arr_pt, arr_ca = ktCO2e_dense_w_nan()
    ktco2e_pt = arr_pt[idx_of_sector[sector], idx_of_ghg[ghg], :, idx_of_year[year]]
    ktco2e_ca = arr_ca[idx_of_sector[sector], idx_of_ghg[ghg], idx_of_year[year]]

    if abs(ktco2e_ca) <= 1 or (sector, ghg) in near_zero_sector_ghgs():
        # Case 1:
        # 1kt CO2e, aka very small across all provinces and territories for all years
        ca_dist = dist.Normal(0, 1)
        # pt dists whose variances add up to 1
        pt_dists = [dist.Normal(0, np.sqrt(1 / 13))] * 13
    else:
        pc_unc_2023 = uncertainty_percent_by_IPCC_Sector_GHG_2023()[sector, ghg]
        pc_unc_1990 = uncertainty_percent_by_IPCC_Sector_GHG_1990()[sector, ghg]
        unc = max(
            np.interp(year, [1990, 2023], [pc_unc_1990, pc_unc_2023]) / 100,
            0.01) # at least 1% uncertainty
        if np.isnan(unc):
            # happens with wastewater CO2 only case I think
            unc = 1.0 # 100% uncertainty

        if sector in [IPCC_Sector.Harvested_Wood_Products,
                      IPCC_Sector.Forest_Land,
                      IPCC_Sector.Cropland,
                      IPCC_Sector.Settlements, # some provinces sometimes have negative settlement emissions
                     ]:
            # Case 2: sectors that can be negative or positive
            rolloff = 100 # kt CO2e
            eps = 1
            ca_dist = SymmetricBlendedLogNormal.rolloff_relerr(
                mu=ktco2e_ca,
                rolloff=rolloff,
                relerr=unc)

            pt_dists = []
            pt_total_abs = np.nansum(abs(ktco2e_pt))

            for pt in PT:
                if pt == PT.XX:
                    continue
                if np.isnan(ktco2e_pt[idx_of_pt[pt]]):
                    pt_dist = SymmetricBlendedLogNormal.rolloff_relerr(
                        mu=0,
                        rolloff=rolloff,
                        relerr=max(unc, 1.0),
                        )
                else:
                    pt_dist = SymmetricBlendedLogNormal.rolloff_relerr(
                        mu=ktco2e_pt[idx_of_pt[pt]],
                        rolloff=rolloff,# * abs(ktco2e_pt[idx_of_pt[pt]]) / abs(ktco2e_ca) + eps,
                        relerr=unc,
                        )
                pt_dists.append(pt_dist)
        else:
            # Case 2: sectors that cannot be negative
            eps = 1
            pos_ktco2e_ca = max(ktco2e_ca, 0.1)
            try:
                ca_dist = SymmetricBlendedLogNormal.rolloff_relerr(
                    mu=pos_ktco2e_ca,
                    rolloff=pos_ktco2e_ca * .1 + eps, # small -> one-sided dist
                    relerr=unc)
            except Exception as e:
                raise Exception(sector, ghg, ktco2e_ca, unc) from e

            pt_dists = []
            pt_total = np.nansum(ktco2e_pt)
            assert np.nanmin(ktco2e_pt) >= 0, (sector, ghg, ktco2e_pt)

            def positive_nearzero_dist():
                return SymmetricBlendedLogNormal.rolloff_relerr(
                    mu=.5 * eps,
                    rolloff=.01 * eps, # very little no negative mass
                    relerr=.5
                    )

            for pt in PT:
                if pt == PT.XX:
                    continue

                if pt == PT.NU and year <= 1998:
                    # Nunavut created in 1999
                    pt_dist = positive_nearzero_dist()

                elif np.isnan(ktco2e_pt[idx_of_pt[pt]]):
                    # pt emissions for this year are not known
                    pt_nanmean_over_time = np.nanmean(arr_pt[idx_of_sector[sector], idx_of_ghg[ghg], idx_of_pt[pt], :])
                    if sector == IPCC_Sector.SCS__Oil_and_Gas_Extraction and pt == PT.NT and year < 1999:
                        # human judgement: no oil & gas sector in NT at this time
                        pt_dist = positive_nearzero_dist()
                    elif np.isnan(pt_nanmean_over_time): 
                        # emissions for this PT were never reported, let's assume they were 0.
                        pt_dist = positive_nearzero_dist()
                    else:
                        # emissions for this PT were sometimes reported
                        pt_dist = SymmetricBlendedLogNormal.rolloff_relerr(
                            mu=max(pt_nanmean_over_time, eps),
                            rolloff=max(pt_nanmean_over_time, eps) * .1, # small relative to mu -> one-sided dist
                            relerr=0.7, # uncertainty on order of mean
                            )
                else:
                    try:
                        if ktco2e_pt[idx_of_pt[pt]] < eps:
                            # typically, ktco2e_pt[idx_of_pt[pt]] is listed as being actually 0
                            pt_dist = positive_nearzero_dist()
                        else:
                            pt_dist = SymmetricBlendedLogNormal.rolloff_relerr(
                                mu=ktco2e_pt[idx_of_pt[pt]],
                                rolloff=.1 * ktco2e_pt[idx_of_pt[pt]], # small relative to mu -> one-sided dist
                                relerr=unc
                                )
                    except Exception as e:
                        raise Exception(sector, ghg, pt, ktco2e_pt[idx_of_pt[pt]]) from e
                pt_dists.append(pt_dist)

    return ca_dist, pt_dists


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


# TODO: DELETE
def NIR2025_EmissionsResults():
    state = State(
        name=f'foo',
        t_start=1990 * u.years)
    state.add_project(NIR2025())
    state.run_until(2024 * u.years)
    rval = state.compute_annual_emissions()
    return rval


def uncertainty_percent_by_IPCC_Sector_GHG_2023():
    return uncertainty_percent_by_IPCC_Sector_GHG(
        percent_col='Combined Uncertainty (%) - 2023')


def uncertainty_percent_by_IPCC_Sector_GHG_1990():
    return uncertainty_percent_by_IPCC_Sector_GHG(
        percent_col='Combined Uncertainty (%) - Base Year')

@cache
def uncertainty_percent_by_IPCC_Sector_GHG(percent_col:str):
    df = load_uncertainty()
    rval = {}
    for record in df.iloc:
        desc = record['IPCC_Source_Category']
        ghg = GHG(record['Gas'])
        pc = record[percent_col]

        if desc == "Fuel Combustion - Public Electricity and Heat Production":
            rval[IPCC_Sector.SCS__Public_Electricity_and_Heat, ghg] = pc

        elif desc == "Fuel Combustion - Petroleum Refining":
            rval[IPCC_Sector.SCS__Petroleum_Refining_Industries, ghg] = pc

        elif desc == "Fuel Combustion - Manufacture of Solid Fuels and Other Energy Industries":
            rval[IPCC_Sector.SCS__Oil_and_Gas_Extraction, ghg] = pc

        elif desc == "Fuel Combustion - Manufacturing Industries and Construction":
            rval[IPCC_Sector.SCS__Manufacturing__NonFerrous, ghg] = pc
            rval[IPCC_Sector.SCS__Manufacturing__Pulp_and_Paper, ghg] = pc
            rval[IPCC_Sector.SCS__Manufacturing__Chemical, ghg] = pc
            rval[IPCC_Sector.SCS__Manufacturing__Cement, ghg] = pc
            rval[IPCC_Sector.SCS__Manufacturing__Other, ghg] = pc
            rval[IPCC_Sector.SCS__Manufacturing__Iron_and_Steel, ghg] = pc
            rval[IPCC_Sector.SCS__Construction, ghg] = pc
            rval[IPCC_Sector.SCS__Agriculture_and_Forestry, ghg] = pc
            rval[IPCC_Sector.SCS__Mining, ghg] = pc

        elif desc == "Fuel Combustion -Off-Roadb":
            rval[IPCC_Sector.Transport__Other__Agriculture_and_Forestry, ghg] = pc
            rval[IPCC_Sector.Transport__Other__Commercial_and_Institutional, ghg] = pc
            rval[IPCC_Sector.Transport__Other__Mfg_Mining_Construction, ghg] = pc
            rval[IPCC_Sector.Transport__Other__Residential, ghg] = pc
            rval[IPCC_Sector.Transport__Other__Other, ghg] = pc

        elif desc == "Fuel Combustion - Civil Aviation":
            rval[IPCC_Sector.Transport__Air__Domestic_Civil, ghg] = pc

        elif desc == "Fuel Combustion - Road Transportation":
            rval[IPCC_Sector.Transport__Road__Light_Duty_Gasoline_Vehicles, ghg] = pc
            rval[IPCC_Sector.Transport__Road__Light_Duty_Gasoline_Trucks, ghg] = pc
            rval[IPCC_Sector.Transport__Road__Heavy_Duty_Gasoline_Vehicles, ghg] = pc
            rval[IPCC_Sector.Transport__Road__Motorcycles, ghg] = pc
            rval[IPCC_Sector.Transport__Road__Light_Duty_Diesel_Vehicles, ghg] = pc
            rval[IPCC_Sector.Transport__Road__Light_Duty_Diesel_Trucks, ghg] = pc
            rval[IPCC_Sector.Transport__Road__Propane_and_Natural_Gas_Vehicles, ghg] = pc
            rval[IPCC_Sector.Transport__Road__Heavy_Duty_Diesel_Vehicles, ghg] = pc

        elif desc == "Fuel Combustion - Railways":
            rval[IPCC_Sector.Transport__Rail, ghg] = pc

        elif desc == "Fuel Combustion - Navigation":
            rval[IPCC_Sector.Transport__Marine__Domestic, ghg] = pc

        elif desc == "Fuel Combustion - Pipeline Transport":
             rval[IPCC_Sector.Transport__Other__Pipeline, ghg] = pc

        elif desc == "Fuel Combustion - Other Sectors":
            rval[IPCC_Sector.SCS__Commercial_and_Institutional, ghg] = pc
            rval[IPCC_Sector.SCS__Residential, ghg] = pc

        elif desc == "Fuel Combustion - Fishing":
            rval[IPCC_Sector.Transport__Marine__Fishing, ghg] = pc

        elif desc == "Fuel Combustion - Other (Military Aviation)":
            rval[IPCC_Sector.Transport__Air__Military, ghg] = pc

        elif desc == "Fuel Combustion - Other (Military Navigation)":
            rval[IPCC_Sector.Transport__Marine__Military, ghg] = pc

        elif desc == "Fugitive Sources - Coal Mining":
            rval[IPCC_Sector.Fugitive__Coal, ghg] = pc

        elif desc == "Fugitive Sources - Oil & Gas":
            rval[IPCC_Sector.Fugitive__Oil, ghg] = pc
            rval[IPCC_Sector.Fugitive__Natural_Gas, ghg] = pc

        elif desc == "Fugitive Sources - Venting":
            rval[IPCC_Sector.Fugitive__Venting, ghg] = pc

        elif desc == "Fugitive Sources - Flaring":
            rval[IPCC_Sector.Fugitive__Flaring, ghg] = pc

        elif desc == "Fugitive Sources - Venting & Flaring": # for CO2
            rval[IPCC_Sector.Fugitive__Flaring, ghg] = pc
            rval[IPCC_Sector.Fugitive__Venting, ghg] = pc

        elif desc == "CO2 Transport and Storage":
            rval[IPCC_Sector.CO2_Transport_and_Storage, ghg] = pc

        elif desc == "IPPU - Cement Production":
            rval[IPCC_Sector.Cement_Production, ghg] = pc

        elif desc == "IPPU - Lime Production":
            rval[IPCC_Sector.Lime_Production, ghg] = pc
            
        elif desc == "IPPU - Glass Production":
            pass
        elif desc == "IPPU - Other Uses of Soda Ash":
            pass

        elif desc == "IPPU - Other (Magnesite Use)":
            pass
        elif desc == "IPPU - Other (Limestone and Dolomite Use)":
            rval[IPCC_Sector.Mineral_Product_Use, ghg] = pc

        elif desc == "IPPU - Ammonia Production":
            rval[IPCC_Sector.Ammonia_Production, ghg] = pc

        elif desc == "IPPU - Nitric Acid Production":
            rval[IPCC_Sector.Nitric_Acid_Production, ghg] = pc
        elif desc == "IPPU - Adipic Acid Production":
            rval[IPCC_Sector.Adipic_Acid_Production, ghg] = pc
        elif desc == "IPPU - Soda Ash Production":
            pass

        elif desc == "IPPU - Petrochemical and Carbon Black Production":
            rval[IPCC_Sector.Petrochemical_and_Carbon_Black_Production, ghg] = pc
        elif desc == "IPPU - Petrochemical and Carbon Black Production (including carbide production)":
            rval[IPCC_Sector.Petrochemical_and_Carbon_Black_Production, ghg] = pc

        elif desc == "IPPU - Fluorochemical Production":
            # HFCs
            rval[IPCC_Sector.Production_and_Consumption_of_Halocarbons, ghg] = pc

        elif desc == "IPPU - Iron and Steel Production":
            rval[IPCC_Sector.Iron_and_Steel_Production, ghg] = pc
        elif desc == "IPPU - Aluminium Production":
            rval[IPCC_Sector.Aluminium_Production, ghg] = pc
        elif desc == "IPPU - Magnesium Production":
            rval[IPCC_Sector.Magnesium_Production_and_Casting, ghg] = pc
        elif desc == "IPPU - Non-Energy Products from Fuels and Solvent Use Other - Other (Other and Undifferentiated)":
            rval[IPCC_Sector.Non_Energy_Products_from_Fuels_and_Solvent_Use, ghg] = pc

        elif desc == "IPPU - Non-Energy Products from Fuels and Solvent Use Other - Other (Use of Urea in SCR Vehicles)":
            rval[IPCC_Sector.Non_Energy_Products_from_Fuels_and_Solvent_Use, ghg] = pc

        elif desc == "IPPU - Integrated Circuit or Semiconductor":
            # PFCs, SF6, NF3
            rval[IPCC_Sector.Production_and_Consumption_of_Halocarbons, ghg] = pc
            pass
        elif desc == "IPPU - Other Emissive Applications":
            pass
        elif desc == "IPPU - Product Uses as Substitutes for Ozone Depleting Substances":
            # HFCs
            rval[IPCC_Sector.Other_Product_Manufacture_and_Use, ghg] = pc
            pass
        elif desc == "IPPU - Electrical Equipment":
            # SF6
            rval[IPCC_Sector.Other_Product_Manufacture_and_Use, ghg] = pc
            pass
        elif desc == "IPPU - Other (Medical Applications of N2O)":
            #N2O
            rval[IPCC_Sector.Other_Product_Manufacture_and_Use, ghg] = pc
        elif desc == "IPPU - Other (Uses of N2O for Propellant)":
            pass # percents are the same as above (medical applications)
        elif desc == "IPPU - Other Contained Product Uses":
            # PFCs
            rval[IPCC_Sector.Other_Product_Manufacture_and_Use, ghg] = pc

        elif desc == "Agriculture - Total CH4": # aggregate
            pass
        elif desc == "Agriculture - Enteric Fermentation":
            rval[IPCC_Sector.Enteric_Fermentation, ghg] = pc
        elif desc == "Agriculture - Manure Management ":
            # CH4
            rval[IPCC_Sector.Manure_Management, ghg] = pc
        elif desc == "Agriculture - Field Burning of Agricultural Residues":
            rval[IPCC_Sector.Field_Burning_of_Agricultural_Residues, ghg] = pc
        elif desc == "Agriculture - Total N2O": # aggregate
            pass
        elif desc == "Agriculture - Manure Management Direct Emissions ":
            # N2O
            rval[IPCC_Sector.Manure_Management, ghg] = pc
            pass
        elif desc == "Agriculture - Manure Management Indirect Emissions ":
            # N2O
            # Admittedly an error to pass, but correct behaviour doesn't seem
            # possible.
            # This and the row above both offer uncertainty estimates for the N2O
            # from Manure Management. Which to use?  This is a smaller
            # contributor to IPCC_Sector.Manure_Management than the direct
            # emissions above, so use the pc from Direct Emissions
            pass
        elif desc == "Agriculture - Direct Agriculture Soils ":
            rval[IPCC_Sector.Agricultural_Soils_Direct, ghg] = pc
        elif desc == "Agriculture - Indirect Agriculture Soils":
            rval[IPCC_Sector.Agricultural_Soils_Indirect, ghg] = pc
        elif desc == "Agriculture - Total CO2":
            pass
        elif desc == "Agriculture - Limestone CaCO3":
            pass
        elif desc == "Agriculture - Urea Application":
            # CO2
            # both this and "Limestone CaCO3" above offer percentage
            # uncertainty for the sector. The error estimates are similar,
            # so just going with this one.
            rval[IPCC_Sector.Liming_Urea_Other, ghg] = pc
        elif desc == "Agriculture - Other Carbon-Containing Fertilizers":
            pass
        elif desc == "LULUCF - Forest Land Remaining Forest Land":
            rval[IPCC_Sector.Forest_Land, ghg] = pc
        elif desc == "LULUCF - Land Converted to Forest Land":
            pass
        elif desc == "LULUCF - Cropland ":
            rval[IPCC_Sector.Cropland, ghg] = pc
        elif desc == "LULUCF - Grassland":
            rval[IPCC_Sector.Grassland, ghg] = pc
        elif desc == "LULUCF - Wetlands ":
            rval[IPCC_Sector.Wetlands, ghg] = pc
        elif desc == "LULUCF - Settlements ":
            rval[IPCC_Sector.Settlements, ghg] = pc
        elif desc == "LULUCF - Conversion of Forest Land ":
            pass
        elif desc == "LULUCF - Harvested Wood Products (HWP)":
            rval[IPCC_Sector.Harvested_Wood_Products, ghg] = pc
        elif desc == "Solid Waste Disposal - Managed Waste Disposal Sites":
            rval[IPCC_Sector.Municipal_Solid_Waste_Landfills, ghg] = pc
        elif desc == "Biological Treatment of Solid Waste - Composting":
            rval[IPCC_Sector.Biological_Treatment_of_Solid_Waste, ghg] = pc
        elif desc == "Biological Treatment of Solid Waste - Anerobic Digestion - Industrial & Municipal Facilities":
            rval[IPCC_Sector.Industrial_Wood_Waste_Landfills, ghg] = pc

        elif desc == "Incineration and Open Burning of Waste - Waste Incineration":
            rval[IPCC_Sector.Incineration_and_Open_Burning_Waste, ghg] = pc
        elif desc == "Wastewater Treatment and Discharge":
            rval[IPCC_Sector.Municipal_Wastewater_Treatment_and_Discharge, ghg] = pc
            rval[IPCC_Sector.Industrial_Wastewater_Treatment_and_Discharge, ghg] = pc
        else:
            raise NotImplementedError(desc)

    # extra guesses
    assert (IPCC_Sector.Cropland, GHG.CH4) not in rval
    rval[IPCC_Sector.Cropland, GHG.CH4] = 45. # kind of an average of Conversion of Forest Land and GrassLand
    rval[IPCC_Sector.Settlements, GHG.CH4] = 45. # kind of an average of Conversion of Forest Land and GrassLand
    rval[IPCC_Sector.Settlements, GHG.N2O] = 45. # kind of an average of Conversion of Forest Land and GrassLand
    return rval


if __name__ == '__main__':
    if 0:
        df = load_uncertainty()
        foo = {}
        for record in df.iloc:
            foo.setdefault(record['IPCC_Source_Category'], {})
            foo[record['IPCC_Source_Category']].setdefault(record['Gas'], {})

        assert len(foo) == len(IPCC_Sector)
        for key, sector in zip(foo, IPCC_Sector):
            print(f'uncertainty_IPCC_Category["{key}"] = {sector}')
    else:
        uncertainty_percent_by_IPCC_Sector_GHG_2023()


