import functools

import pandas as pd
import numpy as np

from . import enums
from .ureg import u 
from .sc_utils import zip_table_to_dataframe
from . import objtensor
from . import sts

Factors = {'units': 1, 'thousands': 1000}

unit_by_fuel_type = {
    enums.CoalType.CanadianBituminous: ('Canadian bituminous coal', 'Metric tonnes', u.tonne_coal_bit),
    enums.CoalType.CanadianSubbituminous: ('Canadian subbituminous coal', 'Metric tonnes', u.tonne_coal_subbit),
    enums.CoalType.ImportedBituminous: ('Imported bituminous coal', 'Metric tonnes', u.tonne_coal_bit),
    enums.CoalType.ImportedSubbituminous: ('Imported subbituminous coal', 'Metric tonnes', u.tonne_coal_subbit),
    enums.CoalType.Lignite: ('Lignite', 'Metric tonnes', u.tonne_lignite),
    enums.FuelType.NaturalGasMkt: ('Natural gas', 'Cubic metres', u.m3_NG_mk),
    enums.FuelType.HeavyFuelOil: ('Total heavy fuel oil', 'Kilolitres', u.kilolitres_HFO),
    enums.FuelType.LightFuelOil: ('Light fuel oil', 'Kilolitres', u.kilolitres_LFO),
    enums.FuelType.Diesel: ('Diesel', 'Kilolitres', u.kilolitres_diesel),
    enums.FuelType.PetCoke: ('Petroleum coke', 'Metric tonnes', u.tonne_petcoke),
    enums.FuelType.Wood: ('Wood', 'Metric tonnes', u.tonne_wood_mc25),
    enums.FuelType.Methane: ('Methane', 'Cubic metres', u.m3_methane),
    enums.FuelType.StillGas: ('Other gaseous fuels', 'Cubic metres', u.m3_stillgas),
}

_actual_PTs = [pt for pt in enums.PT if pt != enums.PT.XX]

def set_ptxx(rval_pt, rval_ca):
    assert rval_pt.ndim == 2
    assert rval_ca.ndim == 1
    for key in rval_ca.dims[0]:
        eps = 1e-4
        try:
            # sum up the values listed for provinces and territories
            times = sts.union_times(rval_pt[key, _actual_PTs])
            pt_total = rval_pt[key, _actual_PTs].apply(
                functools.partial(sts.with_default_zero, times=times)).sum()
            xx = rval_ca[key] - pt_total
            # xx may be a pint quantity of a SparseTimeSeries
            if isinstance(xx, sts.STS):
                assert np.isnan(xx.values[0])
                # It can happen, e.g. in the case of "Other" energy sources
                # in Alberta, 2015 in product 20, that the provincial
                # data adds up to more than the federal data, when using
                # 0 for all missing entries. So this assertion isn't always
                # valid.
                #for tt, vv in zip(xx.times, xx.values[1:]):
                #    assert vv >= -eps, (tt, vv)
                if np.allclose(xx.values[1:], 0):
                    # in the interest of keeping things compact, if the
                    # difference is about 0, then just put in a zero
                    xx = 0 * xx.v_unit
            else:
                assert xx.magnitude >= -eps
            rval_pt[key, enums.PT.XX] = xx
        except Exception as exc:
            raise RuntimeError(key) from exc


@functools.cache
def Archived_Electric_Power_Generation_Annual_Fuel_Consumed_by_Electrical_Utility():
    df = zip_table_to_dataframe("25-10-0017-01")
    df['REF_DATE'] = df['REF_DATE'].apply(pd.to_numeric)
    assert 'int' in str(df.REF_DATE.values.dtype)

    rval_pt = objtensor.empty(unit_by_fuel_type, enums.PT)
    rval_ca = objtensor.empty(unit_by_fuel_type)

    for fuel_type, (name, exp_uom, unit) in unit_by_fuel_type.items():
        fuel_df = df[df['Fuel consumed for electric power generation'] == name]
        assert len(fuel_df) > 10, fuel_type
        rval_pt[fuel_type, :] = 0.0 * unit

        support_years = np.arange(2005, 2022) * u.years

        for (pt_name,), pt_df in fuel_df.groupby(['GEO']):
            uom, = list(set(pt_df.UOM.values))
            assert uom == exp_uom
            factor_str, = list(set(pt_df.SCALAR_FACTOR.values))
            factor = Factors[factor_str]
            rep = sts.annual_report2(
                years=pt_df.REF_DATE.values,
                values=(pt_df.VALUE.values * factor),
                v_unit=unit,
                include_nan_values=False)
            # these can be missing or sensored, but either way
            # skipping is okay, b/c (a) it is set to zero by the usum with support
            # and then sensored contributions to the national total are associated
            # with pseudo-province PT.XX
            assert max(pt_df.REF_DATE.values) <= 2021
            assert min(pt_df.REF_DATE.values) >= 2005
            rep.setdefault_zero(support_years)
            if pt_name == 'Canada':
                rval_ca[fuel_type] = rep
            else:
                rval_pt[fuel_type, enums.PT(pt_name)] = rep

    set_ptxx(rval_pt, rval_ca)
    return rval_pt, rval_ca


@functools.cache
def Electric_Power_Annual_Generation_by_Class_of_Producer():
    df = zip_table_to_dataframe("25-10-0020-01")
    df['REF_DATE'] = df['REF_DATE'].apply(pd.to_numeric)
    EP = enums.ElectricityProducer
    EGT = enums.ElectricityGenerationTech
    PT = enums.PT
    rval_pt = objtensor.empty(EP, EGT, PT)
    rval_ca = objtensor.empty(EP, EGT)

    rval_pt[:] = 0 * u.megawatt_hours
    rval_ca[:] = 0 * u.megawatt_hours

    items = {
        EP.Utilities: ('Electricity producer, electric utilities', 'Megawatt hours'),
        EP.Industry: ('Electricity producer, industries', 'Megawatt hours'),
    }
    unit = u.megawatt_hours

    for ep, (producer_name, exp_uom) in items.items():
        ep_df = df[df['Class of electricity producer'] == producer_name]
        for (pt_name, gen_type), pt_df in ep_df.groupby(['GEO', 'Type of electricity generation']):
            if gen_type.startswith('Total'):
                continue
            uom_str, = set(pt_df.UOM.values)
            assert uom_str == exp_uom
            factor_str, = set(pt_df.SCALAR_FACTOR.values)
            factor = Factors[factor_str]
            as_d = dict(zip(pt_df.REF_DATE.values, pt_df.VALUE.values))
            egt = EGT(gen_type)
            upper_valid_year = {
                EGT.CombustionTurbine: 2014,
                EGT.ConventionalSteam: 2014,
                EGT.InternalCombustion: 2014,
            }.get(egt, 2025)
            for year in range(2005, upper_valid_year):
                # XXX Underestimates missing data
                as_d.setdefault(year, 0)
                if np.isnan(as_d[year]):
                    #assert pt_name.lower() != 'canada', (ep, gen_type, year)
                    # we'll set it to zero here, and then
                    # pick up the difference in the PT.XX
                    as_d[year] = 0
                    # XXX There is still trouble though, data can be
                    # unavailable or sensored. The PT.XX mechanism
                    # does not help with unavailable data, and that is
                    # the most common. Sometimes it appears safe to estimate
                    # 0 for unavailable data, but not always.
                    # The most obviously non-zero unavailable data
                    # is the national totals for CombustionTurbine after
                    # 2014

            times, values = zip(*list(sorted(as_d.items())))
            rep = sts.annual_report2(
                years=times,
                values=[vv * factor for vv in values],
                include_nan_values=False,
                v_unit=unit)
            if pt_name.lower() == 'canada':
                rval_ca[ep, egt] = rep
            else:
                rval_pt[ep, egt, PT(pt_name)] = rep
        set_ptxx(rval_pt[ep], rval_ca[ep])

    return rval_pt, rval_ca
