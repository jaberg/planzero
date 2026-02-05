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
}


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

        support = sts.annual_report(
            times=np.arange(2005, 2022) * u.years,
            values=np.zeros(2022 - 2005) * unit)

        for (pt_name,), pt_df in fuel_df.groupby(['GEO']):
            uom, = list(set(pt_df.UOM.values))
            assert uom == exp_uom
            factor_str, = list(set(pt_df.SCALAR_FACTOR.values))
            factor = Factors[factor_str]
            rep = sts.annual_report(
                times=pt_df.REF_DATE.values * u.years,
                values=(pt_df.VALUE.values * factor) * unit,
                skip_nan_values=True)
            # these can be missing or sensored, but either way
            # skipping is okay, b/c (a) it is set to zero by the usum with support
            # and then sensored contributions to the national total are associated
            # with pseudo-province PT.XX
            assert max(pt_df.REF_DATE.values) <= 2021
            assert min(pt_df.REF_DATE.values) >= 2005
            rep_w_support = sts.usum([rep, support])
            if pt_name == 'Canada':
                rval_ca[fuel_type] = rep_w_support
            else:
                rval_pt[fuel_type, enums.PT(pt_name)] = rep_w_support
        xx = rval_ca[fuel_type] - sts.usum(rval_pt[fuel_type])
        assert np.isnan(xx.values[0])
        if not np.allclose(xx.values[1:], 0):
            rval_pt[fuel_type, enums.PT.XX] = xx

    return rval_pt, rval_ca


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

    for ep, (producer_name, exp_uom) in items.items():
        df = df[df['Class of electricity producer'] == producer_name]
        for (pt_name, gen_type), pt_df in df.groupby(['GEO', 'Type of electricity generation']):
            if gen_type.startswith('Total'):
                continue
            uom, = list(set(pt_df.UOM.values))
            assert uom == exp_uom
            factor_str, = list(set(pt_df.SCALAR_FACTOR.values))
            factor = Factors[factor_str]
            rep = sts.annual_report(
                times=pt_df.REF_DATE.values * u.years,
                values=pt_df.VALUE.values * u.Quantity(f'{factor} {uom.lower()}'),
                skip_nan_values=True)
            egt = EGT(gen_type)
            assert rep is not None
            if pt_name.lower() == 'canada':
                rval_ca[ep, egt] = rep
            else:
                rval_pt[ep, egt, PT(pt_name)] = rep
    for ep in EP:
        for egt in EGT:
            try:
                xx = rval_ca[ep, egt] - sts.usum(rval_pt[ep, egt])
            except Exception as exc:
                raise RuntimeError(ep, egt) from exc
            assert np.isnan(xx.values[0])
            if not np.allclose(xx.values[1:], 0):
                rval_pt[ep, egt, PT.XX] = xx

    return rval_pt, rval_ca
