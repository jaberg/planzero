import pandas as pd
import numpy as np

from .enums import CoalType, FuelType, PT
from .ureg import u 
from .sc_utils import zip_table_to_dataframe
from . import objtensor
from . import sts

unit_by_fuel_type = {
    CoalType.CanadianBituminous: ('Canadian bituminous coal', 'Metric tonnes', u.tonne_coal_bit),
    CoalType.CanadianSubbituminous: ('Canadian subbituminous coal', 'Metric tonnes', u.tonne_coal_subbit),
    CoalType.ImportedBituminous: ('Imported bituminous coal', 'Metric tonnes', u.tonne_coal_bit),
    CoalType.ImportedSubbituminous: ('Imported subbituminous coal', 'Metric tonnes', u.tonne_coal_subbit),
    CoalType.Lignite: ('Lignite', 'Metric tonnes', u.tonne_lignite),
    FuelType.NaturalGasMkt: ('Natural gas', 'Cubic metres', u.m3_NG_mk),
    FuelType.HeavyFuelOil: ('Total heavy fuel oil', 'Kilolitres', u.kilolitres_HFO),
    FuelType.LightFuelOil: ('Light fuel oil', 'Kilolitres', u.kilolitres_LFO),
    FuelType.Diesel: ('Diesel', 'Kilolitres', u.kilolitres_diesel),
}


def Archived_Electric_Power_Generation_Annual_Fuel_Consumed_by_Electrical_Utility():
    df = zip_table_to_dataframe("25-10-0017-01")
    df['REF_DATE'] = df['REF_DATE'].apply(pd.to_numeric)
    assert 'int' in str(df.REF_DATE.values.dtype)

    rval_pt = objtensor.empty(unit_by_fuel_type, PT)
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
            factor = {'units': 1, 'thousands': 1000}[factor_str]
            rep = sts.annual_report(
                times=pt_df.REF_DATE.values * u.years,
                values=(pt_df.VALUE.values * factor) * unit,
                skip_nan_values=True)
            assert max(pt_df.REF_DATE.values) <= 2021
            assert min(pt_df.REF_DATE.values) >= 2005
            rep_w_support = sts.usum([rep, support])
            if pt_name == 'Canada':
                rval_ca[fuel_type] = rep_w_support
            else:
                rval_pt[fuel_type, PT(pt_name)] = rep_w_support
        xx = rval_ca[fuel_type] - sts.usum(rval_pt[fuel_type])
        assert np.isnan(xx.values[0])
        if not np.allclose(xx.values[1:], 0):
            rval_pt[fuel_type, PT.XX] = xx

    return rval_pt, rval_ca
