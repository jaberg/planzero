import matplotlib.pyplot as plt

from . import eccc_nir_annex6
from . import eccc_nir_annex13

from .enums import FuelType, GHG, Geo
from .ptvalues import PTValues, PTDim, national_total
from .sts import annual_report, STSDict
from .ureg import u, litres_by_fuel_type
from .sc_nir import Archived_Electric_Power_Generation_Annual_Fuel_Consumed_by_Electrical_Utility
from .ghgvalues import GWP_100



def consumption():
    scp = Archived_Electric_Power_Generation_Annual_Fuel_Consumed_by_Electrical_Utility()
    val_d = {}
    for fuel in OtherFuelsDim:
       ptv = scp.ptv_fuel_type(fuel)
       for (pt,), val in ptv.val_d.items():
           val_d[fuel, pt] = val
    fallback = STSDict(
        val_d={ft: 0 * litres_by_fuel_type[ft] for ft in OtherFuelsDim},
        broadcast=[False, True],
        dims=[OtherFuelsDim, PTDim])
    return STSDict(
        val_d=val_d,
        broadcast=[False, False],
        dims=[OtherFuelsDim, PTDim],
        fallback=fallback)


def coefficients():
    val_d = {}
    data = eccc_nir_annex6.data_a6_1_6
    ftmap = {
        'Light Fuel Oil': FuelType.LightFuelOil,
        'Heavy Fuel Oil': FuelType.HeavyFuelOil,
        'Diesel': FuelType.Diesel}

    for ii, (fuel_type_str, sector) in enumerate(zip(data['Fuel Type'], data['Sector'])):
        if sector == 'Electric Utilities' and fuel_type_str in ftmap:
            for ghg in [GHG.CO2, GHG.CH4, GHG.N2O]:
                coef = data[f'{ghg.value} (g/L)'][ii]
                fuel_type = ftmap[fuel_type_str]
                top = u.Quantity(f'{coef} g_{ghg.value}')
                val_d[ghg, fuel_type] = top / litres_by_fuel_type[fuel_type]
        elif sector == "Refineries and Others" and fuel_type_str == 'Diesel':
            for ghg in [GHG.CO2, GHG.CH4, GHG.N2O]:
                coef = data[f'{ghg.value} (g/L)'][ii]
                fuel_type = ftmap[fuel_type_str]
                top = u.Quantity(f'{coef} g_{ghg.value}')
                val_d[ghg, fuel_type] = top / litres_by_fuel_type[fuel_type]
    assert len(val_d) == 9

    # fill in dense values, rather than use fallback because units
    # are different for every element
    for ghg in GHG:
        for fuel_type in OtherFuelsDim:
            val_d.setdefault(
                (ghg, fuel_type),
                u.Quantity(f'0 g_{ghg.value}') / litres_by_fuel_type[fuel_type])

    return STSDict(
        val_d=val_d,
        dims=[GHG, OtherFuelsDim],
        broadcast=[False, False])


