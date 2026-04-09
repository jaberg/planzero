import array
import numpy as np

from .ureg import u
from .sc_3210013001 import Livestock, Livestock_nonsums
from . import objtensor
from . import sts

def table_A3p4_11(year_min_inclusive=1990, year_max_inclusive=2023):
    # Table A3.4-11: CH4 Emission factors for enteric fermentation in cattle,
    # selected years
    # kg CH4 / head / yr

    headers = ['Year', 'Dairy cows', 'Dairy Heifers', 'Bulls',
               'Beef Cows', 'Beef Heifers', 'Heifers for slaughter', 'Steers',
               'Calves']
    numbers = [
        [1990, 115.4, 79.4, 108.0, 105.9, 82.5, 44.7, 41.4, 43.8],
        [2005, 125.0, 77.2, 119.9, 114.4, 87.0, 52.8, 46.0, 43.6],
        [2017, 138.1, 76.7, 130.2, 120.9, 91.4, 53.7, 48.4, 43.8],
        [2018, 139.6, 76.7, 125.5, 120.6, 91.4, 54.0, 48.7, 43.9],
        [2019, 142.1, 76.7, 124.2, 120.4, 91.2, 54.2, 49.3, 43.8],
        [2020, 142.9, 76.7, 124.2, 120.5, 91.3, 54.2, 49.6, 43.9],
        [2021, 145.3, 76.7, 127.6, 120.0, 90.9, 54.2, 49.7, 43.9],
        [2022, 145.8, 76.7, 123.2, 120.7, 91.4, 54.6, 49.9, 43.9],
        [2023, 145.6, 76.7, 124.6, 121.5, 91.8, 54.4, 49.6, 44.0],
    ]
    assert len(headers) == len(numbers)
    numbers_T = np.asarray(numbers).T
    del numbers
    def ar2(colidx):
        years = np.arange(year_min_inclusive, year_max_inclusive + 1)
        values = np.interp(years, numbers_T[0], numbers_T[colidx])
        return sts.STS(
            times=array.array('d', years),
            t_unit=u.years,
            values=array.array('d', [values[0]] + list(values)),
            v_unit=u.kg_CH4 / u.cattle / u.year,
            interpolation=sts.InterpolationMode.current)
    rval = objtensor.empty(Livestock_nonsums)
    rval[Livestock.DairyCows] = ar2(1)
    rval[Livestock.DairyHeifers] = ar2(2)
    rval[Livestock.Bulls] = ar2(3)
    rval[Livestock.BeefCows] = ar2(4)
    rval[Livestock.BeefHeifers] = ar2(5)
    rval[Livestock.SlaughterHeifers] = ar2(6)
    rval[Livestock.Steers] = ar2(7)
    rval[Livestock.Calves] = ar2(8)
    return rval
