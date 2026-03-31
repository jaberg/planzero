import enum
import functools

import pandas as pd

from .ureg import u
from .enums import PT
from .sc_utils import zip_table_to_dataframe, Factors, times_values

from . import objtensor
from . import sts


class SurveyDate(str, enum.Enum):
    Jul1 = 'At July 1'
    Jan1 = 'At January 1'


class Livestock(str, enum.Enum):
    TotalCattle = 'Total cattle'
    Bulls = 'Bulls, 1 year and over'
    DairyCows = 'Dairy cows'
    BeefCows = 'Beef cows'
    TotalHeifers = 'Total heifers'
    DairyHeifers = 'Heifers for dairy replacement'
    TotalBeefHeifers = 'Total beef heifers'
    BeefHeifers = 'Heifers for beef replacement'
    SlaughterHeifers = 'Heifers for slaughter'
    Steers = 'Steers, 1 year and over'
    Calves = 'Calves, under 1 year'

Livestock_nonsums = [
    Livestock.Bulls,
    Livestock.DairyCows,
    Livestock.BeefCows,
    Livestock.DairyHeifers,
    Livestock.BeefHeifers,
    Livestock.SlaughterHeifers,
    Livestock.Steers,
    Livestock.Calves]


class FarmType(str, enum.Enum):
    AllCattle = 'On all cattle operations'
    Dairy = 'On dairy operations'
    Beef = 'On beef operations'
    CowCalf = 'On cow calf operations'
    FeederStocker = 'On feeder and stocker operations'
    Feeding = 'On feeding operations'


@functools.cache
def number_of_cattle_by_class_and_farm_type():
    df = zip_table_to_dataframe("32-10-0130-01")
    df['REF_DATE'] = df['REF_DATE'].apply(pd.to_numeric)

    dct_pt = {}
    dct_ca = {}
    for row in df.iloc:
        survey_date = SurveyDate(row['Survey date'])
        livestock = Livestock(row['Livestock'])
        farm_type = FarmType(row['Farm type'])
        year = row['REF_DATE']
        value = row['VALUE'] * 1000
        if row['GEO'] == 'Canada':
            dct_ca.setdefault((survey_date, livestock, farm_type), {})[year] = value
            pass
        elif row['GEO'] in ('Eastern provinces', 'Atlantic provinces', 'Western provinces'):
            pass
        else:
            pt = PT(row['GEO'])
            dct_pt.setdefault((survey_date, livestock, farm_type, pt), {})[year] = value

    rval_pt = objtensor.empty(SurveyDate, Livestock, FarmType, PT)
    rval_pt[:] = 0 * u.cattle
    for key, val_by_year in dct_pt.items():
        years, values = zip(*sorted(val_by_year.items()))
        rval_pt[*key] = sts.annual_report2(years, values, v_unit=u.cattle)

    rval_ca = objtensor.empty(SurveyDate, Livestock, FarmType)
    rval_ca[:] = 0 * u.cattle
    for key, val_by_year in dct_ca.items():
        years, values = zip(*sorted(val_by_year.items()))
        rval_ca[*key] = sts.annual_report2(years, values, v_unit=u.cattle)
    return rval_pt, rval_ca


if __name__ == '__main__':
    rval_pt, rval_ca = number_of_cattle_by_class_and_farm_type()
    print(rval_ca[SurveyDate.Jan1, Livestock.TotalCattle, FarmType.AllCattle])
