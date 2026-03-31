import numpy as np

from . import ipcc_canada
from .ureg import u

_echart_years = ipcc_canada.echart_years()
url_est_nir = 'https://github.com/jaberg/planzero/blob/main/planzero/est_nir.py'

def _rstrip_data(ts, years=_echart_years):
    vals = list(ts.query([yy * u.years for yy in years]))
    while vals and (vals[-1].magnitude == 0 or np.isnan(vals[-1].magnitude)):
        vals.pop()
    rval = [
        {'value': 0 if np.isnan(vv.magnitude) else vv.magnitude,
         'url': url_est_nir}
         for vv in vals]
    return rval


def _echart_reference_NIR_values(catpathww, years=_echart_years):
    v_by_yr = ipcc_canada.annual_sector_Mt_CO2e_by_year(catpathww)
    values = []
    for yr in years:
        try:
            values.append(v_by_yr[yr])
        except KeyError:
            break
    return values
