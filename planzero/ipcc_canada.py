import os
import sys

import numpy as np
import pandas as pd
# from https://data-donnees.az.ec.gc.ca/data/substances/monitor/canada-s-official-greenhouse-gas-inventory/A-IPCC-Sector?lang=en

inv = pd.read_csv(os.path.join(os.environ['PLANZERO_DATA'], 'EN_GHG_IPCC_Can_Prov_Terr.csv'))

inv['CategoryPathWithWhitespace'] = (
    #inv['Source'].fillna('').astype(str) + '/' +
    inv['Category'].fillna('').astype(str) + '/' +
    inv['Sub-category'].fillna('').astype(str) + '/' +
    inv['Sub-sub-category'].fillna('').astype(str)
).str.rstrip('/')
assert not inv['CategoryPathWithWhitespace'].isna().any()

inv['CO2eq'] = inv['CO2eq'].replace('x', float('nan')).astype(float) # why 50_000??
inv['Year'] = inv['Year'].astype(float)

# non-aggregate rows relating to the entire country (remove subtotals)
non_agg = inv[ (inv['Region'].isin(['Canada', 'canada'])) & (inv['Total'] != 'y')]


def echart_years():
    return [float(x) for x in range(1990, 2060)]


def echart_series_Mt(catpath, name=None):
    non_agg_years = list(set(non_agg['Year'].unique()))
    non_agg_years.sort()
    datalen = len(non_agg_years)
    assert ('kt',) == inv['Unit'].unique()
    assert non_agg_years == echart_years()[:datalen]
    catpath_data = non_agg[non_agg['CategoryPathWithWhitespace'] == catpaths[catpath]]
    data = [(int(year), float(co2eq) / 1000)
            for year, co2eq in catpath_data[['Year', 'CO2eq']].values]
    data.sort()
    assert len(data) == datalen, (catpath, len(data), datalen)
    rval = dict(
        name=name or catpath,
        type='line',
        stack='Total',
        areaStyle={},
        emphasis={'focus': 'series'},
        encode={'x': 'year', 'y': 'value'},
        data=[{'year': year,
               'value': co2eq,
               'url': f'/ipcc-sectors/{catpath}'.replace(' ', '_')}
              for year, co2eq in data])
    return rval


def echart_series_all_Mt():
    non_agg_years = list(set(non_agg['Year'].unique()))
    non_agg_years.sort()
    datalen = len(non_agg_years)
    assert ('kt',) == inv['Unit'].unique()
    assert non_agg_years == echart_years()[:datalen]
    for_sorting = []
    for catpath in catpaths:
        values = non_agg[non_agg['CategoryPathWithWhitespace'] == catpaths[catpath]]['CO2eq'].values
        if len(values) != datalen:
            raise RuntimeError('echart_series_all_Mt: Warning wrong length of data', catpath, values)
        if values.max() <= 0:
            # all negative
            # 1 / min makes it look right in eChart with big negative at the
            # bottom
            for_sorting.append((1 / values.min(), catpath, values))
        elif values.min() >= 0:
            # all positive
            for_sorting.append((values.max(), catpath, values))
        else:
            for_sorting.append((1 / values.min(),
                                catpath + ' (sink years)',
                                np.minimum(values, 0)))
            for_sorting.append((values.max(),
                                catpath + ' (source years)',
                                np.maximum(values, 0)))
    for amt, catpath, values in sorted(for_sorting):
        yield dict(
            name=catpath,
            catpath=catpath,
            list_raw_data=[float(vv) / 1000 for vv in values],
            type='line',
            lineStyle={'width': 2},
            select={'itemStyle': {'borderWidth': 20}}, # make invisible hit area wide
            stack='Total',
            areaStyle={},
            emphasis={'focus': 'series'},
            data=[{'value': float(vv) / 1000,
                   'url': f'/ipcc-sectors/{catpath}'.replace(' ', '_')}
                  for vv in values])


catpaths = {str(cpw).replace(' ', '_'): str(cpw) for cpw in non_agg['CategoryPathWithWhitespace'].values}


def net_emissions_total():
    can = inv[inv.Region.isin(['Canada', 'canada'])]
    total_without = can[can.Source == 'Total'].CO2eq.values / 1000
    lulucf = can[(can.Source == 'Land Use, Land-Use Change and Forestry')
                 & (can.Category.isna())].CO2eq.values / 1000
    rval = total_without + lulucf
    assert len(rval) == 2024 - 1990
    return rval


def net_emissions_total_without_LULUCF():
    can = inv[inv.Region.isin(['Canada', 'canada'])]
    rval = can[can.Source == 'Total'].CO2eq.values / 1000
    assert len(rval) == 2024 - 1990
    return rval


def CNZEAA_targets():
    net_emissions = net_emissions_total()
    baseline = net_emissions[2005 - 1990]
    return np.interp(
        echart_years(),
        [2005, 2030, 2035, 2050],
        [baseline, baseline * .6, baseline * .525, 0.0])
