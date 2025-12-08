
import pandas as pd
# from https://data-donnees.az.ec.gc.ca/data/substances/monitor/canada-s-official-greenhouse-gas-inventory/A-IPCC-Sector?lang=en

inv = pd.read_csv('EN_GHG_IPCC_Can_Prov_Terr.csv')

inv['CategoryPath'] = (
    inv['Category'].fillna('').astype(str) + '/' +
    inv['Sub-category'].fillna('').astype(str) + '/' +
    inv['Sub-sub-category'].fillna('').astype(str)
).str.rstrip('/')
assert not inv['CategoryPath'].isna().any()

inv['CO2eq'] = inv['CO2eq'].replace('x', 50_000).astype(float)
inv['Year'] = inv['Year'].astype(float)

# non-aggregate rows relating to the entire country (remove subtotals)
non_agg = inv[ (inv['Region'].isin(['Canada', 'canada'])) & (inv['Total'] != 'y')]

def echart_years():
    return [float(x) for x in sorted(non_agg['Year'].unique())]

def echart_series_Mt(catpath):
    datalen = len(echart_years())
    assert ('kt',) == inv['Unit'].unique()
    data = [float(x) / 1000 for x in non_agg[non_agg['CategoryPath'] == catpath]['CO2eq'].values]
    assert len(data) == datalen, (catpath, len(data), datalen)
    return  dict(
        name=catpath,
        type='line',
        stack='Total',
        areaStyle={},
        emphasis={'focus': 'series'},
        data=[{'value': datum, 'url': f'/ipcc-sectors/{catpath}'.replace(' ', '_')} for datum in data])

def echart_series_all_Mt(only_positive=False):
    datalen = len(echart_years())
    assert ('kt',) == inv['Unit'].unique()
    # assume all category paths have same most-recent year??
    keys = non_agg[non_agg['Year'] == non_agg['Year'].max()][['CO2eq', 'CategoryPath']]
    for amt, catpath in sorted([(amt, catpath) for amt, catpath in keys.values]):
        data = [float(x) / 1000 for x in non_agg[non_agg['CategoryPath'] == catpath]['CO2eq'].values]
        assert len(data) == datalen
        if all(x >= 0 for x in data) or not only_positive:
            yield dict(
                name=catpath,
                catpath=catpath,
                list_raw_data=data,
                type='line',
                stack='Total',
                areaStyle={},
                emphasis={'focus': 'series'},
                data=[{'value': datum, 'url': f'/ipcc-sectors/{catpath}'.replace(' ', '_')} for datum in data])


catpaths = set([cp.replace(' ', '_') for cp in non_agg['CategoryPath'].values])
