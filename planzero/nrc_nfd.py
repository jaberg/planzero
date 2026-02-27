"""
Natural Resources Canada (NRC) - National Forerst Database (NFD)

# NFD * files downloaded from
http://nfdp.ccfm.org/en/download.php

"""
import functools
import math, enum
import pandas as pd

from . import ureg as u, enums
from . import objtensor, sts
from .ureg import m3_by_roundwood_species_group


@functools.cache
def net_merchantable_volume_harvested():
    rval = objtensor.empty(
        enums.RoundwoodSpeciesGroup,
        enums.RoundwoodProductCategory,
        enums.RoundwoodTenure,
        enums.PT)

    nfd_roundwood_harvested = pd.read_csv(
        'data/NFD - Net Merchantable Volume of Roundwood Harvested by Category and Ownership - EN FR.csv',
        encoding='latin_1')

    tv_by_key = {}
    min_year = 2000
    max_year = 2000

    for row in nfd_roundwood_harvested.iloc:
        year = row.Year
        min_year = min(year, min_year)
        max_year = max(year, max_year)
        species_group = enums.RoundwoodSpeciesGroup(row['Species group'])
        product_category = enums.RoundwoodProductCategory(row['Category'])
        tenure = enums.RoundwoodTenure(row['Tenure (En)'])
        volume_m3 = row['Volume (cubic metres) (En)']
        jurisdiction = enums.PT(row.Jurisdiction)
        if math.isnan(volume_m3) or volume_m3 < 10:
            # There are lots of NaNs... I hope it's okay to interpret them as zero!
            # There's a row about hardwood in Manitoba in 2007 where it looks like a duplicate with a tiny value of 1.0 m3
            continue
        else:
            key = (species_group, product_category, tenure, jurisdiction)
            u_year = year * u.years
            u_val = volume_m3 * m3_by_roundwood_species_group[key[0]]
            tv_by_key.setdefault(key, {}).setdefault(u_year, u_val)
            assert tv_by_key[key][u_year] == u_val, (key, u_year, u_val, tv_by_key[key][u_year])

    # fill in missing values as 0 in empty slots
    for species_group in enums.RoundwoodSpeciesGroup:
        rval[species_group] = 0 * m3_by_roundwood_species_group[species_group]
    # ... and in empty years
    assert min_year < 2000
    assert max_year > 2000
    for (sg, _, _, _), val_by_year in tv_by_key.items():
        for year in range(min_year, max_year):
            val_by_year.setdefault(year * u.years,
                                   0 * m3_by_roundwood_species_group[sg])

    for key, tv in tv_by_key.items():
        times, values = zip(*sorted(tv.items())) # chronological
        if times:
            try:
                rval[*key] = sts.annual_report(times=times, values=values)
            except Exception as exc:
                raise RuntimeError(key, times, values) from exc

    return rval
