from . import ghgrp

import numpy as np
import array

def test_facility_ids():
    for foo in ghgrp.GHGRP_IDs():
        pass


def test_facility_by_NAICS():
    for foo in ghgrp.facilities_by_NAICS():
        pass


def test_NAICS_source_emissions():
    ghgrp.GHG_NAICS_source_emissions(ghgrp.ESKey.TotalEmissionsFromSource_tCO2e)


def test_NAICS_emissions():
    ghgrp.NAICS_emissions(ghgrp.EKey.TotalEmissions_tCO2e)


def test_perfect_double_counting():
    nse = ghgrp.GHG_NAICS_source_emissions(nan_value_as_zero=True)
    ne = ghgrp.NAICS_emissions(ghgrp.EKey.TotalEmissions_tCO2e)

    nse_sum = (ghgrp.GWP_100[:, None, None] * nse).sum()
    ne_sum = ne.sum()

    # if this test fails, update the comments on NAICS_source_emissions and NAICS_emissions
    # and consider renaming those functions as their semantics will have changed :/
    assert  nse_sum.times[-2:] == ne_sum.times[-2:] == array.array('d', [2022, 2023])
    assert np.allclose(nse_sum.values[-2:], ne_sum.values[-2:], atol=2) # even 1 works as of 2023, but seems excessively strict


def test_source_emissions_backfill_proportions():
    foo = ghgrp.source_emissions_backfill_proportions()
    assert isinstance(foo, dict)


def test_GHG_NAICS_source_emissions_backfilled():
    foo = ghgrp.GHG_NAICS_source_emissions_backfilled()
