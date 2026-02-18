from .ipcc_canada import *
from . import enums

def test_iter_works():
    n = 0
    for foo in echart_series_all_Mt():
        n += 1
    assert n == len(enums.IPCC_Sector)
