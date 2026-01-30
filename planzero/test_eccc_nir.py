import pytest

from .eccc_nir_annex6 import *
from .eccc_nir import *


def test_sts_a6_1_1():
    NaturalGasFactorsCO2.sts_a6_1_1()

def test_sts_a6_1_2():
    NaturalGasFactorsCO2.sts_a6_1_2()

def test_ngf_emissions_from_electricity_generation():
    NaturalGasFactorsCO2.provincial_utilities_emissions_from_electricity_generation_2005_to_2013()
    NaturalGasFactorsCO2.provincial_industries_emissions_from_electricity_generation_2005_to_2013()
