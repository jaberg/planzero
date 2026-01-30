import pytest

from .eccc_nir_annex6 import *
from .eccc_nir import *


def test_ngf_emissions_from_electricity_generation():
    plot_delta_natural_gas_for_electricity_generation()
