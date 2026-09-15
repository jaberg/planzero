"""
These operators are not methods so that this file doesn't need to be
imported by the webserver.
"""

from .annual_emission_results import AnnualEmissionResults
from .enums import GHG, PT, Activity, IPCC_Sector


def aer_sectors(aer:AnnualEmissionResults) -> set[IPCC_Sector]:
    return {sector for (sector, _, _) in aer.ktCO2e_sample}
