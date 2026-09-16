"""
These operators are not methods so that this file doesn't need to be
imported by the webserver.
"""

import numpy as np

from .annual_emission_results import AnnualEmissionResults
from .annual_subsidy_results import (
        DetailedAnnualProgramBalances,
        NationalAnnualProgramBalances)
from .enums import GHG, PT, Activity, GovernmentProgram, IPCC_Sector


def aer_sectors(aer:AnnualEmissionResults) -> set[IPCC_Sector]:
    return {sector for (sector, _, _) in aer.ktCO2e_sample}


def programs(program_balances:DetailedAnnualProgramBalances) -> set[GovernmentProgram]:
    return {program for (program, _, _) in program_balances.CAD_sample}


def napb_from_dapb(
        dapb:DetailedAnnualProgramBalances
        ) -> NationalAnnualProgramBalances:
    tmp = {}
    # group-by program
    for (program, _, _), sample in dapb.CAD_sample.items():
        tmp.setdefault(program, []).append(sample)
    napb = NationalAnnualProgramBalances(
            CAD_sample={key: sum(vals) for key, vals in tmp.items()},
            years=dapb.years,
            n_samples=dapb.n_samples)
    napb.CAD_sample[GovernmentProgram.Net] = sum(napb.CAD_sample.values())
    return napb

