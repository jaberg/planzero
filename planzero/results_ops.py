"""
These operators are not methods so that this file doesn't need to be
imported by the webserver.
"""

import numpy as np

from .annual_emission_results import AnnualEmissionResults, aer_total_with_LULUCF
from .annual_subsidy_results import (
    DetailedAnnualProgramBalances,
    NationalAnnualProgramBalances,
)
from .enums import GovernmentProgram, IPCC_Sector


def programs(program_balances:DetailedAnnualProgramBalances) -> set[GovernmentProgram]:
    return {program for (program, _, _) in program_balances.CAD_sample}


def napb_refresh_net(
        napb:NationalAnnualProgramBalances
        ) -> NationalAnnualProgramBalances:
    tmp = dict(napb.CAD_sample)
    if GovernmentProgram.Net in tmp:
        del tmp[GovernmentProgram.Net]
    tmp[GovernmentProgram.Net] = sum(tmp.values())
    rval = NationalAnnualProgramBalances(
            CAD_sample=tmp,
            years=napb.years,
            n_samples=napb.n_samples)
    return rval


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
    rval = napb_refresh_net(napb)
    return rval


def napb_scale(
        napb:NationalAnnualProgramBalances,
        amt:float
        ) -> NationalAnnualProgramBalances:
    rval = NationalAnnualProgramBalances(
            CAD_sample={key: val * amt for key, val in napb.CAD_sample.items()},
            years=napb.years,
            n_samples=napb.n_samples)
    return rval


def cost_per_tCO2e(
        napb:NationalAnnualProgramBalances,
        aer:AnnualEmissionResults,
        ) -> np.ndarray: # (n_samples,)
    assert napb.n_samples == aer.n_samples
    total_cost = napb.CAD_sample[GovernmentProgram.Net].sum(axis=1)
    total_tCO2e = aer_total_with_LULUCF(aer).sum(axis=0) * 1000 # b/c aer is ktCO2e
    return total_cost / total_tCO2e
