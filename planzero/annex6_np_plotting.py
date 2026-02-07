"""
This file is meant to be imported by ECCC-NIR.ipynb
"""
from . import enums
from . import annex6_np, ptvalues

def plot_A6_1_1():
    ptvalues.scatter(
        annex6_np.A6_1_1(),
        title='Marketable Natural Gas CO2 Emission Factors',
        legend_loc='upper right',
        ignore_vals_above=1e9)


def plot_A6_1_2():
    ptvalues.scatter(
        annex6_np.A6_1_2(),
        title='Non-Marketable Natural Gas CO2 Emission Factors',
        legend_loc='lower left',
        ignore_vals_above=1e9)


def plot_A6_1_3_and_1_4_producer_consumption_CH4():
    ot = annex6_np.A6_1_3_and_1_4()[enums.GHG.CH4, enums.NaturalGasUser.Producer]
    fig, ax = ptvalues.scatter(
        ot,
        title='Non-Marketable Natural Gas CH4 Emission Factors',
        legend_loc='lower left')
    eps = .1
    ax.text(2007, ot[enums.PT.NL].magnitude + eps, 'Newfoundland and Labrador')
    ax.text(2007, ot[enums.PT.PE].magnitude + eps, 'Other provinces and territories')
