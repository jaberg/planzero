import matplotlib.pyplot as plt

from .enums import CoalType
from .ghgvalues import GWP_100

from .ureg import u, kg_by_ghg, kilotonne_by_coal_type
from . import ureg
from . import sc_np
from . import objtensor
from . import sts
from . import annex6_np
from . import eccc_nir_annex13
from . import enums
from . import ptvalues


class EstAnnex13ElectricityFromCoal(object):
    """Return CO2e by PT"""
    def __init__(self):

        self.prov_consumption, self.national_consumption = \
                sc_np.Archived_Electric_Power_Generation_Annual_Fuel_Consumed_by_Electrical_Utility()
        self.emission_factors = annex6_np.A6_1_10_and_12()
        self.emissions = (self.emission_factors * self.prov_consumption[CoalType]).sum(enums.CoalType)
        self.co2e = GWP_100 @ self.emissions

    def plot_consumption(self):
        ptvalues.scatter_subplots(
            self.prov_consumption[enums.CoalType],
            v_unit_by_outer_key=kilotonne_by_coal_type,
            legend_loc='upper right')

    def plot_emissions(self):
        GHG = enums.GHG
        ptvalues.scatter_subplots(
            self.emissions[[GHG.CO2, GHG.CH4, GHG.N2O]],
            v_unit_by_outer_key={
                GHG.CO2: u.kilotonne_CO2,
                GHG.CH4: u.tonne_CH4,
                GHG.N2O: u.tonne_N2O,
            },
            legend_loc='upper right')

    def plot_vs_annex13_target(self):
        plt.figure()
        est = self.co2e.sum(enums.PT).to(u.kilotonne_CO2e)
        est.plot(label='Estimate')
        a13_ng = eccc_nir_annex13.national_electricity_CO2e_from_combustion()['coal']
        a13_ng.to(est.v_unit).plot(label='Annex13 (Target)')
        plt.title('Emissions: Electricity from Coal')
        plt.legend(loc='upper right')
        plt.xlim(2004, max(max(est.times), max(a13_ng.times)) + 1)


class EstAnnex13ElectricityFromNaturalGas(object):
    def __init__(self):

        # https://en.wikipedia.org/wiki/Natural_gas#Energy_content,_statistics,_and_pricing
        # Wikipedia says 39
        # also Gemini says there is a standard value used by Statistics Canada and National Energy Board
        # of 38.32, but no citation
        Natural_Gas_gross_heating_value_marketable = 38.32 * u.MJ / u.m3_NG_mk

        # Gemini says 40 to 45, no citation given
        # but could be corroborated by averaging the various typical gases? What about moisture?
        Natural_Gas_gross_heating_value_nonmarketable = 42 * u.MJ / u.m3_NG_nmk

        # This is a guess
        Natural_Gas_Peaker_thermal_efficiency = .30

        EP = enums.ElectricityProducer
        EGT = enums.ElectricityGenerationTech
        PT = enums.PT
        GHG = enums.GHG
        NGU = enums.NaturalGasUser

        thermal_efficiency = objtensor.empty(EP, enums.PT)
        thermal_efficiency[:] = Natural_Gas_Peaker_thermal_efficiency
        thermal_efficiency[EP.Utilities, PT.ON] = .5 # more use of combined cycle
        thermal_efficiency[EP.Utilities] *= Natural_Gas_gross_heating_value_marketable
        thermal_efficiency[EP.Industry] *= Natural_Gas_gross_heating_value_nonmarketable

        prov_power_gen, _ = sc_np.Electric_Power_Annual_Generation_by_Class_of_Producer()
        prov_ng_vol = prov_power_gen[:, EGT.CombustionTurbine] / thermal_efficiency

        emission_factors = objtensor.empty(GHG, EP, PT)
        T34 = annex6_np.A6_1_3_and_1_4()
        emission_factors[GHG.CO2, EP.Utilities] = annex6_np.A6_1_1()
        emission_factors[GHG.CO2, EP.Industry] = annex6_np.A6_1_2()
        emission_factors[[GHG.CH4, GHG.N2O], EP.Utilities] = T34[:, NGU.ElectricUtilities]
        emission_factors[[GHG.CH4, GHG.N2O], EP.Industry] = T34[:, NGU.Producer]
        for ghg in [GHG.HFCs, GHG.PFCs, GHG.SF6, GHG.NF3]:
            emission_factors[ghg, EP.Utilities] = 0 * kg_by_ghg[ghg] / u.m3_NG_mk
            emission_factors[ghg, EP.Industry] = 0 * kg_by_ghg[ghg] / u.m3_NG_nmk

        self.thermal_efficiency = thermal_efficiency
        self.prov_power_gen = prov_power_gen
        self.prov_ng_vol = prov_ng_vol
        self.emission_factors = emission_factors
        self.emissions_by_ep = (emission_factors * prov_ng_vol)
        self.emissions = self.emissions_by_ep.sum(EP)
        self.co2e = GWP_100 @ self.emissions


    def plot_provincial_volumes(self):
        fig, axs = ptvalues.scatter_subplots(
            self.prov_ng_vol,
            v_unit_by_outer_key={
                enums.ElectricityProducer.Utilities: u.mega_m3_NG_mk,
                enums.ElectricityProducer.Industry: u.mega_m3_NG_nmk},
            legend_loc='lower right')
        fig.set_size_inches(12, 5)

    def plot_emissions_by_utilities(self):
        GHG = enums.GHG
        EP = enums.ElectricityProducer
        fig, axs = ptvalues.scatter_subplots(
            self.emissions_by_ep[[GHG.CO2, GHG.CH4, GHG.N2O], EP.Utilities],
            v_unit_by_outer_key={
                GHG.CO2: u.kilotonne_CO2,
                GHG.CH4: u.tonne_CH4,
                GHG.N2O: u.tonne_N2O,
            },
            legend_loc='lower right')
        fig.set_size_inches(12, 5)
        axs[0].set_xlim(2004, 2014)

    def plot_emissions_by_natural_gas_producers(self):
        GHG = enums.GHG
        EP = enums.ElectricityProducer
        fig, axs = ptvalues.scatter_subplots(
            self.emissions_by_ep[[GHG.CO2, GHG.CH4, GHG.N2O], EP.Industry],
            v_unit_by_outer_key={
                GHG.CO2: u.kilotonne_CO2,
                GHG.CH4: u.tonne_CH4,
                GHG.N2O: u.tonne_N2O,
            },
            legend_loc='lower right')
        fig.set_size_inches(12, 5)
        axs[0].set_xlim(2004, 2014)
        axs[1].set_xlim(2004, 2014)


    def plot_vs_annex13_target(self):
        plt.figure()
        est = self.co2e.sum(enums.PT).to(u.kilotonne_CO2e)
        est.plot(label='Estimate')
        a13_ng = eccc_nir_annex13.national_electricity_CO2e_from_combustion()['natural_gas']
        a13_ng.to(est.v_unit).plot(label='Annex13 (Target)')
        plt.title('Emissions: Electricity from Natural Gas')
        plt.legend(loc='upper left')
        plt.xlim(2004, max(max(est.times), max(a13_ng.times)) + 1)


class EstAnnex13ElectricityFromOther(object):

    """
    Estimate the emissions associated with "Other Fuels" in Annex 13,
    which I take to be:

    * Diesel
    * Light fuel oil
    * Heavy fuel oil
    """

    def __init__(self):
        self.prov_consumption, self.national_consumption = \
                sc_np.Archived_Electric_Power_Generation_Annual_Fuel_Consumed_by_Electrical_Utility()

        self.emission_factors_lhk = annex6_np.A6_1_6_LFO_HFO_Kerosene()
        self.emission_factors_dg = annex6_np.A6_1_6_Diesel_and_Gasoline()

        FT = enums.FuelType
        RPP_User = enums.RPP_User

        # afaik the only RPP_User producing electricity is ElectricUtilities
        # GHG x LightAndHeavy x PT
        LightAndHeavyOil = [FT.LightFuelOil, FT.HeavyFuelOil]
        self.emissions_LH_pt = (
            self.emission_factors_lhk[:, LightAndHeavyOil, RPP_User.ElectricUtilities, None]
            * self.prov_consumption[LightAndHeavyOil])

        # GHG x PT
        self.emissions_diesel_pt = (
            self.emission_factors_dg[:, FT.Diesel, None]
            * self.prov_consumption[FT.Diesel])

        self.emissions = (
            self.emissions_LH_pt.sum(LightAndHeavyOil)
            + self.emissions_diesel_pt)

        self.co2e = GWP_100 @ self.emissions

    def plot_consumption(self):
        FT = enums.FuelType
        ptvalues.scatter_subplots(
            self.prov_consumption[
                [FT.LightFuelOil, FT.HeavyFuelOil, FT.Diesel]],
            #v_unit_by_outer_key=kilotonne_by_coal_type,
            legend_loc='upper right')

    def plot_emissions(self):
        GHG = enums.GHG
        ptvalues.scatter_subplots(
            self.emissions[[GHG.CO2, GHG.CH4, GHG.N2O]],
            v_unit_by_outer_key={
                GHG.CO2: u.kilotonne_CO2,
                GHG.CH4: u.tonne_CH4,
                GHG.N2O: u.tonne_N2O,
            },
            legend_loc='upper right')


    def plot_vs_annex13_target(self):
        estimate = self.co2e.sum(enums.PT)

        a13_ng = eccc_nir_annex13.national_electricity_CO2e_from_combustion()['other_fuels']
        target = a13_ng.to(estimate.v_unit)

        plt.figure()
        estimate.plot(label='Estimate')
        target.plot(label='Annex13 (Target)')
        plt.title('Emissions: Electricity from Other Fuels')
        plt.legend(loc='upper left')
        plt.xlim(2004, max(max(estimate.times), max(target.times)) + 1)
