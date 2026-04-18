import array
import functools

import matplotlib.pyplot as plt
import numpy as np

from .est_nir_util import (
    _rstrip_data,
    _echart_years,
    _echart_reference_NIR_values,
    )
from .ghgvalues import GWP_100, GHG_zero_kg
from .ureg import (u, kg_by_ghg,
                   kilotonne_by_coal_type,
                   kt_by_ghg)
from . import aer
from . import ghgrp
from . import naics
from . import ureg
from . import sc_np
from . import objtensor
from . import sts
from . import annex6_np
from . import eccc_nir
from . import eccc_nir_annex6
from . import eccc_nir_annex9
from . import eccc_nir_annex13
from . import enums
from . import ptvalues
from . import ipcc_canada
from . import sc_2510003001 # supply and demand of primary and secondary energy
from . import sc_25_10_0084_01 # electric power generation fuel consumed cost of fuel

from .html import (
    EChartTitle,
    EChartXAxis,
    EChartYAxis,
    EChartSeriesStackElem,
    EChartSeriesBase,
    EChartLineStyle,
    EChartItemStyle,
    StackedAreaEChart)

GHG = enums.GHG
PT = enums.PT
IPCC = enums.IPCC_Sector
CoalType = enums.CoalType
est_nir_years = np.arange(1995, 2025) * u.years


def GHG_PT_zeros():
    rval = objtensor.empty(GHG, PT)
    rval[:] = GHG_zero_kg()[:, None]
    return rval


class EstAnnex13ElectricityFromCoal(object):
    """Return CO2e by PT"""
    def __init__(self):

        self.prov_consumption_a, self.national_consumption_a = \
                sc_np.Archived_Electric_Power_Generation_Annual_Fuel_Consumed_by_Electrical_Utility()
        self.prov_consumption_b, self.national_consumption_b = \
                sc_2510003001.supply_and_demand_of_primary_and_secondary_energy()

        # CoalTypes x PT
        self.coaltypes_until_2022 = self.prov_consumption_a[CoalType]

        # PT
        SDC = sc_2510003001.Supply_And_Demand_Characteristics
        support_years = functools.partial(sts.with_default_zero, times=est_nir_years)
        self.total_coal_transformed_by_utilities = self.prov_consumption_b[
            sc_2510003001.Fuel_Type.Total_Coal,
            SDC.Transformed_to_Electricity_by_Utilities].apply(support_years)

        self.prov_consumption = self.coaltypes_until_2022.apply(support_years)
        # XXX guessing a coaltype with which to associate the
        # aggregate data Im' using for years after 2022
        # smarter: use the historically-most-used type for each province?
        # keep the same proportions? find the data?
        self.prov_consumption[CoalType.CanadianBituminous] \
            += (self.total_coal_transformed_by_utilities
                * sts.STS.zero_one(2021.5 * u.years,
                                   v_unit=u.kg_coal_bit / u.kg_coal))

        # GHG x CoalType x PT
        self.emission_factors = annex6_np.A6_1_10_and_12()

        # GHG x PT
        self.emissions = (
            self.emission_factors * self.prov_consumption[CoalType]).sum(enums.CoalType)

        # PT
        def to_CO2e(x):
            if all(vv == 0 for vv in x.values):
                return 0 * u.kt_CO2e
            else:
                return x.to(u.kt_CO2e)

        self.co2e = (GWP_100 @ self.emissions).apply(to_CO2e)

    def plot_consumption(self):
        ptvalues.scatter_subplots(
            self.prov_consumption[enums.CoalType],
            v_unit_by_outer_key=kilotonne_by_coal_type,
            legend_loc='upper right')

    def plot_emissions(self):
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
        NGU = enums.NaturalGasUser

        thermal_efficiency = objtensor.empty(EP, enums.PT)
        thermal_efficiency[:] = Natural_Gas_Peaker_thermal_efficiency
        thermal_efficiency[EP.Utilities, PT.ON] = .5 # more use of combined cycle
        thermal_efficiency[EP.Utilities] *= Natural_Gas_gross_heating_value_marketable
        thermal_efficiency[EP.Industry] *= Natural_Gas_gross_heating_value_nonmarketable

        prov_power_gen, _ = sc_np.Electric_Power_Annual_Generation_by_Class_of_Producer()

        # prov_ng_vol_2005_to_2013 is indexed by class of producer and PT
        # and it only covers years 2005-2013
        prov_ng_vol_2005_to_2013 = prov_power_gen[:, EGT.CombustionTurbine] / thermal_efficiency

        # this other disposition table covers years prior to 2005 and after 2013
        SDC = sc_2510003001.Supply_And_Demand_Characteristics
        supply_and_demand_pt, supply_and_demand_ca \
                = sc_2510003001.supply_and_demand_of_primary_and_secondary_energy()
        prov_ng_vol_pre2005_post2013 = supply_and_demand_pt[sc_2510003001.Fuel_Type.Natural_Gas]

        # usum is used to union the years covered
        support_years = functools.partial(sts.with_default_zero, times=est_nir_years)
        #foo = sts.annual_report(times=[2000 * u.years], values=[1 * u.kg])
        #bar = support_years(foo)
        prov_ng_vol = objtensor.empty(EP, enums.PT)
        prov_ng_vol[EP.Utilities] = (
            prov_ng_vol_2005_to_2013[EP.Utilities].apply(support_years)
            + prov_ng_vol_pre2005_post2013[SDC.Transformed_to_Electricity_by_Utilities].apply(support_years))
        prov_ng_vol[EP.Industry] = (
            prov_ng_vol_2005_to_2013[EP.Industry].apply(support_years)
            + prov_ng_vol_pre2005_post2013[SDC.Transformed_to_Electricity_by_Industry].apply(support_years))

        emission_factors = objtensor.empty(GHG, EP, PT)
        T34 = annex6_np.A6_1_3_and_1_4()
        emission_factors[GHG.CO2, EP.Utilities] = annex6_np.A6_1_1()
        emission_factors[GHG.CO2, EP.Industry] = annex6_np.A6_1_2()
        # I thought that industry consumption of natural gas for electricity generation
        # was mostly by producers, using non-marketable natural gas. That assumption
        # is at least somewhat false, in that Annex 6 Table 1-2 suggests that Quebec
        # did not use non-marketable natural gas, but did generate electricity by industry.
        # I am just patching up the emission factors here, but this feels like a symptom,
        # and perhaps there are significantly many other industrial electricity generators
        # burning marketable natural gas to do so.
        emission_factors[GHG.CO2, EP.Industry, PT.QC] \
                = emission_factors[GHG.CO2, EP.Utilities, PT.QC] * (1 * u.m3_NG_mk / u.m3_NG_nmk)

        emission_factors[[GHG.CH4, GHG.N2O], EP.Utilities] = T34[:, NGU.ElectricUtilities]
        emission_factors[[GHG.CH4, GHG.N2O], EP.Industry] = T34[:, NGU.Producer] # see comment above
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
        assert self.emissions_by_ep.ndim == 3
        self.co2e_by_ep = (GWP_100[:, None, None] * self.emissions_by_ep).sum(0)

    def plot_provincial_volumes(self):
        fig, axs = ptvalues.scatter_subplots(
            self.prov_ng_vol,
            v_unit_by_outer_key={
                enums.ElectricityProducer.Utilities: u.mega_m3_NG_mk,
                enums.ElectricityProducer.Industry: u.mega_m3_NG_nmk},
            legend_loc='lower right')
        fig.set_size_inches(12, 5)

    def plot_emissions_by_utilities(self, as_co2e=False):
        GHG = enums.GHG
        EP = enums.ElectricityProducer
        if as_co2e:
            ot = GWP_100[:, None, None] * self.emissions_by_ep
            v_unit_by_outer_key = {
                GHG.CO2: u.kt_CO2e,
                GHG.CH4: u.kt_CO2e,
                GHG.N2O: u.kt_CO2e,
            }
        else:
            ot = self.emissions_by_ep
            v_unit_by_outer_key = {
                GHG.CO2: u.kilotonne_CO2,
                GHG.CH4: u.tonne_CH4,
                GHG.N2O: u.tonne_N2O,
            }
        fig, axs = ptvalues.scatter_subplots(
            ot[[GHG.CO2, GHG.CH4, GHG.N2O], EP.Utilities],
            v_unit_by_outer_key=v_unit_by_outer_key,
            legend_loc='upper left')
        fig.set_size_inches(12, 5)
        for ax in axs:
            ax.set_xlim(2000, 2025)

    def plot_emissions_by_natural_gas_producers(self, as_co2e=False):
        GHG = enums.GHG
        EP = enums.ElectricityProducer
        if as_co2e:
            ot = GWP_100[:, None, None] * self.emissions_by_ep
            v_unit_by_outer_key = {
                GHG.CO2: u.kt_CO2e,
                GHG.CH4: u.kt_CO2e,
                GHG.N2O: u.kt_CO2e,
            }
        else:
            ot = self.emissions_by_ep
            v_unit_by_outer_key = {
                GHG.CO2: u.kilotonne_CO2,
                GHG.CH4: u.tonne_CH4,
                GHG.N2O: u.tonne_N2O,
            }
        fig, axs = ptvalues.scatter_subplots(
            ot[[GHG.CO2, GHG.CH4, GHG.N2O], EP.Industry],
            v_unit_by_outer_key=v_unit_by_outer_key,
            legend_loc='upper left')
        fig.set_size_inches(12, 5)
        for ax in axs:
            ax.set_xlim(2000, 2025)

    def plot_vs_annex13_target(self):
        EP = enums.ElectricityProducer
        plt.figure()

        est = self.co2e.sum(enums.PT).to(u.kilotonne_CO2e)
        est.plot(label='Estimate')
        a13_ng = eccc_nir_annex13.national_electricity_CO2e_from_combustion()['natural_gas']
        a13_ng.to(est.v_unit).plot(label='Annex13 (Target)')
        plt.title('Emissions: Electricity from Natural Gas')

        est_util = self.co2e_by_ep[EP.Utilities].sum(enums.PT).to(u.kilotonne_CO2e)
        est_util.plot(label='Est (Utilities)')

        est_ind = self.co2e_by_ep[EP.Industry].sum(enums.PT).to(u.kilotonne_CO2e)
        est_ind.plot(label='Est (Industry)')

        plt.xlim(2004, max(max(est.times), max(a13_ng.times)) + 1)
        plt.legend(loc='upper left')


class EstAnnex13ElectricityFromOther(object):

    """
    Estimate the emissions associated with "Other Fuels" in Annex 13,
    which I take to be (because of SC-25-10-0017-1):
    TODO: go through the remaining fuel types in table 25-10-0084-01.

    * Diesel
    * Light fuel oil
    * Heavy fuel oil
    * Petroleum coke
    * Wood
    * Other Solid Fuels
    * Methane
    * Other Gaseous Fuels
    * Propane (so small can be ignored)
    """

    def init_LightHeavy(self):
        FT = enums.FuelType
        FT2 = sc_25_10_0084_01.FuelType
        RPP_User = enums.RPP_User
        self.LightAndHeavyOil = [FT.LightFuelOil, FT.HeavyFuelOil]

        self.emission_factors_lhk = annex6_np.A6_1_6_LFO_HFO_Kerosene()

        self.consumption_LFO = (
            self.cutover * self.prov_consumption[FT.LightFuelOil].apply(self.support_years)
            + self.fuel_consumed_pt[FT2.LightFuelOil].apply(self.support_years))
        self.consumption_HFO = (
            self.cutover * self.prov_consumption[FT.HeavyFuelOil].apply(self.support_years)
            + self.fuel_consumed_pt[FT2.TotalHeavyFuelOil].apply(self.support_years))

        # afaik the only RPP_User producing electricity is ElectricUtilities
        # GHG x LightAndHeavy x PT
        self.emissions_LFO_pt = (
            self.emission_factors_lhk[:, FT.LightFuelOil, RPP_User.ElectricUtilities, None]
            * self.consumption_LFO)
        self.emissions_HFO_pt = (
            self.emission_factors_lhk[:, FT.HeavyFuelOil, RPP_User.ElectricUtilities, None]
            * self.consumption_HFO)

    def init_Diesel(self):
        FT = enums.FuelType
        FT2 = sc_25_10_0084_01.FuelType

        self.emission_factors_dg = annex6_np.A6_1_6_Diesel_and_Gasoline()
        # GHG x PT
        self.emissions_diesel_pt = (
            self.emission_factors_dg[:, FT.Diesel, None]
            * (self.cutover * self.prov_consumption[FT.Diesel].apply(self.support_years)
               + self.fuel_consumed_pt[FT2.Diesel].apply(self.support_years)))

    def init_PetCoke(self):
        FT = enums.FuelType
        FT2 = sc_25_10_0084_01.FuelType
        RPP_User = enums.RPP_User
        self.emission_factors_ps = annex6_np.A6_1_7_and_1_8_and_1_9()

        # It seems odd, but ECCC-NIR Annex 6 clearly describes Petroleum Coke
        # in terms of volume, rather than mass.

        # this is Google Gemini's estimate of how to convert between
        # mass and volume for petcoke, for the purpose of e.g. shipping
        petcoke_bulk_density = 0.85 * u.kg_petcoke / u.l_petcoke

        # and for some more theoretical equivalence of solid pieces,
        # which makes the estimation of the Annex13 totals come out better
        petcoke_material_density = 1.30 * u.kg_petcoke / u.l_petcoke

        self.emissions_petcoke_pt = (
            self.emission_factors_ps[:, FT.PetCoke, RPP_User.RefineriesAndOthers, None]
            * ((self.cutover * self.prov_consumption[FT.PetCoke].apply(self.support_years)
                + self.fuel_consumed_pt[FT2.PetCoke].apply(self.support_years))
               / petcoke_material_density)
        )

    def init_Wood(self):
        # According to note (a) in A6.6-1, CO2 from burning wood isn't
        # counted in the NIR, although CH4 and N2O is counted.
        # (It is assumed that this carbon is already in circulation)
        #
        # The note on the table reads: All greenhouse gas (GHG) emissions,
        # including CO2 emissions from biomass burned in managed forests
        # (wildfires and controlled burning), are reported under Land-Use,
        # Land-use Change and Forestry (LULUCF) and excluded from national
        # inventory totals.
        #
        # My interpretation is that if, hypothetically, wildfires and wood
        # harvest for fuel were to consume all of Canada's forests, then
        # it still wouldn't count toward CO2 emissions in the NIR. I'm
        # confused why that would be the case, but here in the
        # non-hypothetical world, that is not in fact happening, so
        # it's a moot point.

        self.emissions_wood_pt = objtensor.empty(GHG, enums.PT)
        for ghg in [GHG.CO2, GHG.HFCs, GHG.PFCs, GHG.SF6, GHG.NF3]:
            self.emissions_wood_pt[ghg] = 0 * kg_by_ghg[ghg]

        FT = enums.FuelType
        FT2 = sc_25_10_0084_01.FuelType
        self.consumption_wood = (
            self.cutover * self.prov_consumption[FT.Wood].apply(self.support_years)
            + (self.fuel_consumed_pt[FT2.Wood].apply(self.support_years)
               * (.5 * u.kg_wood_mc25 / u.kg_wood_mc50)))

        self.emissions_wood_pt[GHG.CH4] = (
            self.consumption_wood
            * (0.1 * u.g_CH4 / u.kg_wood_mc25))
        self.emissions_wood_pt[GHG.N2O] = (
            self.consumption_wood
            * (0.07 * u.g_N2O / u.kg_wood_mc25))

    def init_OtherSolidFuels(self):
        # I'm including this category because it is included
        # by Stats-Can table 17.
        #
        # For for emission factors, the closest thing I could find was
        # Annex6 Table 7.2 about municipal waste incineration,
        # but there it said that the emission factor should only be
        # applied to fossil carbon waste. I intuitively thought
        # "who throws out fossil carbon?" and just moved on.
        pass

    def init_Methane(self):
        FT = enums.FuelType
        self.emissions_methane_pt = objtensor.empty(GHG, enums.PT)
        for ghg in [GHG.HFCs, GHG.PFCs, GHG.SF6, GHG.NF3]:
            self.emissions_methane_pt[ghg] = 0 * kg_by_ghg[ghg]

        # why is there a separate category for methane, as
        # distinct from natural gas?

        # I'm just eyeballing the year-by-year tables in Annex 6
        # relating to marketable natural gas:

        self.emissions_methane_pt[GHG.CO2] = self.prov_consumption[FT.Methane].apply(self.support_years) \
                    * (1900 * u.g_CO2 / u.m3_methane)
        self.emissions_methane_pt[GHG.CH4] = self.prov_consumption[FT.Methane].apply(self.support_years) \
                    * (.49 * u.g_CH4 / u.m3_methane)
        self.emissions_methane_pt[GHG.N2O] = self.prov_consumption[FT.Methane].apply(self.support_years) \
                    * (.049 * u.g_N2O / u.m3_methane)

    def init_OtherGaseousFuels(self):
        # The comment on this row of the StatsCan table
        # mentions "refinery fuel gas" which is also sometimes
        # called "still gas" in e.g. the ECCC NIR Annex6,
        # so I'll use those emission factors for the whole "Other Gaseous
        # Fuels" category.
        FT = enums.FuelType
        FT2 = sc_25_10_0084_01.FuelType
        RPP_User = enums.RPP_User

        # TODO: confirm that UpgradingFacilities use PetCoke
        # for heat (to create more RPPs and PetCoke), not electricity
        self.emissions_stillgas_pt = (
            self.cutover * self.emission_factors_ps[:, FT.StillGas, RPP_User.RefineriesAndOthers, None]
            * (self.prov_consumption[FT.StillGas].apply(self.support_years)
               + (self.fuel_consumed_pt[FT2.OtherGaseous].apply(self.support_years) * (1 * u.m3_stillgas / u.m3))))

    def __init__(self):
        NAICS = enums.NAICS
        self.prov_consumption, self.national_consumption = \
                sc_np.Archived_Electric_Power_Generation_Annual_Fuel_Consumed_by_Electrical_Utility()

        # what's the relationship between this table and the one above, I'm not sure
        self.epgfc_pt, self.epgfc_ca = \
                sc_25_10_0084_01.electric_power_generation_fuel_consumed_cost_of_fuel()
        self.fuel_consumed_pt = self.epgfc_pt[
            sc_25_10_0084_01.MetaFuelType.Fuel_Consumed, :, NAICS.Electricity_Producers__Utilities]

        self.support_years = functools.partial(sts.with_default_zero, times=est_nir_years)
        self.cutover = sts.STS.one_zero(2019.5 * u.years)

        self.init_LightHeavy()
        self.init_Diesel()
        self.init_PetCoke()
        self.init_Wood()
        self.init_OtherSolidFuels()
        self.init_Methane()
        self.init_OtherGaseousFuels()

        self.emissions = (
            self.emissions_LFO_pt
            + self.emissions_HFO_pt
            + self.emissions_diesel_pt
            + self.emissions_petcoke_pt
            + self.emissions_wood_pt
            + self.emissions_methane_pt
            + self.emissions_stillgas_pt
        )

        self.co2e = GWP_100 @ self.emissions

    def plot_consumption(self):
        FT = enums.FuelType
        ptvalues.scatter(
            self.consumption_wood,
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
            legend_loc='upper left')

    def plot_vs_annex13_target(self):
        estimate = self.co2e.sum(enums.PT).to(u.kilotonne_CO2e)

        a13_ng = eccc_nir_annex13.national_electricity_CO2e_from_combustion()['other_fuels']
        target = a13_ng.to(estimate.v_unit)

        plt.figure()
        estimate.plot(label='Estimate')
        target.plot(label='Annex13 (Target)')
        plt.title('Emissions: Electricity from Other Fuels')
        plt.legend(loc='upper right')
        plt.xlim(2004, max(max(estimate.times), max(target.times)) + 1)


class EstAnnex13ElectricityEmissionsTotal(object):
    def __init__(self):
        self.from_coal = EstAnnex13ElectricityFromCoal()
        self.from_ng = EstAnnex13ElectricityFromNaturalGas()
        self.from_other = EstAnnex13ElectricityFromOther()

        self.total_co2e = (
            self.from_coal.co2e.sum(enums.PT).to(u.kilotonne_CO2e)
            + self.from_ng.co2e.sum(enums.PT).to(u.kilotonne_CO2e)
            + self.from_other.co2e.sum(enums.PT).to(u.kilotonne_CO2e))

    def update_A9_emissions(self, emissions_sectoral_pt):
        for src in [self.from_coal, self.from_ng, self.from_other]:
            emissions_sectoral_pt[:, IPCC.SCS__Public_Electricity_and_Heat] += src.emissions

    def plot_vs_annex13_target(self):
        a13_ng = eccc_nir_annex13.national_electricity_CO2e_from_combustion()['combustion']
        target = a13_ng.to(self.total_co2e.v_unit)

        plt.figure()
        self.total_co2e.plot(label='Estimate', alpha=.5)
        target.plot(label='Annex13 (Target)', alpha=.5)
        plt.title('Emissions: Electricity from Combustion')
        plt.legend(loc='upper right')
        plt.xlim(2004, max(max(estimate.times), max(target.times)) + 1)

    def echart_by_utilities(self):
        years = ipcc_canada.echart_years()
        values = _echart_reference_NIR_values('Stationary Combustion Sources/Public Electricity and Heat Production')

        sts_coal = self.from_coal.co2e.sum(enums.PT).to(u.megatonne_CO2e)
        sts_ng = self.from_ng.co2e_by_ep[
            enums.ElectricityProducer.Utilities].sum(enums.PT).to(u.megatonne_CO2e)
        sts_other = self.from_other.co2e.sum(enums.PT).to(u.megatonne_CO2e)

        return StackedAreaEChart(
            div_id='ipcc_chart_public_electricity',
            title=EChartTitle(
                text='Emissions from Generation of Public Electricity',
                subtext='Hover over data points to see fuel type,'
                ' click through to source file est_nir.py'),
            xAxis=EChartXAxis(data=years),
            yAxis=EChartYAxis(name='Emissions (Mt CO2e)'),
            stacked_series=[
                EChartSeriesStackElem(
                    name='Coal',
                    data=_rstrip_data(sts_coal)),
                EChartSeriesStackElem(
                    name='Natural Gas',
                    data=_rstrip_data(sts_ng)),
                EChartSeriesStackElem(
                    name='Other Fuels',
                    data=_rstrip_data(sts_other)),
            ],
            other_series=[
                EChartSeriesBase(
                    name='NIR Sector Total',
                    lineStyle=EChartLineStyle(color='#303030'),
                    itemStyle=EChartItemStyle(color='#303030'),
                    data=values)])

from .est_nir_hwp import  EstForestAndHarvestedWoodProducts
from .est_nir_venting import EstFugitive_OilandNaturalGas_Venting


NAICS6 = naics.NAICS6

class Est_Energy_SCS_OilAndGas_Extraction(object):

    naics6_oil_and_gas_extraction = (
        NAICS6.Oil_and_gas_extraction__except_oil_sands,
        NAICS6.Conventional_Oil_and_Gas_Extraction,
        NAICS6.NonConventional_Oil_Extraction,
        NAICS6.Insitu_oil_sands_extraction,
        NAICS6.Mined_oil_sands_extraction,
    )

    def init_ghgrp(self):
        EmissionSource = ghgrp.EmissionSource
        nse = ghgrp.GHG_NAICS_source_emissions_backfilled()
        for naics_code in self.naics6_oil_and_gas_extraction:
            self.emissions_by_label[f'GHGRP-registered ({naics_code})']\
                    = nse[:,
                          naics_code,
                          EmissionSource.StationaryFuelCombustion
                         ]

    def init_petrinex_SK(self):
        from .petrinex import (
            ActivityID,
            ProductID,
            FacilityType,
            petrinex_annual_summary)

        extraction_facility_types = [
            FacilityType.Battery,
            FacilityType.GasPlant,
            FacilityType.GasGatheringSystem,
            FacilityType.CustomTreating,
            FacilityType.TankTerminal, # is this extraction??
            FacilityType.FreshFormationWaterSource,
        ]

        self.petrinex_SK = petrinex_annual_summary(PT.SK,
                                                   include_ghgrp=False,
                                                   large_emitter_cutoff=5.0)
        pSK = self.petrinex_SK[ProductID.Gas, ActivityID.Fuel, extraction_facility_types].sum(
            extraction_facility_types)
        emissions = GHG_PT_zeros()
        factor = 2441
        assert all(val == factor for val in eccc_nir_annex6.data_a6_1_2['SK'])
        emissions[GHG.CO2, PT.SK] = (
            sts.with_default_zero(pSK, self.years_u)
            * (factor * u.g_CO2 / u.m3))

        idx = 2
        assert eccc_nir_annex6.df_a6_1_3["Emission Factor Source"][idx]\
                == "Producer Consumption (Non-marketable)"
        emissions[GHG.CH4, PT.SK] = sts.with_default_zero(
            (pSK
             * (eccc_nir_annex6.df_a6_1_3["CH4 (g/m3)"][idx]
                * u.g_CH4 / u.m3)
            ),
            self.years_u)

        emissions[GHG.N2O, PT.SK] = (
            pSK
            * (eccc_nir_annex6.df_a6_1_3["N2O (g/m3)"][idx]
                * u.g_N2O / u.m3)).setdefault_zero(self.years_u)

        self.emissions_by_label['Small Facilities (Saskatchewan)'] = emissions

    def init_petrinex_AB(self):
        from .petrinex import (
            ActivityID,
            ProductID,
            FacilityType,
            petrinex_annual_summary)

        extraction_facility_types = [
            FacilityType.Battery,
            FacilityType.GasPlant,
            FacilityType.GasGatheringSystem,
            FacilityType.InjectionFacility,
        ]

        self.petrinex_AB = petrinex_annual_summary(pt=PT.AB,
                                                   include_ghgrp=False,
                                                   large_emitter_cutoff=5.0)
        pAB = self.petrinex_AB[ProductID.Gas, ActivityID.Fuel, extraction_facility_types].sum(
            extraction_facility_types)
        emissions = GHG_PT_zeros()
        co2_factors = sts.annual_report2(
            years=eccc_nir_annex6.df_a6_1_2.Year,
            values=eccc_nir_annex6.df_a6_1_2.AB,
            v_unit=u.g_CO2 / u.m3) # gas / NG_nmk is implicit
        emissions[GHG.CO2, PT.AB] = sts.with_default_zero(pAB * co2_factors, self.years_u)

        # hack in some old emissions. We know they weren't zero, so how about back-filling
        # the same number as from 2022
        idx_2022, valid_2022 = emissions[GHG.CO2, PT.AB]._idx_of_time(2022 * u.years)
        assert valid_2022
        assert idx_2022 > 2
        emissions[GHG.CO2, PT.AB].values[1:idx_2022] \
                = array.array('d', [emissions[GHG.CO2, PT.AB].values[idx_2022]] * (idx_2022 - 1))

        ch4_factors = sts.annual_report2(
            years=eccc_nir_annex6.df_a6_1_4.Year,
            values=eccc_nir_annex6.df_a6_1_4.AB,
            v_unit=u.g_CH4 / u.m3)
        emissions[GHG.CH4, PT.AB] = sts.with_default_zero(pAB * ch4_factors, self.years_u)

        idx = 2
        assert eccc_nir_annex6.df_a6_1_3["Emission Factor Source"][idx]\
                == "Producer Consumption (Non-marketable)"
        emissions[GHG.N2O, PT.AB] = (
            pAB
            * (eccc_nir_annex6.df_a6_1_3["N2O (g/m3)"][idx]
                * u.g_N2O / u.m3)).setdefault_zero(self.years_u)

        self.emissions_by_label['Small Facilities (Alberta)'] = emissions

        # TODO: injection:
        # *do* count the fuel used for injection
        # and also count the removed carbon that's injected

    def init_bc(self):
        from .bc_pi import bc_provincial_inventory, Sector as BC_Sector
        pi = bc_provincial_inventory()

        emissions = GHG_PT_zeros()
        emissions[:, PT.BC] = pi[:, BC_Sector.Oil_and_Gas_Extraction]

        # TODO: subtract ghgrp facilities
        EmissionSource = ghgrp.EmissionSource
        nse = ghgrp.GHG_NAICS_source_emissions_backfilled()
        bc_nse = nse[:,
                     self.naics6_oil_and_gas_extraction,
                     EmissionSource.StationaryFuelCombustion,
                     PT.BC].sum(1)
        emissions[GHG.CO2, PT.BC] -= sts.with_default_zero(bc_nse[GHG.CO2], self.years_u)
        emissions[GHG.CH4, PT.BC] -= sts.with_default_zero(bc_nse[GHG.CH4], self.years_u)
        emissions[GHG.N2O, PT.BC] -= sts.with_default_zero(bc_nse[GHG.N2O], self.years_u)

        self.emissions_by_label['Small Facilities (British Columbia)'] = emissions

    def __init__(self):
        self.emissions_by_label = {}
        self.years = ipcc_canada.echart_years()
        self.years_u = [yy * u.years for yy in self.years]
        self.init_petrinex_AB()
        self.init_petrinex_SK()
        #self.init_bc() # this is based on NIR, possibly double-counts GHGRP facilities
        self.init_ghgrp()

    def update_A9_emissions(self, emissions_sectoral_pt):
        for label, emissions in self.emissions_by_label.items():
            emissions_sectoral_pt[:, IPCC.SCS__Oil_and_Gas_Extraction] += emissions

    def echart(self):
        years = ipcc_canada.echart_years()
        values = _echart_reference_NIR_values('Stationary Combustion Sources/Oil and Gas Extraction')

        return StackedAreaEChart(
            div_id='ipcc_chart_scs_oilandgas_extraction',
            title=EChartTitle(
                text='Oil and Gas Extraction (Stationary Combustion Sources)',
                subtext='Hover over data points to see emissions by usage,'
                ' click through to source file est_nir.py'),
            xAxis=EChartXAxis(data=years),
            yAxis=EChartYAxis(name='Emissions (Mt CO2e)'),
            stacked_series=[
                EChartSeriesStackElem(
                    name=label,
                    data=_rstrip_data((GWP_100 @ emissions).sum(PT).to(u.Mt_CO2e)))
                for label, emissions in self.emissions_by_label.items()
            ],
            other_series=[
                EChartSeriesBase(
                    name='NIR Sector Total',
                    lineStyle=EChartLineStyle(color='#303030'),
                    itemStyle=EChartItemStyle(color='#303030'),
                    data=values),
            ])


class Est_SCS_Residential(object):

    def init_neud_end_use(self):
        from . import neud
        co2e_by_end_use = neud.ghg_emissions_excl_electricity_by_end_use()
        for end_use in [
            neud.EndUse.WaterHeating,
            neud.EndUse.SpaceHeating,
        ]:
            emissions = GHG_PT_zeros()
            emissions[GHG.CO2] = co2e_by_end_use[end_use] * (1 * u.Mt_CO2 / u.Mt_CO2e)
            self.emissions_by_label[end_use.value] = emissions
            # XXX
            # This data appears to add emissions from wood combustion,
            # which it should perhaps not, if we aren't counting firewood as a negative emission in HWP.
            # Subtracting off wood emissions is a small fraction, and does not appear trivial to do
            # based on the formatting of data in the NEUD.

    def __init__(self):
        self.emissions_by_label = {}
        self.years = ipcc_canada.echart_years()
        self.years_u = [yy * u.years for yy in self.years]
        self.init_neud_end_use()

    def update_A9_emissions(self, emissions_sectoral_pt):
        for label, emissions in self.emissions_by_label.items():
            emissions_sectoral_pt[:, IPCC.SCS__Residential] += emissions

    def echart(self):
        years = ipcc_canada.echart_years()
        values = _echart_reference_NIR_values('Stationary Combustion Sources/Residential')

        return StackedAreaEChart(
            div_id='ipcc_chart_scs_residential',
            title=EChartTitle(
                text='Residential (Stationary Combustion Sources)',
                subtext='Hover over data points to see emissions by usage,'
                ' click through to source file est_nir.py'),
            xAxis=EChartXAxis(data=years),
            yAxis=EChartYAxis(name='Emissions (Mt CO2e)'),
            stacked_series=[
                EChartSeriesStackElem(
                    name=label,
                    data=_rstrip_data((GWP_100 @ emissions).sum(PT).to(u.Mt_CO2e)))
                for label, emissions in self.emissions_by_label.items()
            ],
            other_series=[
                EChartSeriesBase(
                    name='NIR Sector Total',
                    lineStyle=EChartLineStyle(color='#303030'),
                    itemStyle=EChartItemStyle(color='#303030'),
                    data=values),
            ])


class Est_Transport(object):

    def __init__(self):
        self.emissions_by_label = {}
        self.years = ipcc_canada.echart_years()
        self.years_u = [yy * u.years for yy in self.years]
        self.init_neud_transportation_mode()

    def echart(self):
        years = ipcc_canada.echart_years()
        values = _echart_reference_NIR_values(self.catpath_with_whitespace)

        return StackedAreaEChart(
            div_id=self.echart_div_id,
            title=EChartTitle(
                text=self.echart_title,
                subtext='Hover over data points to see emissions by usage,'
                ' click through to source file est_nir.py'),
            xAxis=EChartXAxis(data=years),
            yAxis=EChartYAxis(name='Emissions (Mt CO2e)'),
            stacked_series=[
                EChartSeriesStackElem(
                    name=label,
                    data=_rstrip_data((GWP_100 @ emissions).sum(PT).to(u.Mt_CO2e)))
                for label, emissions in self.emissions_by_label.items()
            ],
            other_series=[
                EChartSeriesBase(
                    name='NIR Sector Total',
                    lineStyle=EChartLineStyle(color='#303030'),
                    itemStyle=EChartItemStyle(color='#303030'),
                    data=values),
            ])


class Est_Transport_LightDutyGasolineTrucks(Est_Transport):

    # All 

    # Active registrations by vehicle type 2017-2024
    # https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=2310030801

    # Road-grade Fuel sold
    # https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=2310006601

    # New motor vehicle sales
    # https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=2010008601
    
    catpath_with_whitespace = 'Transport/Road Transportation/Light-Duty Gasoline Trucks'
    echart_title = 'Light-Duty Trucks (Mobile Combustion Sources)'
    echart_div_id = 'ipcc_chart_mcs_light_duty_gasoline_trucks'

    def init_neud_transportation_mode(self):
        from . import neud
        co2e_by_tmode = neud.ghg_emissions_by_transportation_mode()
        for tmode in [
            neud.TransportationMode.FreightLightTrucks, # stable, echart show below
            neud.TransportationMode.PassengerLightTrucks, # increasing, echart show above
        ]:
            emissions = GHG_PT_zeros()
            emissions[GHG.CO2] = co2e_by_tmode[tmode] * (1 * u.Mt_CO2 / u.Mt_CO2e)
            self.emissions_by_label[f'{tmode.value} (NEUD data)'] = emissions
            # XXX this table doesn't break out light trucks by fuel
            # so it includes some diesel light trucks

    def update_A9_emissions(self, emissions_sectoral_pt):
        for label, emissions in self.emissions_by_label.items():
            emissions_sectoral_pt[:, IPCC.Transport__Road__Light_Duty_Gasoline_Trucks] += emissions


class Est_Transport_LightDutyGasolineVehicles(Est_Transport):

    catpath_with_whitespace = 'Transport/Road Transportation/Light-Duty Gasoline Vehicles'
    echart_title = 'Light-Duty Gasoline Vehicles (Mobile Combustion Sources)'
    echart_div_id = 'ipcc_chart_mcs_light_duty_gasoline_cars'

    def init_neud_transportation_mode(self):
        from . import neud
        co2e_by_tmode = neud.ghg_emissions_by_transportation_mode()
        for tmode in [
            neud.TransportationMode.Cars,
        ]:
            emissions = GHG_PT_zeros()
            emissions[GHG.CO2] = co2e_by_tmode[tmode] * (1 * u.Mt_CO2 / u.Mt_CO2e)
            self.emissions_by_label[f'{tmode.value} (NEUD data)'] = emissions

    def update_A9_emissions(self, emissions_sectoral_pt):
        for label, emissions in self.emissions_by_label.items():
            emissions_sectoral_pt[:, IPCC.Transport__Road__Light_Duty_Gasoline_Vehicles] += emissions


class Est_Transport_HeavyDutyDieselVehicles(Est_Transport):

    catpath_with_whitespace = 'Transport/Road Transportation/Heavy-Duty Diesel Vehicles'
    echart_title = 'Heavy-Duty Diesel Vehicles (Mobile Combustion Sources)'
    echart_div_id = 'ipcc_chart_mcs_heavy_duty_diesel_vehicles'

    def init_neud_transportation_mode(self):
        from . import neud

        medtruck_co2e_by_fuel = neud.medium_truck_ghg_emissions_by_fuel()
        medium_emissions = GHG_PT_zeros()
        medium_emissions[GHG.CO2] = (
            medtruck_co2e_by_fuel[neud.TransportationEnergySource.Diesel]
            * (1 * u.Mt_CO2 / u.Mt_CO2e))

        co2e_by_tmode = neud.ghg_emissions_by_transportation_mode()
        heavy_emissions = GHG_PT_zeros()
        heavy_emissions[GHG.CO2] = (
            co2e_by_tmode[neud.TransportationMode.HeavyTrucks]
            * (1 * u.Mt_CO2 / u.Mt_CO2e))

        bus_co2e_by_fuel = neud.bus_ghg_emissions_by_fuel()
        bus_emissions = GHG_PT_zeros()
        bus_emissions[GHG.CO2] = (
            bus_co2e_by_fuel[neud.TransportationEnergySource.Diesel]
            * (1 * u.Mt_CO2 / u.Mt_CO2e))

        # Ready by eye from chart here:
        # https://440megatonnes.ca/insight/biofuels-in-canada-are-rising-will-emissions-fall
        biodiesel_and_renewable_diesel = sts.STS(
            times=array.array('d', [      2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]),
            t_unit=u.years,
            values=array.array('d', [0.0, .007, .016, .018, 0.02, .020, .020, .022, .024, .026, .030, .035, .035, .036, .067, .069]),
            v_unit=u.dimensionless,
            interpolation = sts.InterpolationMode.current)

        self.emissions_by_label[f'Bus (NEUD data)'] = bus_emissions * (1 - biodiesel_and_renewable_diesel)
        self.emissions_by_label[f'Medium-Duty (NEUD data)'] = medium_emissions * (1 - biodiesel_and_renewable_diesel)
        self.emissions_by_label[f'Heavy-Duty (NEUD data)'] = heavy_emissions * (1 - biodiesel_and_renewable_diesel)
        # see also here for projected production capacity
        # https://www.cer-rec.gc.ca/en/data-analysis/energy-markets/renewable-energy-canada/an-overview-of-bioenergy.html


    def update_A9_emissions(self, emissions_sectoral_pt):
        for label, emissions in self.emissions_by_label.items():
            emissions_sectoral_pt[:, IPCC.Transport__Road__Heavy_Duty_Diesel_Vehicles] += emissions


class Est_EntericFermentation(object):
    # TODO: share code with BovinePopulation in barriers.py
    def __init__(self):
        from .sc_3210013001 import (
            FarmType, Livestock, Livestock_nonsums, SurveyDate,
            number_of_cattle_by_class_and_farm_type)
        from .eccc_nir_annex3p4 import table_A3p4_11
        self.emissions_by_label = {}
        self.cattle_pt, self.cattle_ca = number_of_cattle_by_class_and_farm_type()
        emfac = table_A3p4_11()
        for livestock in Livestock_nonsums:
            cattle_jan1 = self.cattle_pt[SurveyDate.Jan1, livestock, FarmType.AllCattle]
            cattle_jul1 = self.cattle_pt[SurveyDate.Jul1, livestock, FarmType.AllCattle]
            emissions = GHG_PT_zeros()
            emissions[GHG.CH4] += cattle_jan1 * emfac[livestock, None] * (.5 * u.year)
            emissions[GHG.CH4] += cattle_jul1 * emfac[livestock, None] * (.5 * u.year)
            self.emissions_by_label[livestock.value] = emissions
        self.echart_div_id = 'echart_div_enteric_fermentation'
        self.echart_title = 'Enteric Fermentation'

    def update_A9_emissions(self, emissions_sectoral_pt):
        for label, emissions in self.emissions_by_label.items():
            emissions_sectoral_pt[:, IPCC.Enteric_Fermentation] += emissions

    def echart(self):
        years = ipcc_canada.echart_years()
        values = _echart_reference_NIR_values('Enteric Fermentation')

        return StackedAreaEChart(
            div_id=self.echart_div_id,
            title=EChartTitle(
                text=self.echart_title,
                subtext='Hover over data points to see emissions by usage,'
                ' click through to source file est_nir.py'),
            xAxis=EChartXAxis(data=years),
            yAxis=EChartYAxis(name='Emissions (Mt CO2e)'),
            stacked_series=[
                EChartSeriesStackElem(
                    name=label,
                    data=_rstrip_data((GWP_100 @ emissions).sum(PT).to(u.Mt_CO2e)))
                for label, emissions in self.emissions_by_label.items()
            ],
            other_series=[
                EChartSeriesBase(
                    name='NIR Sector Total',
                    lineStyle=EChartLineStyle(color='#303030'),
                    itemStyle=EChartItemStyle(color='#303030'),
                    data=values),
            ])


class EstSectorEmissions(object):

    def __init__(self):
        self.sectoral_emissions = objtensor.empty(GHG, IPCC, PT)
        for ghg, kt in kt_by_ghg.items():
            self.sectoral_emissions[ghg] = 0 * kt

        EstAnnex13ElectricityEmissionsTotal().update_A9_emissions(self.sectoral_emissions)
        EstForestAndHarvestedWoodProducts().update_A9_emissions(self.sectoral_emissions)
        EstFugitive_OilandNaturalGas_Venting().update_A9_emissions(self.sectoral_emissions)
        Est_Energy_SCS_OilAndGas_Extraction().update_A9_emissions(self.sectoral_emissions)
        Est_SCS_Residential().update_A9_emissions(self.sectoral_emissions)
        Est_Transport_LightDutyGasolineTrucks().update_A9_emissions(self.sectoral_emissions)
        Est_Transport_LightDutyGasolineVehicles().update_A9_emissions(self.sectoral_emissions)
        Est_Transport_HeavyDutyDieselVehicles().update_A9_emissions(self.sectoral_emissions)
        Est_EntericFermentation().update_A9_emissions(self.sectoral_emissions)

    def max_gap_2005(self, thresh_Mt=1000):
        a9_2005_total = eccc_nir_annex9.emissions_by_IPCC_sector(2005, 'Total_CO2e')
        estimate = GWP_100 @ self.sectoral_emissions.sum(enums.PT)
        gaps = []
        for keys, buf_offset in estimate.ravel_keys_offsets():
            sector, = keys
            try:
                est_2005 = estimate.buf[buf_offset].query(2005 * u.years)
            except AttributeError:
                est_2005 = estimate.buf[buf_offset]
                assert est_2005.magnitude == 0
            gaps.append((abs(est_2005 - a9_2005_total[sector]).to('kt_CO2e'), sector))

        max_gap = None
        for gap, sector in reversed(sorted(gaps)):
            if max_gap is None:
                max_gap = gap
            if gap.magnitude > thresh_Mt:
                print(f'{gap:.2f} {sector}')
        return max_gap
