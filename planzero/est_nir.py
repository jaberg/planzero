import matplotlib.pyplot as plt
import numpy as np

from .enums import CoalType
from .ghgvalues import GWP_100

from .ureg import (u, kg_by_ghg,
                   kilotonne_by_coal_type,
                   kt_by_ghg,
                   m3_by_roundwood_species_group)
from . import ureg
from . import sc_np
from . import objtensor
from . import sts
from . import annex6_np
from . import eccc_nir_annex6
from . import eccc_nir_annex9
from . import eccc_nir_annex13
from . import enums
from . import ptvalues
from . import nrc_nfd
from . import ipcc_canada

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
    which I take to be (because of SC-25-10-0017-1):

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
        RPP_User = enums.RPP_User
        self.LightAndHeavyOil = [FT.LightFuelOil, FT.HeavyFuelOil]

        self.emission_factors_lhk = annex6_np.A6_1_6_LFO_HFO_Kerosene()

        # afaik the only RPP_User producing electricity is ElectricUtilities
        # GHG x LightAndHeavy x PT
        self.emissions_LH_pt = (
            self.emission_factors_lhk[:, self.LightAndHeavyOil, RPP_User.ElectricUtilities, None]
            * self.prov_consumption[self.LightAndHeavyOil])

    def init_Diesel(self):
        FT = enums.FuelType

        self.emission_factors_dg = annex6_np.A6_1_6_Diesel_and_Gasoline()
        # GHG x PT
        self.emissions_diesel_pt = (
            self.emission_factors_dg[:, FT.Diesel, None]
            * self.prov_consumption[FT.Diesel])

    def init_PetCoke(self):
        FT = enums.FuelType
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
            * (self.prov_consumption[FT.PetCoke] / petcoke_material_density)
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

        FT = enums.FuelType
        self.emissions_wood_pt = objtensor.empty(GHG, enums.PT)
        for ghg in [GHG.CO2, GHG.HFCs, GHG.PFCs, GHG.SF6, GHG.NF3]:
            self.emissions_wood_pt[ghg] = 0 * kg_by_ghg[ghg]

        self.emissions_wood_pt[GHG.CH4] = self.prov_consumption[FT.Wood] \
                    * (0.1 * u.g_CH4 / u.kg_wood_mc25)
        self.emissions_wood_pt[GHG.N2O] = self.prov_consumption[FT.Wood] \
                    * (0.07 * u.g_N2O / u.kg_wood_mc25)

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

        self.emissions_methane_pt[GHG.CO2] = self.prov_consumption[FT.Methane] \
                    * (1900 * u.g_CO2 / u.m3_methane)
        self.emissions_methane_pt[GHG.CH4] = self.prov_consumption[FT.Methane] \
                    * (.49 * u.g_CH4 / u.m3_methane)
        self.emissions_methane_pt[GHG.N2O] = self.prov_consumption[FT.Methane] \
                    * (.049 * u.g_N2O / u.m3_methane)

    def init_OtherGaseousFuels(self):
        # The comment on this row of the StatsCan table
        # mentions "refinery fuel gas" which is also sometimes
        # called "still gas" in e.g. the ECCC NIR Annex6,
        # so I'll use those emission factors for the whole "Other Gaseous
        # Fuels" category.
        FT = enums.FuelType
        RPP_User = enums.RPP_User

        # TODO: confirm that UpgradingFacilities use PetCoke
        # for heat (to create more RPPs and PetCoke), not electricity
        self.emissions_stillgas_pt = (
            self.emission_factors_ps[:, FT.StillGas, RPP_User.RefineriesAndOthers, None]
            * self.prov_consumption[FT.StillGas]
        )
        

    def __init__(self):
        self.prov_consumption, self.national_consumption = \
                sc_np.Archived_Electric_Power_Generation_Annual_Fuel_Consumed_by_Electrical_Utility()

        self.init_LightHeavy()
        self.init_Diesel()
        self.init_PetCoke()
        self.init_Wood()
        #self.init_OtherSolidFuels()
        self.init_Methane()
        self.init_OtherGaseousFuels()

        self.emissions = (
            self.emissions_LH_pt.sum(self.LightAndHeavyOil)
            + self.emissions_diesel_pt
            + self.emissions_petcoke_pt
            + self.emissions_wood_pt
            + self.emissions_methane_pt
            + self.emissions_stillgas_pt
        )

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

    def echart(self):
        non_agg = ipcc_canada.non_agg
        years = ipcc_canada.echart_years()
        values = non_agg[non_agg['CategoryPathWithWhitespace'] == 'Stationary Combustion Sources/Public Electricity and Heat Production']['CO2eq'].values

        sts_coal = self.from_coal.co2e.sum(enums.PT).to(u.megatonne_CO2e)
        sts_ng = self.from_ng.co2e.sum(enums.PT).to(u.megatonne_CO2e)
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
                    data=[
                        {'value': 0 if np.isnan(vv.magnitude) else vv.magnitude,
                         'url': 'https://github.com/jaberg/planzero/blob/main/planzero/est_nir.py'}
                        for vv in sts_coal.query([yy * u.years for yy in years])],
                    ),
                EChartSeriesStackElem(
                    name='Natural Gas',
                    data=[
                        {'value': 0 if np.isnan(vv.magnitude) else vv.magnitude,
                         'url': 'https://github.com/jaberg/planzero/blob/main/planzero/est_nir.py'}
                        for vv in sts_ng.query([yy * u.years for yy in years])],
                    ),
                EChartSeriesStackElem(
                    name='Other Fuels',
                    data=[
                        {'value': 0 if np.isnan(vv.magnitude) else vv.magnitude,
                         'url': 'https://github.com/jaberg/planzero/blob/main/planzero/est_nir.py'}
                        for vv in sts_other.query([yy * u.years for yy in years])],
                    ),
            ],
            other_series=[
                EChartSeriesBase(
                    name='NIR Sector Total',
                    data=values / 1000)])


class EstForestAndHarvestedWoodProducts(object):
    """
    Canada's forests are modelled here as constant-size, constant-composition
    ecosystems, which buffer atmospheric carbon. This model ignores types and
    location of forest, although it does recognize the different density of
    softwood and hardwood species groups.

    The model is that the tree population absorb carbon from the atmosphere at
    a constant rate, and releases it at a variable rate in three ways:

    1. harvesting for durable products (transfers C to products & atmosphere)
    2. decay after insect infestation
    3. combustion due to wild fire

    Data supporting this simple model is drawn from 
    http://nfdp.ccfm.org/en/download.php

    """

    C_coef = .5 * u.kg_carbon / u.kg_wood_od # Annex6 Table 5-1
    CO2_coef = 3.667 * u.kg_CO2 / u.kg_carbon # molecule weight ratio

    def __init__(self):
        # This initial value doesn't really matter, it isn't trying
        # to reflect a physical quantity.
        self.buffered_C_1990 = 0 * u.megatonne_wood_mc25
        self.decay_horizon = 2025 * u.years

        # This rate of accumulation does matter, the intention is to
        # approximately match the trend of forestry sector emissions reported
        # by the ECCC. This is not a very interesting model, in that it
        # doesn't attempt to reveal anything about why the trend is the value
        # that it is. NRCan does world-class bottom-up forest carbon
        # modelling, including with open-source software, although I couldn't
        # yet find public data for that software. It would be interesting
        # future work to use that model instead.
        #self.reforestation_rate = 50 * u.megatonne_wood_mc25 / u.year

        self.net_harvested_w_tenure = nrc_nfd.net_merchantable_volume_harvested()
        # ignore tenure
        self.net_harvested = self.net_harvested_w_tenure.sum(enums.RoundwoodTenure)

        SG = enums.RoundwoodSpeciesGroup
        RPC = enums.RoundwoodProductCategory

        # how much of each GHG is transferred from Forest_Land -> HWP
        self.HWP_captured = objtensor.empty(GHG, RPC, PT)
        # how much of each GHG is transferred from HWP -> Atmosphere
        self.HWP_released = objtensor.empty(GHG, RPC, PT)

        for ghg, kt in kt_by_ghg.items():
            self.HWP_captured[ghg] = 0 * kt
            self.HWP_released[ghg] = 0 * kt

        for cat in RPC:
            if cat == RPC.Logs_and_Bolts:
                self._logs_and_bolts()
            elif cat == RPC.Other_Industrial_Roundwood:
                self._other_industrial_roundwood()
            elif cat == RPC.Fuelwood_and_Firewood:
                self._fuelwood_and_firewood()
            elif cat == RPC.Pulpwood:
                self._pulpwood()
            else:
                raise NotImplementedError(cat)

        self.forest_emissions = self.HWP_captured.sum(RPC)
        self.HWP_emissions = self.HWP_released.sum(RPC) - self.forest_emissions

    def _delayed_CO2_released(self, CO2_by_pt, rpc, halflife):
        for pt in PT:
            report = CO2_by_pt[pt]
            if isinstance(report, sts.SparseTimeSeries):
                report = sts.annual_report_decay(
                    report, halflife * u.years, self.decay_horizon)
            self.HWP_released[GHG.CO2, rpc, pt] = report

    def _logs_and_bolts(self):
        # The carbon content of net harvested timber is accounted as
        # being instantaneously decremented from the forest (considered as part of the atmosphere) and slowly
        # incremented in the HWP sector. This is physically
        # unrealistic in that the CO2 was absorbed slowly into the
        # tree over the tree's lifetime, not all at once.
        # sector. But the HWP increment
        #
        # Gemini suggests "Mill Efficiency Factor" of 50-60%
        # ends up as long-term lumber, and the figure can be
        # found in NIR. The rest will be end up combusted (though
        # possibly some paper?).

        idx = eccc_nir_annex6.data_a6_5_2['Description'].index(
            'Species-weighted average density, Sawnwood')
        assert eccc_nir_annex6.data_a6_5_2['Units'][idx] == 'od tonne per m3'
        density_magnitude = eccc_nir_annex6.data_a6_5_2['Value'][idx]
        mill_efficiency = .7

        SG = enums.RoundwoodSpeciesGroup
        RPC = enums.RoundwoodProductCategory
        CO2_by_sg = objtensor.empty(SG, PT)
        for sg in SG:
            avg_density = (
                density_magnitude
                * (u.tonne_wood_od / m3_by_roundwood_species_group[sg]))
            CO2_by_sg[sg] = (
                self.net_harvested[sg, RPC.Logs_and_Bolts]
                * (mill_efficiency * avg_density * self.C_coef * self.CO2_coef))
            # TODO: What about N2O and CH4 - do these products go to landfill?

        # will be deducted from Forest_Land
        CO2_by_pt = CO2_by_sg.sum(SG)
        self.HWP_captured[GHG.CO2, RPC.Logs_and_Bolts] = CO2_by_pt
        self._delayed_CO2_released(
            CO2_by_pt, RPC.Logs_and_Bolts,
            halflife=eccc_nir_annex6.dict_a6_5_3['Canada', 'Sawnwood'])

    def _other_industrial_roundwood(self):
        # pilings, railway ties, electricity poles
        # similar to Logs and Bolts but lasts slightly longer
        idx = eccc_nir_annex6.data_a6_5_2['Description'].index(
            'Species-weighted average density, Other industrial roundwood')
        assert eccc_nir_annex6.data_a6_5_2['Units'][idx] == 'od tonne per m3'
        density_magnitude = eccc_nir_annex6.data_a6_5_2['Value'][idx]
        mill_efficiency = .7

        SG = enums.RoundwoodSpeciesGroup
        RPC = enums.RoundwoodProductCategory
        CO2_by_sg = objtensor.empty(SG, PT)
        for sg in SG:
            avg_density = (
                density_magnitude
                * (u.tonne_wood_od / m3_by_roundwood_species_group[sg]))
            CO2_by_sg[sg] = (
                self.net_harvested[sg, RPC.Other_Industrial_Roundwood]
                * (mill_efficiency * avg_density * self.C_coef * self.CO2_coef))
            # What about N2O and CH4 - do these products go to landfill?

        # will be deducted from Forest_Land
        CO2_by_pt = CO2_by_sg.sum(SG)
        self.HWP_captured[GHG.CO2, RPC.Other_Industrial_Roundwood] = CO2_by_pt
        self._delayed_CO2_released(
            CO2_by_pt, RPC.Other_Industrial_Roundwood,
            halflife=eccc_nir_annex6.dict_a6_5_3['Canada', 'Other industrial roundwood'])

    def _fuelwood_and_firewood(self):
        # Fuelwood CO2 is considered to released
        # back to the atmosphere from whence it came
        # but N2O and CH4 are considered new emissions
        idx = eccc_nir_annex6.data_a6_5_2['Description'].index(
            'Species-weighted average density, Bioenergy')
        assert eccc_nir_annex6.data_a6_5_2['Units'][idx] == 'od tonne per m3'
        density_raw = eccc_nir_annex6.data_a6_5_2['Value'][idx]

        SG = enums.RoundwoodSpeciesGroup
        RPC = enums.RoundwoodProductCategory
        for sg in SG:
            # extra 15% moisture content, supposing oven-dried is 10% moisture
            # content, and mc25 is, by definition, 25.
            avg_density = (
                (density_raw + .15)
                * (u.tonne_wood_mc25 / m3_by_roundwood_species_group[sg]))

            mass_wood_mc25 = self.net_harvested[sg, RPC.Fuelwood_and_Firewood] * avg_density
            # eyeballing table Annex6 6-1
            # using coefficients for residential combustion
            CH4_coef = 5.0 * u.g_CH4 / u.kg_wood_mc25
            N2O_coef = 0.06 * u.g_N2O / u.kg_wood_mc25
            self.HWP_released[GHG.CH4, RPC.Fuelwood_and_Firewood] \
                    = CH4_coef * mass_wood_mc25
            self.HWP_released[GHG.N2O, RPC.Fuelwood_and_Firewood] \
                    = N2O_coef * mass_wood_mc25

    def _pulpwood(self):
        # Pulpwood C is considered to be removed from
        # forestry sector when it's harvested, but then
        # returned to the atmosphere over the following years
        idx = eccc_nir_annex6.data_a6_5_2['Description'].index(
            'Species-weighted average density, Roundwood')
        assert eccc_nir_annex6.data_a6_5_2['Units'][idx] == 'od tonne per m3'
        density_magnitude = eccc_nir_annex6.data_a6_5_2['Value'][idx]

        SG = enums.RoundwoodSpeciesGroup
        RPC = enums.RoundwoodProductCategory
        CO2_by_sg = objtensor.empty(SG, PT)
        for sg in SG:
            avg_density = (
                density_magnitude
                * (u.tonne_wood_od / m3_by_roundwood_species_group[sg]))
            CO2_by_sg[sg] = (
                self.net_harvested[sg, RPC.Pulpwood]
                * (avg_density * self.C_coef * self.CO2_coef))

            # The CH4 and N2O are supposed to be picked up by Waste sectors,
            # such as landfill. TODO: figure out a lower bound on landfill
            # emissions?

        CO2_by_pt = CO2_by_sg.sum(SG)
        self.HWP_captured[GHG.CO2, RPC.Pulpwood] = CO2_by_pt
        self._delayed_CO2_released(
            CO2_by_pt, RPC.Pulpwood,
            halflife=eccc_nir_annex6.dict_a6_5_3['Canada', 'Pulp and paper'])

    def plot_forest_emissions(self):
        GHG = enums.GHG
        ptvalues.scatter_subplots(
            self.forest_emissions[[GHG.CO2, GHG.CH4, GHG.N2O]],
            v_unit_by_outer_key={
                GHG.CO2: u.kilotonne_CO2,
                GHG.CH4: u.tonne_CH4,
                GHG.N2O: u.tonne_N2O,
            },
            legend_loc='upper left')

    def plot_HWP_emissions(self):
        GHG = enums.GHG
        ptvalues.scatter_subplots(
            self.HWP_emissions[[GHG.CO2, GHG.CH4, GHG.N2O]],
            v_unit_by_outer_key={
                GHG.CO2: u.kilotonne_CO2,
                GHG.CH4: u.tonne_CH4,
                GHG.N2O: u.tonne_N2O,
            },
            legend_loc='upper left')

    def plot_total_co2e(self):
        a9_totals = {
            year: eccc_nir_annex9.emissions_by_IPCC_sector(year, 'Total_CO2e')
            for year in range(1990, 2020)}

        fig, (ax0, ax1) = plt.subplots(1, 2)

        forest = (GWP_100 @ self.forest_emissions.sum(enums.PT)).to(u.kt_CO2e)
        ax0.scatter(
            forest.times,
            forest.values[1:],
            label='Estimate')
        ax0.scatter(
            [int(yr) for yr in forest.times if yr in a9_totals],
            [a9_totals[yr][IPCC.Forest_Land].to(u.kt_CO2e).magnitude
             for yr in forest.times if yr in a9_totals],
            label='NIR')
        ax0.set_title('Forest')

        hwp = (GWP_100 @ self.HWP_emissions.sum(enums.PT)).to(u.kt_CO2e)
        ax1.scatter(
            hwp.times,
            hwp.values[1:],
            label='Estimate')
        ax1.scatter(
            [int(yr) for yr in hwp.times if yr in a9_totals],
            [a9_totals[int(yr)][IPCC.Harvested_Wood_Products].to(u.kt_CO2e).magnitude
             for yr in hwp.times if yr in a9_totals],
            label='NIR')
        ax1.set_title('HWP')

    def update_A9_emissions(self, sectoral_emissions):
        sectoral_emissions[:, IPCC.Forest_Land] \
                += self.forest_emissions
        sectoral_emissions[:, IPCC.Harvested_Wood_Products] \
                += self.HWP_emissions


class EstSectorEmissions(object):

    def __init__(self):
        self.sectoral_emissions = objtensor.empty(GHG, IPCC, PT)
        for ghg, kt in kt_by_ghg.items():
            self.sectoral_emissions[ghg] = 0 * kt

        EstAnnex13ElectricityEmissionsTotal().update_A9_emissions(self.sectoral_emissions)
        EstForestAndHarvestedWoodProducts().update_A9_emissions(self.sectoral_emissions)

    def max_gap_2005(self):
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
            else:
                max_gap = max(max_gap, gap)
            print('* ' if gap.magnitude > 100 else '  ', gap, sector)
        return max_gap
