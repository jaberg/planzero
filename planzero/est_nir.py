import matplotlib.pyplot as plt

from .enums import CoalType
from .ghgvalues import GWP_100

from .ureg import u, kg_by_ghg, kilotonne_by_coal_type, kt_by_ghg
from . import ureg
from . import sc_np
from . import objtensor
from . import sts
from . import annex6_np
from . import eccc_nir_annex13
from . import eccc_nir_annex9
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

        GHG = enums.GHG
        FT = enums.FuelType
        self.emissions_wood_pt = objtensor.empty(GHG, enums.PT)
        for ghg in [GHG.CO2, GHG.HFCs, GHG.PFCs, GHG.SF6, GHG.NF3]:
            self.emissions_wood_pt[ghg] = 0 * kg_by_ghg[ghg]

        self.emissions_wood_pt[GHG.CH4] = self.prov_consumption[FT.Wood] \
                    * (0.1 * u.g_CH4 / u.kg_wood)
        self.emissions_wood_pt[GHG.N2O] = self.prov_consumption[FT.Wood] \
                    * (0.07 * u.g_N2O / u.kg_wood)

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
        GHG = enums.GHG
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
            emissions_sectoral_pt[:, enums.IPCC_Sector.SCS__Public_Electricity_and_Heat] += src.emissions

    def plot_vs_annex13_target(self):
        a13_ng = eccc_nir_annex13.national_electricity_CO2e_from_combustion()['combustion']
        target = a13_ng.to(self.total_co2e.v_unit)

        plt.figure()
        self.total_co2e.plot(label='Estimate', alpha=.5)
        target.plot(label='Annex13 (Target)', alpha=.5)
        plt.title('Emissions: Electricity from Combustion')
        plt.legend(loc='upper right')
        plt.xlim(2004, max(max(estimate.times), max(target.times)) + 1)


class EstSectorEmissions(object):

    def __init__(self):
        self.sectoral_emissions = objtensor.empty(enums.GHG, enums.IPCC_Sector, enums.PT)
        for ghg, kt in kt_by_ghg.items():
            self.sectoral_emissions[ghg] = 0 * kt

        EstAnnex13ElectricityEmissionsTotal().update_A9_emissions(self.sectoral_emissions)

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
            #print(a9_2005_total)
            #print(a9_2005_total[enums.IPCC_Sector.Grassland])
            #print(a9_2005_total[enums.IPCC_Sector('Grassland')])
            gaps.append((abs(est_2005 - a9_2005_total[sector]).to('kt_CO2e'), sector))

        max_gap = None
        for gap, sector in reversed(sorted(gaps)):
            if max_gap is None:
                max_gap = gap
            else:
                max_gap = max(max_gap, gap)
            print('* ' if gap.magnitude > 100 else '  ', gap, sector)
        return max_gap
