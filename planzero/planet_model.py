from pydantic import BaseModel, computed_field
import numpy as np

from .my_functools import cache

from . import enums
from .enums import GHG, PT, IPCC_Sector
from .sts import STS, InterpolationMode
from .base import (
    DynamicElement,
    BaseScenarioProject,
    SparseTimeSeries,
    ureg as u,
    )
from .ghgvalues import GWP_100
from .sim import SiteSimulation

surface_area_of_earth = 5.1e14 * u.m * u.m

molar_mass_CH4 = 16.0 * u.g / u.mol
molar_mass_CO2 = 44.0 * u.g / u.mol
molar_mass_N2O = 44.01 * u.g / u.mol

atmospheric_conc_per_mass_CO2 = (1 * u.ppm) / (7.8 * u.gigatonne_CO2)
atmospheric_conc_per_mass_CH4 = (1 * u.ppm) / (7.8 * u.gigatonne_CH4) * molar_mass_CO2 / molar_mass_CH4
atmospheric_conc_per_mass_N2O = (1 * u.ppm) / (7.8 * u.gigatonne_N2O) * molar_mass_CO2 / molar_mass_N2O
atmospheric_conc_per_mass_HFC = (1 * u.ppb) / (18.0 * u.megatonne_HFC) # HFC-134a (most common HFC)
atmospheric_conc_per_mass_PFC = (1 * u.ppb) / (15.6 * u.megatonne_PFC) # CF4 (most common PFC)
atmospheric_conc_per_mass_SF6 = (1 * u.ppb) / (25.9 * u.megatonne_SF6)
atmospheric_conc_per_mass_NF3 = (1 * u.ppb) / (12.6 * u.megatonne_NF3)


deltaF_coef_N2O = 0.12 * u.watt / (u.m * u.m) * surface_area_of_earth
deltaF_coef_HFC = 0.16 * u.watt / (u.m * u.m) * surface_area_of_earth
deltaF_coef_PFC = 0.08 * u.watt / (u.m * u.m) * surface_area_of_earth
deltaF_coef_SF6 = 0.57 * u.watt / (u.m * u.m) * surface_area_of_earth
deltaF_coef_NF3 = 0.21 * u.watt / (u.m * u.m) * surface_area_of_earth


# TODO: use values in .ghgvalues
CO2_GWP_100 = 1.0 * u.kg_CO2e / u.kg_CO2
CH4_GWP_100 = 28.0 * u.kg_CO2e / u.kg_CH4
N2O_GWP_100 = 265.0 * u.kg_CO2e / u.kg_N2O
HFC_GWP_100 = 1_430 * u.kg_CO2e / u.kg_HFC
PFC_GWP_100 = 6_630 * u.kg_CO2e / u.kg_PFC
SF6_GWP_100 = 23_500 * u.kg_CO2e / u.kg_SF6
NF3_GWP_100 = 16_100 * u.kg_CO2e / u.kg_NF3



class AtmosphericChemistry(BaseScenarioProject):
    """Combine GHG emissions into a CO2e estimate using GWP-100 emission factors,
    and also simulate a simple radiative forcing and planetary heating model.
    """
    methane_decay_timescale:float = 10.0

    may_register_emissions:bool = False
    requires_emissions_registration_closed:bool = True

    stepsize:object
    decay_N2O:object
    decay_HFC:object
    decay_PFC:object
    decay_SF6:object
    decay_NF3:object

    def __init__(self, stepsize=1.0 * u.years):
        super().__init__(
            stepsize=stepsize,
            decay_N2O=(1 - stepsize / (114 * u.years)),
            decay_HFC=(1 - stepsize / (14 * u.years)),
            decay_PFC=1.0,
            decay_SF6=1.0,
            decay_NF3=1.0,
            )

    def sectoral_emissions_contributors_ish(self, state):
        sectoral_emissions_contributors = {}
        for key_by_driver in state.registries['driver'].values():
            for driver, sts_key in key_by_driver.items():
                state.declare_read_current_sts(self, sts_key)
                ghg = GHG(sts_key[len('impulse_'):])
                sectoral_emissions_contributors['Forest_Land'] = {
                    ghg: [sts_key]}
        return sectoral_emissions_contributors

    def on_add_project(self, state):
        sectoral_emissions_contributors = self.sectoral_emissions_contributors_ish(state)

        with state.defining(self) as ctx:
            for catpath, contributors in sectoral_emissions_contributors.items():
                any_CO2e_contributors = False
                if contributors.get(GHG.CO2, []):
                    setattr(ctx, f'Predicted_Annual_Emitted_CO2_mass_{catpath}',
                            SparseTimeSeries(unit=u.kt_CO2, t_unit=u.year))
                    any_CO2e_contributors = True
                if contributors.get(GHG.CH4, []):
                    setattr(ctx, f'Predicted_Annual_Emitted_CH4_mass_{catpath}',
                            SparseTimeSeries(unit=u.kt_CH4, t_unit=u.year))
                    any_CO2e_contributors = True
                if contributors.get(GHG.N2O, []):
                    setattr(ctx, f'Predicted_Annual_Emitted_N2O_mass_{catpath}',
                            SparseTimeSeries(unit=u.kt_N2O, t_unit=u.year))
                    any_CO2e_contributors = True
                if contributors.get(GHG.HFCs, []):
                    setattr(ctx, f'Predicted_Annual_Emitted_HFC_mass_{catpath}',
                            SparseTimeSeries(unit=u.kt_HFC, t_unit=u.year))
                    any_CO2e_contributors = True
                if contributors.get(GHG.PFCs, []):
                    setattr(ctx, f'Predicted_Annual_Emitted_PFC_mass_{catpath}',
                            SparseTimeSeries(unit=u.kt_PFC, t_unit=u.year))
                    any_CO2e_contributors = True
                if contributors.get(GHG.SF6, []):
                    setattr(ctx, f'Predicted_Annual_Emitted_SF6_mass_{catpath}',
                            SparseTimeSeries(unit=u.kt_SF6, t_unit=u.year))
                    any_CO2e_contributors = True
                if contributors.get(GHG.NF3, []):
                    setattr(ctx, f'Predicted_Annual_Emitted_NF3_mass_{catpath}',
                            SparseTimeSeries(unit=u.kt_NF3, t_unit=u.year))
                    any_CO2e_contributors = True
                if any_CO2e_contributors:
                    setattr(ctx, f'Predicted_Annual_Emitted_CO2e_mass_{catpath}',
                            SparseTimeSeries(unit=u.kt_CO2e, t_unit=u.year))

            ctx.Predicted_Annual_Emitted_CO2_mass = SparseTimeSeries(
                unit=u.kt_CO2, interpolation='no_interpolation', t_unit=u.year)
            ctx.Predicted_Annual_Emitted_CH4_mass = SparseTimeSeries(
                unit=u.kt_CH4, interpolation='no_interpolation', t_unit=u.year)
            ctx.Predicted_Annual_Emitted_N2O_mass = SparseTimeSeries(
                unit=u.kt_N2O, interpolation='no_interpolation', t_unit=u.year)
            ctx.Predicted_Annual_Emitted_HFC_mass = SparseTimeSeries(
                unit=u.kt_HFC, interpolation='no_interpolation', t_unit=u.year)
            ctx.Predicted_Annual_Emitted_PFC_mass = SparseTimeSeries(
                unit=u.kt_PFC, interpolation='no_interpolation', t_unit=u.year)
            ctx.Predicted_Annual_Emitted_SF6_mass = SparseTimeSeries(
                unit=u.kt_SF6, interpolation='no_interpolation', t_unit=u.year)
            ctx.Predicted_Annual_Emitted_NF3_mass = SparseTimeSeries(
                unit=u.kt_NF3, interpolation='no_interpolation', t_unit=u.year)
            ctx.Predicted_Annual_Emitted_CO2e_mass = SparseTimeSeries(
                unit=u.kt_CO2e, interpolation='no_interpolation', t_unit=u.year)

            # XXX : what year do these numbers represent? How can this be a default value
            # when the simulated years are a parameter of the state?
            ctx.Atmospheric_CO2_conc = SparseTimeSeries(unit=u.ppm, default_value=400.0 * u.ppm, t_unit=u.year)
            ctx.Atmospheric_CH4_conc = SparseTimeSeries(unit=u.ppb, default_value=1775.0 * u.ppb, t_unit=u.year)
            ctx.Atmospheric_N2O_conc = SparseTimeSeries(unit=u.ppb, default_value=336.0 * u.ppb, t_unit=u.year)

            # Gemini says these data are from NOAA and are accurate for January 2026
            ctx.Atmospheric_HFC_conc = SparseTimeSeries(unit=u.ppb, default_value=0.1345 * u.ppb, t_unit=u.year)
            ctx.Atmospheric_PFC_conc = SparseTimeSeries(unit=u.ppb, default_value=0.0902 * u.ppb, t_unit=u.year)
            ctx.Atmospheric_SF6_conc = SparseTimeSeries(unit=u.ppb, default_value=0.0124 * u.ppb, t_unit=u.year)
            ctx.Atmospheric_NF3_conc = SparseTimeSeries(unit=u.ppb, default_value=0.0036 * u.ppb, t_unit=u.year)

            ctx.DeltaF_CO2 = SparseTimeSeries(unit=u.petawatt, t_unit=u.year)
            ctx.DeltaF_CH4 = SparseTimeSeries(unit=u.petawatt, t_unit=u.year)
            ctx.DeltaF_N2O = SparseTimeSeries(unit=u.petawatt, t_unit=u.year)
            ctx.DeltaF_HFC = SparseTimeSeries(unit=u.petawatt, t_unit=u.year)
            ctx.DeltaF_PFC = SparseTimeSeries(unit=u.petawatt, t_unit=u.year)
            ctx.DeltaF_SF6 = SparseTimeSeries(unit=u.petawatt, t_unit=u.year)
            ctx.DeltaF_NF3 = SparseTimeSeries(unit=u.petawatt, t_unit=u.year)
            ctx.DeltaF_forcing = SparseTimeSeries(unit=u.petawatt, t_unit=u.year)
            ctx.DeltaF_feedback = SparseTimeSeries(unit=u.petawatt, t_unit=u.year)

            # Heat Energy forcing is the heat equivalent to net annual cashflow, an annual integral
            ctx.Annual_Heat_Energy_forcing = SparseTimeSeries(default_value=0 * u.exajoule, t_unit=u.year)
            ctx.Cumulative_Heat_Energy_forcing = SparseTimeSeries(default_value=0 * u.exajoule, t_unit=u.year)
            ctx.Heat_Energy_imbalance = SparseTimeSeries(unit=u.exajoule, t_unit=u.year)
            ctx.Cumulative_Heat_Energy = SparseTimeSeries(default_value=0.0 * u.exajoule, t_unit=u.year)
            ctx.Ocean_Temperature_Anomaly = SparseTimeSeries(default_value=1.3 * u.kelvin, t_unit=u.year)

        return int(state.t_now.to(u.years).magnitude + 1) * u.years

    def step(self, state, current):
        sectoral_emissions_contributors = self.sectoral_emissions_contributors_ish(state)

        # add up annual emissions from registry
        annual_CO2_mass = 0 * u.kt_CO2
        annual_CH4_mass = 0 * u.kt_CH4
        annual_N2O_mass = 0 * u.kt_N2O
        annual_HFC_mass = 0 * u.kt_HFC
        annual_PFC_mass = 0 * u.kt_PFC
        annual_SF6_mass = 0 * u.kt_SF6
        annual_NF3_mass = 0 * u.kt_NF3

        for catpath, contributors in sectoral_emissions_contributors.items():
            catpath_CO2e_mass = 0 * u.kg_CO2e
            any_CO2e_contributors = False

            catpath_CO2_contributors = contributors.get(GHG.CO2, [])
            if catpath_CO2_contributors:
                catpath_CO2_mass = sum(getattr(current, sts_key) * (1 * u.year)
                                       for sts_key in catpath_CO2_contributors)
                try:
                    setattr(current, f'Predicted_Annual_Emitted_CO2_mass_{catpath}', catpath_CO2_mass)
                except:
                    print(catpath_CO2_contributors)
                    raise
                catpath_CO2e_mass += catpath_CO2_mass * CO2_GWP_100
                annual_CO2_mass += catpath_CO2_mass
                any_CO2e_contributors = True

            catpath_CH4_contributors = contributors.get(GHG.CH4, [])
            if catpath_CH4_contributors:
                catpath_CH4_mass = sum(getattr(current, sts_key) * (1 * u.year) for sts_key in catpath_CH4_contributors)
                try:
                    setattr(current, f'Predicted_Annual_Emitted_CH4_mass_{catpath}', catpath_CH4_mass)
                except:
                    print(catpath_CH4_contributors)
                    raise
                catpath_CO2e_mass += catpath_CH4_mass * CH4_GWP_100
                annual_CH4_mass += catpath_CH4_mass
                any_CO2e_contributors = True

            catpath_N2O_contributors = contributors.get(GHG.N2O, [])
            if catpath_N2O_contributors:
                catpath_N2O_mass = sum(getattr(current, sts_key) * (1 * u.year) for sts_key in catpath_N2O_contributors)
                try:
                    setattr(current, f'Predicted_Annual_Emitted_N2O_mass_{catpath}', catpath_N2O_mass)
                except:
                    print(catpath_N2O_contributors)
                    raise
                catpath_CO2e_mass += catpath_N2O_mass * N2O_GWP_100
                annual_N2O_mass += catpath_N2O_mass
                any_CO2e_contributors = True

            catpath_HFC_contributors = contributors.get(GHG.HFCs, [])
            if catpath_HFC_contributors:
                catpath_HFC_mass = sum(getattr(current, sts_key) * (1 * u.year) for sts_key in catpath_HFC_contributors)
                setattr(current, f'Predicted_Annual_Emitted_HFC_mass_{catpath}', catpath_HFC_mass)
                catpath_CO2e_mass += catpath_HFC_mass * HFC_GWP_100
                annual_HFC_mass += catpath_HFC_mass
                any_CO2e_contributors = True

            catpath_PFC_contributors = contributors.get(GHG.PFCs, [])
            if catpath_PFC_contributors:
                catpath_PFC_mass = sum(getattr(current, sts_key) * (1 * u.year) for sts_key in catpath_PFC_contributors)
                setattr(current, f'Predicted_Annual_Emitted_PFC_mass_{catpath}', catpath_PFC_mass)
                catpath_CO2e_mass += catpath_PFC_mass * PFC_GWP_100
                annual_PFC_mass += catpath_PFC_mass
                any_CO2e_contributors = True

            catpath_SF6_contributors = contributors.get(GHG.SF6, [])
            if catpath_SF6_contributors:
                catpath_SF6_mass = sum(getattr(current, sts_key) * (1 * u.year) for sts_key in catpath_SF6_contributors)
                setattr(current, f'Predicted_Annual_Emitted_SF6_mass_{catpath}', catpath_SF6_mass)
                catpath_CO2e_mass += catpath_SF6_mass * SF6_GWP_100
                annual_SF6_mass += catpath_SF6_mass
                any_CO2e_contributors = True

            catpath_NF3_contributors = contributors.get(GHG.NF3, [])
            if catpath_NF3_contributors:
                catpath_NF3_mass = sum(getattr(current, sts_key) * (1 * u.year) for sts_key in catpath_NF3_contributors)
                setattr(current, f'Predicted_Annual_Emitted_NF3_mass_{catpath}', catpath_NF3_mass)
                catpath_CO2e_mass += catpath_NF3_mass * NF3_GWP_100
                annual_NF3_mass += catpath_NF3_mass
                any_CO2e_contributors = True

            if any_CO2e_contributors:
                setattr(current, f'Predicted_Annual_Emitted_CO2e_mass_{catpath}', catpath_CO2e_mass)


        current.Predicted_Annual_Emitted_CO2_mass = annual_CO2_mass
        current.Predicted_Annual_Emitted_CH4_mass = annual_CH4_mass
        current.Predicted_Annual_Emitted_N2O_mass = annual_N2O_mass
        current.Predicted_Annual_Emitted_HFC_mass = annual_HFC_mass
        current.Predicted_Annual_Emitted_PFC_mass = annual_PFC_mass
        current.Predicted_Annual_Emitted_SF6_mass = annual_SF6_mass
        current.Predicted_Annual_Emitted_NF3_mass = annual_NF3_mass
        current.Predicted_Annual_Emitted_CO2e_mass = (
            CO2_GWP_100 * annual_CO2_mass
            + CH4_GWP_100 * annual_CH4_mass
            + N2O_GWP_100 * annual_N2O_mass
            + HFC_GWP_100 * annual_HFC_mass
            + PFC_GWP_100 * annual_PFC_mass
            + SF6_GWP_100 * annual_SF6_mass
            + NF3_GWP_100 * annual_NF3_mass
        )

        fraction_of_emitted_CO2_that_becomes_atmospheric = .45

        # apply an atmospheric climate model
        annual_CO2_mass_atmospheric = (
            annual_CO2_mass
            * fraction_of_emitted_CO2_that_becomes_atmospheric)
        annual_CH4_mass_atmospheric = annual_CH4_mass * 1.0 # no such discounting of CH4

        annual_emitted_CO2_in_atmosphere_as_concentration = (
            annual_CO2_mass_atmospheric
            * atmospheric_conc_per_mass_CO2)

        annual_emitted_CH4_in_atmosphere_as_concentration = (
            annual_CH4_mass_atmospheric
            / (2.78 * u.megatonne_CH4 / u.ppb)
        ).to(u.ppb)

        # TODO this should be multiplied by stepsize, not 1 year implicitly,
        #      and this process should be tested for robustness to step size
        tau_ch4 = 12.0 # years
        annual_ch4_to_co2_decay = (
            state.latest.Atmospheric_CH4_conc
            / tau_ch4)

        current.Atmospheric_CH4_conc = (
            state.latest.Atmospheric_CH4_conc
            + annual_emitted_CH4_in_atmosphere_as_concentration
            + 180 * u.ppb # baseline from other sources
            - annual_ch4_to_co2_decay)

        # no decay is assumed for CO2
        current.Atmospheric_CO2_conc = (
            state.latest.Atmospheric_CO2_conc
            + annual_emitted_CO2_in_atmosphere_as_concentration
            + 2 * u.ppm # baseline from other sources
            + (annual_ch4_to_co2_decay
               * fraction_of_emitted_CO2_that_becomes_atmospheric)
        )

        reference_CO2_conc = 280.0 * u.ppm
        current.DeltaF_CO2 = (
            5.35 * u.watt / (u.m * u.m)
            * surface_area_of_earth
            * np.log(current.Atmospheric_CO2_conc.to(u.ppm).magnitude
                     / reference_CO2_conc.to(u.ppm).magnitude))

        reference_CH4_conc = 722.0 * u.ppb
        current.DeltaF_CH4 = (
            0.036 * u.watt / (u.m * u.m)
            * surface_area_of_earth
            * (np.sqrt(current.Atmospheric_CH4_conc.to(u.ppb).magnitude)
               - np.sqrt(reference_CH4_conc.to(u.ppb).magnitude)))

        self.step_N2O(state, current, annual_N2O_mass)
        self.step_HFC(state, current, annual_HFC_mass)
        self.step_PFC(state, current, annual_PFC_mass)
        self.step_SF6(state, current, annual_SF6_mass)
        self.step_NF3(state, current, annual_NF3_mass)

        current.DeltaF_forcing = (
            current.DeltaF_CO2
            + current.DeltaF_CH4
            + current.DeltaF_N2O
            + current.DeltaF_HFC
            + current.DeltaF_PFC
            + current.DeltaF_SF6
            + current.DeltaF_NF3
        )

        current.DeltaF_feedback = (
            -1.3 * u.watt / (u.m * u.m) / u.kelvin
            * surface_area_of_earth
            * state.latest.Ocean_Temperature_Anomaly)

        current.Annual_Heat_Energy_forcing = (
            self.stepsize # integrate over duration of stepsize aka 1 year
            * current.DeltaF_forcing)

        current.Cumulative_Heat_Energy_forcing = (
            state.latest.Cumulative_Heat_Energy_forcing
            + self.stepsize # integrate over duration of stepsize aka 1 year
            * current.DeltaF_forcing)

        current.Heat_Energy_imbalance = (
            self.stepsize # integrate over duration of stepsize aka 1 year
            * (current.DeltaF_forcing + current.DeltaF_feedback))

        specific_heat_of_top_200m_of_ocean = 151200.0 * u.exajoule / u.kelvin * 2
        current.Ocean_Temperature_Anomaly = (
            state.latest.Ocean_Temperature_Anomaly
            + (current.Heat_Energy_imbalance
               / (specific_heat_of_top_200m_of_ocean)))

        current.Cumulative_Heat_Energy = (
            state.latest.Cumulative_Heat_Energy
            + current.Heat_Energy_imbalance)

        return int(state.t_now.to(u.years).magnitude + 1) * u.years

    def step_N2O(self, state, current, annual_N2O_mass):
        conc = state.latest.Atmospheric_N2O_conc

        conc += atmospheric_conc_per_mass_N2O * annual_N2O_mass
        conc *= self.decay_N2O
        current.Atmospheric_N2O_conc = conc

        reference_N2O_conc = 270.0 * u.ppb

        current.DeltaF_N2O = (
            deltaF_coef_N2O
            * (np.sqrt(conc.to(u.ppb).magnitude)
               - np.sqrt(reference_N2O_conc.to(u.ppb).magnitude)))

    def step_HFC(self, state, current, annual_HFC_mass):
        conc = state.latest.Atmospheric_HFC_conc

        conc += atmospheric_conc_per_mass_HFC * annual_HFC_mass
        conc *= self.decay_HFC
        current.Atmospheric_HFC_conc = conc

        current.DeltaF_HFC = deltaF_coef_HFC * conc.to(u.ppb).magnitude

    def step_PFC(self, state, current, annual_PFC_mass):
        conc = state.latest.Atmospheric_PFC_conc

        conc += atmospheric_conc_per_mass_PFC * annual_PFC_mass
        conc *= self.decay_PFC
        current.Atmospheric_PFC_conc = conc

        current.DeltaF_PFC = deltaF_coef_PFC * conc.to(u.ppb).magnitude

    def step_SF6(self, state, current, annual_SF6_mass):
        conc = state.latest.Atmospheric_SF6_conc

        conc += atmospheric_conc_per_mass_SF6 * annual_SF6_mass
        conc *= self.decay_SF6
        current.Atmospheric_SF6_conc = conc

        current.DeltaF_SF6 = deltaF_coef_SF6 * conc.to(u.ppb).magnitude

    def step_NF3(self, state, current, annual_NF3_mass):
        conc = state.latest.Atmospheric_NF3_conc

        conc += atmospheric_conc_per_mass_NF3 * annual_NF3_mass
        conc *= self.decay_NF3
        current.Atmospheric_NF3_conc = conc

        current.DeltaF_NF3 = deltaF_coef_NF3 * conc.to(u.ppb).magnitude


from .strategies.strategy2 import Strategy2


class EmissionsImpulseResponse(Strategy2):
    """A hypothetical emissions source for testing Planet_Model """
    ghg:object
    impulse_co2e:object = 1_000_000 * u.kg_CO2e

    @computed_field
    def ipcc_sectors(self) -> list[object]:
        return []

    def on_add_project(self, state):
        rate = self.impulse_co2e / GWP_100[self.ghg] / u.year
        state.declare_sts(
            self,
            sts=SparseTimeSeries(
                times=[2000 * u.year, 2001 * u.year],
                values=[1 * rate, 0 * rate],
                default_value=0 * rate),
            name=f'impulse_{self.ghg.value}',
            write=True)

        state.declare_sts(
            self, 
            sts=SparseTimeSeries(default_value=1.0 * u.dimensionless),
            name=f'factor_{self.ghg.value}',
            write=True)

        state.register_driver(
            pt=PT.XX,
            driver=f'Hypothetical Emissions Impulse {self.ghg.value}',
            sts_key=f'impulse_{self.ghg.value}')
        state.register_emission_factor(
            pt=PT.XX,
            driver=f'Hypothetical Emissions Impulse {self.ghg.value}',
            sts_key=f'factor_{self.ghg.value}',
            ipcc_sector=IPCC_Sector.Forest_Land, # have to choose something
            ghg=self.ghg)


class Planet_Model(SiteSimulation):
    """Visualize the simulation of a simple planetary heat model
    as driven by hypothetical impulse-responses of greenhouse gases
    (this is not a model of Canada's sectoral emissions)."""

    @computed_field
    def t_start_year(self) -> int:
        return 2000

    def dynamic_elements(self) -> list[DynamicElement]:
        rval = [AtmosphericChemistry()]
        rval.extend(
            [EmissionsImpulseResponse(
                ghg=ghg,
                identifier=f'EmissionsImpulseResponse_{ghg.value}')
             for ghg in GHG])
        return rval
