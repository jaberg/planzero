"""
https://www.mckinsey.com/features/mckinsey-center-for-future-mobility/our-insights/battery-2035-building-new-advantages
"""

from . import BaseScenarioProject
from . import SparseTimeSeries
from . import ureg as u

class LithiumIonBatteryTechnology(BaseScenarioProject):
    """
    LFP aka Lithium Iron Phosphate batteries.
    Currently these are about 80% of Chinas production capacity, used for
    almost everything.

    In the future, it is possible that solid state batteries and/or
    Lithium-sulfur take over high-energy-density applications, and Sodium-Ion
    batteries take over applications where weight is not so important and cost
    is more important.
    """

    def __init__(self):
        super().__init__()
        self.stepsize = 1.0 * u.years

    def on_add_project(self, state):

        with state.defining(self) as ctx:

            ctx.LithiumIonBatteryTechnology_gravimetric_energy_density = SparseTimeSeries(
                default_value=175.0 * u.watt * u.hour / u.kg)

            ctx.LithiumIonBatteryTechnology_volumetric_energy_density = SparseTimeSeries(
                default_value=400 * u.watt * u.hour / u.liter)

            # N.B. this doesn't reflect tariffs or subsidies
            ctx.LithiumIonBatteryTechnology_cost = SparseTimeSeries(
                default_value=55 * u.USD / (u.kilowatt * u.hour))

        return 2030 * u.years

    def step(self, state, current):

        if state.t_now < 2031 * u.years:
            current.LithiumIonBatteryTechnology_gravimetric_energy_density = 200.0 * u.watt * u.hour / u.kg
            current.LithiumIonBatteryTechnology_cost = 55 * u.USD / (u.kilowatt * u.hour)
            return 2035 * u.years

        elif state.t_now < 2036 * u.years:
            current.LithiumIonBatteryTechnology_gravimetric_energy_density = 225.0 * u.watt * u.hour / u.kg
            current.LithiumIonBatteryTechnology_cost = 53 * u.USD / (u.kilowatt * u.hour)
            return 2040 * u.years

        elif state.t_now < 2041 * u.years:
            current.LithiumIonBatteryTechnology_gravimetric_energy_density = 250.0 * u.watt * u.hour / u.kg
            # TODO: look up some cost projections from e.g. NREL
            # and factor in inflation
            current.LithiumIonBatteryTechnology_cost = 50 * u.USD / (u.kilowatt * u.hour)

            # this technology will be mature
            return None


class SodiumIonBatteryTechnology(BaseScenarioProject):
    """
    Sodium-Ion batteries are expected to track similar performance to
    lithium-ion batteries, but with lower density and lower cost.
    """

    def __init__(self):
        super().__init__()

    def on_add_project(self, state):

        with state.defining(self) as ctx:

            ctx.SodiumIonBatteryTechnology_gravimetric_energy_density = SparseTimeSeries(
                default_value=150.0 * u.watt * u.hour / u.kg)

            ctx.SodiumIonBatteryTechnology_volumetric_energy_density = SparseTimeSeries(
                default_value=400 * u.watt * u.hour / u.liter)

            # N.B. this doesn't reflect tariffs or subsidies
            ctx.SodiumIonBatteryTechnology_cost = SparseTimeSeries(
                default_value=75 * u.USD / (u.kilowatt * u.hour))

        return 2030 * u.years

    def step(self, state, current):

        if state.t_now < 2031 * u.years:
            current.SodiumIonBatteryTechnology_gravimetric_energy_density = 180.0 * u.watt * u.hour / u.kg
            current.SodiumIonBatteryTechnology_cost = 55 * u.USD / (u.kilowatt * u.hour)
            return 2032 * u.years

        elif state.t_now < 2033 * u.years:
            current.SodiumIonBatteryTechnology_gravimetric_energy_density = 200.0 * u.watt * u.hour / u.kg
            current.SodiumIonBatteryTechnology_cost = 50 * u.USD / (u.kilowatt * u.hour)
            return 2035 * u.years

        elif state.t_now < 2036 * u.years:
            current.SodiumIonBatteryTechnology_gravimetric_energy_density = 200.0 * u.watt * u.hour / u.kg
            current.SodiumIonBatteryTechnology_cost = 45 * u.USD / (u.kilowatt * u.hour)
            return 2040 * u.years

        elif state.t_now < 2041 * u.years:
            current.SodiumIonBatteryTechnology_gravimetric_energy_density = 220.0 * u.watt * u.hour / u.kg
            # TODO: look up some cost projections from e.g. NREL
            # and factor in inflation
            current.SodiumIonBatteryTechnology_cost = 40 * u.USD / (u.kilowatt * u.hour)

            # this technology will be mature
            return None


class MarineBatteryTechnology(BaseScenarioProject):

    def __init__(self):
        super().__init__()

    def on_add_project(self, state):
        with state.defining(self) as ctx:
            ctx.MarineBatteryTechnology_gravimetric_energy_density = SparseTimeSeries(
                default_value=175.0 * u.watt * u.hour / u.kg)

            ctx.MarineBatteryTechnology_volumetric_energy_density = SparseTimeSeries(
                default_value=400 * u.watt * u.hour / u.liter)

            # This is based on Gemini3 conjecture about pricing on Corvus'
            # "Blue Whale" system, for which public pricing is not available.
            ctx.MarineBatteryTechnology_system_cost = SparseTimeSeries(
                default_value=300 * u.USD / (u.kilowatt * u.hour))

        return 2030 * u.years

    def step(self, state, current):

        # marine batteries are built differently, and they are a niche
        # compared to e.g. grid-scale or EV batteries. Lithium batteries
        # are notoriously flammable, and marine lithium batteries must
        # never experience runaway thermal on a ship at sea.
        lithium_marine_multiplier = 300.0 / 55.0

        # I'm just hoping this number is lower because sodium is
        # inherently less flammable, but marine applications will always
        # be niche compared to grid BESS systems.
        sodium_marine_multiplier = lithium_marine_multiplier * .5

        lithium_ion_cost = state.latest.LithiumIonBatteryTechnology_cost
        sodium_ion_cost = state.latest.SodiumIonBatteryTechnology_cost

        if lithium_ion_cost * lithium_marine_multiplier < sodium_ion_cost * sodium_marine_multiplier:
            current.MarineBatteryTechnology_gravimetric_energy_density = state.latest.LithiumIonBatteryTechnology_gravimetric_energy_density
            current.MarineBatteryTechnology_volumetric_energy_density = state.latest.LithiumIonBatteryTechnology_volumetric_energy_density
            current.MarineBatteryTechnology_system_cost = lithium_ion_cost * lithium_marine_multiplier
        else:
            current.MarineBatteryTechnology_gravimetric_energy_density = state.latest.SodiumIonBatteryTechnology_gravimetric_energy_density
            current.MarineBatteryTechnology_volumetric_energy_density = state.latest.SodiumIonBatteryTechnology_volumetric_energy_density
            current.MarineBatteryTechnology_system_cost = sodium_ion_cost * sodium_marine_multiplier

        return state.t_now + 1.0 * u.year


# TODO: solid state batteries for EVs
# TODO: Lithium Sulfur
