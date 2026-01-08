from . import BaseScenarioProject
from . import SparseTimeSeries
from . import ureg as u

class IPCC_Transport_Marine_DomesticNavigation_Model(BaseScenarioProject):
    def __init__(self):
        super().__init__()
        self.stepsize = 1.0 * u.years

    def on_add_project(self, state):
        with state.requiring_current(self) as ctx:

            # PacificLogBargeForecast
            ctx.pacific_log_barge_CO2 = SparseTimeSeries(
                default_value=0 * u.kg)

            # TODO: Lakers_CO2_Forecast
            # TODO: Lakes_Tug_Barge_CO2_Forecast
            ctx.Lakers_CO2 = SparseTimeSeries(
                default_value=0 * u.kg)

            # TODO: BC Ferries
            # TODO: Atlantic Ferries
            # TODO: Northern Resupply

            ctx.Other_Domestic_Navigation_CO2 = SparseTimeSeries(
                default_value=1.5 * u.Mt)

        state.register_emission('Transport/Marine/Domestic_Navigation', 'CO2', 'Lakers_CO2')
        state.register_emission('Transport/Marine/Domestic_Navigation', 'CO2', 'Other_Domestic_Navigation_CO2')


class PacificLogBarges(BaseScenarioProject):
    """
    This class represents the combined activities of e.g.
    * Seaspan
    * others

    who own and operate barges and tugs that move logs from remote sites along
    the BC coast to lumber mills on Vancouver Island, and on the mainland near
    e.g. the mouth of the Fraser river.
    """

    def __init__(self):
        super().__init__()
        self.stepsize = 1.0 * u.years

    def on_add_project(self, state):
        with state.requiring_current(self) as ctx:
            ctx.n_pacific_log_tugs_ZEV_constructed = SparseTimeSeries(
                default_value=0 * u.dimensionless)

        with state.defining(self) as ctx:

            ctx.n_pacific_log_tugs = SparseTimeSeries(
                default_value=10 * u.dimensionless)

            # Gemini's best guess was there are about 10 tugs pulling logs down the BC Coast
            ctx.n_pacific_log_tugs_diesel = SparseTimeSeries(
                default_value=10 * u.dimensionless)

            ctx.n_pacific_log_tugs_ZEV = SparseTimeSeries(
                default_value=0 * u.dimensionless)

            ctx.pacific_log_barge_CO2 = SparseTimeSeries(
                default_value=0 * u.kg)

        state.register_emission('Transport/Marine/Domestic_Navigation', 'CO2',
                                'pacific_log_barge_CO2')
        return state.t_now

    def step(self, state, current):
        current.n_pacific_log_tugs_ZEV = current.n_pacific_log_tugs_ZEV_constructed
        current.n_pacific_log_tugs_diesel = (
            current.n_pacific_log_tugs
            - current.n_pacific_log_tugs_ZEV)
        current.n_pacific_log_tugs = (
            current.n_pacific_log_tugs_diesel
            + current.n_pacific_log_tugs_ZEV)
        current.pacific_log_barge_CO2 = (
            2.68 * u.kg / u.liter
            * 400 * u.liter / u.hour
            * current.n_pacific_log_tugs_diesel
            * (300 / 365) # working most days
            * self.stepsize)

        return state.t_now + self.stepsize


class GreatLakesFreight(BaseScenarioProject):
    """
    This class represents the combined activities of e.g.
    * Algoma Central
    * CSL
    * McNeil

    The logic of this class is that it defines (as an assumption) a certain amount of
    ton miles of dry bulk and ignores liquid bulk freight (probably < 10% of
    freight) and container freight (a negligible amount if not actually zero).
    The projection will be a simple extrapolation of historical trends.

    Given the freight work to be done, this class determines how much of that work is
    assigned to the various available vessel types.
    * 1970s-era diesel laker freighter
    * new diesel laker freighter (e.g. equinox-class from Algoma)
    * a hypothesized battery-electric vessel

    This class includes assumptions about the fuel efficiency and crew costs
    of each vessel type, so it calculates the following outputs:
    * amount of marine diesel consumed
    * n. crew employed
    * CO2 emissions produced
    """

    def __init__(self):
        super().__init__()
        self.stepsize = 1.0 * u.years

    def on_add_project(self, state):

        with state.requiring_current(self) as ctx:

            ctx.n_great_lakes_available_freight_tugs = SparseTimeSeries(
                default_value=0 * u.dimensionless)

            ctx.n_great_lakes_available_battery_freight_tugs = SparseTimeSeries(
                default_value=0 * u.dimensionless)

            # assume plenty of capacity in this vessel class
            # it doesn't hurt to have extra here, because we're not modelling
            # how these ships are financed.
            ctx.n_great_lakes_available_freighters_dry_bulk = SparseTimeSeries(
                default_value=100 * u.dimensionless)

        with state.defining(self) as ctx:

            # freight ton miles are considered an input to the model, based
            # on economic demand.
            # This class accesses their historical values and projects future
            # ones.
            ctx.great_lakes_freight_dry_bulk_ton_miles = SparseTimeSeries(
                default_value=100_000_000_000 * u.tonne * u.nautical_mile)

            # The model calculates how many (including fractional) active vessels
            # there are of each type, based on how many vessels are available.
            # The logic is that available battery freight tugs are used first,
            # then diesel tugs are utilized until the available barges are all
            # used, then dedicated freighters meet the the remainder of demand.

            ctx.n_great_lakes_active_battery_freight_tugs = SparseTimeSeries(
                default_value=0 * u.dimensionless)

            ctx.n_great_lakes_active_freighters_dry_bulk = SparseTimeSeries(
                default_value=0 * u.dimensionless)

            ctx.great_lakes_freight_CO2 = SparseTimeSeries(
                default_value=0 * u.kg)

        state.register_emission('Transport/Marine/Domestic_Navigation', 'CO2',
                                'great_lakes_freight_CO2')
        #state.register_employment
        #state.register_diesel_consumption
        #state.register_electricity_consumption
        return state.t_now

    def step(self, state, current):
        average_dry_bulk_backhaul_efficiency = .7
        average_freighter_duty_cycle_efficiency = (1.0 + average_dry_bulk_backhaul_efficiency) / 2.0
        average_freighter_capacity = 30_000 * u.tonne
        freighter_average_speed = 6 * u.knots # average over 12 month period

        current.n_great_lakes_active_freighters_dry_bulk = (
            current.great_lakes_freight_dry_bulk_ton_miles
            / (
                average_freighter_capacity * average_freighter_duty_cycle_efficiency
                * freighter_average_speed * u.year)
        ).to(u.dimensionless)

        current.great_lakes_freight_CO2 = (
            current.n_great_lakes_active_freighters_dry_bulk
            * (26 * u.tonnes / u.day) # diesel consumption
            * (3) # tonnes CO2e / tonne diesel
            * (300 / 365) # working most days
            * self.stepsize)

        return state.t_now + self.stepsize
