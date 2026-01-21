from .base import ureg as u
from .base import BaseScenarioProject, SparseTimeSeries

class IPCC_Transport_RoadTransportation_HeavyDutyDieselVehicles(BaseScenarioProject):
    stepsize:object = 1.0 * u.years

    def on_add_project(self, state):
        with state.requiring_current(self) as ctx:
            ctx.human_population = SparseTimeSeries(state.t_now, 27_685_730 * u.people)
            ctx.Other_HeavyDutyDieselVehicles_ZEV_fraction = SparseTimeSeries(
                default_value=0 * u.dimensionless)

        with state.defining(self) as ctx:
            # https://www.canada.ca/en/treasury-board-secretariat/services/innovation/greening-government/government-canada-greenhouse-gas-emissions-inventory.html
            # not exactly using ^^ but guessing based on that and Gemini estimation

            # TODO: factor in CO2e footprint of manufacturing each type of vehicle

            # TODO: factor in the emissions of electricity generation in each province
            #       and the population of each province

            ctx.Other_HeavyDutyDieselVehicles_CO2 = SparseTimeSeries(
                default_value=0 * u.Mt)

        state.register_emission(
            'Transport/Road_Transportation/Heavy-Duty_Diesel_Vehicles', 'CO2',
            'Other_HeavyDutyDieselVehicles_CO2')

        return state.t_now + self.stepsize

    def step(self, state, current):
        coefficient = 900.0 * u.kg / u.people
        current.Other_HeavyDutyDieselVehicles_CO2 = (
            current.human_population * coefficient
            * (1 * u.dimensionless - current.Other_HeavyDutyDieselVehicles_ZEV_fraction))
        return state.t_now + self.stepsize
