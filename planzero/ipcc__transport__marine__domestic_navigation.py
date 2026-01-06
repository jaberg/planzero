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

            ctx.Lakers_CO2 = SparseTimeSeries(
                default_value=0 * u.kg)

            ctx.Other_Domestic_Navigation_CO2 = SparseTimeSeries(
                default_value=3.4 * u.Mt)

        state.register_emission('Transport/Marine/Domestic_Navigation', 'CO2', 'Lakers_CO2')
        state.register_emission('Transport/Marine/Domestic_Navigation', 'CO2', 'Other_Domestic_Navigation_CO2')


class PacificLogBargeForecast(BaseScenarioProject):
    # somehow link/merge with stakeholders.PacificLogBarges

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
