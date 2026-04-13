
import functools

from pydantic import BaseModel
import numpy as np

from .ureg import u
from .base import State

from .html import (
    EChartTitle,
    EChartXAxis,
    EChartYAxis,
    EChartSeriesStackElem,
    EChartSeriesBase,
    EChartSeriesData,
    EChartLineStyle,
    EChartItemStyle,
    StackedAreaEChart)

from . import barriers
from . import ipcc_canada
from . import scenarios


class SimulationResult(BaseModel):

    state: object

    by_ipcc_sector: StackedAreaEChart

    @classmethod
    def from_state_scenario(cls, state, scenario):
        sim_years_ints = np.arange(1995, 2100)
        sim_years = [tt * u.years for tt in sim_years_ints]

        by_ipcc_sector = StackedAreaEChart(
            div_id='by_ipcc_sector',
            title=EChartTitle(
                text='Simulated Emissions by IPCC Sector',
                subtext='Hover over data points to see sector labels'),
            xAxis=EChartXAxis(data=sim_years_ints),
            yAxis=[
                EChartYAxis(name='Emissions (Mt CO2e)'),
                EChartYAxis(name='Annual Subsidies (CAD, billions)')],
            stacked_series=[
                EChartSeriesStackElem(
                    name=f'Simulated {catpath}',
                    data=EChartSeriesData(
                        state.sts[f'Predicted_Annual_Emitted_CO2e_mass_{catpath}'],
                        times=sim_years,
                        v_unit=u.Mt_CO2e,
                        url=None))
                for catpath, contributors in state.sectoral_emissions_contributors.items()
                if contributors
            ],
            other_series=[
                EChartSeriesBase(
                    name='Federal Target',
                    lineStyle=EChartLineStyle(type='dotted', color='#606060'),
                    itemStyle=EChartItemStyle(color='#606060'),
                    data=ipcc_canada.CNZEAA_targets()),
                EChartSeriesBase(
                    name='Historical Net Total (without LULUCF)',
                    lineStyle=EChartLineStyle(color='#303030'),
                    itemStyle=EChartItemStyle(color='#303030'),
                    data=ipcc_canada.net_emissions_total_without_LULUCF()),
                EChartSeriesBase(
                    name='Historical Net Total (with LULUCF)',
                    lineStyle=EChartLineStyle(color='#303030'),
                    itemStyle=EChartItemStyle(color='#303030'),
                    data=ipcc_canada.net_emissions_total()),
                EChartSeriesBase(
                    name='Subsidies Required',
                    yAxisIndex=1,
                    lineStyle=EChartLineStyle(type='dotted', color='#600000'),
                    itemStyle=EChartItemStyle(color='#600000'),
                    data=EChartSeriesData(
                        state.sts[f'AnnualSubsidyTotal'],
                        times=sim_years,
                        v_unit=u.giga_CAD,
                        url=None)),
            ])
        return SimulationResult(
            state=state,
            by_ipcc_sector=by_ipcc_sector,
            )

from .base import AtmosphericChemistry, SubsidyAccounting

@functools.cache
def sim_scenario(scenario_name):
    scenario = scenarios.scenarios[scenario_name]
    state = State(
        name=f'State_{scenario_name}',
        t_start=scenario.t_start_year * u.years)
    state.add_projects(scenario.dynelems)
    state.add_project(AtmosphericChemistry())
    state.add_project(SubsidyAccounting())
    state.run_until(2100 * u.years)
    return SimulationResult.from_state_scenario(state, scenario)
