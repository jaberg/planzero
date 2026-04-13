
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
        sim_years_ints = np.arange(1990, 2090)
        assert (sim_years_ints[:20] == ipcc_canada.echart_years()[:20]).all()
        sim_years = [tt * u.years for tt in sim_years_ints]
        for_sorting = []
        for catpath, contributors in state.sectoral_emissions_contributors.items():
            if not contributors:
                continue
            data = EChartSeriesData(
                state.sts[f'Predicted_Annual_Emitted_CO2e_mass_{catpath}'],
                times=sim_years,
                v_unit=u.Mt_CO2e,
                url=None)
            values = [vdict['value'] for vdict in data]
            if max(values) <= 0:
                # all negative
                for_sorting.append((1.0 / min(values), catpath, data))
            elif min(values) >= 0:
                # all positive
                for_sorting.append((max(values), catpath, data))
            else:
                # mix of positive and negative entries
                sink_years = [
                    dict(vdict, value=min(vdict['value'], 0))
                    for vdict in data]
                source_years = [
                    dict(vdict, value=max(vdict['value'], 0))
                    for vdict in data]
                for_sorting.append((1.0 / min(values), catpath + '(sink years)', sink_years))
                for_sorting.append((max(values), catpath + '(source years)', source_years))

        by_ipcc_sector = StackedAreaEChart(
            div_id='by_ipcc_sector',
            title=EChartTitle(
                text=f'Simulated Emissions by IPCC Sector: {scenario.name} scenario',
                subtext='Hover over data points to see sector labels'),
            xAxis=EChartXAxis(data=sim_years_ints),
            yAxis=[
                EChartYAxis(name='Emissions (Mt CO2e)'),
                EChartYAxis(name='Annual Subsidies (CAD, billions)')],
            stacked_series=[
                EChartSeriesStackElem(
                    name=f'Simulated {catpath_plus}',
                    data=data,
                    emphasis={'disabled': 1}, # prevents visual corruption on my computer
                    )
                for _, catpath_plus, data in sorted(for_sorting)
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

from .base import (
    AtmosphericChemistry,
    SubsidyAccounting,
    Other_NIR_Historical_Actuals,
    )

@functools.cache
def sim_scenario(scenario_name):
    scenario = scenarios.scenarios[scenario_name]
    state = State(
        name=f'State_{scenario_name}',
        t_start=scenario.t_start_year * u.years)
    state.add_projects(scenario.dynelems)
    state.add_project(Other_NIR_Historical_Actuals())
    state.add_project(AtmosphericChemistry())
    state.add_project(SubsidyAccounting())
    state.run_until(2100 * u.years)
    return SimulationResult.from_state_scenario(state, scenario)
