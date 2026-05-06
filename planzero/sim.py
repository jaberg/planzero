
from .my_functools import cache
from functools import cached_property

from pydantic import BaseModel, computed_field
import numpy as np

from .enums import IPCC_Sector
from .ureg import u
from .base import State
from .base import Other_NIR_Historical_Actuals

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

from . import ipcc_canada
from .ghgvalues import GHG, GWP_100


class SimulationResult(BaseModel):

    simulation_name: str

    state: object
    ablations: dict[str, object] = {}

    @cached_property
    def year_ints(self) -> list[int]:
        start = int(self.state.t_start.to(u.years).magnitude)
        now = int(self.state._t_now.to(u.years).magnitude)
        assert now > start, (self.state.t_start, self.state._t_now)
        return list(range(start, now))

    @cached_property
    def year_times(self) -> list[object]:
        return [tt * u.years for tt in self.year_ints]


    @staticmethod
    def annual_sector_Mt_CO2e_by_year(ipcc_sector) -> dict[int, float]:
        df = ipcc_canada.non_agg[
            ipcc_canada.non_agg['CategoryPathWithWhitespace'] \
            == ipcc_sector.catpath_with_whitespace]
        values = df['CO2eq'].values / 1000
        years = df['Year']
        rval = {int(year): float(val) for year, val in zip(years, values)}
        return rval

    @cached_property
    def by_ipcc_sector(self) -> StackedAreaEChart:
        emres = self.state.compute_annual_emissions()
        sector_totals = {
            ipcc_sector: emres.total(
                only_ipcc_sector=ipcc_sector,
                return_None_instead_of_zero=True)
            for ipcc_sector in IPCC_Sector}

        for_sorting = []
        for ipcc_sector, sector_total in sector_totals.items():
            if sector_total is None:
                continue
            data = EChartSeriesData(
                sector_total,
                times=self.year_times,
                v_unit=u.Mt_CO2e,
                url=f'/simulations/{self.simulation_name.lower()}/ipcc-sectors/{ipcc_sector.catpath_no_whitespace}/')
            values = [vdict['value'] for vdict in data]
            if max(values) <= 0:
                # all negative
                for_sorting.append((1.0 / min(values),
                                    ipcc_sector.catpath_with_whitespace,
                                    data))
            elif min(values) >= 0:
                # all positive
                for_sorting.append((max(values),
                                    ipcc_sector.catpath_with_whitespace,
                                    data))
            else:
                # mix of positive and negative entries
                sink_years = [
                    dict(vdict, value=min(vdict['value'], 0))
                    for vdict in data]
                source_years = [
                    dict(vdict, value=max(vdict['value'], 0))
                    for vdict in data]
                for_sorting.append(
                    (1.0 / min(values),
                     ipcc_sector.catpath_with_whitespace + ' (sink years)',
                     sink_years))
                for_sorting.append(
                    (max(values),
                     ipcc_sector.catpath_with_whitespace + ' (source years)',
                     source_years))

        return StackedAreaEChart(
            div_id='by_ipcc_sector',
            title=EChartTitle(
                text=f'Emissions by IPCC Sector: {self.simulation_name} simulation',
                subtext='Hover over data points to see sector labels'),
            xAxis=EChartXAxis(data=self.year_ints),
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
                        self.state.compute_annual_subsidies().total(),
                        times=self.year_times,
                        v_unit=u.giga_CAD,
                        url=None)),
            ])

    def echart_ipcc_sector_reference_NIR_values(self, ipcc_sector):
        v_by_yr = self.annual_sector_Mt_CO2e_by_year(ipcc_sector)
        values = []
        for yr in self.year_ints:
            try:
                values.append(v_by_yr[yr])
            except KeyError:
                break
        return values

    def echart_ipcc_sector(self, catpath) -> StackedAreaEChart:
        """Return an EChart that shows the emissions contributions to this
        sector in the base scenario.
        """
        ipcc_sector = IPCC_Sector.from_catpath(catpath)
        emres = self.state.compute_annual_emissions()

        return StackedAreaEChart(
            div_id=f'echart_ipcc_sector_{catpath.replace("/", "_")}',
            title=EChartTitle(
                text=f'{ipcc_sector.value} ({self.simulation_name} simulation)',
                subtext='Hover over data points to see emissions by usage,'),
            xAxis=EChartXAxis(data=self.year_ints),
            yAxis=EChartYAxis(name='Emissions (Mt CO2e)'),
            stacked_series=[
                EChartSeriesStackElem(
                    name=f'{driver}',
                    data=EChartSeriesData(
                        emres.total(only_ipcc_sector=ipcc_sector,
                                    only_driver=driver),
                        times=self.year_times,
                        v_unit=u.Mt_CO2e,
                        url=None),
                    emphasis={'disabled': 1}, # prevents visual corruption on my computer
                    )
                for driver in emres.drivers
            ],
            other_series=[
                EChartSeriesBase(
                    name='NIR Sector Total',
                    lineStyle=EChartLineStyle(color='#303030'),
                    itemStyle=EChartItemStyle(color='#303030'),
                    data=self.echart_ipcc_sector_reference_NIR_values(ipcc_sector)),
            ])

from .base import DynamicElement

site_simulations = {}

class SiteSimulation(BaseModel):
    """A specific simulation (no caller configuration, all pre-loaded)
    to be listed on the site's simulation page.
    """

    @classmethod
    def __init_subclass__(cls):
        super().__init_subclass__()
        site_simulations[cls.__name__] = cls()

    @computed_field
    def short_description(self) -> str:
        return self.__class__.__doc__

    @computed_field
    def t_start_year(self) -> int:
        raise NotImplementedError()

    def dynamic_elements(self) -> list[DynamicElement]:
        # TODO: move to model
        raise NotImplementedError()

    def dynelems_by_id(self, identifier):
        # convenience method called in strategies.html jinja2 template
        return [de for de in self.dynamic_elements()
                if de.identifier == identifier]



class NIR2025(SiteSimulation):
    """Visualize the data from National Greenhouse Gas Inventory Report
    NIR-2025."""

    @computed_field
    def t_start_year(self) -> int:
        return 1990

    def dynamic_elements(self) -> list[DynamicElement]:
        return [Other_NIR_Historical_Actuals()]

    #"""Extend statistical trends in emissions contributions"""
from . import cattle 


class Scaling(SiteSimulation):
    """Model maximal deployment of existing products"""

    @computed_field
    def t_start_year(self) -> int:
        return 1990

    def dynamic_elements(self) -> list[DynamicElement]:
        return [
            cattle.Cattle_Population_AR(),
            cattle.Bovaer_Adoption_Limit(),
            cattle.Bovaer_Production_Emission_Factors(),
            cattle.Cattle_Enteric_Emission_Rates_NIR2025_Bovaer(),
            cattle.Bovaer_Purchase_Cost(),
            cattle.Bovaer_Farm_Subsidy(),
            cattle.Bovaer_Monitoring(),
            cattle.Scale_Bovaer(),

            # standard for vis
            Other_NIR_Historical_Actuals(),
        ]

@cache
def simulation_result(simulation_name) -> SimulationResult:
    site_sim = site_simulations[simulation_name]

    def run_sim(exclude_name=None):
        state = State(
            name=f'State_{simulation_name}' + (f'_minus_{exclude_name}' if exclude_name else ''),
            t_start=site_sim.t_start_year * u.years)
        if exclude_name:
            dynelems = [d for d in site_sim.dynamic_elements() if d.identifier != exclude_name]
        else:
            dynelems = site_sim.dynamic_elements()
        state.add_projects(dynelems)
        state.run_until(2100 * u.years)
        assert state._t_now >= 2100 * u.years
        return state

    baseline_state = run_sim()
    ablations = {}
    for d in site_sim.dynamic_elements():
        if 'strategy' in d.tags:
            name = d.identifier
            ablations[name] = run_sim(exclude_name=name)

    return SimulationResult(
            simulation_name=simulation_name,
            state=baseline_state,
            ablations=ablations)
