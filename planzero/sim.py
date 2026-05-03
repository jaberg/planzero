
from .my_functools import cache
from functools import cached_property

from pydantic import BaseModel, computed_field
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
        for_sorting = []
        for catpath, contributors in self.state.sectoral_emissions_contributors.items():
            if not contributors:
                continue
            data = EChartSeriesData(
                self.state.sts[f'Predicted_Annual_Emitted_CO2e_mass_{catpath}'],
                times=self.year_times,
                v_unit=u.Mt_CO2e,
                url=f'/simulations/{self.simulation_name.lower()}/ipcc-sectors/{catpath}/')
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
                        self.state.sts[f'AnnualSubsidyTotal'],
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
        from .enums import IPCC_Sector
        ipcc_sector = IPCC_Sector.from_catpath(catpath)

        return StackedAreaEChart(
            div_id=f'echart_ipcc_sector_{catpath.replace("/", "_")}',
            title=EChartTitle(
                text=f'{ipcc_sector.value} ({self.simulation_name} simulation)',
                subtext='Hover over data points to see emissions by usage,'),
            xAxis=EChartXAxis(data=self.year_ints),
            yAxis=EChartYAxis(name='Emissions (Mt CO2e)'),
            stacked_series=[
                EChartSeriesStackElem(
                    name=f'{sts_id}',
                    data=EChartSeriesData(
                        self.state.sts[sts_id] * GWP_100[GHG(ghg)],
                        times=self.year_times,
                        v_unit=u.Mt_CO2e,
                        url=None),
                    emphasis={'disabled': 1}, # prevents visual corruption on my computer
                    )
                for ghg, contribs in self.state.sectoral_emissions_contributors[
                    ipcc_sector.catpath_no_whitespace].items()
                for sts_id in contribs if contribs
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


from .base import (
    AtmosphericChemistry,
    SubsidyAccounting,
    Other_NIR_Historical_Actuals,
    )


class Extrapolation(SiteSimulation):
    """Extend statistical trends in emissions contributions"""

    @computed_field
    def t_start_year(self) -> int:
        return 1990

    def dynamic_elements(self) -> list[DynamicElement]:
        return [
            # standard for viz
            Other_NIR_Historical_Actuals(),
            AtmosphericChemistry(),
            SubsidyAccounting(),
        ]

from .cattle import (
    Cattle_Population,
    Bovaer_Adoption_Limit,
    Cattle_Enteric_Emissions,
    Bovaer_Monitoring,
    )

from .csfs import (
    Reduce_Methane_per_Cattle_Head,
    Reduce_Population_Cattle,
    )

from .strategies.strategy2 import (
    Scale_Bovaer,
    )


class Scaling(SiteSimulation):
    """Model maximal deployment of existing products"""

    @computed_field
    def t_start_year(self) -> int:
        return 1990

    def dynamic_elements(self) -> list[DynamicElement]:
        return [
            ### Barriers
            # cattle & Bovaer
            Cattle_Population(),
            Bovaer_Adoption_Limit(),
            Cattle_Enteric_Emissions(),
            Bovaer_Monitoring(),

            ### CSFs
            # Enteric Fermentation
            Reduce_Methane_per_Cattle_Head(),
            Reduce_Population_Cattle(),

            ### Strategies
            Scale_Bovaer(),

            # standard for vis
            Other_NIR_Historical_Actuals(),
            AtmosphericChemistry(),
            SubsidyAccounting(),
        ]

@cache
def simulation_result(simulation_name) -> SimulationResult:
    site_sim = site_simulations[simulation_name]

    def run_sim(exclude_name=None):
        state = State(
            name=f'State_{simulation_name}' + (f'_minus_{exclude_name}' if exclude_name else ''),
            t_start=site_sim.t_start_year * u.years)
        if exclude_name:
            dynelems = [d for d in site_sim.dynamic_elements() if d.__class__.__name__ != exclude_name]
        else:
            dynelems = site_sim.dynamic_elements()
        state.add_projects(dynelems)
        state.run_until(2100 * u.years)
        return state

    baseline_state = run_sim()
    ablations = {}
    for d in site_sim.dynamic_elements():
        if 'strategy' in d.tags:
            name = d.__class__.__name__
            ablations[name] = run_sim(exclude_name=name)

    return SimulationResult(
            simulation_name=simulation_name,
            state=baseline_state,
            ablations=ablations)
