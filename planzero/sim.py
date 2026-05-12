
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
                url=f'/simulations/{self.simulation_name}/ipcc-sectors/{ipcc_sector.catpath_no_whitespace}/')
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
                text=f'Emissions by IPCC Sector: {self.simulation_name.replace("_", " ")} simulation',
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
                text=f'{ipcc_sector.value} ({self.simulation_name.replace("_", " ")} simulation)',
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
                for driver in emres.drivers_by_sector(ipcc_sector)
            ],
            other_series=[
                EChartSeriesBase(
                    name='NIR Sector Total',
                    lineStyle=EChartLineStyle(color='#303030'),
                    itemStyle=EChartItemStyle(color='#303030'),
                    data=self.echart_ipcc_sector_reference_NIR_values(ipcc_sector)),
            ])

    def strategy_emissions_diffs(self, strategy_name:str, eps_kt:float) -> dict:
        baseline_emres = self.state.compute_annual_emissions()
        ablated_state = self.ablations.get(strategy_name)
        if not ablated_state:
            raise ValueError(f"Strategy not found in this simulation: {strategy_name}")
        ablated_emres = ablated_state.compute_annual_emissions()

        sector_diffs = {}

        for ipcc_sector in IPCC_Sector:
            base_total = baseline_emres.total(only_ipcc_sector=ipcc_sector, return_None_instead_of_zero=True)
            abl_total = ablated_emres.total(only_ipcc_sector=ipcc_sector, return_None_instead_of_zero=True)
            
            if base_total is None and abl_total is None:
                continue
                
            if base_total is None:
                diff = -abl_total
            elif abl_total is None:
                diff = base_total
            else:
                diff = base_total - abl_total

            assert diff.v_unit == u.kt_CO2e, diff.v_unit
            if np.abs(diff.values[1:]).max() > eps_kt:
                sector_diffs[ipcc_sector] = diff

        return sector_diffs

    def strategy_impact_echart(self, strategy_name: str) -> StackedAreaEChart:
        baseline_emres = self.state.compute_annual_emissions()
        ablated_state = self.ablations.get(strategy_name)
        if not ablated_state:
            raise ValueError(f"Strategy not found in this simulation: {strategy_name}")
        ablated_emres = ablated_state.compute_annual_emissions()
        
        sector_diffs = self.strategy_emissions_diffs(strategy_name, eps_kt=1)

        for_sorting = []
        all_positive = True
        all_negative = True
        for ipcc_sector, diff_ts in sector_diffs.items():
            if diff_ts is None:
                continue
            data = EChartSeriesData(
                diff_ts,
                times=self.year_times,
                v_unit=u.kt_CO2e,
                url=f'/simulations/{self.simulation_name}/ipcc-sectors/{ipcc_sector.catpath_no_whitespace}/'
            )
            
            values = [vdict['value'] for vdict in data]
                
            if max(values) <= 0:
                # all negative
                for_sorting.append((1.0 / min(values),
                                    ipcc_sector.catpath_with_whitespace,
                                    data))
                all_positive = False
            elif min(values) >= 0:
                # all positive
                for_sorting.append((max(values),
                                    ipcc_sector.catpath_with_whitespace,
                                    data))
                all_negative = False
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
                all_positive = False
                all_negative = False

        if all_positive or all_negative:
            other_series = []
        else:
            baseline_total = baseline_emres.total()
            ablated_total = ablated_emres.total()
            impact_data = EChartSeriesData(
                baseline_total - ablated_total,
                times=self.year_times,
                v_unit=u.kt_CO2e,
                url=None
            )
            other_series=[
                EChartSeriesBase(
                    name='Net Emissions Delta',
                    lineStyle=EChartLineStyle(color='#303030', width=2),
                    itemStyle=EChartItemStyle(color='#303030'),
                    data=impact_data,
                )
            ]

        return StackedAreaEChart(
            div_id='impact_chart',
            title=EChartTitle(
                text=f'Emissions Impact: {strategy_name.replace("_", " ")}',
                subtext=f'Annual kt CO2e saved in {self.simulation_name}'),
            xAxis=EChartXAxis(data=self.year_ints),
            yAxis=[EChartYAxis(name='Emissions Delta (kt CO2e)')],
            stacked_series=[
                EChartSeriesStackElem(
                    name=catpath_plus,
                    data=data,
                    emphasis={'disabled': 1},
                )
                for _, catpath_plus, data in sorted(for_sorting)
            ],
            other_series=other_series,
            )

    def strategy_subsidies_diffs(self, strategy_name:str, eps_CAD:float) -> dict:
        ablated_state = self.ablations.get(strategy_name)
        if not ablated_state:
            raise ValueError(f"Strategy not found in this simulation: {strategy_name}")
        baseline_subs = self.state.compute_annual_subsidies()
        ablated_subs = ablated_state.compute_annual_subsidies()

        diffs = {}

        for key, base_ts in baseline_subs.by_program_reason_pt_driver.items():
            abl_ts = ablated_subs.by_program_reason_pt_driver[key]
            prog, reas, pt, driver = key
            diff_key = (prog, reas)
            diff = base_ts - abl_ts
            assert diff.v_unit == u.mega_CAD, diff.v_unit
            if np.abs(diff.values[1:]).max() > (eps_CAD / 1_000_000):
                if diff_key in diffs:
                    diffs[diff_key] += diff
                else:
                    diffs[diff_key] = diff

        return diffs

    def strategy_subsidies_echart(self, strategy_name: str) -> StackedAreaEChart:

        subsidy_diffs = self.strategy_subsidies_diffs(strategy_name,
                                                      eps_CAD=100_000.)
        if not subsidy_diffs:
            return None

        for_sorting = [
            (np.max(diff.values[1:]),
             key,
             diff)
            for key, diff in subsidy_diffs.items() ]

        subsidies_chart = StackedAreaEChart(
            div_id='subsidies_chart',
            title=EChartTitle(
                text=f'Subsidies Impact: {strategy_name}',
                subtext=f'Annual cost of subsidies in {self.simulation_name}'),
            xAxis=EChartXAxis(data=self.year_ints),
            yAxis=EChartYAxis(name='Subsidies Required (CAD, Millions)'),
            stacked_series=[
                EChartSeriesStackElem(
                    name=f'{program.value}: {reason}',
                    data=EChartSeriesData(
                        diff,
                        times=self.year_times,
                        v_unit=u.mega_CAD,
                        url=None,),
                    emphasis={'disabled': 1},
                )
                for _, (program, reason), diff in sorted(for_sorting)
            ],
            other_series=[])
        return subsidies_chart


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
    def show_on_simulations_page(self) -> bool:
        return True

    @computed_field
    def short_description(self) -> str:
        return self.__class__.__doc__

    @computed_field
    def t_start_year(self) -> int:
        raise NotImplementedError()

    @computed_field
    def t_stop_year(self) -> int:
        return 2100

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
        state.run_until(site_sim.t_stop_year * u.years)
        assert state._t_now >= site_sim.t_stop_year * u.years
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
