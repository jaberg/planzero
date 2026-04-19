import math

from . import strategy
from . import stakeholders
from . import battery_tug
from . import battery_freighter

from .. import SparseTimeSeries
from .. import ureg as u

from .strategy import Strategy, StrategyPage, StrategyPageSection, HTML_raw, HTML_Markdown
from .ideas import Idea

from .strategy2 import Strategy2, strategies


# carbon capture
# negative-carbon cement
# enhanced rock weathering (e.g. UNDO, Carbon Run)
# Buying an EV
# Steel production
# Fertilizer production
# Farm machinery
# on-farm biogas capture
# rooftop solar
# highway umbrella solar
# photovoltaic sail boats for cargo
# tethered balloon heat sinks, wind turbines, and solar farms, and transportation medium
# what about India's cattle population!?
# new process for hydrogen peroxide: https://interestingengineering.com/innovation/solar-hydrogen-peroxide-cornell-breakthrough

class ComboA(Strategy):
    """
    A set of recommended strategies for which forecasts are available.
    """

    stepsize:object = 1.0 * u.years

    def __init__(self):
        super().__init__(
            title='Combo A',
            ipcc_catpaths=[],
            is_idea=False,
            )
        self.init_add_subprojects([
            battery_tug.BC_BatteryTug(),
            Force_Government_ZEVs(),
        ])


    def on_add_project(self, state):
        with state.requiring_current(self) as ctx:
            for proj in self._sub_projects:
                setattr(ctx, proj.after_tax_cashflow_name, SparseTimeSeries(
                    default_value=0 * u.MCAD))
        with state.defining(self) as ctx:
            setattr(ctx, self.after_tax_cashflow_name, SparseTimeSeries(
                default_value=0 * u.MCAD))
        return state.t_now

    def step(self, state, current):
        cashflow = 0 * u.MCAD
        for proj in self._sub_projects:
            cashflow += getattr(current, proj.after_tax_cashflow_name)
        setattr(current, self.after_tax_cashflow_name, cashflow)
        return state.t_now + self.stepsize

    def strategy_page_section_members(self):
        from .strategy import HTML_P, HTML_UL, HTML_raw
        # TODO: make a table like on the strategies page so you can sort by significance
        return strategy.StrategyPageSection(
            identifier='elems',
            title='Strategies making up Combo A',
            elements=[
                HTML_P(
                    elements=[
                        HTML_UL(
                            lis=[HTML_raw(raw=f'<a href="{proj.url_path}">{proj.title}</a>: {proj.description}')
                                 for proj in self._sub_projects])]),
            ])

    def strategy_page(self, project_comparison):
        return strategy.StrategyPage(
            show_table_of_contents=True,
            sections=[
                self.strategy_page_section_members(),
                self.strategy_page_section_environmental_model(project_comparison),
            ])


assumptions_markdown = """
### Modelling assumptions

* Government vehicle fleet produces about 2.5% of all Light-Duty Gasoline emissions
* National vehicle fleet size scales with national population
* Vehicles don't get cheaper, average vehicle TCO is equal for ICE and ZEV (conservative).
"""

class Force_Government_ZEVs(Strategy):

    """
    The technology of ZEVs and PHEVs is maturing, the municipal and civil services of the country are 
    already evaluating these vehicle types extensively, and with some success I believe.
    It would be meaningful for our governments to push the transition to this new technology with their significant scale of operations.
    """

    start_time:object = 2022 * u.years
    end_time:object = 2035 * u.years
    stepsize:object = 1.0 * u.years

    def __init__(self):
        super().__init__(
            title="Government ZEV mandate",
            description="Force civilian federal, provincial, and municipality-owned fleets to transition almost completely to EVs",
            ipcc_catpaths= ['Transport/Road_Transportation/Light-Duty_Gasoline_Trucks'],
            )

    def on_add_project(self, state):
        with state.defining(self) as ctx:
            ctx.Government_LightDutyGasolineTrucks_ZEV_fraction = SparseTimeSeries(
                default_value=0 * u.dimensionless)
            setattr(ctx, self.after_tax_cashflow_name, SparseTimeSeries(
                default_value=0 * u.MCAD))
        return state.t_now

    def step(self, state, current):
        if state.t_now >= self.start_time:
            ramp_duration = self.end_time - self.start_time
            ramp_time = state.t_now - self.start_time
            if ramp_time >= ramp_duration:
                current.Government_LightDutyGasolineTrucks_ZEV_fraction = 1 * u.dimensionless
            else:
                current.Government_LightDutyGasolineTrucks_ZEV_fraction = (
                    ramp_time / ramp_duration * u.dimensionless)
        # assume that on average, the total cost of ownership nets out about 0
        # which based on work with Roger Martin seems plausible
        # do not assume batteries get cheaper, or vehicles get cheaper
        setattr(
            current,
            self.after_tax_cashflow_name, 
            0 * u.CAD)
        return state.t_now + self.stepsize

    def strategy_page(self, project_comparison):
        return StrategyPage(
            show_table_of_contents=True,
            sections=[
                self.section_rollout(project_comparison),
                self.strategy_page_section_environmental_model(project_comparison),
            ])

    def section_rollout(self, project_comparison):
        rval = StrategyPageSection(
            identifier='rollout',
            title='Expected Policy Rollout',
            elements=[])

        rval.elements.append(HTML_Markdown(content=assumptions_markdown.format(**locals())))

        rval.append_str_as_paragraph(f"""
        Input assumption: the shape of the fraction of vehicle roles transitioning to ZEVs.
        """)
        rval.elements.append(
            HTML_raw(
                raw=self.project_graph_svg(
                    dict(
                        sts_key='Government_LightDutyGasolineTrucks_ZEV_fraction',
                        t_unit='years'),
                    project_comparison.state_A,
                    project_comparison)))

        rval.append_str_as_paragraph(f"""
        Estimated CO2 emissions from government light-duty gasoline trucks (TODO: include grid emissions, and at least note [externalized] manufacturing emissions).
        """)
        rval.elements.append(
            HTML_raw(
                raw=self.project_graph_svg(
                    dict(
                        sts_key='Government_LightDutyGasolineTrucks_CO2',
                        t_unit='years'),
                    project_comparison.state_A,
                    project_comparison)))

        return rval


def standard_strategies():
    return [
            ComboA(),
            battery_tug.BC_BatteryTug(),
            Force_Government_ZEVs(),
        ]

def _add_strategies_as_ideas():
    for cls in strategy.Strategy_subclasses:
        obj = cls()
        if obj.is_idea:
            idea = Idea(
                descr=obj.description,
                ipcc_catpaths=obj.ipcc_catpaths,
                full_name=obj.title)
            setattr(stakeholders.ideas, obj.identifier, idea)
_add_strategies_as_ideas()
