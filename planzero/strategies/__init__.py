import math

from . import strategy
from . import stakeholders
from . import battery_tug
from . import battery_freighter

from .. import Project, SparseTimeSeries
from .. import ureg as u

from .strategy import Strategy, StrategyPage, StrategyPageSection, HTML_raw, HTML_Markdown
from .ideas import Idea


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
            NationalBovaerMandate(),
            battery_tug.BC_BatteryTug(),
            Force_Government_ZEVs(),
            #battery_freighter.BatteryFreighter(),
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

from ..base import GeometricBovinePopulationForecast
bovaer_assumptions_markdown = """
### Modelling assumptions

* Price of Bovaer, per year: {self.bovaer_price}
* Methane per head, per year: {foo.methane_per_head_per_year}
* Methane reduction due to Bovaer: {methred}
"""
# Yowsers... surely there's better syntax to look up these constants?

class NationalBovaerMandate(Strategy):

    """
    Bovaer
    <a href="https://www.dsm-firmenich.com/anh/news/press-releases/2024/2024-01-31-canada-approves-bovaer-as-first-feed-ingredient-to-reduce-methane-emissions-from-cattle.html">Bovaer</a>
    is a feed supplement that reduces the methane produced by bovine digestion.
    A national Bovaer mandate would phase in the use of Bovaer nation-wide for all cattle.
    There is no obvious economic benefit (or harm) to farmers for using Bovaer, so
    it would be appropriate for a governing body to pay for the additive and drive adoption through regulation.
    """

    peak_year: object = 2035 * u.year
    shoulder_years: object = 5 * u.years
    stepsize: object = 1.0 * u.years
    bovaer_price: object = 150 * u.CAD / u.cattle / u.year

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + math.exp(-x))

    def __init__(self):
        super().__init__(
            title="National Bovaer Mandate",
            description="Compel cattle farmers to administer Bovaer with regulation and subsidies",
            ipcc_catpaths=['Enteric_Fermentation'],
            )

    def on_add_project(self, state):
        with state.requiring_latest(self) as ctx:
            ctx.bovine_population_on_bovaer = SparseTimeSeries(
                default_value=0 * u.cattle)
        with state.defining(self) as ctx:
            ctx.bovine_population_fraction_on_bovaer = SparseTimeSeries(
                default_value=0 * u.dimensionless)
            setattr(ctx, self.after_tax_cashflow_name, SparseTimeSeries(
                default_value=0 * u.MCAD))
        return state.t_now

    def step(self, state, current):
        zval = (
            (state.t_now - self.peak_year)
            / self.shoulder_years)
        current.bovine_population_fraction_on_bovaer = self.sigmoid(
            zval.to('dimensionless').magnitude) * u.dimensionless
            
        setattr(
            current,
            self.after_tax_cashflow_name, 
            -state.latest.bovine_population_on_bovaer
            * self.bovaer_price * self.stepsize)

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

        foo = project_comparison.state_A.projects['GeometricBovinePopulationForecast']
        methred = 1 - foo.bovaer_methane_reduction_fraction

        rval.elements.append(HTML_Markdown(content=bovaer_assumptions_markdown.format(**locals())))

        rval.append_str_as_paragraph(f"""
        This analysis uses the following place-holder projection of the national bovine herd size, that extrapolates a very gradual decline from the current size.
        """)
        rval.elements.append(
            HTML_raw(
                raw=self.project_graph_svg(
                    dict(
                        sts_key='bovine_population',
                        t_unit='years'),
                    project_comparison.state_A,
                    project_comparison)))

        rval.append_str_as_paragraph(f"""
        This project supposes a sigmoidal adoption curve of Bovaer, centered at year {self.peak_year.to('years').magnitude}.
        """)
        rval.elements.append(
            HTML_raw(
                raw=self.project_graph_svg(
                    dict(
                        sts_key='bovine_population_fraction_on_bovaer',
                        t_unit='years'),
                    project_comparison.state_A,
                    project_comparison)))

        rval.append_str_as_paragraph(f"""
        We therefore see a dropping annual emissions of bovine methane, partly through the adoption of Bovaer, and partly due to the gradual reduction in population.
        """)
        rval.elements.append(
            HTML_raw(
                raw=self.project_graph_svg(
                    dict(
                        sts_key='bovine_methane',
                        t_unit='years'),
                    project_comparison.state_A,
                    project_comparison)))

        rval.append_str_as_paragraph(f"""
        As Bovaer is adopted, the impact on Canada's emissions is modulated by the number of cattle.
        The size of Canada's national herd has declined over the last decade, the future is not known.
        """)
        rval.elements.append(
            HTML_raw(
                raw=self.project_graph_svg(
                    dict(
                        sts_key='bovine_population_on_bovaer',
                        t_unit='years'),
                    project_comparison.state_A,
                    project_comparison)))

        rval.append_str_as_paragraph(f"""
        The impact on national methane emissions from so-called enteric fermentation is expected to be significant.
        """)
        rval.elements.append(
            HTML_raw(
                raw=self.project_graph_svg(
                    dict(
                        sts_key='Predicted_Annual_Emitted_CH4_mass',
                        t_unit='years',
                        figtype='plot vs baseline'),
                    project_comparison.state_A,
                    project_comparison)))

        rval.append_str_as_paragraph(f"""
        The impact on global atmospheric methane concentration is expected to be noticeable.
        """)
        rval.elements.append(
            HTML_raw(
                raw=self.project_graph_svg(
                    dict(
                        sts_key='Atmospheric_CH4_conc',
                        t_unit='years',
                        figtype='plot vs baseline'),
                    project_comparison.state_A,
                    project_comparison)))

        rval.append_str_as_paragraph(f"""
        The impact on global heat forcing is too small to see on a graph of that phenomenon.
        """)
        rval.elements.append(
            HTML_raw(
                raw=self.project_graph_svg(
                    dict(
                        sts_key='DeltaF_forcing',
                        t_unit='years',
                        figtype='plot vs baseline'),
                    project_comparison.state_A,
                    project_comparison)))

        rval.append_str_as_paragraph(f"""
        Still, a visualization of the difference in global heat forcing reveals the shape of the impact over time.
        The datapoints in this curve are used to compute the Net Present Heat for the project, by adding up the energy associated with each year (modulated by the future discount factor).
        """)
        rval.elements.append(
            HTML_raw(
                raw=self.project_graph_svg(
                    dict(
                        sts_key='DeltaF_forcing',
                        t_unit='years',
                        figtype='plot delta'),
                    project_comparison.state_A,
                    project_comparison)))

        rval.append_str_as_paragraph(f"""
        In other terms, the difference in global heat forcing due to a national Bovaer mandate can be quantified as a small change in (upward) temperature trajectory for the top 200m of the world's oceans.
        """)
        rval.elements.append(
            HTML_raw(
                raw=self.project_graph_svg(
                    dict(
                        sts_key='Ocean_Temperature_Anomaly',
                        t_unit='years',
                        figtype='plot delta'),
                    project_comparison.state_A,
                    project_comparison)))

        rval.append_str_as_paragraph(f"""
        In terms of financial modelling, the project assumes a price of Bovaer ( {self.bovaer_price} ) that remains constant for the next 200 years. At the scale of production associated with national adoption in Canada and in other countries, this is arguably an over-estimate.
        The curve is simply the product of the population on Bovaer with the price per head.
        The datapoints in this curve are used to compute the Net Present Value for the project.
        """)
        rval.elements.append(
            HTML_raw(
                raw=self.project_graph_svg(
                    dict(
                        sts_key=self.after_tax_cashflow_name,
                        t_unit='years',
                        figtype='plot'),
                    project_comparison.state_A,
                    project_comparison)))

        return rval


def standard_strategies():
    return [
            ComboA(),
            NationalBovaerMandate(),
            battery_tug.BC_BatteryTug(),
            Force_Government_ZEVs(),
            #battery_freighter.BatteryFreighter(),
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
