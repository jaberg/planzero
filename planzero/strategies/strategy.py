from pydantic import BaseModel
import datetime

from ..html import HTML_element, HTML_raw, HTML_P, HTML_UL

class ActionStep(BaseModel):
    descr: str
    who: str
    due: datetime.datetime
    costs: float # CAD


class Stakeholder(BaseModel):
    name: str
    url: str

from .. import Project

Strategy_subclasses = []

class Strategy(Project):

    # short description, typically nota full sentence
    title: str

    # the strategy for which the action plan is developed
    description: str | None

    # objectives supported:
    ipcc_catpaths: list[str]

    # in corporate env: owner, implementation team
    # in open-source env:

    # publicly pursuing some version of this sort of project
    implementations: list[Stakeholder] = []

    # interested parties: interested in talking about this sort of project
    interested_parties: list[Stakeholder] = []

    # outputs / deliverables
    # generic roll-up keys to which this project contributes (e.g. freight, electricity)
    output_keys: list[str] = []
    other_outputs: list[str] = [] # e.g. organizations

    # inputs / costs
    # generic roll-up keys upon which this project relies (e.g. batteries, labour, electricity)
    input_keys: list[str] = []
    other_inputs: list[str] = [] # e.g. "lots of fresh water" which maybe we don't model yet

    # dates  (see timelines)
    start_date: datetime.datetime | None = None
    end_date: datetime.datetime | None = None

    action_steps: list[ActionStep] = []

    # strategies are not physical processes, they should not produce emissions
    # they can modulate the rates of production and utilization of physical assets
    # that are already simulated (and which can generate emissions)
    may_register_emissions:bool = False

    after_tax_cashflow_name: str

    # True for Strategies that develop ideas
    # False for e.g. ComboA
    is_idea:bool = True

    @classmethod
    def __init_subclass__(cls):
        super().__init_subclass__()
        Strategy_subclasses.append(cls)

    def __init__(self, **kwargs):
        if 'after_tax_cashflow_name' not in kwargs:
            kwargs = dict(kwargs, after_tax_cashflow_name=f'{self.__class__.__name__}_AfterTaxCashFlow')
        if 'description' not in kwargs:
            kwargs = dict(kwargs, description=self.__class__.__doc__)
        super().__init__(**kwargs)

    def init_add_subprojects(self, sub_projects):
        super().init_add_subprojects(sub_projects)
        catpaths = set(self.ipcc_catpaths) 
        for proj in sub_projects:
            catpaths |= set(proj.ipcc_catpaths)
        self.ipcc_catpaths = list(sorted(catpaths))

        # TODO: same union trick for output_keys, input_keys, etc.

    @property
    def url_path(self):
        return f'/strategies/{self.identifier}/'

    def strategy_page_section_environmental_model(self, project_comparison):
        rval = StrategyPageSection(
            identifier='national_emissions_impact_section',
            title='National Emissions Impact',
            elements=[])

        rval.append_str_as_paragraph("""
        Below: the impact of this strategy on atmospheric CO2 concentration.
        """)
        rval.elements.append(
            HTML_raw(
                raw=self.project_graph_svg(
                    dict(
                        sts_key='Atmospheric_CO2_conc',
                        t_unit='years',
                        figtype='plot vs baseline'),
                    project_comparison.state_A,
                    project_comparison)))

        rval.append_str_as_paragraph("""
        Below: the difference in global heat forcing.
        The datapoints in this curve are used to compute the Net Present Heat
        for the project, by adding up the energy associated with each year
        (modulated by the future discount factor).
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

        rval.append_str_as_paragraph("""
        In terms of temperature, the difference in global heat forcing can be quantified as equivalent to a small change in (upward) temperature trajectory for the top 200m of the world's oceans.
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

        return rval


class StrategyPageSection(HTML_element):
    identifier: str
    title: str
    elements: list[HTML_element]
    level:int = 2

    def as_html(self):
        elements = '\n'.join(elem.as_html() for elem in self.elements)
        return f'<h{self.level} id={self.identifier}>{self.title}</h{self.level}>{elements}'

    def append_str_as_paragraph(self, para_text):
        self.elements.append(HTML_P(elements=[HTML_raw(raw=para_text)]))

import markdown

class HTML_Markdown(HTML_element):
    content: str

    def as_html(self):
        html_output = markdown.markdown(self.content)
        return html_output

import sympy

class HTML_Math(HTML_element):
    sympy:object = sympy

    def as_html(self):
        raise NotImplementedError() # override-me


class StrategyPage(BaseModel):
    show_table_of_contents:bool = True
    sections: list[StrategyPageSection] = []

    def table_of_contents_html_list(self):
        lis = ''.join(f'<li><a href="#{section.identifier}">{section.title}</a></li>'
                      for section in self.sections)
        return f'<ol>{lis}</ol>'
