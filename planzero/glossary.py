"""
This file  should be imported last among library code files,
so that it can import objects throughout the library, and retrieve
their line numbers for constructing github links.
"""
import functools
import jinja2
from pydantic import BaseModel, computed_field

from .singleton_registry import SingletonRegistry

registry = SingletonRegistry()

class AKA_Registry(object):
    def __init__(self):
        self.aliases = {}

    def refresh_alias_list(self):
        self.aliases = {}
        for clsname, obj in registry.items():
            assert clsname not in self.aliases
            self.aliases[clsname] = clsname
            for alias in obj.all_names:
                assert clsname == self.aliases.setdefault(alias, clsname)

    def __getitem__(self, key):
        try:
            clsname = self.aliases[key]
        except KeyError:
            self.refresh_alias_list()
            clsname = self.aliases[key]
        return registry[clsname]

aka_registry = AKA_Registry()


def siteref(term, text=None):
    try:
        return aka_registry[term].site_reference(text or term)
    except KeyError as exc:
        raise exc


from .blog import latex
from . import blog
from . import barriers
from . import cattle
from . import strategies
from .sts import STS
from .base import DynamicElement
from .html import coderef_url


class GlossaryTerm(BaseModel):

    reserved: bool = False

    @computed_field
    def definition(self) -> str:
        assert self.__class__.__doc__, self.__class__
        return self.__class__.__doc__

    @computed_field
    def definition_html(self) -> str:
        if self.definition.startswith('<p>'):
            source = self.definition
        else:
            source = f'<p>{self.definition}</p>'

        template = jinja2.Template(source=source)
        rval = template.render(self.template_globals())
        return rval

    @property
    def see_also(self) -> dict[str, str]:
        return {}

    @computed_field
    def aka(self) -> list[str]:
        return []

    def pretty_names(self):
        pretty_names = set(name.replace('_', ' ') for name in self.all_names)
        return list(sorted(pretty_names))

    @computed_field
    def all_names(self) -> list[str]:
        rval = [self.__class__.__name__]
        #rval = []
        if '_' in self.__class__.__name__:
            rval.append(self.__class__.__name__.replace('_', ' '))
        return rval + self.aka

    @computed_field
    def as_discussed_in_posts(self) -> list[tuple[object, str, str]]:
        # post, hashtarget, intro txt
        return []

    @computed_field
    def code_links(self) -> dict[str, str]:
        rval = {}
        for txt, thing in self.code_refs.items():
            if isinstance(thing, str) and thing.startswith('http'):
                rval[txt] = thing
            elif isinstance(thing, functools._lru_cache_wrapper):
                while hasattr(thing, '__wrapped__'):
                    thing = thing.__wrapped__
                rval[txt] = coderef_url(thing)
            else:
                rval[txt] = coderef_url(thing)
        return rval

    @property
    def code_refs(self) -> dict[str, object]:
        return {}

    @classmethod
    def __init_subclass__(cls):
        super().__init_subclass__()
        if getattr(cls, 'include_in_registry', True): # default to True for historical reasons
            registry.add_class(cls)

    def template_globals(self) -> dict[str, object]:
        def lref(term, text=None):
            return aka_registry[term].local_ref(text)

        return dict(
            CO2e=latex(r'\mathrm{CO}_2\mathrm e '),
            CO2=latex(r'\mathrm{CO}_2'),
            CH4=latex(r'\mathrm{CH}_4'),
            N2O=latex(r"\mathrm N_2 \mathrm O"),
            SF6=latex(r"\mathrm{SF}_6"),
            NF3=latex(r"\mathrm{NF}_3"),
            degrees=latex(r'^\circ'),
            lref=lref,
        )

    def local_ref(self, text=None) -> str:
        if text is None:
            text = self.__class__.__name__.replace('_', ' ')
        return f'<a href="#{self.__class__.__name__}">{text}</a>'


    def site_reference(self, text=None) -> str:
        if text is None:
            text = self.__class__.__name__.replace('_', ' ')
        return f'<a href="/glossary#{self.__class__.__name__}">{text}</a>'


class Concentrated_Distribution(GlossaryTerm):
    """
    A concentrated distribution is a probability distribution
    with most of its probability mass on a small number of possible values
    compared to the full set of possible values. It's an informal description
    of a distribution. The opposite of a concentrated distribution over
    a finite set of possibilities would be a uniform distribution,
    which is a well-defined distribution. There's not really such a thing
    as the opposite of a concentrated distribution over e.g. all real numbers,
    but one distribution may be described as being more concentrated than another.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Probability_Distribution': 'a distribution may be more or less concentrated',
        }


class Time_Series(GlossaryTerm):
    """A PlanZero time series is a modelling data structure for representing
    a time-varying quantity in some unit of measure.
    The key attributes of a time series are:
    <ul>
        <li>a sequence of numeric values the series takes</li>
        <li>a sequence of times marking <i>when</i> the series takes these values</li>
        <li>a unit of measure for the values</li>
        <li>an interpolation mode, indicating how to interpret value for times other than the enumerated ones</li>
    </ul>
    """

    @computed_field
    def as_discussed_in_posts(self) -> list[tuple[object, str, str]]:
        return [
            (blog.Glossary(), '#timeseries', "see section on Time Series"),
        ]

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Unit_of_Measure': 'the values of a time series are associated with a single unit of measure',
            'Time_Series_Interpolation_Mode': 'the rule for determining value for un-mentioned times',
            'Dynamic Element': 'Dynamic Elements define time series',
            'Simulation': 'Simulation constructs Scenarios from Models',
            'Scenario': 'Scenarios are sets of time series covering a common time interval',
        }

    @property
    def code_refs(self) -> dict[str, object]:
        return {
            'Time Series base class': STS,
        }

class Time_Series_Interpolation_Mode(GlossaryTerm):
    """<p>The time series interpolation mode is a mechanism that is partly for
    convenience and partly for error prevention.
    There are currently two possible interpolation modes.
    The value of a time series at times other than those explicitly mentioned is either
    <ul>
        <li>"current", defined to be the most recent value of the series</li>
        <li>"no interpolation", which leaves such values undefined</li>
    </ul>
    The interpolation mode of a time series is configured by the initialization logic
    of a dynamic element when it creates the time series.
    Time series that are well-defined for continuous ranges of time, such as emission rates,
    should typically be configured with "current" interpolation.
    Time series that are well-defined only for specific points in time, such as annual totals,
    should typically be configured with "no interpolation" as the interpolation mode.
    </p>
    """
    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Time_Series': 'the data structure time series',
            'Dynamic_Element': (
                "Dynamic elements define each time series'"
                " interpolation mode in their initialization logic"),
        }

    @computed_field
    def as_discussed_in_posts(self) -> list[tuple[object, str, str]]:
        return [
            (blog.Glossary(), '#timeseries', "see section on Time Series"),
        ]


class Unit_of_Measure(GlossaryTerm):
    """A PlanZero unit of measure is one that is registered
    in the ureg.py file, using the <a
    href="https://pint.readthedocs.io/en/stable/">Pint unit package</a>.
    The registry includes standard units of measure in e.g. the metric SI
    system in addition to various more traditional ones, and also
    custom units related to PlanZero modelling such as types of coal,
    greenhouse gases, farms, and vehicles.
    """

    @property
    def code_refs(self) -> dict[str, object]:
        return {
            'ureg.py': 'https://github.com/jaberg/planzero/blob/main/planzero/ureg.py',
        }

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Time Series': 'Time Series are associated with a unit of measure',
            'Dynamic Element': 'Dynamic Element initialization logic configures the unit of measure of each time series in a simulation',
        }


class Dynamic_Element(GlossaryTerm):
    """A PlanZero modelling data structure for representing a modelling
    assumption, and defining one or more
    {{lref("Time Series", "time series")|safe}}.
    A dynamic element is expected to be a Python code object, that is
    a subclass of either a
    {{lref("Strategy")|safe}} or a
    {{lref("Barrier")|safe}}.
    <p>
    Dynamic elements provide two important kinds of logic for definining time series:
    initialization logic and recurrence logic.
    Initialization logic creates time series, sets their unit of measure,
    their interpolation mode, and any initial values each time series should take.
    The Initialization logic of a dynamic element
    cannot refer to the values of other time series that aren't created by that element.
    The recurrence logic of a dynamic element
    can use the values of other time series to update its own time series for
    times up to and including the current simulation time.
    </p>

    """

    @computed_field
    def as_discussed_in_posts(self) -> list[tuple[object, str, str]]:
        return [
            (blog.Glossary(), '#dynelem', "see section on Dynamic Elements"),
        ]

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Time Series': 'timeseries are the inputs and outputs of dynamic elements',
            'Model': 'models are sets of dynamic elements',
            'Simulation': 'dynamic elements provide the initialization and recurrence logic to define time series by simulation',
        }

    @property
    def code_refs(self) -> dict[str, object]:
        return {
            'Dynamic Element base class': DynamicElement,
        }


class Strategy(GlossaryTerm):
    """<p>A Strategy is a {{lref("Dynamic Element", "dynamic element")|safe}}
    that is meant to represent an inititive that could be undertaken within
    a model.
    Strategies are optional; they can be omitted without sacrificing the validity of
    a model.
    Indeed, simulating models with and without a strategy is how
    strategies are evaluated in the Simulations on the PlanZero site.
    This is called Ablative Analysis.
    </p>
    <p>I adapt the term from {{lref("EGFS")|safe}} 
    where Strategies were defined as "broad activities required to achieve a goal, create a critical condition, or overcome a barrier."</p>
    """ 

    @computed_field
    def as_discussed_in_posts(self) -> list[tuple[object, str, str]]:
        return [
            (blog.Glossary(), '#strategy', "see modelling sub-section on strategies"),
        ]

    @property
    def code_refs(self) -> dict[str, object]:
        return {
            'Strategy base class': strategies.Strategy2,
            'Example Strategy: Scale Bovaer': strategies.strategy2.Scale_Bovaer,
        }

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Ablative_Analysis': 'A strategy is evaluated in the context of a model by comparing scenarios with and without the strategy',
            'Simulation': 'The construction of a scenarios from a model, sometimes including the ablative analysis of strategies',
            'Simulations_Section': 'the <a href="/models/#sim">Simulation-based models</a> of the site features the strategies of each scenario',
            'Barrier': 'Barriers are the other kind of dynamic element in a model, which define KPIs, and relate the time series of a model to one another',
            'Model': 'Models are sets of dynamic elements, which may include Strategies',
            'Dynamic_Element': 'at a computational level, a strategy is a type of dynamic element',
            'EGFS': 'The Executive Guide to Facilitating Strategy provided the term definition adapted here in PlanZero',
        }



class Barrier(GlossaryTerm):
    """<p>A Barrier is a 
    {{lref("Dynamic Element", "dynamic element")|safe}}
    that is not optional, that is, one whose omission would sacrifice the
    validity of a model.</p>
    <p>
    PlanZero terminology may feel a bit cynical in this regard, but in
    this terminology, all of the following would qualify as barriers:
    <ul>
    <li>regulations</li>
    <li>the life cycle of assets</li>
    <li>consumer behaviour</li>
    <li>the length of research and development cycles</li>
    <li>return on investment requirements</li>
    <li>the predictions of climate models</li>
    <li>the laws of physics</li>
    </ul>
    </p>
    <p>I borrow the term from {{lref("EGFS")|safe}} but its
    use in a computational modelling framework is, admittedly, a stretch.
    </p>
    """ 

    @computed_field
    def as_discussed_in_posts(self) -> dict[str, str]:
        return [
            (blog.Glossary(),
             '#barrier',
             'see section "Barriers: Connecting Strategies to Outcomes"'),
        ]

    @property
    def code_refs(self) -> dict[str, object]:
        return {
            'Barrier base class': barriers.Barrier,
            'Example Barrier class: Bovaer Adoption Limit': cattle.Bovaer_Adoption_Limit,
        }

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Strategy': 'a dynamic element designed to change the input to one or more barriers',
            'Model': 'a set of dynamic elements, including barriers, that make a prediction',
            'NIR_Model': "a model of Canada's future emissions",
            'Simulation': 'the computation of scenarios from models',
            'EGFS': "PlanZero adopts the Barrier term and definition from The Executive Guide to Facilitating Strategy.",
        }


class IPCC_Sector_Contribution(GlossaryTerm):
    """The emissions associated with an IPCC Sector are generally
    computed as coming from one or more sources, each of which
    is associated by PlanZero with the product of a driver and an emission factor.
    This product is called a IPCC Sector Contribution.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'NIR_Model': "a model of Canada's future emissions",
            'Emission Factor': 'the factor of proportionality between a driver and an IPCC Sector Contribution (generally a time series)',
            'Driver': 'A quantity of activity or physical stock that causes emissions (generally a time series)',
            'IPCC Sector': (
                'the emissions for an IPCC Sector are calculated'
                ' by summing one or more IPCC Sector Contributions'),
            'KPI': 'IPCC Sector Contributions are Derived KPIs',
        }


class Emission_Factor(GlossaryTerm):
    """<p>
    An emission factor is a constant of proportionality between
    an emission driver and the amount of some emitted greenhouse gas.
    </p>"""

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'KPI': "an emission factor is one of PlanZero's base KPI types",
            'NIR_Model': "a model of Canada's future emissions",
            'Driver': "a quantity of activity or physical stock that causes emissions in proportion to one or more emission factors",
            'Greenhouse Gas': "an emission factor is a constant of proportionality to one greenhouse gas",
            'Emissions': "the result of multiplying an emission factor by a driver",
        }


class Driver(GlossaryTerm):
    """<p>A driver, in the context of a model,
    is a quantity of activity or of physical stock,
    that is typically associated with a province or territory.
    {{lref("Barrier")|safe}} dynamic elements can register
    {{lref("Time Series", "time series")|safe}} as drivers.
    Drivers are meant to drive emission KPIs via emission factors,
    and subsidy KPIs via subsidy factors.
    </p>

    <p>
    IPCC sector emissions are typically a sum of products (e.g. amount of activity
    multiplied by emissions per unit of activity,
    summed over one or more activities that count toward the category);
    in this typical case, each of the emission-contributing activities
    corresponds
    to a driver, and
    the emission of each greenhouse gas per unit of activity is referred to as
    an {{lref("Emission Factor")|safe}}.
    </p>
    <p>
    An Driver is a time series, whose unit is typically an amount of
    activity (in whatever unit is appropriate for the emissions source) totalled
    per year in a non-interpolating time series.
    </p>"""

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'NIR_Model': 'a model of national emissions',
            'Emission Factor': 'the constant of proportionality of a driver to emissions',
            'Subsidy Factor': 'the constant of proportionality of a driver to subsidy',
            'Barrier': 'Typically, Barriers are the dynamic elements that define Drivers',
        }



class Subsidy_Factor(GlossaryTerm):
    """<p>A subsidy factor is a constant of proportionality between
    a {{lref("Driver")|safe}} and a requirement of a real or hypothetical funding program.
    Subsidy Factors are Base KPIs of NIR Models.
    </p>
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'NIR_Model': 'a model of national emissions, and of subsidy requirements',
            'Driver': 'A level of activity or physical stock that drives a subsidy requirement by multiplication with a subsidy factor',
            'Subsidy_Requirement': 'the annual subsidy amount associated with a driver',
            'KPI': 'A standard metric associated with an NIR Model, subsidy factors are Base KPIs',
        }


class Subsidy_Requirement(GlossaryTerm):
    """A subsidy requirement is derived KPI representing an amount of
    funding required for something on an annual basis.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'NIR_Model': 'a model of national emissions, and also of subsidies',
            'Driver': 'A level of activity or physical stock that drives a subsidy requirement',
            'Subsidy_Factor': 'The product of a Driver and a Subsidy Factor is a Subsidy Requirement',
            'IPCC_Sector_Contribution': "the analogous term to a Subsidy Requirement in the calculation of emissions",
            'KPI': 'Subsidy Requirements are a Derived KPI',
        }


class Subsidy_Program(GlossaryTerm):
    """A Subsidy Program is an actual or hypothetical government program,
    at any level of government, that directly or indirectly funds a set of
    related activities.
    A subsidy program is to total subsidies, 
    as an IPCC Sector is to total national emissions.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'NIR_Model': 'a model of national emissions and subsidies',
            'Driver': 'A level of activity or physical stock that drives a subsidy requirement',
            'Subsidy_Factor': 'The product of a Driver and a Subsidy Factor is a Subsidy Requirement',
            'IPCC_Sector_Contribution': "the analogous term to a Subsidy Requirement in the calculation of emissions",
            'KPI': 'Subsidy Requirements are a Derived KPI',
            'Subsidy_Requirement': 'Subsidy Program totals are calculated by adding up one or more Subsidy Requirements, associated with different Drivers',
        }


class Critical_Success_Factor(GlossaryTerm):
    """<p>A Critical Success Factor
    is a necessary condition of a KPI to achieve an objective.
    For example, the KPI must be within a certain range of values
    for any or all of some period of time.
    The term Critical Success Factor has a
    <a href="https://en.wikipedia.org/wiki/Critical_success_factor">long history</a>.
    PlanZero's use of the term is based on the definition from {{lref("EGFS")|safe}}, 
    which is a "key conditions that must be created to achieve one or more objectives."
    </p>
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'EGFS': 'The Executive Guide to Facilitating Strategy',
            'IPCC_Sector_Contribution': 'an emissions contribution to an IPCC Sector',
            'KPI': 'Key Performance Indicators are the time series contemplated by Critical Success Factors',
        }

    @computed_field
    def as_discussed_in_posts(self) -> dict[str, str]:
        return [
            (blog.Glossary(),
             '#formalizing_csfs',
             'see section "Emissions and Costs as Critical Success Factors"'),
        ]

    @computed_field
    def aka(self) -> list[str]:
        return ['CSF']


class Key_Performance_Indicator(GlossaryTerm):
    """
    A Key Performance Indicator (KPI) in PlanZero is a time series that
    is registered to participate in the calculation of emissions or subsidies.
    A <i>base KPI</i> is a driver, an emission factor, or a subsidy factor.
    A <i>derived KPI</i> is the product of a driver with an emission factor,
    or the product of a driver with a subsidy factor, or the result of summing
    together other derived KPIs toward e.g. national totals.
    """

    @computed_field
    def aka(self) -> list[str]:
        return ['KPI', 'Base KPI', 'Derived KPI']

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'CSF': "target values for target times of a KPI time series, in order to achieve an objective",
            'NIR_Model': "a model of Canada's future emissions",
            'Driver': 'Base KPI in emissions calculation, referring to a level of activity or physical stock',
            'Emission Factor': 'Base KPI in emissions calculations, referring to the factor of proportionality between a driver and an IPCC Sector Contribution',
            'IPCC_Sector_Contribution': (
                'The Derived KPIs added together to'
                ' calculate various emissions totals'),
            'Time Series': 'KPIs are time-varying quantities'
        }



class NIR_Model(GlossaryTerm):
    """
    An NIR model is a model that generates
    emissions KPIs corresponding to the emission amounts in a
    National Inventory Report.
    </p><p>
    NIR Models are also the fully-featured models featured
    in the PlanZero <a href="/models/#sim/">Simulation-based models</a> section.
    In addition to KPIs relating to emissions, these models
    include KPIs relating to subsidy programs.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'NIR': 'A National Inventory Report published by the ECCC with historical Canadian emissions data, organized by IPCC Sector',
            'Model': 'Any set of dynamic elements makes a model, NIR models have dynamic elements that declare drivers and emission factors',
            'Driver': 'A level of activity or physical stock that drives emissions',
            'Emission Factor': 'A factor of proportionality of how much of a greenhouse gas is emitted by a driver',
            'IPCC_Sector': 'NIR models calculate emissions for each IPCC Sector',
            'IPCC_Sector_Contribution': (
                'the emissions calculated by an NIR model are sums'
                ' across per-sector contributions'),
            'KPI': 'an emissions contribution to an IPCC Sector',
            'Simulations_Section': 'NIR models are analyzed in this section of the PlanZero site',
            'Subsidy_Program': 'NIR models calculate subsidy requirements for each Subsidy Program',
            'Subsidy_Requirement': (
                'the total subsidies calculated by an NIR model are sums'
                ' across per-program requirements'),
        }


class Model(GlossaryTerm):
    """A model, in PlanZero, is a set of time series and dynamic elements that
    can be simulated to generate one or more possible scenarios.
    A model can be either deterministic or stochastic. 
    """

    @computed_field
    def as_discussed_in_posts(self) -> list[tuple[object, str, str]]:
        return [
            (blog.Glossary(),
             '#introduction',
             'see sections "<a href="/blog/2026-04-19-glossary/#computation">Computation and Simulation</a>" '
             'and "<a href="/blog/2026-04-19-glossary/#modelling">Modelling National Emissions</a>"'),
        ]

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'NIR_Model': """Model of Canada's national emissions in the style
            of the National Inventory Reports submitted to UNFCCC""",
            'Deterministic_Model': "A model that corresponds to a unique scenario",
            'Stochastic_Model': "A model that corresponds to a distribution over possible scenarios",
            'Simulation': (
                "Simulation is the building of a scenario with the"
                " initialization and recurrence logic in a model's dynamic"
                " elements"),
            'Scenario': (
                'A scenario is the set of time series that results from'
                ' simulating a model'),
        }


class Stochastic_Model(GlossaryTerm):
    """A stochastic model corresponds to a distribution over possible
    scenarios.
    Simulating a stochastic model using pseudo-random numbers draws
    a sample from this distribution."""

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'NIR_Model': """Model of Canada's national emissions in the style
            of the National Inventory Reports submitted to UNFCCC""",
            'Deterministic_Model': "A model that corresponds to a unique scenario",
            'Model': "A set of dynamic elements that can be simulated",
            'Simulation': 'The procedure for converting a stochastic model to one or more scenarios'
        }

    @computed_field
    def aka(self) -> list[str]:
        return ['Probabilistic Model']


class Deterministic_Model(GlossaryTerm):
    """A deterministic model is a model that corresponds to a specific
    scenario, and has no randomness.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'NIR_Model': """Model of Canada's national emissions in the style
            of the National Inventory Reports submitted to UNFCCC""",
            'Model': "A set of dynamic elements that can be simulated",
            'Stochastic_Model': "A model that corresponds to a distribution over possible scenarios",
        }


class Simulations_Section(GlossaryTerm):
    """The Simulations section of the planzero.ca website:
    <a href="/models/#sim">https://planzero.ca/models/</a>"""

    # XXX this is now the "Models Tab"

    # XXX Change term Section -> Tab, because individual pages have "Sections"

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'NIR_Model': """Models of Canada's national emissions provide content for the Simulations Section""",
            'Model': "A set of dynamic elements that can be simulated",
            'Simulation': 'The algorithm for generating the data for the Simulations Section',
            'Ablative_Analysis': (
                'For each simulation, each strategy in the '
                'Simulations Section is evaluated by comparing scenarios'
                ' with and without that strategy'),
        }

class About_Section(GlossaryTerm):
    """<p>The "About" section of the planzero.ca website:
    <a href="/about/">planzero.ca/about</a></p>"""

    @computed_field
    def aka(self) -> list[str]:
        return ['About Page']


class Scenario(GlossaryTerm):
    """A scenario is a set of time series covering a common time period.
    Typically in PlanZero it is the result of simulating a model.
    """

    @computed_field
    def as_discussed_in_posts(self) -> list[tuple[object, str, str]]:
        return [
            (blog.Glossary(), '#scenario', "see simulation sub-section on scenarios"),
        ]

    @computed_field
    def aka(self) -> list[str]:
        return ['Trajectory', 'Rollout (noun)']

    @property
    def code_refs(self) -> dict[str, object]:
        from . import base
        return {
            'Scenario is called "State" in the source code': base.State,
        }

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Deterministic_Model': "A model that corresponds to a unique scenario",
            'Stochastic_Model': "A model that corresponds to a distribution over possible scenarios",
            'Rollout': "A scenario is sometimes called a model rollout",
            'Time Series': "A time series is a data structure representing a time-varying quantity, and a Scenario corresponds to a set of them",
            'Simulation': 'The construction of a Scenario from a model',
            #'Simulations_Section': 'the <a href="/models/#sim">Simulation-based models section</a> of the site analyzes and compares scenarios',
        }


class Model_Metric(GlossaryTerm):
    """A formula, procedure or rule for associating
    a single number to a {{lref("Model")|safe}}.
    For example, NIR models could be evaluated in terms of
    prediction accuracy."""

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Model': "A set of assumptions that can be simulated to produce one or more scenarios",
            'NIR Model': "NIR models have standard KPIs, which could support standard metrics",
            'KPI': (
                'A Key Performance Indicator is a standard'
                ' time series associated with multiple models'),
        }


class Git(GlossaryTerm):
    """<p><a href="https://git-scm.com/">Git</a> is a "free and open source
    distributed version control system designed to handle everything from
    small to very large projects with speed and efficiency."</p>
    <p>PlanZero is developed with git, and hosted on
    {{lref("GitHub")|safe}} as a 
    <a href="https://github.com/jaberg/planzero">public code repository</a>.</p>
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'GitHub': "git hosting for PlanZero",
            'Git Branch': "maintain versions of code with git branches",
            'Git Commit': "record source file changes to a git branch",
            'Git Merge': "merge changes from one branch into another",
        }

class Git_Merge(GlossaryTerm):
    """A git "merge" is the operation of combining two
    {{lref("Git Branch", "branches")|safe}} together into a coherent set
    of files, extending one branch with
    {{lref("Git Commit", "commits")|safe}} from the other.
    Sometimes some fixing up is required, for example if both branches have
    changed the same part of the same file, but usually merging is mostly
    automatic. Merging is how people can contribute their changes
    back to the {{lref("Main Branch", "main branch")|safe}}.
    Merging can be done locally on a development computer,
    or on {{lref("GitHub")|safe}} via a {{lref("GitHub Pull Request", "pull request")|safe}}.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Git_Branch': "a merge operation joins two branches",
            'GitHub_Pull_Request': "a request to merge branches made via GitHub",
        }

class Git_Branch(GlossaryTerm):
    """A git branch is a way of tracking a single version of a set of files.
    Technically, it is a sequence of git commits (sometimes a graph)
    to source files leading from the initially empty project
    to some version that's full of files.
    The PlanZero site is populated by deploying the "main" branch.
    Anyone is welcome to suggest changes to main by creating a pull request on github,
    requesting that the main branch merge changes from another branch that
    they've created.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Main_Branch': "the branch from which the PlanZero site is generated",
            'GitHub_Pull_Request': "a request to merge branches made via GitHub",
            'Git_Commit': "change a branch by adding a commit, which incorporates changes to local files",
            'Git_Merge': "changes from one branch can be merged into another",
            'Git_Graph': "the commits and merges to a branch define a directed acyclic graph with a single source node (the beginning of development) and a single sink node (the current state of the branch)",
        }


class Main_Branch(GlossaryTerm):
    """PlanZero on GitHub generally has multiple git branches.
    The "main branch" is special, in that it is the one used to deploy the
    {{lref("PlanZero Site", "PlanZero site")|safe}}.
    New improvements to the codebase should be {{lref("Git Merge", "merged")|safe}}
    to the main branch.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Git_Branch': "a version of a codebase",
            'GitHub_Pull_Request': (
                "a request to merge one branch into another is the"
                " preferred way to update the main branch of a project,"
                " because it prompts discussion and consideration"),
        }


class PlanZero_Site(GlossaryTerm):
    """The PlanZero site is the website hosted at <a
    href="https://planzero.ca">https://planzero.ca</a>.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Main_Branch': "the code from which the site is generated",
            'PlanZero': "A collective term for the site, the supporting code, and the project to develop them",
            "GitHub_Repository": "The development hub for PlanZero on GitHub",
        }


class GitHub_Repository(GlossaryTerm):
    """A GitHub code repository, or repo, is a GitHub-defined entity,
    it is a major point of configuration,
    especially for billing and permissions.
    A repo contains all of the code and files
    for one or more git branches.
    A repo may be created from some standard initial state (such as being empty),
    or it may be created by forking another repo.
    My ("James Bergstra", aka "jaberg") PlanZero repo is <a
    href="https://github.com/jaberg/planzero">here</a>.
    If a repo is not created by forking, then it may be called a "Source Repo"
    or "Upstream Repo".
    """

    @computed_field
    def aka(self) -> list[str]:
        return ['GitHub Repo', 'repository', 'repo']

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'GitHub': "the site that hosts GitHub code repositories",
            'Git': "the version control system upon which GitHub operates",
            'GitHub_Fork': "A repo can be forked to create a downstream copy of an upstream repo"
        }

class GitHub_Fork(GlossaryTerm):
    """Public GitHub {{lref("repo", "repos")|safe}} can be "forked" by users
    who wish to make and publish their own modifications. If you have a GitHub
    account, you can <a href="https://github.com/jaberg/planzero/fork">fork
    the PlanZero codebase</a>, to make
    {{lref("Git Branch", "branches")|safe}} with your own modifications,
    and then submit them back as
    {{lref("GitHub Pull Request", "pull requests")|safe}}.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Git_Branch': "a codebase version within a fork",
            'GitHub_Pull_Request': "a request to merge branches, possibly across forks",
            'GitHub_Repository': "A GitHub repo may be a fork of another repo",
        }


class Git_Commit(GlossaryTerm):
    """A "commit" is an increment of change to a codebase, across one or more
    changed files. It is a node in the graph of changes that make up a git
    repository.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Git_Branch': "a named graph of commits corresponding to a single version of set of files",
        }

    @computed_field
    def aka(self) -> list[str]:
        return ['Commit']

class Git_Graph(GlossaryTerm):
    """A set of commits and merges that build on one another form a graph
    representing all of the development on a project.
    Try visualizing the graph for any git project by using purpose-built visual tools, 
or running a command such as <pre>git log --all --decorate --oneline --graph</pre>
    """

    @computed_field
    def aka(self) -> list[str]:
        return ['Git History']

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Git_Commit': "a node of a git [change] graph",
            'GitHub Repo': "a git graph hosted on GitHub",
            'Git_Branch': "a named subgraph of ancestors of a particular commit",
        }


class GitHub(GlossaryTerm):
    """<p>GitHub (<a href="https://github.com/">site</a>, <a href="https://en.wikipedia.org/wiki/GitHub">wikipedia</a>) is a web service for using the git version
    control system over the internet to collaborate on software projects. Circa 2023, it was the world's largest source code host, with over 100 million developers, and 420 million code repositories.</p>
    """

    @computed_field
    def aka(self) -> list[str]:
        return ['GH']

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Git': "version control software upon which GitHub is based",
            'GitHub_Repository': "git branches corresponding to one or more versions of a project's code and files",
            'GitHub_Fork': "a copy of a GitHub repository into another user's GitHub account",
            'GitHub_Issue': "a future-work item on the PlanZero project",
        }


class GitHub_Issue(GlossaryTerm):
    """GitHub issues, such as the <a href="https://github.com/jaberg/planzero/issues">PlanZero issues</a>,
    are notices / reminders of future work.
    Issues are used differently by different projects, some projects don't use
    GitHub's issues at all.
    PlanZero uses GitHub issues to record a variety of ideas and To-Do items,
    but in particular, those that have been mentioned in posts.
    """

    @computed_field
    def aka(self) -> list[str]:
        return ['gh issue']

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'GitHub': """Code hosting for PlanZero""",
            'Post': """A report on a piece of work on PlanZero, sometimes linking to GitHub issues corresponding to next steps beyond the scope of the post itself""",
        }


class GitHub_Pull_Request(GlossaryTerm):
    """One of GitHub's main features is a web interface for people
    to suggest code changes to each other. Pull requests
    enable asynchronous loosely-coupled development over the net by
    giving contributors and developers a place to talk about the changes,
    and possibly even {{lref("Git Merge", "merge")|safe}} those changes.
    See <a href="https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests">Github documentation</a> for full description of this capability.
    """

    @computed_field
    def aka(self) -> list[str]:
        return ['pull request', 'PR']

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Git_Branch': "a pull request is a request to merge two branches",
            'Main_Branch': "submit a pull request to this branch when a new development is ready",
            'GitHub_Fork': "typically a pull request represents a request to merge code from a branch in one fork (maintained by one person) into a branch on another fork (maintained by another person)"
        }


class EGFS(GlossaryTerm):
    """<p>Wilkinson, M., <i>The Executive Guide to Facilitating Strategy: featuring the Drivers Model</i>, Atlanta: Leadership Strategies Publishing, 2011. <a href="https://www.amazon.ca/Executive-Guide-Facilitating-Strategy/dp/0972245812">Amazon</a></p>
    <p>
    I've appropriated (hopefully not misappropriated) terms
    of implementation planning ({{lref("Critical Success Factor")|safe}},
    {{lref("Barrier")|safe}},
    {{lref("Strategy")|safe}})
    as defined in this book, to structure the visualization of model simulation.
    I have also tried to insert aspects of directional strategy (mission,
    vision, guiding principles, positioning) in the project overview on the
    <a href="/about/">About page</a>.
    </p>
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Strategy': "an ablatable model element, used to represent a strategy in a model",
            'Barrier': "a non-ablatable model element, used to model the interactions of time series with KPIs",
            'CSF': "target values for target times of a KPI time series, in order to achieve an objective",
            'KPI': "a time series registered to participate in emissions or subsidy calculations",
        }

    @computed_field
    def as_discussed_in_posts(self) -> list[tuple[object, str, str]]:
        return [
            (blog.Glossary(), '#modelling', "see section on Modelling National Emissions"),
        ]

class NIR(GlossaryTerm):
    """A National [Greenhouse Gas] Inventory Report,
    is published annually by Environment and Climate Change Canada, and
    submitted to the UNFCCC Secretariat.
    Specific reports are referred to as e.g. NIR-2023, NIR-2024, NIR-2025,
    an so on in PlanZero.
    The first report was <a href="https://publications.gc.ca/site/eng/9.506002/publication.html">NIR-2004</a>.
    The most recent as-of writing is <a href="https://www.canada.ca/en/environment-climate-change/services/climate-change/greenhouse-gas-emissions/sources-sinks-executive-summary-2026.html">NIR-2026</a>.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            "UNFCCC": "The United Nations Framework Convention on Climate Change is the primary international treaty aimed at stabilizing greenhouse gas concentrations in the atmosphere",
            "IPCC Sector": "PlanZero term for the most granular categories used in NIR documents",
            "CNZEAA": (
                "The Canadian Net-Zero Accountability Act commits ECCC"
                " to problishing National Inventory Reports, setting"
                " emissions targets, and developing a plan to hit those targets"),
        }


class National_Greenhouse_Gas_Inventory(GlossaryTerm):
    """Canada's
    <a
    href="https://www.canada.ca/en/environment-climate-change/services/climate-change/greenhouse-gas-emissions/inventory.html">National
    Greenhouse Gas Inventory</a> tracks emissions
    across Canada, and is the basis for the annual
    {{lref("NIR", "National Inventory Reports")|safe}}.
    """

    @computed_field
    def aka(self) -> list[str]:
        return ['NGGI']

    @property
    def see_also(self) -> dict[str, str]:
        return {
            "ECCC": "Environment and Climate Change Canada maintains and publishes the NGGI",
            "NIR": "National Inventory Reports are prepared from the NGGI",
        }


class National_Energy_Use_Database(GlossaryTerm):
    """<p>The <a href="https://oee.nrcan.gc.ca/corporate/statistics/neud/dpa/data_e/databases.cfm">National Energy Usage Database</a>, is
    developed and maintained by {{lref("Natural Resources Canada")|safe}}.
    It reflects energy usage and movement by economic sector.
    </p>
    <p>
    PlanZero uses this data to quantify drivers of emissions in certain
    {{lref("IPCC Sector", "IPCC sectors")|safe}}.
    </p>"""

    @computed_field
    def aka(self) -> list[str]:
        return ['NEUD']

    @property
    def code_refs(self) -> dict[str, object]:
        from . import neud
        return {
            'NEUD access, neud.py': neud,
        }

    @computed_field
    def as_discussed_in_posts(self) -> list[tuple[object, str, str]]:
        return [
            (blog.IPCC_HeavyDutyDieselVehicles(),
             '#quant',
             "see section Estimating Emissions from the National Energy Use Database"),
            (blog.IPCC_MCS_LightGasolineCarsAndTrucks(),
             '#neud',
             "see section An Estimator Based on Data from the National Energy Use Database"),
            (blog.IPCC_SCS_Residential(),
             '#estimation',
             "see section Estimating Residential Stationary Combustion Emissions"),
        ]

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'NRCan': "Natural Resources Canada develops and publishes the NEUD",
        }

class Statistics_Canada(GlossaryTerm):
    """<a href="https://www.statcan.gc.ca/en/start">Statistics Canada</a>,
    Canada's national statistical office.
    Much of the data used in PlanZero is drawn from Statistics Canada tables,
    downloaded via open-source project
    <a href="https://github.com/ianepreston/stats_can">stats_can</a>.
    """
    
    @computed_field
    def aka(self) -> list[str]:
        return ['StatsCan']

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'NGGI': 'The National Greenhouse Gas Inventory draws on data products from Statistics Canada',
            'NEUD': 'The National Energy Use Database draws on data products from Statistics Canada',
        }


class Natural_Resources_Canada(GlossaryTerm):
    """<p><a href="https://natural-resources.canada.ca/">Natural Resources
    Canada</a> (sometimes, NRCan) is a federal ministry
    "<i>Committed to improving the quality of
    life of Canadians by ensuring the country’s abundant natural resources are
    developed sustainably, competitively and inclusively.</i>".
    </p>
    """

    @computed_field
    def aka(self) -> list[str]:
        return ['NRCan']

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'ECCC': 'Environment_and_Climate_Change_Canada is a peer federal ministry',
            'NEUD': "NRCan maintains and publishes the National Energy Use Database",
        }


class Environment_and_Climate_Change_Canada(GlossaryTerm):
    """<p><a href="https://www.canada.ca/en/environment-climate-change.html">Federal ministry</a> "<i>protecting and conserving our natural heritage,
    predicting weather and environmental conditions, preventing and managing
    pollution, promoting clean growth and a sustainable environment for
    present and future generations.</i>"</p>
    <p>
    ECCC, among many activities and responsibilities,
    prepares the annual {{lref("NIR")|safe}}, and maintains
    the federal plan for <a href="https://www.canada.ca/en/services/environment/weather/climatechange/climate-plan/net-zero-emissions-2050.html">Net-Zero by 2050</a>.
    </p>
    """

    @computed_field
    def aka(self) -> list[str]:
        return ['ECCC', '"E-triple-C"']

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Natural_Resources_Canada': 'peer federal ministry',
            "NIR": "National Inventory Reports are prepared by ECCC",
            "NGGI": "National Greenhouse Gas Inventory is maintained by ECCC",
        }

class Net_Zero(GlossaryTerm):
    """Net-zero is a hypothetical state, in which
    greenhouse gas {{lref("Emissions", "emissions")|safe}} across the sectors of an economy, average
    out to zero.
    The goal of the PlanZero project is to explore models of how Canada might
    achieve net-zero.
    """

    @computed_field
    def aka(self) -> list[str]:
        return ['Net-Zero']

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'ECCC': 'The ECCC is preparing a plan for Canada to reach net-zero',
            "Paris Agreement": "The Paris Agreement implores countries to reach net-zero by mid-century",
            "PlanZero": "PlanZero is this project, to model how Canada might reach net-zero",
        }


class PlanZero(GlossaryTerm):
    """This site and its supporting code, as well as the project of building
    and continually improving the site and its supporting code,
    is collectivey referred to as "PlanZero"
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'ECCC': 'The ECCC is preparing a plan for Canada to reach net-zero',
            "About Section": "Read more about this project on the About page",
            "GitHub_Repository": "The development hub for PlanZero on GitHub",
            "PlanZero_Site": "The site you probably used to access this page",
        }


class International_Panel_on_Climate_Change(GlossaryTerm):
    """<a href="https://www.ipcc.ch/">International Panel on Climate Change
    (IPCC)</a> is
    the United Nations body for assessing the science related to climate
    change.
    Among many activities, and across many working groups,
    the IPCC maintains the methodology for emissions estimation
    that is to be used by signatories of the Paris Agreement.
    Canada uses this methodology to prepare its annual {{lref("NIR")|safe}}.
    """

    @computed_field
    def aka(self) -> list[str]:
        return ['IPCC']

    @property
    def see_also(self) -> dict[str, str]:
        return {
            "Paris Agreement": "The international agreement on reporting guidelines for greenhouse gases",
            "UNFCCC": "The United Nations Framework Convention on Climate Change is the primary international treaty aimed at stabilizing greenhouse gas concentrations in the atmosphere",
            "IPCC Sector": "PlanZero term for the most granular categories used in NIR documents",
            'NIR': "National Inventory Reports of Greenhous Gases, prepared according to UNFCCC guidelines, which were developed by the IPCC",
            "CNZEAA": (
                "The Canadian Net-Zero Accountability Act, legislation to uphold"
                " Canada's responsibilities under the Paris Agreement"),
        }


class IPCC_Sector(GlossaryTerm):
    """IPCC Sector is a PlanZero term, referring to
    an economic area for which Canada tracks emissions
    in the {{lref("NGGI")|safe}}.
    Each such sector is analyzed in accordance
    with IPCC emissions reporting guidelines,
    but the UNFCCC leaves the level of granularity of each sector
    up to each country, in order to report on its progress appropriately.
    In PlanZero posts,
    the term almost always refers to a sector that is not
    a subtotal of other sectors.
    PlanZero uses a set of 71 IPCC Sectors, which match
    the ones used in NIR-2025.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'NIR': "National Inventory Reports of Greenhous Gases, organized in terms of [what PlanZero refers to as] IPCC Sectors",
            'NGGI': 'The National Greenhouse Gas Inventory of emissions from which Canada produces annual reports',
            "IPCC": "The International Panel on Climate Change, a scientific body that produces climate reports for the UN, and reporting guidelines for Paris Agreement signatories",
            "Paris Agreement": "The international agreement on reporting guidelines for greenhouse gases",
            "UNFCCC": "The United Nations Framework Convention on Climate Change is the primary international treaty aimed at stabilizing greenhouse gas concentrations in the atmosphere",
            "NIR_Model": "A PlanZero model that simulates emission amounts for IPCC Sectors",
        }


class UNFCCC(GlossaryTerm):
    """United Nations Framework Convention on Climate Change is the primary
    international treety aimed at stabilizing greenhouse gas concentrations in the atmosphere.
    It has as an objective to limit global temperature rise to "well under 2{{degrees}}C".
    It defined the annual "Conference Of the Parties (COP)" meetings to assess progress and
    negotiate new treaties and initiatives, such as the Paris Agreement, which is based
    upon the UNFCCC.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            "IPCC": "The International Panel on Climate Change, a scientific body that provides scientific input for ongoing UNFCCC initiatives",
            "CNZEAA": (
                "The Canadian Net-Zero Accountability Act, legislation to uphold"
                " Canada's responsibilities under the Paris Agreement"),
            'Paris Agreement': (
                "The international agreement that signatories would"
                " report emissions in standard ways, and set"
                " Nationally Determined Contribution (NDC) targets"),
        }


class Paris_Agreement(GlossaryTerm):
    """The Paris Agreement is an international treaty committing signatories
    to report emissions in a standard way (as per IPCC recommendation),
    to set their own emissions targets (Nationally Determined Contributions, NDCs),
    and to strengthen those targets over time, ideally achieving net-zero emissions
    by mid-century, and limiting global warming to 1.5{{degrees}}C.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            "IPCC": (
                "The International Panel on Climate Change,"
                " the scientific body that defined the reporting guidelines adopted"
                " by the Paris Agreement"),
            "CNZEAA": (
                "The Canadian Net-Zero Accountability Act, legislation to uphold"
                " Canada's responsibilities under the Paris Agreement"),
            "UNFCCC": (
                "Reports prepared by Paris Agreement signatories are"
                " delivered to the Secretariat of the"
                " United Nations Framework on Climate Change"),
        }


class CNZEAA(GlossaryTerm):
    """The Canadian Net-Zero Accountability Act legislates federal ministries and agencies
    to uphold Canada's obligations under the Paris Agreement.
    """

    @computed_field
    def aka(self) -> list[str]:
        return ['Canadian Net-Zero Accountability Act']

    @property
    def see_also(self) -> dict[str, str]:
        return {
            "ECCC": (
                "Environment and Climate Change Canada is responsible for"
                " delivering an annual NIR to the UNFCCC"),
            "UNFCCC": (
                "As a Paris Agreement signatory, Canada"
                " delivers an annual report to the Secretariat of the"
                " United Nations Framework on Climate Change"),
            "NIR": (
                "National Inventory Reports, published by ECCC, are one of the"
                " most critical data sources for PlanZero"
            )
        }


class Code(GlossaryTerm):
    """PlanZero is an open-source project, with [source] code on GitHub."""
    @property
    def see_also(self) -> dict[str, str]:
        return {
            "GitHub_Repository": "The development hub for PlanZero on GitHub",
            "Python": "PlanZero's source code is mostly written in the Python programming language",
        }


class Python(GlossaryTerm):
    """<p>PlanZero is implemented in the <a href="https://www.python.org/">Python programming language</a></p>"""

    @property
    def see_also(self) -> dict[str, str]:
        return {
            "GitHub_Repository": "The development hub for PlanZero on GitHub",
            "Code": "PlanZero's source code is mostly written in the Python programming language",
        }


class Emissions(GlossaryTerm):
    """The combined emissions of the seven greenhouse gases, often quantified
    in units of {{CO2e|safe}}.
    Canada estimates rates of emissions in the
    {{lref("National Greenhouse Gas Inventory")|safe}}.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Driver': 'a level of activity or physical stock, multiplied by an emission factor to calculate emissions',
            'Emission_Factor': 'a factor of proportionality between drivers and emissions',
            'KPI': 'Emissions are Derived KPIs',
            'NIR_Model': "a model of Canada's future emissions, calculated by summing up emissions across IPCC Sectors",
            "IPCC Sector": "the lowest levels of granularity in National Greenhouse Gas Inventory",
            "National Greenhouse Gas Inventory": "the data from which NIRs are generated"
        }



class Petrinex(GlossaryTerm):
    """<p>Petrinex <i>facilitates efficient, standardized, safe, and
    accurate management and exchange of "data of record" information essential
    for:
    <ol>
        <li>the administration of Alberta's, British Columbia's, Saskatchewan's, Manitoba's, and Indian Oil and Gas Canada's royalty frameworks and regulatory enforcement, and</li>
        <li>the commercial operation of the upstream, midstream and downstream petroleum sector.</li>
    </ol></i>
    In PlanZero, Petrinex is a data source for emissions data relating to
    the oil and gas sector in Alberta and Saskatchewan.
    </p>
    """
    
    @computed_field
    def as_discussed_in_posts(self) -> list[tuple[object, str, str]]:
        return [
            (blog.IPCC_SCS_OilAndGas_Exploration(),
             '#petrinex',
             "see section Estimating Stationary Combustion Emissions from Oil and Gas Extraction"),
            (blog.IPCC_VentingNaturalGas(),
             '#petrinex',
             "see section Estimating Venting Emissions"),
        ]

    @property
    def code_refs(self) -> dict[str, object]:
        from . import petrinex
        return {
            'Petrinex access, petrinex.py': petrinex,
        }


class Rollout(GlossaryTerm):
    """Rollout is a conventionally used term in modelling and simulation,
    which means one of a few related things depending on the context of usage.
    A Scenario is constructed from a model by executing the
    initialization and recurrence logic of the model's dynamic elements.
    The recurrence logic is generally executed repeatedly, extending
    time series one time step at a time, until all of the time series
    in all of the dynamic elements are extended to at least the
    end-time of the simulation. This proces of repeated extension
    is sometimes called "rolling out" the model, perhaps like a carpet
    that unrolls to take up more space and reveal the intracacies of its
    design. Sometimes "rollout" is used as a noun, to refer to the
    resulting scenario, rather than the algorithm that built it.
    """

    @computed_field
    def as_discussed_in_posts(self) -> list[tuple[object, str, str]]:
        return [
            (blog.Glossary(), '#simulation', "see sub-sections about simulation and scenarios"),
        ]

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Model': 'a set of dynamic elements that can be simulated (rolled out)',
            "Simulation": "The process of building a scenario (rolling out the model)",
            "Scenario": (
                "The set of time series that result"
                " from simulating a model (a scenario may be called a rollout)"),
            'Time Series': "the data structures being rolled out, or making up a rollout",
            'Dynamic Element': "the model elements being rolled out",
        }


class Simulation(GlossaryTerm):
    """Simulation in PlanZero refers to the
    the algorithm for building one or more Scenarios from a Model.
    Models comprise dynamic elements, which provide initialization
    and recurrence logic for defining time series.
    Simulation is the algorithm of using initialization logic and then
    iterative recurrence logic to extend time series forward in time (roll them out)
    until all of the time series cover a required time interval.
    </p>
    <p>The strategies, in a model with strategies, are evaluated
    for PlanZero's Simulations Section by an Ablative Analysis. That is to
    say, the model with all of the strategies is simulated, and the model
    without a given strategy is also simulated, and the difference in resulting scenarios
    is shown on the site as the impact of the strategy.
    Depending on context, simulation refer to either the creation of these individual scenarios,
    or the creation of all of the scenarios necessary for ablative analysis.
    </p>
    <p>A deterministic model corresponds to a single scenario,
    whereas a non-deterministic model corresponds to a distribution over possible scenarios.
    Simulation of a non-deterministic model means sampling from this distribution over scenarios
    by generating scenarios using pseudo-random numbers.
    Ablative analysis of a strategy in a non-deterministic model means sampling from
    the distributions both with and without the strategy. Simulation, in this context,
    may refer to the computation of any or all of the scenarios involved.
    </p>
    """

    @computed_field
    def as_discussed_in_posts(self) -> list[tuple[object, str, str]]:
        return [
            (blog.Glossary(), '#simulation', "see sub-section about simulation"),
        ]

    @property
    def code_refs(self) -> dict[str, object]:
        from . import sim
        return {
            'The <code>simulation_result(...)</code> function implements PlanZero simulation': sim.simulation_result,
        }

    @computed_field
    def aka(self) -> list[str]:
        return ['Rollout (verb)']

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Model': 'a set of dynamic elements that can be simulated (rolled out)',
            'Deterministic_Model': 'a model that corresponds to a single scenario',
            'Stochastic_Model': 'a model that corresponds to a distribution over scenarios',
            'Ablative_Analysis': 'the evaluation of a strategy by comparing scenarios with and without the strategy',
            "Scenario": (
                "The set of time series that result"
                " from simulating a model"),
            'Time Series': "the data structures built up by simulation",
            'Dynamic Element': "the model elements providing initialization and recurrence logic",
            "Rollout": "a synonym for either simulation or scenario, depending on context",
            "Simulations_Section": "Pages on the PlanZero site showing simulation results",
        }

class Ablative_Analysis(GlossaryTerm):
    """
    Ablative analysis in PlanZero is the study of models with and without
    certain elements (namely strategies), in order to characterize the impact
    of those elements on the behaviour of the whole model.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Strategy': 'Strategies within models are are ablated in order to assess their impact.',
            'Simulation': 'Ablative analysis is part of model simulation.',
            'Simulations_Section': 'Strategies within simulations are evaluated and characterized based on an ablative analysis.',
        }


class Post(GlossaryTerm):
    """A post on this site (see e.g. <a href="/posts/">https://planzero.ca/posts/</a>).
    </p><p>
    The <a href="/about/">About</a> page
    includes some guidelines for post content.
    """

    @computed_field
    def as_discussed_in_posts(self) -> list[tuple[object, str, str]]:
        return [
            (blog.About(), '#posting-policy', "see sections on Guidance Re: Posts"),
        ]


class Draft_Status(GlossaryTerm):
    """A PlanZero Post may be in Draft Status,
    in which case it is still subject to significant change.
    A post that is not in Draft Status should not be materially changed,
    they should only be changed to include
    clarifications and annoted links to relevant newer content.
    """

    @computed_field
    def as_discussed_in_posts(self) -> list[tuple[object, str, str]]:
        return [
            (blog.About(), '#posting-policy', "see sections on Guidance Re: Posts"),
        ]

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'About_Section': 'more guidelines around draft status and posts generally',
            'Post': 'Narrative descriptions of updates to PlanZero, which may be in Draft Status if the work is still in progress',
        }


class Greenhouse_Gas(GlossaryTerm):
    """One, or a mixture, of the seven gases (or families of gases) assessed by the IPCC as
    a significant factor in trapping heat within the Earth planetary system.
    They are {{CO2|safe}}, {{CH4|safe}}, {{N2O|safe}},
    HFCs, PFCs, {{NF3|safe}}, and {{SF6|safe}}.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'IPCC': 'International Panel on Climate Change defines greenhouse gases and reporting standards for signatories to the Paris Agreement',
            'NGGI': 'National Greenhouse Gas Inventory tracks Canadian greenhouse gas emissions',
        }

    @computed_field
    def as_discussed_in_posts(self) -> list[tuple[object, str, str]]:
        return [
            (blog.GHG_Emissions(), '#h2_ghg', "see section on Greenhouse gases"),
        ]


class Inference_Algorithm(GlossaryTerm):
    """An inference algorithm samples from the posterior distribution of latent variables in a probabilistic model,
    often by applying Bayes' rule systematically across a formal specification of the model.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Stochastic_Model': 'Inference algorithms only make sense for stochastic models',
            'Posterior_Distribution': 'Inference algorithms sample from posterior distributions',
            'Latent_Variable': 'Inference algorithms provide samples for latent variables'
        }

    @computed_field
    def as_discussed_in_posts(self) -> list[tuple[object, str, str]]:
        return [
            #(blog.TwoProbabilisticModels(), '#', "see most of first half of post"),
        ]


class Unknown_Variable(GlossaryTerm):
    """An unknown variable is an unknown scalar- or vector-valued
    term in a probabilistic model. Often it could be known, measured
    or estimated, but we may lack sufficient information or equipment to do so.
    Sometimes it's not theoretically possible to measure something (like
    the number of cars on the road 10 years in the future,
    because it hasn't happened yet) but we can still include it in a model
    as an unknown variable.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Stochastic_Model': 'A model associates unknown variables with probability distributions',
            'Probability_Distribution': 'In order to reason about a variable without knowing exactly what it is, we assume a set of values that it might have using a probability distribution.',
        }

    @computed_field
    def as_discussed_in_posts(self) -> list[tuple[object, str, str]]:
        return [
                (blog.StaticNormals(), '#appendix-notation', "Appendix 1: Probabilistic Modelling Notation"),
        ]

    @computed_field
    def aka(self) -> list[str]:
        return ['Random Variable']


class Predictive_Model(GlossaryTerm):
    """
    A model that includes probability distributions for
    events in the future.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Stochastic_Model': 'a more general class of models, which may or may not include distributions related to future events',
        }


class Probability_Density_Function(GlossaryTerm):
    """A mathematical function that is never negative,
    whose area under the curve is 1, and which is used to
    describe the probability distrubtion for a random variable.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Unknown_Variable': 'a variable whose true value is not known, but with which we can associate a probability distribution',
        }


class Probability_Distribution(GlossaryTerm):
    """A probability distribution is the act or result of
    distributing probability across the possibly true values of a
    unknown variable.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Unknown_Variable': 'a variable whose true value is not known, but with which we can associate a probability distribution',
            'Probability_Density_Function': 'a description of how to distribute probability over a continuous set of possible values, such as a range of real numbers',
        }

    @computed_field
    def as_discussed_in_posts(self) -> list[tuple[object, str, str]]:
        return [
            (blog.ProbabilisticNIR2025(), '#appendix-model-distribution', "Appendix 1 is an introduction to probabilistic modelling"),
        ]


class Joint_Distribution(GlossaryTerm):
    """A joint probability distribution is act or result of
    distributing probability across possible values for multiple
    unknown variables. If the probability of each value for each variable
    does not depend on the others, then that variable is said to be "independent"
    of the rest, but that's the exception to the rule that generally,
    variables are not independent of each another.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
                'Probability_Distribution': 'Joint distributions are probability distributions, over multiple unknown variables.',
                'Stochastic_Model': 'Probabilistic models define joint distributions over variables',
                }


class Conditional_Distribution(GlossaryTerm):
    """A conditional distribution is a probability distribution
    over one or more unknown variables in a model,
    conditioned on assumptions about other variables in the model.
    If a conditional distribution is over multiple variables,
    then it is called a conditional joint distribution.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
                'Probability_Distribution': 'Conditional distributions are probability distributions.',
                'Stochastic_Model': 'Probabilistic models can be used to derive conditional distributions among their variables',
                }


class Posterior_Distribution(GlossaryTerm):
    """A posterior distribution is the result of conditioning on stochastic model on the observation of data.
    The observation of data generally changes the distribution of probability for the remaining variables, called latent variables.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
                'Inference_Algorithm': 'Inference in stochastic models can sometimes be done by standard algorithms.',
                'Conditional_Distribution': 'Posterior distributions are conditional distributions in which the conditioning is on observed data',
                'Latent_Variable': 'Unobserved unknown variables in a probabilistic model are called latent variables; a posterior distribution is over latent variables.',
                }

class Latent_Variable(GlossaryTerm):
    """A latent variable in a probabilistic model is one that remains unobserved when the model is conditioned on data. """
    @property
    def see_also(self) -> dict[str, str]:
        return {
                'Inference_Algorithm': 'Inference of latent variables can sometimes be done by standard algorithms.',
                'Posterior_Distribution': 'A posterior distribution is over latent variables.',
                }


class Credible_Interval(GlossaryTerm):
    """A <a href="https://en.wikipedia.org/wiki/Credible_interval">credible interval</a> is the set of values that a
    random variable in a probabilistic model might most-credibly take.
    For example, a 95% credible interval is the smallest interval containing
    the 95% most-probable values for the random variable.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Unknown_Variable': 'the model element associated with a credible interval',
            'Probabilistic Model': 'credible intervals arise only in probabilistic modelling',
            'Inference_Algorithm': 'the main use of credible intervals is to characterize the uncertainty remaining after inference of the latent random variables in a model',
        }

    @computed_field
    def as_discussed_in_posts(self) -> list[tuple[object, str, str]]:
        return [
            (blog.ProbabilisticNIR2025(), '#appendix-visualizing', "credibility intervals are used to visualize probabilistic models"),
        ]


class Bayesian_Inference(GlossaryTerm):
    """Bayesian inference is the use of
    <a href="https://en.wikipedia.org/wiki/Bayes%27_theorem">Bayes' Theorem</a>
    to define a posterior distribution over latent variables in a probabilistic
    model, by conditioning on the observation of data.
    Bayesian inference is one of the most compelling
    features of probabilistic models.
    Generally the result of inference is a sample of latent-variable values
    drawn from the posterior distribution by an algorithm, although in
    certain models Bayesian inference can be done analytically.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Probabilistic Model': 'Only probabilistic models support Bayesian inference',
            'Inference_Algorithm': 'In probabilistic models that are defined within probabilistic programming systems, Bayesian inference can often be implemented automatically',
        }

    @computed_field
    def as_discussed_in_posts(self) -> list[tuple[object, str, str]]:
        return [
            #(blog.TwoProbabilisticModels(), '#', "see most of first half of post"),
        ]


class Pending_Prediction_Challenge_Results(GlossaryTerm):
    """
    Prediction challenge results may be described as "Pending". That is
    because it is not possible to compute them yet.
    The prediction time has passed for the challenge, but the publication
    of the predicted data (the challenge's evaluation time) has either
    (a) not yet ocurred, or at least (b)
    not yet been incorporated into the PlanZero site.
    """
    # TODO: Prediction Time
    # TODO: Prediction Challenge

    @computed_field
    def as_discussed_in_posts(self) -> list[tuple[object, str, str]]:
        return [
            (blog.registry['StaticNormals'], '#prenir', "prediction time vs. evaluation time"),
        ]

class Prediction_Challenge(GlossaryTerm):
    """
    A Prediction Challenge in PlanZero is a machine learning problem statement
    about predicting emissions before they are published by an NIR.
    A challenge will specify what emissions to predict,
    for what regions, for what years, and how the predictions will be evaluated.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'NIR_Prediction': 'NIR prediction is the only type of prediction challenge in PlanZero so far',
            'Prediction_Date': 'The deadline after which new information cannot be used for prediction',
        }



class Prediction_Date(GlossaryTerm):
    """
    A Prediction Date is a point in time at which a prediction must be made,
    in the context of e.g. a prediction challenge. Information published
    after the prediction date cannot be used for prediction.
    """
    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Prediction_Challenge': 'Prediction challenges have prediction dates',
        }

class NIR_Prediction(GlossaryTerm):
    """
    In the context of PlanZero, NIR prediction
    refers either generally to the challenge of predicting
    the final years' emissions in an National greenhouse gas Inventory Report (NIR)
    prior to the report's publication,
    or specifically to the PlanZero machine learning problem statements
    reflecting that challenge with the weighted-KL divergence model
    evaluation metric.
    """

    @computed_field
    def as_discussed_in_posts(self) -> list[tuple[object, str, str]]:
        return [
            (blog.StaticNormals(), '#appendix-prenir', "Introduces the Pre-NIR-2025-06 prediction challenge, which uses a weighted KL-divergence to compare model predictions to reference distributions."),
        ]

    @computed_field
    def aka(self) -> list[str]:
        return ['PreNIR']

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Weighted_KL_Divergence': 'The scoring method for model predictions',
            'Prediction_Challenge': 'NIR prediction is the only type of prediction challenge in PlanZero so far',
        }

class Weighted_KL_Divergence(GlossaryTerm):
    """In PlanZero, the term "weighted KL divergence" typically refers
    to the specific weighting of KL divergences between actual and predicted
    emissions that is used to score NIR-predicting models probabilistic models.
    """

    @computed_field
    def as_discussed_in_posts(self) -> list[tuple[object, str, str]]:
        return [
            (blog.ProbabilisticNIR2025(), '#appendix-evaluating', "Develops the weighted KL divergence formula for scoring NIR predictions against a reference probabilistic interprentation of an NIR"),
            (blog.StaticNormals(), '#appendix-prenir', "Introduces the Pre-NIR-2025-06 prediction challenge, which uses a weighted KL-divergence to compare model predictions to reference distributions."),
        ]

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Model_Metric': 'Weighted KL divergence is used to score how well (technicall, how poorly) a predictive model approximates a reference one.',
        }

class Symmetric_Blended_Log_Normal_Distribution(GlossaryTerm):

    """<p>A symmetric blended log-normal distribution is a
    custom parametric probability distribution used in PlanZero
    to model quantities that are characterized by relative uncertainty,
    but which may sometimes be both negative and positive, such as
    emissions in certain sectors.
    </p>
    <p>The details of this distribution can be found in a
<a href="https://github.com/jaberg/planzero/blob/d1972c58f1b8463be5fb3462e7b1c3179d273fa0/SymmetricBlendedLogNormal.ipynb">Jupyter notebook</a>
hosted on PlanZero's GitHub page.</p>
    """

    @computed_field
    def aka(self) -> list[str]:
        return ['SBLN']

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Probability_Distribution': 'The SBLN is a probability distribution',
        }

    @computed_field
    def as_discussed_in_posts(self) -> list[tuple[object, str, str]]:
        return [
            (blog.ProbabilisticNIR2025(), '#appendix-sbln', "Probabilistic NIR2025, Appendix 2"),
        ]
