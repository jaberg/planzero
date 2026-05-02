"""
This file  should be imported last among library code files,
so that it can import objects throughout the library, and retrieve
their line numbers for constructing github links.
"""
import jinja2
from pydantic import BaseModel, computed_field
import inspect

glossary_terms = {} # classname -> Singleton instance
glossary_terms_w_aka = {} # string -> Singleton instance

def siteref(term, text=None):
    try:
        return glossary_terms_w_aka[term].site_reference(text or term)
    except KeyError as exc:
        try:
            return glossary_terms[term].site_reference(text)
        except KeyError as fallback_exc:
            raise exc


from .blog import latex
from . import barriers
from . import cattle
from . import csfs
from . import strategies
from .sts import STS
from .base import DynamicElement


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

    @computed_field
    def all_names(self) -> list[str]:
        #rval = [self.__class__.__name__]
        rval = []
        rval.append(self.__class__.__name__.replace('_', ' '))
        return rval + self.aka

    @computed_field
    def as_discussed_in_posts(self) -> dict[str, str]:
        return {}

    @computed_field
    def code_links(self) -> dict[str, str]:
        rval = {}
        for txt, cls in self.code_refs.items():
            file_path = inspect.getsourcefile(cls)
            assert file_path.startswith('/mnt/planzero')
            file_path = file_path[5:]
            lines, line_number = inspect.getsourcelines(cls)
            url = f'https://github.com/jaberg/planzero/blob/main/{file_path}#{line_number}'
            rval[txt] = url

        return rval

    @property
    def code_refs(self) -> dict[str, object]:
        return {}

    @classmethod
    def __init_subclass__(cls):
        super().__init_subclass__()
        obj = cls()
        assert cls.__name__ not in glossary_terms
        glossary_terms[cls.__name__] = obj

        for name in obj.all_names:
            assert name not in glossary_terms_w_aka, name
            glossary_terms_w_aka[name] = obj

    def template_globals(self) -> dict[str, object]:
        def lref(term, text=None):
            return glossary_terms_w_aka[term].local_ref(text)
        return dict(
            CO2e=latex(r'\mathrm{CO}_2\mathrm e '),
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



class Time_Series(GlossaryTerm):
    """A PlanZero modelling data structure for representing a time-varying
    quantity.
    """

    @property
    def code_refs(self) -> dict[str, object]:
        return {
            'Time Series base class': STS,
        }


class Dynamic_Element(GlossaryTerm):
    """A PlanZero modelling data structure for representing a modelling
    assumption, that one or more
    {{lref("Time Series", "time series")|safe}} follows a formula.
    A dynamic element is expected to be a Python code object, that is
    a subclass of either a
    {{lref("Strategy")|safe}}, a
    {{lref("Barrier")|safe}}, or a
    {{lref("Critical Success Factor")|safe}}.
    """

    @property
    def code_refs(self) -> dict[str, object]:
        return {
            'Dynamic Element base class': DynamicElement,
        }


class Strategy(GlossaryTerm):
    """<p>A Strategy is a {{lref("Dynamic Element", "dynamic element")|safe}}
    that is optional, that can be omitted without sacrificing the validity of
    a model.
    Typically a strategy directly affects a small number of time series, and
    indirectly, through those, affects the evolution of more time series via
    barriers.
    </p>
    <p>
    Defining Strategy in this way enables
    {{lref("Ablative Analysis")|safe}} as a standard
    part of {{lref("Simulation")|safe}}.
    </p>
    <p>I borrow the term from {{lref("EGFS")|safe}} but risk mis-appropriating it
    as the use in a computational modelling framework is, admittedly, a stretch.
    </p>
    """ 

    @property
    def code_refs(self) -> dict[str, object]:
        return {
            'Strategy base class': strategies.Strategy2,
            'Example Strategy: Scale Bovaer': strategies.strategy2.ScaleBovaer,
        }


class Barrier(GlossaryTerm):
    """<p>A Barrier is a 
    {{lref("Dynamic Element", "dynamic element")|safe}}
    that is not optional, that is, one whose omission would sacrifice the
    validity of a model.</p>
    <p>
    PlanZero terminology may feel a bit cynical in this regard, but yes, in
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
    <p>I borrow the term from {{lref("EGFS")|safe}} but risk mis-appropriating it
    as the use in a computational modelling framework is, admittedly, a stretch.
    </p>
    """ 

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
        }


class IPCC_Sector_Contributor(GlossaryTerm):
    """<p>A {{lref("Dynamic Element", "dynamic element")|safe}}
    (part of a {{lref("Model", "model")|safe}})
    that represents a contribution to an {{lref("IPCC Sector")|safe}}.
    Category emissions are typically a sum of products (e.g. amount of activity
    multiplied by emissions per unit of activity,
    summed over one or more activities that count toward the category);
    in this typical case, each of the things being summed is an
    IPCC-Sector Contributor.
    </p>
    <p>
    This dynamic element defines a single timeseries for each greenhouse gas
    that is emitted, whose unit is a rate of mass (of gas) per unit time.
    </p>
    """


class Emission_Factor(GlossaryTerm):
    """<p>
    Category emissions are typically a sum of products (e.g. amount of activity
    multiplied by emissions per unit of activity,
    summed over one or more activities that count toward the category);
    in this typical case, each of the emission-contributing activities corresponds
    to an {{lref("Emissions Source")|safe}} and
    the emission of each greenhouse gas per unit of activity is referred to as
    an Emission Factor.
    </p>
    <p>
    An Emission Factor is a time series, whose unit is an amount of
    mass (of greenhouse gas) per unit activity (or if not "activity",
    whatever makes sense for the {{lref("Emissions Source", "emission
    contributor")|safe}}).
    </p>
    """


class Emissions_Source(GlossaryTerm):
    """<p>
    Category emissions are typically a sum of products (e.g. amount of activity
    multiplied by emissions per unit of activity,
    summed over one or more activities that count toward the category);
    in this typical case, each of the emission-contributing activities
    corresponds
    to an {{lref("Emissions Source")|safe}} and
    the emission of each greenhouse gas per unit of activity is referred to as
    an Emission Factor.
    </p>
    <p>
    An Emission Source is a time series, whose unit is typically an amount of
    activity (in whatever unit is appropriate for the emissions source) per
    unit time.</p>"""


class Critical_Success_Factor(GlossaryTerm):
    """<p>A Critical Success Factor is a dynamic element tied to an
    {{lref("IPCC Sector")|safe}},
    representing a mathematical decomposition of what
    would be required to reduce or maintain emissions in that category.
    Category emissions are typically a sum of products (e.g. amount of activity
    multiplied by emissions per unit of activity for one or more activities),
    and the summed terms are typically the Critical Success Factors (e.g. 
    each emission-contributing activity is one Critical Success Factor).
    </p>
    TODO EXAMPLE
    <p>
    Critical Success Factors are used in the visualization and communication
    of the effects of strategies and barriers.
    Althouth the scope (or granularity) of critical success factors is not
    generally obvious,
    I hope that PlanZero can make itself useful with a relatively stable set.
    They are defined to be one-per-emissions-contributing-activity so that
    PlanZero can visualize the emissions contributing to a sector as a stacked
    line chart of Critical Success Factors.
    </p>
    <p>
    The term Critical Success Factor has a <a href="https://en.wikipedia.org/wiki/Critical_success_factor">long history</a>.
    I believe I'm fairly appropriating it, I learned about it from
    {{lref("EGFS")|safe}}.
    </p>
    """

    @computed_field
    def as_discussed_in_posts(self) -> dict[str, str]:
        return {'A Glossary of terms...':
                '/blog/2026-04-19-glossary#critical_success_factor'}

    @computed_field
    def aka(self) -> list[str]:
        return ['CSF']

    @property
    def code_refs(self) -> dict[str, object]:
        return {
            'Critical Success Factor base class': csfs.CSFs,
            'Example CSF: Large Oil and Gas Extraction Operations': csfs.CSFs, # TODO
        }



class NIR_Model(GlossaryTerm):
    """
    An NIR model is a model that can generate time series corresponding to
    emissions predictions in the form of a National Inventory Report.
    In PlanZero, an NIR model is also expected to generate
    {{lref("Critical Success Factor", "critical success factors")|safe}},
    so that it can be visualized in the models section.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Model': 'more-general term',
            'Models Section': 'models section of PlanZero website',
            'Critical Success Factor': 'an emissions contribution to an IPCC Sector',
        }


class Model(GlossaryTerm):
    """A model, in PlanZero, is a set of time series and dynamic elements that
    can be simulated to generate one or more possible scenarios.
    A model can be either deterministic or stochastic. 
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'NIR_Model': """Model of Canada's national emissions in the style
            of the National Inventory Reports submitted to UNFCCC""",
            'Deterministic_Model': "A model that corresponds to a unique scenario",
            'Stochastic_Model': "A model that corresponds to a distribution over possible scenarios",
        }


class Stochastic_Model(GlossaryTerm):
    """A non-deterministic model, which corresponds to a set of possible
    scenarios, rather than a single one. Simulating a stochastic model
    requires choosing a random seed. Simulating a stochastic model with
    different random seeds results in different scenarios. These scenarios
    follow some distribution over possible outcomes, as defined by the model.
    """

class Deterministic_Model(GlossaryTerm):
    """A deterministic model is a model that corresponds to a specific
    scenario, and has no randomness.
    """


class Models_Section(GlossaryTerm):
    """The models section of the planzero.ca website:
    https://planzero.ca/models/"""


class About_Section(GlossaryTerm):
    """<p>The "About" section of the planzero.ca website:
    <a href="/about/">planzero.ca/about</a></p>"""


class Scenario(GlossaryTerm):
    """A scenario is a set of time series.
    Typically in PlanZero it is the result of simulating a model.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Deterministic_Model': "A model that corresponds to a unique scenario",
            'Stochastic_Model': "A model that corresponds to a distribution over possible scenarios",
        }


class Model_Metric(GlossaryTerm):
    """A formula, procedure or rule for associating
    a single number to a {{lref("Model")|safe}}.
    Typically,
    for the purpose of ranking models in terms of e.g. prediction accuracy."""

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Model': "A set of assumptions that can be simulated to produce one or more scenarios",
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
    """A sequence of git commits (sometimes a graph)
    to source files leading from the initially empty project
    to some version that's full of files.
    The site is populated by deploying the "main" branch.
    Anyone is welcome to suggest changes to main by creating a pull request on github,
    requesting that the main branch merge changes from another branch that
    they've created.
    """

class Main_Branch(GlossaryTerm):
    """PlanZero on GitHub generally has multiple branches.
    The "main branch" is special, in that it is the one used to deploy the
    {{lref("PlanZero Site", "PlanZero site")|safe}}.
    New improvements to the codebase should be {{lref("Git Merge", "merged")|safe}}
    to the main branch.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Git_Branch': "a version of a codebase",
        }

class PlanZero_Site(GlossaryTerm):
    """The PlanZero site is the website hosted at <a
    href="https://planzero.ca">https://planzero.ca</a>.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Main_Branch': "the code from which the site is generated",
        }

class Repository(GlossaryTerm):
    """Code repository, on GitHub"""

    @computed_field
    def aka(self) -> list[str]:
        return ['repo']


class Git_Fork(GlossaryTerm):
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
        }

class Git_Commit(GlossaryTerm):
    """A "commit" is an increment of change to a codebase, across one or more
    changed files.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Git_Branch': "a sequence of commits",
        }

class GitHub(GlossaryTerm):
    """<p>GitHub (<a href="https://github.com/">site</a>, <a href="https://en.wikipedia.org/wiki/GitHub">wikipedia</a>) is a web service for using the git version
    control system over the internet to collaborate on software projects. Circa 2023, it was the world's largest source code host, with over 100 million developers, and 420 million code repositories.</p>
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'GitHub_Issue': "a future-work item on the PlanZero project",
        }


class GitHub_Issue(GlossaryTerm):
    """Link to GH issues page, explain how they're used in PlanZero
    """

    @computed_field
    def aka(self) -> list[str]:
        return ['gh issue']

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'GitHub': """Code hosting for PlanZero""",
        }


class GitHub_Pull_Request(GlossaryTerm):
    """One of GitHub's main features is a web interface for users
    to suggest changes to open source project code. Pull requests
    enable asynchronous loosely-coupled development over the net by
    giving contributors and developers a place to talk about the changes,
    and implementing the {{lref("Git Merge", "merging")|safe}} of changes.
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
        }


class EGFS(GlossaryTerm):
    """<p><a href="https://www.amazon.ca/Executive-Guide-Facilitating-Strategy/dp/0972245812">Executive Guide to Facilitating Strategy, a book by Michael Wilkinson</a>.</p>
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


class NIR(GlossaryTerm):
    """National Inventory Report,
    published annually by Environment and Climate Change Canada, and
    submitted to the UNFCCC Secretariat.
    Specific reports are referred to as e.g. NIR-2023, NIR-2024, NIR-2025,
    an so on in PlanZero.
    The first report was <a href="https://publications.gc.ca/site/eng/9.506002/publication.html">NIR-2004</a>.
    The most recent as-of writing is <a href="https://www.canada.ca/en/environment-climate-change/services/climate-change/greenhouse-gas-emissions/sources-sinks-executive-summary-2026.html">NIR-2026</a>.
    """

class National_Greenhouse_Gas_Inventory(GlossaryTerm):
    """Canada's
    <a
    href="https://www.canada.ca/en/environment-climate-change/services/climate-change/greenhouse-gas-emissions/inventory.html">National
    Greenhouse Gas Inventory</a> tracks emissions
    across Canada, and is the basis for the annual
    {{lref("NIR", "National Inventory Reports")|safe}}.
    """


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


class Natural_Resources_Canada(GlossaryTerm):
    """<p><a href="https://natural-resources.canada.ca/">Natural Resources
    Canada</a> (sometimes, NRCan) is a federal ministry
    "<i>Committed to improving the quality of
    life of Canadians by ensuring the country’s abundant natural resources are
    developed sustainably, competitively and inclusively.</i>".
    </p>

    <p>PlanZero use the {{lref("NEUD")|safe}}, published by NRCan.</p>
    """

    @computed_field
    def aka(self) -> list[str]:
        return ['NRCan']

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Environment_and_Climate_Change_Canada': 'peer federal ministry'
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
            'Natural_Resources_Canada': 'peer federal ministry'
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


class IPCC_Sector(GlossaryTerm):
    """An economic area for which Canada reports emissions
    in at least one {{lref("NIR")|safe}}, in accordance
    with IPCC emissions reporting guidelines.
    In PlanZero posts,
    the term almost always refers to a sector that is not
    a subtotal of other sectors.
    """


class UNFCCC(GlossaryTerm):
    """United Nations Framework Convention on Climate Change
    """


class Python(GlossaryTerm):
    """<p>PlanZero is implemented in the <a href="https://www.python.org/">Python programming language</a></p>"""


class Emissions(GlossaryTerm):
    """The combined emissions of the seven greenhouse gases, often quantified
    in units of {{CO2e|safe}}.
    Canada estimates rates of emissions in the
    {{lref("National Greenhouse Gas Inventory")|safe}}.
    """


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


class Rollout(GlossaryTerm):
    """The step by step creation of a scenario, by simulating a model."""


class Simulation(GlossaryTerm):
    """The algorithm of computing a scenario for a model by computing
    the recurrence in dynamic elements.

    Simulation

    TODO: talk about temporal dependencies, and latest vs current dependence.
    """

class Ablative_Analysis(GlossaryTerm):
    """
    For models with strategies, simulation involves rolling out the model
    with all of the strategies enabled, and then with each
    individual strategy being omitted.
    """
