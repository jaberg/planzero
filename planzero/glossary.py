"""
This file  should be imported last among library code files,
so that it can import objects throughout the library, and retrieve
their line numbers for constructing github links.
"""
import jinja2
from pydantic import BaseModel, computed_field

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

    @computed_field
    def all_names(self) -> list[str]:
        #rval = [self.__class__.__name__]
        rval = []
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
            else:
                rval[txt] = coderef_url(thing)
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

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Unit_of_Measure': 'the values of a time series are associated with a single unit of measure',
            'Time_Series_Interpolation_Mode': 'the rule for determining value for un-mentioned times',
        }

    @property
    def code_refs(self) -> dict[str, object]:
        return {
            'Time Series base class': STS,
        }

class Time_Series_Interpolation_Mode(GlossaryTerm):
    """<p>The time series interpolation mode is a mechanism that is partly for
    convenience and partly for error prevention.
    There are currently two interpolation modes.
    The value of a time series at times other than those explicitly mentioned is either
    <ul>
        <li>"current", defined to be the most recent value of the series</li>
        <li>"no interpolation", which leaves such values undefined</li>
    </ul>
    </p>
    """
    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Time_Series': 'the data structure time series',
        }


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


class Dynamic_Element(GlossaryTerm):
    """A PlanZero modelling data structure for representing a modelling
    assumption, and defining one or more
    {{lref("Time Series", "time series")|safe}}.
    A dynamic element is expected to be a Python code object, that is
    a subclass of either a
    {{lref("Strategy")|safe}} or a
    {{lref("Barrier")|safe}}.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Time Series': 'timeseries are the inputs and outputs of dynamic elements',
            'Model': 'models are sets of dynamic elements',
            'Simulation': 'dynamic elements provide the initialization and recurrence logic to define time series by simulation',
        }

    @computed_field
    def as_discussed_in_posts(self) -> list[tuple[object, str, str]]:
        return [
            (blog.Glossary(), '#dynelem', "see section on Computation and Simulation"),
        ]

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
            'Example Strategy: Scale Bovaer': strategies.strategy2.Scale_Bovaer,
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
        }



class Subsidy_Factor(GlossaryTerm):
    """<p>A subsidy factor is a constant of proportionality between
    a {{lref("Driver")|safe}} and a real or hypothetical funding program.</p>
    """


class Critical_Success_Factor(GlossaryTerm):
    """<p>A Critical Success Factor
    is a necessary condition of a KPI to achieve an objective.
    For example, the KPI must be within a certain range of values
    for any or all of some period of time.
    The term Critical Success Factor has a
    <a href="https://en.wikipedia.org/wiki/Critical_success_factor">long history</a>.
    PlanZero's use of the term is based on the definition from {{lref("EGFS")|safe}}.
    </p>
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'EGFS': 'The Executive Guide to Facilitating Strategy',
            'Models Section': 'models section of PlanZero website',
            'Critical Success Factor': 'an emissions contribution to an IPCC Sector',
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
        return ['KPI']

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'CSF': "target values for target times of a KPI time series, in order to achieve an objective",
            'NIR_Model': "a model of Canada's future emissions",
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

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'NIR_Model': """Model of Canada's national emissions in the style
            of the National Inventory Reports submitted to UNFCCC""",
            'Deterministic_Model': "A model that corresponds to a unique scenario",
            'Model': "A set of dynamic elements that can be simulated",
        }

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
    <a href="/simulations/">https://planzero.ca/simulations/</a>"""


class About_Section(GlossaryTerm):
    """<p>The "About" section of the planzero.ca website:
    <a href="/about/">planzero.ca/about</a></p>"""


class Scenario(GlossaryTerm):
    """A scenario is a set of time series covering a common time period.
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
        return ['repo', 'repository']

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'GitHub': "the site that hosts GitHub code repositories",
            'Git': "the version control system upon which GitHub operates",
            'Git_Fork': "A repo can be forked to create a downstream copy of an upstream repo"
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
            'Git_Repo': "a copy of a git graph",
            'Git_Branch': "a named subgraph of ancestors of a particular commit",
        }


class GitHub(GlossaryTerm):
    """<p>GitHub (<a href="https://github.com/">site</a>, <a href="https://en.wikipedia.org/wiki/GitHub">wikipedia</a>) is a web service for using the git version
    control system over the internet to collaborate on software projects. Circa 2023, it was the world's largest source code host, with over 100 million developers, and 420 million code repositories.</p>
    """

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
            'Git_Fork': "typically a pull request represents a request to merge code from a branch in one fork (maintained by one person) into a branch on another fork (maintained by another person)"
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

    @computed_field
    def aka(self) -> list[str]:
        return ['NGGI']


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
    """IPCC Sector is a PlanZero term, for
    an economic area for which Canada tracks emissions
    in the {{lref("NGGI")|safe}}, in accordance
    with IPCC emissions reporting guidelines.
    In PlanZero posts,
    the term almost always refers to a sector that is not
    a subtotal of other sectors.
    PlanZero uses a set of 71 IPCC Sectors, which match
    the ones used in NIR-2025.
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
    Ablative analysis in PlanZero is the study of models with and without
    certain elements (namely strategies), in order to characterize the impact
    of those elements on the behaviour of the whole model.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Strategy': 'ablative analysis is used to evaluate strategies',
            'Simulation': 'implements ablative analysis',
        }


class Post(GlossaryTerm):
    """A post on this site (see e.g. <a href="/posts/">https://planzero.ca/posts/</a>).
    </p><p>
    The <a href="/about/">About</a> page
    includes some guidelines for post content.
    """


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

