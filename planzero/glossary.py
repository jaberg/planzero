"""
This file  should be imported last among library code files,
so that it can import objects throughout the library, and retrieve
their line numbers for constructing github links.
"""
import jinja2
from pydantic import BaseModel, computed_field
import inspect

class MyClass:
    pass

glossary_terms = {} # classname -> Singleton instance

from .blog import latex
from . import barriers
from . import cattle
from . import csfs

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

    @computed_field
    def see_also(self) -> dict[str, str]:
        return {}

    @computed_field
    def all_names(self) -> list[str]:
        return [self.__class__.__name__]

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
        glossary_terms[cls.__name__] = cls()

    def template_globals(self) -> dict[str, object]:
        def local_ref(term, text=None):
            return glossary_terms[term].local_ref(text)
        return dict(
            CO2e=latex(r'\mathrm{CO}_2\mathrm e '),
            glossary_terms=glossary_terms,
            local_ref=local_ref,
        )

    def local_ref(self, text=None) -> str:
        if text is None:
            text = self.__class__.__name__.replace('_', ' ')
        return f'<a href="#{self.__class__.__name__}">{text}</a>'


    def abs_ref(self, text=None) -> str:
        if text is None:
            text = self.__class__.__name__.replace('_', ' ')
        return f'<a href="/glossary#{self.__class__.__name__}">{text}</a>'



class Time_Series(GlossaryTerm):
    """A PlanZero modelling data structure for representing a time-varying
    quantity.
    """


class Dynamic_Element(GlossaryTerm):
    """A PlanZero modelling data structure for representing a modelling
    assumption, that one or more
    {{local_ref("Time_Series", "time series")|safe}} follows a formula.
    A dynamic element is expected to be a Python code object, that is
    a subclass of either a
    {{local_ref("Strategy")|safe}}, a
    {{local_ref("Barrier")|safe}}, or a
    {{local_ref("Critical_Success_Factor")|safe}}.
    """

class Strategy(GlossaryTerm):
    """A Strategy is a {{local_ref("Dynamic_Element", "dynamic element")|safe}}
    that can act on investment""" 

class Public_Policy(GlossaryTerm):
    """A Public Policy is a {{local_ref("Dynamic_Element", "dynamic element")|safe}}
    that can act on public funds.""" 


class Barrier(GlossaryTerm):
    """<p>A Barrier is a 
    {{local_ref("Dynamic_Element", "dynamic element")|safe}}
    that, unlike a 
    {{local_ref("Strategy")|safe}} or
    {{local_ref("Public_Policy")|safe}},
    acts regardless of investment.</p>

    <p>
    PlanZero terminology may feel a bit cynical in this regard, but yes, in
    this terminology, all of the following would qualify as barriers:
    <ul>
    <li>regulations</li>
    <li>the life cycle of assets</li>
    <li>consumer behaviour</li>
    <li>the length of research and development cycles</li>
    <li>return on investment requirements</li>
    <li>the laws of physics</li>
    <li>the predictions of climate models</li>
    </ul>
    Strategies and Public Policies may typically be omitted from a model in
    order to mean "not doing them", but removing a Barrier from a model
    generally invalidates the model's output.
    </p>
    """ 

    @property
    def code_refs(self) -> dict[str, object]:
        return {
            'Barrier base class': barriers.Barrier,
            'Example Barrier class: Bovaer Adoption Limit': cattle.Bovaer_Adoption_Limit,
        }


class Critical_Success_Factor(GlossaryTerm):
    """<p>A Critical Success Factor is a dynamic element tied to an
    {{local_ref("IPCC_Sector")|safe}},
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
    {{local_ref("EGFS")|safe}}.
    </p>
    """

    @computed_field
    def as_discussed_in_posts(self) -> dict[str, str]:
        return {'A Glossary of terms...':
                '/blog/2026-04-19-glossary#critical_success_factor'}

    @computed_field
    def all_names(self) -> list[str]:
        # TODO: use these
        return ['Critical Success Factor', 'CSF']

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
    In PlanZero, an NIR model is also expected to generated critical success
    factors,
    so that it can be visualized in the models section.
    """

    @computed_field
    def see_also() -> dict[str, str]:
        return {
            'Model': 'more-general term',
            'Models Section': 'models section of PlanZero website',
        }


class Model(GlossaryTerm):
    """A model, in PlanZero, is a set of time series and dynamic elements that
    can be simulated to generate one or more possible scenarios.
    A model can be either deterministic or stochastic. 
    """

    @computed_field
    def see_also() -> dict[str, str]:
        return {
            'NIR_Model': """Model of Canada's national emissions in the style
            of the National Inventory Reports submitted to UNFCCC""",
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


class Model_Metric(GlossaryTerm):
    """A single number associated with a Model, often defined
    for the purpose of ranking how good different models are at something."""

    @computed_field
    def see_also() -> dict[str, str]:
        return {
            'Model': "A set of time series",
        }

class Git(GlossaryTerm):
    """<p><a href="https://git-scm.com/">Git</a> is a "free and open source
    distributed version control system designed to handle everything from
    small to very large projects with speed and efficiency."</p>
    <p>PlanZero is developed with git, and hosted on
    {{local_ref("GitHub")|safe}} as a 
    <a href="https://github.com/jaberg/planzero">public code repository</a>.</p>
    """

class Git_Merge(GlossaryTerm):
    """A git "merge" is the operation of combining two
    {{local_ref("Git_Branch", "branches")|safe}} together into a coherent set
    of files, extending one branch with
    {{local_ref("Git_Commit", "commits")|safe}} from the other.
    Sometimes some fixing up is required, for example if both branches have
    changed the same part of the same file, but usually merging is mostly
    automatic. Merging is how people can contribute their changes
    back to the {{local_ref("Main_Branch", "main branch")|safe}}.
    Merging can be done locally on a development computer,
    or on {{local_ref("GitHub")|safe}} via a {{local_ref("GitHub_Pull_Request", "pull request")|safe}}.
    """

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
    """The site is populated by deploying the "main" branch. (LINK TO IT)
    """

class Git_Fork(GlossaryTerm):
    """Fork this repo - that's a thing you can do, go for it.
    """

class Git_Commit(GlossaryTerm):
    """Open source revision-control system
    """

class GitHub(GlossaryTerm):
    """Link to GH project page, explain how github is used
    """

    @computed_field
    def see_also() -> dict[str, str]:
        return {
            'GitHub_Issue': "a TODO item on the PlanZero project",
        }


class GitHub_Issue(GlossaryTerm):
    """Link to GH issues page, explain how they're used in PlanZero
    """

    @computed_field
    def abbreviations() -> list[str]:
        return ['gh issue']

    @computed_field
    def see_also() -> dict[str, str]:
        return {
            'GitHub': """Code hosting for PlanZero""",
        }


class GitHub_Pull_Request(GlossaryTerm):
    """One of GitHub's main features is a web interface for users
    to suggest changes to open source project code. Pull requests
    enable asynchronous loosely-coupled development over the net by
    giving contributors and developers a place to talk about the changes,
    and implementing the {{local_ref("Git_Merge", "merging")|safe}} of changes.
    See <a href="https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests">Github documentation</a> for full description of this capability.
    """


class EGFS(GlossaryTerm):
    """<p><a href="https://www.amazon.ca/Executive-Guide-Facilitating-Strategy/dp/0972245812">Executive Guide to Facilitating Strategy, a book by Michael Wilkinson</a>.</p>
    <p>
    I've appropriated (hopefully not misappropriated) terms
    of implementation planning ({{local_ref("Critical_Success_Factor")|safe}},
    {{local_ref("Barrier")|safe}},
    {{local_ref("Strategy")|safe}})
    as defined in this book, to structure the visualization of model simulation.
    I have also tried to insert aspects of directional strategy (mission,
    vision, guiding principles, positioning) in the project overview on the
    <a href="/about/">About page</a>.
    </p>
    """


class NIR(GlossaryTerm):
    """National Inventory Report, published annually by Environment and Climate Change Canada, and submitted to the UNFCCC Secretariat.
    Specific reports are referred to as e.g. NIR-2023, NIR-2025, an so on.
    """


class NEUD(GlossaryTerm):
    """National Energy Usage Database, developed by Natural Resources Canada
    """


class StatsCan(GlossaryTerm):
    """Statistics Canada
    """


class NRCan(GlossaryTerm):
    """Natural Resources Canada
    """


class ECCC(GlossaryTerm):
    """Environment and Climate Change Canada
    """


class IPCC(GlossaryTerm):
    """International Panel on Climate Change
    """

class IPCC_Sector(GlossaryTerm):
    """An economic area for which Canada reports emissions
    in at least one {{local_ref("NIR")|safe}}, in accordance
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
    in units of {{CO2e|safe}}."""
