"""
This file  should be imported last among library code files,
so that it can import objects throughout the library, and retrieve
their line numbers for constructing github links.

Guidelines for Glossary Terms
-----------------------------

Multiple meanings: some terms (e.g. "rollout" at one point) have multiple meanings.
(e.g. "rollout" as noun, "roll out" as verb). These should be listed as multiple Glossary Terms,
and cross-linked via their "see_also" attributes, which should clarify the differences / relationships
between the related terms.

The dictionary returned by `see_also` of a GlossaryTerm A should have keys that are
the class name of glossary terms (e.g. terms, B and C, different from A),
and values that are one or more full sentences each.
The sentences should relate the term of the GlossaryTerm class to the terms in the see_also dictionary
(e.g. the sentence(s) associated with B should relate A and B;
the sentence(s) associated with C should relate A and C.)
"""
import functools
import re

import jinja2
from pydantic import BaseModel

from . import blog
from .html import coderef_url, latex
from .singleton_registry import SingletonRegistry

registry = SingletonRegistry()

class AKA_Registry:
    def __init__(self):
        self.aliases = {}

    def refresh_alias_list(self):
        self.aliases = {}
        for clsname, obj in registry.items():
            assert clsname not in self.aliases
            self.aliases[clsname] = clsname
            for alias in obj.all_names:
                assert clsname == self.aliases.setdefault(alias, clsname), (alias, clsname)

    def __getitem__(self, key):
        try:
            clsname = self.aliases[key]
        except KeyError:
            self.refresh_alias_list()
            clsname = self.aliases[key]
        return registry[clsname]

    def __contains__(self, key):
        if key in self.aliases and self.aliases[key] in registry:
            return True
        self.refresh_alias_list()
        rval = key in self.aliases and self.aliases[key] in registry
        return rval


aka_registry = AKA_Registry()

# Block-level tags that must be flattened to inline text before they can be
# used as tooltip content inside a <span> (tooltip content is inserted into
# <p> blocks, so any block tag would close the surrounding paragraph).
_BLOCK_TAG_RE = re.compile(
    r'</?(?:p|div|ul|ol|li|h[1-6]|blockquote|pre|table|thead|tbody|tr|td|th|figure|figcaption)\b[^>]*>',
    re.IGNORECASE,
)

def siteref(term, text=None):
    return aka_registry[term].site_reference(text or term)


def template_globals(tooltips: bool = False, site_links: bool = False) -> dict[str, object]:
    def lref(term, text=None):
        obj = aka_registry[term]
        href = obj.site_href if site_links else obj.local_href
        text = obj._ref_text(text)
        if tooltips:
            return obj._glossary_ref_markup(href, text)
        return obj._plain_ref(href, text)

    return {
        'CO2e': latex(r'\mathrm{CO}_2\mathrm e '),
        'CO2': latex(r'\mathrm{CO}_2'),
        'CH4': latex(r'\mathrm{CH}_4'),
        'N2O': latex(r"\mathrm N_2 \mathrm O"),
        'SF6': latex(r"\mathrm{SF}_6"),
        'NF3': latex(r"\mathrm{NF}_3"),
        'degrees': latex(r'^\circ'),
        'lref': lref,
    }


class GlossaryTerm(BaseModel):

    reserved: bool = False

    @property
    def definition(self) -> str:
        assert self.__class__.__doc__, self.__class__
        return self.__class__.__doc__

    def _render_definition(self, globals_: dict[str, object]) -> str:
        source = self.definition
        if not source.startswith('<p>'):
            source = f'<p>{source}</p>'

        template = jinja2.Template(source=source)
        return template.render(globals_)

    @property
    def plain_definition_html(self) -> str:
        return self._render_definition(template_globals())

    @property
    def definition_html(self) -> str:
        # Rendered for page display: cross-references carry tooltips.
        return self._render_definition(template_globals(tooltips=True))

    @property
    def hover_definition_html(self) -> str:
        # Tooltip content is rendered outside the glossary page, so lref links
        # must target the glossary page. Keep them plain to avoid recursion
        # into another term's tooltip_html.
        return self._render_definition(template_globals(site_links=True))

    @property
    def tooltip_html(self) -> str:
        # Flatten the hover definition into inline (phrasing) content so it can
        # be safely embedded as a tooltip <span> inside a paragraph.
        html = self.hover_definition_html.strip()
        # Keep list items readable when flattened.
        html = html.replace('<li>', ' \u00b7 ')
        html = _BLOCK_TAG_RE.sub(' ', html)
        return re.sub(r'\s+', ' ', html).strip()
    @property
    def see_also(self) -> dict[str, str]:
        return {}

    @property
    def aka(self) -> list[str]:
        return []

    def pretty_names(self) -> list[str]:
        """Return the names of a term as they should appear in e.g. term list and per-term header.
        The term's preferred form should be first, followed by pretty, de-duplicated AKA names.
        """
        # used in glossary.html in the term list and per-term header
        list_pretty = [name.replace('_', ' ') for name in self.all_names]
        assert list_pretty
        first_pretty = list_pretty[0]
        set_pretty = set(list_pretty)

        assert len(set_pretty) >= 1
        if len(set_pretty) == 1:
            return [first_pretty]
        else:
            return [first_pretty] + sorted({name for name in set_pretty if name != first_pretty})

    @property
    def all_names(self) -> list[str]:
        rval = [self.__class__.__name__]
        #rval = []
        if '_' in self.__class__.__name__:
            rval.append(self.__class__.__name__.replace('_', ' '))
        return rval + self.aka

    @property
    def as_discussed_in_posts(self) -> list[tuple[object, str, str]]:
        # post, hashtarget, intro txt
        return []

    @property
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

    def _ref_text(self, text=None) -> str:
        if text is None:
            return self.__class__.__name__.replace('_', ' ')
        return text

    @property
    def local_href(self) -> str:
        return f'#{self.__class__.__name__}'

    @property
    def site_href(self) -> str:
        return f'/glossary#{self.__class__.__name__}'

    def _plain_ref(self, href: str, text=None) -> str:
        return f'<a href="{href}">{self._ref_text(text)}</a>'

    def _glossary_ref_markup(self, href: str, text: str) -> str:
        return (
            f'<span class="glossary-ref">'
            f'<a href="{href}">{text}</a>'
            f'<span class="glossary-ref-tip" aria-hidden="true">{self.tooltip_html}</span>'
            f'</span>'
        )

    def _plain_local_ref(self, text=None) -> str:
        return self._plain_ref(self.local_href, text)

    def local_ref(self, text=None) -> str:
        return self._glossary_ref_markup(self.local_href, self._ref_text(text))

    def site_reference(self, text=None) -> str:
        return self._glossary_ref_markup(self.site_href, self._ref_text(text))


class Ablative_Analysis(GlossaryTerm):
    """
    Ablative analysis in PlanZero is the study of models with and without
    certain elements (namely strategies), in order to characterize the impact
    of those elements on the behaviour of the whole model.
    """

    @property
    def aka(self) -> list[str]:
        return ['Ablation Study']

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Strategy': 'Strategies within models are are ablated in order to assess their impact.',
            'Simulation': 'Ablative analysis is part of model simulation.',
        }


class About_Section(GlossaryTerm):
    """<p>The "About" section of the planzero.ca website:
    <a href="/about/">planzero.ca/about</a></p>"""

    @property
    def aka(self) -> list[str]:
        return ['About Page']


class Barrier(GlossaryTerm):
    """<p>Barrier is a term adapted from {{lref("EGFS")|safe}},
    where it means <i>an existing or potential challenge that hinders the achievement of one or more objectives</i>.
    In PlanZero, the term is used on per-sector analysis pages to call out
    key challenges in reducing emissions in each sector.
    </p>
    <p>
    In the future, Barrier call-outs on per-sector analysis pages may link to specific
    structural elements within certain models, which formalize some aspect of the Barrier.
    For now, they are not formally part of PlanZero models.
    </p>
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
                'Strategy': "a related EGFS term, for a project or initiative that could address a barrier",
                'IPCC_Sector': "An NIR emissions reporting sector, compatible with IPCC reporting guidelines",
                'Structural_Element': "a Model Element that defines variables within a PlanZero model",
                'EGFS': "the Barrier term and definition are adapted from The Executive Guide to Facilitating Strategy",
        }


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
            'Inference_Algorithm': 'In probabilistic models that are defined within probabilistic programming systems, Bayesian inference can often be implemented automatically',
        }

    @property
    def as_discussed_in_posts(self) -> list[tuple[object, str, str]]:
        return [
            #(blog.TwoProbabilisticModels(), '#', "see most of first half of post"),
        ]


class CNZEAA(GlossaryTerm):
    """The Canadian Net-Zero Accountability Act legislates federal ministries and agencies
    to uphold Canada's obligations under the Paris Agreement.
    """

    @property
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


class Cashflow_Variables(GlossaryTerm):
    """Cashflow variables are the set of variables in a PlanZero model related to cashflow.
    The Posterior phase values of cashflow variables are used to
    calculate statistics and render summary graphics on the PlanZero site.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'PlanZero_Model': 'May have cashflow variables, or not',
            'Special_Project_Entity': 'Cashflow in a PlanZero model is between a pair of SPEs',
        }

    @property
    def as_discussed_in_posts(self) -> list[tuple[object, str, str]]:
        return [
            (blog.Glossary(), '#semantic-variable-taxonomy', "Taxonomy of Model Variables by semantics"),
        ]


class Code(GlossaryTerm):
    """PlanZero is an open-source project, with [source] code on GitHub."""
    @property
    def see_also(self) -> dict[str, str]:
        return {
            "GitHub_Repository": "The development hub for PlanZero on GitHub",
            "Python": "PlanZero's source code is mostly written in the Python programming language",
        }


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
                'Probabilistic_Model': 'Probabilistic models can be used to derive conditional distributions among their variables',
                }


class Credible_Interval(GlossaryTerm):
    """A <a href="https://en.wikipedia.org/wiki/Credible_interval">credible interval</a> is the set of values that a
    random variable in a probabilistic model might most-credibly take.
    For example, a 95% credible interval is the smallest interval containing
    the 95% most-probable values for the random variable.
    </p>
    <p>
    In PlanZero, Credibility and Confidence are used interchangeably.
    In the field of statistics, the terms have distinct meanings that apply to
    Bayesian and frequentist estimators respectively.
    PlanZero strives to use Bayesian reasoning,
    but with MCMC methods there can be significant uncertainty due to the
    limited sample size, which is the the sort of uncertainty that frequentist
    methods deal with. Candidly I don't honestly know whether either term is
    more appropriate for the estimates that arise from PlanZero modelling.
    """

    @property
    def aka(self) -> list[str]:
        return ['Confidence Interval']

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Unknown_Variable': 'the model element associated with a credible interval',
            'Probabilistic Model': 'credible intervals arise only in probabilistic modelling',
            'Inference_Algorithm': 'the main use of credible intervals is to characterize the uncertainty remaining after inference of the latent random variables in a model',
        }

    @property
    def as_discussed_in_posts(self) -> list[tuple[object, str, str]]:
        return [
            (blog.ProbabilisticNIR2025(), '#appendix-visualizing', "credibility intervals are used to visualize probabilistic models"),
        ]


class Critical_Success_Factor(GlossaryTerm):
    """<p>Critical Success Factor (CSF) is a term adapted from {{lref("EGFS")|safe}},
    where it means a
    <i>key condition that must be created to achieve one or more objectives</i>.
    Wikipedia offers additional context on the 
    <a href="https://en.wikipedia.org/wiki/Critical_success_factor">history of the term</a>.
    In PlanZero, the term is used on per-sector analysis pages to call out
    necessary conditions to achieve net-zero (or net-negative) emissions in that sector.
    </p>
    <p>
    In the future, these call-outs may be formalized within PlanZero to link to Strategy Elements
    involved in achieving CSFs in particular models.
    For now, they are not formally part of PlanZero models.
    </p>
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
                'IPCC_Sector': "An NIR emissions reporting sector, compatible with IPCC reporting guidelines",
                'KPI': 'Key Performance Indicators sometimes provide scales on which Critical Success Factors are defined as thresholds',
                'EGFS': "the CSF term and definition are adapted from The Executive Guide to Facilitating Strategy",
        }

    @property
    def aka(self) -> list[str]:
        return ['CSF']


class Draft_Status(GlossaryTerm):
    """A PlanZero Post may be in Draft Status,
    in which case it is still subject to significant change.
    A post that is not in Draft Status should not be materially changed,
    they should only be changed to include
    clarifications and annoted links to relevant newer content.
    """

    @property
    def as_discussed_in_posts(self) -> list[tuple[object, str, str]]:
        return [
            (blog.About(), '#draft-status', "see section on Draft Status"),
        ]

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'About_Section': 'more guidelines around draft status and posts generally',
            'Post': 'Narrative descriptions of updates to PlanZero, which may be in Draft Status if the work is still in progress',
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

    @property
    def as_discussed_in_posts(self) -> list[tuple[object, str, str]]:
        return [
            (blog.Glossary(), '#model-elements', "see strategic management and communication in the Model Elements section within Appendix 1"),
        ]


class Emission_Factor(GlossaryTerm):
    """<p>
    An emission factor is a constant of proportionality between
    an emission driver and the amount of some emitted greenhouse gas.
    </p>"""

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'KPI': "an emission factor is one of PlanZero's base KPI types",
            'PlanZero_Model': "a model of Canada's future emissions",
            'Greenhouse Gas': "an emission factor is a constant of proportionality to one greenhouse gas",
            'Emissions': "the result of multiplying an emission factor by a driver",
        }


class Emission_Reduction_Efficiency(GlossaryTerm):
    """A strategy's emission reduction efficiency is the cost (in CAD) per ton of abated {{CO2e}}"""

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Strategy_Element': 'emission reduction efficiency is one of the key metrics associated with strategies',
        }


class Emission_Variables(GlossaryTerm):
    """Emission variables are the set of variables in a PlanZero model related to emissions.
    The Posterior phase values of emission variables are used to
    calculate statistics and render summary graphics on the PlanZero site.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'PlanZero_Model': 'Always have a full set of emission variables',
        }

    @property
    def as_discussed_in_posts(self) -> list[tuple[object, str, str]]:
        return [
            (blog.Glossary(), '#semantic-variable-taxonomy', "Taxonomy of Model Variables by semantics"),
        ]


class Emissions(GlossaryTerm):
    """The combined emissions of the seven greenhouse gases, often quantified
    in units of {{CO2e|safe}}.
    Canada estimates rates of emissions in the
    {{lref("National Greenhouse Gas Inventory")|safe}}.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Emission_Factor': 'a factor of proportionality between drivers and emissions',
            'PlanZero_Model': "a model of Canada's future emissions, calculated by summing up emissions across IPCC Sectors",
            "IPCC_Sector": "the lowest levels of granularity in National Greenhouse Gas Inventory",
            "National Greenhouse Gas Inventory": "the data from which NIRs are generated"
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

    @property
    def aka(self) -> list[str]:
        return ['ECCC', '"E-triple-C"']

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Natural_Resources_Canada': 'peer federal ministry',
            "NIR": "National Inventory Reports are prepared by ECCC",
            "NGGI": "National Greenhouse Gas Inventory is maintained by ECCC",
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


class GitHub(GlossaryTerm):
    """<p>GitHub (<a href="https://github.com/">site</a>, <a href="https://en.wikipedia.org/wiki/GitHub">wikipedia</a>) is a web service for using the git version
    control system over the internet to collaborate on software projects. Circa 2023, it was the world's largest source code host, with over 100 million developers, and 420 million code repositories.</p>
    """

    @property
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


class GitHub_Issue(GlossaryTerm):
    """GitHub issues, such as the <a href="https://github.com/jaberg/planzero/issues">PlanZero issues</a>,
    are notices / reminders of future work.
    Issues are used differently by different projects, some projects don't use
    GitHub's issues at all.
    PlanZero uses GitHub issues to record a variety of ideas and To-Do items,
    but in particular, those that have been mentioned in posts.
    """

    @property
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

    @property
    def aka(self) -> list[str]:
        return ['pull request', 'PR']

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Git_Branch': "a pull request is a request to merge two branches",
            'Main_Branch': "submit a pull request to this branch when a new development is ready",
            'GitHub_Fork': "typically a pull request represents a request to merge code from a branch in one fork (maintained by one person) into a branch on another fork (maintained by another person)"
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

    @property
    def aka(self) -> list[str]:
        return ['GitHub Repo', 'repository', 'repo']

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'GitHub': "the site that hosts GitHub code repositories",
            'Git': "the version control system upon which GitHub operates",
            'GitHub_Fork': "A repo can be forked to create a downstream copy of an upstream repo"
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

    @property
    def aka(self) -> list[str]:
        return ['Commit']


class Git_Graph(GlossaryTerm):
    """A set of commits and merges that build on one another form a graph
    representing all of the development on a project.
    Try visualizing the graph for any git project by using purpose-built visual tools, 
or running a command such as <pre>git log --all --decorate --oneline --graph</pre>
    """

    @property
    def aka(self) -> list[str]:
        return ['Git History']

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Git_Commit': "a node of a git [change] graph",
            'GitHub Repo': "a git graph hosted on GitHub",
            'Git_Branch': "a named subgraph of ancestors of a particular commit",
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

    @property
    def as_discussed_in_posts(self) -> list[tuple[object, str, str]]:
        return [
            (blog.GHG_Emissions(), '#h2_ghg', "see section on Greenhouse gases"),
        ]


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
            "PlanZero_Model": "A PlanZero Model models emission amounts for IPCC Sectors",
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
            'PlanZero_Model': "a model of Canada's future emissions",
            'Emission Factor': 'the factor of proportionality between a driver and an IPCC Sector Contribution (generally a time series)',
            'IPCC Sector': (
                'the emissions for an IPCC Sector are calculated'
                ' by summing one or more IPCC Sector Contributions'),
            'KPI': 'IPCC Sector Contributions are Derived KPIs',
        }


class Inference_Algorithm(GlossaryTerm):
    """An inference algorithm samples from the posterior distribution of latent variables in a probabilistic model,
    often by applying Bayes' rule systematically across a formal specification of the model.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Probabilistic_Model': 'Inference algorithms operate on probabilistic models',
            'Posterior_Distribution': 'Inference algorithms sample from posterior distributions',
            'Latent_Variable': 'Inference algorithms provide samples for latent variables'
        }

    @property
    def as_discussed_in_posts(self) -> list[tuple[object, str, str]]:
        return [
            #(blog.TwoProbabilisticModels(), '#', "see most of first half of post"),
        ]


class Inference_Group(GlossaryTerm):
    """
    An inference group is a set of structural and strategy elements of
    a PlanZero model for which an inference algorithm can operate independently
    of other inference groups.
    PlanZero models are large, and inference groups serve to define smaller submodels
    upon which inference algorithms can operate effectively.
    """

    @property
    def as_discussed_in_posts(self) -> list[tuple[object, str, str]]:
        return [
                (blog.Glossary(), '#inference-groups', "Section: Inference Groups in Appendix 1: Model Architecture"),
        ]


class Inference_Phase(GlossaryTerm):
    """
    Inference is the second of the three phases of evaluating a PlanZero model.
    Inference is the relatively-long computation of sampling values for unobserved variables
    according to their posterior distribution under the model.
    Inference typically uses an MCMC sampling algorithm.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Model_Evaluation_Phase': 'Prior, Inference, and Posterior',
            'Unobserved_Variable': 'A variable with a posterior distribution in a PlanZero model',
            'Observed_Variable': 'A variable associated with conditioning data in a PlanZero model',
        }

    @property
    def as_discussed_in_posts(self) -> list[tuple[object, str, str]]:
        return [
                (blog.Glossary(), '#phases', "Appendix 2: Phases of inference"),
        ]


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

    @property
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
                'Probabilistic_Model': 'Probabilistic models define joint distributions over variables',
                }


class Key_Performance_Indicator(GlossaryTerm):
    """
    A Key Performance Indicator, in general, is any metric that is an important
    measure of system, or organization performance.
    KPIs are not formally part of PlanZero modelling, but PlanZero Models are meant to include model variables
    corresponding to many KPIs of Canada's emissions and emissions-reduction plans.
    """

    @property
    def aka(self) -> list[str]:
        return ['KPI', 'Base KPI', 'Derived KPI']

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'CSF': "Sometimes it is easy to express a CSF in terms of a KPI (e.g. emissions &le; 0)",
            'Emission_Variables': "Emission Variables model KPIs of Canada's emission-reduction plans",
        }


class Known_Variable(GlossaryTerm):
    """A known variable is a scalar- or vector-valued
    term in a PlanZero model that is known from the outset of modelling,
    with no dependency on an unknown variable.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Unknown_Variable': 'A variable modelled in terms of a probability distribution',
            'Model_Variable': 'the more general class of variables in a PlanZero Model'
        }

    @property
    def as_discussed_in_posts(self) -> list[tuple[object, str, str]]:
        return [
                (blog.StaticNormals(), '#appendix-notation', "Appendix 1: Probabilistic Modelling Notation"),
        ]


class Latent_Variable(GlossaryTerm):
    """A latent variable in a probabilistic model is one that remains unobserved when the model is conditioned on data.
    Each time step in a time series variable is a unique variable,
    so earlier time steps might be observed and later time steps (especially future time steps)
    may be latent.
    """
    @property
    def see_also(self) -> dict[str, str]:
        return {
                'Inference_Algorithm': 'Inference of latent variables can sometimes be done by standard algorithms.',
                'Posterior_Distribution': 'A posterior distribution is over latent variables.',
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


class Model_Construction(GlossaryTerm):
    """Model construction is the computational work of
    using model elements to define variables in a PlanZero model.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Model_Construction_Subphase': "The subphases of model construction",
            'Model_Element': 'Model elements provide variable definitions',
            'Model_Variable': 'Model variables provide connectivity between model elements'
        }

    @property
    def as_discussed_in_posts(self) -> list[tuple[object, str, str]]:
        return [
                (blog.Glossary(), '#subphases', "Appendix 2: Model-construction subphases"),
        ]


class Model_Construction_Subphase(GlossaryTerm):
    """Model construction subphases are preparation (before annual iterations),
    annual iterations, and post-processing (after annual iterations). Model elements
    can participate in these subphases to define model variables. The kinds of variables
    that can be defined (and accessed) differ from phase to phase.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Model_Construction': "Is made up of subphases of model construction",
            'Model_Element': 'Model elements provide variable definitions by participating in one or more construction subphases',
            'Model_Variable': 'The rules for Model variable definitions and access differ from subphase to subphase'
        }

    @property
    def as_discussed_in_posts(self) -> list[tuple[object, str, str]]:
        return [
                (blog.Glossary(), '#subphases', "Appendix 2: Model-construction subphases"),
        ]


class Model_Construction_Subphase_Annual_Iteration(GlossaryTerm):
    """The model construction subphase of annual iteration defines annual variables and carry variables.
    Model element logic operating in this subphase can access any variable defined in the preparation subphase,
    and some current and next values of annual and carry variables, so long as all the definitions have a dependency ordering.
    The annual iterations are handled with NumPyro's scan operation, so that a PlanZero model can be configured to forecast
    any number of future years.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Model_Construction': "Is made up of subphases of model construction",
        }

    @property
    def aka(self) -> list[str]:
        return ['Simulation']


    @property
    def as_discussed_in_posts(self) -> list[tuple[object, str, str]]:
        return [
                (blog.Glossary(), '#subphases', "Appendix 2: Model-construction subphases"),
        ]


class Model_Element(GlossaryTerm):
    """An implementation element of a PlanZero model,
    providing one or more computationl subphase implementations that define model variables.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'PlanZero_Model': "the general form of PlanZero probabilistic models",
        }

    @property
    def code_refs(self) -> dict[str, object]:
        from .nir_model import NIR_ModelElement
        from .prob import SiteModelElement
        from .scenario_model import ModelElement
        return {
                'ModelElement (basic model participation)': ModelElement,
                'NIR ModelElement (define emissions and cashflow)': NIR_ModelElement,
                'Site ModelElement (page documentation elements)': SiteModelElement,
        }



class Model_Evaluation(GlossaryTerm):
    """Model evaluation is the computational work of
    determining statistics of the variables of a PlanZero model.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
                'PlanZero_Model': "Model evaluation refers implicitly to PlanZero model evaluation",
                'Model_Evaluation_Phase': "PlanZero performs model evaluation in three broad phases: Prior, Inference, and Posterior",
                'Model_Construction_Subphase': "Each evaluation phase builds some or all of a PlanZero model using the same construction subphases",
                'Model_Metric': 'Model-level metrics are calculated in the last evaluation phase'
        }

    @property
    def as_discussed_in_posts(self) -> list[tuple[object, str, str]]:
        return [
                (blog.Glossary(), '#appendix-semantics-evaluation', "Appendix 2: Model Semantics and Evaluation"),
        ]

class Model_Evaluation_Phase(GlossaryTerm):
    """There are three phases of PlanZero model evaluation: Prior, Inference, and Posterior.
    The prior evaluation is quick, and used primarily for basic model validation.
    Inference is slow, and done piecewise on Inference Groups to draw samples from the posterior
    distribution of unobserved variables.
    The Posterior phase is a reconstruction of the model using the posterior samples instead of prior distributions
    to define unknown variables.
    Statistics of the Posterior phase are presented on this site as the results of model evaluation.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Inference_Phase': "The inference phase generally uses an MCMC algorithm to draw posterior samples from PlanZero model",
        }

    @property
    def as_discussed_in_posts(self) -> list[tuple[object, str, str]]:
        return [
                (blog.Glossary(), '#phases', "Appendix 2: Phases of inference"),
        ]


class Model_Explorer(GlossaryTerm):
    """The <a href="/model-explorer/">Model Explorer</a> is a resource here on the site
    that supports interactive exploration of inference groups and model element
    interaction via variables.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
                'PlanZero_Model': "The Model Explorer works on PlanZero models",
                'Inference_Group': "The Model Explorer looks at one inference group at a time",
                'Model_Explorer_Trace': "a detailed view of elements and variables in terms of model construction subphases",
                'Model_Explorer_Overview': "an inference-group-level view of connectivity between model elements",
        }

    @property
    def as_discussed_in_posts(self) -> list[tuple[object, str, str]]:
        return [
                (blog.Glossary(), '#overview', "Overview and Appendix 1"),
        ]


class Model_Explorer_Overview(GlossaryTerm):
    """The Model Explorer - Overview tab is a visualization
    tab within the Model Explorer that shows the interaction of
    model elements within an inference group.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
                'Inference_Group': "The Model Explorer looks at one inference group at a time",
                'Model_Explorer': "an interactive visualization tool on the PlanZero site",
        }

    @property
    def as_discussed_in_posts(self) -> list[tuple[object, str, str]]:
        return [
                (blog.Glossary(), '#overview', "Overview and Appendix 1"),
        ]


class Model_Explorer_Trace(GlossaryTerm):
    """The Model Explorer - Trace tab is a visualization
    tab within the Model Explorer that shows the interaction of
    model elements and variables as a result of model construction.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
                'Inference_Group': "The Model Explorer looks at one inference group at a time",
                'Model_Explorer': "an interactive visualization tool on the PlanZero site",
        }

    @property
    def as_discussed_in_posts(self) -> list[tuple[object, str, str]]:
        return [
                (blog.Glossary(), '#overview', "Overview and Appendix 1"),
        ]



class Model_Metric(GlossaryTerm):
    """A formula, procedure or rule for associating
    a single number to a {{lref("PlanZero_Model")|safe}}.
    For example, NIR models could be evaluated in terms of
    prediction accuracy."""

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Weighted_KL_Divergence': "A prediction-quality metric for PlanZero Models",
        }


class Model_Variable(GlossaryTerm):
    """A model variable is a known or unknown variable in a PlanZero Model.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'PlanZero_Model': "the general form of models on the PlanZero site",
            'Model_Element': "the unit of modelling assumptions that relate variables in a PlanZero Model",
        }


class NIR2025_Model(GlossaryTerm):
    """The NIR2025 model is a PlanZero model, whose defining feature
    is that it strictly reflects the data of the NIR-2025 without imposing any
    modelling assumptions upon it.
    """


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

    @property
    def as_discussed_in_posts(self) -> list[tuple[object, str, str]]:
        return [
            (blog.StaticNormals(), '#appendix-prenir', "Introduces the Pre-NIR-2025-06 prediction challenge, which uses a weighted KL-divergence to compare model predictions to reference distributions."),
        ]

    @property
    def aka(self) -> list[str]:
        return ['PreNIR']

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Weighted_KL_Divergence': 'The scoring method for model predictions',
            'Prediction_Challenge': 'NIR prediction is the only type of prediction challenge in PlanZero so far',
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

    @property
    def aka(self) -> list[str]:
        return ['NEUD']

    @property
    def code_refs(self) -> dict[str, object]:
        from . import neud
        return {
            'NEUD access, neud.py': neud,
        }

    @property
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


class National_Greenhouse_Gas_Inventory(GlossaryTerm):
    """Canada's
    <a
    href="https://www.canada.ca/en/environment-climate-change/services/climate-change/greenhouse-gas-emissions/inventory.html">National
    Greenhouse Gas Inventory</a> tracks emissions
    across Canada, and is the basis for the annual
    {{lref("NIR", "National Inventory Reports")|safe}}.
    """

    @property
    def aka(self) -> list[str]:
        return ['NGGI']

    @property
    def see_also(self) -> dict[str, str]:
        return {
            "ECCC": "Environment and Climate Change Canada maintains and publishes the NGGI",
            "NIR": "National Inventory Reports are prepared from the NGGI",
        }


class National_Inventory_Report(GlossaryTerm):
    """A National [Greenhouse Gas] Inventory Report,
    is published annually by Environment and Climate Change Canada, and
    submitted to the UNFCCC Secretariat.
    Specific reports are referred to as e.g. NIR-2023, NIR-2024, NIR-2025,
    an so on in PlanZero.
    The first report was <a href="https://publications.gc.ca/site/eng/9.506002/publication.html">NIR-2004</a>.
    """

    @property
    def aka(self) -> list[str]:
        return ['NIR']

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


class Natural_Resources_Canada(GlossaryTerm):
    """<p><a href="https://natural-resources.canada.ca/">Natural Resources
    Canada</a> (sometimes, NRCan) is a federal ministry
    "<i>Committed to improving the quality of
    life of Canadians by ensuring the country’s abundant natural resources are
    developed sustainably, competitively and inclusively.</i>".
    </p>
    """

    @property
    def aka(self) -> list[str]:
        return ['NRCan']

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'ECCC': 'Environment_and_Climate_Change_Canada is a peer federal ministry',
            'NEUD': "NRCan maintains and publishes the National Energy Use Database",
        }


class Net_Zero(GlossaryTerm):
    """Net-zero is a hypothetical state, in which
    greenhouse gas {{lref("Emissions", "emissions")|safe}} across the sectors of an economy, average
    out to zero.
    The goal of the PlanZero project is to explore models of how Canada might
    achieve net-zero.
    """

    @property
    def aka(self) -> list[str]:
        return ['Net-Zero']

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'ECCC': 'The ECCC is preparing a plan for Canada to reach net-zero',
            "Paris Agreement": "The Paris Agreement implores countries to reach net-zero by mid-century",
            "PlanZero": "PlanZero is this project, to model how Canada might reach net-zero",
        }


class Observed_Variable(GlossaryTerm):
    """An observed variable is an unknown variable for which we have
    data that is modelled a sample from the variable's distribution.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Unobserved_Variable': 'A variable for which we have no direct reading of its probable value',
            'Unknown_Variable': 'A variable in a probabilistic model',
        }

    @property
    def as_discussed_in_posts(self) -> list[tuple[object, str, str]]:
        return [
                (blog.Glossary(), '#model-variables', "Section: Model Variables in Appendix 1: Model Architecture"),
        ]


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

    @property
    def as_discussed_in_posts(self) -> list[tuple[object, str, str]]:
        return [
            (blog.registry['StaticNormals'], '#appendix-prenir', "prediction time vs. evaluation time"),
        ]


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
    
    @property
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


class PlanZero_Model(GlossaryTerm):
    """A PlanZero Model is the general form of model on the PlanZero site.
    It is a graph of Model Variables and Model Elements,
    the Unknown Variables of which make up a probabilistic model.
    A PlanZero Model includes variables for annual emissions matching the structure of recent
    National Inventory Reports, as well as variables for annual cashflow between economic entities called Special Project Entities (SPEs)
    as a result of any strategies in the model, as well as any other variables involved in modelling emissions and cashflows.
    """

    @property
    def aka(self) -> list[str]:
        return ['NIR Model', 'Model']

    @property
    def hover_definition_html(self) -> str:
        return "<p>A PlanZero Model is the general form of model on the PlanZero site.</p>"

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Model_Variable': "a scalar- or vector-valued variable in a model",
            'Model_Element': "a set of assumptions for defining one or more variables in a model",
            'Probabilistic_Model': "the joint distribution over unknown variables in a model correspond to a directed probabilistic graphical model",
        }

    @property
    def as_discussed_in_posts(self) -> list[tuple[object, str, str]]:
        return [
            (blog.Glossary(), '#overview', "this entire post is dedicated to documenting PlanZero Models"),
        ]


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


class Post(GlossaryTerm):
    """A post on this site (see e.g. <a href="/posts/">https://planzero.ca/posts/</a>).
    </p><p>
    The <a href="/about/">About</a> page
    includes some guidelines for post content.
    """

    @property
    def as_discussed_in_posts(self) -> list[tuple[object, str, str]]:
        return [
            (blog.About(), '#stability', "see several sections regarding the handling of Posts"),
        ]


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


class Posterior_Phase(GlossaryTerm):
    """
    The Posterior phase is the last of the three phases of evaluating a PlanZero model.
    The Posterior phase is a reconstruction of the model using the posterior samples instead of prior distributions
    to define unknown variables.
    Statistics of the Posterior phase are presented on this site as the results of model evaluation.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Model_Evaluation_Phase': 'Prior, Inference, and Posterior',
            'Unobserved_Variable': 'A variable with a posterior distribution in a PlanZero model',
        }

    @property
    def as_discussed_in_posts(self) -> list[tuple[object, str, str]]:
        return [
                (blog.Glossary(), '#phases', "Appendix 2: Phases of inference"),
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


class Predictive_Model(GlossaryTerm):
    """
    A model that includes probability distributions for events in the future is called a <i>predictive</i> model.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'NIR2025_Model': 'The NIR2025 model is not a predictive PlanZero model',
            'Scaling_Model': 'The Scaling model is a predictive PlanZero model',
        }


class Prior_Phase(GlossaryTerm):
    """
    The Prior phase is the first of the three phases of evaluating a PlanZero model.
    The Prior phase establishes prior distributions over unknown variables,
    and validates that inference groups are statistically independent.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Model_Evaluation_Phase': 'Prior, Inference, and Posterior',
            'Unknown_Variable': 'All unknown variables are associated with a prior distribution by a PlanZero model',
        }

    @property
    def as_discussed_in_posts(self) -> list[tuple[object, str, str]]:
        return [
                (blog.Glossary(), '#phases', "Appendix 2: Phases of inference"),
        ]


class Probabilistic_Model(GlossaryTerm):
    """A probabilistic model, in general, is a specification of probability distribution over one or more unknown variables.
    In PlanZero, it typically means the distribution of probability
    over the unknown variables in a PlanZero model.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
                'Probability_Distribution': 'A model defines a joint probability distribution over all its variables',
                'PlanZero_Model': 'PlanZero models are probabilistic models, encompassing NIR emissions data',
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
    PlanZero uses this standard term in the standard way, as defined more fully by e.g.
    <a href="https://en.wikipedia.org/wiki/Probability_distribution">wikipedia</a>
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Unknown_Variable': 'a variable whose true value is not known, but with which we can associate a probability distribution',
            'Probability_Density_Function': 'a description of how to distribute probability over a continuous set of possible values, such as a range of real numbers',
        }

    @property
    def as_discussed_in_posts(self) -> list[tuple[object, str, str]]:
        return [
            (blog.ProbabilisticNIR2025(), '#appendix-models', "Appendix 1 is an introduction to probabilistic modelling"),
        ]


class Python(GlossaryTerm):
    """<p>PlanZero is implemented in the <a href="https://www.python.org/">Python programming language</a></p>"""

    @property
    def see_also(self) -> dict[str, str]:
        return {
            "GitHub_Repository": "The development hub for PlanZero on GitHub",
            "Code": "PlanZero's source code is mostly written in the Python programming language",
        }


class Scaling_Model(GlossaryTerm):
    """The Scaling model is a PlanZero model, whose defining feature
    is that it includes hypothetical adoption and standing of sufficiently-mature
    technology and products.
    """


class Scenario(GlossaryTerm):
    """A scenario is a set of time series covering a common time period.
    Typically in PlanZero it is the result of simulating a model.
    """


class Special_Project_Entity(GlossaryTerm):
    """SPEs are abstract economic entities with cash balances.
    SPEs represent non-overlapping sets of actual or hypothetical individuals and corporate entities.
    """

    @property
    def aka(self) -> list[str]:
        return ['SPE']

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Emission_Reduction_Efficiency': 'for a strategy, the cost (in CAD) per ton of abated emissions',
            }

    @property
    def as_discussed_in_posts(self) -> list[tuple[object, str, str]]:
        return [
                (blog.Glossary(), '#cashflow-variables', "SPEs are introduced in relation to cashflow variables"),
        ]


class Static_Normals_Model(GlossaryTerm):
    """
    The Static Normals model is a PlanZero model, whose defining feature
    is that it assumes no change over time of any emissions, and attributes
    all year-to-year differences in the data to observation noise.
    """



class Statistics_Canada(GlossaryTerm):
    """<a href="https://www.statcan.gc.ca/en/start">Statistics Canada</a>,
    Canada's national statistical office.
    Much of the data used in PlanZero is drawn from Statistics Canada tables,
    downloaded via open-source project
    <a href="https://github.com/ianepreston/stats_can">stats_can</a>.
    """
    
    @property
    def aka(self) -> list[str]:
        return ['StatsCan']

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'NGGI': 'The National Greenhouse Gas Inventory draws on data products from Statistics Canada',
            'NEUD': 'The National Energy Use Database draws on data products from Statistics Canada',
        }


class Strategy(GlossaryTerm):
    """<p>Strategy is a term adapted from {{lref("EGFS")|safe}} (which aims near the standard usage definition),
    where it means
    a <i>broad activity required to achieve a goal, create a critical condition, or overcome a barrier</i>.
    In PlanZero, the term is used in two ways.
    Firstly, it is used on per-sector analysis pages to call out things that could be done
    to reduce emissions in a sector.
    Secondly, it is sometimes used to describe Strategy Elements, which are formal model elements
    used to evaluate strategy ideas within PlanZero models.
    </p>
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Barrier': 'a related EGFS term, for hinderances to achieving a goal, which may be overcome by strategies',
            'EGFS': 'The Executive Guide to Facilitating Strategy, providing strategic management term definitions',
        }


class Strategy_Element(GlossaryTerm):
    """A strategy element in a PlanZero model stands for
    a project or initiative that may be undertaken (or have been undertaken) in the real world (that is, outside the model).
    PlanZero model evaluation includes the assessment of strategy <i>impact</i>,
    by evaluating and comparing variants with and without the strategy.
    """

    @property
    def aka(self) -> list[str]:
        return []

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Model_Element': 'Strategy elements are a type of Model Element',
            'Ablative_Analysis': 'The impact of strategy elements is assessed by ablative analysis',
            'Structural_Element': 'Structural elements are the other type of Model Element',
        }


class Structural_Element(GlossaryTerm):
    """Structural elements are model elements within scenario models that stand for
    the modelled relationship between variables.
    A scenario model must include structural elements, and
    unlike strategy elements, structural elements will not be ablated.
    Typically, most (if not all) elements in a model are structural ones.
    """

    @property
    def aka(self) -> list[str]:
        return []

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Model_Element': 'Structural elements are a type of Model Element',
            'Strategy_Element': 'Strategy elements work by overriding default values accessed by structural elements'
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

    @property
    def aka(self) -> list[str]:
        return ['SBLN']

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Probability_Distribution': 'The SBLN is a probability distribution',
        }

    @property
    def as_discussed_in_posts(self) -> list[tuple[object, str, str]]:
        return [
            (blog.ProbabilisticNIR2025(), '#appendix-sbln', "Probabilistic NIR2025, Appendix 2"),
        ]


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


class Unknown_Variable(GlossaryTerm):
    """An unknown variable is an unknown scalar- or vector-valued
    term in a probabilistic model.
    The model provides a probability distribution over its possible values.
    That distribution may depend on the values of other known and unknown variables in the model.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Probabilistic_Model': 'A model associates unknown variables with probability distributions',
            'Probability_Distribution': 'In order to reason about a variable without knowing exactly what it is, we assume a set of values that it might have using a probability distribution.',
            'Observed_Variable': 'A variable for which we have some [generally-]noisy data exemplifying its probable value',
            'Unobserved_Variable': 'A variable for which we have no direct data',
        }

    @property
    def as_discussed_in_posts(self) -> list[tuple[object, str, str]]:
        return [
                (blog.StaticNormals(), '#appendix-notation', "Appendix 1: Probabilistic Modelling Notation"),
        ]

    @property
    def aka(self) -> list[str]:
        return ['Random Variable']


class Unobserved_Variable(GlossaryTerm):
    """An unobserved variable is an unknown variable for which we have no
    direct readings to inform a probability distribution over its possible values.
    Inference algorithms draw samples for unobserved variables based on data
    associated with observed variables.
    """

    @property
    def see_also(self) -> dict[str, str]:
        return {
            'Observed_Variable': 'A variable for which we have some [noisy] data exemplifying its probable value',
            'Unknown_Variable': 'A variable in a probabilistic model',
        }

    @property
    def as_discussed_in_posts(self) -> list[tuple[object, str, str]]:
        return [
                (blog.Glossary(), '#model-variables', "Section: Model Variables in Appendix 1: Model Structure"),
        ]


class Weighted_KL_Divergence(GlossaryTerm):
    """In PlanZero, the term "weighted KL divergence" typically refers
    to the specific weighting of KL divergences between actual and predicted
    emissions that is used to score NIR-predicting models probabilistic models.
    """

    @property
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
