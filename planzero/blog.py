import enum
from pydantic import BaseModel
import datetime

from . import enums
from . import est_nir

_classes = []
_blogs_by_url_filename = {}
_blogs_sorted_by_date = []


class BlogTag(str, enum.Enum):
    NIR_Modelling = "NIR Modelling"
    ScalingStrategies = 'Scaling Strategies'
    Scenarios = 'Scenarios'
    About = 'About'


class BlogPost(BaseModel):

    date: datetime.datetime
    title: str
    about: str
    url_filename: str
    author: str
    published: bool = True
    draft: bool = False # for published posts, show draft text anyway
    concept_only: bool = False # there is no html for this post object
    tags: set[str] = set()

    def __init__(self, **kwargs):
        if 'about' not in kwargs:
            kwargs = dict(kwargs, about=self.__class__.__doc__)
        super().__init__(**kwargs)

    @classmethod
    def __init_subclass__(cls):
        super().__init_subclass__()
        _classes.append(cls)


from .planet_model import emissions_impulse_response_project_evaluation
from . import enums
from .ureg import u

from io import StringIO
from .html import HTML_element
from .html import HTML_Math_Latex
import matplotlib.pyplot as plt
from . import ipcc_canada


def latex(latex, display='inline'): # display inline or block
    return HTML_Math_Latex(latex=latex, display=display).as_html()


class HTML_Matplotlib_Figure(HTML_element):

    def as_html(self):
        self.build_figure()
        svg_buffer = StringIO()
        plt.savefig(svg_buffer, format="svg")
        plt.close()
        svg_string = svg_buffer.getvalue()
        return svg_string


class UncertaintyReductionForCattleEnteric(BlogPost):
    # Reducing Uncertainty with a better model

    """
    introduces a new scenario based on statistical modelling and
    extrapolation of current trends. This "extrapolating" scenario is
    especially useful for near-term forecasting (near-casting).
    Near-casting can be more accurate than simply
    re-using prior-year estimates because Statistics Canada releases some indicator
    variables with less delay than the ECCC releases the annual NIR.
    """
    def __init__(self):
        super().__init__(
            date=datetime.datetime(2026, 4, 21),
            title='Uncertainty in Scenario Forecasting',
            url_filename="2026-04-21-nearcasting", # rename?
            author="James Bergstra",
            tags={BlogTag.Scenarios, BlogTag.NIR_Modelling},
            concept_only=True,
            published=False,
            draft=True,
            )


class Uncertainty(BlogPost):

    """
    introduces a new scenario based on statistical modelling and
    extrapolation of current trends. This "extrapolating" scenario is
    especially useful for near-term forecasting (near-casting).
    Near-casting can be more accurate than simply
    re-using prior-year estimates because Statistics Canada releases some indicator
    variables with less delay than the ECCC releases the annual NIR.
    """
    def __init__(self):
        super().__init__(
            date=datetime.datetime(2026, 4, 21),
            title='Uncertainty in Scenario Forecasting',
            url_filename="2026-04-21-nearcasting", # rename?
            author="James Bergstra",
            tags={BlogTag.Scenarios, BlogTag.NIR_Modelling},
            concept_only=True,
            published=False,
            draft=True,
            )

class GPR_Extrapolation(BlogPost):
    """This post introduces probabilistic forecasting to PlanZero.
    There are lots of ways the future could go!
    PlanZero aspires to model a representative set of futures in terms of the
    sorts of time series that make up its scenarios.
    This post introduces the statistical modelling technique of Gaussian Process Regression
    to generate future scenarios that similar in mean, variance, and
    rate of change to the past.
    These models don't incorporate driving factors such as climate influences,
    foreign exchange rates, international supply and demand, or trade policies,
    but they probably improve the obviously-wrong-looking flat lines currently
    extending to the right of year 2023 for each emission sector.
    """
    # * extends models
    #
    #

    def __init__(self):
        super().__init__(
            date=datetime.datetime(2026, 4, 25),
            title='Improving Baseline Emission Estimates with Gaussian Process Regression',
            url_filename="2026-04-25-GPR", # rename?
            author="James Bergstra",
            tags={BlogTag.Scenarios, BlogTag.NIR_Modelling},
            concept_only=True,
            published=False,
            draft=True,
            )

class Glossary(BlogPost):
    """Another post adding to the About page: a list of terms and acronyms used commonly in posts,
    including some with specific meanings in the context of PlanZero
    modelling.
    """
    # renames scenarios -> models
    def __init__(self):
        super().__init__(
            date=datetime.datetime(2026, 4, 20),
            title='New: a glossary of terms used in specific ways across multiple posts',
            url_filename="2026-04-20-glossary",
            author="James Bergstra",
            tags={BlogTag.About,},
            concept_only=True,
            draft=True,
            )

class About(BlogPost):
    """Briefly going meta: this post explains
    (1) why the About page has been rewritten, and
    (2) how posts themselves are meant to work as a mechanism for developing PlanZero.
    The content of this post also now appears on the About page.
    """
    def __init__(self):
        super().__init__(
            date=datetime.datetime(2026, 4, 12),
            title='About this project: a rearticulation',
            url_filename="2026-04-12-about",
            author="James Bergstra",
            tags={BlogTag.About,},
            draft=True,
            )


class ScenarioModelling(BlogPost):
    """This post introduces a modelling framework for PlanZero.
    The framework formalizes the ideas of critical success factors (CSFs), barriers, strategies,
    and scenarios.
    This post introduces a "Scaling" scenario that estimates what can be achieved by scaling
    currently-available products.
    The scaling scenario launches with just a single product: Bovaer. A new
    series of posts will develop other scaling strategies.
    """
    def __init__(self):
        super().__init__(
            date=datetime.datetime(2026, 4, 3),
            title='A Bovaer Strategy and a Scenario Modelling Framework',
            url_filename="2026-04-03-scenario",
            author="James Bergstra",
            tags={BlogTag.ScalingStrategies,
                  enums.IPCC_Sector.Enteric_Fermentation,
                 },
            draft=True,
            )


class IPCC_HeavyDutyDieselVehicles(BlogPost):
    """Eighth in the sector-by-sector National Greenhouse Gas Inventory series:
    heavy-duty diesel vehicles, such as medium and large freight vehicles,
    buses, and municipal refuse trucks.
    """
    est_nir: object
    terms: dict[str, str]
    def __init__(self):
        super().__init__(
            date=datetime.datetime(2026, 4, 1),
            title='Heavy-Duty Diesel Vehicles: Emissions Calculations',
            url_filename="2026-04-01-heavy-duty-diesel",
            author="James Bergstra",
            tags={BlogTag.NIR_Modelling,
                  enums.IPCC_Sector.Transport__Road__Heavy_Duty_Diesel_Vehicles,
                 },
            est_nir=est_nir,
            terms=dict(),
            )


class IPCC_EntericFermentation(BlogPost):
    """Seventh in the sector-by-sector National Greenhouse Gas Inventory series:
    enteric fermentation, the emission of methane from the digestive systems of all
    livestock, but especially ruminants, and most especially cattle.
    """
    est_nir: object
    terms: dict[str, str]
    def __init__(self):
        super().__init__(
            date=datetime.datetime(2026, 3, 31),
            title='Enteric Fermentation: Emissions Calculations',
            url_filename="2026-03-31-enteric",
            author="James Bergstra",
            tags={BlogTag.NIR_Modelling,
                  enums.IPCC_Sector.Enteric_Fermentation,
                 },
            est_nir=est_nir,
            terms=dict(),
            )

class IPCC_MCS_LightGasolineCarsAndTrucks(BlogPost):
    """Sixth in the sector-by-sector National Greenhouse Gas Inventory series:
    energy to power light-duty gasoline cars and trucks (including SUVs, minivans, and cargo vans).
    A transition to EVs seems to be the sector's clearest pathway to decarbonization.
    """
    est_nir: object
    terms: dict[str, str]
    def __init__(self):
        super().__init__(
            date=datetime.datetime(2026, 3, 30),
            title='Cars and Trucks: Emissions Calculations',
            url_filename="2026-03-30-light-duty-gasoline-trucks",
            author="James Bergstra",
            tags={BlogTag.NIR_Modelling,
                  enums.IPCC_Sector.Transport__Road__Light_Duty_Gasoline_Trucks,
                  enums.IPCC_Sector.Transport__Road__Light_Duty_Gasoline_Vehicles,
                 },
            est_nir=est_nir,
            terms=dict(
                gas_combustion=latex(
                    r"2~\mathrm C_8 \mathrm H_{18} + 25~\mathrm O_2 \rightarrow 16~\mathrm C \mathrm O_2 + 18 ~\mathrm H_2 \mathrm O",
                    display='block'),
                ),
            )


class IPCC_SCS_Residential(BlogPost):
    """Fifth in the sector-by-sector National Greenhouse Gas Inventory series:
    residential stationary combustion. Energy from stationary combustion
    within residential buildings is used predominantly to heat living spaces
    and provide hot water.
    Heat-pumps and ongoing insulation improvements
    promise a viable pathway to decarbonization in this sector.
    """
    est_nir: object
    def __init__(self):
        super().__init__(
            date=datetime.datetime(2026, 3, 26),
            title='Residential Stationary Combustion Sources: Emissions Calculations',
            url_filename="2026-03-26-scs-residential",
            author="James Bergstra",
            tags={BlogTag.NIR_Modelling,
                  enums.IPCC_Sector.SCS__Residential,
                 },
            est_nir=est_nir,
            )


class IPCC_SCS_OilAndGas_Exploration(BlogPost):
    """Fourth in the sector-by-sector series on the National Greenhouse Gas Inventory computation:
    stationary combustion sources involved in the extraction of oil and gas.
    Energy from stationary combustion is used directly and indirectly to drive pumps,
    compressors, separators, and diverse aspects of conventional wells, gathering systems, gas plants, and
    bitumen upgrading operations.
    """
    est_nir:object
    def __init__(self):
        super().__init__(
            date=datetime.datetime(2026, 3, 11),
            title='Stationary Combustion to Extract Oil and Gas: Emissions Calculations',
            url_filename="2026-03-11-og-extraction",
            author="James Bergstra",
            tags={BlogTag.NIR_Modelling,
                  enums.IPCC_Sector.SCS__Oil_and_Gas_Extraction,
                 },
            est_nir=est_nir,
            )

class IPCC_VentingNaturalGas(BlogPost):
    """Third in the sector-by-sector series on the National Greenhouse Gas Inventory computation:
    the venting of emissions from oil and gas systems.
    Venting refers to the intentional or engineered release of greenhouse gases across within the oil and gas sector.
    The re-engineering of the sector to avoid such releases is well underway, but venting still accounts for 5.5%
    of Canada's annual emissions total, at least as of 2023.
    """
    est_nir:object
    def __init__(self):
        super().__init__(
            date=datetime.datetime(2026, 3, 2),
            title='Oil and Natural Gas Venting: Emissions Calculations',
            url_filename="2026-03-02-venting",
            author="James Bergstra",
            tags={BlogTag.NIR_Modelling,
                  enums.IPCC_Sector.Fugitive__Venting,
                 },
            est_nir=est_nir,
            )

class IPCC_ForestAndHWP(BlogPost):
    """Second in the sector-by-sector National Greenhouse Gas Inventory computation:
    Harvested Wood Products and Forest Land.
    Data from Natural Resources Canada on harvested wood volume
    supports a satisfactory estimate of Harvested Wood Products emissions,
    and a first step toward a Forest Land estimate.
    """
    est_nir:object
    def __init__(self):
        super().__init__(
            date=datetime.datetime(2026, 2, 22),
            title='Emissions calculations for Harvested Wood Products and Forest Land',
            url_filename="2026-02-22-forest-hwp",
            author="James Bergstra",
            tags={BlogTag.NIR_Modelling,
                  enums.IPCC_Sector.Harvested_Wood_Products,
                  enums.IPCC_Sector.Forest_Land,
                 },
            est_nir=est_nir,
            )


class IPCC_PublicElectricity(BlogPost):
    """The first in a series of posts replicating the sector-by-sector computation of
    Canada's National Greenhouse Gas Inventory: Public Electricity and Heat.
    As it is first, it also introduces the sectors of the IPCC reporting guidelines,
    and the 71 sectors with which Canada reports its greenhouse gas inventory.
    """
    est_nir:object
    def __init__(self):
        super().__init__(
            date=datetime.datetime(2026, 2, 12),
            title='Emission calculations for Public Electricity and Heat',
            url_filename="2026-02-12-public-electricity",
            author="James Bergstra",
            tags={BlogTag.NIR_Modelling,
                  enums.IPCC_Sector.SCS__Public_Electricity_and_Heat,
                 },
            est_nir=est_nir,
            )


class CNZEAA(BlogPost):
    """A brief introduction to the Canadian Net-Zero Emissions Accountability
    Act, the federal implementation of Canada’s obligations under the Paris
    Accords.
    """
    CNZEAA_targets:list[float]
    net_emissions_total_without_LULUCF:list[float]
    net_emissions_total:list[float]
    def __init__(self):
        super().__init__(
            date=datetime.datetime(2026, 2, 2),
            title="Paris Accords and the CNZEAA",
            url_filename="2026-02-02-cnzeaa",
            author="James Bergstra",
            CNZEAA_targets=list(ipcc_canada.CNZEAA_targets()),
            net_emissions_total_without_LULUCF=list(ipcc_canada.net_emissions_total_without_LULUCF()),
            net_emissions_total=list(ipcc_canada.net_emissions_total()),
            )


class GHG_Emissions_CO2e_v_Heat(HTML_Matplotlib_Figure):
    peval:object
    sts_key:str
    title:str
    legend_loc:str = 'upper right'
    add_circle:bool = False

    def build_figure(self):
        fig, ax = plt.subplots()
        plt.title(self.title)
        years = [year for year in range(2000, 2101)]
        #years = [year for year in range(1990, 2101)]
        for ghg in enums.GHG:
            comp = self.peval.comparisons[ghg]
            years_pint = [year * u.year for year in years]
            energy_A = comp.state_A.sts[self.sts_key].query(years_pint)
            energy_B = comp.state_B.sts[self.sts_key].query(years_pint)
            plt.plot(years,
                     (energy_A - energy_B).to('terajoules').magnitude,
                     label=ghg)
        plt.legend(loc=self.legend_loc)
        if self.add_circle:
            import matplotlib.patches
            circle = matplotlib.patches.Ellipse((2100, 1550), width=10, height=1300, color='blue', alpha=.2)
            ax.add_patch(circle)
            plt.annotate('These emissions are supposed\n'
                         'to trap similar amounts of heat\n'
                         'after 100 years. The simulation\n'
                         'does this to within factor of 2.2x\n'
                         'which I think is okay.',
                         xytext=(2049, 25),
                         xy=(2100, 900),
                         arrowprops=dict(width=1))
        #plt.grid()
        plt.xlabel(f'Time (years)')
        plt.ylabel(f'Heat (terajoules)')


class GHG_Emissions(BlogPost):
    """
    What are greenhouse gases and what do they have to do with
    climate?  This is, I hope, the first post in a series developing
    various plans to achieve a net-zero economy in Canada. It outlines the
    terms in which net-zero is defined, and documents planzero's simple
    climate model.
    """
    equations: dict[str, str]
    a: str
    figure_svgs: dict[str, str]

    def __init__(self):
        equations = dict(
            CO2=latex(r'\mathrm{CO}_2'),
            CO2e=latex(r'\mathrm{CO}_2\mathrm e '),
            CH4=latex(r"\mathrm{CH}_4"),
            N2O=latex(r"\mathrm N_2 \mathrm O"),
            SF6=latex(r"\mathrm{SF}_6"),
            NF3=latex(r"\mathrm{NF}_3"),
            C=latex("C"),
            C0=latex("C_0"),
            W_m2=latex("W / m^2"),
            CO2_df=latex(r"5.35 \ln(C / C_0))"), # W/m^2
            delta_C_left=latex(r"\frac{\Delta C}{dt}"),
            CH4_delta_C_right=latex(r"-C / (12~\mathrm{years})"),
            CH4_df=latex(r"0.036 (\sqrt{C} - \sqrt{C0})"), # W/m^2
            N2O_df=latex(r"0.12 (\sqrt{C} - \sqrt{C0})"), # W/m^2
            N2O_delta_C_right=latex(r"-C / (114~\mathrm{years})"),
            GWP_100=latex(r"\mathrm{GWP}_{100}"),
            HFC_delta_C_right=latex(r"-C / (14~\mathrm{years})"),
            HFC_df=latex(r"0.16 C"),
            PFC_df=latex(r"0.08 C"),
            SF6_df=latex(r"0.57 C"),
            NF3_df=latex(r"0.21 C"),
            )
        peval = emissions_impulse_response_project_evaluation(
            impulse_co2e=1_000_000 * u.kg_CO2e,
            years=100)

        super().__init__(
            date=datetime.datetime(2026, 1, 21),
            title="A Model of Greenhouse Gas Emissions",
            url_filename="2026-01-21-unfccc",
            author="James Bergstra",
            a="bar",
            equations=equations,
            figure_svgs=dict(
                co2e_v_heat_remaining=GHG_Emissions_CO2e_v_Heat(
                    peval=peval,
                    sts_key='Cumulative_Heat_Energy',
                    title="Heat Remaining After 1-year CO2e-equivalent Emissions",
                    legend_loc='upper right').as_html(),
                co2e_v_heat_forcing=GHG_Emissions_CO2e_v_Heat(
                    peval=peval,
                    sts_key='Cumulative_Heat_Energy_forcing',
                    title="Cumulative GHG-Trapped Heat",
                    add_circle=True,
                    legend_loc='upper left').as_html(),
            ))


class Contributing(BlogPost):
    """Coming back from the winter break, I thought I'd write about how
    I myself should contribute to this site; partly to get back in gear, and
    partly to encourage collaboration."""
    def __init__(self):
        super().__init__(
            date=datetime.datetime(2026, 1, 6),
            title="Contributing (even for myself)",
            url_filename="2026-01-06-contributing",
            tags={BlogTag.About,},
            author="James Bergstra")


class HowMightWe(BlogPost):
    """Plan Zero is an independent research project to work publicly toward
    understanding how Canada might achieve net-zero emissions."""
    def __init__(self):
        super().__init__(
            date=datetime.datetime(2025, 12, 5),
            title='How might Canada achieve Net-Zero?',
            url_filename="2025-12-05-first-post",
            tags={BlogTag.About,},
            author="James Bergstra")


def init_blogs_by_url_filename():
    global _blogs_sorted_by_date
    for cls in _classes:
        obj = cls()
        if not obj.concept_only:
            _blogs_by_url_filename[obj.url_filename] = obj
            _blogs_sorted_by_date.append(obj)
    _blogs_sorted_by_date.sort(key=lambda x: x.date, reverse=True)

# TODO: handle this in the metaclass, update the _blogs_sorted_by_date on access
init_blogs_by_url_filename()

def blogs_by_tag(tag):
    for blog in _blogs_sorted_by_date:
        if tag in blog.tags:
            yield blog
