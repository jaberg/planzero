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


class BlogPost(BaseModel):

    date: datetime.datetime
    title: str
    about: str
    url_filename: str
    author: str
    published: bool = True
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


def latex(latex, display='inline'):
    return HTML_Math_Latex(latex=latex, display=display).as_html()


class HTML_Matplotlib_Figure(HTML_element):

    def as_html(self):
        self.build_figure()
        svg_buffer = StringIO()
        plt.savefig(svg_buffer, format="svg")
        plt.close()
        svg_string = svg_buffer.getvalue()
        return svg_string

class IPCC_ForestAndHWP(BlogPost):
    """Second in the sector-by-sector National Greenhouse Gas Inventory computation:
    Forest Land and Harvested Wood Products
    """
    def __init__(self):
        super().__init__(
            date=datetime.datetime(2026, 2, 22),
            title='Replicating emissions calculations for "Forest Land" and "Harvested Wood Products"',
            url_filename="2026-02-22-forest-hwp",
            author="James Bergstra",
            published=False,
            tags={BlogTag.NIR_Modelling,
                  enums.IPCC_Sector.Harvested_Wood_Products,
                  enums.IPCC_Sector.Forest_Land,
                 },
            )


class IPCC_PublicElectricity(BlogPost):
    """First in a series replicating the sector-by-sector computation of
    Canada's National Greenhouse Gas Inventory: Public Electricity and Heat.
    Also introduces the sectors of the IPCC reporting guidelines, and the
    71 sectors with which Canada reports its greenhouse gas inventory.
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
            author="James Bergstra")


class HowMightWe(BlogPost):
    """Plan Zero is an independent research project to work publicly toward
    understanding how Canada might achieve net-zero emissions."""
    def __init__(self):
        super().__init__(
            date=datetime.datetime(2025, 12, 5),
            title='How might Canada achieve Net-Zero?',
            url_filename="2025-12-05-first-post",
            author="James Bergstra")


def init_blogs_by_url_filename():
    global _blogs_sorted_by_date
    for cls in _classes:
        obj = cls()
        _blogs_by_url_filename[obj.url_filename] = obj
        _blogs_sorted_by_date.append(obj)
    _blogs_sorted_by_date.sort(key=lambda x: x.date, reverse=True)

def blogs_by_tag(tag):
    for blog in _blogs_sorted_by_date:
        if tag in blog.tags:
            yield blog
