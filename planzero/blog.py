import datetime
import enum

from pydantic import BaseModel

from . import enums, est_nir, sim

_classes = []
_blogs_by_url_filename = {}
_blogs_sorted_by_date = []
registry = {}



class BlogTag(str, enum.Enum):
    """
    These correspond to points where we show "Posts Developing This Page"
    lists. Blogs should not be tagged with arbitrary strings.
    """
    NIR_Modelling = "NIR Modelling"
    Strategies = 'Strategies'
    BarrierModelling = 'Barrier Modelling'
    About = 'About' # bottom of About page
    StaticNormals = "Static Normals"  # Static Normals model pages


class BlogStatus(str, enum.Enum):
    Done = "Done"
    Draft = "Draft"
    Planned = "Planned"


class BlogPost(BaseModel):

    date: datetime.date
    title: str
    about: str
    url_filename: str
    author: str

    status: BlogStatus

    published: bool = True
    # published=False means post will not appear on website, regardless of status.
    # This attribute is useful for overriding to render a page locally
    # that cannot be rendered from e.g. github workflows.
    # github workflows run with PLANZERO_HOME_SHOW_UNPUBLISHED_POSTS=0
    #
    # TODO: rename this "add to registry" or something

    tags: set[str] = set()
    # TODO: enforce BlogTags e.g.
    # tags: set[BlogTag] = set()
    # But also consider that IPCC_Sector tags are also
    # valid as blog tags. What else?
    # Models, Strategies, Barriers... anything else?

    @property
    def draft(self):
        return self.status == BlogStatus.Draft

    @property
    def done(self):
        return self.status == BlogStatus.Done

    @property
    def planned(self):
        return self.status == BlogStatus.Planned

    @property
    def siteref(self):
        return f'/post/{self.url_filename}'

    def __init__(self, **kwargs):
        if 'about' not in kwargs:
            kwargs = dict(kwargs, about=self.__class__.__doc__)
        super().__init__(**kwargs)

    @classmethod
    def __init_subclass__(cls):
        super().__init_subclass__()
        _classes.append(cls)

    @property
    def status_title_html(self):
        if self.status == BlogStatus.Planned:
            return f"[Planned] {self.title}"
        elif self.status == BlogStatus.Draft:
            return f"[Draft] {self.title}"
        elif self.status == BlogStatus.Done:
            return f"{self.title}"
        else:
            raise NotImplementedError(self.status)

    def generate_assets(self):
        pass


from . import ipcc_canada
from .html import HTML_Math_Latex, HTML_Matplotlib_Figure
from .ureg import u


def latex(latex, display='inline'): # display inline or block
    return HTML_Math_Latex(latex=latex, display=display).as_html()

from .nir_ar2_post import AR2

class Glossary(BlogPost):
    """This post announces a new page:
    a glossary of terms and acronyms with specific meanings in the context of
    PlanZero posts.
    This glossary also introduces modelling terminology to support future posts.
    The modelling terminology is used to reframe the NIR-reconstruction
    project within languages of both strategic management and of statistical
    modelling."""
    def __init__(self):
        super().__init__(
            date=datetime.date(2026, 9, 15),
            title='New: the PlanZero glossary',
            url_filename="2026-04-19-glossary",
            author="James Bergstra",
            #tags={
                # BlogTag.About, # Does it? How?
                # BlogTag.Strategies,
                # BlogTag.BarrierModelling,
                # BlogTag.NIR_Modelling,
            #},
            status=BlogStatus.Planned,
            )


class ProbabilisticBovaer(BlogPost):
    """
    This post revisits the Bovaer strategy
    (<a href="/blog/2026-04-03-bovaer">Modelling a Bovaer Strategy</a>)
    and adapts it to the Static Normals probabilistic model.
    This post deprecates the non-probabilistic "Scaling" model
    and in fact all of the non-probabilistic models in the Models tab in favour
    of probabilistic modelling generally.
    """

    def __init__(self):
        super().__init__(
            date=datetime.date(2026, 8, 30),
            title='Emission Reduction Strategies in Probabilistic Models: Another Look at Bovaer',
            url_filename="2026-07-22-prob-bovaer",
            author="James Bergstra",
            #tags={
                  ## TODO: the probabilistic models?
                  #enums.IPCC_Sector.Enteric_Fermentation,
                 #},
            status=BlogStatus.Planned,
            )


from .nir_static_normals_post import StaticNormals


class ProbabilisticNIR2025(BlogPost):
    """This post introduces a PlanZero's first probabilistic model:
    an interpretation of the NIR-2025 data including its uncertainty
    estimates. The PlanZero site now includes a Models tab
    with a section for probabilistic models, which represent and visualize
    emissions uncertainty. This treatment of uncertainty is a fundamental
    aspect of PlanZero's future modelling work.
    This post also introduces "Planned" status for posts as a mechanism for
    communicating roadmap and organizing ongoing work.
    """

    def __init__(self):
        super().__init__(
            date=datetime.date(2026, 5, 20),
            title='A Probabilistic NIR-2025',
            url_filename="2026-05-20-prob-nir",
            author="James Bergstra",
            status=BlogStatus.Draft,
            tags={
                BlogTag.NIR_Modelling,
                BlogTag.About,
                },
            )

    @staticmethod
    def generate_asset_annex2_rows(path):
        from . import nir2025
        from .enums import GHG, IPCC_Sector
        from .html import html_by_ghg
        table = dict(nir2025.annex2_source_category_by_sector())
        assert (IPCC_Sector.Cropland, GHG.CH4) not in table
        table[IPCC_Sector.Cropland, GHG.CH4] = 'Blend: Conversion of Forest Land and Grass Land'
        table[IPCC_Sector.Settlements, GHG.CH4] = 'Blend: Conversion of Forest Land and Grass Land'
        table[IPCC_Sector.Settlements, GHG.N2O] = 'Blend: Conversion of Forest Land and Grass Land'
        with open(path, 'w') as ofile:
            for sector in IPCC_Sector:
                sources = {
                    table[sector, ghg] for ghg in GHG
                     if (sector, ghg) in table}
                if len(sources) == 0:
                    raise NotImplementedError() # doesn't happen
                    ofile.write(f'<tr><td>{sector.catpath_with_whitespace}</td>')
                    ofile.write('<td></td>')
                    ofile.write('<td>undefined</td></tr>\n')
                elif len(sources) == 1:
                    source_cat, = sources
                    ofile.write(f'<tr><td>{sector.catpath_with_whitespace}</td>')
                    ofile.write('<td>All</td>')
                    ofile.write(f'<td>{source_cat}</td></tr>\n')
                else:
                    for ghg in GHG:
                        if (sector, ghg) in table:
                            ofile.write(f'<tr><td>{sector.catpath_with_whitespace}</td>')
                            ofile.write(f'<td>{html_by_ghg[ghg]}</td>')
                            ofile.write(f'<td>{table[sector, ghg]}</td></tr>\n')

    @classmethod
    def generate_assets(cls):
        from . import prob
        from .enums import GHG, IPCC_Sector

        base = 'html/blog/2026-05-20-prob-nir'
        model_name = 'NIR2025'
        site_inference = prob.site_inferences['NIR2025']
        site_inference.uncertain_sparkline_matrix_echart(
            div_id=f"{model_name}_all_sectors",
            v_unit="Mt_CO2e").save_as(
                f'{base}-{model_name}-all_sectors.html')
        site_inference.uncertain_sparkline_matrix_echart(
            div_id=f"{model_name}_all_sectors2",
            v_unit="Mt_CO2e").save_as(
                f'{base}-{model_name}-all_sectors2.html')
        site_inference.sector_echart(
            sector=IPCC_Sector.SCS__Public_Electricity_and_Heat,
            ghg=None,
            v_unit="Mt_CO2e").save_as(
                f'{base}-{model_name}-PEH.html')
        site_inference.sector_echart(
            sector=IPCC_Sector.SCS__Public_Electricity_and_Heat,
            ghg=GHG.CO2,
            v_unit="Mt_CO2e").save_as(
                f'{base}-{model_name}-PEH-CO2.html')

        cls.generate_asset_annex2_rows(
                f'{base}-{model_name}-annex2_rows.html')


    def figure_uncertainty_hist(self,):
        import numpy as np
        from .nir2025 import load_uncertainty
        unc_df = load_uncertainty()
        def log_squash(x):
            return np.sign(x) * np.log1p(abs(x))

        class RVAL(HTML_Matplotlib_Figure):
            def build_figure_plt(self, plt):
                fig, ax0, = plt.subplots(1, 1, figsize=(6, 4))
                ax0.set_title("Sector-Gas Emissions Uncertainty for year 2023 in NIR-2025")
                ax0.scatter(
                    log_squash(unc_df['Emissions (ktCO2eq) - 2023']),
                    unc_df['Emission Factor Uncertainty (%) - 2023'],
                    s=10,
                    alpha=.3,
                    )
                ax0.set_xlabel('Log-scaled emissions $m$ (ktCO2eq)')
                ax0.set_ylabel('Assessed uncertainty $u$ ($100u$%)')
                xticks = np.asarray([-100_000,
                                     -10_000,
                                     -1000,
                                     -100,
                                     -10,
                                     0,
                                     10,
                                     100,
                                     1000,
                                     10_000,
                                     100_000])
                ax0.set_xticks(
                    log_squash(xticks),
                    [xt if abs(xt) <= 1000 else f'{xt//1000}k' for xt in xticks])
                plt.tight_layout()
        return RVAL()

    @staticmethod
    def figure_pdf():
        import numpy as np
        import numpyro.distributions as dist
        class RVAL(HTML_Matplotlib_Figure):
            def build_figure_plt(self, plt):
                (fig, ax) = plt.subplots(1, 1, figsize=(8, 4))

                ax.set_title('Probability Assessment Using a "Log-Normal" Probability Density Function')
                x = np.linspace(0, 1.4, 100)
                ax.plot(x, np.exp(dist.LogNormal(-1, 0.7).log_prob(x)))

                A = .5
                B = 1.1
                from scipy.stats import lognorm
                cdf_A, cdf_B = lognorm.cdf([A, B], s=.7, scale=np.exp(-1))

                ax.set_xticks(
                    [0, .2, .4,  A , .6, .8, 1.0,  B , 1.2, 1.4],
                    [0, .2, .4, 'A', .6, .8, 1.0, 'B', 1.2, 1.4],
                    )
                x2 = np.linspace(A, B, 50)
                ax.fill_between(
                    x2,
                    #np.exp(dist.LogNormal(-1, 0.7).log_prob(x2)),
                    lognorm.pdf(x2, s=.7, scale=np.exp(-1)),
                    color='skyblue',
                    alpha=0.4)
                ax.set_xlabel("Possible values of the unknown variable, X")
                ax.set_ylabel("Probability density")
                ax.text(A + .05, .15,
                        f"$\\mathrm{{P}}(A < X < B) \\approx {cdf_B - cdf_A:.2f}$")

                plt.tight_layout()
        return RVAL()

    @staticmethod
    def figure_SBLN():
        import numpy as np
        from .enums import IPCC_Sector, GHG, PT
        from .nir2025 import (
            ktCO2e_numpyro_dist_pt_ca,
            idx_of_pt)

        def axtexts(ax, texts, left_offset=0.05):
            for ii, text_str in enumerate(texts):
                ax.text(left_offset, .95 - ii * .07, text_str,
                        transform=ax.transAxes,
                        verticalalignment='top',
                        horizontalalignment='left')

        class RVAL(HTML_Matplotlib_Figure):
            def build_figure_plt(self, plt):
                fig, ((ax0, ax1), (ax2, ax3)) \
                        = plt.subplots(2, 2, figsize=(10, 7))

                ca_dist, pt_dists = ktCO2e_numpyro_dist_pt_ca(
                    sector=IPCC_Sector.SCS__Public_Electricity_and_Heat,
                    ghg=GHG.CO2,
                    year=1995)
                x = np.linspace(0, 150, 500)
                ax0.plot(
                    x,
                    np.exp(ca_dist.log_prob(x * 1000)))
                ax0.set_title(r"National Public Electricity $\mathrm{CO_2}$ Emissions in 1995")
                ax0.set_xlabel(r"Emissions ($\mathrm{MtCO_2e}$)")
                ax0.set_ylabel("Probability Density")
                axtexts(
                    ax0,
                    [f"$\\mu={ca_dist.mu / 1000:.2f}~MtCO_2e$",
                     f"$\\rho={ca_dist.rolloff / 1000:.2f}~MtCO_2e$",
                     f"$\\sigma={ca_dist.relerr * 100:.2f}$%",
                    ])

                x = np.linspace(0, 150, 500)
                pt_dist = pt_dists[idx_of_pt[PT.NU]]
                ax1.plot(
                    x,
                    np.exp(pt_dist.log_prob(x)))
                ax1.set_title(r"Nunavut Public Electricity $\mathrm{CO_2}$ Emissions in 1995")
                ax1.set_xlabel(r"Emissions ($\mathrm{ktCO_2e}$)")
                ax1.set_ylabel("Probability Density")
                axtexts(
                    ax1,
                    [f"$\\mu={pt_dist.mu:.2f}~ktCO_2e$",
                     f"$\\rho={pt_dist.rolloff:.2f}~ktCO_2e$",
                     f"$\\sigma={pt_dist.relerr * 100:.2f}$%",
                    ],
                    left_offset=.6,
                    )

                ca_dist, pt_dists = ktCO2e_numpyro_dist_pt_ca(
                    sector=IPCC_Sector.Forest_Land,
                    ghg=GHG.CO2,
                    year=2009)
                x = np.linspace(-50, 350, 500)
                ax = ax2
                ax.plot(
                    x,
                    np.exp(ca_dist.log_prob(x * 1000)))
                ax.set_title(r"National Forest Land $\mathrm{CO_2}$ Emissions in 2009")
                ax.set_xlabel(r"Emissions ($\mathrm{MtCO_2e}$)")
                ax.set_ylabel("Probability Density")
                axtexts(
                    ax,
                    [f"$\\mu={ca_dist.mu / 1000:.2f}~MtCO_2e$",
                     f"$\\rho={ca_dist.rolloff / 1000:.2f}~MtCO_2e$",
                     f"$\\sigma={ca_dist.relerr * 100:.2f}$%",
                    ],
                    left_offset=.6,
                    )

                x = np.linspace(-50, 350, 500)
                pt_dist = pt_dists[idx_of_pt[PT.ON]]
                ax = ax3
                ax.plot(
                    x,
                    np.exp(pt_dist.log_prob(x * 1000)))
                ax.set_title(r"Ontario Forest Land $\mathrm{CO_2}$ Emissions in 2009")
                ax.set_xlabel(r"Emissions ($\mathrm{MtCO_2e}$)")
                ax.set_ylabel("Probability Density")
                axtexts(
                    ax,
                    [f"$\\mu={pt_dist.mu / 1000:.2f}~MtCO_2e$",
                     f"$\\rho={pt_dist.rolloff / 1000:.2f}~MtCO_2e$",
                     f"$\\sigma={pt_dist.relerr * 100:.2f}$%",
                    ],
                    left_offset=.6,
                    )

                plt.tight_layout()
        return RVAL()


class About(BlogPost):
    """Going meta: Refining the vision and mission,
    acknowledging the project contributors, and 
    explaining how posts are meant to work as a mechanism for developing PlanZero.
    """
    def __init__(self):
        super().__init__(
            date=datetime.date(2026, 4, 23),
            title='About this project: rewriting and expanding planzero.ca/about',
            url_filename="2026-04-12-about",
            status=BlogStatus.Done,
            author="James Bergstra",
            tags={BlogTag.About,
                 },
            )


class ModellingBovaer(BlogPost):
    """
    This post breaks from the sector-by-sector
    analysis of the National Greenhouse Gas Inventory
    to do a bit of modelling: what would happen if Canada's beef
    and dairy farmers gradually transitioned to administering the feed additive Bovaer,
    which reduces methane emissions? A PlanZero model finds that it could eventually
    remove 10Mt of annual emissions, and cost about $222 per tonne removed.
    """
    def __init__(self):
        super().__init__(
            date=datetime.date(2026, 4, 9),
            title='Modelling a Bovaer Strategy',
            url_filename="2026-04-03-bovaer",
            author="James Bergstra",
            status=BlogStatus.Done,
            tags={BlogTag.BarrierModelling,
                  enums.IPCC_Sector.Enteric_Fermentation,
                  'Scale_Bovaer',
                 },
            draft=False,
            )

    @staticmethod
    def generate_assets():
        scaling = sim.simulation_result('Scaling')
        scaling.by_ipcc_sector.save_as(
            'html/blog/2026-04-03-bovaer_by-ipcc-sector.html')
        scaling.strategy_impact_echart('Scale_Bovaer').save_as(
            'html/blog/2026-04-03-bovaer_strategy-emissions.html')
        scaling.strategy_subsidies_echart('Scale_Bovaer').save_as(
            'html/blog/2026-04-03-bovaer_strategy-subsidies.html')


class IPCC_HeavyDutyDieselVehicles(BlogPost):
    """Eighth in the sector-by-sector National Greenhouse Gas Inventory series:
    heavy-duty diesel vehicles, such as medium and large freight vehicles,
    buses, and municipal refuse trucks.
    """
    est_nir: object
    terms: dict[str, str]
    def __init__(self):
        super().__init__(
            date=datetime.date(2026, 4, 1),
            title='Heavy-Duty Diesel Vehicles: Emissions Calculations',
            url_filename="2026-04-01-heavy-duty-diesel",
            author="James Bergstra",
            status=BlogStatus.Done,
            tags={BlogTag.NIR_Modelling,
                  enums.IPCC_Sector.Transport__Road__Heavy_Duty_Diesel_Vehicles,
                 },
            est_nir=est_nir,
            terms={},
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
            date=datetime.date(2026, 3, 31),
            title='Enteric Fermentation: Emissions Calculations',
            url_filename="2026-03-31-enteric",
            author="James Bergstra",
            status=BlogStatus.Done,
            tags={BlogTag.NIR_Modelling,
                  enums.IPCC_Sector.Enteric_Fermentation,
                 },
            est_nir=est_nir,
            terms={},
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
            date=datetime.date(2026, 3, 30),
            title='Cars and Trucks: Emissions Calculations',
            url_filename="2026-03-30-light-duty-gasoline-trucks",
            author="James Bergstra",
            status=BlogStatus.Done,
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
            date=datetime.date(2026, 3, 26),
            title='Residential Stationary Combustion Sources: Emissions Calculations',
            url_filename="2026-03-26-scs-residential",
            author="James Bergstra",
            status=BlogStatus.Done,
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
            date=datetime.date(2026, 3, 11),
            title='Stationary Combustion to Extract Oil and Gas: Emissions Calculations',
            url_filename="2026-03-11-og-extraction",
            author="James Bergstra",
            status=BlogStatus.Done,
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
            date=datetime.date(2026, 3, 2),
            title='Oil and Natural Gas Venting: Emissions Calculations',
            url_filename="2026-03-02-venting",
            author="James Bergstra",
            status=BlogStatus.Done,
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
            date=datetime.date(2026, 2, 22),
            title='Emissions calculations for Harvested Wood Products and Forest Land',
            url_filename="2026-02-22-forest-hwp",
            author="James Bergstra",
            status=BlogStatus.Done,
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
            date=datetime.date(2026, 2, 12),
            title='Emission calculations for Public Electricity and Heat',
            url_filename="2026-02-12-public-electricity",
            author="James Bergstra",
            status=BlogStatus.Done,
            tags={BlogTag.NIR_Modelling,
                  enums.IPCC_Sector.SCS__Public_Electricity_and_Heat,
                 },
            est_nir=est_nir,
            )


class CNZEAA(BlogPost):
    """A brief introduction to the Canadian Net-Zero Emissions Accountability
    Act, the federal implementation of Canada’s obligations under the Paris
    Agreement.
    """
    CNZEAA_targets:list[float]
    net_emissions_total_without_LULUCF:list[float]
    net_emissions_total:list[float]
    def __init__(self):
        super().__init__(
            date=datetime.date(2026, 2, 2),
            title="The Paris Agreement and the CNZEAA",
            url_filename="2026-02-02-cnzeaa",
            author="James Bergstra",
            status=BlogStatus.Done,
            CNZEAA_targets=list(ipcc_canada.CNZEAA_targets()),
            net_emissions_total_without_LULUCF=list(ipcc_canada.net_emissions_total_without_LULUCF()),
            net_emissions_total=list(ipcc_canada.net_emissions_total()),
            )


class GHG_Emissions_CO2e_v_Heat(HTML_Matplotlib_Figure):
    sim_result:object
    sts_key:str
    title:str
    legend_loc:str = 'upper right'
    add_circle:bool = False

    def build_figure_plt(self, plt):
        fig, ax = plt.subplots()
        plt.title(self.title)
        years = [year for year in range(2000, 2101)]
        #years = [year for year in range(1990, 2101)]
        for ghg in enums.GHG:
            years_pint = [year * u.year for year in years]
            state_A = self.sim_result.state
            state_B = self.sim_result.ablations[f'EmissionsImpulseResponse_{ghg.value}']
            energy_A = state_A.sts[self.sts_key].query(years_pint)
            energy_B = state_B.sts[self.sts_key].query(years_pint)
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

        super().__init__(
            date=datetime.date(2026, 1, 21),
            title="A Model of Greenhouse Gas Emissions",
            url_filename="2026-01-21-unfccc",
            author="James Bergstra",
            status=BlogStatus.Done,
            a="bar",
            equations=equations,
            figure_svgs=dict(
                co2e_v_heat_remaining=GHG_Emissions_CO2e_v_Heat(
                    sim_result=sim.simulation_result('Planet_Model'),
                    sts_key='Cumulative_Heat_Energy',
                    title="Heat Remaining After 1-year CO2e-equivalent Emissions",
                    legend_loc='upper right').as_html(),
                co2e_v_heat_forcing=GHG_Emissions_CO2e_v_Heat(
                    sim_result=sim.simulation_result('Planet_Model'),
                    sts_key='Cumulative_Heat_Energy_forcing',
                    title="Cumulative GHG-Trapped Heat",
                    add_circle=True,
                    legend_loc='upper left').as_html(),
            ),
            tags=[
                'EmissionsImpulseResponse_CO2',
                'EmissionsImpulseResponse_CH4',
                'EmissionsImpulseResponse_N2O',
                'EmissionsImpulseResponse_HFCs',
                'EmissionsImpulseResponse_PFCs',
                'EmissionsImpulseResponse_SF6',
                'EmissionsImpulseResponse_NF3',
            ],
            )


class Contributing(BlogPost):
    """Coming back from the winter break, I thought I'd write about how
    I myself should contribute to this site; partly to get back in gear, and
    partly to encourage collaboration."""
    def __init__(self):
        super().__init__(
            date=datetime.date(2026, 1, 6),
            title="Contributing (even for myself)",
            url_filename="2026-01-06-contributing",
            status=BlogStatus.Done,
            tags={BlogTag.About,},
            author="James Bergstra")


class HowMightWe(BlogPost):
    """Plan Zero is an independent research project to work publicly toward
    understanding how Canada might achieve net-zero emissions."""
    def __init__(self):
        super().__init__(
            date=datetime.date(2025, 12, 5),
            title='How might Canada achieve Net-Zero?',
            url_filename="2025-12-05-first-post",
            status=BlogStatus.Done,
            tags={BlogTag.About,},
            author="James Bergstra")


def init_blogs_by_url_filename():
    global _blogs_sorted_by_date
    for cls in _classes:
        obj = cls()
        if obj.published:
            registry[cls.__name__] = obj
            _blogs_by_url_filename[obj.url_filename] = obj
            _blogs_sorted_by_date.append(obj)
    _blogs_sorted_by_date.sort(key=lambda x: x.date, reverse=True)

# TODO: handle this in the metaclass, update the _blogs_sorted_by_date on access
init_blogs_by_url_filename()

def blogs_by_tag(tag):
    for blog in _blogs_sorted_by_date:
        if tag in blog.tags:
            yield blog
