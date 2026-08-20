import enum
from pydantic import BaseModel
import datetime

from . import enums
from . import est_nir
from . import sim

from .enums import col_by_pt, col_ca

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

    date: datetime.datetime
    title: str
    about: str
    url_filename: str
    author: str

    status: BlogStatus
    published: bool = True

    # TODO: enforce BlogTags e.g.
    # tags: set[BlogTag] = set()
    # But also consider that IPCC_Sector tags are also
    # valid as blog tags. What else?
    # Models, Strategies, Barriers... anything else?
    tags: set[str] = set()

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


class AR2(BlogPost):
    """
    This post introduces
    a new baseline probabilistic model based on two-step autoregressive
    processes (AR-2).
    Unlike the Static Normals model it does not assume year-to-year variations
    are all measurement error, it assumes that they are nearly accurate
    and that emissions change according to trends.
    This model estimates national, provincial and territorial emissions
    for every sector and greenhouse gas.
    Its near-term predictions are relatively precise, and longer-term
    predictions are less certain.
    """

    def __init__(self):
        super().__init__(
            date=datetime.datetime(2026, 10, 28),
            title='Autoregressive Process Modelling of NIR-2025',
            url_filename="2026-05-26-probabilistic-modelling",
            author="James Bergstra",
            #tags={BlogTag.NIR_Modelling,
                 #},
            status=BlogStatus.Planned,
            )

    @staticmethod
    def generate_assets():
        from . import prob
        from .enums import IPCC_Sector, GHG

        for model_name, site_inference in prob.site_inferences.items():
            base = 'html/blog/2026-04-26-probabilistic-modelling'
            if model_name in ('Static_Normals', 'AR2'):
                site_inference.uncertain_sparkline_matrix_echart(
                    div_id=f"{model_name}_all_sectors",
                    v_unit="Mt_CO2e").save_as(
                        f'{base}-{model_name}-all_sectors.html')
                site_inference.sector_echart(
                    sector=IPCC_Sector.Harvested_Wood_Products,
                    ghg=GHG.CO2,
                    v_unit="Mt_CO2e").save_as(
                        f'{base}-{model_name}-HWP.html')
            else:
                continue

    def ar2_model(self, sector, ghg):
        from .nir_ar2 import NIR2025_AR2, load_config_samples
        config, grouped_samples = load_config_samples(sector, ghg)
        model = NIR2025_AR2(
            sector=sector,
            ghg=ghg,
            future_idx=NIR2025_AR2.idx_of_last_train_year(2022))
        model.grouped_samples = grouped_samples
        model.post_samples = {
            key: val.reshape(-1, *val.shape[2:])
            for key, val in grouped_samples.items()}
        return model

    def const_sector_ghg(self, ax, sector, ghg):
        from .nir_constant_predictor import NIR2025_Model
        from . import nir2025
        import numpy as np
        import jax.numpy as jnp
        from numpyro.diagnostics import hpdi
        PT = enums.PT

        model = NIR2025_Model.posterior_inference(
            sector=sector, ghg=ghg)
        predictions = model.predictions()

        mean_mu = jnp.mean(model.post_samples['mu'], axis=0)
        hpdi_mu_pt = hpdi(model.post_samples['mu'], 0.95)
        hpdi_mu_ca = hpdi(
            jnp.sum(model.post_samples['mu'], axis=1),
            0.95)

        # spread
        x = nir2025.nir2025_year_ints
        spread_ca = hpdi(predictions['obs_ca'], 0.95)
        ax.fill_between(
            x,
            np.ones(len(x)) * spread_ca[0] * model.scale,
            np.ones(len(x)) * spread_ca[1] * model.scale,
            alpha=0.1,
            interpolate=True,
            color=col_ca,
            )
        # mean
        ax.axhline(
            jnp.mean(predictions['obs_ca']) * model.scale,
            color=col_ca,
        )
        # data
        ax.scatter(
            nir2025.nir2025_year_ints,
            model.jnp_ca,
            color=col_ca,
        )

        for ii, pt in enumerate(PT):
            if pt == PT.XX:
                continue
            # spread
            spread_pt = hpdi(predictions['obs_pt'][ii], 0.95)
            ax.fill_between(
                x,
                np.ones(len(x)) * spread_pt[0] * model.scale,
                np.ones(len(x)) * spread_pt[1] * model.scale,
                alpha=0.1,
                interpolate=True,
                color=col_by_pt[pt],
                )
            # mean
            ax.axhline(
                jnp.mean(predictions['obs_pt'][ii]) * model.scale,
                color=col_ca,
            )
            # data
            ax.scatter(
                nir2025.nir2025_year_ints,
                model.jnp_pt[ii],
                color=col_by_pt[pt],
            )

    def const_sector_ghg_pt(self, ax, sector, ghg, pt, list_idx, model, predictions):
        from . import nir2025
        import numpy as np
        import jax.numpy as jnp
        from numpyro.diagnostics import hpdi

        scale = model.scale

        x = nir2025.nir2025_year_ints
        if pt is None:
            # spread
            spread_ca = hpdi(predictions['obs_ca'], 0.95)
            ax.fill_between(
                x,
                np.ones(len(x)) * spread_ca[0] * scale,
                np.ones(len(x)) * spread_ca[1] * scale,
                alpha=0.1,
                interpolate=True,
                color=col_ca,
                )
            # mean
            ax.axhline(
                jnp.mean(predictions['obs_ca']) * scale,
                c=col_ca,
            )
            # data
            ax.scatter(
                nir2025.nir2025_year_ints,
                model.jnp_ca * scale / model.scale,
                color=col_ca,
            )
            for ii, pt in enumerate(enums.PT):
                if pt == enums.PT.XX:
                    continue
                ax.scatter(
                    nir2025.nir2025_year_ints,
                    model.jnp_pt[ii] * scale / model.scale,
                    color=col_by_pt[pt],
                )
        else:
            # spread
            #print('obs_pt shape', predictions['obs_pt'].shape)
            spread_pt = hpdi(predictions['obs_pt'][:, list_idx], 0.95)
            #print('mu shape', model.post_samples['mu'].shape)
            spread_pt_mu = hpdi(model.post_samples['mu'][:, list_idx], 0.95)
            #print('spread_pt shape', spread_pt.shape)
            ax.fill_between(
                x,
                np.ones(len(x)) * spread_pt[0] * scale,
                np.ones(len(x)) * spread_pt[1] * scale,
                alpha=0.1,
                interpolate=True,
                color=col_by_pt[pt],
                )
            ax.fill_between(
                x,
                np.ones(len(x)) * spread_pt_mu[0] * scale,
                np.ones(len(x)) * spread_pt_mu[1] * scale,
                alpha=0.2,
                interpolate=True,
                color=col_by_pt[pt],
                )
            # latent mean
            ax.axhline(
                jnp.mean(predictions['obs_pt'][:, list_idx]) * scale,
                c=col_by_pt[pt],
                ls=':',
                label='estimated mu',
            )
            # data
            ax.scatter(
                nir2025.nir2025_year_ints,
                model.jnp_pt[list_idx] * scale / model.scale,
                color=col_by_pt[pt],
                label='data',
            )
            # data mean
            ax.axhline(
                np.nanmean(model.jnp_pt[list_idx] * scale / model.scale),
                color=col_by_pt[pt],
                ls='-',
                label='data mean',
            )
            lbound = min(0,
                           spread_pt[0] * scale,
                           np.nanmin(model.jnp_pt[list_idx] * scale / model.scale))
            ubound = max(0,
                           spread_pt[1] * scale,
                           np.nanmax(model.jnp_pt[list_idx] * scale / model.scale))
            ludiff = ubound - lbound
            ax.set_ylim(
                lbound - .05 * ludiff,
                ubound + .05 * ludiff)


    def foo(self,):
        assert 0
        IPCC_Sector = enums.IPCC_Sector
        GHG = enums.GHG
        sector_ghg_list = [
            (IPCC_Sector.SCS__Public_Electricity_and_Heat,
             GHG.CO2),
            (IPCC_Sector.Ammonia_Production,
             GHG.CO2),
            (IPCC_Sector.Cement_Production,
             GHG.CO2),
            (IPCC_Sector.SCS__Commercial_and_Institutional,
             GHG.CO2),
            (IPCC_Sector.Forest_Land,
             GHG.CO2),
            (IPCC_Sector.Harvested_Wood_Products,
             GHG.CO2),
            (IPCC_Sector.Lime_Production,
             GHG.CO2),
            (IPCC_Sector.Nitric_Acid_Production,
             GHG.N2O),
            (IPCC_Sector.Non_Energy_Products_from_Fuels_and_Solvent_Use,
             GHG.CO2),
            (IPCC_Sector.Non_Energy_Products_from_Fuels_and_Solvent_Use,
             GHG.N2O),
            (IPCC_Sector.Petrochemical_and_Carbon_Black_Production,
             GHG.CO2),
            (IPCC_Sector.SCS__Petroleum_Refining_Industries,
             GHG.CO2),
        ]

        class RVAL(HTML_Matplotlib_Figure):
            def build_figure(_):
                n_cols = 3
                fig, axs = plt.subplots(4, n_cols, figsize=(10, 12))
                for row, axrow in enumerate(axs):
                    for col, ax in enumerate(axrow):
                        list_idx = col + row * n_cols
                        try:
                            sector, ghg = sector_ghg_list[list_idx]
                        except IndexError:
                            break
                        self.const_sector_ghg( ax, sector, ghg)
                        if col == 0:
                            ax.set_ylabel('Emissions (CO2e)')
                        ax.set_title(sector.value)
                plt.tight_layout()
        return RVAL()


    def static_normals_HWP_CO2(self,):
        IPCC_Sector = enums.IPCC_Sector
        GHG = enums.GHG
        PT = enums.PT
        sector = IPCC_Sector.SCS__Commercial_and_Institutional
        sector = IPCC_Sector.Harvested_Wood_Products
        ghg = GHG.CO2

        from .nir_constant_predictor import NIR2025_Model
        model = NIR2025_Model.posterior_inference(
            sector=sector, ghg=ghg)
        predictions = model.predictions()

        class RVAL(HTML_Matplotlib_Figure):
            def build_figure(_):
                n_cols = 2
                fig, axs = plt.subplots(7, n_cols, figsize=(9, 17))
                for row, axrow in enumerate(axs):
                    for col, ax in enumerate(axrow):
                        list_idx = col + row * n_cols
                        if list_idx == 0:
                            pt = None
                        else:
                            pt = list(enums.PT)[list_idx - 1]
                        self.const_sector_ghg_pt(
                            ax, sector, ghg, pt, list_idx - 1, model, predictions)
                        if col == 0:
                            ax.set_ylabel('Emissions (CO2e)')
                        ax.set_title(pt.value if pt else "Canada")
                        if pt:
                            ax.legend(loc='lower right')
                plt.tight_layout()
        return RVAL()

    def figure_ar2_rhat(self):
        IPCC_Sector = enums.IPCC_Sector
        GHG = enums.GHG
        sector = IPCC_Sector.Harvested_Wood_Products
        ghg = GHG.CO2
        model = self.ar2_model(sector, ghg)
        grouped_samples = model.grouped_samples
        from numpyro.diagnostics import summary
        diagnostics = summary(grouped_samples, prob=0.90, group_by_chain=True)
        class RVAL(HTML_Matplotlib_Figure):
            def build_figure(_):

                for param, stats in diagnostics.items():
                    plt.hist(stats['r_hat'].flatten(), alpha=.2, label=param)

                plt.legend(loc='upper right')
        return RVAL()

    def ar2_sector_ghg_pt(self, ax, sector, ghg, pt, list_idx, model):
        from . import nir2025
        import numpy as np
        import jax.numpy as jnp
        import jax.random as jrandom
        from numpyro.diagnostics import hpdi
        from . import nir_ar2

        rng_key = jrandom.key(1234)
        np_rng = np.random.default_rng(12345)
        rng_key, rng_key_ = jrandom.split(rng_key)

        n_mu_timesteps_to_2022 = 31 # for NIR-2025
        n_mu_timesteps_to_2050 = n_mu_timesteps_to_2022 + 27

        samples = nir_ar2.samples_with_extended_mu(
            model.post_samples,
            n_steps=n_mu_timesteps_to_2050 - n_mu_timesteps_to_2022,
            pt_rms=model.pt_rms,
            np_rng=np_rng)
        samples = nir_ar2.sample_past_given_mu(
            samples,
            relerr=model.relerr,
            pad_mu0_m1=True,
            key=rng_key_)

        rec_pt = samples['past-pt']
        rec_ca = samples['past-ca']
        rec_yrs = np.arange(1990, 2050)

        scale = model.scale / 1000 # convert to Mt

        if pt is None:
            # spread
            spread_ca = hpdi(rec_ca, 0.95)
            ax.fill_between(
                rec_yrs,
                spread_ca[0] * scale,
                spread_ca[1] * scale,
                alpha=0.1,
                interpolate=True,
                color=col_ca,
                label='$X$ 95% CI',
                )
            # mean
            ax.plot(
                rec_yrs,
                rec_ca.mean(axis=0) * scale,
                c=col_ca,
                label='$X$ mean',
                ls=":",
            )
            # data
            ax.scatter(
                nir2025.nir2025_year_ints,
                model.jnp_ca * scale / model.scale,
                color=col_ca,
                s=20,
                label='NIR-2025 data',
            )
            for ii, pt in enumerate(enums.PT):
                if pt == enums.PT.XX:
                    continue
                ax.scatter(
                    nir2025.nir2025_year_ints,
                    model.jnp_pt[ii] * scale / model.scale,
                    color=col_by_pt[pt],
                    alpha=.3,
                    s=20,
                )

        else:
            future_idx = len(model.scaled_ca)
            obs_valid = jnp.isfinite(model.scaled_pt[:, :future_idx])
            approx_obs = jnp.where(
                obs_valid,
                model.scaled_pt[:, :future_idx],
                jnp.nanmean(model.scaled_pt[:, :future_idx], axis=1, keepdims=True))

            mu = nir_ar2.complete_mu(samples)
            mu_spread = hpdi(mu, 0.95)
            mu_years = np.arange(mu.shape[1]) + 1990
            ax.fill_between(
                list(mu_years),
                mu_spread[0, :, list_idx] * scale,
                mu_spread[1, :, list_idx] * scale,
                alpha=0.2,
                interpolate=True,
                color=col_by_pt[pt],
                label=r'$\mu$ 95% CI',
                )

            spread_pt_ii = hpdi(rec_pt[:, :, list_idx], 0.95)
            ax.fill_between(
                rec_yrs,
                spread_pt_ii[0] * scale,
                spread_pt_ii[1] * scale,
                alpha=0.1,
                interpolate=True,
                color=col_by_pt[pt],
                label=r'$X$ 95% CI',
                )
            ax.plot(
                rec_yrs,
                np.mean(spread_pt_ii, axis=0) * scale,
                color=col_by_pt[pt],
                label=r'$X$ mean = $\mu$ mean',
                ls=':',
                )

            ax.scatter(
                nir2025.nir2025_year_ints,
                model.jnp_pt[list_idx] * scale / model.scale,
                color=col_by_pt[pt],
                label='NIR-2025 data',
                s=20,
            )

    def figure_ar2_hwp(self,):
        IPCC_Sector = enums.IPCC_Sector
        GHG = enums.GHG
        PT = enums.PT
        sector = IPCC_Sector.Harvested_Wood_Products
        ghg = GHG.CO2

        model = self.ar2_model(sector, ghg)

        import numpy as np

        class RVAL(HTML_Matplotlib_Figure):
            def build_figure(_):
                n_cols = 2
                fig, axs = plt.subplots(7, n_cols, figsize=(9, 17))
                for row, axrow in enumerate(axs):
                    for col, ax in enumerate(axrow):
                        list_idx = col + row * n_cols
                        if list_idx == 0:
                            pt = None
                        else:
                            pt = list(enums.PT)[list_idx - 1]
                        self.ar2_sector_ghg_pt(
                            ax, sector, ghg, pt, list_idx - 1,
                            model)
                        alpha_1 = np.mean(model.post_samples['alpha_1'][:, list_idx - 1])
                        alpha_2 = np.mean(model.post_samples['alpha_2'][:, list_idx - 1])
                        if pt is None:
                            ax.set_title("Canada (with regional subtotals)")
                        else:
                            ax.set_title(f'{pt.value} [$\\bar c_1={3 * alpha_1 - .5:.2f}$, $\\bar c_2={3 * alpha_2 - 1.5:.2f}$]')
                        ax.legend(loc='lower right')
                plt.tight_layout()
        return RVAL()

    def figure_normal(self,):
        import numpy as np
        import numpyro.distributions as dist
        class RVAL(HTML_Matplotlib_Figure):
            def build_figure(self):
                fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4))
                ax0.set_title("Normal(0, 1)")
                x = np.linspace(-3, 3, 100)
                ax0.plot(x, np.exp(dist.Normal(0, 1).log_prob(x)))
                ax0.set_ylabel('Density')

                ax1.set_title("LogNormal(-1, .7)")
                x = np.linspace(0, 1, 100)
                ax1.plot(x, np.exp(dist.LogNormal(-1, 0.7).log_prob(x)))
                plt.tight_layout()
        return RVAL()


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
            date=datetime.datetime(2026, 9, 15),
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
            date=datetime.datetime(2026, 8, 30),
            title='Emission Reduction Strategies in Probabilistic Models: Another Look at Bovaer',
            url_filename="2026-07-22-prob-bovaer",
            author="James Bergstra",
            #tags={
                  ## TODO: the probabilistic models?
                  #enums.IPCC_Sector.Enteric_Fermentation,
                 #},
            status=BlogStatus.Planned,
            )


class PreNIR(BlogPost):

    #PlanZero is largely about predicting Canada's future emissions,
    #both with and without strategies aimed at reducing those emissions.
    #It is difficult to make credible predictions. One way to build credibility
    #for a model is to show that it predicted the recent past on the basis of the distant past,
    #and so therefore it should be trusted to predict the future on the basis of up-to-date data.
    """
    This post introduces the "<a href="/predictions">Predictions</a>" tab, with PlanZero's first prediction challenge:
    to predict the latest year of data (2023) from NIR-2025 on the basis of
    data that was available 12 months prior to its publication (Pre-NIR-2025-12).
    This challenge is simulated using NIR-2025's data until 2022 instead of NIR-2024,
    but lays groundwork for future prediction challenges Pre-NIR-2026-12 and Pre-NIR-2027-12
    that will not be simulated.
    """

    def __init__(self):
        super().__init__(
            date=datetime.datetime(2026, 7, 22),
            title='Prediction of National Inventory Reports',
            url_filename="2026-06-19-yoly-2025",
            author="James Bergstra",
            #tags={BlogTag.NIR_Modelling,},
            status=BlogStatus.Planned,
            )

    def figure_YOLY_2025(self):
        class RVAL(HTML_Matplotlib_Figure):
            def build_figure(_):
                import matplotlib.pyplot as plt
                import numpy as np
                from datetime import datetime

                sources = {
                    "NIR-2024": {
                        "pub_date": "2024-05-03",
                        "train_start": "1990-01-01",
                        "train_end": "2022-12-31",
                    },
                    "NIR-2025": {
                        "pub_date": "2025-04-16",
                        "train_start": "1990-01-01",
                        "train_end": "2022-12-31",
                        "test_start": "2023-01-01",
                        "test_end": "2023-12-31",
                    },
                    "Hypothetically admissible extra data)": {
                        "pub_date": "2023-03-31",
                        "train_start": "2000-01-01",
                        "train_end": "2022-12-31",
                    },
                    "Data set not available at prediction time": {
                        "pub_date": "2024-09-30",
                        "train_start": "2000-01-01",
                        "train_end": "2022-12-31",
                    }
                }

                fig, ax = plt.subplots(figsize=(8, 3.5))

                for ii, (dset_name, dset_info) in enumerate(sources.items()):
                    for key, val in list(dset_info.items()):
                        dset_info[key] = datetime.strptime(val, "%Y-%m-%d")
                    y = -ii

                    # Solid bar: fully available data
                    ax.barh(y, dset_info["train_end"] - dset_info["train_start"], left=dset_info["train_start"],
                            height=0.5, color='steelblue', alpha=0.85,
                            label='Training data' if ii == 0 else None,
                           )
                    last_date = dset_info["train_end"]
                    if 'test_start' in dset_info:
                        ax.barh(y, dset_info["test_end"] - dset_info["test_start"], left=dset_info["test_start"],
                                height=0.5, color='lightblue', alpha=0.85,
                                label="Evaluation data" if ii == 0 else None)
                        last_date = dset_info["test_end"]

                    # Lag bar: not yet published
                    ax.barh(y, dset_info["pub_date"] - last_date, left=last_date,
                            height=0.5, color='tomato', alpha=0.35,
                            hatch='..', edgecolor='tomato',
                            label="Publication delay" if ii == 0 else None)

                prediction_time = datetime.strptime("2024-05-16", "%Y-%m-%d")
                ax.axvline(prediction_time, color='crimson', linestyle='--', lw=1.5, label='Prediction time')
                ax.axvline(datetime.now(), color='black', linestyle='--', lw=1.5, label='Today')

                ax.set_yticks([-ii for ii in range(len(sources))])
                ax.set_yticklabels([src for src in sources])
                ax.set_xlabel("Date")
                #ax.xaxis.set_major_formatter(lambda x, _: f"–{int(TODAY-x)}d" if x < TODAY else "Today")
                #ax.set_title("The NIR-2025 Year-Out, Last-Year (YOLY-2025) prediction challenge (of year 2023)")
                plt.legend(loc='lower left')
                plt.tight_layout()
        return RVAL()

    def figure_YOLY_2025_violinplots(self):
        class RVAL(HTML_Matplotlib_Figure):
            def build_figure(_):
                return
                import planzero
                from planzero.nir_ar2 import NIR2025_AR2
                IPCC_Sector = planzero.enums.IPCC_Sector
                GHG = planzero.enums.GHG

                sector = IPCC_Sector.Harvested_Wood_Products
                ghg = GHG.CO2
                model = self.ar2_model(sector, ghg)

                from planzero.enums import col_by_pt, col_ca, PT
                rcon = model.reconstructed_past()
                from .nir_constant_predictor import NIR2025_Model
                const_model = NIR2025_Model.posterior_inference(
                    sector=IPCC_Sector.Harvested_Wood_Products,
                    ghg=GHG.CO2,
                )
                const_predictions = const_model.predictions()

                from . import nir2025
                arr_pt, arr_ca = nir2025.ktCO2e_dense_w_nan()
                ca_2023 = rcon['past-ca-1'][:, -1]
                #print(rcon['past-pt-1'].shape)
                import matplotlib.pyplot as plt
                def monochrome_violinplot(x, ydata, c, label=None):
                    violin_parts = ax.violinplot([ydata], positions=[x], orientation='horizontal')
                    violin_parts['bodies'][0].set_color(c)
                    for part_name in ['cmaxes', 'cmins', 'cbars']:
                        line_collection = violin_parts[part_name]
                        line_collection.set_color(c)
                fig, ax = plt.subplots(figsize=[8, 6])
                monochrome_violinplot(0, ca_2023 * model.scale, col_ca)
                monochrome_violinplot(-1, const_predictions['obs_ca'] * model.scale, col_ca)
                for ii, pt in enumerate(planzero.enums.PT):
                    if pt == planzero.enums.PT.XX:
                        continue
                    monochrome_violinplot(
                        -2 * (ii + 1),
                        rcon['past-pt-1'][:, -1, ii] * model.scale,
                        col_by_pt[pt],
                    )
                    monochrome_violinplot(
                        -2 * (ii + 1) - 1,
                        const_predictions['obs_pt'][:, ii] * model.scale,
                        col_by_pt[pt])
                    plt.text(22_000, -2 * (ii + 1) - 0.8, pt.two_letter_code(), fontsize=12)
                    plt.text(12_000, -2 * (ii + 1) - 0.2, "AR2")
                    plt.text(12_000, -2 * (ii + 1) - 1.2, "Const")
                plt.text(12_000, -2 * (-1 + 1) - 0.2, "AR2")
                plt.text(12_000, -2 * (-1 + 1) - 1.2, "Const")
                plt.text(22_000, - 0.8, 'CA', fontsize=12)

                plt.scatter(#[1],
                            [arr_ca[nir2025.idx_of_sector[IPCC_Sector.Harvested_Wood_Products],
                                    nir2025.idx_of_ghg[GHG.CO2],
                                    -1]] * 2,
                            [0, -1],
                            c=col_ca,
                            marker='o',
                            label="Solid dots: NIR-2025's 2023 emissions",
                )
                plt.scatter(arr_pt[nir2025.idx_of_sector[IPCC_Sector.Harvested_Wood_Products],
                                    nir2025.idx_of_ghg[GHG.CO2],
                                    :, # PT but not XX
                                    -1],
                            [-y * 2 for y in range(1, 14)],
                            c=[col_by_pt[pt] for pt in PT if pt != PT.XX],
                            marker='o')
                plt.scatter(arr_pt[nir2025.idx_of_sector[IPCC_Sector.Harvested_Wood_Products],
                                    nir2025.idx_of_ghg[GHG.CO2],
                                    :, # PT but not XX
                                    -1],
                            [-y * 2 - 1 for y in range(1, 14)],
                            c=[col_by_pt[pt] for pt in PT if pt != PT.XX],
                            marker='o')

                if 0:
                    plt.yticks(
                        [-ii * 2 - .5 for ii in range(14)],
                        ['Canada'] + [pt.value for pt in PT if pt != PT.XX])
                elif 0:
                    plt.yticks(
                        [-ii for ii in range(14 * 2)],
                        ['Const' if ii % 2 else 'AR2' for ii in range(14 * 2)])
                else:
                    plt.yticks([])
                plt.xlabel('Predicted emissions (kt CO2e)')
                plt.title('Predictions of CO2 from Harvested Wood Products')
                ax.yaxis.tick_right()
                #plt.xlim(-130_000, 20_000)
                plt.legend(loc='lower left')
                plt.tight_layout()

        return RVAL()


    def log_prob_2023_AR2(self):
        return float('nan')
        import numpy as np
        import planzero
        from planzero.nir_ar2 import NIR2025_AR2
        from planzero import nir2025

        IPCC_Sector = planzero.enums.IPCC_Sector
        GHG = planzero.enums.GHG

        sector = IPCC_Sector.Harvested_Wood_Products
        ghg = GHG.CO2
        model = self.ar2_model(sector, ghg)

        arr_pt, arr_ca = nir2025.ktCO2e_dense_w_nan()
        mu_ca = np.sum(model.post_samples['mu'], axis=2)

        arr_idx_of_2023 = arr_ca.shape[2] - 1
        mu_idx_of_2023 = arr_idx_of_2023 - 2 # 2 from it being AR2 model


        eval_mu = np.zeros((1000, 14))
        eval_mu[:, :13] = model.post_samples['mu'][:, mu_idx_of_2023]
        eval_mu[:, 13] = mu_ca[:, mu_idx_of_2023]

        eval_sigma = np.zeros(14)
        eval_sigma[:13] = model.noise_ca * model.pt_rms
        eval_sigma[13] = model.noise_ca

        eval_obs = np.zeros(14)
        eval_obs[:13] = arr_pt[
            nir2025.idx_of_sector[model.sector],
            nir2025.idx_of_ghg[model.ghg],
            :,
            arr_idx_of_2023]
        eval_obs[13] = arr_ca[
            nir2025.idx_of_sector[model.sector],
            nir2025.idx_of_ghg[model.ghg],
            arr_idx_of_2023]

        import numpyro.distributions as dist
        log_probs = dist.Normal(
            eval_mu * model.scale,
            eval_sigma * model.scale
        ).log_prob(
            eval_obs
        )
        logprob_X = np.sum(log_probs, axis=1)
        from scipy.special import logsumexp
        rval = logsumexp(logprob_X, b=1.0 / len(logprob_X))

        return rval




class StaticNormals(BlogPost):
    """
    This post introduces a "Static Normals" probabilistic model to PlanZero 
    that is the first to fit overarching assumptions to data, and the first
    to make predictions.
    The overarching assumption is that regional sub-totals should add up to
    national totals (which is often the case in the data, but not always).
    The predictions are trivial: that all years are the same.
    These predictions aren't very precise because the model
    assumes variation from year to year
    is all measurement error and the historical range of variation will continue indefinitely.
    Static Normals is intended as a baseline in PlanZero against which more complex
    models will be judged: they shouldn't be poorer predictors of the future
    than Static Normals.
    """

    def __init__(self):
        super().__init__(
            date=datetime.datetime(2026, 6, 28),
            title='Static Normals',
            # html/blog/2026-06-28-static-normals.html
            url_filename="2026-06-28-static-normals",
            author="James Bergstra",
            tags={#BlogTag.NIR_Modelling,
                  #'Static_Normals',
                  BlogTag.About,
                 },
            status=BlogStatus.Planned,
            )


class ProbabilisticNIR2025(BlogPost):
    """This post introduces a PlanZero's first probabilistic model:
    an interpretation of the NIR-2025 data including its uncertainty
    estimates. The PlanZero site now includes a Models tab
    with a section for probabilistic models, which represent and visualize
    emissions uncertainty. This treatment of uncertainty is a fundamental
    aspect of PlanZero's future modelling work.
    This post also introduces "Planned" status for posts as a mechanism for communicating
    roadmap and organizing ongoing work.
    """

    def __init__(self):
        super().__init__(
            date=datetime.datetime(2026, 5, 20),
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
        from .enums import IPCC_Sector, GHG
        from .html import html_by_ghg
        table = dict(nir2025.annex2_source_category_by_sector())
        assert (IPCC_Sector.Cropland, GHG.CH4) not in table
        table[IPCC_Sector.Cropland, GHG.CH4] = 'Blend: Conversion of Forest Land and Grass Land'
        table[IPCC_Sector.Settlements, GHG.CH4] = 'Blend: Conversion of Forest Land and Grass Land'
        table[IPCC_Sector.Settlements, GHG.N2O] = 'Blend: Conversion of Forest Land and Grass Land'
        with open(path, 'w') as ofile:
            for sector in IPCC_Sector:
                sources = set(
                    [table[sector, ghg] for ghg in GHG
                     if (sector, ghg) in table])
                if len(sources) == 0:
                    raise NotImplementedError() # doesn't happen
                    ofile.write(f'<tr><td>{sector.catpath_with_whitespace}</td>')
                    ofile.write(f'<td></td>')
                    ofile.write(f'<td>undefined</td></tr>\n')
                elif len(sources) == 1:
                    source_cat, = sources
                    ofile.write(f'<tr><td>{sector.catpath_with_whitespace}</td>')
                    ofile.write(f'<td>All</td>')
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
        from .enums import IPCC_Sector, GHG

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
            def build_figure(self):
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
            def build_figure(self):
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
            def build_figure(self):
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
                ax0.set_title("National Public Electricity $\mathrm{CO_2}$ Emissions in 1995")
                ax0.set_xlabel("Emissions ($\mathrm{MtCO_2e}$)")
                ax0.set_ylabel("Probability Density")
                axtexts(
                    ax0,
                    [f"$\mu={ca_dist.mu / 1000:.2f}~MtCO_2e$",
                     f"$\\rho={ca_dist.rolloff / 1000:.2f}~MtCO_2e$",
                     f"$\sigma={ca_dist.relerr * 100:.2f}$%",
                    ])

                x = np.linspace(0, 150, 500)
                pt_dist = pt_dists[idx_of_pt[PT.NU]]
                ax1.plot(
                    x,
                    np.exp(pt_dist.log_prob(x)))
                ax1.set_title("Nunavut Public Electricity $\mathrm{CO_2}$ Emissions in 1995")
                ax1.set_xlabel("Emissions ($\mathrm{ktCO_2e}$)")
                ax1.set_ylabel("Probability Density")
                axtexts(
                    ax1,
                    [f"$\mu={pt_dist.mu:.2f}~ktCO_2e$",
                     f"$\\rho={pt_dist.rolloff:.2f}~ktCO_2e$",
                     f"$\sigma={pt_dist.relerr * 100:.2f}$%",
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
                ax.set_title("National Forest Land $\mathrm{CO_2}$ Emissions in 2009")
                ax.set_xlabel("Emissions ($\mathrm{MtCO_2e}$)")
                ax.set_ylabel("Probability Density")
                axtexts(
                    ax,
                    [f"$\mu={ca_dist.mu / 1000:.2f}~MtCO_2e$",
                     f"$\\rho={ca_dist.rolloff / 1000:.2f}~MtCO_2e$",
                     f"$\sigma={ca_dist.relerr * 100:.2f}$%",
                    ],
                    left_offset=.6,
                    )

                x = np.linspace(-50, 350, 500)
                pt_dist = pt_dists[idx_of_pt[PT.ON]]
                ax = ax3
                ax.plot(
                    x,
                    np.exp(pt_dist.log_prob(x * 1000)))
                ax.set_title("Ontario Forest Land $\mathrm{CO_2}$ Emissions in 2009")
                ax.set_xlabel("Emissions ($\mathrm{MtCO_2e}$)")
                ax.set_ylabel("Probability Density")
                axtexts(
                    ax,
                    [f"$\mu={pt_dist.mu / 1000:.2f}~MtCO_2e$",
                     f"$\\rho={pt_dist.rolloff / 1000:.2f}~MtCO_2e$",
                     f"$\sigma={pt_dist.relerr * 100:.2f}$%",
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
            date=datetime.datetime(2026, 4, 23),
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
            date=datetime.datetime(2026, 4, 9),
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
            date=datetime.datetime(2026, 4, 1),
            title='Heavy-Duty Diesel Vehicles: Emissions Calculations',
            url_filename="2026-04-01-heavy-duty-diesel",
            author="James Bergstra",
            status=BlogStatus.Done,
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
            status=BlogStatus.Done,
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
            date=datetime.datetime(2026, 3, 26),
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
            date=datetime.datetime(2026, 3, 11),
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
            date=datetime.datetime(2026, 3, 2),
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
            date=datetime.datetime(2026, 2, 22),
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
            date=datetime.datetime(2026, 2, 12),
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
            date=datetime.datetime(2026, 2, 2),
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

    def build_figure(self):
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
            date=datetime.datetime(2026, 1, 21),
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
            date=datetime.datetime(2026, 1, 6),
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
            date=datetime.datetime(2025, 12, 5),
            title='How might Canada achieve Net-Zero?',
            url_filename="2025-12-05-first-post",
            status=BlogStatus.Done,
            tags={BlogTag.About,},
            author="James Bergstra")


def init_blogs_by_url_filename():
    global _blogs_sorted_by_date
    for cls in _classes:
        obj = cls()
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
