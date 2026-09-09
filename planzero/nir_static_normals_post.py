import datetime

from .blog import BlogPost, BlogStatus, BlogTag
from .enums import GHG, IPCC_Sector
from .html import HTML_Matplotlib_Figure


class StaticNormals(BlogPost):
    """
    This post introduces a "Static Normals" probabilistic model to PlanZero
    which is the first to impose modelling assumptions on the NIR,
    and the first to make predictions about future NIR emissions.
    The predictions are trivial: all years are the same (static, rather than dynamic).
    This model is intended as a baseline against which more complex
    future models can be judged.
    Several appendices to this post introduce probabilistic modelling notation,
    Bayesian inference, the Pre-NIR-2025-m06 emissions prediction challenge, and
    describe changes to PlanZero's build system to support probabilistic models.
    """

    def __init__(self):
        super().__init__(
            date=datetime.date(2026, 6, 28),
            title='Static Normals: A baseline predictive model',
            # html/blog/2026-06-28-static-normals.html
            url_filename="2026-06-28-static-normals",
            author="James Bergstra",
            tags={#BlogTag.NIR_Modelling,
                  #'Static_Normals',
                  BlogTag.About,
                 },
            status=BlogStatus.Draft,
            )

    def figure_normal(self,):
        import numpy as np
        import numpyro.distributions as dist

        class RVAL(HTML_Matplotlib_Figure):
            def build_figure_plt(self, plt):
                _, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4))
                ax0.set_title("Normal")
                x = np.linspace(-3, 3, 100)
                ax0.plot(x, np.exp(dist.Normal(0, 1).log_prob(x)), label=r"$\mu=0, \sigma^2=1$")
                ax0.plot(x, np.exp(dist.Normal(-1, 0.7).log_prob(x)), label=r"$\mu=-1, \sigma^2=0.7$")
                ax0.set_ylabel('Density')
                ax0.legend(loc='upper right')

                ax1.set_title("Lognormal")
                x = np.linspace(0, 1, 100)
                ax1.plot(x, np.exp(dist.LogNormal(0, 1).log_prob(x)), label=r"$\mu=0, \sigma^2=1$")
                ax1.plot(x, np.exp(dist.LogNormal(-1, 0.7).log_prob(x)), label=r"$\mu=-1, \sigma^2=0.7$")
                ax1.legend(loc='upper right')
                plt.tight_layout()
        return RVAL()

    def generate_assets(self):
        from . import prob
        model_name = 'Static_Normals_2024_12_31'
        site_inference = prob.registry[model_name]
        base = f'html/blog/{self.url_filename}'

        site_inference.uncertain_sparkline_matrix_echart(
            div_id=f"{model_name}_all_sectors",
            v_unit="Mt_CO2e").save_as(
                f'{base}-{model_name}-all_sectors.html')
        site_inference.uncertain_sparkline_matrix_echart(
            div_id=f"{model_name}_all_sectors_B",
            v_unit="Mt_CO2e").save_as(
                f'{base}-{model_name}-all_sectors_B.html')
        site_inference.sector_echart(
            sector=IPCC_Sector.Harvested_Wood_Products,
            ghg=GHG.CO2,
            v_unit="Mt_CO2e").save_as(
                f'{base}-{model_name}-HWP.html')

    def figure_YOLY_2025(self):
        class RVAL(HTML_Matplotlib_Figure):
            def build_figure_plt(_, plt):
                import matplotlib.pyplot as plt
                import numpy as np
                dt = datetime.datetime

                sources = {
                    "NIR-2024": {
                        "pub_date": "2024-05-03",
                        "train_start": "1990-01-01",
                        "train_end": "2022-12-31",
                    },
                    "NIR-2025": {
                        "pub_date": "2025-04-16",
                        #"train_start": "1990-01-01",
                        #"train_end": "2022-12-31",
                        "test_start": "2023-01-01",
                        #"test_start": "1990-01-01",
                        "test_end": "2023-12-31",
                    },
                    #"Hypothetically admissible extra data)": {
                        #"pub_date": "2023-03-31",
                        #"train_start": "2000-01-01",
                        #"train_end": "2022-12-31",
                    #},
                    #"Data set not available at prediction time": {
                        #"pub_date": "2024-09-30",
                        #"train_start": "2000-01-01",
                        #"train_end": "2022-12-31",
                    #}
                }

                fig, ax = plt.subplots(figsize=(8, 3.5))

                for ii, (dset_name, dset_info) in enumerate(sources.items()):
                    for key, val in list(dset_info.items()):
                        dset_info[key] = dt.strptime(val, "%Y-%m-%d")
                    y = -ii

                    # Solid bar: fully available data
                    last_date = None
                    if 'train_start' in dset_info:
                        ax.barh(y, dset_info["train_end"] - dset_info["train_start"], left=dset_info["train_start"],
                                height=0.5, color='steelblue', alpha=0.85,
                                label='Conditioning data' if ii == 0 else None,
                               )
                        last_date = dset_info["train_end"]
                    if 'test_start' in dset_info:
                        ax.barh(y, dset_info["test_end"] - dset_info["test_start"], left=dset_info["test_start"],
                                height=0.5, color='lightblue', alpha=0.85,
                                label="Evaluation data" if ii == 1 else None)
                        last_date = dset_info["test_end"]

                    # Lag bar: not yet published
                    ax.barh(y, dset_info["pub_date"] - last_date, left=last_date,
                            height=0.5, color='tomato', alpha=0.35,
                            hatch='..', edgecolor='tomato',
                            label="Publication delay" if ii == 0 else None)

                prediction_time = dt.strptime("2024-12-31", "%Y-%m-%d")
                ax.axvline(prediction_time,
                           color='crimson', linestyle='--', lw=1.5,
                           label='Prediction date')
                ax.axvline(dt.now(), color='black', linestyle='--', lw=1.5, label='Today')

                ax.set_yticks([-ii for ii in range(len(sources))])
                ax.set_yticklabels([src for src in sources])
                ax.set_xlabel("Date")
                #ax.xaxis.set_major_formatter(lambda x, _: f"–{int(TODAY-x)}d" if x < TODAY else "Today")
                #ax.set_title("The NIR-2025 Year-Out, Last-Year (YOLY-2025) prediction challenge (of year 2023)")
                plt.legend(loc='center left')
                plt.title("Data usage in Pre-NIR-2025-m04")
                plt.tight_layout()
        return RVAL()
