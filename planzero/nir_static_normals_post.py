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
                        dset_info[key] = dt.strptime(val, "%Y-%m-%d")
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

                prediction_time = dt.strptime("2024-05-16", "%Y-%m-%d")
                ax.axvline(prediction_time, color='crimson', linestyle='--', lw=1.5, label='Prediction time')
                ax.axvline(dt.now(), color='black', linestyle='--', lw=1.5, label='Today')

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
            def build_figure_plt(_, plt):
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
