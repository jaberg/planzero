import datetime

from .blog import BlogPost, BlogStatus, BlogTag
from .enums import GHG, IPCC_Sector
from .html import HTML_Matplotlib_Figure


class StaticNormals(BlogPost):
    """
    This post introduces a "Static Normals" probabilistic model to PlanZero
    which is the first to impose overarching assumptions on the NIR,
    and the first to make predictions about future NIR emissions.
    The predictions are trivial: all years are the same (static, rather than dynamic).
    This model is intended as a baseline against which more complex
    models can be judged; they shouldn't be poorer predictors of the future
    than Static Normals.
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
            status=BlogStatus.Planned,
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
