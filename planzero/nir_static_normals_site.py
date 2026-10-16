import datetime

from .nir_static_normals_model import StaticNormals_ScenarioModel
from .prob import SiteInference as SiteModel


class Static_Normals_2024_12_31(SiteModel):
    """Emissions per province and territory,
    and per greenhouse gas, modelled as
    non-time-varying Normal distributions.
    National totals are modelled as the sums of provincial
    and territorial totals.
    This is a baseline model, not intended to be accurate.
    """

    include_in_registry:bool = True

    def __init__(self):
        self.data_cutoff = datetime.date(year=2024, month=12, day=31)
        # registry_name = (
        #         f'StaticNormals_data-cutoff'
        #         f'-{data_cutoff.year}'
        #         f'-{data_cutoff.month}'
        #         f'-{data_cutoff.day}'
        #         '_lfy-2050')
        registry_name = 'Static_Normals_2024_12_31'
        scenario_model = StaticNormals_ScenarioModel(
                data_cutoff=self.data_cutoff,
                last_forecast_year=2050,
                )
        super().__init__(
                scenario_model_registry_name=registry_name,
                scenario_model=scenario_model)

    def one_line_description(self) -> str:
        return "Very simple baseline model - non-time-varying estimates"

    @property
    def show_on_models_page(self) -> bool:
        return True

    def challenge_netzero_2050(self):
        return self.scenario_model.p_net_zero_2050(self.comp)

    def prediction_scores_prenir_2025_m04(self):
        return self.scenario_model.prediction_scores_prenir_2025_m04(self.comp)

    def show_prediction_quality(self):
        assert self.scenario_model.data_cutoff == datetime.date(year=2024, month=12, day=31)
        return True

    def challenge_result_url(self, challenge_name) -> str:
        if challenge_name in (
                'PreNIR_2025_m04',
                'NetZero_2050',
                ):
            return f'/models/prob/{self.model_id}/#{challenge_name}'
        else:
            return ''

    @property
    def pretty_name(self) -> str:
        return 'Static Normals (data-cutoff: 2024-12-31)'

    @property
    def posts_developing_this_page(self) -> list[str]:
        return [
                'StaticNormals', # 2026-06-28
                'ProbabilisticNIR2025', # 2026-05-20
                ]
