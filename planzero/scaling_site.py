import datetime

from . import prob
from .enums import IPCC_Sector
from .scaling_model import Scaling_ScenarioModel


class ScalingSiteInference(prob.SiteInference):
    """Maximal deployment of available, modelled products,
    based on a baseline assumption of static emissions amounts."""

    include_in_registry:bool = True

    def __init__(self):
        data_cutoff = datetime.date(year=2024, month=12, day=31)
        # registry_name = (
        #         f'StaticNormals_data-cutoff'
        #         f'-{data_cutoff.year}'
        #         f'-{data_cutoff.month}'
        #         f'-{data_cutoff.day}'
        #         '_lfy-2050')
        registry_name = 'Scaling_2024_12_31'
        scenario_model = Scaling_ScenarioModel(
                data_cutoff=data_cutoff,
                last_forecast_year=2050,
                num_warmup=1500,
                num_samples=500,
                )
        super().__init__(
                scenario_model_registry_name=registry_name,
                scenario_model=scenario_model,
                )

    @property
    def pretty_name(self) -> str:
        return 'Scaling'

    @property
    def url_name(self) -> str:
        return 'Scaling'

    def one_line_description(self):
        return "Hypothetical scaling of emission-reduction measures within an otherwise non-time-varying economy"

    @property
    def show_on_models_page(self) -> bool:
        return True

    def sectors_affected_by_strategy(self, strategy_id:str) -> set[IPCC_Sector]:
        if strategy_id.startswith('Scale_Bovaer'):
            return {IPCC_Sector.Enteric_Fermentation,
                    IPCC_Sector.Other_Product_Manufacture_and_Use,}
        else:
            raise KeyError(strategy_id)

    @property
    def posts_developing_this_page(self) -> list[str]:
        return [
                'ProbabilisticBovaer',
                'StaticNormals',
                'ModellingBovaer',
                ]

    def challenge_result_url(self, challenge_name) -> str:
        if challenge_name in (
                'PreNIR_2025_m04',
                'NetZero_2050',
                ):
            return f'/models/{self.url_name}/#{challenge_name}'
        else:
            return ''
