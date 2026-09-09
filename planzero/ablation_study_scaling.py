from typing import ClassVar

from pydantic import computed_field

from . import cattle, prob
from .ablation import AblationStudy
from .sparkline_echart_helper import (
        #PseudoRegion,
        #PseudoSectors,
        RegionalSparklineEChartHelperBase,
        SparklineEChartHelperBase,
)


class SparklineEChartHelper(SparklineEChartHelperBase):
    pass

class RegionalSparklineEChartHelper(RegionalSparklineEChartHelperBase):
    pass


class ScalingSiteInference(prob.SiteInference):
    """Maximal deployment of existing products"""

    strategy_id: str|None

    @computed_field
    def ablation_study_id(self) -> str|None:
        return 'Scaling'

    @property
    def identifier(self) -> str:
        if self.strategy_id is None:
            return f'ScalingStudy_All_Strategies'
        else:
            return f'ScalingStudy_wo_{self.strategy_id}'

    @computed_field
    def pretty_name(self) -> str:
        if self.strategy_id is None:
            return 'Scaling (All strategies)'
        else:
            return f'Scaling (minus {self.strategy_id})'

    @computed_field
    def show_on_models_page(self) -> bool:
        return (self.strategy_id is None)


class Scaling(AblationStudy):

    include_in_registry: ClassVar[bool] = True

    def barriers(self) -> list:
        return [
            cattle.Cattle_Population_AR(),
            cattle.Bovaer_Adoption_Limit(),
            cattle.Bovaer_Production_Emission_Factors(),
            cattle.Cattle_Enteric_Emission_Rates_NIR2025_Bovaer(),
            cattle.Bovaer_Purchase_Cost(),
            cattle.Bovaer_Farm_Subsidy(),
            cattle.Bovaer_Monitoring(),
        ]

    @property
    def strategy_ids(self) -> list[str]:
        return ['Scale_Bovaer']

    def install_site_inferences(self):
        site_inf = ScalingSiteInference(strategy_id=None)
        prob.registry[site_inf.identifier] = site_inf

        for strategy_id in self.strategy_ids:
            site_inf = ScalingSiteInference(strategy_id=strategy_id)
