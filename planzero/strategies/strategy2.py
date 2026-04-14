
from pydantic import Field, computed_field

from ..ureg import u
from ..enums import IPCC_Sector, StandardScenarios, PT
from ..base import DynamicElement
from .. import sts
from .. import objtensor

strategies = {} # classname -> Singleton instance


class Strategy2(DynamicElement):

    @classmethod
    def __init_subclass__(cls):
        super().__init_subclass__()
        strategies[cls.__name__] = cls()

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.tags.add('strategy')

    @computed_field
    def ipcc_sector_values(self) -> list[str]:
        return [sec.value for sec in self.ipcc_sectors]


class ScaleBovaer(Strategy2):

    @computed_field
    def short_description(self) -> str:
        return f"Use as much Bovaer as farmers will take"

    @computed_field
    def ipcc_sectors(self) -> list[object]:
        return [IPCC_Sector.Enteric_Fermentation,
                IPCC_Sector.Other_Product_Manufacture_and_Use, # sync with BovinePopulation
               ]

    @computed_field
    def scenarios(self) -> list[object]:
        return [StandardScenarios.Scaling]

    @computed_field
    def research(self) -> dict[str, str]:
        return {
            'bovaer dairy safety NIH': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC8603004'
        }

    def on_add_project(self, state):
        with state.defining(self) as ctx:
            ctx.bovine_population_fraction_on_bovaer = sts.SparseTimeSeries(
                default_value=0 * u.dimensionless)
        return 2025 * u.years

    def step(self, state, current):
        # The strategy here, is to use as much Bovaer as farmers will take
        max_fraction = state.latest.max_fraction_of_cattle_on_bovaer
        current.bovine_population_fraction_on_bovaer = max_fraction
        return state.t_now + 1 * u.years
