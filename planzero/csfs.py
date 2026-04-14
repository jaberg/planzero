from pydantic import computed_field

from .base import DynamicElement
from .enums import IPCC_Sector, StandardScenarios
from .ureg import u

from . import sts

csfs = {} # classname -> Singleton instance


class CSFs(DynamicElement):

    @computed_field
    def kpi_targets(self) -> list[str]:
        return []

    @classmethod
    def __init_subclass__(cls):
        super().__init_subclass__()
        csfs[cls.__name__] = cls()

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.tags.add('CSF')

    @computed_field
    def ipcc_sector_values(self) -> list[str]:
        return [sec.value for sec in self.ipcc_sectors]


class EntericFermentation_CSFs(CSFs):
    """All formalized CSFs related to Enteric Fermentation
    """

    @computed_field
    def kpi_targets(self) -> dict[str, str | float]:
        return {
            'bovine_methane_per_head': 'Minimize',
            'bovine_headcount': 'Minimize',
        }

    @computed_field
    def ipcc_sectors(self) -> list[object]:
        return [IPCC_Sector.Enteric_Fermentation]

    @computed_field
    def scenarios(self) -> list[object]:
        return [StandardScenarios.Scaling]


    def on_add_project(self, state):
        state.declare_read_current_sts(self, 'bovine_headcount')
        state.declare_read_current_sts(self, 'bovine_methane_rate')
        with state.defining(self) as ctx:
            ctx.bovine_methane_per_head = sts.SparseTimeSeries(
                default_value=0 * u.kg_CH4 / u.cattle / u.year,
                t_unit=u.year)

    def step(self, state, current):
        current.bovine_methane_per_head = (
            current.bovine_methane_rate
            / current.bovine_headcount)
