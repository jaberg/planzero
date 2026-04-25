from pydantic import computed_field

from .base import DynamicElement
from .enums import IPCC_Sector, StandardScenarios
from .ureg import u

from . import sts

csfs = {} # classname -> Singleton instance


class CSFs(DynamicElement):

    @classmethod
    def __init_subclass__(cls):
        super().__init_subclass__()
        csfs[cls.__name__] = cls()

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.tags.add('CSF')

    @computed_field
    def kpi_name(self) -> str:
        raise NotImplementedError()


class Reduce_Methane_per_Cattle_Head(CSFs):
    """Enteric fermentation emissions would be reduced if each head of cattle
    emitted less.
    """

    @computed_field
    def kpi_name(self) -> str:
        return 'bovine_methane_per_head'

    @computed_field
    def target_value(self) -> float:
        return -float('inf') # means "minimize", there's no mechanism to say 0 is bound

    @computed_field
    def ipcc_sectors(self) -> list[object]:
        return [IPCC_Sector.Enteric_Fermentation]

    @computed_field
    def scenarios(self) -> list[object]:
        return [StandardScenarios.Scaling]

    def on_add_project(self, state):
        state.declare_read_current_sts(self, 'total_cattle_headcount')
        state.declare_read_current_sts(self, 'bovine_methane_rate')
        with state.defining(self) as ctx:
            ctx.bovine_methane_per_head = sts.SparseTimeSeries(
                default_value=0 * u.kg_CH4 / u.cattle / u.year,
                t_unit=u.year)

    def step(self, state, current):
        current.bovine_methane_per_head = (
            current.bovine_methane_rate
            / current.bovine_headcount)


class Reduce_Population_Cattle(CSFs):
    """Enteric fermentation emissions would be reduced if there were fewer head of cattle.
    """

    @computed_field
    def kpi_name(self) -> str:
        return 'bovine_headcount'

    @computed_field
    def target_value(self) -> float:
        return -float('inf') # means "minimize", there's no mechanism to say 0 is bound

    @computed_field
    def ipcc_sectors(self) -> list[object]:
        return [IPCC_Sector.Enteric_Fermentation]

    @computed_field
    def scenarios(self) -> list[object]:
        return [StandardScenarios.Scaling]
