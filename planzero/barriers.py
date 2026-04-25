from pydantic import Field, computed_field

from .ureg import u
from .enums import IPCC_Sector, StandardScenarios, PT
from .base import DynamicElement
from . import sts
from . import objtensor

barriers = {} # classname -> Singleton instance


class Barrier(DynamicElement):

    @classmethod
    def __init_subclass__(cls):
        super().__init_subclass__()
        barriers[cls.__name__] = cls()

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.tags.add('barrier')

    @computed_field
    def ipcc_sector_values(self) -> list[str]:
        return [sec.value for sec in self.ipcc_sectors]


from . import cattle
