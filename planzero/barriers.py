from pydantic import Field, computed_field

from .ureg import u
from .enums import IPCC_Sector, PT
from .base import DynamicElement
from . import sts
from . import objtensor

# TODO: don't make a registry of these,
# let each AblationStudy / Model define its own barriers
# Then also don't import cattle.py below
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
    def short_description(self) -> str:
        return self.__class__.__doc__



from . import cattle
