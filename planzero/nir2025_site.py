from .nir2025_model import NIR2025_ScenarioModel
from .prob import SiteInference as SiteModel


class NIR2025(SiteModel):
    """Emissions per province and territory,
    and per greenhouse gas, are taken from the 2025 National Inventory Report data,
    incorporating uncertainty estimates from Annex 2 of the report.
    """

    include_in_registry:bool = True

    def one_line_description(self) -> str:
        return "Emissions with uncertainty from NIR-2025"

    def __init__(self):
        super().__init__(
                scenario_model_registry_name='NIR2025',
                scenario_model=NIR2025_ScenarioModel(),
                )

    @property
    def predicted_emissions_2050_MtCO2e_bounds_str(self) -> str:
        return "N/A"
