from .enums import GHG
from . import objtensor
from .ureg import u


# TODO: these values can change over time, consider using STS instead of
# scalars
GWP_100 = objtensor.from_dict({
    GHG.CO2: 1.0 * u.kg_CO2e / u.kg_CO2,
    GHG.CH4: 28.0 * u.kg_CO2e / u.kg_CH4,
    GHG.N2O: 265.0 * u.kg_CO2e / u.kg_N2O,
    GHG.HFCs: 1_430.0 * u.kg_CO2e / u.kg_HFC,
    GHG.PFCs: 6_630.0 * u.kg_CO2e / u.kg_PFC,
    GHG.SF6: 23_500.0 * u.kg_CO2e / u.kg_SF6,
    GHG.NF3: 16_100.0 * u.kg_CO2e / u.kg_NF3})
