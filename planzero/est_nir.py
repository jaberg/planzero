
from .enums import CoalType
from .ghgvalues import GWP_100

from .ureg import u, kg_by_ghg
from . import ureg
from . import sc_np
from . import objtensor
from . import sts
from . import annex6_np
from . import enums


def est_annex13_electricity_from_coal():
    """Return CO2e by PT"""

    prov_consumption, national_consumption = sc_np.Archived_Electric_Power_Generation_Annual_Fuel_Consumed_by_Electrical_Utility()
    emission_factors = annex6_np.A6_1_10_and_12()
    emissions = (emission_factors * prov_consumption[CoalType]).sum(enums.CoalType)
    co2e = GWP_100 @ emissions
    return co2e


class EstAnnex13ElectricityFromNG(object):
    def __init__(self):

        # https://en.wikipedia.org/wiki/Natural_gas#Energy_content,_statistics,_and_pricing
        # Wikipedia says 39
        # also Gemini says there is a standard value used by Statistics Canada and National Energy Board
        # of 38.32, but no citation
        Natural_Gas_gross_heating_value_marketable = 38.32 * u.MJ / u.m3_NG_mk

        # Gemini says 40 to 45, no citation given
        # but could be corroborated by averaging the various typical gases? What about moisture?
        Natural_Gas_gross_heating_value_nonmarketable = 42 * u.MJ / u.m3_NG_nmk

        # This is a guess
        Natural_Gas_Peaker_thermal_efficiency = .30

        EP = enums.ElectricityProducer
        EGT = enums.ElectricityGenerationTech
        PT = enums.PT
        GHG = enums.GHG
        NGU = enums.NaturalGasUser

        thermal_efficiency = objtensor.empty(EP, enums.PT)
        thermal_efficiency[:] = Natural_Gas_Peaker_thermal_efficiency
        thermal_efficiency[EP.Utilities, PT.ON] = .5 # more use of combined cycle
        thermal_efficiency[EP.Utilities] *= Natural_Gas_gross_heating_value_marketable
        thermal_efficiency[EP.Industry] *= Natural_Gas_gross_heating_value_nonmarketable

        prov_power_gen, _ = sc_np.Electric_Power_Annual_Generation_by_Class_of_Producer()
        prov_ng_vol = prov_power_gen[:, EGT.CombustionTurbine] / thermal_efficiency

        emission_factors = objtensor.empty(GHG, EP, PT)
        T34 = annex6_np.A6_1_3_and_1_4()
        emission_factors[GHG.CO2, EP.Utilities] = annex6_np.A6_1_1()
        emission_factors[GHG.CO2, EP.Industry] = annex6_np.A6_1_2()
        emission_factors[[GHG.CH4, GHG.N2O], EP.Utilities] = T34[:, NGU.ElectricUtilities]
        emission_factors[[GHG.CH4, GHG.N2O], EP.Industry] = T34[:, NGU.Producer]
        for ghg in [GHG.HFCs, GHG.PFCs, GHG.SF6, GHG.NF3]:
            emission_factors[ghg, EP.Utilities] = 0 * kg_by_ghg[ghg] / u.m3_NG_mk
            emission_factors[ghg, EP.Industry] = 0 * kg_by_ghg[ghg] / u.m3_NG_nmk

        self.thermal_efficiency = thermal_efficiency
        self.prov_power_gen = prov_power_gen
        self.prov_ng_vol = prov_ng_vol
        self.emission_factors = emission_factors
        self.emissions = (emission_factors * prov_ng_vol).sum(EP)
        self.co2e = GWP_100 @ self.emissions
