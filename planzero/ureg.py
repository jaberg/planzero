import pint
from . import enums

ureg = pint.UnitRegistry()

ureg.define('CAD = [currency]')
ureg.define('USD = 1.35 CAD')

ureg.define('people = [human_population]')
ureg.define('cattle = [bovine_population]')

ureg.define('fraction = [] = frac')
ureg.define('ppm = 1e-6 fraction')
ureg.define('ppb = 1e-9 fraction')

for mass_substance in ['coal_bit', 'coal_subbit', 'lignite',
                       'CO2', 'CO2e', 'CH4', 'N2O', 'HFC', 'PFC', 'SF6', 'NF3']:
    ureg.define(f"kg_{mass_substance} = [mass_{mass_substance}]")
    ureg.define(f"tonne_{mass_substance} = 1000 * kg_{mass_substance}")
    ureg.define(f"kilotonne_{mass_substance} = 1_000_000 * kg_{mass_substance} = kt_{mass_substance}")
    ureg.define(f"megatonne_{mass_substance} = 1_000_000_000 * kg_{mass_substance} = Mt_{mass_substance}")
    ureg.define(f"gigatonne_{mass_substance} = 1_000_000_000_000 * kg_{mass_substance} = Gt_{mass_substance}")

    ureg.define(f"pounds_{mass_substance} = 1 / 2.20462 * kg_{mass_substance}")

    ureg.define(f"g_{mass_substance} = 0.001 * kg_{mass_substance}")

# TODO: move to test file
assert ureg.Quantity("1 Mt_CO2").to('kg_CO2').magnitude == 1_000_000_000

for volume_substance in ['NG_mk', 'NG_nmk',
                         'diesel',
                         'gasoline',
                         'LFO',  # Light Fuel Oil
                         'HFO',  # Heavy Fuel Oil
                        ]:
    ureg.define(f"m3_{volume_substance} = [volume_{volume_substance}]")
    ureg.define(f"l_{volume_substance} = .001 * m3_{volume_substance}")
    ureg.define(f"litres_{volume_substance} = .001 * m3_{volume_substance}")
    ureg.define(f"kilolitres_{volume_substance} = m3_{volume_substance}")

u = ureg
u.m3 = u.m ** 3

litres_by_fuel_type = {
    enums.FuelType.LightFuelOil: u.l_LFO,
    enums.FuelType.HeavyFuelOil: u.l_HFO,
    enums.FuelType.Diesel: u.l_diesel,
    enums.FuelType.Gasoline: u.l_gasoline,
}

kg_by_ghg = {
    enums.GHG.CO2: u.kg_CO2,
    enums.GHG.CH4: u.kg_CH4,
    enums.GHG.N2O: u.kg_N2O,
    enums.GHG.HFCs: u.kg_HFC,
    enums.GHG.PFCs: u.kg_PFC,
    enums.GHG.SF6: u.kg_SF6,
    enums.GHG.NF3: u.kg_NF3,
}

kg_by_coal_type = {
    enums.CoalType.CanadianBituminous: u.kg_coal_bit,
    enums.CoalType.CanadianSubbituminous: u.kg_coal_bit,
    enums.CoalType.Lignite: u.kg_lignite,
    enums.CoalType.ImportedBituminous: u.kg_coal_subbit,
    enums.CoalType.ImportedSubbituminous: u.kg_coal_subbit,
}


from .enums import * # XXX
