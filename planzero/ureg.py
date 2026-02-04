import pint
from .enums import FuelType

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
    FuelType.LightFuelOil: u.l_LFO,
    FuelType.HeavyFuelOil: u.l_HFO,
    FuelType.Diesel: u.l_diesel,
}


from .enums import *
