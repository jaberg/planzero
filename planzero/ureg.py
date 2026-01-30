import enum
import pint

ureg = pint.UnitRegistry()

ureg.define('CAD = [currency]')
ureg.define('USD = 1.35 CAD')

ureg.define('people = [human_population]')
ureg.define('cattle = [bovine_population]')

ureg.define('fraction = [] = frac')
ureg.define('ppm = 1e-6 fraction')
ureg.define('ppb = 1e-9 fraction')

for mass_substance in ['coal', 'CO2', 'CO2e', 'CH4', 'N2O']:
    ureg.define(f"kg_{mass_substance} = [mass_{mass_substance}]")
    ureg.define(f"tonne_{mass_substance} = 1000 * kg_{mass_substance}")
    ureg.define(f"kilotonne_{mass_substance} = 1_000_000 * kg_{mass_substance} = kt_{mass_substance}")
    ureg.define(f"megatonne_{mass_substance} = 1_000_000_000 * kg_{mass_substance} = Mt_{mass_substance}")

    # TODO: move to test file
    assert ureg.Quantity("1 Mt_coal").to('kg_coal').magnitude == 1_000_000_000

u = ureg
u.m3 = u.m ** 3


class Geo(str, enum.Enum):
    CA = 'Canada'
    BC = 'British Columbia'
    AB = 'Alberta'
    SK = 'Saskatchewan'
    MB = 'Manitoba'
    ON = 'Ontario'
    QC = 'Quebec'
    NS = 'Nova Scotia'
    NB = 'New Brunswick'
    PE = 'Prince Edward Island'
    NL = 'Newfoundland and Labrador'
    YT = 'Yukon'
    NT = 'Northwest Territories'
    NU = 'Nunavut'

    @classmethod
    def codes(cls):
        for thing in cls:
            return str(thing)[4:]

    def code(self):
        rval = str(self)[4:]
        assert len(rval) == 2
        return rval

    @classmethod
    def provinces_and_territories(cls):
        for thing in cls:
            if thing != Geo.CA:
                yield thing


class ElectricityGenerationTech(str, enum.Enum):
    Hydro = 'Hydraulic turbine'
    Tidal = 'Tidal power turbine'
    Wind = 'Wind power turbine'
    ConventionalSteam = 'Conventional steam turbine'
    Nuclear = 'Nuclear steam turbine'
    InternalCombustion = 'Internal combustion turbine'
    CombustionTurbine = 'Combustion turbine'
    Solar = 'Solar'
    Other = 'Other types of electricity generation'
    Geothermal = 'Geothermal'
