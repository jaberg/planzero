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
