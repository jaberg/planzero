import enum


class ProvinceTerritory(str, enum.Enum):
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
    # used in PTValues for the delta between public provincial
    # totals and the public national total, which may include sensored
    # provincial subtotals.
    XX = 'Unknown Province or Territory'

    def two_letter_code(self):
        return str(self)[-2:]

PT = ProvinceTerritory


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


# Coal Types used in StatsCan Table 25-10-0017-01
class CoalType(str, enum.Enum):
    CanadianBituminous = 'Canadian bituminous coal'
    CanadianSubbituminous = 'Canadian subbituminous coal'
    Lignite = 'Lignite'
    ImportedBituminous = 'Imported bituminous coal'
    ImportedSubbituminous = 'Imported subbituminous coal'


class FuelType(str, enum.Enum):
    # Used in Archived table 25-10-0017-01
    LightFuelOil = 'Light fuel oil'
    HeavyFuelOil = 'Heavy fuel oil'
    Diesel = 'Diesel'
    Wood = 'Wood'
    OtherSolid = "Other solid fuels"
    OtherGaseous = "Other gaseous fuels"

    # Used in NIR Annex6 Table 1-6
    Gasoline = 'Motor gasoline'
    Kerosene = 'Kerosene'
    PetCoke = 'Petroleum coke'
    StillGas = 'Still gas'

    Methane = 'Methane'

    NaturalGasMkt = 'Natural gas'
    NaturalGasNonMkt = 'Natural gas (non-marketable)'


class GHG(str, enum.Enum):
    CO2 = 'CO2'
    CH4 = 'CH4'
    N2O = 'N2O'
    HFCs = 'HFCs'
    PFCs = 'PFCs'
    SF6 = 'SF6'
    NF3 = 'NF3'


class ElectricityProducer(str, enum.Enum):
    Utilities = 'Utilities'
    Industry = 'Industry' # Aka producers sometimes, of e.g. Natural Gas


# from ECCC NIR Annex 6 Table 1-3
class NaturalGasUser(str, enum.Enum):
    ElectricUtilities = 'Electric Utilities'
    Industrial = 'Industrial'
    Producer = 'Producer Consumption'
    Pipelines = 'Pipelines'
    Cement = 'Cement'
    Manufacturing = 'Manufacturing Industries'
    Residential = 'Residential'
    Construction = 'Construction'
    Commercial = 'Commercial/Institutional'
    Agriculture = 'Agriculture'


# from ECCC NIR Annex 6 Table 1-6 and 1-3
class RPP_User(str, enum.Enum):
    ElectricUtilities = "Electric Utilities"
    Industrial = "Industrial"
    Producer = "Producer Consumption"
    Residential = "Residential"
    Commercial = "Construction, Forestry, Public Admin, Commercial/Institutional"
    RefineriesAndOthers = "Refineries and Others"
    UpgradingFacilities = "Upgrading Facilities"
