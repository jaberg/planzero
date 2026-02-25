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


class NAICS(str, enum.Enum):
    # Used in SC 25-10-0084-01
    # TODO: use this instead of ElectricityProducer above?
    Electricity_Producers__Total = 'Total all classes of electricity'
    Electricity_Producers__Utilities = 'Electricity producers, electricity utilities [2211]'
    Electricity_Producers__Industries = 'Electricity producers, industries'


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


# Used to organize
class IPCC_Sector_Group(str, enum.Enum):
    Energy = "Energy"
    Industrial_Processes_and_Product_Use = "Industrial Processes and Product Use"
    Agriculture = 'Agriculture'
    Waste = 'Waste'
    LULUCF = 'Land Use, Land Use Change, and Forestry'
    
    Stationary_Combustion_Sources = 'Stationary Combustion Sources'
    Manufacturing = 'Manufacturing'
    Transport = 'Transport'
    Aviation = 'Aviation'
    Road_Transport = 'Road Transport'
    Marine_Transport = 'Marine Transport'
    Other_Transport = 'Other Transport'
    Fugitive_Sources = 'Fugitive Sources'
    Fugitive_Sources__Oil_and_Natural_Gas = 'Fugitive Sources, Oil and Natural Gas'


# Leaf nodes of Canada's IPCC taxonomy
class IPCC_Sector(str, enum.Enum):
    # Energy
    SCS__Public_Electricity_and_Heat = 'Public Electricity and Heat Production'
    SCS__Petroleum_Refining_Industries = 'Petroleum Refining Industries' # Producer consumption for heat & electricity
    SCS__Oil_and_Gas_Extraction = 'Oil and Gas Extraction'
    SCS__Mining = 'Mining'

    SCS__Manufacturing__Iron_and_Steel = 'Iron and Steel'
    SCS__Manufacturing__NonFerrous = 'Non-Ferrous Metals'
    SCS__Manufacturing__Chemical = 'Chemical'
    SCS__Manufacturing__Pulp_and_Paper = 'Pulp and Paper'
    SCS__Manufacturing__Cement = 'Cement'
    SCS__Manufacturing__Other = 'Other Manufacturing'

    SCS__Construction = 'Construction'
    SCS__Commercial_and_Institutional = 'Commercial and Institutional'
    SCS__Residential = 'Residential'
    SCS__Agriculture_and_Forestry = 'Agriculture and Forestry'

    Transport__Air__Domestic_Civil = 'Domestic Aviation (Civil)'
    Transport__Air__Military = 'Military'

    Transport__Road__Light_Duty_Gasoline_Vehicles = 'Light-Duty Gasoline Vehicles'
    Transport__Road__Light_Duty_Gasoline_Trucks = 'Light-Duty Gasoline Trucks'
    Transport__Road__Heavy_Duty_Gasoline_Vehicles = 'Heavy-Duty Gasoline Vehicles'
    Transport__Road__Motorcycles = 'Motorcycles'
    Transport__Road__Light_Duty_Diesel_Vehicles = 'Light-Duty Diesel Vehicles'
    Transport__Road__Light_Duty_Diesel_Trucks = 'Light-Duty Diesel Trucks'
    Transport__Road__Heavy_Duty_Diesel_Vehicles = 'Heavy-Duty Diesel Vehicles'
    Transport__Road__Propane_and_Natural_Gas_Vehicles = 'Propane and Natural Gas Vehicles'

    Transport__Rail = 'Railways'

    Transport__Marine__Domestic = 'Domestic Navigation'
    Transport__Marine__Fishing = 'Fishing'
    Transport__Marine__Military = 'Military Water-Borne Navigation'

    Transport__Other__Agriculture_and_Forestry = 'Off-Road Agriculture and Forestry'
    Transport__Other__Commercial_and_Institutional = 'Off-Road Commercial and Institutional'
    Transport__Other__Mfg_Mining_Construction = 'Off-Road Manufacturing, Mining and Construction'
    Transport__Other__Residential = 'Off-Road Residential'
    Transport__Other__Other = 'Off-Road Other Transportation'
    Transport__Other__Pipeline = 'Pipeline Transport'

    Fugitive__Coal = 'Coal Mining'
    Fugitive__Oil = 'Oil'
    Fugitive__Natural_Gas = 'Natural Gas'
    Fugitive__Venting = 'Venting'
    Fugitive__Flaring = 'Flaring'

    CO2_Transport_and_Storage = 'CO2 Transport and Storage'

    # Industrial
    Cement_Production = 'Cement Production'
    Lime_Production = 'Lime Production'
    Mineral_Product_Use = 'Mineral Product Use'

    Ammonia_Production = 'Ammonia Production'
    Nitric_Acid_Production = 'Nitric Acid Production'
    Adipic_Acid_Production = 'Adipic Acid Production'
    Petrochemical_and_Carbon_Black_Production = 'Petrochemical and Carbon Black Production'

    Iron_and_Steel_Production = 'Iron and Steel Production'
    Aluminium_Production = 'Aluminium Production'
    Magnesium_Production_and_Casting = 'Magnesium Production and Casting'

    Production_and_Consumption_of_Halocarbons = 'Production and Consumption of Halocarbons, SF6 and NF3d'
    Non_Energy_Products_from_Fuels_and_Solvent_Use = 'Non-Energy Products from Fuels and Solvent Use'
    Other_Product_Manufacture_and_Use = 'Other Product Manufacture and Use'

    # Agriculture
    Enteric_Fermentation = 'Enteric Fermentation'
    Manure_Management = 'Manure Management'
    Agricultural_Soils_Direct = 'Direct Sources'
    Agricultural_Soils_Indirect = 'Indirect Sources'
    Field_Burning_of_Agricultural_Residues = 'Field Burning of Agricultural Residues'
    Liming_Urea_Other = 'Liming, Urea Application and Other Carbon-Containing Fertilizers'

    # Waste
    Municipal_Solid_Waste_Landfills = 'Municipal Solid Waste Landfills'
    Industrial_Wood_Waste_Landfills = 'Industrial Wood Waste Landfills'
    Biological_Treatment_of_Solid_Waste = 'Biological Treatment of Solid Waste'
    Incineration_and_Open_Burning_Waste = 'Incineration and Open Burning of Waste'
    Municipal_Wastewater_Treatment_and_Discharge = 'Municipal Wastewater Treatment and Discharge'
    Industrial_Wastewater_Treatment_and_Discharge = 'Industrial Wastewater Treatment and Discharge'

    # Land use, land use change, and forestry
    Forest_Land = 'Forest Land'
    Cropland = 'Cropland'
    Grassland = 'Grassland'
    Wetlands = 'Wetlands'
    Settlements = 'Settlements'

    Harvested_Wood_Products = 'Harvested Wood Products'


class RoundwoodSpeciesGroup(str, enum.Enum):
    Unspecified = 'Unspecified'
    Softwoods = 'Softwoods'
    Hardwoods = 'Hardwoods'


class RoundwoodProductCategory(str, enum.Enum):
    Logs_and_Bolts = 'Logs and bolts'
    Other_Industrial_Roundwood = 'Other industrial roundwood'
    Fuelwood_and_Firewood = 'Fuelwood*b and firewood*c'
    Pulpwood = 'Pulpwood'


class RoundwoodTenure(str, enum.Enum):
    Unspecified = 'Unspecified'
    Private_Land = 'Private land'
    Federal_Land = 'Federal land'
    Provincial_Land = 'Provincial land'
