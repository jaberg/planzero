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

    @staticmethod
    def from_catpath(catpath):
        # may have whitespace or not
        try:
            return IPCC_Sector_from_catpath_no_whitespace[catpath]
        except KeyError:
            return IPCC_Sector_from_catpath_with_whitespace[catpath]



class StandardScenarios(str, enum.Enum):
    Scaling = 'scaling'


IPCC_Sector.SCS__Public_Electricity_and_Heat.catpath_no_whitespace = 'Stationary_Combustion_Sources/Public_Electricity_and_Heat_Production'
IPCC_Sector.SCS__Public_Electricity_and_Heat.catpath_with_whitespace = 'Stationary Combustion Sources/Public Electricity and Heat Production'
IPCC_Sector.SCS__Petroleum_Refining_Industries.catpath_no_whitespace = 'Stationary_Combustion_Sources/Petroleum_Refining_Industries'
IPCC_Sector.SCS__Petroleum_Refining_Industries.catpath_with_whitespace = 'Stationary Combustion Sources/Petroleum Refining Industries'
IPCC_Sector.SCS__Oil_and_Gas_Extraction.catpath_no_whitespace = 'Stationary_Combustion_Sources/Oil_and_Gas_Extraction'
IPCC_Sector.SCS__Oil_and_Gas_Extraction.catpath_with_whitespace = 'Stationary Combustion Sources/Oil and Gas Extraction'
IPCC_Sector.SCS__Mining.catpath_no_whitespace = 'Stationary_Combustion_Sources/Mining'
IPCC_Sector.SCS__Mining.catpath_with_whitespace = 'Stationary Combustion Sources/Mining'
IPCC_Sector.SCS__Manufacturing__Iron_and_Steel.catpath_no_whitespace = 'Stationary_Combustion_Sources/Manufacturing_Industries/Iron_and_Steel'
IPCC_Sector.SCS__Manufacturing__Iron_and_Steel.catpath_with_whitespace = 'Stationary Combustion Sources/Manufacturing Industries/Iron and Steel'
IPCC_Sector.SCS__Manufacturing__NonFerrous.catpath_no_whitespace = 'Stationary_Combustion_Sources/Manufacturing_Industries/Non-Ferrous_Metals'
IPCC_Sector.SCS__Manufacturing__NonFerrous.catpath_with_whitespace = 'Stationary Combustion Sources/Manufacturing Industries/Non-Ferrous Metals'
IPCC_Sector.SCS__Manufacturing__Chemical.catpath_no_whitespace = 'Stationary_Combustion_Sources/Manufacturing_Industries/Chemical'
IPCC_Sector.SCS__Manufacturing__Chemical.catpath_with_whitespace = 'Stationary Combustion Sources/Manufacturing Industries/Chemical'
IPCC_Sector.SCS__Manufacturing__Pulp_and_Paper.catpath_no_whitespace = 'Stationary_Combustion_Sources/Manufacturing_Industries/Pulp_and_Paper'
IPCC_Sector.SCS__Manufacturing__Pulp_and_Paper.catpath_with_whitespace = 'Stationary Combustion Sources/Manufacturing Industries/Pulp and Paper'
IPCC_Sector.SCS__Manufacturing__Cement.catpath_no_whitespace = 'Stationary_Combustion_Sources/Manufacturing_Industries/Cement'
IPCC_Sector.SCS__Manufacturing__Cement.catpath_with_whitespace = 'Stationary Combustion Sources/Manufacturing Industries/Cement'
IPCC_Sector.SCS__Manufacturing__Other.catpath_no_whitespace = 'Stationary_Combustion_Sources/Manufacturing_Industries/Other_Manufacturing'
IPCC_Sector.SCS__Manufacturing__Other.catpath_with_whitespace = 'Stationary Combustion Sources/Manufacturing Industries/Other Manufacturing'
IPCC_Sector.SCS__Construction.catpath_no_whitespace = 'Stationary_Combustion_Sources/Construction'
IPCC_Sector.SCS__Construction.catpath_with_whitespace = 'Stationary Combustion Sources/Construction'
IPCC_Sector.SCS__Commercial_and_Institutional.catpath_no_whitespace = 'Stationary_Combustion_Sources/Commercial_and_Institutional'
IPCC_Sector.SCS__Commercial_and_Institutional.catpath_with_whitespace = 'Stationary Combustion Sources/Commercial and Institutional'
IPCC_Sector.SCS__Residential.catpath_no_whitespace = 'Stationary_Combustion_Sources/Residential'
IPCC_Sector.SCS__Residential.catpath_with_whitespace = 'Stationary Combustion Sources/Residential'
IPCC_Sector.SCS__Agriculture_and_Forestry.catpath_no_whitespace = 'Stationary_Combustion_Sources/Agriculture_and_Forestry'
IPCC_Sector.SCS__Agriculture_and_Forestry.catpath_with_whitespace = 'Stationary Combustion Sources/Agriculture and Forestry'
IPCC_Sector.Transport__Air__Domestic_Civil.catpath_no_whitespace = 'Transport/Aviation/Domestic_Aviation_(Civil)'
IPCC_Sector.Transport__Air__Domestic_Civil.catpath_with_whitespace = 'Transport/Aviation/Domestic Aviation (Civil)'
IPCC_Sector.Transport__Air__Military.catpath_no_whitespace = 'Transport/Aviation/Military'
IPCC_Sector.Transport__Air__Military.catpath_with_whitespace = 'Transport/Aviation/Military'
IPCC_Sector.Transport__Road__Light_Duty_Gasoline_Vehicles.catpath_no_whitespace = 'Transport/Road_Transportation/Light-Duty_Gasoline_Vehicles'
IPCC_Sector.Transport__Road__Light_Duty_Gasoline_Vehicles.catpath_with_whitespace = 'Transport/Road Transportation/Light-Duty Gasoline Vehicles'
IPCC_Sector.Transport__Road__Light_Duty_Gasoline_Trucks.catpath_no_whitespace = 'Transport/Road_Transportation/Light-Duty_Gasoline_Trucks'
IPCC_Sector.Transport__Road__Light_Duty_Gasoline_Trucks.catpath_with_whitespace = 'Transport/Road Transportation/Light-Duty Gasoline Trucks'
IPCC_Sector.Transport__Road__Heavy_Duty_Gasoline_Vehicles.catpath_no_whitespace = 'Transport/Road_Transportation/Heavy-Duty_Gasoline_Vehicles'
IPCC_Sector.Transport__Road__Heavy_Duty_Gasoline_Vehicles.catpath_with_whitespace = 'Transport/Road Transportation/Heavy-Duty Gasoline Vehicles'
IPCC_Sector.Transport__Road__Motorcycles.catpath_no_whitespace = 'Transport/Road_Transportation/Motorcycles'
IPCC_Sector.Transport__Road__Motorcycles.catpath_with_whitespace = 'Transport/Road Transportation/Motorcycles'
IPCC_Sector.Transport__Road__Light_Duty_Diesel_Vehicles.catpath_no_whitespace = 'Transport/Road_Transportation/Light-Duty_Diesel_Vehicles'
IPCC_Sector.Transport__Road__Light_Duty_Diesel_Vehicles.catpath_with_whitespace = 'Transport/Road Transportation/Light-Duty Diesel Vehicles'
IPCC_Sector.Transport__Road__Light_Duty_Diesel_Trucks.catpath_no_whitespace = 'Transport/Road_Transportation/Light-Duty_Diesel_Trucks'
IPCC_Sector.Transport__Road__Light_Duty_Diesel_Trucks.catpath_with_whitespace = 'Transport/Road Transportation/Light-Duty Diesel Trucks'
IPCC_Sector.Transport__Road__Heavy_Duty_Diesel_Vehicles.catpath_no_whitespace = 'Transport/Road_Transportation/Heavy-Duty_Diesel_Vehicles'
IPCC_Sector.Transport__Road__Heavy_Duty_Diesel_Vehicles.catpath_with_whitespace = 'Transport/Road Transportation/Heavy-Duty Diesel Vehicles'
IPCC_Sector.Transport__Road__Propane_and_Natural_Gas_Vehicles.catpath_no_whitespace = 'Transport/Road_Transportation/Propane_and_Natural_Gas_Vehicles'
IPCC_Sector.Transport__Road__Propane_and_Natural_Gas_Vehicles.catpath_with_whitespace = 'Transport/Road Transportation/Propane and Natural Gas Vehicles'
IPCC_Sector.Transport__Rail.catpath_no_whitespace = 'Transport/Railways'
IPCC_Sector.Transport__Rail.catpath_with_whitespace = 'Transport/Railways'
IPCC_Sector.Transport__Marine__Domestic.catpath_no_whitespace = 'Transport/Marine/Domestic_Navigation'
IPCC_Sector.Transport__Marine__Domestic.catpath_with_whitespace = 'Transport/Marine/Domestic Navigation'
IPCC_Sector.Transport__Marine__Fishing.catpath_no_whitespace = 'Transport/Marine/Fishing'
IPCC_Sector.Transport__Marine__Fishing.catpath_with_whitespace = 'Transport/Marine/Fishing'
IPCC_Sector.Transport__Marine__Military.catpath_no_whitespace = 'Transport/Marine/Military_Water-Borne_Navigation'
IPCC_Sector.Transport__Marine__Military.catpath_with_whitespace = 'Transport/Marine/Military Water-Borne Navigation'
IPCC_Sector.Transport__Other__Agriculture_and_Forestry.catpath_no_whitespace = 'Transport/Other_Transportation/Off-Road_Agriculture_and_Forestry'
IPCC_Sector.Transport__Other__Agriculture_and_Forestry.catpath_with_whitespace = 'Transport/Other Transportation/Off-Road Agriculture and Forestry'
IPCC_Sector.Transport__Other__Commercial_and_Institutional.catpath_no_whitespace = 'Transport/Other_Transportation/Off-Road_Commercial_and_Institutional'
IPCC_Sector.Transport__Other__Commercial_and_Institutional.catpath_with_whitespace = 'Transport/Other Transportation/Off-Road Commercial and Institutional'
IPCC_Sector.Transport__Other__Mfg_Mining_Construction.catpath_no_whitespace = 'Transport/Other_Transportation/Off-Road_Manufacturing,_Mining_and_Construction'
IPCC_Sector.Transport__Other__Mfg_Mining_Construction.catpath_with_whitespace = 'Transport/Other Transportation/Off-Road Manufacturing, Mining and Construction'
IPCC_Sector.Transport__Other__Residential.catpath_no_whitespace = 'Transport/Other_Transportation/Off-Road_Residential'
IPCC_Sector.Transport__Other__Residential.catpath_with_whitespace = 'Transport/Other Transportation/Off-Road Residential'
IPCC_Sector.Transport__Other__Other.catpath_no_whitespace = 'Transport/Other_Transportation/Off-Road_Other_Transportation'
IPCC_Sector.Transport__Other__Other.catpath_with_whitespace = 'Transport/Other Transportation/Off-Road Other Transportation'
IPCC_Sector.Transport__Other__Pipeline.catpath_no_whitespace = 'Transport/Other_Transportation/Pipeline_Transport'
IPCC_Sector.Transport__Other__Pipeline.catpath_with_whitespace = 'Transport/Other Transportation/Pipeline Transport'
IPCC_Sector.Fugitive__Coal.catpath_no_whitespace = 'Fugitive_Sources/Coal_Mining'
IPCC_Sector.Fugitive__Coal.catpath_with_whitespace = 'Fugitive Sources/Coal Mining'
IPCC_Sector.Fugitive__Oil.catpath_no_whitespace = 'Fugitive_Sources/Oil_and_Natural_Gas/Oil'
IPCC_Sector.Fugitive__Oil.catpath_with_whitespace = 'Fugitive Sources/Oil and Natural Gas/Oil'
IPCC_Sector.Fugitive__Natural_Gas.catpath_no_whitespace = 'Fugitive_Sources/Oil_and_Natural_Gas/Natural_Gas'
IPCC_Sector.Fugitive__Natural_Gas.catpath_with_whitespace = 'Fugitive Sources/Oil and Natural Gas/Natural Gas'
IPCC_Sector.Fugitive__Venting.catpath_no_whitespace = 'Fugitive_Sources/Oil_and_Natural_Gas/Venting'
IPCC_Sector.Fugitive__Venting.catpath_with_whitespace = 'Fugitive Sources/Oil and Natural Gas/Venting'
IPCC_Sector.Fugitive__Flaring.catpath_no_whitespace = 'Fugitive_Sources/Oil_and_Natural_Gas/Flaring'
IPCC_Sector.Fugitive__Flaring.catpath_with_whitespace = 'Fugitive Sources/Oil and Natural Gas/Flaring'
IPCC_Sector.CO2_Transport_and_Storage.catpath_no_whitespace = 'CO2_Transport_and_Storage'
IPCC_Sector.CO2_Transport_and_Storage.catpath_with_whitespace = 'CO2 Transport and Storage'
IPCC_Sector.Cement_Production.catpath_no_whitespace = 'Mineral_Products/Cement_Production'
IPCC_Sector.Cement_Production.catpath_with_whitespace = 'Mineral Products/Cement Production'
IPCC_Sector.Lime_Production.catpath_no_whitespace = 'Mineral_Products/Lime_Production'
IPCC_Sector.Lime_Production.catpath_with_whitespace = 'Mineral Products/Lime Production'
IPCC_Sector.Mineral_Product_Use.catpath_no_whitespace = 'Mineral_Products/Mineral_Product_Use'
IPCC_Sector.Mineral_Product_Use.catpath_with_whitespace = 'Mineral Products/Mineral Product Use'
IPCC_Sector.Ammonia_Production.catpath_no_whitespace = 'Chemical_Industry/Ammonia_Production'
IPCC_Sector.Ammonia_Production.catpath_with_whitespace = 'Chemical Industry/Ammonia Production'
IPCC_Sector.Nitric_Acid_Production.catpath_no_whitespace = 'Chemical_Industry/Nitric_Acid_Production'
IPCC_Sector.Nitric_Acid_Production.catpath_with_whitespace = 'Chemical Industry/Nitric Acid Production'
IPCC_Sector.Adipic_Acid_Production.catpath_no_whitespace = 'Chemical_Industry/Adipic_Acid_Production'
IPCC_Sector.Adipic_Acid_Production.catpath_with_whitespace = 'Chemical Industry/Adipic Acid Production'
IPCC_Sector.Petrochemical_and_Carbon_Black_Production.catpath_no_whitespace = 'Chemical_Industry/Petrochemical_and_Carbon_Black_Production'
IPCC_Sector.Petrochemical_and_Carbon_Black_Production.catpath_with_whitespace = 'Chemical Industry/Petrochemical and Carbon Black Production'
IPCC_Sector.Iron_and_Steel_Production.catpath_no_whitespace = 'Metal_Production/Iron_and_Steel_Production'
IPCC_Sector.Iron_and_Steel_Production.catpath_with_whitespace = 'Metal Production/Iron and Steel Production'
IPCC_Sector.Aluminium_Production.catpath_no_whitespace = 'Metal_Production/Aluminium_Production'
IPCC_Sector.Aluminium_Production.catpath_with_whitespace = 'Metal Production/Aluminium Production'
IPCC_Sector.Magnesium_Production_and_Casting.catpath_no_whitespace = 'Metal_Production/SF6_Used_in_Magnesium_Smelters_and_Casters'
IPCC_Sector.Magnesium_Production_and_Casting.catpath_with_whitespace = 'Metal Production/SF6 Used in Magnesium Smelters and Casters'
IPCC_Sector.Production_and_Consumption_of_Halocarbons.catpath_no_whitespace = 'Production_and_Consumption_of_Halocarbons,_SF6_and_NF3'
IPCC_Sector.Production_and_Consumption_of_Halocarbons.catpath_with_whitespace = 'Production and Consumption of Halocarbons, SF6 and NF3'
IPCC_Sector.Non_Energy_Products_from_Fuels_and_Solvent_Use.catpath_no_whitespace = 'Non-Energy_Products_from_Fuels_and_Solvent_Use'
IPCC_Sector.Non_Energy_Products_from_Fuels_and_Solvent_Use.catpath_with_whitespace = 'Non-Energy Products from Fuels and Solvent Use'
IPCC_Sector.Other_Product_Manufacture_and_Use.catpath_no_whitespace = 'Other_Product_Manufacture_and_Use'
IPCC_Sector.Other_Product_Manufacture_and_Use.catpath_with_whitespace = 'Other Product Manufacture and Use'
IPCC_Sector.Enteric_Fermentation.catpath_no_whitespace = 'Enteric_Fermentation'
IPCC_Sector.Enteric_Fermentation.catpath_with_whitespace = 'Enteric Fermentation'
IPCC_Sector.Manure_Management.catpath_no_whitespace = 'Manure_Management'
IPCC_Sector.Manure_Management.catpath_with_whitespace = 'Manure Management'
IPCC_Sector.Agricultural_Soils_Direct.catpath_no_whitespace = 'Agricultural_Soils/Direct_Sources'
IPCC_Sector.Agricultural_Soils_Direct.catpath_with_whitespace = 'Agricultural Soils/Direct Sources'
IPCC_Sector.Agricultural_Soils_Indirect.catpath_no_whitespace = 'Agricultural_Soils/Indirect_Sources'
IPCC_Sector.Agricultural_Soils_Indirect.catpath_with_whitespace = 'Agricultural Soils/Indirect Sources'
IPCC_Sector.Field_Burning_of_Agricultural_Residues.catpath_no_whitespace = 'Field_Burning_of_Agricultural_Residues'
IPCC_Sector.Field_Burning_of_Agricultural_Residues.catpath_with_whitespace = 'Field Burning of Agricultural Residues'
IPCC_Sector.Liming_Urea_Other.catpath_no_whitespace = 'Liming,_Urea_Application_and_Other_Carbon-Containing_Fertilizers'
IPCC_Sector.Liming_Urea_Other.catpath_with_whitespace = 'Liming, Urea Application and Other Carbon-Containing Fertilizers'
IPCC_Sector.Municipal_Solid_Waste_Landfills.catpath_no_whitespace = 'Municipal_Solid_Waste_Landfills'
IPCC_Sector.Municipal_Solid_Waste_Landfills.catpath_with_whitespace = 'Municipal Solid Waste Landfills'
IPCC_Sector.Industrial_Wood_Waste_Landfills.catpath_no_whitespace = 'Industrial_Wood_Waste_Lanfills'
IPCC_Sector.Industrial_Wood_Waste_Landfills.catpath_with_whitespace = 'Industrial Wood Waste Lanfills'
IPCC_Sector.Biological_Treatment_of_Solid_Waste.catpath_no_whitespace = 'Biological_Treatment_of_Solid_Waste'
IPCC_Sector.Biological_Treatment_of_Solid_Waste.catpath_with_whitespace = 'Biological Treatment of Solid Waste'
IPCC_Sector.Incineration_and_Open_Burning_Waste.catpath_no_whitespace = 'Incineration_and_Open_Burning_of_Waste'
IPCC_Sector.Incineration_and_Open_Burning_Waste.catpath_with_whitespace = 'Incineration and Open Burning of Waste'
IPCC_Sector.Municipal_Wastewater_Treatment_and_Discharge.catpath_no_whitespace = 'Municipal_Wastewater_Treatment_and_Discharge'
IPCC_Sector.Municipal_Wastewater_Treatment_and_Discharge.catpath_with_whitespace = 'Municipal Wastewater Treatment and Discharge'
IPCC_Sector.Industrial_Wastewater_Treatment_and_Discharge.catpath_no_whitespace = 'Industrial_Wastewater_and_Discharge'
IPCC_Sector.Industrial_Wastewater_Treatment_and_Discharge.catpath_with_whitespace = 'Industrial Wastewater and Discharge'
IPCC_Sector.Forest_Land.catpath_no_whitespace = 'Forest_Land'
IPCC_Sector.Forest_Land.catpath_with_whitespace = 'Forest Land'
IPCC_Sector.Cropland.catpath_no_whitespace = 'Cropland'
IPCC_Sector.Cropland.catpath_with_whitespace = 'Cropland'
IPCC_Sector.Grassland.catpath_no_whitespace = 'Grassland'
IPCC_Sector.Grassland.catpath_with_whitespace = 'Grassland'
IPCC_Sector.Wetlands.catpath_no_whitespace = 'Wetlands'
IPCC_Sector.Wetlands.catpath_with_whitespace = 'Wetlands'
IPCC_Sector.Settlements.catpath_no_whitespace = 'Settlements'
IPCC_Sector.Settlements.catpath_with_whitespace = 'Settlements'
IPCC_Sector.Harvested_Wood_Products.catpath_no_whitespace = 'Harvested_Wood_Products'
IPCC_Sector.Harvested_Wood_Products.catpath_with_whitespace = 'Harvested Wood Products'

IPCC_Sector_from_catpath_no_whitespace = {
    ipcc_sector.catpath_no_whitespace: ipcc_sector
    for ipcc_sector in IPCC_Sector}
IPCC_Sector_from_catpath_with_whitespace = {
    ipcc_sector.catpath_with_whitespace: ipcc_sector
    for ipcc_sector in IPCC_Sector}
