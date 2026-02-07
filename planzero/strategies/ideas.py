from .. import ipcc_canada
from .stakeholders import ss, Stakeholders
from .stakeholders import orgs, Org

class Ideas(object):
    def __init__(self):
        self._ideas = {}

    def __setattr__(self, name, idea):
        if name.startswith('_'):
            self.__dict__[name] = idea
        elif name in self._ideas:
            raise AttributeError(f'attribute "{name}" is already assigned')
        else:
            self._ideas[name] = idea
            idea.on_assign(name)
        return idea

    def __getattr__(self, name):
        try:
            return self._ideas[name]
        except KeyError:
            raise AttributeError(name)

    def items(self):
        for item in self._ideas.items():
            yield item

    def get(self, name):
        return self._ideas.get(name)

ideas = Ideas()

class Idea(object):
    def __init__(self, *, who=None, descr:str=None, for_whom=(), ipcc_catpaths=(), urls=(), full_name=None):
        self.who = who
        if isinstance(for_whom, Stakeholders):
            self.for_whom = [for_whom]
        else:
            self.for_whom = for_whom
        self.descr = descr
        self.ipcc_catpaths = set(ipcc_catpaths)
        for catpath in self.ipcc_catpaths:
            assert catpath in ipcc_canada.catpaths
        self.urls = urls
        self.name = None
        self._full_name = full_name

    def on_assign(self, name):
        if self.name is None:
            self.name = name
        else:
            assert self.name == name

    @property
    def full_name(self):
        return self._full_name or self.name.replace('_', ' ')

class RegulationIdea(Idea):
    """A regulation change"""


class NewcoIdea(Idea):
    """A project that could be profitable"""
    pass


class EducationIdea(Idea):
    """A project that could deliver a public good"""


class PurchaseIdea(Idea):
    """A purchasing decision should be profitable"""
    pass


class DonationIdea(Idea):
    """A donation decision that might be worthwhile"""
    pass


class ResearchIdea(Idea):
    """A research initiative that might be grant-worthy."""
    # TODO: by specific granting agency, e.g. NSERC
    pass

ss.People_Desiring_Net_Zero = Stakeholders()

ss.Voters = Stakeholders(
    ipcc_catpaths=ipcc_canada.catpaths,
    )


ss.Organizations_discouraging_beef_and_milk_consumption = Stakeholders(
    descr="Organizations discouraging people from consuming beef and milk",
    ipcc_catpaths=['Enteric_Fermentation'],
    )

ss.Organizations_encouraging_beef_and_milk_consumption = Stakeholders(
    descr="Organizations encouraging people from consuming beef and milk",
    ipcc_catpaths=['Enteric_Fermentation'],
    )

ideas.education_to_discourage_beef_and_milk = EducationIdea(
    who=ss.Organizations_discouraging_beef_and_milk_consumption.new_org(org_type='charity'),
    for_whom=ss.People_Desiring_Net_Zero,
    descr="Public education / PR discouraging people from consuming beef and milk",
    ipcc_catpaths=['Enteric_Fermentation'],
    )

ss.Charitable_Donors = Stakeholders()

ideas.donation_to_discourage_beef_and_milk = DonationIdea(
    who=ss.Charitable_Donors,
    descr="Support existing organizations advocating that people refrain from consumption of beef and milk",
    for_whom=ss.Organizations_discouraging_beef_and_milk_consumption,
    ipcc_catpaths=['Enteric_Fermentation'],
    )

ss.NSERC_Funded_Academics = Stakeholders()

ideas.improve_meat_substitutes = ResearchIdea(
    who=ss.NSERC_Funded_Academics,
    descr="Research improved meat substitutes",
    for_whom=ss.Makers_of_Beef_Substitutes,
    ipcc_catpaths=['Enteric_Fermentation'],
    )

ideas.improve_dairy_substitutes = ResearchIdea(
    who=ss.NSERC_Funded_Academics,
    descr="Research improved milk substitutes",
    for_whom=ss.Makers_of_Milk_Substitutes,
    ipcc_catpaths=['Enteric_Fermentation'],
    )

ideas.filter_methane_in_barns = NewcoIdea(
    who=ss.Barn_Equipment_Makers.new_org(),
    descr="Develop barn equipment to filter methane in ventilation systems",
    for_whom=[ss.Beef_Farmers, ss.Dairy_Farmers],
    ipcc_catpaths=['Enteric_Fermentation'],
    )

ideas.better_bovaer = NewcoIdea(
    who=ss.Feed_Additive_Companies.new_org(),
    descr="Develop a more financially-efficient alternative to Bovaer",
    for_whom=[ss.Beef_Farmers, ss.Dairy_Farmers],
    ipcc_catpaths=['Enteric_Fermentation'],
    )

# has the likely effect of shifting the emissions to other countries?
# as well as being unpopular with, say, dairy farmers?
ideas.dismantle_dairy_supply_management = RegulationIdea(
    who=orgs.Canada,
    for_whom=ss.People_Desiring_Net_Zero,
    descr="Remove supply management for dairy, collapse domestric dairy farming industry",
    ipcc_catpaths=['Enteric_Fermentation'],
    )


ss.Corn_Farmers = Stakeholders(
    descr="Collectively grow Canada's corn",
    ipcc_catpaths=['Cropland'],
    )

ss.Wheat_Farmers = Stakeholders(
    descr="Collectively grow Canada's wheat",
    ipcc_catpaths=['Cropland'],
    )
ss.Seed_Companies = Stakeholders(
    descr="Collectively sell the seeds for Canada's farmed land",
    ipcc_catpaths=['Cropland'],
    )

ideas.nitrogen_fixing_wheat = NewcoIdea(
    who=ss.Seed_Companies.new_org(),
    descr="Develop a wheat variant that fixes nitrogen",
    urls=["https://www.ucdavis.edu/food/news/wheat-makes-its-own-fertilizer"],
    for_whom=ss.Wheat_Farmers,
    ipcc_catpaths=['Cropland'],
    )


def ideas_by_catpath(catpath):
    for name, idea in ideas.items():
        if catpath in idea.ipcc_catpaths:
            yield idea


orgs.CanadaSteamshipLines = Org(
    full_name="Canada Steamship Lines (CSL)",
    url="https://cslships.com/")
orgs.AlgomaCentral = Org(
    full_name="Algoma Central Corporation",
    url="https://www.algonet.com/")
orgs.Fednav = Org(
    url="https://www.fednav.com/")
orgs.McKeil = Org(
    full_name="McKeil Marine",
    url="https://mckeil.com/")
orgs.Norvic = Org(
    full_name="Norvic Shipping",
    url="https://norvicshipping.com/company/")


ss.Great_Lakes_Dry_Bulk_Shipping_Companies = Stakeholders(
    ipcc_catpaths=['Transport/Marine/Domestic_Navigation'],
    notable_members=[
        orgs.CanadaSteamshipLines,
        orgs.AlgomaCentral,
        orgs.McKeil,
    ])
"""
<ul>
    <li>Transport large amounts of bulk cargo picked up at ports, and delivered to ports, at predictable times, moved at low price per ton mile?</li>
    <li>Sometimes customers want steady movement (e.g. from mines)</li>
    <li>Sometimes customers want seasonal movement (e.g. grain)</li>
    <li>Sometimes customers want irregular movement? </li>
    <li>Buy fuel from oil companies, refuel at ports</li>
    <li>Subcontract ship construction, delivery, repair</li>
    <li>May Subcontract ship operation</li>
    <li>May have to perform or subcontract cargo hold cleaning</li>
</ul>
<p>
What are their main concerns?
<ul>
    <li>Price of fuel?</li>
    <li>Price of labour?</li>
    <li>Price of ship mortgage financing?</li>
    <li>Canals close for winter, can't compete with rail and truck?</li>
</ul>
</p>
"""

ss.Great_Lakes_Liquid_Bulk_Shipping_Companies = Stakeholders(
    ipcc_catpaths=['Transport/Marine/Domestic_Navigation'],
    notable_members=[
        orgs.McKeil,
    ])

ss.Tug_and_Tow_Companies = Stakeholders(
    ipcc_catpaths=['Transport/Marine/Domestic_Navigation'],
    )

ss.Great_Lakes_Tug_and_Tow_Companies = Stakeholders(
    ipcc_catpaths=['Transport/Marine/Domestic_Navigation'],
    )

ss.Pacific_Tug_and_Tow_Companies = Stakeholders(
    ipcc_catpaths=['Transport/Marine/Domestic_Navigation'],
    )

ss.Atlantic_Tug_and_Tow_Companies = Stakeholders(
    ipcc_catpaths=['Transport/Marine/Domestic_Navigation'],
    )

orgs.Imperial_Oil = Org()
# fully integrated oil company
# owns Esso brand
# operate the Kearl oil sands mind and Cold Lake in-situ (thermal) mining project
# Canadian arm of ExxonMobil, which owns 69.6% of Imperial_Oil

orgs.Suncor = Org()
# fully integrated oil company
# owns Petro Canada brand
# also operates oil sands extraction e.g. "Base Plant" and "Fort Hills"
# operate refineries in Edmonton, Sarnia, Montreal

orgs.Cenovus_Energy = Org()
# massive in-situ (thermal) operator, specializing in SAGD technology
# operates Foster Creek, Christina Lake projects
# now an integrated oil company with puchase of Husky brand (esp. for trucks)

orgs.Canadian_Natural_Resources_Ltd_CNRL = Org()
# largest oil and gas producer in Canada by volume
# own Horizon mine and now Albian Sands mine (bought from Shell)
# an "upstream" company, not a fully-integrated one
# they sell crude oil domestically and internationally

orgs.Shell_Canada = Org()
# Canadian arm of Royal Dutch Shell

ss.Oil_and_Gas_Extraction_Companies = Stakeholders(
    ipcc_catpaths=[
        'Stationary_Combustion_Sources/Oil_and_Gas_Extraction',
    ],
    notable_members=[
        orgs.Imperial_Oil,
        orgs.Suncor,
        orgs.Cenovus_Energy,
        orgs.Canadian_Natural_Resources_Ltd_CNRL,
        orgs.Shell_Canada, # is it involved still?
    ])

ss.Oil_Refinery_Operators = Stakeholders(
    ipcc_catpaths=[
        'Stationary_Combustion_Sources/Oil_and_Gas_Extraction', # supplied from these operations
    ],
    notable_members=[
        orgs.Imperial_Oil,
        orgs.Suncor,
        orgs.Cenovus_Energy,
    ])

ss.International_Oil_Refinery_Operators = Stakeholders(
    ipcc_catpaths=[
        'Stationary_Combustion_Sources/Oil_and_Gas_Extraction', # supplied from these operations
    ])

ss.Domestic_Fuel_Vendors_Marine_Diesel = Stakeholders(
    ipcc_catpaths=[
        'Transport/Marine/Domestic_Navigation',
    ],
    notable_members=[
        orgs.Imperial_Oil,
        orgs.Suncor,
        orgs.Shell_Canada,
    ])

ss.Domestic_Fuel_Customers_Marine_Diesel = Stakeholders(
    ipcc_catpaths=[
        'Transport/Marine/Domestic_Navigation',
        'Stationary_Combustion_Sources/Oil_and_Gas_Extraction', # supplied from these operations
    ])

ss.International_Dry_Bulk_Shipping = Stakeholders(
    ipcc_catpaths=['Transport/Marine/Domestic_Navigation'],
    )

ss.International_Liquid_Bulk_Shipping = Stakeholders(
    ipcc_catpaths=['Transport/Marine/Domestic_Navigation'],
    )

ss.International_Container_Shipping = Stakeholders(
    ipcc_catpaths=['Transport/Marine/Domestic_Navigation'],
    )

orgs.Port_of_Vancouver = Org()
orgs.Port_of_Montreal = Org()
orgs.Port_of_Hamilton = Org()
orgs.Port_of_Thunder_Bay = Org()
orgs.Port_of_Halifax = Org()
orgs.Port_of_St_Johns = Org()
orgs.Port_of_Toronto = Org()
orgs.Port_of_Quebec_City = Org()

ss.Ports = Stakeholders(
    ipcc_catpaths=['Transport/Marine/Domestic_Navigation'],
    notable_members=[
        orgs.Port_of_Vancouver,
        orgs.Port_of_Montreal,
        orgs.Port_of_Hamilton,
        orgs.Port_of_Thunder_Bay,
        orgs.Port_of_Halifax,
        orgs.Port_of_St_Johns,
        orgs.Port_of_Toronto,
        orgs.Port_of_Quebec_City,
    ])

orgs.Great_Lakes_Grain = Org(url="https://www.greatlakesgrain.com/About-Us")
orgs.PnH_Crop_Inputs_and_Grain = Org(
    full_name="Parish and Heimbecker",
    url="https://parrishandheimbecker-ag.com/")
ss.Great_Lakes_Grain_Movers = Stakeholders(
    ipcc_catpaths=['Transport/Marine/Domestic_Navigation'],
    notable_members=[
        orgs.Great_Lakes_Grain,
        orgs.PnH_Crop_Inputs_and_Grain,
    ])

orgs.Nutrien = Org()
orgs.Mosaic_Company = Org()
orgs.Compass_Minerals = Org()
orgs.KpS_Potash_Canada = Org()

 # TODO: super-set of Potash and Coal
ss.Mining_Companies = Stakeholders(
    ipcc_catpaths=[
        'Forest_Land',
    ])

ss.Potash_Mining_Companies = Stakeholders(
    ipcc_catpaths=['Transport/Marine/Domestic_Navigation'],
    urls=["https://natural-resources.canada.ca/minerals-mining/mining-data-statistics-analysis/minerals-metals-facts/potash-facts",],
    notable_members=[
        orgs.Nutrien,
        orgs.Mosaic_Company,
        orgs.Compass_Minerals,
        orgs.KpS_Potash_Canada])

ss.Coal_Mining_Companies = Stakeholders(
    ipcc_catpaths=['Transport/Marine/Domestic_Navigation'],
    )

orgs.Newfoundland = Org()
orgs.New_Brunswick = Org()
orgs.Prince_Edward_Island = Org()
orgs.Nova_Scotia = Org()
orgs.Quebec = Org()
orgs.Ontario = Org()
orgs.Manitoba = Org()
orgs.Saskatchewan = Org()
orgs.Alberta = Org()
orgs.British_Columbia = Org()
orgs.Nunavut = Org()
orgs.NorthWestTerritories = Org()
orgs.Yukon = Org()

ss.Coal_Mining_Provinces = Stakeholders(
    notable_members=[
        orgs.Alberta,
        orgs.Saskatchewan,
        orgs.Nova_Scotia,
        orgs.New_Brunswick,
    ])

orgs.BC_Ferries = Org()
orgs.Canadian_Ferry_Association = Org(url="https://canadianferry.ca/")

ss.Ferry_Operators = Stakeholders(
    ipcc_catpaths=['Transport/Marine/Domestic_Navigation'],
    notable_members=[orgs.BC_Ferries, orgs.Canadian_Ferry_Association],
    )

ss.Pacific_Logging_Marine_Transport_Companies = Stakeholders(
    ipcc_catpaths=['Transport/Marine/Domestic_Navigation'],
    )

ss.Northern_Supply_Marine_Transport_Companies = Stakeholders(
    ipcc_catpaths=['Transport/Marine/Domestic_Navigation'],
    notable_members=[orgs.Fednav])

orgs.Seaspan = Org(
    url="https://www.seaspan.com/")
orgs.Davie = Org()

ss.Ship_Component_Builders = Stakeholders(
    ipcc_catpaths=['Transport/Marine/Domestic_Navigation'],
    notable_members=[
    ])

ss.Ship_Builders = Stakeholders(
    ipcc_catpaths=['Transport/Marine/Domestic_Navigation'],
    notable_members=[
        orgs.Seaspan,
        orgs.Davie,
    ]
    )

ss.Ship_Designers = Stakeholders(
    ipcc_catpaths=['Transport/Marine/Domestic_Navigation'],
    )

ss.Ship_Maintenance_Companies = Stakeholders(
    ipcc_catpaths=['Transport/Marine/Domestic_Navigation'],
    )

ss.Mariners = Stakeholders(
    ipcc_catpaths=['Transport/Marine/Domestic_Navigation'],
    )

ss.Mariner_Training_Institutions = Stakeholders(
    ipcc_catpaths=['Transport/Marine/Domestic_Navigation'],
    )

ss.Steel_Mills = Stakeholders(
    ipcc_catpaths=['Transport/Marine/Domestic_Navigation'],
    )

ss.Dredging_Companies = Stakeholders(
    ipcc_catpaths=['Transport/Marine/Domestic_Navigation'],
    )

ss.Marine_Construction_Companies = Stakeholders(
    ipcc_catpaths=['Transport/Marine/Domestic_Navigation'],
    )

orgs.International_Maritime_Organization = Org()
orgs.Green_Marine = Org()

ss.Marine_Policy_Advocacy_Groups = Stakeholders(
    ipcc_catpaths=['Transport/Marine/Domestic_Navigation'],
    notable_members=[
        orgs.International_Maritime_Organization,
        orgs.Green_Marine,
    ])

ss.Harbour_Ferry_Customers = Stakeholders()
ss.Newfoundland_Ferry_Customers = Stakeholders()
ss.Victoria_Island_Ferry_Customers = Stakeholders()

ideas.battery_water_taxis = NewcoIdea(
    who=ss.Ferry_Operators.new_org(),
    descr="Operate a battery-electric water-taxi service",
    for_whom=[ss.Harbour_Ferry_Customers],
    ipcc_catpaths=['Transport/Marine/Domestic_Navigation'],
    )

ideas.hydrofoil_ferry_service = NewcoIdea(
    who=ss.Ferry_Operators.new_org(),
    descr="Operate a hydrofoil ferry service",
    for_whom=[ss.Newfoundland_Ferry_Customers, ss.Victoria_Island_Ferry_Customers],
    ipcc_catpaths=['Transport/Marine/Domestic_Navigation'],
    )

# this may also be a good way to get between Montreal and QC?
# maybe along 401 around Toronto?
# maybe along  QC, Montreal, Toronto, Niagara, Detroit, Cleveland, Chicago?
ideas.ferry_service_ground_effect = NewcoIdea(
    who=ss.Ferry_Operators.new_org(),
    descr="Operate a wing-in-ground-effect ferry service",
    for_whom=[ss.Newfoundland_Ferry_Customers, ss.Victoria_Island_Ferry_Customers],
    ipcc_catpaths=['Transport/Marine/Domestic_Navigation'],
    )

ideas.replace_aging_ferries_with_battery_electric = RegulationIdea(
    who=orgs.Canada,
    descr="Prefer replacing aging ferries with battery-electric designs",
    ipcc_catpaths=['Transport/Marine/Domestic_Navigation'],
    for_whom=[ss.People_Desiring_Net_Zero],
    urls=["https://www.damen.com/vessels/ferries"],
    )

ideas.scrub_emissions_from_marine_exhaust = RegulationIdea(
    who=orgs.Canada,
    descr="Require GHG emission scrubbing from vessel exhaust",
    for_whom=[ss.People_Desiring_Net_Zero],
    ipcc_catpaths=['Transport/Marine/Domestic_Navigation'],
    )


ideas.autonomous_hopper_barges_for_dredging = NewcoIdea(
    who=ss.Tug_and_Tow_Companies.new_org(),
    descr="Autonomous battery-electric hopper barges for dredging operations",
    for_whom=[ss.Marine_Construction_Companies, ss.Dredging_Companies, ss.Ports],
    ipcc_catpaths=['Transport/Marine/Domestic_Navigation'],
    )

ideas.retrofit_wingsails_on_freighters = NewcoIdea(
    who=ss.Ship_Component_Builders.new_org(),
    descr="Retrofit wingsails on Great Lakes freighters",
    for_whom=ss.Great_Lakes_Dry_Bulk_Shipping_Companies,
    ipcc_catpaths=['Transport/Marine/Domestic_Navigation'],
    )

# replace the chichimaun with battery ferry (who, for whom?)
# upgrade the chichimaun to use battery? (who, for whom?)

ideas.power_ships_by_LNG_and_pyrolysis = NewcoIdea(
    who=ss.Ship_Component_Builders.new_org(),
    descr="Power zero-emissions ships with LNG via methane pyrolysis and a hydrogen fuel cell",
    for_whom=[
        ss.Ship_Designers, ss.Ship_Builders,
    ],
    ipcc_catpaths=['Transport/Marine/Domestic_Navigation'],
    )

ideas.power_ships_with_eMethanol = NewcoIdea(
    #who=ss.Ship_Component_Builders.new_org(),
    descr="Something about e-Methanol",
    #for_whom=[ ss.Ship_Designers, ss.Ship_Builders, ],
    ipcc_catpaths=['Transport/Marine/Domestic_Navigation'],
    )

# something about e-Methanol for shipping
# what's the status of e-Methanol production, can a vessel fuel
# at both ends of a journey, or carry enough fuel to return to home port?



ss.Oil_and_Gas_Extraction_Industry_Employees = Stakeholders(
    ipcc_catpaths=[
        'Stationary_Combustion_Sources/Oil_and_Gas_Extraction', # paid by these activities
    ])

ss.Domestic_Fuel_Customers_Diesel = Stakeholders(
    ipcc_catpaths=[
        'Stationary_Combustion_Sources/Oil_and_Gas_Extraction', # end-customers of these activities
    ])

ss.Domestic_Fuel_Customers_Gasoline = Stakeholders(
    ipcc_catpaths=[
        'Stationary_Combustion_Sources/Oil_and_Gas_Extraction', # end-customers of these activities
        'Transport/Road_Transportation/Light-Duty_Gasoline_Trucks',
    ])

ss.Domestic_Fuel_Customers_Heating_Oil = Stakeholders(
    ipcc_catpaths=[
        'Stationary_Combustion_Sources/Oil_and_Gas_Extraction', # end-customers of these activities
        'Stationary_Combustion_Sources/Public_Electricity_and_Heat_Production',
    ])
ss.Domestic_Fuel_Customers_Natural_Gas = Stakeholders(
    ipcc_catpaths=[
        'Stationary_Combustion_Sources/Oil_and_Gas_Extraction', # end-customers of these activities
        'Stationary_Combustion_Sources/Public_Electricity_and_Heat_Production',
    ])

ss.International_Fuel_Customers = Stakeholders(
    ipcc_catpaths=[
        'Stationary_Combustion_Sources/Oil_and_Gas_Extraction', # end-customers of these activities
    ])

ss.Domestic_Fuel_Vendors_Gasoline = Stakeholders(
    ipcc_catpaths=[
        'Transport/Road_Transportation/Light-Duty_Gasoline_Trucks',
    ],
    notable_members=[
        orgs.Imperial_Oil,
        orgs.Suncor,
        orgs.Cenovus_Energy,
        orgs.Shell_Canada,
    ])
ss.Domestic_Fuel_Vendors_Heating_Oil = Stakeholders(
    ipcc_catpaths=[
        'Stationary_Combustion_Sources/Oil_and_Gas_Extraction', # end-customers of these activities
        'Stationary_Combustion_Sources/Public_Electricity_and_Heat_Production',
    ])
ss.Domestic_Fuel_Vendors_Natural_Gas = Stakeholders(
    ipcc_catpaths=[
        'Stationary_Combustion_Sources/Oil_and_Gas_Extraction', # end-customers of these activities
        'Stationary_Combustion_Sources/Public_Electricity_and_Heat_Production',
    ])

ss.Equipment_Vendors_Gas_Turbines = Stakeholders(
    ipcc_catpaths=[
        'Stationary_Combustion_Sources/Oil_and_Gas_Extraction', # use gas turbines for extraction
        'Stationary_Combustion_Sources/Public_Electricity_and_Heat_Production',
    ])

ss.Equipment_Vendors_Steam_Boilers = Stakeholders(
    ipcc_catpaths=[
        'Stationary_Combustion_Sources/Oil_and_Gas_Extraction', # use gas turbines for extraction
        'Stationary_Combustion_Sources/Public_Electricity_and_Heat_Production',
    ])

ss.Equipment_Vendors_Process_Heaters = Stakeholders(
    ipcc_catpaths=[
        'Stationary_Combustion_Sources/Oil_and_Gas_Extraction', # use gas turbines for extraction
    ])

ss.Equipment_Vendors_Glycol_Reboilers = Stakeholders(
    ipcc_catpaths=[
        'Stationary_Combustion_Sources/Oil_and_Gas_Extraction', # use gas turbines for extraction
    ])

ideas.require_gas_turbine_exhaust_scrubbers = RegulationIdea(
    who=ss.Regulators,
    descr="Incentivize/require gas turbines to have exhaust scrubbers",
    for_whom=[ss.People_Desiring_Net_Zero],
    ipcc_catpaths=[
        'Stationary_Combustion_Sources/Oil_and_Gas_Extraction', # gas turbines used in oil sands projects
    ])

ss.Oil_and_Gas_Project_Developers = Stakeholders()

ideas.extract_oil_with_renewable_electricity = NewcoIdea(
    who=ss.Oil_and_Gas_Project_Developers.new_org(),
    descr="Design and implement an oil sands project based on renewable process energy",
    for_whom=[ss.Oil_and_Gas_Extraction_Companies],
    ipcc_catpaths=[
        'Stationary_Combustion_Sources/Oil_and_Gas_Extraction', # gas turbines used in oil sands projects
    ])

ideas.ban_oil_sands_mining = RegulationIdea(
    who=ss.Regulators,
    descr="Forbid new oil sands projects and site expansion, find new jobs for industry staff",
    for_whom=[ss.People_Desiring_Net_Zero],
    ipcc_catpaths=[
        'Stationary_Combustion_Sources/Oil_and_Gas_Extraction', # gas turbines used in oil sands projects
    ])

ideas.limit_oil_and_gas_exports = RegulationIdea(
    who=ss.Regulators,
    descr="Limit and/or discourage (e.g. tax) Canada's energy exports",
    for_whom=[ss.People_Desiring_Net_Zero],
    ipcc_catpaths=[
        'Stationary_Combustion_Sources/Oil_and_Gas_Extraction', # gas turbines used in oil sands projects
    ])

ss.Methane_Pyrolysis_Companies = Stakeholders() # is this actually a stakeholder group?
ideas.power_oil_sands_extraction_via_methane_pyrolysis = NewcoIdea(
    who=ss.Methane_Pyrolysis_Companies.new_org(),
    descr="Generate heat and electricity via methane pyrolysis and hydrogen fuel cells",
    for_whom=[ss.Oil_and_Gas_Extraction_Companies],
    ipcc_catpaths=[
        'Stationary_Combustion_Sources/Oil_and_Gas_Extraction', # gas turbines used in oil sands projects
    ])


ss.Heat_Pump_Installation_Companies = Stakeholders()

ideas.heat_pumps_for_homes = NewcoIdea(
    who=ss.Heat_Pump_Installation_Companies.new_org(),
    descr="Reduce demand for furnace oil and natural gas by deploying heat pumps for homes",
    for_whom=[
        ss.Domestic_Fuel_Customers_Heating_Oil,
        ss.Domestic_Fuel_Customers_Natural_Gas,
    ],
    ipcc_catpaths=[
        #'Stationary_Combustion_Sources/Oil_and_Gas_Extraction', # reduce some domestic demand
    ])

ideas.heat_pumps_for_commercial = NewcoIdea(
    who=ss.Heat_Pump_Installation_Companies.new_org(),
    descr="Reduce demand for heating oil and natural gas by deploying heat pump technology for condominiums and commercial buildings",
    for_whom=[
        ss.Domestic_Fuel_Customers_Heating_Oil,
        ss.Domestic_Fuel_Customers_Natural_Gas,
    ],
    ipcc_catpaths=[
        #'Stationary_Combustion_Sources/Oil_and_Gas_Extraction', # reduce some domestic demand
    ])

ss.Chinese_Car_Companies = Stakeholders()

ss.Light_Duty_Vehicle_Owners = Stakeholders(
    ipcc_catpaths=[
        'Stationary_Combustion_Sources/Oil_and_Gas_Extraction', # end-customers of these activities
        'Transport/Road_Transportation/Light-Duty_Gasoline_Trucks',
    ])

ideas.sell_chinese_EVs = NewcoIdea(
    who=ss.Chinese_Car_Companies.new_org(), # e.g. BYD Canada
    descr="Sell some quota of Chinese EVs (increasing supply of ZEVs)",
    for_whom=[
        ss.Light_Duty_Vehicle_Owners,
    ],
    ipcc_catpaths=[
        'Transport/Road_Transportation/Light-Duty_Gasoline_Trucks', # reduce some domestic demand
    ])

ideas.combo_a = Idea(
    descr="Combo A: a set of promising projects.")


# TODO: idea small autonomous RoRo to ferry trucks


## Light Duty Gasoline Trucks

ss.Light_Duty_Gasoline_Truck_Vendors = Stakeholders(
    ipcc_catpaths=[
        'Transport/Road_Transportation/Light-Duty_Gasoline_Trucks',
    ])

ss.Light_Duty_Gasoline_Truck_Mechanics = Stakeholders(
    ipcc_catpaths=[
        'Transport/Road_Transportation/Light-Duty_Gasoline_Trucks',
    ])
ss.Light_Duty_Gasoline_Truck_Insurers = Stakeholders(
    ipcc_catpaths=[
        'Transport/Road_Transportation/Light-Duty_Gasoline_Trucks',
    ])

ss.Home_Buyers = Stakeholders()
ss.Individuals_and_Families = Stakeholders(
    ipcc_catpaths=[
        'Transport/Road_Transportation/Light-Duty_Gasoline_Trucks',
    ])

ss.Light_Duty_Truck_Fleet_Operators = Stakeholders(
    ipcc_catpaths=[
        'Transport/Road_Transportation/Light-Duty_Gasoline_Trucks',
    ])

ss.Parking_Lot_Owners = Stakeholders(
    ipcc_catpaths=[
        'Transport/Road_Transportation/Light-Duty_Gasoline_Trucks',
    ])

ideas.embed_wireless_chargers_in_roads = NewcoIdea(
    descr="Embed wireless chargers in roads",
    urls=["https://www.purdue.edu/newsroom/2025/Q4/first-highway-segment-in-u-s-wirelessly-charges-electric-heavy-duty-truck-while-driving/"],
    ipcc_catpaths=[
        'Transport/Road_Transportation/Light-Duty_Gasoline_Trucks',
        # TODO: heavy vehicles, diesel, trucks, all road vehicles
    ])

ideas.incentivize_EVs_indefinitely = RegulationIdea(
    who=orgs.Canada,
    descr="Subsidize EV and EV-charger purchases indefinitely with a gasoline tax",
    for_whom=[ss.People_Desiring_Net_Zero],
    ipcc_catpaths=[
        'Transport/Road_Transportation/Light-Duty_Gasoline_Trucks',
    ])

ideas.force_government_fleet_to_go_green = RegulationIdea(
    who=orgs.Canada,
    descr="Force civilian federal, provincial, and municipality-owned fleets to transition almost completely to EVs",
    for_whom=[ss.People_Desiring_Net_Zero],
    ipcc_catpaths=[
        'Transport/Road_Transportation/Light-Duty_Gasoline_Trucks',
    ])

ideas.force_military_fleet_to_go_green = RegulationIdea(
    who=orgs.Canada,
    descr="Force military-owned fleets to transition to EVs",
    for_whom=[ss.People_Desiring_Net_Zero],
    ipcc_catpaths=[
        'Transport/Road_Transportation/Light-Duty_Gasoline_Trucks',
    ])


ss.Timber_Harvesting_Companies = Stakeholders(
    ipcc_catpaths=[
        'Forest_Land',
    ])
ss.Commercial_Real_Estate_Developers = Stakeholders(
    ipcc_catpaths=[
        'Forest_Land',
    ])
ss.Construction_Companies = Stakeholders(
    ipcc_catpaths=[
        'Forest_Land', # they are a major user of the timber
    ])

ss.Farmers = Stakeholders(
    ipcc_catpaths=[
        'Forest_Land',
        'Cropland',
        #'Stationary_Combustion_Sources/Public_Electricity_and_Heat_Production',
    ])
"""
<p>To-do: What are this group's private problems relating to public electricity and heat?<p>
<ul>
    <li>They buy electricity, much like any other commercial customer, mostly for constant energy loads
        <ul>
            <li>Running a co-located home</li>
            <li>Dairy: cooling milk, vacuum lines, water heating, ventilation</li>
            <li>Greenhouses: lighting, ventilation, robotics</li>
            <li>Poultry: lighting, ventilation, robotics</li>
            <li>Pork: lighting, ventilation, robotics</li>
            <li>Field: aeration, irrigation, robotics</li>
        </ul>
    </li>
    <li>They are able to implement behind-the-meter wind and/or solar because they tend to own lots of land, and simple buildings with big roofs.
        They can use photovoltaic power opportunistically with or without battery buffering, but loads tend to be constant rather than aligned to solar energy.</li>
    <li>Grow crops. Don't like things that shade fields, or remove water from the soil.</li>
</ul>
"""


ideas.vertically_integrated_mass_timber_construction_co = NewcoIdea(
    descr="Vertically-integrated mass-timber construction company",
    for_whom=[ss.Commercial_Real_Estate_Developers],
    ipcc_catpaths=[
        'Forest_Land',
        'Harvested_Wood_Products',
        'Stationary_Combustion_Sources/Construction',
    ])

ideas.enhanced_rock_weathering_for_managed_forests = NewcoIdea(
    descr="Enhanced rock weathering product for managed forests, selling carbon credits",
    for_whom=[ss.Timber_Harvesting_Companies], # sell carbon credits?
    ipcc_catpaths=[
        'Forest_Land',
        'Liming,_Urea_Application_and_Other_Carbon-Containing_Fertilizers',
    ])

ideas.something_to_suppress_insect_infestations = NewcoIdea(
    descr="Something to suppress insect infestations",
    for_whom=[ss.Timber_Harvesting_Companies], # sell carbon credits?
    ipcc_catpaths=[
        'Forest_Land',
    ])

ideas.something_to_suppress_forest_fires = NewcoIdea(
    descr="Something to suppress forest fires",
    for_whom=[ss.Timber_Harvesting_Companies], # sell carbon credits?
    ipcc_catpaths=[
        'Forest_Land',
    ])

ss.Building_Owners = Stakeholders()

# geo-engineering TODO: how much heat could be vented this way? who could pay for it? who would benefit and how? Could it be built in Southern USA or Mexico?
# "www.MEER.org",

ideas.deploy_sky_windows = NewcoIdea(
    descr='Cool buildings with so-called "sky windows" - heat bypasses greenhouse gases',
    for_whom=[ss.Building_Owners],
    urls=["https://www.skycoolsystems.com/technology/",
          "https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/adsu.202400948", # geo-engineering
         ],
    ipcc_catpaths=[
        'Stationary_Combustion_Sources/Public_Electricity_and_Heat_Production',
    ])

ideas.deploy_highly_reflective_paint_on_flat_roofs = NewcoIdea(
    descr='Roof buildings with highly-reflective ultra-white paint',
    for_whom=[ss.Building_Owners],
    urls=[],
    ipcc_catpaths=[
        'Stationary_Combustion_Sources/Public_Electricity_and_Heat_Production',
    ])

ss.Public_Electricity_Customers = Stakeholders(
    ipcc_catpaths=[
        'Stationary_Combustion_Sources/Oil_and_Gas_Extraction', # end-customers of these activities
        'Stationary_Combustion_Sources/Public_Electricity_and_Heat_Production',
    ])

ss.Public_Heat_Customers = Stakeholders(
    ipcc_catpaths=[
        'Stationary_Combustion_Sources/Oil_and_Gas_Extraction', # end-customers of these activities
        'Stationary_Combustion_Sources/Public_Electricity_and_Heat_Production',
    ])

ideas.deploy_wind_farms_in_cropland = NewcoIdea(
    descr='Deploy wind farms in cropland',
    for_whom=[ss.Public_Electricity_Customers],
    urls=[],
    ipcc_catpaths=[
        'Stationary_Combustion_Sources/Public_Electricity_and_Heat_Production',
    ])

ideas.deploy_wind_farms_in_forests = NewcoIdea(
    descr='Deploy wind farms in managed forests',
    for_whom=[ss.Public_Electricity_Customers],
    urls=[],
    ipcc_catpaths=[
        'Stationary_Combustion_Sources/Public_Electricity_and_Heat_Production',
    ])

ideas.deploy_wind_farms_on_waterways = NewcoIdea(
    descr='Deploy wind farms in waterways',
    for_whom=[ss.Public_Electricity_Customers],
    urls=[],
    ipcc_catpaths=[
        'Stationary_Combustion_Sources/Public_Electricity_and_Heat_Production',
    ])


ideas.convert_forest_to_solar_farms = NewcoIdea(
    descr='Convert managed forest to solar farms',
    for_whom=[ss.Public_Electricity_Customers],
    urls=[],
    ipcc_catpaths=[
        'Stationary_Combustion_Sources/Public_Electricity_and_Heat_Production',
    ])

ideas.convert_cropland_to_solar_farms = NewcoIdea(
    descr='Convert cropland to solar farms',
    for_whom=[ss.Public_Electricity_Customers],
    urls=[],
    ipcc_catpaths=[
        'Stationary_Combustion_Sources/Public_Electricity_and_Heat_Production',
    ])

ideas.deploy_solar_farms_on_water = NewcoIdea(
    descr='Convert water surfaces to solar farms',
    for_whom=[ss.Public_Electricity_Customers],
    urls=[],
    ipcc_catpaths=[
        'Stationary_Combustion_Sources/Public_Electricity_and_Heat_Production',
    ])

ideas.deploy_tidal_turbines = NewcoIdea(
    descr='Convert tidal flows to electricity',
    for_whom=[ss.Public_Electricity_Customers],
    urls=[],
    ipcc_catpaths=[
        'Stationary_Combustion_Sources/Public_Electricity_and_Heat_Production',
    ])


ss.Public_Electricity_Utilities = Stakeholders(
    ipcc_catpaths=[
        'Stationary_Combustion_Sources/Oil_and_Gas_Extraction', # end-customers of these activities
        'Stationary_Combustion_Sources/Public_Electricity_and_Heat_Production',
    ])

ss.Public_Heat_Utilities = Stakeholders(
    ipcc_catpaths=[
        'Stationary_Combustion_Sources/Oil_and_Gas_Extraction', # end-customers of these activities
        'Stationary_Combustion_Sources/Public_Electricity_and_Heat_Production',
    ])

ss.Public_Electricity_Generators_Photovoltaic = Stakeholders(
    ipcc_catpaths=[
        'Stationary_Combustion_Sources/Public_Electricity_and_Heat_Production',
    ])

ss.Public_Electricity_Generators_Wind = Stakeholders(
    ipcc_catpaths=[
        'Stationary_Combustion_Sources/Public_Electricity_and_Heat_Production',
    ])

ss.Wind_Turbine_Equipment_Vendors = Stakeholders(
    ipcc_catpaths=[
        'Stationary_Combustion_Sources/Public_Electricity_and_Heat_Production',
    ])

ss.Photovoltaic_Equipment_Vendors = Stakeholders(
    ipcc_catpaths=[
        'Stationary_Combustion_Sources/Public_Electricity_and_Heat_Production',
    ])

ss.CCS_Equipment_Vendors = Stakeholders(
    ipcc_catpaths=[
        'Stationary_Combustion_Sources/Public_Electricity_and_Heat_Production',
    ])


# Tidal power:
# https://www.orbitalmarine.com/
# They have installations planned / installed in Nova Scotia already.
# Heard through email from Guillermo around Jan 11, 2026



# Heat buffering better than clay
# https://interestingengineering.com/innovation/thermal-battery-for-buildings

# AtmosZero heat pump steam boiler
# https://interestingengineering.com/energy/atmoszero-electrifies-centuries-old-steam-boiler

# High-power THz-range microwave heating for plasma ... helpful for Methane Pyrolysis?
# https://interestingengineering.com/energy/uk-firm-microwave-fusion-energy

# https://interestingengineering.com/energy/worlds-first-hydrogen-pilot
# https://www.globenewswire.com/news-release/2026/02/03/3230886/0/en/Vema-Hydrogen-Drills-Pilot-Wells-in-Quebec-for-World-s-First-Engineered-Mineral-Hydrogen-Test-Project.html


# https://interestingengineering.com/energy/finland-sand-battery-heat-transfer
