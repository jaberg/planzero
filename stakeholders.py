# sets are generally partially overlapping, but sets can be subsets, disjoint
import ipcc_canada


class Stakeholder_Sets(object):
    """A set of stakeholders is a set of explicit or implicit individuals and/or
    organizations with common issues, properties
    """
    def __init__(self):
        self._stakeholder_sets = {}

    def __setattr__(self, name, stakeholders):
        if name.startswith('_'):
            self.__dict__[name] = stakeholders
            return

        if isinstance(stakeholders, Stakeholders) and name not in self._stakeholder_sets:
            self._stakeholder_sets[name] = stakeholders
            stakeholders.on_assign(name)
        else:
            raise NotImplementedError(org)

    def __getattr__(self, name):
        try:
            return self._stakeholder_sets[name]
        except KeyError:
            raise AttributeError(name)

    def items(self):
        for item in self._stakeholder_sets.items():
            yield item

    # may be in product competition with another set of stakeholders
    # may be in land use competition with another set of stakeholders
    # TODO: consider also a type system of "Products" which
    #       have a set relationship: some stakeholder groups are the unique
    #       producers of certain products "e.g. beef"
    #       and simultaneously the non-unique producers of other products e.g. "food"
    # The purpose would be to
    # * enable graph algorithms for managing stakeholder maps
    # * align the types in this sytem with whatever StatsCan tracks in terms of e.g.
    #   amounts of things being produced, bought, sold, etc.
    # * detect competitive pressures in the building of stakeholder maps (e.g.
    #   if we reduce the number of X, it will increase the numbers of competing
    #   things in those product areas)

class Stakeholders(object):

    def __init__(self, *,
                 name=None,
                 descr=None,
                 notable_members=None,
                 full_name=None,
                 #unique_product=None, # a product produced only by members of this group
                 ipcc_catpaths=(),
                 customers=(),
                 suppliers=(),
                 urls=(),
                ):
        self.name = name
        self._full_name = full_name
        self.descr = descr
        self.notable_members = notable_members or []
        self.ipcc_catpaths =  set(ipcc_catpaths)
        for catpath in self.ipcc_catpaths:
            assert catpath in ipcc_canada.catpaths, catpath
        self.customers = customers
        self.suppliers = suppliers
        self.urls = urls
        self.newco_counter = 1

    @property
    def full_name(self):
        return self._full_name or self.name.replace('_', ' ')

    def on_assign(self, name):
        if self.name is None:
            self.name = name
        else:
            assert self.name == name

    def new_org_name(self):
        name = f'{self.name}_NewCo'
        while name in orgs._orgs:
            self.newco_counter += 1
            name = f'{self.name}_NewCo#{self.newco_counter}'
        return name

    def new_org(self, *, name=None, org=None, **kwargs):
        if kwargs:
            assert org is None
            org = Org(name=name or self.new_org_name(), **kwargs)
            rval = orgs.new_org(org)
        elif name is None and org is None:
            rval = orgs.new_org(Org(name=self.new_org_name()))
        elif name is None and org is not None:
            rval = orgs.new_org(org)
        elif name is not None and org is None:
            rval = orgs.new_org(Org(name=name))
        else:
            assert name == org.name
            rval = orgs.new_org(org)
        assert rval.name
        return rval


class Organizations(object):
    def __init__(self):
        self._orgs = {}

    def __setattr__(self, name, org):
        if name.startswith('_'):
            self.__dict__[name] = org
            return

        if isinstance(org, Org) and name not in self._orgs:
            self._orgs[name] = org
            org.on_assign(name)
        else:
            raise NotImplementedError(org)

    def __getattr__(self, name):
        try:
            return self._orgs[name]
        except KeyError:
            raise AttributeError(name)

    def new_org(self, org):
        assert isinstance(org.name, str) and org.name
        self._orgs[org.name] = org
        return org

    def items(self):
        for item in self._orgs.items():
            yield item


# class Organization
class Org(object):
    def __init__(self, *, name=None, full_name=None, ipcc_catpaths=(), org_type=None, descr=None, url=None):
        self.name = name
        self._full_name = full_name
        self.ipcc_catpaths = set(ipcc_catpaths )
        self.descr = descr
        for catpath in self.ipcc_catpaths:
            assert catpath in ipcc_canada.catpaths
        self.org_type = org_type
        self.url = url

    @property
    def full_name(self):
        return self._full_name or self.name.replace('_', ' ')

    def on_assign(self, name):
        if self.name is None:
            self.name = name
        else:
            assert self.name == name


orgs = Organizations()
orgs.Canada = Org(
    full_name="Government of Canada",
    ipcc_catpaths=ipcc_canada.catpaths
    )

orgs.Beyond_Meat_Inc = Org()
orgs.Impossible_Foods = Org()

orgs.FSM_Firmenich = Org(descr="Maker of Bovaer")


ss = Stakeholder_Sets()
ss.Regulators = Stakeholders(
    ipcc_catpaths=ipcc_canada.catpaths,
    notable_members=[
        orgs.Canada,
    ])

ss.Makers_of_Beef_Substitutes = Stakeholders(
    ipcc_catpaths=['Enteric_Fermentation'],
    notable_members=[
        orgs.Beyond_Meat_Inc,
        orgs.Impossible_Foods])

ss.Makers_of_Milk_Substitutes = Stakeholders(
    ipcc_catpaths=['Enteric_Fermentation'],
    )

ss.Beef_Farmers = Stakeholders(
    descr="Collectively own Canada's beef herd",
    ipcc_catpaths=['Enteric_Fermentation'],
    )
ss.Dairy_Farmers = Stakeholders(
    descr="Collectively own Canada's dairy herd",
    ipcc_catpaths=['Enteric_Fermentation'],
    )
ss.Barn_Equipment_Makers = Stakeholders(
    ipcc_catpaths=['Enteric_Fermentation'],
    )
ss.Feed_Additive_Companies = Stakeholders(
    ipcc_catpaths=['Enteric_Fermentation'],
    notable_members=[orgs.FSM_Firmenich],
    )

ss.Feed_Growing_Farmers = Stakeholders(
    customers=[ss.Beef_Farmers, ss.Dairy_Farmers],
    ipcc_catpaths=['Enteric_Fermentation'],
    )
ss.Butchers_Meat_Packers = Stakeholders(
    suppliers=[ss.Beef_Farmers],
    ipcc_catpaths=['Enteric_Fermentation'],
    )
ss.Dairies = Stakeholders(
    suppliers=[ss.Dairy_Farmers],
    ipcc_catpaths=['Enteric_Fermentation'],
    )
ss.Beef_Consumers = Stakeholders(
    suppliers=[ss.Butchers_Meat_Packers],
    ipcc_catpaths=['Enteric_Fermentation'],
    )
ss.Milk_Consumers = Stakeholders(
    suppliers=[ss.Dairies],
    ipcc_catpaths=['Enteric_Fermentation'],
    )


class AssetClasses(object):
    def __init__(self):
        self._acs = {}

    def __setattr__(self, name, ac):
        if name.startswith('_'):
            self.__dict__[name] = ac
            return

        if isinstance(ac, AssetClass) and name not in self._acs:
            self._acs[name] = ac
            ac.on_assign(name)
        else:
            raise NotImplementedError(ac)

    def __getattr__(self, name):
        try:
            return self._acs[name]
        except KeyError:
            raise AttributeError(name)


ac = AssetClasses()


class AssetClass(object):
    """A set of similar assets, owned collectively by a particular set of
    Stakeholders.
    """
    def __init__(self, *, name=None, descr=None, owners=None, ipcc_catpath=None):
        self.name = name
        self.descr = descr
        self.owners = owners

        # to which ipcc sector category does this contribute emissions?
        if ipcc_catpath is None or ipcc_catpath in ipcc_canada.catpaths:
            self.ipcc_catpath = ipcc_catpath
        else:
            raise NotImplementedError(ipcc_catpath)

    @property
    def owners(self):
        return self._owners

    @owners.setter
    def owners(self, owners):
        assert owners is None or isinstance(owners, (Stakeholders, Org))
        self._owners = owners

    def on_assign(self, name):
        if self.name is None:
            self.name = name
        else:
            assert self.name == name

    @property
    def annual_emitted_CH4_name(self):
        return f'annual_emitted_CH4_{self.name}'

    @property
    def annual_emitted_CO2_name(self):
        return f'annual_emitted_CO2_{self.name}'


class BeefCattle(AssetClass):
    def __init__(self):
        super().__init__(owners=ss.Beef_Farmers)

ac.Beef_Cattle = BeefCattle()


class DairyCattle(AssetClass):
    def __init__(self):
        super().__init__(owners=ss.Dairy_Farmers)

ac.Dairy_Cattle = DairyCattle()



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

ideas.national_bovaer_mandate = RegulationIdea(
    who=orgs.Canada,
    for_whom=ss.People_Desiring_Net_Zero,
    descr="Compel cattle farmers to administer Bovaer",
    ipcc_catpaths=['Enteric_Fermentation'],
    )

ideas.credit_bovaer = RegulationIdea(
    who=orgs.Canada,
    for_whom=[ss.Beef_Farmers, ss.Dairy_Farmers],
    descr="Recognize Bovaer usage with carbon credits",
    ipcc_catpaths=['Enteric_Fermentation'],
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

ss.Oil_and_Gas_Extraction_Companies = Stakeholders(
    ipcc_catpaths=['Stationary_Combustion_Sources/Oil_and_Gas_Extraction'],
    notable_members=[
        orgs.Imperial_Oil,
        orgs.Suncor,
        orgs.Cenovus_Energy,
        orgs.Canadian_Natural_Resources_Ltd_CNRL,
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

ss.Marine_Diesel_Vendors = Stakeholders(
    ipcc_catpaths=[
        'Transport/Marine/Domestic_Navigation',
    ],
    notable_members=[
        orgs.Imperial_Oil,
        orgs.Suncor,
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


ideas.battery_tugs_w_aux_solar_barges = NewcoIdea(
    full_name="Battery-powered tugboats with auxiliary solar barges",
    who=ss.Tug_and_Tow_Companies.new_org(),
    descr="Solar, wind, and/or battery-electric tugboats for large barges",
    for_whom=[ss.Great_Lakes_Dry_Bulk_Shipping_Companies,
              ss.Great_Lakes_Liquid_Bulk_Shipping_Companies,
              ss.Pacific_Logging_Marine_Transport_Companies,
             ],
    ipcc_catpaths=['Transport/Marine/Domestic_Navigation'],
    urls=["https://www.seaspan.com/stories/log-barging-101/"],
    )
# autonomous version of ^^ doesn't make very much financial difference, fuel savings is much larger


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


class DieselDryBulkFreighters(AssetClass):
    pass


class PacificLogBarges(AssetClass):
    # Gemini estimates there about 10 of these in operation
    # made by Seaspan and Rivtow
    # Gemini states they are moved by tug, they are not self-propelled
    # Gemini states they are moved by tugs with 3000-6000 HP
    # they currently move about 10-15k tons / ship
    # they make about 50 trips a year
    # they move about 7_000_000 tons annually
    # the trips average 400 km
    # the freight task is 7_000_000 x 400 = 2.8 billion tonne-km
    # the associated emissions are 35-40 tonnes CO2e, including empty back-haul
    # backhaul travel is typically done faster at a lower-efficiency speed, uses 60-70% fuel / km
    # burn marine diesel oil or marine gas oil, not bunker
    # future fleet burns LNG, and may be hybrid-electric

    def __init__(self):
        super().__init__(owners=ss.Pacific_Logging_Marine_Transport_Companies)
        self.n_barges = 10

ac.Pacific_Log_Barges = PacificLogBarges()

class PassiveBarge3000DWT(AssetClass):
    pass

ac.Diesel_Dry_Bulk_Freighters = DieselDryBulkFreighters()
ac.PassiveBarge3000DWT = PassiveBarge3000DWT()


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
    ])

ss.Domestic_Fuel_Customers_Heating_Oil = Stakeholders(
    ipcc_catpaths=[
        'Stationary_Combustion_Sources/Oil_and_Gas_Extraction', # end-customers of these activities
    ])
ss.Domestic_Fuel_Customers_Natural_Gas = Stakeholders(
    ipcc_catpaths=[
        'Stationary_Combustion_Sources/Oil_and_Gas_Extraction', # end-customers of these activities
    ])

ss.International_Fuel_Customers = Stakeholders(
    ipcc_catpaths=[
        'Stationary_Combustion_Sources/Oil_and_Gas_Extraction', # end-customers of these activities
    ])

ss.Equipment_Vendors_Gas_Turbines = Stakeholders(
    ipcc_catpaths=[
        'Stationary_Combustion_Sources/Oil_and_Gas_Extraction', # use gas turbines for extraction
    ])

ss.Equipment_Vendors_Steam_Boilers = Stakeholders(
    ipcc_catpaths=[
        'Stationary_Combustion_Sources/Oil_and_Gas_Extraction', # use gas turbines for extraction
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

ideas.sell_chinese_EVs = NewcoIdea(
    who=ss.Chinese_Car_Companies.new_org(), # e.g. BYD Canada
    descr="Reduce demand for gasoline by selling Chinese EVs (increasing supply of ZEVs)",
    for_whom=[
        ss.Domestic_Fuel_Customers_Gasoline,
    ],
    ipcc_catpaths=[
        #'Stationary_Combustion_Sources/Oil_and_Gas_Extraction', # reduce some domestic demand
    ])


# TODO: idea small autonomous RoRo to ferry trucks
