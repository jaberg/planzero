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

class Stakeholders(object):

    newco_counter = 1

    def __init__(self, *,
                 name=None,
                 descr=None,
                 notable_members=None,
                 full_name=None,
                 ipcc_catpaths=(),
                 customers=(),
                 suppliers=(),
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

    @property
    def full_name(self):
        return self._full_name or self.name.replace('_', ' ')

    def on_assign(self, name):
        if self.name is None:
            self.name = name
        else:
            assert self.name == name

    def new_org_name(self):
        name = f'{self.name}_(NewCo #{Stakeholders.newco_counter})'
        assert name not in orgs._orgs
        Stakeholders.newco_counter += 1
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


class Org(object):
    def __init__(self, *, name=None, full_name=None, ipcc_catpaths=(), org_type=None, descr=None):
        self.name = name
        self._full_name = full_name
        self.ipcc_catpaths = set(ipcc_catpaths )
        self.descr = descr
        for catpath in self.ipcc_catpaths:
            assert catpath in ipcc_canada.catpaths
        self.org_type = org_type

    @property
    def full_name(self):
        return self._full_name or self.name.replace('_', ' ')

    def on_assign(self, name):
        if self.name is None:
            self.name = name
        else:
            assert self.name == name


orgs = Organizations()
orgs.gov_of_canada = Org(
    full_name="Government of Canada",
    ipcc_catpaths=ipcc_canada.catpaths
    )

orgs.Beyond_Meat_Inc = Org()
orgs.Impossible_Foods = Org()

orgs.FSM_Firmenich = Org(descr="Maker of Bovaer")


ss = Stakeholder_Sets()

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
    )
ss.Butchers_Meat_Packers = Stakeholders(
    suppliers=[ss.Beef_Farmers],
    )
ss.Dairies = Stakeholders(
    suppliers=[ss.Dairy_Farmers],
    )
ss.Beef_Consumers = Stakeholders(
    suppliers=[ss.Butchers_Meat_Packers],
    )
ss.Milk_Consumers = Stakeholders(
    suppliers=[ss.Dairies],
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



class DieselDryBulkFreighters(AssetClass):
    pass


ac.Diesel_Dry_Bulk_Freighters = DieselDryBulkFreighters()


orgs.CanadaSteamshipLines = Org()
orgs.AlgomaCentral = Org()


ss.GreatLakesDryBulkShippingCompanies = Stakeholders(
    notable_members=[
        orgs.CanadaSteamshipLines,
        orgs.AlgomaCentral])


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
    def __init__(self, *, who=None, descr:str=None, for_whom=(), ipcc_catpaths=(), urls=()):
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
        self._full_name = None

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

ideas.national_bovaer_mandate = RegulationIdea(
    who=orgs.gov_of_canada,
    for_whom=ss.People_Desiring_Net_Zero,
    descr="Force cattle farmers to administer Bovaer",
    ipcc_catpaths=['Enteric_Fermentation'],
    )

ideas.credit_bovaer = RegulationIdea(
    who=orgs.gov_of_canada,
    for_whom=[ss.Beef_Farmers, ss.Dairy_Farmers],
    descr="Recognize Bovaer usage with carbon credits",
    ipcc_catpaths=['Enteric_Fermentation'],
    )

ss.Organizations_discouraging_beef_and_milk_consumption = Stakeholders(
    descr="Organizations discouraging people from consuming beef and milk",
    ipcc_catpaths=['Enteric_Fermentation'],
    )

ideas.education_to_discourage_beef_and_milk = EducationIdea(
    who=ss.Organizations_discouraging_beef_and_milk_consumption.new_org(org_type='charity'),
    for_whom=ss.People_Desiring_Net_Zero,
    descr="Public education discouraging people from consuming beef and milk",
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
        print(name, idea, catpath, idea.ipcc_catpaths)
        if catpath in idea.ipcc_catpaths:
            yield idea
