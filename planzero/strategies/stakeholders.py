# sets are generally partially overlapping, but sets can be subsets, disjoint
from .. import ipcc_canada


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


from .ideas import *
