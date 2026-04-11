from pydantic import Field, computed_field

from .ureg import u
from .enums import IPCC_Sector, StandardScenarios, PT
from .base import DynamicElement
from . import sts
from . import objtensor

barriers = {} # classname -> Singleton instance


class Barrier(DynamicElement):

    @classmethod
    def __init_subclass__(cls):
        super().__init_subclass__()
        barriers[cls.__name__] = cls()


class BovaerAdoptionLimit(Barrier):
    """Assume that no more than X% of
    farmers will switch to administering Bovaer
    in any given year, but that adoption
    can go all the way to 100%.
    """
    
    @computed_field
    def max_increase_rate(self) -> object:
        return 5.0 * u.percent / u.year

    @computed_field
    def short_descr(self) -> str:
        return f"Bovaer will only be adopted by, at most, {self.max_increase_rate} of cattle operations"

    @computed_field
    def ipcc_sectors(self) -> list[object]:
        return [IPCC_Sector.Enteric_Fermentation]

    @computed_field
    def scenarios(self) -> list[object]:
        return [StandardScenarios.Scaling]

    @computed_field
    def research(self) -> dict[str, str]:
        return {}

    def on_add_project(self, state):
        with state.defining(self) as ctx:
            ctx.max_fraction_of_cattle_on_bovaer = sts.SparseTimeSeries(
                default_value=0 * u.dimensionless)
            ctx.too_many_cattle_on_bovaer = sts.SparseTimeSeries(
                default_value=0 * u.dimensionless)
        # use syntax ctx.too_many_cattle_on_bovaer = Monitor(sts.SparseTimeSeries(...))
        #state.register_monitor('too_many_cattle_on_bovaer')
        return 2025 * u.years

    def step(self, state, current):
        current.max_fraction_of_cattle_on_bovaer = min(
            (state.latest.bovine_population_fraction_on_bovaer
             + self.max_increase_rate * 1.0 * u.year),
            1.0 * u.dimensionless)
        return state.t_now + 1 * u.years


class BovaerPrice(Barrier):

    @computed_field
    def short_descr(self) -> str:
        return f"Bovaer will always cost {self.bovaer_price}"

    @computed_field
    def ipcc_sectors(self) -> list[object]:
        return [IPCC_Sector.Enteric_Fermentation]

    @computed_field
    def scenarios(self) -> list[object]:
        return [StandardScenarios.Scaling]

    @computed_field
    def research(self) -> dict[str, str]:
        return {}

    @computed_field
    def bovaer_price(self) -> object:
        # XXX
        # this is the wrong unit for price, the price should be different for
        # calves because they are smaller, shouldn't it?
        # Can you even give it to calves? Starting at what age?
        return 150 * u.CAD / u.cattle / u.year

    def on_add_project(self, state):
        with state.defining(self) as ctx:
            ctx.bovaer_price = sts.SparseTimeSeries(
                default_value=self.bovaer_price)

# TODO: there will be a cost for monitoring
# https://www.mn.uio.no/geo/english/about/news-and-events/news/2025/combined-drone-satelite-data-and-ground-based-measurements-methane-emissions.html
# It can apparently be done pretty well with drones
# There are approximately 70_000 cattle operations in Canada
# Monitoring might cost 10-20 million / year?

# TODO: Bovaer production (and distribution?) is energy-intensive, has GHG footprint

# TODO: Market mechanism in simulation to determine prices based on supply and demand

# TODO: Output-based carbon pricing model for agriculture


from .eccc_nir_annex3p4 import table_A3p4_11
from .sc_3210013001 import (
    FarmType, Livestock, Livestock_nonsums, SurveyDate,
    number_of_cattle_by_class_and_farm_type)

class BovinePopulation(Barrier):
    """The barrier to net-zero represented by this class is that Canada
    has a lot of cattle. They are modelled as producing methane (although
    less-so if they are fed Bovaer) and they also produce milk and beef.
    The production of leather is not modelled.

    In future, the evolution of the cattle population may be modelled
    (and projected) but for now it is projected as remaining constant.
    """

    @computed_field
    def short_descr(self) -> str:
        return f"Model a constant cattle population"

    @computed_field
    def ipcc_sectors(self) -> list[object]:
        return [IPCC_Sector.Enteric_Fermentation]

    @computed_field
    def scenarios(self) -> list[object]:
        return [StandardScenarios.Scaling]

    @computed_field
    def research(self) -> dict[str, str]:
        return {}

    # TODO: research URLs
    # https://www.ucdavis.edu/food/news/making-cattle-more-sustainable
    # 70% of emissions remain, according to https://www.helsinki.fi/en/news/climate-change/new-feed-additive-can-significantly-reduce-methane-emissions-generated-ruminants-already-dairy-farm
    # https://www.dsm-firmenich.com/anh/news/press-releases/2024/2024-01-31-canada-approves-bovaer-as-first-feed-ingredient-to-reduce-methane-emissions-from-cattle.html
    bovaer_methane_reduction_fraction:float = .625
    # TODO: dairy cattle average is .7
    # TODO: beef cattle reduction is greater, multiplier should be .55

    @computed_field
    def bovaer_actual_vs_nominal(self) -> float:
        """What fraction of cattle nominally on bovaer actually eat it properly?"""
        return .95

    @computed_field
    def bovaer_methane_reduction(self) -> dict[object, object]:
        guess = .4
        return {
            Livestock.Bulls: guess,
            Livestock.DairyCows: 0.30, # https://www.dsm-firmenich.com/anh/products-and-services/products/methane-inhibitors/bovaer.html
            Livestock.BeefCows: .45, # https://www.dsm-firmenich.com/anh/news/press-releases/2024/2024-01-31-canada-approves-bovaer-as-first-feed-ingredient-to-reduce-methane-emissions-from-cattle.html
            Livestock.DairyHeifers: guess,
            Livestock.BeefHeifers: guess,
            Livestock.SlaughterHeifers: guess,
            Livestock.Steers: guess,
            Livestock.Calves: guess,
        }

    @computed_field
    def bovaer_cost(self) -> dict[object, object]:
        # https://www.producer.com/livestock/new-methane-feed-additive-pleases-producers
        return {
            Livestock.Bulls: .50 * u.CAD / u.day / u.cattle,
            Livestock.DairyCows: 0.50 * u.CAD / u.day / u.cattle,
            Livestock.BeefCows: 0.50 * u.CAD / u.day / u.cattle,
            Livestock.DairyHeifers: .40 * u.CAD / u.day / u.cattle,
            Livestock.BeefHeifers: .40 * u.CAD / u.day / u.cattle,
            Livestock.SlaughterHeifers: .40 * u.CAD / u.day / u.cattle,
            Livestock.Steers: .40 * u.CAD / u.day / u.cattle,
            Livestock.Calves: .30 * u.CAD / u.day / u.cattle,
        }

    def on_add_project(self, state):
        stash = state.stash(self)

        with state.requiring_current(self) as ctx:
            # TODO: for each type of cattle, for each province
            ctx.bovine_population_fraction_on_bovaer = sts.SparseTimeSeries(
                default_value=0 * u.dimensionless)
        with state.defining(self) as ctx:
            cattle_pt, cattle_ca = number_of_cattle_by_class_and_farm_type()
            emfac = table_A3p4_11()
            stash.headcounts = []
            stash.on_bovaer = []
            stash.livestock_type = []
            bovine_methane = None
            for livestock in Livestock_nonsums:
                jan1 = cattle_pt[SurveyDate.Jan1, livestock, FarmType.AllCattle]
                jul1 = cattle_pt[SurveyDate.Jul1, livestock, FarmType.AllCattle]
                for pt in PT:
                    jan1_pt = jan1[pt]
                    jul1_pt = jul1[pt]
                    if isinstance(jul1_pt, sts.STS):
                        assert isinstance(jan1_pt, sts.STS)
                        hc = sts.interleave([jan1_pt, jul1_pt.delay(.5 * u.years)])
                        hc.values[0] = 0 # assume 0 instead of undefined
                        state.declare_sts(self, hc, need_current=True)
                        stash.headcounts.append(hc)

                        on_bovaer = sts.SparseTimeSeries(default_value=0 * u.cattle)
                        stash.on_bovaer.append(on_bovaer)
                        state.declare_sts(self, on_bovaer, need_current=True)

                        stash.livestock_type.append(livestock)

                        if bovine_methane is None:
                            bovine_methane = hc * emfac[livestock]
                        else:
                            bovine_methane += hc * emfac[livestock]
                    else:
                        assert not isinstance(jan1_pt, sts.STS)

            ctx.bovine_methane_rate = bovine_methane
            ctx.bovaer_cost = sts.SparseTimeSeries(default_value=0 * u.CAD / u.year, t_unit=u.year)
            ctx.bovaer_cost_annual = sts.SparseTimeSeries(default_value=0 * u.CAD, t_unit=u.year)

            # this ends with the 2025-2026 year being associated with 2025
            correct = bovine_methane.bin_integrals(
                bin_boundaries=[tt * u.year for tt in range(1990, 2026 + 1)])
            assert correct.times[-1] == 2025, correct

            # XXX
            # however the atmospheric chemistry model that runs for 2026
            # runs at the equivalent of Jan 1, 2026, but expects the 2026 numbers to be done
            # which is unfortunate.
            ctx.bovine_methane_annual = correct.delay(1 * u.year)

        state.register_emission('Enteric_Fermentation', 'CH4', 'bovine_methane_annual')
        return 2026.5 * u.year

    def step(self, state, current):
        stash = state.stash(self)
        emfac = table_A3p4_11()

        frac = (
            current.bovine_population_fraction_on_bovaer
            * self.bovaer_actual_vs_nominal)

        # TODO: for each province
        bovine_methane_contribs = []
        bovaer_cost_contribs = []
        for ob, hc, livestock in zip(stash.on_bovaer, stash.headcounts, stash.livestock_type):
            hc_now = hc.query(state.t_now)
            ob_now = hc_now * frac
            assert 0 <= frac <= 1
            ob.append(state.t_now, ob_now)
            emfac_now = emfac[livestock].query(state.t_now)
            bovine_methane_contribs.append(
                ob_now * emfac_now * (1 - self.bovaer_methane_reduction[livestock])
                + (hc_now - ob_now) * emfac_now)
            bovaer_cost_contribs.append(ob_now * self.bovaer_cost[livestock])

        current.bovine_methane_rate = sum(bovine_methane_contribs)
        current.bovaer_cost = sum(bovaer_cost_contribs)

        assert state.t_now.u == u.year
        if state.t_now.magnitude == int(state.t_now.magnitude):
            # XXX this dynamic element defines bovine_methane_annual, yes
            # but *with a delay* that isn't known to the scheduler.
            # This will cause the atmospheric chemistry model to
            # use the wrong value sometimes, and there'll be no warning about it.
            year_end = state.t_now
            year_start = year_end - 1 * u.year
            state.sts['bovine_methane_annual'].append(
                year_end,  # XXX this year is off by one
                state.sts['bovine_methane_rate'].bin_integral(
                    year_start, year_end))
            state.sts['bovaer_cost_annual'].append(
                year_end,  # XXX this year is off by one
                state.sts['bovaer_cost'].bin_integral(
                    year_start, year_end))
        else:
            pass

        # TODO: how much milk and beef are produced?
        # TODO: raise a violation if the ratio of production to Canada's
        # population gets too from historic norms. Should the fix be to assume
        # there will be more cattle? How many? Where would they live, what
        # would they eat, etc?

        return state.t_now + .5 * u.year
