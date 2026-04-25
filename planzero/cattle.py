from functools import cache as memcache

from pydantic import Field, computed_field
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from .ureg import u
from .enums import IPCC_Sector, StandardScenarios, PT
from . import sts
from .barriers import Barrier

from .eccc_nir_annex3p4 import table_A3p4_11
from .sc_3210013001 import (
    FarmType, Livestock, Livestock_nonsums, SurveyDate,
    number_of_cattle_by_class_and_farm_type_combined_surveys)


feature_mask_by_farmtype = {
    FarmType.Dairy: [lt not in [Livestock.BeefCows] for lt in Livestock_nonsums],
    FarmType.Beef: [lt not in [Livestock.DairyCows, Livestock.DairyHeifers] for lt in Livestock_nonsums],
    FarmType.AllCattle: [True for lt in Livestock_nonsums],
}

@memcache
def sorted_years(farm_type):
    # Identify common set of years
    # Each element in data_slice is a SparseTimeSeries or a Quantity (if 0)
    # We want to find the union of all times across all Livestock and PT
    combined_surveys_pt, _ = number_of_cattle_by_class_and_farm_type_combined_surveys()
    data_slice = combined_surveys_pt[:, farm_type, :]
    all_years = set()
    for livestock in Livestock_nonsums:
        for pt in PT:
            val = data_slice[livestock, pt]
            if isinstance(val, sts.STS):
                assert val.t_unit == u.years
                all_years.update(val.times) # these will be floating values that either either whole numbers or whole numbers + 0.5
    
    return list(sorted(all_years))

@memcache
def np_data(farm_type):
    combined_surveys_pt, _ = number_of_cattle_by_class_and_farm_type_combined_surveys()
    data_slice = combined_surveys_pt[:, farm_type, :]
    year_to_idx = {year: i for i, year in enumerate(sorted_years(farm_type))}
    num_surveys = len(sorted_years(farm_type))
    num_livestock_types = len(Livestock_nonsums)
    num_PTs = len(PT)
    
    # Construct 3D array (num_years, num_livestock, num_pts)
    data = np.zeros((num_surveys, num_livestock_types, num_PTs))
    data[:] = float('nan')
    for i, lt in enumerate(Livestock_nonsums):
        for j, pt in enumerate(PT):
            val = data_slice[lt, pt]
            if isinstance(val, sts.STS):
                # SparseTimeSeries might not have all years
                # We assume 0 for missing years based on the plan,
                # but let's be careful and query if it's within range
                #print(livestock, pt, val.times[0])
                for t, v in zip(val.times, val.values[1:]):
                    data[year_to_idx[t], i, j] = v
            else:
                data[:, i, j] = val.magnitude
    return data


def Xy(farm_type, context_size, pt_idx, start_year_inclusive, cutoff_year_inclusive) -> tuple[list, list]:
    """Return a feature matrix, target matrix pair"""
    survey_idxs = [
        ii for ii, t in enumerate(sorted_years(farm_type))
        if (start_year_inclusive <= t <= cutoff_year_inclusive
            and ii >= context_size)]
    X = []
    y = []
    # For each year tau from context_size to t_idx
    X_pre_sliced = np_data(farm_type)[:, feature_mask_by_farmtype[farm_type], pt_idx]
    y_pre_sliced = np_data(farm_type)[:, :, pt_idx]
    for tau_idx in survey_idxs:
        X_tau = X_pre_sliced[tau_idx - context_size : tau_idx].flatten()
        y_tau = y_pre_sliced[tau_idx]
        if np.isfinite(X_tau).all() and np.isfinite(y_tau).all():
            X.append(X_tau)
            y.append(y_tau)
    return X, y

class InsufficientContext(Exception):
    pass

def rollout(farm_type, context_size, pt_idx, start_year, model, scale, n_steps):
    """Return a model rollout, starting with the prediction of start_year"""
    rval = np.full((context_size + n_steps, len(Livestock_nonsums)), np.nan)
    assert n_steps >= 0
    try:
        survey_idx = sorted_years(farm_type).index(start_year - 0.5)
    except ValueError:
        return rval

    if survey_idx < context_size:
        return rval

    rval[:context_size, :] = np_data(farm_type)[
        survey_idx - context_size + 1: survey_idx + 1,
        :,
        pt_idx]

    # proceed with rollouts
    for step in range(n_steps):
        Xmat = rval[step:step + context_size, feature_mask_by_farmtype[farm_type]].reshape(1, -1)
        if not np.isfinite(Xmat).all():
            break
        rval[step + context_size] = model.predict(Xmat * scale) / scale
    return rval


def train_model(farm_type, context_size, t_idx=-1):
    X_train = []
    y_train = []
    for pt_idx, _ in enumerate(PT):
        X_p, y_p = Xy(
            farm_type, 
            context_size,
            pt_idx,
            sorted_years(farm_type)[0],
            sorted_years(farm_type)[t_idx])
        X_train.extend(X_p)
        y_train.extend(y_p)

    min_train_ratio = 2
    if len(X_train) > len(PT) * context_size * min_train_ratio:
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        #model = LinearRegression(fit_intercept=True)
        #model = ElasticNet(alpha=10.1, l1_ratio=.1)
        model=RidgeCV()
        scale = 1e-4
        valid_steps = 5
        model.fit(X_train * scale, y_train * scale)
        # looking at the results of eval_AR123 suggests that the AR
        # model is no better than a cycling baseline after 5 steps
        return model, scale, valid_steps


def eval_AR123(farm_type):
    # This function is meant to be run in jupyter notebook

    local_sorted_years = sorted_years(farm_type)
    num_surveys = len(local_sorted_years)
    num_livestock_types = len(Livestock_nonsums)
    num_PTs = len(PT)
    data = np_data(farm_type)
    
    # Walk-forward validation
    def walk_forward(context_size, n_steps):
        predictions = np.full((n_steps, num_surveys, num_livestock_types, num_PTs), np.nan)
        
        for t_idx in range(context_size, num_surveys):
            # Training data: samples up to year t_idx - 1
            model_scale_valid_steps = train_model(
                farm_type=farm_type,
                context_size=context_size,
                t_idx=t_idx)
            if model_scale_valid_steps:
                model, scale, _ = model_scale_valid_steps
                # Predict for year t_idx
                for p in range(num_PTs):
                    try:
                        pred = rollout(
                            farm_type=farm_type,
                            context_size=context_size,
                            pt_idx=p,
                            start_year=local_sorted_years[t_idx],
                            model=model,
                            scale=scale,
                            n_steps=n_steps)
                        predictions[:, t_idx, :, p] = pred[context_size:]
                    except InsufficientContext:
                        pass

        return predictions

    n_step_rollout = 10
    # Baseline: Constant predictor (previous year)
    baseline_preds = np.full((n_step_rollout, num_surveys, num_livestock_types, num_PTs), np.nan)
    for t_idx in range(2, num_surveys):
        baseline_preds[::2, t_idx] = data[t_idx - 2]
        baseline_preds[1::2, t_idx] = data[t_idx - 1]

    # AR models
    ar_max_context = 8
    ar_preds = [walk_forward(ii, n_steps=n_step_rollout)
                for ii in range(1, ar_max_context)]

    # Aggregation: Sum over PT and Livestock for total Canada dairy cattle
    true_total = np.sum(data, axis=(1, 2))
    baseline_total = np.sum(baseline_preds, axis=(2, 3))
    ar_totals = [np.sum(ar_pred, axis=(2, 3)) for ar_pred in ar_preds]
    ar_max_total = ar_totals[-1]
    
    # For MAE, only use years where all models predicted
    eval_mask = ~np.isnan(ar_max_total[0])
    eval_years = np.array(local_sorted_years)[eval_mask]

    def steps_eval(foo, metric):
        return np.asarray([
            metric(true_total[eval_mask][step:],
                   foo[step, eval_mask][slice(None, -step) if step else slice(None, None)])
            for step in range(n_step_rollout)])
    
    mae_baseline = steps_eval(baseline_total, mean_absolute_error)
    mae_ars = [steps_eval(ar_total, mean_absolute_error) for ar_total in ar_totals]
    
    print(f"Mean Absolute Error (Total {farm_type}):")
    print(f"  Baseline (Constant): {mae_baseline}")
    for ii, mae in enumerate(mae_ars):
        print(f"  AR({ii + 1}):               {mae}")
    #return baseline_preds, ar_preds

    rmse_baseline = steps_eval(baseline_total, root_mean_squared_error)
    rmse_ars = [steps_eval(ar_total, root_mean_squared_error) for ar_total in ar_totals]
    
    print(f"Root Mean Squared Error (Total {farm_type}):")
    print(f"  Baseline (Constant): {rmse_baseline}")
    for ii, rmse in enumerate(rmse_ars):
        print(f"  AR({ii + 1}):               {rmse}")

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(local_sorted_years, true_total, 'k-', label='True Total (Canada)', linewidth=2)
    plt.plot(local_sorted_years, [baseline_total[0, i] if not np.isnan(baseline_total[0, i]) else np.nan for i in range(num_surveys)], 'r--', label='Baseline Prediction')
    for ii, ar_total in enumerate(ar_totals):
        plt.plot(local_sorted_years, [ar_total[0, i] if not np.isnan(ar_total[0, i]) else np.nan for i in range(num_surveys)], label=f'AR({ii + 1}) Prediction')
    
    plt.title('Walk-Forward Validation: Dairy Cattle Population (Total Canada)')
    plt.xlabel('Year')
    plt.ylabel('Number of Cattle')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



class Cattle_Population(Barrier):

    """Historical actuals followed by auto-regressive model of future cattle
    population, followed by constant cyclic repetition of a constant pair of
    january and july values.

    The autoregressive weights permit a linear modelling of the birth of
    calves from cows, and their growing up as heifers and steers etc.  That
    said, the basic dynamics at that level have started to change since around
    2016 with the innovation of sexed semen. Sexed semen allows about 30% of
    top-producing cows to conceive exclusively female calves, at a sufficient
    rate to maintain the herd on a farm (and satisfy the farm's quota to
    produce milk).  Cows must conceive calves roughly once a year to continue
    to producing milk, so the possibility of sexed semen can also be used for
    the remaining 70% of cows to produce male cross-breeds that are suitable
    for the beef market.  This so-called "beef on dairy" practice changes the
    beef population dynamics.  In terms of the linear autoregressive model,
    this practice changes the rates at which dairy cows yield calves that
    become dairy heifers, and links dairy cows to the steer population.  There
    isn't enough data to train and test a new linear model on the years since
    2016, so I just note this as an un-modelled phenomenon that could be
    reflected in a future model.
    https://www.fcc-fac.ca/en/knowledge/beef-on-dairy-changing-canadas-beef-supply

    The model does not include features of the historical or projected climate (e.g.
    cold spells requiring more feed in winter, drought-triggered selling in summer,
    extreme heat or cold interfering with calving).
    To include this sort of thing, consider deriving features from e.g.
    # https://climate-scenarios.canada.ca/index.php?page=CMIP6-statistical-downscaling

    The model does not include features of historical or projected
    domestic or international supply or demand (e.g. beef, milk, feed, land).

    The model does not include features of price.
    """

    @computed_field
    def farm_type(self) -> object:
        return FarmType.AllCattle

    @property
    def ar_context_size(self):
        return 4

    @computed_field
    def short_description(self) -> str:
        return f"Model cattle population, milk and beef production"

    @computed_field
    def ipcc_sectors(self) -> list[object]:
        return []

    @computed_field
    def scenarios(self) -> list[object]:
        return []

    @computed_field
    def research(self) -> dict[str, str]:
        return {}

    def on_add_project(self, state):
        stash = state.stash(self)
        model, scale, valid_steps = train_model(
            farm_type=self.farm_type,
            context_size=self.ar_context_size)
        combined_surveys_pt, _ = number_of_cattle_by_class_and_farm_type_combined_surveys()

        with state.requiring_current(self) as ctx:
            pass

        with state.defining(self) as ctx:
            stash.headcounts_by_livestock_pt = {}
            for pti, pt in enumerate(PT):
                pt_rollout = rollout(
                    farm_type=self.farm_type,
                    context_size=self.ar_context_size,
                    pt_idx=pti,
                    start_year=sorted_years(self.farm_type)[-1] + 0.5,
                    model=model,
                    scale=scale,
                    n_steps=valid_steps)

                # initialize with historical
                for lti, livestock in enumerate(Livestock_nonsums):
                    hc = combined_surveys_pt[livestock, self.farm_type, pt]
                    if isinstance(hc, sts.STS):
                        hc = hc.copy()
                        hc.values[0] = 0 # assume 0 instead of undefined

                        # extend by model rollout
                        for step in range(valid_steps):
                            hc.append(
                                t=(sorted_years(self.farm_type)[-1]
                                   + (step + 1) * 0.5
                                  ) * hc.t_unit,
                                v=pt_rollout[self.ar_context_size + step, lti] * hc.v_unit)

                        state.declare_sts(self, hc, write=True,
                                          name=f'cattle_population_{livestock.value}_{pt.value}')
                        stash.headcounts_by_livestock_pt[livestock, pt] = hc
                    else:
                        assert hc.magnitude == 0

            headcounts = list(stash.headcounts_by_livestock_pt.values())
            ctx.total_cattle_headcount = sum(headcounts[1:], start=headcounts[0])

        t_step_start = (
            sorted_years(self.farm_type)[-1]
            + (valid_steps + 1) * 0.5
        ) * u.years
        stash.t_step_start = t_step_start # for Cattle_Enteric_Emissions
        return t_step_start

    def step(self, state, current):
        stash = state.stash(self)
        total_cattle_headcount = 0
        for hc in stash.headcounts_by_livestock_pt.values():
            hc_now = hc.values[-2] # value from same time-of-year, prev year
            total_cattle_headcount += hc_now
            hc.append(state.t_now, hc_now * u.cattle)
        current.total_cattle_headcount = total_cattle_headcount * u.cattle
        return state.t_now + .5 * u.year


class Bovaer_Adoption_Limit(Barrier):
    
    @computed_field
    def max_increase_rate(self) -> object:
        return 5.0 * u.percent / u.year

    @computed_field
    def short_description(self) -> str:
        return f"Assume Bovaer will only be adopted by, at most, {self.max_increase_rate} of cattle operations"

    @computed_field
    def description(self) -> str:
        return f"""Assume that no more than {self.max_increase_rate} of
        farmers will switch to administering Bovaer
        in any given year, but that adoption
        can ultimately go all the way to 100%.
        """

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
    def organic_fraction(self) -> float:
        # TODO: make this an STS, it can change over time
        # Approximately 1.5% of Canada's milk is organic, and about 0.7%
        # of beef is organic.
        return 0.01

    def on_add_project(self, state):
        with state.requiring_current(self) as ctx:
            ctx.bovine_population_fraction_on_bovaer = sts.SparseTimeSeries(
                default_value=0 * u.dimensionless, t_unit=u.years)

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
            (current.bovine_population_fraction_on_bovaer
             + self.max_increase_rate * 1.0 * u.year),
            (1 - self.organic_fraction) * u.dimensionless)
        # Apparently Bovaer is not allowed as part of organic production.
        return state.t_now + 1 * u.years

# TODO: there will be a cost for monitoring
# https://www.mn.uio.no/geo/english/about/news-and-events/news/2025/combined-drone-satelite-data-and-ground-based-measurements-methane-emissions.html
# It can apparently be done pretty well with drones
# There are approximately 70_000 cattle operations in Canada
# Monitoring might cost 10-20 million / year?

# TODO: Market mechanism in simulation to determine prices based on supply and demand

# TODO: Output-based carbon pricing model for agriculture


class Cattle_Enteric_Emissions(Barrier):
    """Assume cattle produce methane (less-so if they are fed Bovaer), milk and beef (TODO).
    The evolution of the cattle population is projected to remain constant.
    """

    @computed_field
    def short_description(self) -> str:
        return f"Model cattle population, production of methane (considering Bovaer), milk and beef (TODO)"

    @computed_field
    def ipcc_sectors(self) -> list[object]:
        return [IPCC_Sector.Enteric_Fermentation,
                IPCC_Sector.Other_Product_Manufacture_and_Use, # TODO: Is this the correct sector?
               ]

    @computed_field
    def scenarios(self) -> list[object]:
        return [StandardScenarios.Scaling]

    @computed_field
    def research(self) -> dict[str, str]:
        return {}

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

    # TODO: make these STS variables, not constants. The price might e.g. come down
    @computed_field
    def bovaer_cost(self) -> dict[object, object]:
        # https://www.producer.com/livestock/new-methane-feed-additive-pleases-producers
        return {
            Livestock.Bulls: .50 * u.CAD / u.day / u.cattle,
            Livestock.DairyCows: 0.50 * u.CAD / u.day / u.cattle,
            Livestock.BeefCows: 0.50 * u.CAD / u.day / u.cattle,
            Livestock.DairyHeifers: .35 * u.CAD / u.day / u.cattle,
            Livestock.BeefHeifers: .35 * u.CAD / u.day / u.cattle,
            Livestock.SlaughterHeifers: .35 * u.CAD / u.day / u.cattle,
            Livestock.Steers: .35 * u.CAD / u.day / u.cattle,
            Livestock.Calves: .20 * u.CAD / u.day / u.cattle,
        }

    def on_add_project(self, state):
        stash = state.stash(self)
        hc_by_livestock_pt \
                = state.stashes['Cattle_Population'].headcounts_by_livestock_pt
        rval = state.stashes['Cattle_Population'].t_step_start

        with state.requiring_current(self) as ctx:
            # TODO: for each type of cattle, for each province
            # will be written by Strategy
            ctx.bovine_population_fraction_on_bovaer = sts.SparseTimeSeries(
                default_value=0 * u.dimensionless, t_unit=u.years)

            # I don't know the details of current or actual production processes.
            # This number is chosen based on a conversation with Google Gemini
            # circa April 2026, in which it characterized the production footprint of Bovaer
            # as 20-50 times less in magnitude compared to the emission
            # reduction in enteric fermentation
            ctx.bovaer_production_CO2_per_methane_abated = sts.SparseTimeSeries(
                default_value=45 * u.kg_CO2 / u.cattle / u.year)

            for hc in hc_by_livestock_pt.values():
                state.declare_sts(self, hc, need_current=True)

        with state.defining(self) as ctx:
            emfac = table_A3p4_11()
            stash.headcounts = []
            stash.on_bovaer = []
            #stash.annual_methane_emissions = []
            stash.livestock_type = []
            bovine_methane = None
            for (livestock, pt), hc in hc_by_livestock_pt.items():
                stash.headcounts.append(hc)

                ## XXX this upper bound changes every few months with new statscan data
                #annual_methane_emissions = (emfac[livestock] * hc).bin_integrals(
                #    bin_boundaries=[tt * u.year for tt in range(1990, 2026 + 1)])
                #state.declare_sts(self, hc, need_current=True,
                #                  name=f'annual_methane_emissions_{livestock.value}_{pt.value}')
                #stash.annual_methane_emissions.append(annual_methane_emissions)

                frac_sts = (
                    state.sts['bovine_population_fraction_on_bovaer']
                    * self.bovaer_actual_vs_nominal)

                on_bovaer = frac_sts * hc
                stash.on_bovaer.append(on_bovaer)
                state.declare_sts(self, on_bovaer, write=True,
                                  name=f'headcount_on_Bovaer_{livestock.value}_{pt.value}')

                stash.livestock_type.append(livestock)

                if bovine_methane is None:
                    bovine_methane = hc * emfac[livestock]
                else:
                    bovine_methane += hc * emfac[livestock]

            ctx.bovine_methane_rate = bovine_methane.to(u.kt_CH4 / u.year)
            ctx.bovaer_cost = sts.SparseTimeSeries(default_value=0 * u.mega_CAD / u.year, t_unit=u.year)
            ctx.bovaer_cost_annual = sts.SparseTimeSeries(default_value=0 * u.mega_CAD, t_unit=u.year)
            ctx.bovaer_headcount = sts.SparseTimeSeries(default_value=0 * u.cattle, t_unit=u.year)

            last_complete_year = int(bovine_methane.times[-1])

            correct = bovine_methane.bin_integrals(
                bin_boundaries=[tt * u.year for tt in range(1990, last_complete_year + 1)])

            # XXX there's an off-by-one error in the way the data from the NIR is handled
            #     in this simulation, the reported totals for e.g. 2023 should be associated
            #     with the time point 2024 * u.years, because that's the
            #     trailing integral boundary.
            ctx.bovine_methane_annual = correct.delay(1 * u.year).to(u.kt_CH4)

            # TODO: we shouldn't get ahead of ourselves. The cattle population might be
            # forecast out to 2028, but the strategy of administering Bovaer might start
            # in e.g. 2026.
            #
            # Maybe the state needs some notion of "the present" from which
            # time strategies can start, and beyond which time barriers should not
            # initialize variables that depend (directly or indirectly) on strategies?
            #
            # Or maybe a well-written on_add_project should check the last time on it's
            # input variables, and not advance past that point?
            # (^^ This seems like the right way.)

            ctx.bovaer_production_CO2 = sts.SparseTimeSeries(
                default_value=0 * u.kt_CO2 / u.year)
            ctx.bovaer_production_CO2_annual = sts.SparseTimeSeries(
                default_value=0 * u.kt_CO2,
                t_unit=u.year)

        state.register_emission('Enteric_Fermentation', 'CH4',
                                'bovine_methane_annual')

        # basically guessing at this
        state.register_emission('Other_Product_Manufacture_and_Use', 'CO2',
                                'bovaer_production_CO2_annual')

        state.register_subsidy_requirement('bovaer_cost_annual')
        #assert 0, rval
        return rval

    def step(self, state, current):
        stash = state.stash(self)
        emfac = table_A3p4_11()

        frac = (
            current.bovine_population_fraction_on_bovaer
            * self.bovaer_actual_vs_nominal)

        # TODO: for each province
        bovine_methane_contribs = []
        bovaer_cost_contribs = []
        bovaer_CO2_contribs = []
        bovaer_headcount_contribs = []
        for ob, hc, livestock in zip(stash.on_bovaer, stash.headcounts, stash.livestock_type):
            hc_now = hc.query(state.t_now)
            ob_now = hc_now * frac
            assert 0 <= frac <= 1
            ob.append(state.t_now, ob_now)
            emfac_now = emfac[livestock].query(state.t_now)
            methane_potential = hc_now * emfac_now
            methane_avoided = ob_now * emfac_now * self.bovaer_methane_reduction[livestock]
            bovine_methane_contribs.append(methane_potential - methane_avoided)
            bovaer_cost_contribs.append(ob_now * self.bovaer_cost[livestock])
            bovaer_CO2_contribs.append(
                ob_now * current.bovaer_production_CO2_per_methane_abated)
            bovaer_headcount_contribs.append(ob_now)

        current.bovine_methane_rate = sum(bovine_methane_contribs)
        current.bovaer_cost = sum(bovaer_cost_contribs)
        current.bovaer_production_CO2 = sum(bovaer_CO2_contribs)
        current.bovaer_headcount = sum(bovaer_headcount_contribs)

        assert state.t_now.u == u.year
        if state.t_now.magnitude == int(state.t_now.magnitude):
            # XXX this dynamic element defines bovine_methane_annual, yes
            # but *with a delay* that isn't known to the scheduler.
            # This will cause the atmospheric chemistry model to
            # use the wrong value sometimes, and there'll be no warning about it.
            year_end = state.t_now
            year_start = year_end - 1 * u.year
            # XXX this year may be off by one
            methane_mass = state.sts['bovine_methane_rate'].bin_integral(
                year_start, year_end)
            state.sts['bovine_methane_annual'].append(
                year_end, methane_mass)
            state.sts['bovaer_cost_annual'].append(
                year_end,
                state.sts['bovaer_cost'].bin_integral(
                    year_start, year_end))
            state.sts['bovaer_production_CO2_annual'].append(
                year_end,
                state.sts['bovaer_production_CO2'].bin_integral(
                    year_start, year_end))
        else:
            # TODO: split this annual bit into a separate DynamicElement with
            # a 1-year step size
            pass

        # TODO: how much milk and beef are produced?
        #
        # TODO: raise a violation if the ratio of food production to Canada's
        # population deviates too far from historic norms. Should the fix be
        # to assume there will be more cattle? How many? Where would they
        # live, what would they eat, etc?

        return state.t_now + .5 * u.year


class Bovaer_Monitoring(Barrier):

    @computed_field
    def short_description(self) -> str:
        return f"Assume monitoring Bovaer usage costs {self.paperwork_monitoring} for paperwork and {self.onsite_monitoring} for on-site inspection"

    @computed_field
    def paperwork_monitoring(self) -> object:
        # Gemini made this up
        return 1000 * u.CAD / u.farm / u.year

    @computed_field
    def onsite_monitoring(self) -> object:
        # assume one visit per year at this rate, which Gemini made up
        return 3000 * u.CAD / u.farm / u.year

    @computed_field
    def cattle_per_farm(self) -> object:
        # TODO: pull down actual data from https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=3210015101
        return 160 * u.cattle / u.farm

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
        with state.requiring_current(self) as ctx:
            ctx.bovaer_headcount = sts.SparseTimeSeries(default_value=0 * u.cattle, t_unit=u.year)

        with state.defining(self) as ctx:
            ctx.bovaer_monitoring_cost_annual_total = sts.SparseTimeSeries(
                default_value=0 * u.mega_CAD)

        state.register_subsidy_requirement('bovaer_monitoring_cost_annual_total')

        return int(state.t_now.to('year').magnitude) * u.year

    def step(self, state, current):
        cost_rate = self.onsite_monitoring + self.paperwork_monitoring
        current.bovaer_monitoring_cost_annual_total = (
            cost_rate * 1 * u.year
            / self.cattle_per_farm
            * current.bovaer_headcount)
        return state.t_now + 1 * u.year
