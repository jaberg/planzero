from functools import cache as memcache

import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
import numpy as np
from pydantic import computed_field
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from . import nir2025, sts
from .annual_emission_results import (
    aer_key_normal_mu_ca,
    aer_key_normal_sigma_ca,
    aer_result_key,
)
from .barriers import Barrier
from .challenge import PreNIR_2025_m04
from .eccc_nir_annex3p4 import table_A3p4_11
from .enums import GHG, PT, Activity, IPCC_Sector
from .sc_3210013001 import (
    FarmType,
    Livestock,
    Livestock_nonsums,
    number_of_cattle_by_class_and_farm_type_combined_surveys,
)
from .symmetric_blended_lognormal import kl_divergence_uniform_normal_mixture
from .ureg import u

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



class Cattle_Population_AR(Barrier):

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

    ar_context_size: int = 4 # 2 per year
    farm_type: object = FarmType.AllCattle

    @computed_field
    def short_description(self) -> str:
        return f"Model cattle population, milk and beef production"

    @computed_field
    def cattle_per_farm(self) -> object:
        # TODO: pull down actual data from https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=3210015101
        return 160 * u.cattle / u.farm


    def on_add_project(self, state):
        stash = state.stash(self)
        model, scale, valid_steps = train_model(
            farm_type=self.farm_type,
            context_size=self.ar_context_size)
        combined_surveys_pt, _ = number_of_cattle_by_class_and_farm_type_combined_surveys()

        valid_steps = 0 # disables the use of AR model until blog post about it

        with state.requiring_current(self) as ctx:
            pass

        with state.defining(self) as ctx:
            stash.headcounts_by_livestock_pt = {}
            stash.operations_by_pt = {}
            for pti, pt in enumerate(PT):
                pt_rollout = rollout(
                    farm_type=self.farm_type,
                    context_size=self.ar_context_size,
                    pt_idx=pti,
                    start_year=sorted_years(self.farm_type)[-1] + 0.5,
                    model=model,
                    scale=scale,
                    n_steps=valid_steps)

                cattle_operations = None

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

                        stash.headcounts_by_livestock_pt[livestock, pt] = hc
                        name = f'cattle_population_{livestock.value}_{pt.value}'
                        state.declare_sts(self, hc, write=True, name=name)
                        state.register_driver(pt, livestock, name)

                        # farm count (fc) from head count (hc)
                        if cattle_operations is None:
                            cattle_operations = hc / self.cattle_per_farm
                        else:
                            cattle_operations += hc / self.cattle_per_farm
                    else:
                        assert hc.magnitude == 0

                if cattle_operations is not None:
                    cattle_operations_name = f'cattle_operations_{pt.value}'
                    state.declare_sts(
                        project=self,
                        sts=cattle_operations,
                        write=True,
                        name=cattle_operations_name)
                    state.register_driver(
                        pt,
                        driver='Cattle Operations',
                        sts_key=cattle_operations_name)
                    stash.operations_by_pt[pt] = cattle_operations

        t_step_start = (
            sorted_years(self.farm_type)[-1]
            + (valid_steps + 1) * 0.5
        ) * u.years
        stash.t_step_start = t_step_start # for Cattle_Enteric_Emissions
        return t_step_start

    def step(self, state, current):
        stash = state.stash(self)
        hc_by_pt = {}
        for (_, pt), hc in stash.headcounts_by_livestock_pt.items():
            hc_now = hc.values[-2] # value from same time-of-year, prev year
            hc_by_pt.setdefault(pt, 0)
            hc_by_pt[pt] += hc_now
            hc.append(state.t_now, hc_now * u.cattle)

        for pt, oc in stash.operations_by_pt.items():
            oc.append(state.t_now, hc_by_pt[pt] * u.cattle / self.cattle_per_farm)
        return state.t_now + .5 * u.year


class Bovaer_Adoption_Limit(Barrier):
    
    @computed_field
    def max_increase_rate(self) -> object:
        return 5.0 * u.percent / u.year

    @property
    def prob_max_increase_rate(self) -> tuple[float, float]:
        return (
                self.max_increase_rate.magnitude / 2,
                self.max_increase_rate.magnitude * 2)

    @computed_field
    def short_description(self) -> str:
        return f"Assume Bovaer will only be adopted by, at most, {self.max_increase_rate} of cattle operations, up to a maximum of {(1 - self.organic_fraction) * 100:.1f}%"

    @computed_field
    def description(self) -> str:
        low, high = self.prob_max_increase_rate
        return f"""Assume that no more than {self.max_increase_rate} of
        farmers will switch to administering Bovaer
        in any given year, but that adoption
        can ultimately rise to {(1 - self.organic_fraction) * 100:.1f}%,
        the remainder of whom are organic farmers who won't adopt it.
        </p>
        When used in probabilistic models, assume instead that
        {low}-{high}% of farmers will switch to Bovaer in any given year.
        <p>
        """

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
        return 2025 * u.years

    def step(self, state, current):
        current.max_fraction_of_cattle_on_bovaer = min(
            (current.bovine_population_fraction_on_bovaer
             + self.max_increase_rate * 1.0 * u.year),
            (1 - self.organic_fraction) * u.dimensionless)
        # Apparently Bovaer is not allowed as part of organic production.
        return state.t_now + 1 * u.years

    def annual_scan_init(self, initial_carry, xs, years, constants, jrkey):
        n_samples = constants['sigma_ca'].shape[0]
        initial_carry.setdefault('bovine_population_fraction_on_bovaer', jnp.zeros(n_samples))
        initial_carry['max_fraction_of_cattle_on_bovaer'] = jnp.zeros(n_samples)
        initial_carry['BAL_key'] = jrandom.key(934)

    def annual_scan_step(self, new_carry, y, x, year, carry, constants, outputs):
        n_samples = constants['sigma_ca'].shape[0]
        new_carry['BAL_key'], key = jrandom.split(carry['BAL_key'])
        low, high = self.prob_max_increase_rate
        y['max_increase_fraction'] = jrandom.uniform(
                key, (n_samples,), 'float64', low / 100, high / 100)
        if 'bovine_population_fraction_on_bovaer' in outputs:
            new_carry['bovine_population_fraction_on_bovaer'] \
                    = carry['bovine_population_fraction_on_bovaer']
        new_carry['max_fraction_of_cattle_on_bovaer'] = jnp.minimum(
                (carry['bovine_population_fraction_on_bovaer']
                 + y['max_increase_fraction']),
                (1 - self.organic_fraction))

    @property
    def posts_developing_this_page(self) -> list[str]:
        return ['ProbabilisticBovaer',
                'ModellingBovaer']


class Bovaer_Production_Emission_Factors(Barrier):

    @computed_field
    def short_description(self) -> str:
        return f"""Suppose that embedded/production emission of Bovaer is about {self.rate}."""

    @computed_field
    def description(self) -> str:
        return f"""In deterministic models, this barrier estimates the
        production emissions of Bovaer at {self.rate}.
        </p>
        <p>
        In the probabilistic version of this barrier,
        the logic is based on a "20-50x" reduction in CO2e compared with
        methane emitted from cattle, as suggested by an LLM-AI chat,
        which was the original source for the {self.rate} value.
        For each sampled scenario, a reduction factor
        is sampled uniformly from 20 to 50.
        </p>
        <p>This barrier does not reflect manufacturer or 3rd party
        estimates of emission rates nor does
        it include any change in production emission efficiency over time.
        """

    @computed_field
    def rate(self) -> object:
        return 45 * u.kg_CO2 / u.cattle / u.year

    def on_add_project(self, state):

        with state.requiring_current(self) as ctx:
            # TODO: for each type of cattle, for each province
            # will be written by Strategy
            ctx.bovine_population_fraction_on_bovaer = sts.SparseTimeSeries(
                default_value=0 * u.dimensionless, t_unit=u.years)

        with state.defining(self) as ctx:

            # I don't know the details of current or actual production processes.
            # This number is chosen based on a conversation with Google Gemini
            # circa April 2026, in which it characterized the production footprint of Bovaer
            # as 20-50 times less in magnitude compared to the emission
            # reduction in enteric fermentation
            ctx.bovaer_production_CO2_per_methane_abated = sts.SparseTimeSeries(
                default_value=0 * self.rate,
                t_unit=u.years)

            # TODO: model where the Bovaer is actually produced.
            for pt in PT:
                for livestock in Livestock_nonsums:
                    state.register_emission_factor(
                        sts_key='bovaer_production_CO2_per_methane_abated',
                        ipcc_sector=IPCC_Sector.Other_Product_Manufacture_and_Use,
                        ghg=GHG.CO2,
                        pt=pt,
                        driver=livestock)
        # TODO: revisit after switching from step() to fill()
        # so the start date will be based on where inputs leave off
        return state.stashes['Cattle_Population_AR'].t_step_start

    def step(self, state, current):
        current.bovaer_production_CO2_per_methane_abated = (
            current.bovine_population_fraction_on_bovaer
            * self.rate)
        return state.t_now + 1 * u.year

    def annual_scan_init(self, new_carry, xs, years, constants, jrkey):
        n_samples = constants['sigma_ca'].shape[0]
        key = jrandom.key(934)
        new_carry['bovaer_production_emission_factor'] = jrandom.uniform(
                key, (n_samples,), 'float64',
                20,
                50)

    def annual_scan_step(self, new_carry, y, x, year, carry, constants, outputs):
        new_carry['bovaer_production_emission_factor'] = (
                carry['bovaer_production_emission_factor'])


class Cattle_Enteric_Emission_Rates_NIR2025_Bovaer(Barrier):
    """Assume cattle produce methane (less-so if they are fed Bovaer).

    Defines one emission factor time series per livestock type.
    """

    calculate_KL_divergence_PreNIR_2025_m04: bool = False
    draws_per_posterior_sample:int = 32

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

    def on_add_project(self, state):
        stash = state.stash(self)

        with state.requiring_current(self) as ctx:
            # TODO: for each type of cattle, for each province
            # will be written by Strategy
            ctx.bovine_population_fraction_on_bovaer = sts.SparseTimeSeries(
                default_value=0 * u.dimensionless, t_unit=u.years)

        with state.defining(self) as ctx:
            table = table_A3p4_11()
            stash.emfac = {}

            for livestock in Livestock_nonsums:
                name = f'enteric_fermentation_emission_rate_{livestock.value}'
                stash.emfac[livestock] = table[livestock].copy()
                state.declare_sts(self, stash.emfac[livestock], write=True, name=name)
                for pt in PT:
                    state.register_emission_factor(
                        driver=livestock,
                        pt=pt,
                        ghg=GHG.CH4,
                        ipcc_sector=IPCC_Sector.Enteric_Fermentation,
                        sts_key=name)

        # TODO: revisit after switching from step() to fill()
        # so the start date will be based on where inputs leave off
        return state.stashes['Cattle_Population_AR'].t_step_start

    def step(self, state, current):
        stash = state.stash(self)
        table = table_A3p4_11()
        methane_reduction = self.bovaer_methane_reduction
        actual = self.bovaer_actual_vs_nominal
        for livestock in Livestock_nonsums:
            stash.emfac[livestock].append(
                state.t_now,
                table[livestock].query(state.t_now)
                * (
                    (
                        (1 - current.bovine_population_fraction_on_bovaer)
                        * 1 # full rate
                    )
                    + (
                        current.bovine_population_fraction_on_bovaer
                        * (1 - methane_reduction[livestock] * actual)
                    )
                ))
        return state.t_now + 1 * u.year

    def annual_scan_init(self, new_carry, xs, years, constants, jrkey):
        # from prob_bovaer.py
        num_samples, num_regions = constants['sigma_pt'].shape
        jrkey, tmpkey = jrandom.split(jrkey)
        constants['shift_pt'] = (
                jrandom.normal(tmpkey, (self.draws_per_posterior_sample, num_samples, num_regions))
                * constants['sigma_pt']
                * constants['enteric_ch4_ktCO2e_scale']
                )
        jrkey, tmpkey = jrandom.split(jrkey)
        constants['shift_ca'] = (
                jrandom.normal(tmpkey, (self.draws_per_posterior_sample, num_samples,))
                * constants['sigma_ca']
                * constants['enteric_ch4_ktCO2e_scale']
                )

    def annual_scan_step(self, new_carry, y, x, year, carry, constants, outputs):
        # (n_samples,)
        bovaer_fraction = new_carry['bovine_population_fraction_on_bovaer']
        ef_kg_CH4_per_head = constants['latent_emission_factors']
        from .ghgvalues import GWP_100

        ef_kt_CO2e = ef_kg_CH4_per_head * GWP_100[GHG.CH4].magnitude / 1_000_000

        methane_reduction_potential = jnp.array([self.bovaer_methane_reduction[lt] for lt in Livestock_nonsums])

        actual = self.bovaer_actual_vs_nominal
        # (n_samples, n_livestock_types)
        factor_per_livestock_type = (
                (1 - bovaer_fraction[:, None]) * 1 # full rate
                + (bovaer_fraction[:, None]
                   * (1 - methane_reduction_potential * actual)))

        bovaer_factor_per_livestock_type = (
                bovaer_fraction[:, None]
                * (1 - methane_reduction_potential * actual))

        heads = constants['latent_livestock_counts']
        # broadcast over PTs
        emissions_by_cattle_type = (
                heads
                * (factor_per_livestock_type[:, :, None]
                   * ef_kt_CO2e[:, :, None])
                )

        emissions_by_cattle_type_on_bovaer = (
                heads
                * (bovaer_factor_per_livestock_type[:, :, None]
                   * ef_kt_CO2e[:, :, None]))

        # sum over cattle type to get shape (sample_size, PT)
        # and trim off the PT.XX category because it should have been factored
        # into the headcounts during inference
        y['enteric_fermentation_ktCO2e_pt'] = emissions_by_cattle_type.sum(axis=1)[:, :13]
        y['enteric_fermentation_ktCO2e_ca'] = emissions_by_cattle_type.sum(axis=(1, 2))

        y['enteric_fermentation_ktCO2e_pt_sample'] \
                = y['enteric_fermentation_ktCO2e_pt'] + constants['shift_pt']
        y['enteric_fermentation_ktCO2e_ca_sample'] \
                = y['enteric_fermentation_ktCO2e_ca'] + constants['shift_ca']

        sector = IPCC_Sector.Enteric_Fermentation
        for ghg in GHG:
            result_key = aer_result_key(
                    sector=sector,
                    ghg=ghg,
                    activity=Activity.Farming_Cattle
                    )
            if ghg == GHG.CH4:
                y[result_key] = y['enteric_fermentation_ktCO2e_ca_sample'].reshape(-1)
            else:
                y[result_key] = jnp.zeros((1,))

        # We're ignoring the variance from sigma_pt and sigma_ca here
        # I'm not really sure if that's correct or not.
        emissions_by_cattle_type_on_bovaer_pt \
                = emissions_by_cattle_type_on_bovaer.sum(axis=1)[:, :13]
        emissions_by_cattle_type_on_bovaer_ca \
                = emissions_by_cattle_type_on_bovaer.sum(axis=(1, 2))

        y['bovaer_production_emissions_ktCO2e_ca_sample'] = (
                emissions_by_cattle_type_on_bovaer_ca
                / new_carry['bovaer_production_emission_factor'])
        y['bovaer_production_emissions_ktCO2e_pt_sample'] = (
                emissions_by_cattle_type_on_bovaer_pt
                / new_carry['bovaer_production_emission_factor'][:, None])

    def annual_scan_post(self, jrkey, post_vals, final_carry, ys, xs, years, constants):
        sector = IPCC_Sector.Enteric_Fermentation
        challenge = PreNIR_2025_m04()
        for years_idx_of_2023, year in enumerate(years):
            if year == 2023:
                break
        else:
            raise ValueError('Year 2023 not modelled')

        for years_idx_of_2050, year in enumerate(years):
            if year == 2050:
                break
        else:
            years_idx_of_2050 = None

        mu_pt_2023 = ys['enteric_fermentation_ktCO2e_pt'][years_idx_of_2023]
        mu_ca_2023 = ys['enteric_fermentation_ktCO2e_ca'][years_idx_of_2023]

        if years_idx_of_2050 is not None:
            # mu_pt_2050 = ys['enteric_fermentation_ktCO2e_pt'][years_idx_of_2050]
            mu_ca_2050 = ys['enteric_fermentation_ktCO2e_ca'][years_idx_of_2050]

        # from prob_bovaer.py
        sigma_pt = constants['sigma_pt'] * constants['enteric_ch4_ktCO2e_scale']
        sigma_ca = constants['sigma_ca'] * constants['enteric_ch4_ktCO2e_scale']

        for ghg in GHG:
            if ghg == GHG.CH4:
                real_PTs = [pt for pt in PT if pt != PT.XX]

                ca_dist, pt_dists = nir2025.ktCO2e_numpyro_dist_pt_ca(
                    sector=sector,
                    ghg=ghg,
                    year=2023)

                KL_PT = []

                for jj, pt in enumerate(real_PTs):
                    KL_PT.append(
                            kl_divergence_uniform_normal_mixture(
                                p=pt_dists[jj],
                                q_mu=mu_pt_2023[:, jj],
                                q_sigma=sigma_pt[:, jj]))
                KL_CA = kl_divergence_uniform_normal_mixture(
                        p=ca_dist,
                        q_mu=mu_ca_2023,
                        q_sigma=sigma_ca)

                challenge = PreNIR_2025_m04()
                constants[challenge.key_sector_ghg_pt(sector, ghg)] = jnp.stack(KL_PT)
                constants[challenge.key_sector_ghg_ca(sector, ghg)] = KL_CA

                if years_idx_of_2050 is not None:
                    # use "Other" as activity to help ensure that this is the
                    # only source setting the variable
                    constants[aer_key_normal_mu_ca(sector, ghg, Activity.Other)] = mu_ca_2050
                    constants[aer_key_normal_sigma_ca(sector, ghg, Activity.Other)] = sigma_ca
            else:
                post_vals[challenge.key_sector_ghg_ca(sector, ghg)] = (
                        jnp.zeros(()))
                post_vals[challenge.key_sector_ghg_pt(sector, ghg)] = (
                        jnp.zeros((13,)))

                if years_idx_of_2050 is not None:
                    # use "Other" as activity to help ensure that this is the
                    # only source setting the variable
                    constants[aer_key_normal_mu_ca(sector, ghg, Activity.Other)] = (
                            jnp.zeros(()))
                    constants[aer_key_normal_sigma_ca(sector, ghg, Activity.Other)] = (
                        jnp.ones(()))



# TODO: there will be a cost for monitoring
# https://www.mn.uio.no/geo/english/about/news-and-events/news/2025/combined-drone-satelite-data-and-ground-based-measurements-methane-emissions.html
# It can apparently be done pretty well with drones
# There are approximately 70_000 cattle operations in Canada
# Monitoring might cost 10-20 million / year?


class Bovaer_Monitoring(Barrier):

    @computed_field
    def short_description(self) -> str:
        return f"""Assume administering and monitoring costs
    {self.paperwork_monitoring} for paperwork and {self.onsite_monitoring} for
    on-site inspection"""

    @computed_field
    def description(self) -> str:
        return f"""Assume administering and monitoring costs
    {self.paperwork_monitoring} for paperwork and {self.onsite_monitoring} for
    on-site inspection. Furthermore, estimate that these amounts are paid
    in labour costs, from which income tax is collected back at a rate of
    {self.income_tax_rate * 100}%.
    """

    @computed_field
    def paperwork_monitoring(self) -> object:
        return 1000 * u.CAD / u.farm / u.year

    @computed_field
    def onsite_monitoring(self) -> object:
        return 3000 * u.CAD / u.farm / u.year

    @property
    def income_tax_rate(self) -> float:
        return .25

    @computed_field
    def research(self) -> dict[str, str]:
        return {}

    def on_add_project(self, state):

        with state.requiring_current(self) as ctx:
            # TODO: for each type of cattle, for each province
            # will be written by Strategy
            ctx.bovine_population_fraction_on_bovaer = sts.SparseTimeSeries(
                default_value=0 * u.dimensionless, t_unit=u.years)

        with state.defining(self) as ctx:
            ctx.bovaer_monitoring_admin = sts.SparseTimeSeries(
                default_value=0 * u.CAD / u.farm / u.year,
                t_unit=u.year)
            ctx.bovaer_monitoring_onsite = sts.SparseTimeSeries(
                default_value=0 * u.CAD / u.farm / u.year,
                t_unit=u.year)
            for pt in PT:
                state.register_subsidy_factor(
                    pt=pt,
                    driver='Cattle Operations',
                    sts_key='bovaer_monitoring_admin',
                    program='Bovaer Subsidy',
                    reason='Monitoring - Administration')
                state.register_subsidy_factor(
                    pt=pt,
                    driver='Cattle Operations',
                    sts_key='bovaer_monitoring_onsite',
                    program='Bovaer Subsidy',
                    reason='Monitoring - Onsite')

        # TODO: revisit after switching from step() to fill()
        # so the start date will be based on where inputs leave off
        return state.stashes['Cattle_Population_AR'].t_step_start

    def step(self, state, current):
        bovaer_onsite_rate = self.onsite_monitoring
        bovaer_admin_rate = self.paperwork_monitoring

        current.bovaer_monitoring_admin = (
            bovaer_admin_rate
            * current.bovine_population_fraction_on_bovaer)
        current.bovaer_monitoring_onsite = (
            bovaer_onsite_rate
            * current.bovine_population_fraction_on_bovaer)
        return state.t_now + 1 * u.year

    def annual_scan_init(self, new_carry, xs, years, constants, jrkey):
        pass

    def annual_scan_step(self, new_carry, y, x, year, carry, constants, outputs):
        # num_samples, cattle_types, PT 14
        heads = constants['latent_livestock_counts']
        farms_on_bovaer = (
                heads / 160
                * new_carry['bovine_population_fraction_on_bovaer'][:, None, None])

        farms_on_bovaer_ca = farms_on_bovaer.sum(axis=(1,2))
        y['bovaer_monitoring_admin_ca'] = (
                farms_on_bovaer_ca * self.paperwork_monitoring.magnitude)
        y['bovaer_monitoring_onsite_ca'] = (
                farms_on_bovaer_ca * self.onsite_monitoring.magnitude)

        y['bovaer_monitoring_admin_ca_tax'] = (
                y['bovaer_monitoring_admin_ca'] * self.income_tax_rate)
        y['bovaer_monitoring_onsite_ca_tax'] = (
                y['bovaer_monitoring_onsite_ca'] * self.income_tax_rate)


class Bovaer_Farm_Subsidy(Barrier):

    @computed_field
    def short_description(self) -> str:
        return f"""Pay farmers {self.subsidy_rate} to administer Bovaer."""

    @computed_field
    def description(self) -> str:
        return f"""Model a tax-funded government subsidy program to pay
        cattle farmers {self.subsidy_rate} to administer Bovaer to all
        cattle on their farms, and comply with monitoring protocols.
        It is assumed that this subsidy amount is deductible, not
        subject to income tax.
        """

    @computed_field
    def subsidy_rate(self) -> object:
        return 5000 * u.CAD / u.farm / u.year

    def on_add_project(self, state):

        with state.requiring_current(self) as ctx:
            # TODO: for each type of cattle, for each province
            # will be written by Strategy
            ctx.bovine_population_fraction_on_bovaer = sts.SparseTimeSeries(
                default_value=0 * u.dimensionless, t_unit=u.years)

        with state.defining(self) as ctx:
            ctx.bovaer_farm_subsidy = sts.SparseTimeSeries(
                default_value=0 * u.CAD / u.farm / u.year,
                t_unit=u.year)
            for pt in PT:
                state.register_subsidy_factor(
                    pt=pt,
                    driver='Cattle Operations',
                    sts_key='bovaer_farm_subsidy',
                    program='Bovaer Subsidy',
                    reason='Farm Subsidy')

        # TODO: revisit after switching from step() to fill()
        # so the start date will be based on where inputs leave off
        return state.stashes['Cattle_Population_AR'].t_step_start

    def step(self, state, current):
        bovaer_cost_rate = self.subsidy_rate
        current.bovaer_farm_subsidy = (
            bovaer_cost_rate
            * current.bovine_population_fraction_on_bovaer)
        return state.t_now + 1 * u.year

    def annual_scan_init(self, new_carry, xs, years, constants, jrkey):
        n_samples = constants['sigma_ca'].shape[0]
        new_carry['bovine_population_fraction_on_bovaer'] = jnp.zeros(n_samples)

    def annual_scan_step(self, new_carry, y, x, year, carry, constants, outputs):
        try:
            have_money = carry['tax_funded_budget_for_bovaer'] > 0
        except KeyError:
            have_money = False
        new_carry['bovine_population_fraction_on_bovaer'] = jnp.where(
                have_money,
                new_carry['max_fraction_of_cattle_on_bovaer'],
                carry['bovine_population_fraction_on_bovaer'])

        # num_samples, cattle_types, PT 14
        heads = constants['latent_livestock_counts']
        cattle_per_farm = 160 # look this up somewhere
        subsidy_per_head = self.subsidy_rate.magnitude / cattle_per_farm

        bovaer_heads_pt = (
                heads.sum(axis=1)
                * new_carry['bovine_population_fraction_on_bovaer'][:, None])

        y['bovaer_farm_subsidy_pt'] = subsidy_per_head * bovaer_heads_pt[:, :13]
        y['bovaer_farm_subsidy_ca'] = subsidy_per_head * bovaer_heads_pt.sum(axis=1)

        y['bovaer_farm_subsidy_pt_tax'] = y['bovaer_farm_subsidy_pt'] * .2
        y['bovaer_farm_subsidy_ca_tax'] = y['bovaer_farm_subsidy_ca'] * .2

class Bovaer_Purchase_Cost(Barrier):
    """
    Define subsidy rates for the purchase cost of Bovaer
    """

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

        with state.requiring_current(self) as ctx:
            # TODO: for each type of cattle, for each province
            # will be written by Strategy
            ctx.bovine_population_fraction_on_bovaer = sts.SparseTimeSeries(
                default_value=0 * u.dimensionless, t_unit=u.years)

        for livestock, cost in self.bovaer_cost.items():
            sts_key = f'bovaer_cost_{livestock.value}'
            ts = sts.SparseTimeSeries(
                identifier=sts_key,
                default_value=cost * 0,
                t_unit=u.years)
            state.declare_sts(self, ts, write=True)
            for pt in PT:
                state.register_subsidy_factor(
                    pt=pt,
                    driver=livestock,
                    sts_key=sts_key,
                    program='Bovaer Subsidy',
                    reason='Bovaer Cost')

        # TODO: revisit after switching from step() to fill()
        # so the start date will be based on where inputs leave off
        return state.stashes['Cattle_Population_AR'].t_step_start

    def step(self, state, current):
        current.bovine_population_fraction_on_bovaer
        for livestock, cost in self.bovaer_cost.items():
            sts_key = f'bovaer_cost_{livestock.value}'
            setattr(current, sts_key, cost * current.bovine_population_fraction_on_bovaer)
        return state.t_now + 1 * u.year

    def annual_scan_init(self, new_carry, xs, years, constants, jrkey):
        pass

    def annual_scan_step(self, new_carry, y, x, year, carry, constants, outputs):
        # num_samples, cattle_types, PT 14
        heads = constants['latent_livestock_counts']

        daily_bovaer_cost = [self.bovaer_cost[lt].magnitude for lt in Livestock_nonsums]
        annual_bovaer_cost = jnp.array(daily_bovaer_cost) * 365

        bovaer_costs_pt = (
                (heads * annual_bovaer_cost[:, None]).sum(axis=1)
                * new_carry['bovine_population_fraction_on_bovaer'][:, None])

        y['bovaer_cost_pt'] = bovaer_costs_pt[:, :13]
        y['bovaer_cost_ca'] = bovaer_costs_pt.sum(axis=1)


from .strategies.strategy2 import Scale_Bovaer
