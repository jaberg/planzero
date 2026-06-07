import array
import os

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from scipy.optimize import minimize

# Enable Float64 for Gaussian Process numerical stability
from jax import config
config.update("jax_enable_x64", True)


from . import ipcc_canada
from . import ghgvalues
from .base import State, DynamicElement
from .enums import (
    IPCC_Sector,
    IPCC_Sector_from_catpath_with_whitespace,
    PT,
    GHG)
from .objtensor import ObjectTensor
from .ureg import u, kt_by_ghg
from .sts import SparseTimeSeries, STS
from .my_functools import cache



nir2025_year_ints = np.arange(1990, 2024)


def load_inventory():
    """Return inventory as DataFrame with standard pre-processing
    """
    inv = pd.read_csv(os.path.join(os.environ['PLANZERO_DATA'], 'EN_GHG_IPCC_Can_Prov_Terr.csv'))
    assert ('kt',) == inv['Unit'].unique()

    inv['CategoryPathWithWhitespace'] = (
        inv['Category'].fillna('').astype(str) + '/' +
        inv['Sub-category'].fillna('').astype(str) + '/' +
        inv['Sub-sub-category'].fillna('').astype(str)
    ).str.rstrip('/')
    assert not inv['CategoryPathWithWhitespace'].isna().any()

    # ensure that 'x' means NaN
    non_strs = set()
    for key in ['CO2', 'CH4', 'CH4 (CO2eq)', 'N2O',
                 'N2O (CO2eq)', 'HFCs', 'PFCs', 'SF6', 'NF3', 'CO2eq']:
        for strval in inv[key].values:
            try:
                assert np.isfinite(float(strval))
            except:
                non_strs.add(strval)
    assert non_strs == {'x'}

    for key in ['CO2', 'CH4', 'CH4 (CO2eq)', 'N2O',
                 'N2O (CO2eq)', 'HFCs', 'PFCs', 'SF6', 'NF3', 'CO2eq']:
        inv[key] = inv[key].replace('x', float('nan')).astype(float)
    del key

    # Year could have been cast to int, they're all whole numbers.
    inv['Year'] = inv['Year'].astype(float)
    return inv


idx_of_sector = {sector: ii for ii, sector in enumerate(IPCC_Sector)}
idx_of_ghg = {ghg: ii for ii, ghg in enumerate(GHG)}
idx_of_pt = {pt: ii for ii, pt in enumerate(PT)}
idx_of_yr = {yr: ii for ii, yr in enumerate(nir2025_year_ints)}


@cache
def ktCO2e_dense_w_nan():
    """Return the NIR-2025 in the form of two numpy arrays:
    * arr_pt (IPCC_Sector idx, GHG idx, PT idx, year since 1990)
    * arr_ca (IPCC_Sector idx, GHG idx, year since 1990)

    In arr_pt, censored data is represented with NaN.
    In arr_ca, there is no censored data.
    """
    inv = load_inventory()

    arr_pt = np.zeros((len(IPCC_Sector),
                       len(GHG),
                       len(PT) - 1,
                       len(nir2025_year_ints)),
                      dtype=np.float32)
    arr_ca = np.zeros((len(IPCC_Sector), len(GHG), len(nir2025_year_ints)), dtype=np.float32)

    non_agg = inv[inv['Total'] != 'y']
    for catpathww, nonagg_catpath in non_agg.groupby('CategoryPathWithWhitespace'):
        ipcc_sector = IPCC_Sector_from_catpath_with_whitespace[catpathww]

        for region, region_df in nonagg_catpath.groupby('Region'):
            if region.lower() == 'canada':
                pt = None
            elif region == 'Northwest Territories and Nunavut':
                pt = PT.XX
            else:
                pt = PT(region)

            for ghg in GHG:
                if ghg in [GHG.CH4, GHG.N2O]:
                    values = region_df[f'{ghg.value} (CO2eq)'].values
                else:
                    values = region_df[ghg.value].values
                years = region_df['Year'].values

                for year, val in zip(years, values):
                    if np.isfinite(val):
                        if pt is None:
                            arr_ca[idx_of_sector[ipcc_sector],
                                   idx_of_ghg[ghg],
                                   idx_of_yr[year]] = val
                        elif pt is PT.XX:
                            arr_pt[idx_of_sector[ipcc_sector],
                                   idx_of_ghg[ghg],
                                   idx_of_pt[PT.NT],
                                   idx_of_yr[year]] = float('nan')
                            arr_pt[idx_of_sector[ipcc_sector],
                                   idx_of_ghg[ghg],
                                   idx_of_pt[PT.NU],
                                   idx_of_yr[year]] = float('nan')
                        else:
                            arr_pt[idx_of_sector[ipcc_sector],
                                   idx_of_ghg[ghg],
                                   idx_of_pt[pt],
                                   idx_of_yr[year]] = val
                    else:
                        assert pt is not None
                        arr_pt[idx_of_sector[ipcc_sector],
                               idx_of_ghg[ghg],
                               idx_of_pt[pt],
                               idx_of_yr[year]] = float('nan')

    return arr_pt, arr_ca


def discrepancies_by_sector_ghg(arr_pt, arr_ca, idx_of_sector, idx_of_ghg):
    """Return dictionary
        (sector, ghg) -> [discrepancy for 1990, for 1991, ..., for 2023]
    """
    rval = {}
    for sector, ii in idx_of_sector.items():
        for ghg, jj in idx_of_ghg.items():
            arr_pt_ij = arr_pt[ii, jj, :, :]
            arr_ca_ij = arr_ca[ii, jj, :]
            discrepancy_by_yr = arr_ca_ij - np.nansum(arr_pt_ij, axis=0)
            rval[sector, ghg] = discrepancy_by_yr
    return rval


def idealized_NIR2025_objective(params, L, obs_data, mask, noise_std, jnp_ca,
                                PT_mean, PT_var, CA_mean, CA_var,
                                national_preference
                               ):
    """Pure JAX description of the loss function."""
    n_timesteps, n_PTs = obs_data.shape
    vparams = params.reshape((n_timesteps, n_PTs))

    prior_loss = 0.5 * jnp.sum(params ** 2) * .1

    z = (L @ vparams) * jnp.sqrt(PT_var) + PT_mean

    # XXX assumes that all provincial emissions have the same sign in every year
    #     which in principle seems like a bug, but which in practice seems to
    #     be at least approximately true?
    CA_est = z.sum(axis=1)

    # Likelihood Loss (Masked safely without NaN propagation)
    CA_sqerr = (jnp_ca - CA_est) ** 2
    PT_sqerr = jnp.where(mask, (obs_data - z) ** 2, 0.0)
    likelihood_loss_CA = national_preference * CA_sqerr.sum() / CA_var
    likelihood_loss_PT = (PT_sqerr / PT_var).sum()
    loss = prior_loss + likelihood_loss_CA + likelihood_loss_PT
    return loss


idealized_NIR2025_loss_and_grad_fn = jax.jit(
    jax.value_and_grad(idealized_NIR2025_objective))


def idealized_sector_ghg(sector, ghg, block_pt, block_ca,
                         length_scale,
                         national_preference):
    # In future, the length scale can be parameterized, and optimized,
    # but just hard-code it for now, and make up a number.

    n_PTs, n_timesteps = block_pt.shape
    observed_data = block_pt.T.copy()  # (timesteps, n_PTs)

    # for each year, an equally-distributed portion of the discrepancy
    obs_noise_std = jnp.array(
        abs(np.nansum(block_pt, axis=0) - block_ca) / n_PTs + 1e-6)

    valid_mask_jnp = jnp.array(~np.isnan(observed_data))

    # Replace NaNs with 0.0 so they don't slow down JAX's gradient
    # calculations (neither the zeros nor NaNs should affect the result)
    clean_obs_jnp = jnp.array(
        np.where(np.isnan(observed_data), 0.0, observed_data))
    jnp_ca = jnp.array(block_ca) # 1D

    PT_mean = jnp.nanmean(observed_data, axis=0)
    # the prior variance in PT is tricky, because sometimes
    # all provinces report all zeros, but the national total
    # is a few Mt (e.g. Ammonia Production). The strategy here
    # is to define the prior to be very weak
    PT_var = jnp.maximum(
        jnp.nanvar(observed_data, axis=0) + jnp.var(jnp_ca),
        0.1) # epsilon floor
    assert PT_var.shape == (n_PTs,)
    CA_mean = jnp.mean(jnp_ca)
    CA_var = jnp.maximum(jnp.var(jnp_ca), 0.1)

    # L_jnp: the Cholesky factor matrix
    def rbf_kernel(t1, t2):
        return np.exp(-0.5 * (t1[:, None] - t2[None, :]) ** 2 / length_scale ** 2)
    for eps in [1e-8, 1e-7, 1e-6, 1e-5]:
        try:
            K = (rbf_kernel(nir2025_year_ints, nir2025_year_ints)
                 + 1e-6 * np.eye(n_timesteps))
            L_jnp = jnp.array(np.linalg.cholesky(K))
            break
        except Exception as err:
            print(err)
            continue
    else:
        raise RuntimeError('Matrix so bad it could not be fixed')

    initial_guess = np.zeros(n_PTs * n_timesteps)

    def scipy_solver_wrapper(params):
        """Wrapper that translates JAX arrays back into standard NumPy format
        for SciPy."""
        loss, grad = idealized_NIR2025_loss_and_grad_fn(
            params,
            L_jnp, clean_obs_jnp, valid_mask_jnp, obs_noise_std, jnp_ca,
            PT_mean, PT_var, CA_mean, CA_var,
            national_preference,
            )
        return float(loss), np.array(grad)

    res = minimize(
        scipy_solver_wrapper,
        initial_guess,
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': 500}
    )

    opt_v = res.x.reshape((n_timesteps, n_PTs))
    inferred_series = (L_jnp @ opt_v) * jnp.sqrt(PT_var) + PT_mean
    return inferred_series


@cache
def idealized_NIR2025(discrepancy_threshold_ktCO2e=10,
                      national_preference=10,
                      length_scale=.25,
                     ):
    """Return an idealized NIR2025-like dataset by maximizing likelihood
    of a latent model that reconciles national emissions totals
    across provinces and territories.
    """
    rval = np.zeros(
        (len(IPCC_Sector),
         len(GHG),
         len(PT) - 1, # don't count PT.XX
         len(nir2025_year_ints)))

    idx_of_sector = {sector: ii for ii, sector in enumerate(IPCC_Sector)}
    idx_of_ghg = {ghg: ii for ii, ghg in enumerate(GHG)}
    ktCO2e_pt, ktCO2e_ca = ktCO2e_dense_w_nan()
    discrepancies = discrepancies_by_sector_ghg(ktCO2e_pt, ktCO2e_ca,
                                                idx_of_sector, idx_of_ghg)
    for (sector, ghg), discrepancy_by_yr in discrepancies.items():
        sector_idx = idx_of_sector[sector]
        ghg_idx = idx_of_ghg[ghg]
        if discrepancy_by_yr.max() > discrepancy_threshold_ktCO2e:
            rval[sector_idx, ghg_idx] = idealized_sector_ghg(
                sector, ghg,
                ktCO2e_pt[sector_idx, ghg_idx],
                ktCO2e_ca[sector_idx, ghg_idx],
                length_scale=length_scale,
                national_preference=national_preference).T
        else:
            rval[sector_idx, ghg_idx] = ktCO2e_pt[sector_idx, ghg_idx]
            rval[sector_idx, ghg_idx][np.isnan(rval[sector_idx, ghg_idx])] = 0
            tmp_by_yr = np.sum(rval[sector_idx, ghg_idx], axis=0) + 1e-6
            rval[sector_idx, ghg_idx] *= ktCO2e_ca[sector_idx, ghg_idx] / tmp_by_yr
    return rval


@cache
def co2e_objtensors():
    """Return the NIR-2025 as a pair of ObjectTensors (rval_pt, rval_ca)

    The return values contain the mass of CO2e emitted in each sector, for
    each GHG, and for rval_pt, for each region.

    rval_pt's PT.XX region contains a differencing amount, where necessary so that
    the regional totals match the national total.

    The elements of the returned values are either 0 kt_CO2e constants, or else
    no-interpolation timeseries for year 1990 to 2023 inclusive.
    """

    inv = load_inventory()
    non_agg = inv[inv['Total'] != 'y']

    rval_pt = ObjectTensor.empty(IPCC_Sector, GHG, PT)
    rval_ca = ObjectTensor.empty(IPCC_Sector, GHG)

    rval_pt[:] = 0 * u.kt_CO2e
    rval_ca[:] = 0 * u.kt_CO2e

    for catpathww, nonagg_catpath in non_agg.groupby('CategoryPathWithWhitespace'):
        ipcc_sector = IPCC_Sector_from_catpath_with_whitespace[catpathww]

        for region, region_df in nonagg_catpath.groupby('Region'):
            if region.lower() == 'canada':
                pt = None
            elif region == 'Northwest Territories and Nunavut':
                pt = PT.XX
            else:
                pt = PT(region)

            for ghg in GHG:
                if ghg in [GHG.CH4, GHG.N2O]:
                    values = region_df[f'{ghg.value} (CO2eq)'].values
                else:
                    values = region_df[ghg.value].values
                years = region_df['Year'].values

                kt_by_yr = {int(year): float(val)
                            for year, val in zip(years, values)
                            if np.isfinite(val)}
                if kt_by_yr:
                    ts_zero = STS(
                        times=array.array('d', nir2025_year_ints),
                        t_unit=u.years,
                        values=array.array('d', [float('nan')] + [
                            kt_by_yr.get(yr, 0.0)
                            for yr in nir2025_year_ints]),
                        v_unit=u.kt_CO2e,
                        interpolation='no_interpolation')
                    if any(vv != 0 for vv in ts_zero.values[1:]):
                        if pt is None:
                            rval_ca[ipcc_sector, ghg] = ts_zero
                        else:
                            rval_pt[ipcc_sector, ghg, pt] = ts_zero
                else:
                    # we already set it to zero in initialization above
                    pass

    # now rectify the regional totals to make them match the national ones
    # by adding unassigned CO2e to the PT.XX region.
    # typically this is due to censored values in the provincial reports,
    # but it can be for other reasons too, such as differences in measurement
    # protocols or data sources.
    for sector in IPCC_Sector:
        for ghg in GHG:
            if isinstance(rval_ca[sector, ghg], STS):
                assert np.isfinite(rval_ca[sector, ghg].values[1:]).all()
            diff = rval_ca[sector, ghg] - rval_pt[sector, ghg].sum()
            if not isinstance(diff, STS):
                continue
            if abs(np.asarray(diff.values[1:])).max() < 5:  # kt_CO2e
                continue
            if sector in [IPCC_Sector.Non_Energy_Products_from_Fuels_and_Solvent_Use]:
                # Something's up with this one.
                # Look at e.g.
                #
                # inv[
                #    (inv['Total'] != 'y')
                #    & (inv['Year'] == 1990)
                #    & (inv['CategoryPathWithWhitespace'] == 'Non-Energy Products from Fuels and Solvent Use')
                #    ]
                #
                # Every province's emissions add up to
                # 12_000 kt, while the national total is listed as 5900.
                # It is the only IPCC sector whose provincial estimates exceed
                # the national one.
                #
                # Google Gemini suggested that the top-down national estimate
                # is more accurate because the provinces aren't positioned to
                # estimate this sector very well without double counting
                #
                # In this function, below, we set the PT.XX emissions as
                # negative.
                eps = None
            elif sector in [IPCC_Sector.Incineration_and_Open_Burning_Waste,
                            IPCC_Sector.Industrial_Wastewater_Treatment_and_Discharge]:
                eps = 1.1
                assert (np.asarray(diff.values[1:])
                        >= -eps).all(), (sector, ghg, diff)
            else:
                eps = 1e-3
                assert (np.asarray(diff.values[1:])
                        >= -eps).all(), (sector, ghg, diff)

            rval_pt[sector, ghg, PT.XX] += diff

    return rval_pt, rval_ca


class NIR2025(DynamicElement):

    # this year (inclusive) and previous ones are used by the class
    # set to e.g. 2020 to ignore data from years 2021 and beyond
    last_training_year: int = 3000

    skip_registered_sectors: bool = False

    sub_name: str = ''

    n_year_filter: int = 0

    filter_algo: str|None = None
    fit_start_year:int = 0

    def filter_ts_linear_filter(self, ts, ghg):
        assert self.n_year_filter > 0
        scale = 1.0 / ghgvalues.GWP_100[ghg].magnitude
        times = []
        values = [0]
        windows = []
        for ii, (yr, val) in enumerate(zip(ts.times, ts.values[1:])):
            if yr > self.last_training_year:
                break
            times.append(yr)
            n_prediction_years = 1 # TODO: parameterize this later
            window = ts.values[max(1, ii + 2 - self.n_year_filter - n_prediction_years):ii + 2]
            assert window[-1] == val
            #values.append(scale * np.mean(window[1:]))
            values.append(scale * val)
            if len(window) == (self.n_year_filter + n_prediction_years):
                if yr >= self.fit_start_year:
                    windows.append(window)

        if 1:
            from sklearn.decomposition import PCA
            from sklearn.linear_model import Ridge, RidgeCV, LinearRegression
            windows = np.asarray(windows)

            pca = PCA()
            pca.fit(windows[:, :-1])
            pcawin = pca.transform(windows[:-1, :-1])
            ymean = np.mean(windows[:-1, -1])
            ycen = windows[:-1, -1] - ymean
            ystd = np.std(ycen)
            eps = 1e-6

            model = RidgeCV(fit_intercept=True)
            model.fit(pcawin, ycen / (ystd + eps))
            #model.coef_[:] = .3333
            #print(model.coef_)
            times.append(times[-1] + 1)
            pred_pca = pca.transform([values[-self.n_year_filter:]])
            pred_y = float(model.predict(pred_pca)[0])
            values.append(scale * (ystd * pred_y + ymean))

        rval = STS(
            times=array.array('d', times),
            t_unit=u.years,
            values=array.array('d', values),
            v_unit=kt_by_ghg[ghg] / u.year,
            interpolation='current')
        return rval

    def filter_ts_average(self, ts, ghg):
        assert self.n_year_filter > 0
        scale = 1.0 / ghgvalues.GWP_100[ghg].magnitude
        times = []
        values = [0]
        for ii, (yr, val) in enumerate(zip(ts.times, ts.values[1:])):
            if yr > self.last_training_year:
                break
            times.append(yr)
            window = ts.values[max(1, ii + 2 - self.n_year_filter):ii + 2]
            assert window[-1] == val
            if yr > 2000:
                assert len(window) == 3
            values.append(scale * np.mean(window))

        rval = STS(
            times=array.array('d', times),
            t_unit=u.years,
            values=array.array('d', values),
            v_unit=kt_by_ghg[ghg] / u.year,
            interpolation='current')
        return rval


    def filter_ts(self, ts, ghg):
        if self.filter_algo == 'average':
            return self.filter_ts_average(ts, ghg)

        if self.filter_algo == 'linear_filter':
            return self.filter_ts_linear_filter(ts, ghg)

        assert self.filter_algo is None, self.filter_algo
        scale = 1.0 / ghgvalues.GWP_100[ghg].magnitude
        rval = STS(
            times=array.array('d', [
                yr for yr in ts.times
                if yr <= self.last_training_year]),
            t_unit=u.years,
            values=array.array('d', [0] + [
                scale * val
                for yr, val in zip(ts.times, ts.values[1:])
                if yr <= self.last_training_year]),
            v_unit=kt_by_ghg[ghg] / u.year,
            interpolation='current')
        return rval

    def on_add_project(self, state):

        driver_name = f'NIR-2025 Emissions Placeholder'
        state.declare_sts(
            self,
            sts=SparseTimeSeries(
                identifier=driver_name,
                default_value=1.0 * u.dimensionless,
                t_unit=u.year),
            write=True)
        for pt in PT:
            state.register_driver(
                pt=pt,
                driver=driver_name,
                sts_key=driver_name)

        if self.skip_registered_sectors:
            skip_sectors = {
                ipcc_sector_key
                for by_pt in state.registries['emission_factor'].values()
                for by_ipcc_sector in by_pt.values()
                for ipcc_sector_key in by_ipcc_sector}
        else:
            skip_sectors = {}

        co2e_pt, co2e_ca = co2e_objtensors()

        for (sector, ghg, pt), offset in co2e_pt.ravel_keys_offsets():
            obj = co2e_pt.buf[offset]
            if not isinstance(obj, STS):
                continue
            assert obj.t_unit == u.year
            assert obj.v_unit == u.kt_CO2e

            if sector in skip_sectors:
                continue

            sub_name = f'[{self.sub_name}]' if self.sub_name else ''
            name = f'{self.__class__.__name__}{sub_name} {ghg.value} from {sector.value} in {pt.value}'
            state.declare_sts(
                project=self,
                sts=self.filter_ts(obj, ghg),
                name=name,
                write=True)
            state.register_emission_factor(
                pt=pt,
                driver=driver_name,
                sts_key=name,
                ipcc_sector=sector,
                ghg=ghg)


def latest_data_prediction_algo(last_training_year, horizon):
    state = State(name=f'foo', t_start=1990 * u.years)
    state.add_project(NIR2025(last_training_year=last_training_year))
    state.run_until(horizon * u.years)
    return state


def rolling_average_prediction_algo(last_training_year, horizon, n_year_average):
    state = State(name=f'foo', t_start=1990 * u.years)
    state.add_project(
        NIR2025(
            last_training_year=last_training_year,
            n_year_filter=n_year_average,
            filter_algo='average',
            ))
    state.run_until(horizon * u.years)
    return state


def linear_prediction_algo(last_training_year, horizon, n_year_average, fit_start_year):
    state = State(name=f'foo', t_start=1990 * u.years)
    state.add_project(
        NIR2025(
            last_training_year=last_training_year,
            n_year_filter=n_year_average,
            filter_algo='linear_filter',
            fit_start_year=fit_start_year,
            ))
    state.run_until(horizon * u.years)
    return state


def NIR2025_EmissionsResults():
    state = State(
        name=f'foo',
        t_start=1990 * u.years)
    state.add_project(NIR2025())
    state.run_until(2024 * u.years)
    rval = state.compute_annual_emissions()
    return rval
