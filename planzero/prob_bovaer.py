import sys
from functools import cache as memcache

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.lax import scan
from numpyro.infer import MCMC, NUTS
from pydantic import computed_field

from . import model_db, nir2025, sts
from .eccc_nir_annex3p4 import table_A3p4_11
from .enums import GHG, PT, IPCC_Sector
from .ghgvalues import GWP_100
from .sc_3210013001 import (
    FarmType,
    Livestock,
    Livestock_nonsums,
    SurveyDate,
    number_of_cattle_by_class_and_farm_type_combined_surveys,
)
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
def jnp_data(farm_type):
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
    return jnp.array(data)


@memcache
def jnp_emission_factors(farm_type):
    years_of_interest = sorted_years(farm_type)
    year_to_idx = {year: i for i, year in enumerate(sorted_years(farm_type))}
    table = table_A3p4_11()
    rval = np.zeros((len(years_of_interest), len(Livestock_nonsums)))
    for jj, livestock in enumerate(Livestock_nonsums):
        for ii, year in enumerate(years_of_interest):
            amt = table[livestock].query(year * u.years)
            rval[year_to_idx[year], jj] = amt.to(u.kg_CH4 / u.cattle / u.year).magnitude
    return jnp.array(rval)


@memcache
def enteric_ch4_ktCO2e_scale():
    sector = IPCC_Sector.Enteric_Fermentation
    ghg = GHG.CH4

    arr_pt, arr_ca = nir2025.ktCO2e_dense_w_nan()
    jnp_pt = jnp.array(
        arr_pt[nir2025.idx_of_sector[sector],
               nir2025.idx_of_ghg[ghg]],
        dtype='float64')
    jnp_ca = jnp.array(
        arr_ca[nir2025.idx_of_sector[sector],
               nir2025.idx_of_ghg[ghg]],
        dtype='float64')
    scale = max([
        np.sqrt(np.mean(jnp_ca ** 2)),
        np.sqrt(np.nanmean(np.nansum(jnp_pt ** 2, axis=0))),
        1.0,
    ])
    return scale


inference_keys = [
        #'PT_emissions_ktCO2e',
        'latent_emission_factors',
        'latent_scaled_livestock_counts',
        'latent_scaled_livestock_counts_sigma',
        'latent_livestock_counts',
        'sigma_ca',
        'sigma_pt']


def weighted_constant_model(
        livestock_counts,
        nir_emission_factors,
        data_years,
        start_year_inclusive,
        end_year_exclusive,
        scaled_ktCO2e_pt,
        scaled_ktCO2e_ca,
        scaled_ktCO2e_ca_scale,
        weights,
        ):
    num_PTs = len(PT) # 14, including PT.XX
    num_livestock_types = len(Livestock_nonsums)
    valid_year_mask = (start_year_inclusive <= data_years) & (data_years < end_year_exclusive)

    debug = False

    if 1:
        livestock_scale = jnp.nanmax(livestock_counts)
        latent_scaled_livestock_counts = numpyro.sample(
                "latent_scaled_livestock_counts",
                dist.LogNormal(-1, 2).expand((num_livestock_types, num_PTs,)))

        latent_scaled_livestock_counts_sigma = numpyro.sample(
                "latent_scaled_livestock_counts_sigma",
                dist.LogNormal(-1, 1).expand((num_livestock_types, num_PTs,)))

        # explain head counts
        numpyro.sample(
                "obs_scaled_headcounts",
                dist.Normal(latent_scaled_livestock_counts,
                            latent_scaled_livestock_counts_sigma, #0.01,
                            ).mask(valid_year_mask[:, None, None]),
                obs=jnp.where(
                    valid_year_mask[:, None, None],
                    livestock_counts / livestock_scale, 0))

        numpyro.deterministic(
                "latent_livestock_counts",
                latent_scaled_livestock_counts * livestock_scale)

    if 1:
        latent_emission_factors = numpyro.sample(
                "latent_emission_factors",
                dist.Normal(
                    jnp.array([130, 77, 120.0, 120.0, 90, 52, 45, 44.0]),
                    jnp.array([ 10,  2,   5,     5.0,  4,  5,  5,  1.0])))

        # explain emission factors
        numpyro.sample(
                "obs_emission_factors",
                dist.Normal(latent_emission_factors, 20)\
                        .expand((len(data_years), num_livestock_types))\
                        .mask(valid_year_mask[:, None]),
                obs=jnp.where(valid_year_mask[:, None], nir_emission_factors, 0))

    if 1:
        # Strongly consider e.g. jpu to add units support with jax
        # https://github.com/dfm/jpu
        PT_emissions_kg_CH4 = (
                latent_emission_factors[:, None]
                * latent_scaled_livestock_counts
                * livestock_scale
                ).sum(axis=0)

        PT_emissions_ktCO2e = (
                PT_emissions_kg_CH4
                * (
                    GWP_100[GHG.CH4].magnitude # CH4 -> CO2e
                    / 1_000_000 # kg -> kt
                    ))

        # now act like the standard NIR-estimation model
        n_regions = 13
        # trim out the PT.XX region (could a better approach be implemented?)

        if debug:
            numpyro.deterministic("PT_emissions_ktCO2e", PT_emissions_ktCO2e)
        mu = PT_emissions_ktCO2e[:n_regions] / scaled_ktCO2e_ca_scale

        sigma_pt = numpyro.sample("sigma_pt",
                                  dist.LogNormal(-1.0, 0.7).expand((n_regions,)))
        sigma_ca = numpyro.sample("sigma_ca",
                                  dist.LogNormal(-1.0, 0.7))

        valid_pt = jnp.isfinite(scaled_ktCO2e_pt)
        dist_pt = dist.Normal(mu, sigma_pt).mask(valid_pt)
        obs_pt = jnp.where(valid_pt, scaled_ktCO2e_pt, 0)
        obs_ca = scaled_ktCO2e_ca

        # explain provincial and territorial methane emissions
        with numpyro.handlers.scale(scale=jnp.array(weights).reshape(-1, 1, 1)):
            numpyro.sample("obs_pt", dist_pt, obs=obs_pt)

        # explain national sectoral methane emissions
        with numpyro.handlers.scale(scale=jnp.array(weights).reshape(-1, 1)):
            numpyro.sample("obs_ca",
                           dist.Normal(jnp.sum(mu), sigma_ca),
                           obs=obs_ca)
        if debug:
            numpyro.deterministic("debug_est_ca", jnp.sum(mu))
            numpyro.deterministic("debug_obs_ca", obs_ca)

def inference():
    farm_type = FarmType.AllCattle
    heads = jnp_data(farm_type)
    nir_emission_factors = jnp_emission_factors(farm_type)
    jnp_sorted_years = jnp.array(sorted_years(farm_type))

    sector = IPCC_Sector.Enteric_Fermentation
    ghg = GHG.CH4
    rng_key = jrandom.key(1234)

    scale = enteric_ch4_ktCO2e_scale()
    NIR_emission_sample_size = 100
    real_PTs = [pt for pt in PT if pt != PT.XX]
    rng_key, pt_sample, ca_sample = nir2025.sample_pt_ca(
            rng_key,
            sample_size=NIR_emission_sample_size,
            years=range(1990, 1990 + 33),
            PTs=real_PTs,
            sector=sector,
            ghg=ghg)

    weights = jnp.array(
            [1.0 / NIR_emission_sample_size]
            * NIR_emission_sample_size)

    scaled_ca = jnp.array(ca_sample / scale)
    scaled_pt = jnp.array(pt_sample / scale)

    mcmc = MCMC(NUTS(weighted_constant_model),
                num_warmup=200,
                thinning=1,
                num_samples=500)

    rng_key, rng_key_ = jrandom.split(rng_key)
    mcmc.run(rng_key_,
             livestock_counts=heads,
             nir_emission_factors=nir_emission_factors,
             data_years=jnp_sorted_years,
             start_year_inclusive=1990,
             end_year_exclusive=2023,
             scaled_ktCO2e_pt=scaled_pt,
             scaled_ktCO2e_ca=scaled_ca,
             scaled_ktCO2e_ca_scale=scale,
             weights=weights,
            )
    mcmc.print_summary()
    return mcmc


def cached_inference(recompute=False):
    model_id = model_db.stable_hash(
            str(('prob_bovaer', 1)))
    component_id = model_db.stable_hash(
            str(('enteric_fermentation', model_id)))

    if not recompute:
        grouped_samples = model_db.load_ndarray_group(
            model_id=model_id,
            component_id=component_id,
            group_id='grouped_samples')
        if len(grouped_samples) == len(inference_keys):
            return grouped_samples

        try:
            # re-populate the db registry from files on disk
            model_db.delete_ndarray_group(
                component_id=component_id,
                group_id='grouped_samples',
                )
            grouped_samples = model_db.index_ndarray_group(
                model_id=model_id,
                component_id=component_id,
                group_id='grouped_samples',
                ndarray_d_keys=inference_keys)
            return grouped_samples
        except OSError:
            pass # at least one file is missing, probably all of them

    mcmc = inference()
    grouped_samples = mcmc.get_samples(group_by_chain=True)
    assert len(grouped_samples) == len(inference_keys)
    model_db.delete_ndarray_group(
        component_id=component_id,
        group_id='grouped_samples',
        )
    model_db.save_ndarray_group(
        model_id=model_id,
        component_id=component_id,
        group_id='grouped_samples',
        ndarray_d=grouped_samples)
    return grouped_samples


def samples_from_grouped_samples(grouped_samples):
    sample_size = None
    for gs in grouped_samples.values():
        assert gs.shape[0] == 1
        if sample_size is None:
            sample_size = gs.shape[1]
        else:
            assert sample_size == gs.shape[1]
    samples = {key: gs[0] for key, gs in grouped_samples.items()}
    return samples


from .barriers import Barrier

class Cattle_Population_Static_Normal(Barrier):
    """Static Normal estimates of cattle populations
    and enteric fermentation emission factors, to explain three
    things:
    (1) cattle counts,
    (2) NIR2025 enteric fermentation emission factors,
    (3) NIR2025 overall enteric fermentation emissions (including all livestock).
    Estimates are made for each of 8 cattle populations (beef cows, dairy cows, beef heifers for replacement, beef heifers for slaughter, dairy heifers, calves, steers, and bulls),
    and each province and territory.
    </p>
    <p>
    """

    @computed_field
    def short_description(self) -> str:
        return "Static model of cattle population"

    def annual_scan_init(self, initial_carry, xs, years, constants):
        grouped_samples = cached_inference()
        samples = samples_from_grouped_samples(grouped_samples)
        # (num_samples, cattle_types, PT 14)
        constants['latent_livestock_counts'] = samples['latent_livestock_counts']
        constants['latent_emission_factors'] = samples['latent_emission_factors']
        constants['sigma_ca'] = samples['sigma_ca']
        constants['sigma_pt'] = samples['sigma_pt']
        constants['enteric_ch4_ktCO2e_scale'] = enteric_ch4_ktCO2e_scale()

    def annual_scan_step(self, new_carry, y, x, year, carry, constants, outputs):
        pass

    @property
    def posts_developing_this_page(self) -> list[str]:
        return ['ProbabilisticBovaer']


class InitialCarry:

    _dict: dict[str, jnp.ndarray]  # variable name -> initial variable value
    _setdefaults: dict[str, str]  # variable name -> responsible dynamic element name
    _setitems: dict[str, str]  # variable name -> responsible dynamic element name
    viewer_name: str|None  # name of dynamic element viewing this InitialCarry

    def __init__(self, viewer_name=None, _dict=None, _setdefaults=None, _setitems=None):
        self._dict = {} if _dict is None else _dict
        self._setdefaults = {} if _setdefaults is None else _setdefaults
        self._setitems = {} if _setitems is None else _setitems
        self.viewer_name = viewer_name

    def check_compatibility(self, item, value):
        if item in self._dict:
            ref = self._dict[item]
            if isinstance(ref, (int, float, bool)):
                assert type(ref) == type(value), (ref, value)
            else:
                assert value.shape == ref.shape, (item, ref.shape, value.shape)
                assert value.dtype == ref.dtype, (item, ref.dtype, value.dtype)
        else:
            return True

    def __setitem__(self, item:str, value:jnp.ndarray):
        # it's an error to set the same item from multiple places
        # because there can only be one definition of an item.
        assert item not in self._setitems, (item,)
        self.check_compatibility(item, value)

        # once an item has been set, its setdefaults are all forgotten
        # and subsequent setdefaults won't change item ownership
        assert self.viewer_name is not None
        self._setitems[item] = self.viewer_name
        if item in self._setdefaults:
            del self._setdefaults[item]
        self._dict[item] = value

    def setdefault(self, item, value):
        self.check_compatibility(item, value)
        assert self.viewer_name is not None
        if item in self._setitems:
            # if another dynamic element has already called setitem
            # then let it be.
            pass
        else:
            # it's okay to setdefault multiple times, because this
            # class will track the *first* set-default call, and use
            # that one (arbitrarily) as the working definition of the item.
            self._setdefaults.setdefault(item, self.viewer_name)
            self._dict.setdefault(item, value)

    def view(self, viewer_name):
        assert self.viewer_name is None
        return InitialCarry(
                viewer_name=viewer_name,
                _dict=self._dict,
                _setdefaults=self._setdefaults,
                _setitems=self._setitems,
                )

    def outputs(self, viewer_name):
        rval = {key: owner for key, owner in self._setitems.items()
                if owner == viewer_name}
        rval.update({key: owner for key, owner in self._setdefaults.items()
                     if owner == viewer_name})
        return rval

    def owner(self, item):
        setitem_owner = self._setitems.get(item) == self.viewer_name
        setdefault_owner = self._setdefaults.get(item) == self.viewer_name
        assert not (setitem_owner and setdefault_owner), (item, self.viewer_name)
        rval = setitem_owner or setdefault_owner
        assert rval, item
        return rval

    def items(self):
        return self._dict.items()

    def get(self, item, default_value:jnp.ndarray) -> jnp.ndarray:
        # this function will depend on the order of dynamic element initialization
        # so think carefully about how to handle it.
        raise NotImplementedError(item)

    def __getitem__(self, item) -> jnp.ndarray:
        # TODO: record that this access was attempted,
        # maybe only if it was successful?
        # See also get()
        try:
            return self._dict[item]
        except KeyError:
            raise NotImplementedError(item)


class NewCarry:
    def __init__(self, initial_carry: InitialCarry, new_carry:dict, elem:str, carry:dict):
        self.initial_carry = initial_carry
        self.new_carry = new_carry
        self.carry = carry
        self.viewer_name = elem

    def __setitem__(self, item, value):
        owner = self.initial_carry.owner(item)
        if owner == self.viewer_name:
            self.new_carry[item] = value
        else:
            assert 0, f"element {self.viewer_name} doesn't have write access to {item}, which is owned by {owner}"

        # TODO: verify that if a dynelem is writing

    def setdefault(self, item, default_value):
        raise RuntimeError("avoid using setdefault on new_carry dictionary,"
                           f" use `if '{item}' in outputs:` instead")

    def get(self, item, default_value):
        # this function will depend on the order of dynamic element initialization
        # so think carefully about how to handle it.
        raise NotImplementedError(item)

    def __getitem__(self, item):
        # TODO: record that this access was attempted,
        # maybe only if it was successful?
        # See also get()
        try:
            return self._dict[item]
        except KeyError:
            raise NotImplementedError(item)


class Carry:
    def __init__(self, initial_carry: InitialCarry, carry:dict, elem:str):
        self.initial_carry = initial_carry
        self.carry = carry
        self.viewer_name = elem


class RolloutResult:

    dct:dict

    def __init__(self, dct:dict):
        self.dct = dct

    def __getitem__(self, key):
        return self.dct[key]

    def total_nbytes(self):
        nbytes = 0
        def size_fn(val):
            if isinstance(val, (jnp.ndarray, np.ndarray)):
                rval = val.nbytes
            elif isinstance(val, (int, float, None)):
                # this shouldn't add up to much, so we overestimate
                # to be conservative
                rval = sys.getsizeof(val)
            else:
                raise NotImplementedError(val)
            return rval

        for key in ['ys', 'xs', 'final_carry']:
            try:
                for val in self.dct[key].values():
                    nbytes += size_fn(val)
            except AttributeError as err:
                err.add_note(key)
                raise
        for key in ['constants', 'initial_carry',]:
            for val in self.dct[key]._dict.values():
                nbytes += size_fn(val)
        return nbytes

def batch_rollout_elements(
        elements: dict[str, object],
        n_samples:int,
        ) -> RolloutResult:
    years = jnp.arange(1990, 2050 + 1) # TODO: param

    initial_carry = InitialCarry()
    xs = InitialCarry()
    constants = InitialCarry()

    constants._dict['n_samples'] = n_samples


    for name, elem in elements.items():
        elem.annual_scan_init(
                initial_carry.view(name),
                xs.view(name),
                years,
                constants.view(name))

    ys_ics = []
    nc_ics = []

    def scan_step(carry, x_curtime):
        x, curtime = x_curtime
        y = InitialCarry()
        new_carry = InitialCarry()
        ys_ics.append(y)
        nc_ics.append(new_carry)
        for name, elem in elements.items():
            elem.annual_scan_step(
                    new_carry.view(name),
                    y.view(name),
                    x,
                    curtime,
                    carry,
                    constants._dict,
                    outputs=initial_carry.outputs(name))
        return new_carry._dict, y._dict

    final_carry, ys = scan(
            scan_step,
            initial_carry._dict,
            (xs._dict, years))
    if 0:
        for key, val in final_carry.items():
            if 'float' in str(val.dtype) or 'int' in str(val.dtype):
                print(key, val.shape, val.dtype, val.min(), val.max())
            else:
                # might be jrandom key
                print(key, val.shape, val.dtype)
        for key, val in ys.items():
            print(key, val.shape, val.dtype, val.min(), val.max())
    assert len(ys_ics) == 1
    return RolloutResult({
            'ys': ys,
            'xs': xs._dict,
            'final_carry': final_carry,
            'years': years,
            'initial_carry': initial_carry,
            'constants': constants,
            'xs_ic': xs,
            'ys_ic': ys_ics[-1],
            'nc_ic': nc_ics[-1],
            })


def batch_rollout_barriers(barriers, strategies, elements=None, n_samples=500):
    if elements is None:
        elements = {}
        # loop is preferred to update() in order to show the name in case of duplication
        for d in (barriers, strategies):
            for name, elem in d.items():
                assert name not in elements, name
                elements[name] = elem
    return batch_rollout_elements(
            elements,
            n_samples=n_samples)



def main_debug():
    cached_inference(recompute=True)
    #batch_rollout_barriers()


if __name__ == '__main__':
    main_debug()
