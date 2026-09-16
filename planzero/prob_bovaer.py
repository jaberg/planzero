from functools import cache as memcache
from pydantic import computed_field

from . import sts
from .ureg import u
from .enums import IPCC_Sector, PT, GHG
from .eccc_nir_annex3p4 import table_A3p4_11
from .ghgvalues import GWP_100
from .sc_3210013001 import (
    FarmType, Livestock, Livestock_nonsums, SurveyDate,
    number_of_cattle_by_class_and_farm_type_combined_surveys)
from . import model_db, my_functools, nir2025

import numpy as np
try:
    import jax.numpy as jnp
    import jax.random as jrandom
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS
    from jax.lax import scan
except ImportError:
    pass


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
    """Simple static-normals estimate of Cattle populations
    for each cattle type, and each province and territory.
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


class InitialCarry:
    def __init__(self, elem=None, _dict=None, _setdefaults=None, _setitems=None):
        self._dict = {} if _dict is None else _dict
        self._setdefaults = {} if _setdefaults is None else _setdefaults
        self._setitems = {} if _setitems is None else _setitems
        self.elem = elem

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

    def __setitem__(self, item, value):
        # it's an error to set the same item from multiple places
        # because there can only be one definition of an item.
        assert item not in self._setitems, (item,)
        self.check_compatibility(item, value)

        # once an item has been set, its setdefaults are all forgotten
        # and subsequent setdefaults won't change item ownership
        self._setitems[item] = self.elem
        if item in self._setdefaults:
            del self._setdefaults[item]
        self._dict[item] = value

    def setdefault(self, item, value):
        self.check_compatibility(item, value)
        if item in self._setitems:
            # if another dynamic element has already called setitem
            # then let it be.
            pass
        else:
            # it's okay to setdefault multiple times, because this
            # class will track the *first* set-default call, and use
            # that one (arbitrarily) as the working definition of the item.
            self._setdefaults.setdefault(item, self.elem)
            self._dict.setdefault(item, value)

    def view(self, elem):
        assert self.elem is None
        return InitialCarry(
                elem=elem,
                _dict=self._dict,
                _setdefaults=self._setdefaults,
                _setitems=self._setitems,
                )

    def outputs(self, elem):
        rval = {key: owner for key, owner in self._setitems.items()
                if owner == elem}
        rval.update({key: owner for key, owner in self._setdefaults.items()
                     if owner == elem})
        return rval

    def owner(self, item):
        setitem_owner = self.initial_carry._setitems.get(item) == self.elem
        setdefault_owner = self.initial_carry._setdefaults.get(item) == self.elem
        assert setitem_owner is None or setdefault_owner is None
        rval = setitem_owner or setdefault_owner
        assert rval is not None
        return rval

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


class NewCarry:
    def __init__(self, initial_carry: InitialCarry, new_carry:dict, elem:str, carry:dict):
        self.initial_carry = initial_carry
        self.new_carry = new_carry
        self.carry = carry
        self.elem = elem

    def __setitem__(self, item, value):
        owner = self.initial_carry.owner(item)
        if owner == self.elem:
            self.new_carry[item] = value
        else:
            assert 0, f"element {self.elem} doesn't have write access to {item}, which is owned by {owner}"

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
        self.new_carry = new_carry
        self.elem = elem



def batch_rollout_barriers():
    from . import cattle
    from .strategies.strategy2 import Scale_Bovaer

    barriers = [
            Cattle_Population_Static_Normal(),
            cattle.Bovaer_Adoption_Limit(),
            cattle.Bovaer_Farm_Subsidy(),
            cattle.Bovaer_Production_Emission_Factors(),
            cattle.Cattle_Enteric_Emission_Rates_NIR2025_Bovaer(),
            cattle.Bovaer_Purchase_Cost(),
            cattle.Bovaer_Monitoring(),
            ]
    strategies = [
            Scale_Bovaer(),
            ]

    initial_carry = InitialCarry()
    xs = {}
    constants = {}

    years = jnp.arange(1990, 2050 + 1)

    for elem in barriers + strategies:
        elem.annual_scan_init(initial_carry.view(elem), xs, years, constants)

    def scan_step(carry, x_curtime):
        x, curtime = x_curtime
        y = {}
        new_carry = {}
        for elem in barriers + strategies:
            elem.annual_scan_step(new_carry, y, x, curtime, carry, constants,
                                  outputs=initial_carry.outputs(elem))
        return new_carry, y

    final_carry, ys = scan(scan_step, initial_carry._dict, (xs, years))
    if 0:
        for key, val in final_carry.items():
            if 'float' in str(val.dtype) or 'int' in str(val.dtype):
                print(key, val.shape, val.dtype, val.min(), val.max())
            else:
                # might be jrandom key
                print(key, val.shape, val.dtype)
        for key, val in ys.items():
            print(key, val.shape, val.dtype, val.min(), val.max())
    return {
            'ys': ys,
            'xs': xs,
            'final_carry': final_carry,
            'years': years,
            }


def main_debug():
    cached_inference(recompute=True)
    #batch_rollout_barriers()


if __name__ == '__main__':
    main_debug()
