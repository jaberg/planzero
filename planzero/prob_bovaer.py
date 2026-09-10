from functools import cache as memcache

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
import jax.numpy as jnp
import jax.random as jrandom
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS


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


inference_keys = [
        #'PT_emissions_ktCO2e',
        'latent_emission_factors',
        #'latent_livestock_counts',
        'latent_scaled_livestock_counts',
        'sigma_ca',
        'sigma_pt']


def weighted_constant_model(
        livestock_counts,
        nir_emission_factors,
        data_years,
        start_year_inclusive,
        end_year_exclusive,
        jnp_pt,
        jnp_ca,
        jnp_ca_scale,
        weights,
        ):
    num_PTs = len(PT) # 14, including PT.XX
    num_livestock_types = len(Livestock_nonsums)
    valid_year_mask = (start_year_inclusive <= data_years) & (data_years < end_year_exclusive)

    if 1:
        livestock_scale = jnp.nanmax(livestock_counts)
        latent_scaled_livestock_counts = numpyro.sample(
                "latent_scaled_livestock_counts",
                dist.LogNormal(-1, 2).expand((num_livestock_types, num_PTs,)))

        # explain head counts
        numpyro.sample(
                "obs_scaled_headcounts",
                dist.Normal(latent_scaled_livestock_counts,
                            0.01,
                            ).mask(valid_year_mask[:, None, None]),
                obs=jnp.where(
                    valid_year_mask[:, None, None],
                    livestock_counts / livestock_scale, 0))

        #numpyro.deterministic(
                #"latent_livestock_counts",
                #latent_scaled_livestock_counts * livestock_scale)

    if 1:
        latent_emission_factors = numpyro.sample(
                "latent_emission_factors",
                dist.Normal(
                    jnp.array([130, 77, 120.0, 120.0, 90, 52, 45, 44.0]),
                    jnp.array([ 10,  2,   5,     5.0,  4,  5,  5,  1.0])))

        # explain emission factors
        numpyro.sample(
                "obs_emission_factors",
                dist.Normal(latent_emission_factors, 1)\
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

        #numpyro.deterministic("PT_emissions_ktCO2e", PT_emissions_ktCO2e)
        mu = PT_emissions_ktCO2e[:n_regions] / jnp_ca_scale

        sigma_pt = numpyro.sample("sigma_pt",
                                  dist.LogNormal(-1.0, 0.7).expand((n_regions,)))
        sigma_ca = numpyro.sample("sigma_ca",
                                  dist.LogNormal(-1.0, 0.7))

        valid_pt = jnp.isfinite(jnp_pt)
        dist_pt = dist.Normal(mu, sigma_pt).mask(valid_pt)
        obs_pt = jnp.where(valid_pt, jnp_pt / jnp_ca_scale, 0)
        obs_ca = jnp_ca / jnp_ca_scale

        # explain provincial and territorial methane emissions
        with numpyro.handlers.scale(scale=jnp.array(weights).reshape(-1, 1, 1)):
            numpyro.sample("obs_pt", dist_pt, obs=obs_pt)

        # explain national sectoral methane emissions
        with numpyro.handlers.scale(scale=jnp.array(weights).reshape(-1, 1)):
            numpyro.sample("obs_ca",
                           dist.Normal(jnp.sum(mu), sigma_ca),
                           obs=obs_ca)

def inference():
    farm_type = FarmType.AllCattle
    heads = jnp_data(farm_type)
    nir_emission_factors = jnp_emission_factors(farm_type)
    jnp_sorted_years = jnp.array(sorted_years(farm_type))
    arr_pt, arr_ca = nir2025.ktCO2e_dense_w_nan()

    sector = IPCC_Sector.Enteric_Fermentation
    ghg = GHG.CH4
    rng_key = jrandom.key(1234)

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
             jnp_pt=scaled_pt,
             jnp_ca=scaled_ca,
             jnp_ca_scale=scale,
             weights=weights,
            )
    mcmc.print_summary()
    return mcmc


def cached_inference():
    model_id = model_db.stable_hash(
            str(('prob_bovaer', 1)))
    component_id = model_db.stable_hash(
            str(('enteric_fermentation', model_id)))

    grouped_samples = model_db.load_ndarray_group(
        model_id=model_id,
        component_id=component_id,
        group_id='grouped_samples')
    if len(grouped_samples) == len(inference_keys):
        return grouped_samples

    try:
        # re-populate the db registry from files on disk
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
    model_db.save_ndarray_group(
        model_id=model_id,
        component_id=component_id,
        group_id='grouped_samples',
        ndarray_d=grouped_samples)
    return grouped_samples


def main_debug():
    inference()

if __name__ == '__main__':
    main_debug()
