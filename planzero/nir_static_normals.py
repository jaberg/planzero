"""
Example inference that's fast and represents a lower bound on accuracy.
"""
import datetime

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from scipy.special import logsumexp

from . import model_db, my_functools, nir2025
from .enums import GHG, PT, IPCC_Sector

model_family = 'StaticNormal'
model_version = 2
model_version_description = "Switching to distributional NIR data"


def weighted_constant_model(scaled_pt=None, scaled_ca=None, weights=1):
    n_regions = 13
    mu = numpyro.sample("mu", dist.Normal(0.0, 1.0).expand((n_regions,)))
    sigma_pt = numpyro.sample("sigma_pt",
                              dist.LogNormal(-1.0, 0.7).expand((n_regions,)))
    sigma_ca = numpyro.sample("sigma_ca",
                              dist.LogNormal(-1.0, 0.7))
    if scaled_pt is None:
        dist_pt = dist.Normal(mu, sigma_pt)
        obs_pt = None
    else:
        valid_pt = jnp.isfinite(scaled_pt)
        dist_pt = dist.Normal(mu, sigma_pt).mask(valid_pt)
        obs_pt = jnp.where(valid_pt, scaled_pt, 0)

    with numpyro.handlers.scale(scale=jnp.array(weights).reshape(-1, 1, 1)):
        numpyro.sample("obs_pt", dist_pt, obs=obs_pt)

    with numpyro.handlers.scale(scale=jnp.array(weights).reshape(-1, 1)):
        numpyro.sample("obs_ca",
                       dist.Normal(jnp.sum(mu), sigma_ca),
                       obs=scaled_ca)


def model_id_from_data_cutoff(data_cutoff):
    model_id = 'model_{}'.format(
        model_db.stable_hash(
            str((model_family, model_version, data_cutoff))))
    return model_id


def touch_model(data_cutoff):
    with model_db.connect() as conn:
        cursor = conn.cursor()
        model_id = model_id_from_data_cutoff(data_cutoff)

        try:
            model_db.by_id('Model', model_id=model_id)
        except model_db.NoRecord:
            model_db.insert_model(
                cursor=cursor,
                model_id=model_id,
                family=model_family,
                version=model_version,
                version_description=model_version_description,
                data_cutoff=data_cutoff,
                )


def touch_components(data_cutoff):
    model_id = model_id_from_data_cutoff(data_cutoff)
    arr_pt, arr_ca = nir2025.ktCO2e_dense_w_nan()

    with model_db.connect() as conn:
        cursor = conn.cursor()
        for sector in IPCC_Sector:
            for ghg in GHG:
                component_id='comp_{}'.format(model_db.stable_hash(str(
                    (model_id, ghg.value, sector.value,))))
                try:
                    model_db.by_id('ComponentType', component_id=component_id)
                    print('Found component', component_id, model_id, ghg.value, sector.value, 'skipping')
                    continue
                except model_db.NoRecord:
                    pass

                for pt in PT:
                    if pt != PT.XX:
                        model_db.insert_component_mapping(
                            cursor=cursor,
                            model_id=model_id,
                            ghg=ghg,
                            NIR_sector=sector,
                            region=pt.value,
                            component_id=component_id)
                model_db.insert_component_mapping(
                    cursor=cursor,
                    model_id=model_id,
                    ghg=ghg,
                    NIR_sector=sector,
                    region='Canada',
                    component_id=component_id)

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

                if jnp.nansum(abs(jnp_ca)) <= 1: # 1kt, aka very small
                    model_db.insert_component_type(
                        cursor=cursor,
                        component_id=component_id,
                        component_type='Normal',
                        model_id=model_id)
                    # TODO: estimating that there's a 50% chance the emission will be negative is
                    # silly for most sectors.
                    model_db.insert_component_normal(
                        cursor=cursor,
                        component_id=component_id,
                        location=0,
                        scale=1,
                        v_unit='kt CO2e')
                else:
                    model_db.insert_component_type(
                        cursor=cursor,
                        component_id=component_id,
                        component_type='BayesianNormal',
                        model_id=model_id)
                    model_db.insert_component_bayesian_normal(
                        cursor=cursor,
                        component_id=component_id,
                        num_samples=500,
                        num_warmup=250,
                        thinning=1,
                        seed=0,
                        scale=scale)
                    model_db.insert_task(
                        cursor=cursor,
                        payload={
                            'entrypoint': 'static_normals_inference',
                            'component_id': component_id,
                            })


def entrypoint_static_normals_inference(payload, model_db=model_db):
    component_id = payload['component_id']

    params, = model_db.params_BayesianNormal(component_id)

    # Check if the grouped samples have been saved. If they're
    # present, assume they are correct.
    try:
        grouped_samples = model_db.index_ndarray_group(
            model_id=params['model_id'],
            component_id=component_id,
            group_id='grouped_samples',
            ndarray_d_keys=['mu', 'sigma_pt', 'sigma_ca'])
        if len(grouped_samples) == 3:
            return
        assert 0, (payload, params)
    except OSError:
        pass # at least one file is missing, probably all of them

    # Design pattern:
    # in this function, train/ infer this component by looking at the
    # data_cutoff parameter, and including as many data sources as
    # appropriate, so long as they were published before the cutoff.
    #
    # It is appropriate to dispatch to different implementations depending on
    # the data_cutoff.
    #
    # TODO: each data source should feature a publication date.

    # Emulate use of NIR2024 by using first 33 years' data in NIR2025
    nir2024_publication_date = datetime.date(year=2024, month=4, day=1) # approx
    nir2025_publication_date = datetime.date(year=2025, month=4, day=1) # approx
    if params['data_cutoff'] < nir2024_publication_date:
        raise NotImplementedError(params['data_cutoff'])
    elif params['data_cutoff'] < nir2025_publication_date:
        n_training_years = 33
    else:
        n_training_years = 34

    nir_rng_key = jrandom.key(123)
    NIR_emission_sample_size = 100
    weights = jnp.array([1.0 / NIR_emission_sample_size] * NIR_emission_sample_size)

    ca_sample = np.empty((NIR_emission_sample_size, n_training_years))
    pt_sample = np.empty((NIR_emission_sample_size, n_training_years, 13,))
    real_PTs = [pt for pt in PT if pt != PT.XX]
    for ii, year in enumerate(range(1990, 1990 + n_training_years)):
        ca_dist, pt_dists = nir2025.ktCO2e_numpyro_dist_pt_ca(
            sector=params['sector'],
            ghg=params['ghg'],
            year=year)
        nir_rng_key, rng_key_ = jrandom.split(nir_rng_key)
        ca_sample[:, ii] = ca_dist.sample(rng_key_, (NIR_emission_sample_size,))
        for jj, pt in enumerate(real_PTs):
            nir_rng_key, rng_key_ = jrandom.split(nir_rng_key)
            pt_sample[:, ii, jj,] = pt_dists[jj].sample(rng_key_, (NIR_emission_sample_size,))


    scaled_ca = jnp.array(ca_sample / params['scale'])
    scaled_pt = jnp.array(pt_sample / params['scale'])
    mcmc_rng_key = jrandom.key(params['seed'])

    mcmc = MCMC(NUTS(weighted_constant_model),
                num_warmup=params['num_warmup'],
                thinning=params['thinning'],
                num_samples=params['num_samples'])

    mcmc_rng_key, rng_key_ = jrandom.split(mcmc_rng_key)
    mcmc.run(rng_key_,
             scaled_pt=scaled_pt,
             scaled_ca=scaled_ca,
             weights=weights,
            )
    if payload.get('print_summary'):
        mcmc.print_summary()
    grouped_samples = mcmc.get_samples(group_by_chain=True)
    model_db.save_ndarray_group(
        model_id=params['model_id'],
        component_id=component_id,
        group_id='grouped_samples',
        ndarray_d=grouped_samples)


def inference_work_loop(data_cutoff):
    for payload in model_db.task_completion_iter():
        assert payload['entrypoint'] == 'static_normals_inference'
        entrypoint_static_normals_inference(payload)


def normals_by_sector_ghg(model_id) -> dict:
    normal_components = {
            comp_d['component_id']: comp_d
            for comp_d in model_db.normal_components_by_model(model_id)}
    rval = {}
    for scope_d in model_db.component_scopes_by_model_id(model_id):
        comp_id = scope_d['component_id']
        if comp_id in normal_components:
            rval[scope_d['sector'], scope_d['ghg']] = normal_components[comp_id]
    return rval


def BNs_by_sector_ghg(model_id) -> dict:
    BN_components = {
            comp_d['component_id']: comp_d
            for comp_d in model_db.BayesianNormal_components_by_model(model_id)}
    rval = {}
    for scope_d in model_db.component_scopes_by_model_id(model_id):
        comp_id = scope_d['component_id']
        if comp_id in BN_components:
            rval[scope_d['sector'], scope_d['ghg']] = BN_components[comp_id]
    return rval


def post_samples_from_grouped_samples(grouped_samples):
    rval = {
        key: val.reshape(-1, *val.shape[2:])
        for key, val in grouped_samples.items()}
    return rval


def loglik_NIR_Normal(
    component_id,
    NIR_year,
    emission_year,
    ):
    return float('nan')

    params, = model_db.params_Normal(component_id)

    assert NIR_year == 2025
    arr_pt, arr_ca = nir2025.ktCO2e_dense_w_nan()

    assert emission_year == 2023
    arr_idx_of_2023 = arr_ca.shape[2] - 1

    eval_obs = np.zeros(14)
    eval_obs[:13] = arr_pt[
        nir2025.idx_of_sector[params['sector']],
        nir2025.idx_of_ghg[params['ghg']],
        :,
        arr_idx_of_2023]
    eval_obs[13] = arr_ca[
        nir2025.idx_of_sector[params['sector']],
        nir2025.idx_of_ghg[params['ghg']],
        arr_idx_of_2023]
    valid_mask = np.isfinite(eval_obs)

    # log_probs: 14 elements
    ca_scale = params['scale']
    ca_var = ca_scale ** 2
    pt_var = ca_var / 13 # could do better: not all pts are typically equal
    pt_scale = np.sqrt(pt_var)
    assert params['location'] == 0
    pt_log_probs = dist.Normal(0, pt_scale).log_prob(eval_obs[:13])
    ca_log_prob = dist.Normal(0, ca_scale).log_prob(eval_obs[13])
    # product (log sum) over provinces, territories, and country
    logprob_X = float(np.sum(pt_log_probs[valid_mask[:13]])
                      + (ca_log_prob if valid_mask[13] else 0))

    #print('ll_Normal', logprob_X)
    assert np.isfinite(logprob_X)
    return logprob_X


@my_functools.cache
def loglik_NIR_BayesianNormal(
    component_id,
    NIR_year,
    emission_year,
    ):
    return float('nan')
    assert NIR_year == 2025
    assert emission_year == 2023

    params, = model_db.params_BayesianNormal(component_id)
    grouped_samples = model_db.load_ndarray_group(component_id, 'grouped_samples')
    post_samples = post_samples_from_grouped_samples(grouped_samples)

    sample_size = params['num_samples'] // params['thinning']

    mu = post_samples['mu'] # (sample_size, 13)
    mu_ca = mu.sum(axis=1) # (sample_size,)

    from planzero import nir2025
    arr_pt, arr_ca = nir2025.ktCO2e_dense_w_nan()
    arr_idx_of_2023 = arr_ca.shape[2] - 1

    eval_mu = np.zeros((sample_size, 14))
    eval_mu[:, :13] = mu
    eval_mu[:, 13] = mu_ca
    eval_mu *= params['scale']

    eval_sigma = np.zeros((sample_size, 14))
    eval_sigma[:, :13] = post_samples['sigma_pt'][0]
    eval_sigma[:, 13] = post_samples['sigma_ca']
    eval_sigma *= params['scale']

    eval_obs = np.zeros(14)
    eval_obs[:13] = arr_pt[
        nir2025.idx_of_sector[params['sector']],
        nir2025.idx_of_ghg[params['ghg']],
        :,
        arr_idx_of_2023]
    eval_obs[13] = arr_ca[
        nir2025.idx_of_sector[params['sector']],
        nir2025.idx_of_ghg[params['ghg']],
        arr_idx_of_2023]
    valid_mask = np.isfinite(eval_obs)

    import numpyro.distributions as dist
    # log_probs: sample_size x 14
    log_probs = dist.Normal(eval_mu, eval_sigma).log_prob(eval_obs)
    # product (log sum) over provinces, territories, and country
    # results in sample_size estimates
    logprob_X = np.sum(log_probs[:, valid_mask], axis=1)

    # return them as-is so that they can be averaged after combining
    # the estimates from other ghgs, sectors, etc.
    assert np.isfinite(logprob_X).all()

    ## DEBUG
    #KL_NIR_BayesianNormal(component_id, NIR_year, emission_year)
    ##
    return logprob_X


@my_functools.cache
def KL_NIR_BayesianNormal(
    model_id,
    component_id, # TODO: should be sector, gas, etc?
    NIR_year, # target
    emission_year, # target
    model_db=model_db,
    ):
    assert NIR_year == 2025
    assert emission_year == 2023

    params, = model_db.params_BayesianNormal(component_id)
    grouped_samples = model_db.load_ndarray_group(model_id, component_id, 'grouped_samples')
    post_samples = post_samples_from_grouped_samples(grouped_samples)

    sample_size = params['num_samples'] // params['thinning']

    mu = post_samples['mu'] # (sample_size, 13)
    mu_ca = mu.sum(axis=1) # (sample_size,)

    from planzero import nir2025
    arr_pt, arr_ca = nir2025.ktCO2e_dense_w_nan()
    arr_idx_of_2023 = arr_ca.shape[2] - 1

    eval_mu = np.zeros((sample_size, 14))
    eval_mu[:, :13] = mu
    eval_mu[:, 13] = mu_ca
    eval_mu *= params['scale']

    eval_sigma = np.zeros((sample_size, 14))
    eval_sigma[:, :13] = post_samples['sigma_pt'][0]
    eval_sigma[:, 13] = post_samples['sigma_ca']
    eval_sigma *= params['scale']

    eval_obs = np.zeros(14)
    eval_obs[:13] = arr_pt[
        nir2025.idx_of_sector[params['sector']],
        nir2025.idx_of_ghg[params['ghg']],
        :,
        arr_idx_of_2023]
    eval_obs[13] = arr_ca[
        nir2025.idx_of_sector[params['sector']],
        nir2025.idx_of_ghg[params['ghg']],
        arr_idx_of_2023]
    valid_mask = np.isfinite(eval_obs)

    ca2023_dist, pt2023_dists = nir2025.ktCO2e_numpyro_dist_pt_ca(
        sector=params['sector'],
        ghg=params['ghg'],
        year=2023)

    # KL computation
    kls = []
    rng_key = jrandom.key(123)
    N = 500 # sample size for empirical KL estimation
    for ii, (dist_p, pt) in enumerate(zip(pt2023_dists + [ca2023_dist], PT)):
        rng_key, rng_key_ = jrandom.split(rng_key)
        x = dist_p.sample(rng_key_, (N,))
        log_p = dist_p.log_prob(x)
        assert log_p.shape == (N,)
        expected_log_p = log_p.mean()

        # interpret the MCMC samples as components of large mixture model
        component_dist_q = dist.Normal(eval_mu[:, ii, None], eval_sigma[:, ii, None])
        component_log_q = component_dist_q.log_prob(x)
        assert component_log_q.shape == (sample_size, N)
        from scipy.special import logsumexp

        log_q = logsumexp(component_log_q, b=1.0 / sample_size, axis=0)
        assert log_q.shape == (N,)
        expected_log_q = log_q.mean()
        kl_ii = expected_log_p - expected_log_q
        kls.append(float(kl_ii))
        if params['sector'] == IPCC_Sector.Aluminium_Production and params['ghg'] == GHG.CO2:
            print()
            print(ii, kl_ii, pt)
            print('eval_obs', eval_obs[ii])
            print('P', dist_p, 'mu', dist_p.mu, 'min', min(x), 'mean', np.mean(x), 'max', max(x))
            rng_key, rng_key_ = jrandom.split(rng_key)
            x_q = component_dist_q.sample(rng_key_, (N,)).flatten()
            print('Q', component_dist_q, f'min {min(x_q.flatten())} mean {np.mean(x_q.flatten())} max {max(x_q.flatten())}')

    #print(params['sector'], params['ghg'], sum(kls))
    #print(' ' * 10, kls)
    return sum(kls)


def main_all(args):
    model_db.init_db()
    main_touch_model(args)

    main_touch_components(
        parser.parse_args(
            ['touch_components', '--model-id', 'model_fe117e9d55ab2a02']))
    model_db.main_worker(
        model_db.parser.parse_args(
            ['worker', '--raise-on-failure',
             '--on-complete=exit',
            ]))


if __name__ == '__main__':
    import argparse
    import sys

    # create the top-level parser
    parser = argparse.ArgumentParser(prog='planzero')
    subparsers = parser.add_subparsers(help='subcommand help')

    subparser = subparsers.add_parser('touch_model')
    #subparser.add_argument('--year', type=int, default=2005, help='year')
    subparser.set_defaults(func=main_touch_model)

    subparser = subparsers.add_parser('touch_components')
    subparser.add_argument('--model-id', type=str)
    subparser.set_defaults(func=main_touch_components)

    subparser = subparsers.add_parser('all')
    subparser.set_defaults(func=main_all)

    args = parser.parse_args()
    sys.exit(args.func(args))
