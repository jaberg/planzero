"""
Example inference that's fast and represents a lower bound on accuracy.
"""
import os


import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer import Predictive

from . import nir2025
from .prob import SiteInference
from .enums import IPCC_Sector, GHG, PT, LULUCF_Sectors

from .nir_constant_predictor import constant_model
from . import model_db
from . import my_functools

import hashlib


def stable_hash(data: str) -> str:
    """Returns a stable, cross-session SHA-256 hex string."""
    # 1. Encode the string to bytes
    encoded_data = data.encode('utf-8')

    # 2. Generate and return the hexadecimal digest
    return hashlib.shake_128(encoded_data).hexdigest(8)


def main_touch_model(args):
    with model_db.connect() as conn:
        cursor = conn.cursor()

        family='StaticNormal'
        version = 1
        data_cutoff = '2024-12-31'
        model_id = 'model_{}'.format(stable_hash(str(
            (family, version, data_cutoff))))
        try:
            model_db.by_id('Model', model_id=model_id)
        except model_db.NoRecord:
            model_db.insert_model(
                cursor=cursor,
                model_id=model_id,
                family=family,
                version=version,
                version_description='initial',
                data_cutoff=data_cutoff,
                )


def main_touch_components(args):
    arr_pt, arr_ca = nir2025.ktCO2e_dense_w_nan()
    from .enums import IPCC_Sector, GHG

    model_id = args.model_id
    with model_db.connect() as conn:
        cursor = conn.cursor()
        for sector in IPCC_Sector:
            for ghg in GHG:
                component_id='comp_{}'.format(stable_hash(str(
                    (model_id, ghg.value, sector.value,))))
                print(component_id)
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
                        component_type='Normal')
                    model_db.insert_component_normal(
                        cursor=cursor,
                        component_id=component_id,
                        location=0,
                        scale=1,
                        v_unit=f'kt CO2e')
                else:
                    model_db.insert_component_type(
                        cursor=cursor,
                        component_id=component_id,
                        component_type='BayesianNormal')
                    model_db.insert_component_bayesian_normal(
                        cursor=cursor,
                        component_id=component_id,
                        num_samples=100,
                        num_warmup=250,
                        thinning=1,
                        seed=0,
                        scale=scale)
                    model_db.insert_task(
                        cursor=cursor,
                        payload=dict(
                            entrypoint='static_normals_inference',
                            component_id=component_id,
                            ))


def entrypoint_static_normals_inference(payload):
    component_id = payload['component_id']

    params, = model_db.params_BayesianNormal(component_id)

    arr_pt, arr_ca = nir2025.ktCO2e_dense_w_nan()
    jnp_pt = jnp.array(arr_pt[nir2025.idx_of_sector[params['sector']],
                                   nir2025.idx_of_ghg[params['ghg']]],
                            dtype='float64')
    jnp_ca = jnp.array(arr_ca[nir2025.idx_of_sector[params['sector']],
                                   nir2025.idx_of_ghg[params['ghg']]],
                            dtype='float64')
    scaled_ca = jnp.array(jnp_ca / params['scale'])
    scaled_pt = jnp.array(jnp_pt / params['scale'])
    rng_key = jrandom.key(params['seed'])

    mcmc = MCMC(NUTS(constant_model),
                num_warmup=params['num_warmup'],
                thinning=params['thinning'],
                num_samples=params['num_samples'])

    rng_key, rng_key_ = jrandom.split(rng_key)
    mcmc.run(rng_key_,
             scaled_pt=scaled_pt.T,
             scaled_ca=scaled_ca)
    if payload.get('print_summary'):
        mcmc.print_summary()
    grouped_samples = mcmc.get_samples(group_by_chain=True)
    model_db.save_ndarray_group(component_id, 'grouped_samples', grouped_samples)


def post_samples_from_grouped_samples(grouped_samples):
    rval = {
        key: val.reshape(-1, *val.shape[2:])
        for key, val in grouped_samples.items()}
    return rval


@my_functools.cache
def loglik_NIR_BayesianNormal(
    component_id,
    NIR_year,
    emission_year,
    ):
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

    import numpyro.distributions as dist
    # log_probs: sample_size x 14
    log_probs = dist.Normal(eval_mu, eval_sigma).log_prob(eval_obs)
    # straight sum over samples
    logprob_X = np.sum(log_probs, axis=1)
    rval = {'by_pt': {}}
    for ii, pt in enumerate(PT):
        if pt == PT.XX:
            assert ii == 13
            rval['Canada'] = log_prob_X[ii]
        else:
            rval[pt] = log_prob_X[ii]
    return rval


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

    main_touch_loglik(
        parser.parse_args(
            ['touch_loglik', '--model-id', 'model_fe117e9d55ab2a02']))



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
