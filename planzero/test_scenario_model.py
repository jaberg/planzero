
import jax.numpy as jnp
import numpyro.distributions as dist

from .posterior import *
from .scenario_model import *


class Exponential_Population(ModelElement):

    # derived from BaseModel 
    # avoid defining __init__ if possible

    def inference(self, ie: InferenceElement) -> None:
        ie.general('alpha', sample=True, shape=[])
        ie.annual_X('data', sample=False, shape=[ie.inference_years_dim])
        ie.carry('y_curr', shape=[ie.inference_years_dim],
                 sample_initial=True,
                 sample_next=True,
                 )
        ie.carry('y_prev', shape=[ie.inference_years_dim],
                 sample_initial=False,
                 sample_next=False)
        ie.annual_Y('mu', sample=False, shape=[ie.inference_years_dim])
        ie.general('generated_data', shape=[ie.inference_years_dim],
                   sample=True,
                   observation='data')

        #@ie.define('alpha', sample=True, shape=[])
        #@ie.define_annual('data', sample=False, annual_shape=[])
        #@ie.define_carry('y_curr')
        #@ie.define_carry('y_prev')
        @ie.annual_scan_prep()
        def prep(ws:WorkSpace_Prep) -> None:
            ws.dist.general['alpha'] = dist.Kumaraswamy(1.25, 1.25)
            ws.dist.initial_carry['y_curr'] = dist.Normal()
            ws.val.initial_carry['y_prev'] = jnp.zeros(())
            ws.val.annual_X['data'] = jnp.array(
                    [1000, 1001, 1005, 1006, 1010, 1013, 1020])
        
        #@ie.define_Y('mu')
        @ie.annual_scan_step()
        def step(ws:WorkSpace_Step) -> None:
            coef = -0.5 + 3 * ws.val.general['alpha']
            ws.dist.next_carry['y_curr'] = dist.Normal(
                    coef
                    + 0.5 * ws.val.this_carry['y_curr']
                    + 0.5 * ws.val.this_carry['y_prev']
                    )
            ws.val.next_carry['y_prev'] = ws.val.this_carry['y_curr']
            ws.val.this_Y['mu'] = ws.val.this_carry['y_curr'] * 1000

        #@ie.define('generated_data')
        @ie.annual_scan_post(reads=['data'])
        def post(ws:WorkSpace_Proc) -> None:
            ws.dist.general['generated_data'] = dist.Normal(
                    ws.val.annual_Y['mu'],
                    scale=50.0)


def test_add_inference_add_vars():

    model = AnnualScanModel(n_inference_years=7, n_posterior_years=10)
    model.add_element(Exponential_Population())

    print(model.mv.general_nd)

    print(model.mv.initial_carry_nd)
    print(model.mv.this_carry_nd)
    print(model.mv.next_carry_nd)
    print(model.mv.final_carry_nd)

    print(model.mv.this_X_nd)
    print(model.mv.Xs_nd)

    print(model.mv.this_Y_nd)
    print(model.mv.Ys_nd)

    print("Posteriors")
    print("----------")
    for key, val in model.mv.posterior_nd.items():
        print(key)
        print(val)
        print()
    assert 'alpha' in model.mv.posterior_nd
    assert 'y_curr' not in model.mv.posterior_nd
    assert initial_carry('y_curr') in model.mv.posterior_nd
    assert final_carry('y_curr') in model.mv.posterior_nd
    assert initial_carry('y_prev') not in model.mv.posterior_nd
    assert final_carry('y_prev') not in model.mv.posterior_nd
    assert 'mu' not in model.mv.posterior_nd
    assert 'sample_data' not in model.mv.posterior_nd
    assert 'data' not in model.mv.posterior_nd


def test_add_inference_run_smoke():

    model = AnnualScanModel(n_inference_years=7, n_posterior_years=10)
    model.add_element(Exponential_Population())

    icomp = InferenceComputation(model=model, seed=123)
    icomp.run_once()


def test_add_inference_mcmc_smoke():
    model = AnnualScanModel(n_inference_years=7, n_posterior_years=10)
    model.add_element(Exponential_Population())

    run_mcmc(model=model,
             seed=123,
             num_warmup=10,
             thinning=1,
             num_samples=10)

def test_posterior_smoke():
    model = AnnualScanModel(n_inference_years=7, n_posterior_years=10)
    model.add_element(Exponential_Population())
    mcmc = run_mcmc(model=model,
             seed=123,
             num_warmup=10,
             thinning=1,
             num_samples=10)

    pc = PosteriorComputation(
            model=model,
            grouped_samples=mcmc.get_samples(group_by_chain=True),
            )
