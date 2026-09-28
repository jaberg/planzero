
import jax.numpy as jnp
import numpyro.distributions as dist

from .scenario_model import *


class Exponential_Population(ModelElement):

    # derived from BaseModel 
    # avoid defining __init__ if possible

    def inference(self, ie: InferenceElement) -> None:
        ie.general('alpha', sample=True, shape=[])
        ie.annual_X('data', sample=False, shape=[ie.annual_scan_dim])
        ie.carry('y_curr', shape=[ie.annual_scan_dim],
                 initial_sample=True,
                 next_sample=True,
                 )
        ie.carry('y_prev', shape=[ie.annual_scan_dim],
                 initial_sample=False,
                 next_sample=False)
        ie.annual_Y('mu', sample=True, shape=[ie.annual_scan_dim])
        ie.general('generated_data', shape=[ie.annual_scan_dim],
                   sample=True,
                   observation='data')

        @ie.annual_scan_prep()
        def prep(ws:WorkSpace_AnnualScanPrep_Inference) -> None:
            ws.general['alpha'].dist = dist.Kumaraswamy(1.25, 1.25)
            ws.initial_carry['y_curr'].dist = dist.Normal()
            ws.initial_carry['y_prev'] = 0
            ws.annual_X['data'] = jnp.array(
                    [1000, 1001, 1005, 1006, 1010, 1013, 1020])
        
        @ie.annual_scan_step()
        def step(ws:WorkSpace_AnnualScanStep_Inference) -> None:
            coef = -0.5 + 3 * ws.general['alpha'].sample
            ws.next_carry['y_curr'].dist = dist.Normal(
                    coef
                    + 0.5 * ws.this_carry['y_curr']
                    + 0.5 * ws.this_carry['y_prev']
                    )
            ws.next_carry['y_prev'] = ws.this_carry['y_curr']
            ws.this_Y['mu'] = ws.this_carry['y_curr'] * 1000

        @ie.annual_scan_post(reads=['data'])
        def post(ws:WorkSpace_AnnualScanPost_Inference) -> None:
            ws.general['generated_data'].dist = dist.Normal(
                    ws.annual_Y['mu'],
                    scale=5.0)



def test_add_inference_add_vars():

    model = AnnualScanModel()
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
    assert 'mu' in model.mv.posterior_nd
    assert 'sample_data' not in model.mv.posterior_nd
    assert 'data' not in model.mv.posterior_nd


if 0:
  def test_add_inference_run():

    model = AnnualScanModel()
    model.add_element(Exponential_Population())

    icomp = InferenceComputation(model=model)
    icomp.run()
