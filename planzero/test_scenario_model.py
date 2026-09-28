
import jax.numpy as jnp

from .scenario_model import *


class Exponential_Population(ModelElement):

    # derived from BaseModel 
    # avoid defining __init__ if possible

    def inference(self, ie: InferenceElement) -> None:
        ie.general('alpha', known=False, shape=[])
        ie.annual_X('data', known=True, shape=[ie.annual_scan_dim])
        ie.carry('y_curr', initial_known=False, shape=[ie.annual_scan_dim])
        ie.carry('y_prev', shape=[ie.annual_scan_dim],
                 initial_known=True,
                 final_known=False)
        ie.annual_Y('mu', known=False, shape=[ie.annual_scan_dim])
        ie.general('generated_data', known=True, shape=[ie.annual_scan_dim],
                     observation='data')

        @ie.annual_scan_prep()
        def prep(ws:WorkSpace_AnnualScanPrep_Inference) -> None:
            ws.general_nd['alpha'] = None # sample
            ws.general_nd['data'] = [
                    1000, 1001, 1005, 1006, 1010, 1013, 1020]
            ws.initial_carry_nd['y_curr'] = 0
            ws.initial_carry_nd['y_prev'] = 0
        
        @ie.annual_scan_step()
        def step(ws:WorkSpace_AnnualScanStep_Inference) -> None:
            ws.next_carry_nd['y_curr'] = 0
            ws.next_carry_nd['y_prev'] = 0
            ws.this_Y_nd['mu'] = 0

        @ie.annual_scan_post(reads=['data'])
        def post(ws:WorkSpace_AnnualScanPost_Inference) -> None:
            ws.general_nd['sample_data'] = 0 # sample, and observe data

    def analysis(self, annual_scan_dim: NdarrayDim) -> ModelElementAnalysis:
        rval = ModelElementAnalysis(
            defines={
                General('mean_2'): Variable(value_type=Ndarray()),
                #General(AnnualEmission(sector, gas, activity)): Variable(),
                },
            )

        @rval.annual_scan_prep(reads=[posterior('mu')])
        def annual_scan_prep(ws:WorkSpace_AnnualScanPrep_Analysis) -> None:
            ws.general_nd['mean_2'] = jnp.mean(
                    ws.general_nd[posterior('mu')][:, 2])
            #ws.general_nd[AnnualEmission(sector, gas, activity)] = 0

        return rval



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
    assert final_carry('y_prev') in model.mv.posterior_nd
    assert 'mu' in model.mv.posterior_nd
    assert 'sample_data' not in model.mv.posterior_nd
    assert 'data' not in model.mv.posterior_nd


if 0:
  def test_add_inference_run():

    model = AnnualScanModel()
    model.add_element(Exponential_Population())

    icomp = InferenceComputation(model=model)
    icomp.run()
