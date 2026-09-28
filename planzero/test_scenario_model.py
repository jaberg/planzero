from .scenario_model import *


class Exponential_Population(ModelElement):

    def __init__(self):
        self.data_len = ndarray_dimension('data_len')

    def inference(self):
        def annual_scan_prep(ws):
            ws.general['alpha'] = None # sample
            ws.general['data'] = [
                    1000, 1001, 1005, 1006, 1010, 1013, 1020]
            ws.initial_carry['y_curr'] = 0
            ws.initial_carry['y_prev'] = 0
        
        def annual_scan_step(ws):
            ws.next_carry['y_curr'] = 0
            ws.next_carry['y_prev'] = 0
            ws.y_t['mu'] = 0

        def annual_scan_post(ws):
            ws.general['sample_data'] = 0 # sample, and observe data


        return ModelElementInference(
            defines={
                General('alpha'): UnknownVariable(value_type=Ndarray()),
                AnnualX('data'): Variable(value_type=Ndarray(shape=[self.data_len])),
                AnnualCarry('y_curr'): UnknownVariable(value_type=Ndarray()),
                AnnualCarry('y_prev'): UnknownVariable(value_type=Ndarray()),
                AnnualY('mu'): UnknownVariable(value_type=Ndarray()),
                General('sample_data'): UnknownVariable(
                    value_type=Ndarray(shape=[self.data_len]),
                    # requires=[ScanY('mu')],
                    # definition_phase=InferencePostScan,
                    observation=General('data')),
                },
            annual_scan_prep=annual_scan_prep,
            annual_scan_step=annual_scan_step,
            annual_scan_post=annual_scan_post,
            annual_scan_post_reads=[General('data')],
            )

    def analysis(self):
        def annual_scan_pre(ws):
            ws.general['mean_2'] = jnp.mean(
                    ws.posterior['mu'][:, 2])
            ws.general[AnnualEmission(sector, gas, activity)] = 0

        return AnalysisPhase(
            defines={
                General('mean_2'): Variable(value_type=Ndarray()),
                General(AnnualEmission(sector, gas, activity)): Variable(),
                },
            annual_scan_prep=annual_scan_pre,
            annual_scan_prep_reads=[Posterior('mu')],
            )


def test_0():
    pass

    #model = Model()
    #model.add_element(Exponential_Population())
