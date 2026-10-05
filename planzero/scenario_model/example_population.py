import jax.numpy as jnp
import numpyro.distributions as dist

from .base import (
    ndarray_dim,
    observation,
    observation_valid,
    years_dim,
)
from .computation import (
    ModelElement,
    WorkSpace_Prep,
    WorkSpace_Proc,
    WorkSpace_Step,
    access_val,
    define,
    define_annual,
    define_carry,
    years_key,
)


class Example_Population(ModelElement):

    # derived from pydantic.BaseModel 
    # avoid defining __init__ if possible

    @define('alpha', prior_shape=[])
    @define('data',  prior_shape=[ndarray_dim()])
    @define_carry('y_curr', shape=[])
    @define_carry('y_prev', shape=[])
    def model_element_prepare(self, ws:WorkSpace_Prep):
        ws.dist.general['alpha'] = dist.Kumaraswamy(1.25, 1.25)
        ws.dist.initial_carry['y_curr'] = dist.Normal()
        ws.val.initial_carry['y_prev'] = jnp.zeros(())
        ws.val.general['data'] = jnp.array(
                [1000.0, 1001.1, 1005.2, 1006, 1010, 1013, 1020])

    @access_val('alpha')
    @define_annual('mu', annual_shape=[])
    def model_element_annual_step(self, ws:WorkSpace_Step):
        coef = -0.5 + 3 * ws.val.general['alpha']
        ws.dist.next_carry['y_curr'] = dist.Normal(
                coef
                + 0.5 * ws.val.this_carry['y_curr']
                + 0.5 * ws.val.this_carry['y_prev']
                )
        ws.val.next_carry['y_prev'] = ws.val.this_carry['y_curr']
        ws.val.this_Y['mu'] = ws.val.this_carry['y_curr'] * 1000

    @define('generated_data', prior_shape=[years_dim])
    def model_element_postprocess(self, ws:WorkSpace_Proc):
        assert ws.year_0 == 1990
        if ws.n_years < 7:
            raise NotImplementedError()
        elif ws.n_years == 7:
            ws.val.general[observation('generated_data')] = ws.val.general['data']
        else:
            ws.val.general[observation('generated_data')] = jnp.concatenate(
                    [ws.val.general['data'], jnp.zeros(ws.n_years - 7)])
            ws.val.general[observation_valid('generated_data')] = ws.val.general[years_key] < 1997
        ws.dist.general['generated_data'] = dist.Normal(
                ws.val.general['mu'],
                scale=50.0)
