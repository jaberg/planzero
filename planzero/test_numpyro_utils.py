
import jax.numpy as jnp
import jax.random as jrandom

from numpyro.distributions import Normal
from .numpyro_utils import stack_distributions
from .symmetric_blended_lognormal import SymmetricBlendedLogNormal


def test_stack_distributions_normal():

    d0 = Normal(0, .1)
    d1 = Normal(10, .1)
    d2 = Normal(-10, .1)

    dd = stack_distributions([d0, d1, d2])

    dd_alt = Normal(jnp.array([0, 10, -10]), .1)
    assert dd.batch_shape == dd_alt.batch_shape
    assert dd.event_shape == dd_alt.event_shape

    sample_default = dd.sample(key=jrandom.key(0))
    sample_w_shape = dd.sample(key=jrandom.key(0), sample_shape=(4,))

    assert sample_default.shape == (3,)
    assert sample_w_shape.shape == (4, 3)

    print(sample_w_shape)
    assert jnp.abs(sample_w_shape[:, 0]).max() < 1
    assert jnp.abs(sample_w_shape[:, 1] - 10).max() < 1
    assert jnp.abs(sample_w_shape[:, 2] + 10).max() < 1

    expand_sample = dd.expand((5, 3)).sample(key=jrandom.key(0))
    assert expand_sample.shape == (5, 3)
    assert jnp.abs(expand_sample[:, 0]).max() < 1
    assert jnp.abs(expand_sample[:, 1] - 10).max() < 1
    assert jnp.abs(expand_sample[:, 2] + 10).max() < 1


def test_stack_distributions_symmetric_blended_lognormal():

    d0 = SymmetricBlendedLogNormal.rolloff_relerr(0, .1, .01)
    d1 = SymmetricBlendedLogNormal.rolloff_relerr(10, .1, .01)
    d2 = SymmetricBlendedLogNormal.rolloff_relerr(-10, .1, .01)

    dd = stack_distributions([d0, d1, d2])

    sample_default = dd.sample(key=jrandom.key(0))
    sample_w_shape = dd.sample(key=jrandom.key(0), sample_shape=(4,))

    assert sample_default.shape == (3,)
    assert sample_w_shape.shape == (4, 3)

    print(sample_w_shape)
    assert jnp.abs(sample_w_shape[:, 0]).max() < 1
    assert jnp.abs(sample_w_shape[:, 1] - 10).max() < .1
    assert jnp.abs(sample_w_shape[:, 2] + 10).max() < .1

    expand_sample = dd.expand((5, 3)).sample(key=jrandom.key(0))
    assert expand_sample.shape == (5, 3)
    assert jnp.abs(expand_sample[:, 0]).max() < 1
    assert jnp.abs(expand_sample[:, 1] - 10).max() < .1
    assert jnp.abs(expand_sample[:, 2] + 10).max() < .1
