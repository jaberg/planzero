"""
Johnson's SU distribution isn't in NumPyro, this class was written by Gemini
to be NumPyro compatible.

https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.johnsonsu.html

https://en.wikipedia.org/wiki/Johnson%27s_SU-distribution
"""

# Gemini generated this code

import jax.numpy as jnp
import jax.random as random
from numpyro.distributions import Distribution, constraints
from numpyro.distributions.util import promote_shapes

class JohnsonSU(Distribution):
    # Define the constraints for the parameters so NumPyro knows how to handle
    # them
    arg_constraints = {
        "gamma": constraints.real,       # Shape parameter (skewness)
        "delta": constraints.positive,   # Shape parameter (tail heaviness/kurtosis)
        "loc": constraints.real,         # Location parameter
        "scale": constraints.positive,   # Scale parameter
    }

    # The support is unbounded (all real numbers)
    support = constraints.real

    # Marking parameters as reparametrized helps HMC/NUTS sample more
    # efficiently
    reparametrized_params = ["gamma", "delta", "loc", "scale"]

    def __init__(self, gamma=0., delta=1., loc=0., scale=1., validate_args=None):
        # promote_shapes ensures all parameters are broadcasted to the same
        # batch shape
        self.gamma, self.delta, self.loc, self.scale = promote_shapes(
            gamma, delta, loc, scale)
        batch_shape = jnp.shape(self.gamma)
        super(JohnsonSU, self).__init__(batch_shape=batch_shape,
                                        validate_args=validate_args)

    def value_from_z(self, z):
        return self.loc + self.scale * jnp.sinh((z - self.gamma) / self.delta)

    def z_from_value(self, value, return_y=False):
        if self._validate_args:
            self._validate_sample(value)

        y = (value - self.loc) / self.scale

        # Transform the value back to the standard normal space
        z = self.gamma + self.delta * jnp.arcsinh(y)
        return (z, y) if return_y else z

    def sample(self, key, sample_shape=()):
        """Draws samples from the distribution."""
        shape = sample_shape + self.batch_shape

        # 1. Sample from a standard Normal distribution
        z = random.normal(key, shape=shape)
        return self.value_from_z(z)

    def log_prob(self, value):
        """Evaluates the log probability density for a given value."""
        z, y = self.z_from_value(value, return_y=True)

        # 1. Log PDF of the standard normal distribution
        log_prob_z = -0.5 * jnp.log(2 * jnp.pi) - 0.5 * z**2

        # 2. Log determinant of the Jacobian (accounts for the volume change
        # from the transform)
        log_det_jacobian = (
            jnp.log(self.delta)
            - jnp.log(self.scale) - 0.5 * jnp.log(y**2 + 1.0))

        return log_prob_z + log_det_jacobian


def test_pdfs_match(plot_things=False):
    import numpy as np
    from scipy.stats import johnsonsu

    if plot_things:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1)

    # these values are from the example code on
    # scipy's Johnson's SU page
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.johnsonsu.html
    a, b = 2.55, 2.25
    lb, ub = johnsonsu.support(a, b)
    x = np.linspace(johnsonsu.ppf(0.01, a, b),
                    johnsonsu.ppf(0.99, a, b), 100)
    if plot_things:
        ax.plot(x, johnsonsu.pdf(x, a, b),
               'r-', lw=5, alpha=0.6, label='johnsonsu pdf')
        ax.plot(x, np.exp(JohnsonSU(a, b).log_prob(x)))
    else:
        assert np.allclose(
            johnsonsu.pdf(x, a, b),
            np.exp(JohnsonSU(a, b).log_prob(x)),
            )
