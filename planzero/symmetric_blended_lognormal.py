from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import numpyro.distributions as dist
from jax.typing import ArrayLike
from numpyro.distributions import Distribution, constraints, kl
from numpyro.distributions.util import promote_shapes


class SymmetricBlendedLogNormal(Distribution):
    """A 3-element mixture model implementing a symmatric version of
    log-normal with a dead-zone around 0.
    """
    arg_constraints = {  # noqa: RUF012
        "mu": constraints.real,
        "scale_neg": constraints.positive,
        "scale_norm": constraints.positive,
        "scale_pos": constraints.positive,
        "transition_rate": constraints.positive,
        "norm_dominance": constraints.real,
        "deadzone": constraints.positive,
    }

    @property
    def support(self):
        return constraints.real

    mu: jnp.ndarray
    scale_neg: jnp.ndarray
    scale_norm: jnp.ndarray
    scale_pos: jnp.ndarray
    transition_rate: jnp.ndarray
    norm_dominance: jnp.ndarray
    deadzone: jnp.ndarray

    @classmethod
    def rolloff_relerr(cls, mu, rolloff, relerr, **kwargs):
        """
        mu: the mean of the distribution it's "closeness" to 0 is measured relative to `rolloff`.
        rolloff: if small relative to abs(mu), the distribution is strongly one-sided. Considering mu as a parameter, it controls how sharply the distribution might change as mu approaches 0.
        relerr: if small relative to 1, the distribution is concentrated.
        """
        # relerr <= 0 will trigger constraints violation, catch earlier here
        # assert jnp.all(relerr > 0), relerr.min()

        # actually 0 maybe should work? not tested
        # assert jnp.all(rolloff > 0), rolloff.min()

        return cls(
            mu=mu,
            scale_neg=relerr / 3,
            scale_norm=rolloff * relerr / 3,
            scale_pos=relerr / 3,
            transition_rate=10.0 / rolloff,
            norm_dominance=10.0,
            deadzone=rolloff / 10,
            **kwargs)

    @property
    def rolloff(self):
        return self.deadzone * 10

    @property
    def relerr(self):
        return self.scale_neg * 3

    def __init__(self, mu=0, scale_neg=.1, scale_norm=1.0, scale_pos=.1,
                 transition_rate=1.0, norm_dominance=10.0, deadzone=1,
                 validate_args=None, dtype=jnp.float64):
        """mu is the mean of the distribution """

        # transition_rate dictates how sharply the mixture switches
        # norm_dominance dictates how wide the "normal" regime is around mu=0
        (self.mu, self.scale_neg, self.scale_norm, self.scale_pos,
         self.transition_rate, self.norm_dominance, self.deadzone) = \
            promote_shapes(
                jnp.array(mu, dtype=dtype),
                jnp.array(scale_neg, dtype=dtype),
                jnp.array(scale_norm, dtype=dtype),
                jnp.array(scale_pos, dtype=dtype),
                jnp.array(transition_rate, dtype=dtype),
                jnp.array(norm_dominance, dtype=dtype),
                jnp.array(deadzone, dtype=dtype))

        batch_shape = jnp.shape(self.mu)
        super().__init__(
            batch_shape=batch_shape,
            validate_args=validate_args)

    def __getitem__(self, item):
        # Select a single (or slice of) batch element. Parameters may have
        # shapes that are only broadcastable to ``batch_shape``, so broadcast
        # each one to the full batch shape before indexing.
        batch_shape = self.batch_shape
        kwargs = {}
        for name in self.arg_constraints:
            arr = getattr(self, name)
            broadcasted = jnp.broadcast_to(jnp.asarray(arr), batch_shape) \
                if batch_shape else jnp.asarray(arr)
            kwargs[name] = broadcasted[item]
        kwargs["validate_args"] = self._validate_args
        return type(self)(**kwargs)

    @property
    def variance(self) -> jnp.ndarray:
        # Analytic mixture variance of the three components. This mirrors the
        # moment calculation used by the mixture, including the log-normal
        # component moments (which depend on each component's own scale, not
        # only on the active-log-normal loc).
        log_weights = self._get_log_weights()
        weights = jnp.exp(log_weights)  # batch_shape + (3,)

        log_loc = self._active_lognormal_mu()

        # Negative log-normal component (negated).
        mean_neg = -jnp.exp(log_loc + 0.5 * self.scale_neg ** 2)
        var_neg = (jnp.exp(self.scale_neg ** 2) - 1.0) * jnp.exp(
            2.0 * log_loc + self.scale_neg ** 2)

        # Normal component.
        mean_norm = self.mu
        var_norm = self.scale_norm ** 2

        # Positive log-normal component.
        mean_pos = jnp.exp(log_loc + 0.5 * self.scale_pos ** 2)
        var_pos = (jnp.exp(self.scale_pos ** 2) - 1.0) * jnp.exp(
            2.0 * log_loc + self.scale_pos ** 2)

        comp_means = jnp.stack([mean_neg, mean_norm, mean_pos], axis=-1)
        comp_second_moments = jnp.stack(
            [var_neg + mean_neg ** 2,
             var_norm + mean_norm ** 2,
             var_pos + mean_pos ** 2],
            axis=-1)

        mix_mean = jnp.sum(weights * comp_means, axis=-1)
        mix_second = jnp.sum(weights * comp_second_moments, axis=-1)
        eps = 0 # 1e-20
        raw_rval = mix_second - mix_mean ** 2
        return jnp.where(raw_rval > eps, raw_rval, eps)

    def _get_log_weights(self):
        # Create logits that shift dominance based on the sign and magnitude of mu
        logit_neg = jnp.where(
            -self.mu > self.deadzone,
            (-self.mu - self.deadzone) * self.transition_rate,
            -jnp.inf)
        logit_norm = self.norm_dominance * jnp.ones_like(self.mu)
        logit_pos = jnp.where(
            self.mu > self.deadzone,
            (self.mu - self.deadzone) * self.transition_rate,
            -jnp.inf)

        assert logit_neg.shape == logit_norm.shape == logit_pos.shape
        logits = jnp.stack([logit_neg, logit_norm, logit_pos], axis=-1)
        rval = jax.nn.log_softmax(logits, axis=-1)
        return rval

    def _active_lognormal_mu(self):
        """Return the mu corresponding to the negative or positive lognormal
        that would give it a mean of self.mu
        """
        # mean = exp(log(mu) + scale ** 2 / 2)
        # log_mean = log(mu) + scale ** 2 / 2
        # log_mean - scale ** 2 / 2 = log_mu
        # exp(log_mean - scale ** 2 / 2) = mu
        mean = self.mu
        scale = jnp.where(mean < 0, self.scale_neg, self.scale_pos)
        mu = jnp.log(jnp.abs(mean) + 1e-6) - scale ** 2 / 2
        return mu

    def sample(self, key:ArrayLike|None, sample_shape:tuple=()) -> jnp.ndarray:
        shape = tuple(sample_shape) + self.batch_shape
        if key is None:
            raise NotImplementedError()
        key_comp, key_neg, key_norm, key_pos = jrandom.split(key, 4)

        # 1. Sample which component is active based on weights
        log_weights = self._get_log_weights()
        try:
            comp = dist.Categorical(logits=log_weights, validate_args=False
                                    ).sample(key_comp, sample_shape)
        except ValueError as err:
            err.add_note(f"logits={log_weights}")
            err.add_note(f"max(logits)={log_weights.max()}")
            raise

        log_loc = self._active_lognormal_mu()

        # 3. Sample from all three distributions
        samp_neg = -dist.LogNormal(log_loc, self.scale_neg).sample(
            key_neg, sample_shape)
        samp_norm = dist.Normal(self.mu, self.scale_norm).sample(
            key_norm, sample_shape)
        samp_pos = dist.LogNormal(log_loc, self.scale_pos).sample(
            key_pos, sample_shape)

        # 4. Pick the correct sample based on the categorical draw
        # comp == 0 -> neg, comp == 1 -> norm, comp == 2 -> pos
        rval = jnp.where(comp == 0, samp_neg,
                 jnp.where(comp == 1, samp_norm, samp_pos))
        assert rval.shape == shape
        return rval

    def log_prob(self, value:ArrayLike, intermediates:list[Any]|None=None):
        if self._validate_args:
            self._validate_sample(value)
        if intermediates is not None:
            raise NotImplementedError(intermediates)

        log_weights = self._get_log_weights()
        log_loc = self._active_lognormal_mu()

        # --- SAFE EVALUATION TRICK ---
        # Provide strictly valid dummy values to the LogNormal log_prob to
        # prevent NaNs.  The jnp.where mask will discard the dummy evaluations
        # anyway.
        safe_val_for_neg = jnp.where(value < 0, -value, 1.0)
        safe_val_for_pos = jnp.where(value > 0, value, 1.0)

        # Component 0: Negative Log-Normal
        log_p_neg = jnp.where(
            value < 0,
            dist.LogNormal(log_loc, self.scale_neg).log_prob(safe_val_for_neg),
            -jnp.inf
        )

        # Component 1: Normal
        log_p_norm = dist.Normal(self.mu, self.scale_norm).log_prob(value)

        # Component 2: Positive Log-Normal
        log_p_pos = jnp.where(
            value > 0,
            dist.LogNormal(log_loc, self.scale_pos).log_prob(safe_val_for_pos),
            -jnp.inf
        )

        # Combine using logsumexp:
        # log(w_neg*P_neg + w_norm*P_norm + w_pos*P_pos)
        components_log_prob = jnp.stack([
            log_weights[..., 0] + log_p_neg,
            log_weights[..., 1] + log_p_norm,
            log_weights[..., 2] + log_p_pos
        ], axis=-1)

        rval = jax.nn.logsumexp(components_log_prob, axis=-1)

        if isinstance(value, jnp.ndarray|np.ndarray):
            assert rval.shape == value.shape
        else:
            assert rval.shape == ()
        return rval


def _gauss_hermite(n_quad: int):
    """Return Gauss-Hermite nodes and weights scaled so that
    E[f] ~= sum_i weights_i * f(nodes_i) for a standard normal input f.

    hermgauss gives nodes for the weight exp(-t**2) (a N(0, 1/2) measure);
    scaling the nodes by sqrt(2) converts them to the N(0, 1) measure.
    """
    nodes, weights_unnorm = np.polynomial.hermite.hermgauss(n_quad)
    # hermgauss weights satisfy sum(weights) = sqrt(pi), so divide by sqrt(pi)
    # to turn the exponential-quadrature weights into expectation weights.
    return jnp.asarray(np.sqrt(2.0) * nodes), jnp.asarray(weights_unnorm) / jnp.sqrt(jnp.pi)


def _uniform_normal_mixture_log_prob(
        q_mu:jnp.ndarray, # (M,...)
        q_sigma:jnp.ndarray, # (M,...)
        x:jnp.ndarray, # (N,...)
        ) -> jnp.ndarray: # (N,...)
    """Log-density of an equally-weighted mixture of M normals,
    Q(x) = (1/M) * sum_m N(x | q_mu[m], q_sigma[m]).
    """
    (M, *_mushape) = q_mu.shape
    (_, *_xshape) = x.shape
    assert len(_mushape) == len(_xshape)
    for ii, jj in zip(_mushape, _xshape):
        assert ii == jj or ii == 1 or jj == 1
    components = dist.Normal(q_mu[:, None, ...], q_sigma[:, None, ...])
    log_q = jax.nn.logsumexp(components.log_prob(x), axis=0)
    rval = log_q - jnp.log(M)
    assert rval.shape == x.shape
    return rval




def _uniform_normal_mixture_variance(q_mu, q_sigma):
    q_mu = jnp.asarray(q_mu)
    q_sigma = jnp.asarray(q_sigma)
    mix_mean = jnp.mean(q_mu, axis=0)
    mix_second = jnp.mean(q_sigma ** 2 + q_mu ** 2, axis=0)
    return mix_second - mix_mean ** 2


@jax.jit(static_argnames=['n_quad', 'clip'])
def kl_P_SBLN_Q_uniform_normal_mixture(
        p, # batch_shape
        q_mu, # (N,) + batch_shape
        q_sigma, # (N,) + batch_shape
        n_quad: int = 64,
        clip=True,
        ) -> jnp.ndarray: # batch_shape

    p_shape = p.batch_shape
    v_nodes, v_weights = _gauss_hermite(n_quad)

    nodes = v_nodes.reshape((n_quad,) + (1,) * len(p_shape))
    weights_arr = v_weights.reshape((n_quad,) + (1,) * len(p_shape))

    # parameters of all three components
    # where Y corresponds to the [positive] log-domain
    # where X = exp(Y)
    y_mu = p._active_lognormal_mu()
    y_sigma_pos = p.scale_pos
    y_sigma_neg = p.scale_neg
    x_sigma = p.scale_norm

    # y_nodes: hermite points for positive mixture component
    # (note that the values of y_nodes may be negative numbers)
    y_nodes = nodes * y_sigma_pos + y_mu

    # hermite points for middle term
    x_nodes = nodes * x_sigma + p.mu

    exp_y_nodes = jnp.exp(y_nodes)

    integrand_neg = (
            dist.LogNormal(y_mu, y_sigma_neg).log_prob(exp_y_nodes)
            - _uniform_normal_mixture_log_prob(
                jnp.asarray(q_mu), jnp.asarray(q_sigma), -exp_y_nodes))

    integrand_mid = (
            dist.Normal(p.mu, x_sigma).log_prob(x_nodes)
            - _uniform_normal_mixture_log_prob(
                jnp.asarray(q_mu), jnp.asarray(q_sigma), x_nodes))

    integrand_pos = (
            dist.LogNormal(y_mu, y_sigma_pos).log_prob(exp_y_nodes)
            - _uniform_normal_mixture_log_prob(
                jnp.asarray(q_mu), jnp.asarray(q_sigma), exp_y_nodes))

    # batch_shape + (3,)
    integrand = jnp.stack([
        (integrand_neg * weights_arr).sum(axis=0),
        (integrand_mid * weights_arr).sum(axis=0),
        (integrand_pos * weights_arr).sum(axis=0),
        ], axis=-1)

    # batch_shape + (3,)
    mixture_weights = jnp.exp(p._get_log_weights())

    raw_kl = (integrand * mixture_weights).sum(axis=-1)
    if clip:
        return jnp.maximum(0, raw_kl)
    else:
        return raw_kl


def kl_divergence_uniform_normal_mixture_quad_p(
        p,
        q_mu,
        q_sigma,
        n_quad: int = 64,
        ) -> jnp.ndarray:
    # An alternate quadrature route that approximates the expectation over P
    # by a single moment-matched Gaussian with P's mixture mean and variance.
    # This is useful when the SBLN is the more concentrated distribution and
    # component-tailored nodes may not cover the relevant region well.
    p_shape = p.batch_shape
    v_nodes, v_weights = _gauss_hermite(n_quad)
    # print(v_nodes)

    # Moment-match P.
    p_mean = p.mu
    p_std = jnp.sqrt(p.variance)
    assert p_mean.dtype == jnp.float64, p_mean.dtype
    assert p_std.dtype == jnp.float64
    assert q_mu.dtype == jnp.float64
    assert q_sigma.dtype == jnp.float64

    nodes = v_nodes.reshape((-1,) + (1,) * len(p_shape))
    weights_arr = v_weights.reshape((-1,) + (1,) * len(p_shape))
    x = p_mean + p_std * nodes

    # Re-weight the Gauss-Hermite nodes according to the ratio of the density
    # of P to the density of the Gaussian approximation used by the nodes.
    # Without this factor, the quadrature approximates an expectation under
    # the approximating Gaussian instead of under P.
    std_normal_log_prob = -0.5 * (nodes ** 2) - 0.5 * jnp.log(2.0 * jnp.pi)
    log_ratio = p.log_prob(x) - std_normal_log_prob
    importance_weights_unnorm = weights_arr * jnp.exp(log_ratio)
    importance_weights = importance_weights_unnorm / importance_weights_unnorm.sum()

    integrand = (
            p.log_prob(x)
            - _uniform_normal_mixture_log_prob(
                jnp.asarray(q_mu), jnp.asarray(q_sigma), x))
    raw_kl = (importance_weights * integrand).sum(axis=0)
    if raw_kl.shape == ():
        if raw_kl < -1e-6:
            print('x', x)
            print('p.log_prob(x)', p.log_prob(x))
            print('q.log_prob(x)', _uniform_normal_mixture_log_prob(
                jnp.asarray(q_mu), jnp.asarray(q_sigma), x))
            print('importance_weights', importance_weights)
            print('p log_weights', p._get_log_weights())

        if raw_kl < -1e-6:
            for ac in p.arg_constraints:
                print('p', ac, getattr(p, ac))
            print('p_mean', p_mean)
            print('p_std', p_std)
            print('q_mu', ", ".join(str(x) for x in list(q_mu[:20])))
            print('q_sigma', ", ".join(str(x) for x in list(q_sigma[:20])))
        assert float(raw_kl) > -1e-6

    rval = jnp.maximum(raw_kl, 0.0)
    assert rval.shape == p_shape
    return rval


@jax.jit(static_argnames=['n_quad'])
def kl_P_normal_Q_uniform_normal_mixture(p:dist.Normal,
                                         q_mu, q_sigma,
                                         n_quad: int = 64):
    # arrays of shape (n_quad,)
    v_nodes, v_weights = _gauss_hermite(n_quad)

    # arrays that will broadcast over batch_shape
    p_shape = p.batch_shape
    nodes = v_nodes.reshape((-1,) + (1,) * len(p_shape))
    weights = v_weights.reshape((-1,) + (1,) * len(p_shape))

    def f(x):
        return p.log_prob(x) - _uniform_normal_mixture_log_prob(q_mu, q_sigma, x)

    if p_shape == ():
        kl = (weights * f(p.mean + jnp.sqrt(p.variance) * nodes)).sum()
    else:
        raise NotImplementedError()
    return kl


def kl_divergence_uniform_normal_mixture(p, q_mu, q_sigma, n_quad: int = 64):
    assert q_sigma.shape == q_mu.shape
    # assert all(s > 0 for s in q_sigma), q_sigma  ## messes up jit

    q_shape = q_mu.shape
    p_shape = p.batch_shape
    try:
        assert len(q_shape) >= len(p_shape)
        if len(p_shape):
            qp_shape = q_shape[-len(p_shape):]

            assert len(qp_shape) == len(p_shape)
            for qpsi, psi in zip(qp_shape, p_shape):
                assert qpsi == psi or qpsi == 1
    except AssertionError as err:
        err.add_note('q_shape must explicitly include p_shape dims for broadcasting')
        err.add_note(f'p_shape={p_shape}')
        err.add_note(f'q_shape={q_shape}')
        raise

    if isinstance(p, SymmetricBlendedLogNormal):
        rval = kl_P_SBLN_Q_uniform_normal_mixture(
            p, q_mu, q_sigma, n_quad=n_quad)
    elif isinstance(p, dist.Normal):
        rval = kl_P_normal_Q_uniform_normal_mixture(
            p, q_mu, q_sigma, n_quad=n_quad)
    else:
        raise NotImplementedError(p)

    return rval


@kl.dispatch(dist.TransformedDistribution, dist.Normal)
def kl_divergence(p:dist.TransformedDistribution, q:dist.Normal) -> jnp.ndarray:
    if (len(p.transforms) == 1
        and isinstance(p.transforms[0], dist.transforms.AffineTransform)
        and p.transforms[0].loc == 0
        ):
        # p = base_dist * scale
        #
        # kl(p, q) = kl(base_dist * scale, q)
        # = kl(base_dist, q / scale)
        #
        # shift the normal parametrically
        scale = p.transforms[0].scale
        affine_q_loc = q.loc / scale
        affine_q_scale = q.scale / scale
        affine_q = dist.Normal(affine_q_loc, affine_q_scale)
        return kl.kl_divergence(p.base_dist, affine_q)
    else:
        raise NotImplementedError(p)


@kl.dispatch(SymmetricBlendedLogNormal, dist.Normal)
def kl_divergence(p:SymmetricBlendedLogNormal, q:dist.Normal) -> jnp.ndarray:  # noqa: F811
    n_quad = 32 # future env var?
    if q.batch_shape == ():
        n_missing_dims = 1 + len(p.batch_shape)
        rval = kl_divergence_uniform_normal_mixture(
                p,
                q_mu=jnp.array(q.loc)[(None,) * n_missing_dims],
                q_sigma=jnp.array(q.scale)[(None,) * n_missing_dims],
                n_quad=n_quad,
                )
        assert rval.shape == p.batch_shape
        return rval
    elif q.batch_shape == p.batch_shape:
        # TODO adapt kl_divergence_uniform_normal_mixture to work on broadcasted q_mu, q_sigma
        if isinstance(q.loc, (jnp.ndarray, int, float)):
            q_mu = jnp.zeros_like(p.mu) + q.loc
        else:
            raise NotImplementedError(q.loc)
        if isinstance(q.scale, (jnp.ndarray, int, float)):
            q_sigma = jnp.zeros_like(p.mu) + q.scale
        else:
            raise NotImplementedError(q.loc)
        rval = kl_divergence_uniform_normal_mixture(
                p,
                q_mu=q_mu[None, ...],
                q_sigma=q_sigma[None, ...],
                n_quad=n_quad,
                )
        assert rval.shape == p.batch_shape
        return rval
    else:
        raise NotImplementedError()
