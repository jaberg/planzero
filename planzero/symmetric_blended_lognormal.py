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
                 validate_args=None):
        """mu is the mean of the distribution """

        # transition_rate dictates how sharply the mixture switches
        # norm_dominance dictates how wide the "normal" regime is around mu=0
        (self.mu, self.scale_neg, self.scale_norm, self.scale_pos,
         self.transition_rate, self.norm_dominance, self.deadzone) = \
            promote_shapes(
                mu, scale_neg, scale_norm, scale_pos,
                transition_rate, norm_dominance, deadzone)

        batch_shape = jnp.shape(self.mu)
        super().__init__(
            batch_shape=batch_shape,
            validate_args=validate_args)

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


@jax.jit(static_argnames=['n_quad'])
def kl_divergence_uniform_normal_mixture(
        p,  # SymmetricBlendedLogNormal, with 
        q_mu,  # (M,) + p.batch_shape mixture component means
        q_sigma,  # (M,) + p.batch_shape mixture component scales
        n_quad: int = 64,  # Gauss-Hermite quadrature points per component
        ) -> jnp.ndarray:  # p.batch_shape, KL[P || Q], clamped >= 0
    """Deterministic estimate of KL divergence from P to a uniform mixture
    of normals Q(x) = (1/M) * sum_m N(x | q_mu[m], q_sigma[m]).

    Instead of a Monte-Carlo estimate, the KL is decomposed exactly over P's
    three mixture components,

        KL[P || Q] = sum_c w_c * E_{C_c}[ log P - log Q ],

    and each expectation is evaluated with Gauss-Hermite quadrature on the
    component's natural coordinates:
        - normal component:        x = mu + scale_norm * t
        - negative lognormal:      x = -exp(log_loc + scale_neg * t)
        - positive lognormal:      x =  exp(log_loc + scale_pos * t)

    This converges much faster and more accurately than Monte-Carlo sampling,
    and is deterministic.
    """
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

    def f(x):
        return p.log_prob(x) - _uniform_normal_mixture_log_prob(q_mu, q_sigma, x)

    # arrays of shape (n_quad,)
    v_nodes, v_weights = _gauss_hermite(n_quad)
    # arrays that will broadcast over batch_shape
    nodes = v_nodes.reshape((-1,) + (1,) * len(p_shape))
    weights = v_weights.reshape((-1,) + (1,) * len(p_shape))
    if isinstance(p, SymmetricBlendedLogNormal):
        # batch_shape + (3,) in order [neg, norm, pos]
        log_weights = p._get_log_weights()
        mixture_weights = jnp.exp(log_weights)

        log_loc = p._active_lognormal_mu()
        assert log_loc.shape == mixture_weights.shape[:-1]
        assert mixture_weights.shape[-1] == 3

        e_norm = (weights * f(p.mu + p.scale_norm * nodes)).sum(axis=0)
        e_neg = (weights * f(-jnp.exp(log_loc + p.scale_neg * nodes))).sum(axis=0)
        e_pos = (weights * f(jnp.exp(log_loc + p.scale_pos * nodes))).sum(axis=0)

        kl = (mixture_weights[..., 1] * e_norm
              + mixture_weights[..., 0] * e_neg
              + mixture_weights[..., 2] * e_pos)
        assert kl.shape == p_shape
    elif isinstance(p, dist.Normal):
        if p_shape == ():
            kl = (weights * f(p.mean + jnp.sqrt(p.variance) * nodes)).sum()
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError(p)
    return jnp.maximum(kl, 0.0)


@kl.dispatch(dist.TransformedDistribution, dist.Normal)
def kl_divergence(p:dist.TransformedDistribution, q:dist.Normal) -> jnp.ndarray:
    if (len(p.transforms) == 1
        and isinstance(p.transforms[0], dist.transforms.AffineTransform)
        ):
        # p = base_dist * scale + loc
        #
        # kl(p, q) = kl(base_dist, (q - loc) / scale)
        # = kl(base_dist, q * 1/scale - loc/scale)
        #
        # shift the normal parametrically
        loc = p.transforms[0].loc
        scale = p.transforms[0].scale
        shifted_q_loc = q.loc - loc / scale
        shifted_q_scale = q.scale / scale
        base_q = dist.Normal(shifted_q_loc, shifted_q_scale)
        return kl_divergence(p.base_dist, base_q)
    else:
        raise NotImplementedError(p)


@kl.dispatch(SymmetricBlendedLogNormal, dist.Normal)
def kl_divergence(p:SymmetricBlendedLogNormal, q:dist.Normal) -> jnp.ndarray:  # noqa: F811
    if q.batch_shape == ():
        n_missing_dims = 1 + len(p.batch_shape)
        rval = kl_divergence_uniform_normal_mixture(
                p,
                q_mu=jnp.array(q.loc)[(None,) * n_missing_dims],
                q_sigma=jnp.array(q.scale)[(None,) * n_missing_dims],
                )[0]
    else:
        raise NotImplementedError()
    assert rval.shape == ()
    return rval
