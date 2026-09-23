import functools

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import numpyro.distributions as dist
from numpyro.distributions import Distribution, constraints
from numpyro.distributions.util import promote_shapes


class SymmetricBlendedLogNormal(Distribution):
    """A 3-element mixture model implementing a symmatric version of
    log-normal with a dead-zone around 0.
    """
    arg_constraints = {
        "mu": constraints.real,
        "scale_neg": constraints.positive,
        "scale_norm": constraints.positive,
        "scale_pos": constraints.positive,
        "transition_rate": constraints.positive,
        "norm_dominance": constraints.real,
        "deadzone": constraints.positive,
    }
    
    support = constraints.real
    reparametrized_params = ["mu"]
    
    @classmethod
    def rolloff_relerr(cls, mu, rolloff, relerr, **kwargs):
        """
        mu: the mean of the distribution it's "closeness" to 0 is measured relative to `rolloff`.
        rolloff: if small relative to abs(mu), the distribution is strongly one-sided. Considering mu as a parameter, it controls how sharply the distribution might change as mu approaches 0.
        relerr: if small relative to 1, the distribution is concentrated.
        """
        # relerr <= 0 will trigger constraints violation, catch earlier here
        assert relerr > 0

        assert rolloff > 0 # actually 0 maybe should work? not tested

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
        super(SymmetricBlendedLogNormal, self).__init__(
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
        
        logits = jnp.stack([logit_neg, logit_norm, logit_pos], axis=-1)
        return jax.nn.log_softmax(logits, axis=-1)

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

    def sample(self, key, sample_shape=()):
        shape = sample_shape + self.batch_shape
        assert shape == sample_shape
        key_comp, key_neg, key_norm, key_pos = jrandom.split(key, 4)
        
        # 1. Sample which component is active based on weights
        log_weights = self._get_log_weights()
        try:
            comp = dist.Categorical(logits=log_weights, validate_args=False).sample(
                key_comp, sample_shape)
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
        return jnp.where(comp == 0, samp_neg, 
                 jnp.where(comp == 1, samp_norm, samp_pos))

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
            
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
        
        return jax.nn.logsumexp(components_log_prob, axis=-1)


# @functools.cache  ## messes up jax.jit below
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
        q_mu, # (M,)
        q_sigma, # (M,)
        x, # (N,)
        ) -> jnp.ndarray: # (N,)
    """Log-density of an equally-weighted mixture of M normals,
    Q(x) = (1/M) * sum_m N(x | q_mu[m], q_sigma[m]).
    """
    M, = q_mu.shape
    components = dist.Normal(q_mu[:, None], q_sigma[:, None])
    log_q = jax.nn.logsumexp(components.log_prob(x), axis=0)
    return log_q - jnp.log(M)


@jax.jit
def kl_divergence_uniform_normal_mixture(
        p,  # SymmetricBlendedLogNormal, with batch_shape == ()
        q_mu,  # (M,) mixture component means
        q_sigma,  # (M,) mixture component scales
        n_quad: int = 64,  # Gauss-Hermite quadrature points per component
        ) -> jnp.ndarray:  # scalar KL[P || Q], clamped at 0
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

    def f(x):
        return p.log_prob(x) - _uniform_normal_mixture_log_prob(q_mu, q_sigma, x)

    nodes, weights = _gauss_hermite(n_quad)
    if isinstance(p, SymmetricBlendedLogNormal):
        log_weights = p._get_log_weights() # (3,) in order [neg, norm, pos]
        w_neg, w_norm, w_pos = jnp.exp(log_weights)
        log_loc = p._active_lognormal_mu()

        e_norm = (weights * f(p.mu + p.scale_norm * nodes)).sum()
        e_neg = (weights * f(-jnp.exp(log_loc + p.scale_neg * nodes))).sum()
        e_pos = (weights * f(jnp.exp(log_loc + p.scale_pos * nodes))).sum()

        kl = w_norm * e_norm + w_neg * e_neg + w_pos * e_pos
    elif isinstance(p, dist.Normal):
        kl = (weights * f(p.mean + jnp.sqrt(p.variance) * nodes)).sum()
    else:
        raise NotImplementedError(p)
    return jnp.maximum(kl, 0.0)
