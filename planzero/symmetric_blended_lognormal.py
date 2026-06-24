import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpyro
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
        return cls(
            mu=mu,
            scale_neg=relerr / 3,
            scale_norm=rolloff * relerr / 3,
            scale_pos=relerr / 3,
            transition_rate=10.0 / rolloff,
            norm_dominance=10.0,
            deadzone=rolloff / 10,
            **kwargs)
        
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
        key_comp, key_neg, key_norm, key_pos = jrandom.split(key, 4)
        
        # 1. Sample which component is active based on weights
        log_weights = self._get_log_weights()
        comp = dist.Categorical(logits=log_weights).sample(
            key_comp, sample_shape)
        
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
