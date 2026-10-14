import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpyro.distributions as dist

from .symmetric_blended_lognormal import (
    SymmetricBlendedLogNormal,
    kl_divergence_uniform_normal_mixture,
    kl_divergence_uniform_normal_mixture_quad_p,
    kl_divergence_uniform_normal_mixture_quad_q,
)


def test_gh_invalid():
    sbln = SymmetricBlendedLogNormal.rolloff_relerr(
            mu=1000,
            rolloff=100,
            relerr=.1)
    print(sbln._get_log_weights())
    comp = sbln.sample(jrandom.key(12345), sample_shape=())
    print(comp)
    # This is failing on GH but working on my computer.
    # Looking at source online, suggests that it should be failing
    # because "real" constraint on Categorical forbids -inf.
    # I am trying again with validate_args=False on the Categorical
    # on line 110.


def test_kl_zero_for_perfect_normal_fit():
    # With mu=0 and a large deadzone, P reduces to N(0, 1.0) exactly.
    p = SymmetricBlendedLogNormal(
            mu=0, scale_neg=.1, scale_norm=1.0, scale_pos=.1,
            transition_rate=1.0, norm_dominance=10.0, deadzone=1)
    kl = kl_divergence_uniform_normal_mixture(
            p, q_mu=jnp.array([0.0]), q_sigma=jnp.array([1.0]))
    assert jnp.all(jnp.isfinite(kl))
    assert float(kl) < 1e-6, kl


def test_kl_finite_and_positive():
    cases = [
        (1000, 100, .1),   # strongly positive, one-sided
        (-1000, 100, .1),  # strongly negative, one-sided
        (0.0, 100, 2.0),   # near zero, wide
        (0.5, 10, 1.0),
    ]
    for mu, rolloff, relerr in cases:
        p = SymmetricBlendedLogNormal.rolloff_relerr(
                mu=mu, rolloff=rolloff, relerr=relerr)
        q_mu = jnp.array([mu, mu - rolloff, mu + rolloff])
        q_sigma = jnp.array([relerr * abs(mu) + 1, 10.0, 10.0])
        kl = kl_divergence_uniform_normal_mixture(p, q_mu, q_sigma)
        assert jnp.all(jnp.isfinite(kl)), (mu, kl)
        assert float(kl) >= 0, (mu, kl)


def test_kl_converges_with_quadrature_order():
    p = SymmetricBlendedLogNormal.rolloff_relerr(
            mu=50, rolloff=5, relerr=.2)
    q_mu = jnp.array([30.0, 50.0, 70.0, 90.0])
    q_sigma = jnp.array([1.0, 12.0, 12.0, 120.0])
    kl_16 = kl_divergence_uniform_normal_mixture(p, q_mu, q_sigma, n_quad=16)
    kl_32 = kl_divergence_uniform_normal_mixture(p, q_mu, q_sigma, n_quad=32)
    kl_64 = kl_divergence_uniform_normal_mixture(p, q_mu, q_sigma, n_quad=64)
    kl_128 = kl_divergence_uniform_normal_mixture(p, q_mu, q_sigma, n_quad=128)
    assert bool(jnp.all(jnp.isfinite(jnp.stack([kl_32, kl_64, kl_128]))))
    assert float(jnp.abs(kl_64 - kl_128)) < 1e-11, (kl_64, kl_128)
    assert float(jnp.abs(kl_32 - kl_64)) < 1e-10, (kl_32, kl_64)
    assert float(jnp.abs(kl_16 - kl_32)) < 1e-8, (kl_16, kl_32)


def test_kl_agrees_with_monte_carlo():

    rng_key = jrandom.key(1234)
    p = SymmetricBlendedLogNormal.rolloff_relerr(
            mu=30, rolloff=3, relerr=.2)
    q_mu = jnp.array([15.0, 30.0, 45.0])
    q_sigma = jnp.array([16.0, 16.0, 16.0])

    kl_q = kl_divergence_uniform_normal_mixture_quad_q(p, q_mu, q_sigma, n_quad=128)
    kl_p = kl_divergence_uniform_normal_mixture_quad_p(p, q_mu, q_sigma, n_quad=128)
    kl_blend = kl_divergence_uniform_normal_mixture(p, q_mu, q_sigma, n_quad=128)

    x = p.sample(rng_key, sample_shape=(1_000_000,))
    log_p = p.log_prob(x)
    M, = q_mu.shape
    log_q = dist.Normal(q_mu[:, None], q_sigma[:, None]).log_prob(x)
    log_q = jax.nn.logsumexp(log_q, axis=0) - jnp.log(M)
    kl_mc = (log_p - log_q).mean()

    for kl in [kl_p, kl_q, kl_blend]:
        assert jnp.isfinite(kl) and jnp.isfinite(kl_mc)
        assert float(jnp.abs(kl - kl_mc)) < 0.02, (kl, kl_mc)
        assert 1.85 < kl < 1.86


def test_affine_dispatch():
    p1 = SymmetricBlendedLogNormal.rolloff_relerr(
            mu=jnp.array([1, 2, 3, 4]),
            rolloff=3, relerr=.2)
    p10 = dist.TransformedDistribution(
            p1, dist.transforms.AffineTransform(loc=0, scale=10))

    q1 = dist.Normal(3, 1)
    q10 = dist.Normal(30, 10)

    kl1 = dist.kl_divergence(p1, q1)
    kl10 = dist.kl_divergence(p10, q10)

    print(kl1)
    print(kl10)

    assert jnp.allclose(kl1, kl10)


def test_kl_divergence_shape_p_being_sbln_vec_q_being_normal():
    p = SymmetricBlendedLogNormal.rolloff_relerr(
            mu=jnp.array([10, 20, 30, 40]),
            rolloff=3, relerr=.2)
    q = dist.Normal(30, 1)

    kl_div = dist.kl_divergence(p, q)
    assert kl_div.shape == (4,)

    a, b, c, d = [float(e) for e in kl_div]

    assert a > b > c < d


def test_shape_understanding():
    p = dist.Normal(0, 1)
    assert p.shape() == ()
    assert p.batch_shape == ()
    assert p.event_shape == ()

    p = dist.Normal(jnp.array([0]), 1)
    assert p.shape() == (1,)
    assert p.batch_shape == (1,)
    assert p.event_shape == ()

    p = dist.Normal(jnp.array([0, 0]), 1)
    assert p.shape() == (2,)
    assert p.batch_shape == (2,)
    assert p.event_shape == ()


def test_int_basic_indexing():
    p = SymmetricBlendedLogNormal.rolloff_relerr(
            mu=jnp.array([10, 20, 30, 40]),
            rolloff=3, relerr=.2)
    p0 = p[0]
    p3 = p[3]

    try:
        p[4]
        assert 0
    except IndexError:
        pass

    p3alt = p[-1]

    assert p0.mu == 10
    assert p3.mu == 40
    assert p3alt.mu == 40
