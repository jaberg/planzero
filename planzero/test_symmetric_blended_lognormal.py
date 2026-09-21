import jax.random as jrandom

from .symmetric_blended_lognormal import SymmetricBlendedLogNormal


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
