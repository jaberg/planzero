import jax.random as jrandom

from .annual_emission_results import (
    AER_NotResultKey,
    AnnualEmissionResults,
    aer_parse_result_key,
)
from .barriers import Barrier
from .prob_bovaer import batch_rollout_elements
from .strategies.strategy2 import Strategy2


def compute_annual_emission_results(
        strategies: dict[str, Strategy2],
        barriers: dict[str, Barrier],
        seed_int=12345,
        ) -> AnnualEmissionResults:
    """
    Build up a full NIR AnnualEmissionResults object from the contributions
    of different simulation groups.
    """
    ktCO2e_sample = {}
    years = None

    all_des = {}
    for name, de in strategies.items():
        all_des["Strategy", name] = de
    for name, de in barriers.items():
        all_des["Barrier", name] = de

    des_by_simgroup = {}
    for key, de in all_des.items():
        #assert de.simgroup is not None, (key, de)
        des_by_simgroup.setdefault(de.simgroup, {})[key] = de

    jrkey = jrandom.key(seed_int)
    for simgroup, des in des_by_simgroup.items():
        jrkey, tmpkey = jrandom.split(jrkey)
        results = batch_rollout_elements(
                elements=des,
                n_samples=500,
                jrkey=tmpkey)
        if years is None:
            years = results['years']
        for result_type in ['xs', 'ys', 'constants']:
            for key, val in results[result_type].items():
                try:
                    sector, ghg, activity = aer_parse_result_key(key)
                    assert (sector, ghg, activity) not in ktCO2e_sample
                    ktCO2e_sample[sector, ghg, activity] = val
                except AER_NotResultKey:
                    pass
    assert years is not None
    rval = AnnualEmissionResults(
            ktCO2e_sample=ktCO2e_sample,
            years=years,
            n_samples=16000)
    return rval


# TODO: compute_annual_emission_delta
# works by looking at all of the dependencies, and ignoring things that are independent of a particular strategy
# uses the same random key for computations both with and without the strategy
