from .annual_emission_results import (
    AER_NotResultKey,
    AnnualEmissionResults,
    aer_parse_result_key,
)
from .barriers import Barrier

# TODO: move this here
from .prob_bovaer import batch_rollout_barriers
from .strategies.strategy2 import Strategy2


def compute_annual_emission_results(
        strategies: dict[str, Strategy2],
        barriers: dict[str, Barrier],
        ) -> AnnualEmissionResults:
    """
    Build up a full NIR AnnualEmissionResults object from the contributions
    of different simulation groups.
    """
    n_samples = 500  # TODO calculate this below
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

    for simgroup, des in des_by_simgroup.items():
        results = batch_rollout_barriers(
                strategies=None,
                barriers=None,
                elements=des,
                n_samples=n_samples)
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
            n_samples=n_samples)
    return rval
