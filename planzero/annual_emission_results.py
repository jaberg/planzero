import numpy as np

from .enums import GHG, PT, Activity, IPCC_Sector


def aer_result_key(sector, ghg, activity):
    """Dynamic elements wishing to register
    annual emissions results should do so by defining variables
    by these names as "xs", "ys" or "constants".
    """
    return f'AER_{sector.value}_{ghg.value}_{activity.value}'


class AER_NotResultKey(Exception):
    pass


def aer_parse_result_key(key: str):
    try:
        aer, sector, ghg, activity = key.split('_')
        assert aer == 'AER'
        sector = IPCC_Sector(sector)
        ghg = GHG(ghg)
        activity = Activity(activity)
        return sector, ghg, activity
    except (AssertionError, ValueError, IndexError,
            AttributeError, # in case key is a tuple
            ):
        raise AER_NotResultKey()


class AnnualEmissionResults:
    # Don't inherit from BaseModel because these can be big
    # and the possible aliasing between dictionary entries and cache
    # system is deliberate.

    ktCO2e_sample: dict[tuple[IPCC_Sector, GHG, Activity],
                        np.ndarray] # n_years, n_samples
    years: list[int]
    n_samples: int

    def __init__(self, *, ktCO2e_sample, years, n_samples):
        self.ktCO2e_sample = ktCO2e_sample
        self.years = years
        self.n_samples = n_samples
        for val in ktCO2e_sample.values():
            n_years, _n_samples = val.shape
            assert n_years in (1, len(years))
            assert _n_samples in (1, n_samples), (_n_samples, n_samples)

    def __getitem__(self, item):
        if isinstance(item, IPCC_Sector):
            rval = {(ghg, activity): sample
                    for ((sector, ghg, activity), sample) in self.ktCO2e_sample.items()
                    if sector == item}
        else:
            raise NotImplementedError(item)
        return rval

    def missing_sector_ghgs(self) -> set[tuple[IPCC_Sector, GHG]]:
        rval = {(sector, ghg) for sector in IPCC_Sector for ghg in GHG}
        for sector, ghg, _ in self.ktCO2e_sample:
            rval.discard((sector, ghg))
        return rval


class AnnualEmissionResults_PT:
    # Don't inherit from BaseModel because these can be big
    # and the possible aliasing between dictionary entries and cache
    # system is deliberate.

    ktCO2e_sample_pt: dict[
            tuple[
                IPCC_Sector,
                GHG,
                PT,
                Activity],
            np.ndarray] # n_years x n_samples
    years: list[int]
    n_samples: int

    def __init__(self, *, ktCO2e_sample_pt, years, n_samples):
        self.ktCO2e_sample_pt = ktCO2e_sample_pt
        self.years = years
        self.n_samples = n_samples


def aer_sectors(aer:AnnualEmissionResults) -> set[IPCC_Sector]:
    return {sector for (sector, _, _) in aer.ktCO2e_sample}


def aer_total_with_LULUCF(
        aer:AnnualEmissionResults,
        ) -> np.ndarray: # (n_years, n_samples)
    return sum(aer.ktCO2e_sample.values())


def total_prediction_CI(
        aer: AnnualEmissionResults,
        year:int=2050,
        CI:tuple=(.025, .975),
        ) -> tuple[float, float]:
    missing = aer.missing_sector_ghgs()
    if missing:
        assert 0, missing
    total = aer_total_with_LULUCF(aer)
    n_years, _ = total.shape
    if n_years > 1:
        year_idx = list(aer.years).index(year)
        lower, upper = np.quantile(
                total[year_idx],
                q=CI)
    else:
        lower, upper = np.quantile(
                total[0, :],
                q=CI)
    return lower, upper
