from .enums import GHG, PT, Activity, IPCC_Sector


class AnnualEmissionResults:
    # Don't inherit from BaseModel because these can be big
    # and the possible aliasing between dictionary entries and cache
    # system is deliberate.

    ktCO2e_sample: dict[tuple[IPCC_Sector, GHG, Activity],
                        object] # n_samples x n_years
    years: list[int]
    n_samples: int

    def __init__(self, *, ktCO2e_sample, years, n_samples):
        self.ktCO2e_sample = ktCO2e_sample
        self.years = years
        self.n_samples = n_samples

    def __getitem__(self, item):
        if isinstance(item, IPCC_Sector):
            rval = {(ghg, activity): sample
                    for ((sector, ghg, activity), sample) in self.ktCO2e_sample.items()
                    if sector == item}
        else:
            raise NotImplementedError(item)
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
            object] # n_samples x n_years
    years: list[int]
    n_samples: int

    def __init__(self, *, ktCO2e_sample_pt, years, n_samples):
        self.ktCO2e_sample_pt = ktCO2e_sample_pt
        self.years = years
        self.n_samples = n_samples
