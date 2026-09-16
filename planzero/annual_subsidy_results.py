from .enums import PT, Activity, GovernmentProgram


class DetailedAnnualProgramBalances:
    # Don't inherit from BaseModel because these can be big
    # and the possible aliasing between dictionary entries and cache
    # system is deliberate.

    CAD_sample: dict[tuple[GovernmentProgram,
                           # ProgramComponent,
                           PT,
                           Activity],
                     object] # n_samples x n_years

    years: list[int]
    n_samples: int


class NationalAnnualProgramBalances:
    # Don't inherit from BaseModel because these can be big
    # and the possible aliasing between dictionary entries and cache
    # system is deliberate.

    CAD_sample: dict[GovernmentProgram,
                     object] # n_samples x n_years

    years: list[int]
    n_samples: int
