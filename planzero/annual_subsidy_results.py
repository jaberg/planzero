from .enums import PT, Activity, SubsidyProgram


class SubsidyResults:
    # Don't inherit from BaseModel because these can be big
    # and the possible aliasing between dictionary entries and cache
    # system is deliberate.

    CAD_sample: dict[tuple[SubsidyProgram,
                           # ProgramComponent,
                           PT,
                           Activity],
                     object] # n_samples x n_years

    years: list[int]
    n_samples: int
