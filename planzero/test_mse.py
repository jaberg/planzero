from .mse import *

from functools import partial

import pytest

from .nir2025 import (
    NIR2025_EmissionsResults,
    latest_data_prediction_algo,
    rolling_average_prediction_algo,
    linear_prediction_algo,
    )


def test_0():
    ref_er = NIR2025_EmissionsResults()

    int_year_range = (2023, 2024)
    mse = blended_RMSE(ref_er, ref_er, int_year_range)
    assert mse == 0 

    int_year_range_2 = (2020, 2024)
    mse = blended_RMSE(ref_er, ref_er, int_year_range_2)
    assert mse == 0 


def test_smoke_blended_scores_nir2025_latest_data():
    ref_er = NIR2025_EmissionsResults()

    RMSEs = []
    for last_training_year in [2020, 2021, 2022]:
        RMSEs.append(
            blended_RMSE_algo(
                latest_data_prediction_algo,
                ref_er,
                last_training_year=last_training_year,
                horizon=2024))
        print(RMSEs[-1])


def test_smoke_blended_scores_nir2025_rolling_average():
    ref_er = NIR2025_EmissionsResults()

    RMSEs = []
    for last_training_year in [2020, 2021, 2022]:
        RMSEs.append(
            blended_RMSE_algo(
                partial(rolling_average_prediction_algo,
                        n_year_average=3),
                ref_er,
                last_training_year=last_training_year,
                horizon=2024))
        print(last_training_year, RMSEs[-1])
    assert 0


def test_smoke_blended_scores_nir2025_linear():
    ref_er = NIR2025_EmissionsResults()

    RMSEs = []
    for last_training_year in [2020, 2021, 2022]:
        RMSEs.append(
            blended_RMSE_algo(
                partial(linear_prediction_algo,
                        n_year_average=3,
                        fit_start_year=1990,
                       ),
                ref_er,
                last_training_year=last_training_year,
                horizon=2024))
        print(last_training_year, RMSEs[-1])
    assert 0
