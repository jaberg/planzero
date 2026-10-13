import numpy as np

from .nir_static_normals_site import Static_Normals_2024_12_31


def test_aluminum_co2():
    model = Static_Normals_2024_12_31()
    prenir = model.prediction_scores_prenir_2025_m04()
    print(prenir)
    assert 3.2 < np.isfinite(prenir['weighted_divergence']) < 3.7
