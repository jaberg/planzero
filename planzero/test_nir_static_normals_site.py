import numpy as np

from .nir_static_normals import weighted_KL_score
from .nir_static_normals_site import Static_Normals_2024_12_31


def test_aluminum_co2():
    model = Static_Normals_2024_12_31()
    weighted_div, abs_kts, kls = weighted_KL_score(year=2023, model_id=model.model_id)
    print(weighted_div)
    assert np.isfinite(weighted_div)
