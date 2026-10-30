from .nir_static_normals_site import Static_Normals_2024_12_31
from .scaling_site import ScalingSiteInference


def test_avoid_getting_killed():
    site_infs = [
            Static_Normals_2024_12_31(),
            ScalingSiteInference()]
    for si in site_infs:
        print('compute', si.registry_basename())
        si.registry_compute_model()
        for elem_id in si.strategies:
            print('compute', si.registry_basename(), elem_id)
            si.registry_compute_model(
                    ablate_strategy_elem_id=elem_id)
