from .ablation_study_scaling import scaling_study_singleton
from .endpoints import completed_prob_registry


def test_0():
    reg = completed_prob_registry()
    ss = scaling_study_singleton()
    ll, uu = ss._baseline_siteinf.predicted_emissions_2050_MtCO2e_bounds_ul
    print(ll, uu)
    assert 580 < ll < 600
    assert 870 < uu < 900

    sn = reg['Static_Normals_2024_12_31']
    sn_ll, sn_uu = sn.predicted_emissions_2050_MtCO2e_bounds_ul
    assert 580 < sn_uu < 600
    print(sn_ll, sn_uu)
    assert 580 < sn_ll < 600
    assert 870 < sn_uu < 900
