from .ablation_study_scaling import scaling_study_singleton
from .endpoints import completed_prob_registry


def test_predicted_emissions():
    reg = completed_prob_registry()
    ss = scaling_study_singleton()
    ll, uu = ss._baseline_siteinf.predicted_emissions_2050_MtCO2e_bounds_ul
    print(ll, uu)
    assert 580 < ll < 600
    assert 870 < uu < 900

    sn = reg['Static_Normals_2024_12_31']
    sn_ll, sn_uu = sn.predicted_emissions_2050_MtCO2e_bounds_ul
    print(sn_ll, sn_uu)
    assert 580 < sn_ll < 600
    assert 870 < sn_uu < 900


def test_bovaer_cost_per_ton():
    reg = completed_prob_registry()
    scaling = reg['ScalingStudy_All_Strategies']
    ll, uu = scaling.ablation_study.cost_per_tCO2e('Scale_Bovaer')
    assert 185 < ll < uu < 205
