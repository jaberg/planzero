import datetime
import time

from .ablation_study_scaling import scaling_study_singleton
from .endpoints import completed_prob_registry
from .nir_static_normals import model_id_from_data_cutoff, weighted_KL_score


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


def test_prediction_scores_prenir_2025_m04():
    reg = completed_prob_registry()
    scaling = reg['ScalingStudy_All_Strategies']
    prenir_results = scaling.prediction_scores_prenir_2025_m04()
    weighted_divergence = prenir_results['weighted_divergence']
    print(weighted_divergence)
    assert 3.3 < weighted_divergence < 3.4, weighted_divergence

    data_cutoff = datetime.date(year=2024, month=12, day=31)
    model_id = model_id_from_data_cutoff(data_cutoff)
    t0 = time.time()
    weighted_divergence_, _, _ = weighted_KL_score(
            year=2023,
            model_id=model_id,
            )
    t1 = time.time()
    print(weighted_divergence_, (t1 - t0))
    assert abs(weighted_divergence - weighted_divergence_) < .1
