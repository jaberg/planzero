from .endpoints import *

def test_bovaer_production_emission_factors():
    # this one is causing trouble at time of writing

    assert '/models/sim/Scaling/barriers/Bovaer_Production_Emission_Factors/' in endpoints()
