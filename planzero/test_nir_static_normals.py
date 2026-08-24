import datetime

from .nir_static_normals import *

def test_aluminum_co2():
    class ModelDB:
        def __init__(self):
            self.ndarray_groups = {}

        def params_BayesianNormal(self, component_id):
            return dict(
                scale=5000,
                seed=1234,
                num_samples=100,
                num_warmup=250,
                thinning=1,
                sector=IPCC_Sector.Aluminium_Production,
                ghg=GHG.CO2,
                data_cutoff=datetime.date(year=2024, month=12, day=31),
                ),

        def save_ndarray_group(self, component_id, tag, samples_d):
            self.ndarray_groups[component_id, tag] = samples_d

        def load_ndarray_group(self, component_id, tag):
            return self.ndarray_groups[component_id, tag]

    model_db = ModelDB()

    entrypoint_static_normals_inference(
        payload=dict(
            component_id=None,
            ),
        model_db=model_db)

    KL_NIR_BayesianNormal(component_id=None,
                          NIR_year=2025,
                          emission_year=2023,
                          model_db=model_db,
                         )

