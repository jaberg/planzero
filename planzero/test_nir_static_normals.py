import datetime

import pytest

from .nir_static_normals import *


@pytest.mark.xfail
def test_aluminum_co2():
    class ModelDB:
        def __init__(self):
            self.ndarray_groups = {}

        def params_BayesianNormal(self, component_id):
            return {
                'scale': 5000,
                'seed': 1234,
                'num_samples': 25,
                'num_warmup': 25,
                'thinning': 1,
                'sector': IPCC_Sector.Aluminium_Production,
                'ghg': GHG.CO2,
                'data_cutoff': datetime.date(year=2024, month=12, day=31),
                'model_id': 'model_id_foo',
                },

        def save_ndarray_group(self, model_id, component_id, group_id, ndarray_d):
            self.ndarray_groups[model_id, component_id, group_id] = ndarray_d

        def index_ndarray_group(self, model_id, component_id, group_id, ndarray_d_keys):
            raise OSError()

        def load_ndarray_group(self, model_id, component_id, group_id):
            return self.ndarray_groups[model_id, component_id, group_id]

    model_db = ModelDB()

    entrypoint_static_normals_inference(
        payload={
            'component_id': None,
            },
        model_db=model_db)

    KL_NIR_BayesianNormal(model_id=None,
                          component_id=None,
                          NIR_year=2025,
                          emission_year=2023,
                          model_db=model_db,
                         )

