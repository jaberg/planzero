from .example_model_element_class_parameters import (
    Case,
    PropertyModelElement,
    property_model,
)


def test_vars_are_present():
    elem_1 = PropertyModelElement(n_samples=1)
    elem_2 = PropertyModelElement(n_samples=2)
    assert elem_1._n_samples == 1
    assert elem_1.n_samples == 1
    assert elem_2._n_samples == 2
    assert elem_2.n_samples == 2

def test_add_elements():
    model = property_model()
    elem_1 = model.model_elements['PropertyModelElement(n_samples=1)']
    elem_2 = model.model_elements['PropertyModelElement(n_samples=2)']

    general_nd = model.mv.general_nd
    for n_samples_m1, elem in enumerate([elem_1, elem_2]):
        n_samples = n_samples_m1 + 1
        assert general_nd[elem.prop_key].definition_metadata.value_type.shape[0] == n_samples
        for case in Case:
            assert general_nd[elem.dict_keys[case]].definition_metadata.value_type.shape[0] == n_samples
