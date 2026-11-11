import datetime
import json
import re

import pytest

from .dependency_explorer import (
    build_mcmc_group_dependency_data,
    list_mcmc_groups,
    render_mcmc_group_html_explorer,
)
from .scaling_model import Scaling_ScenarioModel

MCMC_GROUP = "enteric_and_other_product_mfg"

_METHODS = {"prepare", "annual_step", "postprocess"}


@pytest.fixture(scope="module")
def scaling_model():
    return Scaling_ScenarioModel(
            data_cutoff=datetime.date(year=2024, month=12, day=31),
            sectors=[],
            num_warmup=1,
            num_samples=1,
            just_efman_testing=True,
            )


@pytest.fixture(scope="module")
def data(scaling_model):
    return build_mcmc_group_dependency_data(scaling_model, MCMC_GROUP)


def _variables_by_name(data, name):
    return [v for v in data["variables"] if v["name"] == name]


def _variable_by_name(data, name):
    matches = _variables_by_name(data, name)
    assert len(matches) == 1, (name, matches)
    return matches[0]


def _occurrence(data, occurrence_id):
    matches = [o for o in data["occurrences"] if o["id"] == occurrence_id]
    assert len(matches) == 1, occurrence_id
    return matches[0]


def _edge_by_variable(data, variable_id):
    matches = [e for e in data["edges"] if e["variable"] == variable_id]
    assert len(matches) == 1, (variable_id, matches)
    return matches[0]


def test_list_mcmc_groups(scaling_model):
    groups = list_mcmc_groups(scaling_model)
    assert groups == sorted(groups)
    assert len(groups) == len(set(groups))
    assert MCMC_GROUP in groups
    assert None not in groups


def test_payload_shape(data):
    summary = data["summary"]
    assert data["group"] == MCMC_GROUP
    assert summary["n_elements"] == len(data["elements"]) == 12
    assert summary["n_variables"] == len(data["variables"])
    assert summary["n_occurrences"] == len(data["occurrences"])
    assert summary["n_edges"] == len(data["edges"])
    assert summary["n_cross_edges"] == sum(
            1 for edge in data["edges"] if edge["cross_element"])

    element_ids = [element["id"] for element in data["elements"]]
    variable_ids = [variable["id"] for variable in data["variables"]]
    occurrence_ids = [o["id"] for o in data["occurrences"]]
    assert len(element_ids) == len(set(element_ids))
    assert len(variable_ids) == len(set(variable_ids))
    assert len(occurrence_ids) == len(set(occurrence_ids))

    element_id_set = set(element_ids)
    variable_id_set = set(variable_ids)
    occurrence_id_set = set(occurrence_ids)

    for element in data["elements"]:
        assert element["id"] in element_id_set
        assert set(element["methods"]) == _METHODS
        for method in element["methods"].values():
            for occurrence_id in method["defines"]:
                assert occurrence_id in occurrence_id_set
            for occurrence_id in method["accesses"]:
                assert occurrence_id in occurrence_id_set

    for occurrence in data["occurrences"]:
        assert occurrence["element"] in element_id_set
        assert occurrence["method"] in _METHODS
        assert occurrence["direction"] in {"define", "access"}
        assert occurrence["kind"] in {"general", "annual", "carry"}
        assert occurrence["variable_ids"]
        for variable_id in occurrence["variable_ids"]:
            assert variable_id in variable_id_set

    for variable in data["variables"]:
        assert variable["id"] in variable_id_set
        assert variable["defines"] or variable["accesses"]
        for occurrence_id in variable["defines"] + variable["accesses"]:
            assert occurrence_id in occurrence_id_set

    for edge in data["edges"]:
        assert edge["source"] in occurrence_id_set
        assert edge["target"] in occurrence_id_set
        assert edge["variable"] in variable_id_set
        assert edge["source_element"] in element_id_set
        assert edge["target_element"] in element_id_set
        assert edge["cross_element"] == (
                edge["source_element"] != edge["target_element"])


def test_carry_lifecycle_is_normalized(data):
    """InitialCarry/ThisCarry/NextCarry/FinalCarry collapse to one variable."""
    variable = _variable_by_name(data, "bovine_fraction_on_bovaer")
    assert variable["kind"] == "carry"
    assert variable["cross_element"] is True
    assert variable["self_element"] is False
    assert variable["define_elements"] == ["Bovaer_Farm_Subsidy"]
    assert variable["access_elements"] == ["Bovaer_Adoption_Limit"]
    assert len(variable["defines"]) == 1
    assert len(variable["accesses"]) == 1

    edge = _edge_by_variable(data, variable["id"])
    assert edge["source_element"] == "Bovaer_Farm_Subsidy"
    assert edge["target_element"] == "Bovaer_Adoption_Limit"
    assert edge["backward"] is True

    # If the lifecycle wrappers were not normalized, the define occurrence
    # and the this_carry access occurrence would refer to different
    # variables and this edge would not exist.
    define_occurrence = _occurrence(data, variable["defines"][0])
    access_occurrence = _occurrence(data, variable["accesses"][0])
    assert define_occurrence["direction"] == "define"
    assert access_occurrence["direction"] == "access"
    assert set(define_occurrence["variable_ids"]) == {variable["id"]}
    assert set(access_occurrence["variable_ids"]) == {variable["id"]}


def test_carry_access_occurrence_is_backward(data):
    variable = _variable_by_name(data, "max_possible_fraction_on_bovaer")
    access_occurrence = _occurrence(data, variable["accesses"][0])
    assert access_occurrence["element"] == "Bovaer_Farm_Subsidy"
    assert access_occurrence["method"] == "annual_step"
    assert access_occurrence["direction"] == "access"
    assert access_occurrence["name"] == "max_fraction_on_bovaer"
    assert access_occurrence["backward"] is True

    define_occurrence = _occurrence(data, variable["defines"][0])
    assert define_occurrence["backward"] is False


def test_non_carry_occurrences_are_not_backward(data):
    backward = [o for o in data["occurrences"] if o["backward"]]
    assert len(backward) == 2
    assert {o["name"] for o in backward} == {
            "bovine_fraction_on_bovaer",
            "max_fraction_on_bovaer",
            }
    assert all(o["method"] == "annual_step" for o in backward)


def test_cross_and_self_flags(data):
    bovaer_budget = _variable_by_name(data, "bovaer_budget")
    assert bovaer_budget["cross_element"] is True
    assert bovaer_budget["self_element"] is False
    edge = _edge_by_variable(data, bovaer_budget["id"])
    assert edge["source_element"] == "Scale_Bovaer_2030"
    assert edge["target_element"] == "Bovaer_Farm_Subsidy"
    assert edge["cross_element"] is True

    max_increase = _variable_by_name(data, "max_increase_fraction")
    assert max_increase["cross_element"] is False
    assert max_increase["self_element"] is True
    self_edge = _edge_by_variable(data, max_increase["id"])
    assert self_edge["source_element"] == "Bovaer_Adoption_Limit"
    assert self_edge["target_element"] == "Bovaer_Adoption_Limit"
    assert self_edge["cross_element"] is False


def test_matrix_cells_aggregate_distinct_variables(data):
    variables_by_id = {v["id"]: v for v in data["variables"]}
    edges_by_id = {e["id"]: e for e in data["edges"]}

    assert data["matrix"]
    for cell in data["matrix"]:
        assert cell["count"] == len(cell["variable_ids"])
        assert cell["count"] == len(set(cell["variable_ids"]))
        assert cell["edge_ids"]
        for edge_id in cell["edge_ids"]:
            edge = edges_by_id[edge_id]
            assert edge["source_element"] == cell["source"]
            assert edge["target_element"] == cell["target"]
            assert edge["variable"] in cell["variable_ids"]
        assert set(cell["source_methods"]) <= _METHODS
        assert set(cell["target_methods"]) <= _METHODS
        for variable_id in cell["variable_ids"]:
            assert variable_id in variables_by_id

    bovaer_budget = _variable_by_name(data, "bovaer_budget")
    cell = next(
            cell
            for cell in data["matrix"]
            if cell["source"] == "Scale_Bovaer_2030"
            and cell["target"] == "Bovaer_Farm_Subsidy")
    assert cell["count"] == 1
    assert cell["variable_ids"] == [bovaer_budget["id"]]

    self_cell = next(
            cell
            for cell in data["matrix"]
            if cell["source"] == "Bovaer_Adoption_Limit"
            and cell["target"] == "Bovaer_Adoption_Limit")
    max_increase = _variable_by_name(data, "max_increase_fraction")
    assert self_cell["variable_ids"] == [max_increase["id"]]


def test_summary_rankings(data):
    summary = data["summary"]
    fanouts = [
            variable["fan_out"]
            for variable in summary["top_fanout_variables"]
            ]
    assert fanouts == sorted(fanouts, reverse=True)
    assert fanouts
    assert summary["top_fanout_variables"][0]["name"] == "annual_headcounts"

    fanins = [element["fan_in"] for element in summary["top_fanin_elements"]]
    assert fanins == sorted(fanins, reverse=True)
    assert fanins

    fanouts_elements = [
            element["fan_out"]
            for element in summary["top_fanout_elements"]
            ]
    assert fanouts_elements == sorted(fanouts_elements, reverse=True)


def test_site_model_element_metadata(data, scaling_model):
    elements_by_id = {element["id"]: element for element in data["elements"]}

    site_id = "Bovaer_Production_Emissions"
    site_element = elements_by_id[site_id]
    model_element = scaling_model.model_elements[site_id]
    assert site_element["site_model_element"] is True
    assert site_element["pretty_name"] == model_element.pretty_name
    assert site_element["display"] == model_element.pretty_name
    assert site_element["short_description"]
    assert "<" not in site_element["short_description"]
    assert "embedded/production emission" in site_element["short_description"]

    scale_element = elements_by_id["Scale_Bovaer_2030"]
    assert scale_element["display"] == "Scale Bovaer"
    assert scale_element["pretty_name"] == "Scale Bovaer"

    non_site_id = "NIR2024_ModelElement_Enteric Fermentation_CH4"
    non_site_element = elements_by_id[non_site_id]
    assert non_site_element["site_model_element"] is False
    assert non_site_element["short_description"] is None
    assert non_site_element["pretty_name"] == non_site_element["display"]


def test_render_explorer_html(scaling_model, data):
    rendered = render_mcmc_group_html_explorer(
            scaling_model,
            MCMC_GROUP,
            model_name="Scaling")

    assert 'data-mcmc-explorer' in rendered
    assert 'class="mcmc-explorer-data"' in rendered
    assert MCMC_GROUP in rendered
    assert "cytoscape@3.30.2" in rendered
    assert "/assets/js/mcmc-explorer.js" in rendered
    assert "/assets/css/mcmc-explorer.css" in rendered

    match = re.search(
            r'<script type="application/json" class="mcmc-explorer-data">'
            r'(?P<json>.*?)</script>',
            rendered,
            re.DOTALL)
    assert match, rendered
    payload = json.loads(match.group("json"))
    assert payload["group"] == MCMC_GROUP
    assert payload["model"] == "Scaling"
    assert payload["summary"] == data["summary"]
    assert payload["elements"] == data["elements"]
    assert payload["edges"] == data["edges"]


def test_unknown_group_raises(scaling_model):
    with pytest.raises(ValueError, match="unknown mcmc_group"):
        build_mcmc_group_dependency_data(
                scaling_model,
                "this_group_does_not_exist")
