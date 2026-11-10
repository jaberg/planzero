import datetime
import re

import pytest

from .mermaid import render_mcmc_group_mermaid
from .scaling_model import Scaling_ScenarioModel

MCMC_GROUP = "enteric_and_other_product_mfg"


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
def mermaid_text(scaling_model):
    return render_mcmc_group_mermaid(scaling_model, MCMC_GROUP)


_NODE_RE = re.compile(
        r'^\s*(?P<id>[A-Za-z_][A-Za-z0-9_]*)'
        r'\["(?P<label>.*)"\]$')
_EDGE_RE = re.compile(
        r'^\s*(?P<source>[A-Za-z_][A-Za-z0-9_]*)'
        r'\s*-->\|"(?P<label>.*)"\|\s*'
        r'(?P<target>[A-Za-z_][A-Za-z0-9_]*)$')
_SUBGRAPH_RE = re.compile(
        r'^\s*subgraph\s+(?P<id>[A-Za-z_][A-Za-z0-9_]*)'
        r'\["(?P<label>.*)"\]$')


def _nodes(text):
    rval = {}
    for line in text.splitlines():
        match = _NODE_RE.match(line)
        if match:
            rval[match.group("id")] = match.group("label")
    return rval


def _edges(text):
    rval = []
    for line in text.splitlines():
        match = _EDGE_RE.match(line)
        if match:
            rval.append((
                match.group("source"),
                match.group("target"),
                match.group("label"),
                ))
    return rval


def _find_node(nodes, elem_id, method):
    matches = [
            node_id
            for node_id, label in nodes.items()
            if elem_id in label and method in label
            ]
    assert len(matches) == 1, (elem_id, method, matches, nodes)
    return matches[0]


def _assert_edge(text, source_elem, source_method,
                 target_elem, target_method, variable):
    nodes = _nodes(text)
    source = _find_node(nodes, source_elem, source_method)
    target = _find_node(nodes, target_elem, target_method)
    labels = [
            label
            for edge_source, edge_target, label in _edges(text)
            if edge_source == source and edge_target == target
            ]
    assert labels, (source_elem, source_method,
                    target_elem, target_method, source, target)
    assert any(variable in label for label in labels), (variable, labels)


def _scan_block(text):
    lines = text.splitlines()
    start = next(
            ii for ii, line in enumerate(lines)
            if _SUBGRAPH_RE.match(line)
            and "annual scan step" in line)
    depth = 0
    block = []
    for line in lines[start:]:
        if _SUBGRAPH_RE.match(line):
            depth += 1
        block.append(line)
        if line.strip() == "end":
            depth -= 1
            if depth == 0:
                break
    return "\n".join(block)


def test_nodes_are_nonempty_method_pairs(mermaid_text):
    nodes = _nodes(mermaid_text)
    labels = list(nodes.values())

    assert any(
            "Cattle_StaticNormals_ModelElement" in label
            and "model_element_postprocess" in label
            for label in labels)
    assert any(
            "Bovaer_Adoption_Limit" in label
            and "model_element_annual_step" in label
            for label in labels)
    assert any(
            "Bovaer_Farm_Subsidy" in label
            and "model_element_annual_step" in label
            for label in labels)

    # Inherited base methods declare nothing and should not be nodes.
    assert not any(
            "Cattle_StaticNormals_ModelElement" in label
            and "model_element_prepare" in label
            for label in labels)
    assert not any(
            "Bovaer_Production_Emissions" in label
            and "model_element_annual_step" in label
            for label in labels)
    assert not any(
            "Bovaer_Adoption_Limit" in label
            and "model_element_postprocess" in label
            for label in labels)
    assert not any("ModelBuiltIns" in label for label in labels)


def test_dependencies_are_labelled_by_variable(mermaid_text):
    _assert_edge(
            mermaid_text,
            "Scale_Bovaer_2030", "model_element_prepare",
            "Bovaer_Farm_Subsidy", "model_element_annual_step",
            "annual: bovaer_budget")

    _assert_edge(
            mermaid_text,
            "Bovaer_Farm_Subsidy", "model_element_annual_step",
            "Cattle_StaticNormals_ModelElement", "model_element_postprocess",
            "annual: annual_bovaer_fraction")

    _assert_edge(
            mermaid_text,
            "Cattle_StaticNormals_ModelElement", "model_element_postprocess",
            "Bovaer_Production_Emissions", "model_element_postprocess",
            "annual: ktCO2e_abated_by_bovaer_ca")

    _assert_edge(
            mermaid_text,
            "Bovaer_Farm_Subsidy", "model_element_annual_step",
            "Bovaer_Adoption_Limit", "model_element_annual_step",
            "carry: bovaer_fraction")


def test_annual_step_nodes_are_inside_scan_box(mermaid_text):
    block = _scan_block(mermaid_text)
    step_nodes = _nodes(block)

    assert step_nodes
    assert all(
            "model_element_annual_step" in label
            for label in step_nodes.values())

    adoption_label = next(
            label for label in step_nodes.values()
            if "Bovaer_Adoption_Limit" in label)
    assert "reads:" in adoption_label
    assert "annual: max_increase_fraction" in adoption_label
    assert "this carry: bovaer_fraction" in adoption_label
    assert "writes:" in adoption_label
    assert "carry: max_possible_fraction" in adoption_label

    farm_label = next(
            label for label in step_nodes.values()
            if "Bovaer_Farm_Subsidy" in label)
    assert "reads:" in farm_label
    assert "annual: bovaer_budget" in farm_label
    assert "writes:" in farm_label
    assert "annual: annual_bovaer_fraction" in farm_label
    assert "carry: bovaer_fraction" in farm_label


def test_mermaid_structure_is_balanced(mermaid_text):
    lines = mermaid_text.splitlines()
    assert lines[0] == "flowchart LR"
    assert MCMC_GROUP in mermaid_text
    assert sum(
            1 for line in lines if _SUBGRAPH_RE.match(line)
            ) == sum(1 for line in lines if line.strip() == "end")


def test_unknown_mcmc_group_raises(scaling_model):
    with pytest.raises(ValueError, match="unknown mcmc_group"):
        render_mcmc_group_mermaid(scaling_model, "not_a_group")


def test_invalid_direction_raises(scaling_model):
    with pytest.raises(ValueError, match="invalid mermaid direction"):
        render_mcmc_group_mermaid(
                scaling_model, MCMC_GROUP, direction="sideways")
