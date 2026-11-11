import datetime
import re
from html.parser import HTMLParser

import pytest

from .mermaid import (
    render_mcmc_group_html_table,
    render_mcmc_group_mermaid,
)
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


_METHOD_NODE_RE = re.compile(
        r'^\s*(?P<id>m\d+)\["(?P<label>.*)"\]$')
_ROUND_NODE_RE = re.compile(
        r'^\s*(?P<id>v\d+)\("(?P<label>.*)"\)$')
_STADIUM_NODE_RE = re.compile(
        r'^\s*(?P<id>v\d+)\(\["(?P<label>.*)"\]\)$')
_HEXAGON_NODE_RE = re.compile(
        r'^\s*(?P<id>v\d+)\{\{"(?P<label>.*)"\}\}$')
_DEFINITION_EDGE_RE = re.compile(
        r'^\s*(?P<source>m\d+)\s*-->\s*(?P<target>v\d+)$')
_ACCESS_EDGE_RE = re.compile(
        r'^\s*(?P<source>v\d+)\s*(?P<arrow>-->|-\.->)'
        r'\|"(?P<label>.*)"\|\s*(?P<target>m\d+)$')
_SUBGRAPH_RE = re.compile(
        r'^\s*subgraph\s+(?P<id>[A-Za-z_][A-Za-z0-9_]*)'
        r'\["(?P<label>.*)"\]$')

_METHOD_SHORT = {
        "model_element_prepare": "[Prep]",
        "model_element_annual_step": "[Annual Step]",
        "model_element_postprocess": "[Post]",
        }


def _method_nodes(text):
    rval = {}
    for line in text.splitlines():
        match = _METHOD_NODE_RE.match(line)
        if match:
            rval[match.group("id")] = match.group("label")
    return rval


def _variable_nodes(text):
    """Return {node_id: (label, shape)} for every variable node."""
    rval = {}
    for line in text.splitlines():
        for shape, pattern in (
                ("round", _ROUND_NODE_RE),
                ("stadium", _STADIUM_NODE_RE),
                ("hexagon", _HEXAGON_NODE_RE),
                ):
            match = pattern.match(line)
            if match:
                rval[match.group("id")] = (match.group("label"), shape)
                break
    return rval


def _edges(text):
    """Return (source, target, label, backward) tuples for all edges."""
    rval = []
    for line in text.splitlines():
        match = _DEFINITION_EDGE_RE.match(line)
        if match:
            rval.append((
                match.group("source"),
                match.group("target"),
                None,
                False,
                ))
            continue
        match = _ACCESS_EDGE_RE.match(line)
        if match:
            rval.append((
                match.group("source"),
                match.group("target"),
                match.group("label"),
                match.group("arrow") == "-.->",
                ))
    return rval


def _find_method(nodes, elem_id, method):
    short_method = _METHOD_SHORT[method]
    matches = [
            node_id
            for node_id, label in nodes.items()
            if elem_id in label and short_method in label
            ]
    assert len(matches) == 1, (elem_id, method, matches, nodes)
    return matches[0]


def _find_variable(variables, name, shape):
    matches = [
            node_id
            for node_id, (label, node_shape) in variables.items()
            if label == name and node_shape == shape
            ]
    assert len(matches) == 1, (name, shape, matches, variables)
    return matches[0]


def _definition_variable(text, source_elem, source_method,
                         variable_name, variable_shape):
    methods = _method_nodes(text)
    variables = _variable_nodes(text)
    edges = _edges(text)

    source = _find_method(methods, source_elem, source_method)

    # Several elements can define a variable with the same decorator name,
    # so select the one produced by the expected defining method node.
    candidates = [
            node_id
            for node_id, (label, shape) in variables.items()
            if label == variable_name and shape == variable_shape
            ]
    defined_by_source = [
            node_id
            for node_id in candidates
            if (source, node_id, None, False) in edges
            ]
    assert len(defined_by_source) == 1, (
            source_elem, source_method, variable_name, defined_by_source)
    return defined_by_source[0]


def _assert_definition_edge(text, source_elem, source_method,
                            variable_name, variable_shape):
    _definition_variable(
            text, source_elem, source_method, variable_name, variable_shape)


def _assert_path(text, source_elem, source_method, variable_name,
                 variable_shape, target_elem, target_method, access_label,
                 *, backward=False):
    methods = _method_nodes(text)
    edges = _edges(text)

    variable = _definition_variable(
            text, source_elem, source_method, variable_name, variable_shape)
    target = _find_method(methods, target_elem, target_method)

    access_labels = [
            label
            for edge_source, edge_target, label, edge_backward in edges
            if edge_source == variable
            and edge_target == target
            and label is not None
            and edge_backward == backward
            ]
    assert access_labels, (variable_name, target_elem, target_method)
    assert any(
            access_label in label for label in access_labels), (
                    access_label, access_labels)


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


def test_method_nodes_are_nonempty_pairs(mermaid_text):
    nodes = _method_nodes(mermaid_text)
    labels = list(nodes.values())
    prepare = _METHOD_SHORT["model_element_prepare"]
    annual_step = _METHOD_SHORT["model_element_annual_step"]
    postprocess = _METHOD_SHORT["model_element_postprocess"]

    assert any(
            "Cattle_StaticNormals_ModelElement" in label
            and postprocess in label
            for label in labels)
    assert any(
            "Bovaer_Adoption_Limit" in label
            and annual_step in label
            for label in labels)
    assert any(
            "Bovaer_Farm_Subsidy" in label
            and annual_step in label
            for label in labels)

    # Inherited base methods declare nothing and should not be nodes.
    assert not any(
            "Cattle_StaticNormals_ModelElement" in label
            and prepare in label
            for label in labels)
    assert not any(
            "Bovaer_Production_Emissions" in label
            and annual_step in label
            for label in labels)
    assert not any(
            "Bovaer_Adoption_Limit" in label
            and postprocess in label
            for label in labels)
    assert not any("ModelBuiltIns" in label for label in labels)


def test_variable_nodes_use_decorator_names_and_shapes(mermaid_text):
    variables = _variable_nodes(mermaid_text)

    # General variables are rounded.
    assert _find_variable(
            variables, "bovaer_program_launch", "round")

    # Carry variables are stadium-shaped.
    assert _find_variable(
            variables, "bovine_fraction_on_bovaer", "stadium")
    assert _find_variable(
            variables, "max_possible_fraction_on_bovaer", "stadium")

    # Annual variables are hexagonal.
    assert _find_variable(variables, "bovaer_budget", "hexagon")
    assert _find_variable(
            variables, "annual_bovaer_fraction", "hexagon")
    assert _find_variable(
            variables, "ktCO2e_abated_by_bovaer_ca", "hexagon")

    # No variable node should still carry a kind prefix from the old
    # edge-labelled graph.
    assert not any(
            label.startswith(("annual:", "carry:", "general:"))
            for label, _ in variables.values())


def test_dict_valued_decorator_creates_one_variable_node(mermaid_text):
    # ktCO2e_pt_keys resolves to a dict of several VarKeys, but the decorator
    # should still produce exactly one variable node.
    methods = _method_nodes(mermaid_text)
    variables = _variable_nodes(mermaid_text)
    edges = _edges(mermaid_text)

    source = _find_method(
            methods, "Cattle_StaticNormals_ModelElement",
            "model_element_postprocess")
    defined_names = [
            variables[target][0]
            for edge_source, target, _, _ in edges
            if edge_source == source
            ]
    assert defined_names.count("ktCO2e_pt_keys") == 1
    assert defined_names.count("ktCO2e_ca_key") == 1


def test_definition_and_access_paths(mermaid_text):
    _assert_path(
            mermaid_text,
            "Scale_Bovaer_2030", "model_element_prepare",
            "bovaer_budget", "hexagon",
            "Bovaer_Farm_Subsidy", "model_element_annual_step",
            "bovaer_budget")

    _assert_path(
            mermaid_text,
            "Bovaer_Farm_Subsidy", "model_element_annual_step",
            "bovine_fraction_on_bovaer", "stadium",
            "Bovaer_Adoption_Limit", "model_element_annual_step",
            "bovine_fraction_on_bovaer",
            backward=True)

    _assert_path(
            mermaid_text,
            "Bovaer_Adoption_Limit", "model_element_annual_step",
            "max_possible_fraction_on_bovaer", "stadium",
            "Bovaer_Farm_Subsidy", "model_element_annual_step",
            "max_fraction_on_bovaer",
            backward=True)

    _assert_path(
            mermaid_text,
            "Bovaer_Farm_Subsidy", "model_element_annual_step",
            "annual_bovaer_fraction", "hexagon",
            "Cattle_StaticNormals_ModelElement", "model_element_postprocess",
            "annual_bovaer_fraction")

    _assert_path(
            mermaid_text,
            "Cattle_StaticNormals_ModelElement", "model_element_postprocess",
            "ktCO2e_abated_by_bovaer_ca", "hexagon",
            "Bovaer_Production_Emissions", "model_element_postprocess",
            "ktCO2e_abated_by_bovaer_ca")


def test_carry_variables_keep_initial_definers(mermaid_text):
    _assert_definition_edge(
            mermaid_text,
            "Bovaer_Farm_Subsidy", "model_element_prepare",
            "bovine_fraction_on_bovaer", "stadium")
    _assert_definition_edge(
            mermaid_text,
            "Bovaer_Adoption_Limit", "model_element_prepare",
            "max_possible_fraction_on_bovaer", "stadium")


def test_only_this_carry_edges_are_backward(mermaid_text):
    methods = _method_nodes(mermaid_text)
    variables = _variable_nodes(mermaid_text)
    edges = _edges(mermaid_text)

    backward_edges = [
            (source, target, label)
            for source, target, label, backward in edges
            if backward
            ]
    assert backward_edges
    assert any(
            label == "bovine_fraction_on_bovaer"
            for _, _, label in backward_edges)
    assert any(
            label == "max_fraction_on_bovaer"
            for _, _, label in backward_edges)

    annual_step = _METHOD_SHORT["model_element_annual_step"]
    for source, target, _ in backward_edges:
        assert variables[source][1] == "stadium"
        assert annual_step in methods[target]

    # A normal annual access remains a solid forward edge.
    budget = _find_variable(variables, "bovaer_budget", "hexagon")
    farm_step = _find_method(
            methods, "Bovaer_Farm_Subsidy", "model_element_annual_step")
    assert (budget, farm_step, "bovaer_budget", False) in edges


def test_access_edges_use_access_decorator_names(mermaid_text):
    # The property used in the access decorator can differ from the name of
    # the defining variable, as with the dict-valued _contributors property.
    _assert_path(
            mermaid_text,
            "Cattle_StaticNormals_ModelElement", "model_element_postprocess",
            "ktCO2e_pt_keys", "hexagon",
            "Sector_GHG_Observation_IPCC_Sector."
            "Enteric_Fermentation_GHG.CH4", "model_element_postprocess",
            "_contributors")


def test_edges_are_only_method_variable_or_variable_method(mermaid_text):
    edges = _edges(mermaid_text)

    assert edges
    assert all(
            (source.startswith("m") and target.startswith("v"))
            or (source.startswith("v") and target.startswith("m"))
            for source, target, _, _ in edges)

    # Every variable node is produced by a defining method node.
    variables = set(_variable_nodes(mermaid_text))
    defined = {
            target for source, target, _, _ in edges
            if source.startswith("m")
            }
    assert variables == defined

    # Every method-to-variable edge is an unlabelled, solid definition edge.
    assert all(
            label is None and not backward
            for source, _, label, backward in edges
            if source.startswith("m"))
    assert all(
            label is not None
            for source, _, label, _ in edges
            if source.startswith("v"))


def test_annual_step_nodes_are_inside_scan_box(mermaid_text):
    block = _scan_block(mermaid_text)
    step_nodes = _method_nodes(block)

    assert step_nodes
    assert all(
            _METHOD_SHORT["model_element_annual_step"] in label
            for label in step_nodes.values())

    adoption = _find_method(
            step_nodes, "Bovaer_Adoption_Limit",
            "model_element_annual_step")
    farm = _find_method(
            step_nodes, "Bovaer_Farm_Subsidy",
            "model_element_annual_step")
    assert adoption != farm

    # Variable nodes are explicit now, so annual-step labels no longer list
    # their reads and writes.
    assert not any(
            "reads:" in label or "writes:" in label
            for label in step_nodes.values())


def test_variable_nodes_are_outside_scan_box(mermaid_text):
    block = _scan_block(mermaid_text)
    assert not _variable_nodes(block)


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


_HTML_METHOD_COLUMNS = (
        "model_element_prepare",
        "model_element_annual_step",
        "model_element_postprocess",
        )


class _DependencyTableParser(HTMLParser):
    """Collect the headers, rows, and chips of the dependency table."""

    def __init__(self):
        super().__init__()
        self.headers: list[str] = []
        self.rows: list[list[dict]] = []
        self.chips: dict[str, dict] = {}
        self._header: list[str] | None = None
        self._row: list[dict] | None = None
        self._cell: dict | None = None
        self._chip: dict | None = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = attrs.get("class", "").split()
        if tag == "tr":
            self._row = []
        elif tag == "th":
            self._header = []
        elif tag == "td":
            self._cell = {
                    "classes": classes,
                    "element_id": attrs.get("data-element-id"),
                    "text": [],
                    "chips": [],
                    }
        elif tag == "span" and "mcmc-chip" in classes:
            self._chip = {
                    "classes": classes,
                    "id": attrs.get("data-chip-id"),
                    "related": attrs.get("data-related", "").split(),
                    "text": [],
                    }

    def handle_endtag(self, tag):
        if tag == "th" and self._header is not None:
            self.headers.append("".join(self._header).strip())
            self._header = None
        elif tag == "td" and self._cell is not None:
            assert self._row is not None
            self._cell["text"] = "".join(self._cell["text"])
            self._row.append(self._cell)
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None
        elif tag == "span" and self._chip is not None:
            chip = self._chip
            chip["text"] = "".join(chip["text"]).strip()
            self.chips[chip["id"]] = chip
            if self._cell is not None:
                self._cell["chips"].append(chip)
            self._chip = None

    def handle_data(self, data):
        if self._header is not None:
            self._header.append(data)
        elif self._chip is not None:
            self._chip["text"].append(data)
        elif self._cell is not None:
            self._cell["text"].append(data)


@pytest.fixture(scope="module")
def html_table(scaling_model):
    return render_mcmc_group_html_table(scaling_model, MCMC_GROUP)


@pytest.fixture(scope="module")
def parsed_table(html_table):
    parser = _DependencyTableParser()
    parser.feed(html_table)
    return parser


def _row_for_element(parsed_table, elem_id):
    matches = [
            row
            for row in parsed_table.rows
            if row and row[0]["element_id"] == elem_id
            ]
    assert len(matches) == 1, (elem_id, matches)
    return matches[0]


def _find_chip(parsed_table, elem_id, method, name, direction):
    row = _row_for_element(parsed_table, elem_id)
    cell = row[1 + _HTML_METHOD_COLUMNS.index(method)]
    matches = [
            chip
            for chip in cell["chips"]
            if chip["text"] == name
            and f"mcmc-{direction}" in chip["classes"]
            ]
    assert len(matches) == 1, (
            elem_id, method, name, direction, cell["chips"])
    return matches[0]


def _assert_related(parsed_table, define_chip, access_chip):
    assert access_chip["id"] in define_chip["related"], (
            define_chip, access_chip)
    assert define_chip["id"] in access_chip["related"], (
            define_chip, access_chip)


def test_html_table_has_expected_headers(parsed_table):
    assert parsed_table.headers == [
            "Model Element",
            "Preparation",
            "Annual Step",
            "Post-Processing",
            ]


def test_html_table_has_one_row_per_element(scaling_model, parsed_table):
    expected = [
            elem_id
            for elem_id in scaling_model.model_elements
            if scaling_model.mcmc_group_by_elemid.get(elem_id) == MCMC_GROUP
            ]
    actual = [
            row[0]["element_id"]
            for row in parsed_table.rows
            if row
            ]
    assert actual == expected
    assert all(len(row) == 4 for row in parsed_table.rows if row)
    # The first column displays a word-wrappable version of the identifier.
    assert all(
            row[0]["text"] == row[0]["element_id"].replace("_", " ")
            for row in parsed_table.rows
            if row)


def test_html_table_places_properties_in_method_columns(parsed_table):
    _find_chip(
            parsed_table,
            "Scale_Bovaer_2030", "model_element_prepare",
            "bovaer_budget", "define")
    _find_chip(
            parsed_table,
            "Bovaer_Farm_Subsidy", "model_element_annual_step",
            "bovaer_budget", "access")
    _find_chip(
            parsed_table,
            "Bovaer_Farm_Subsidy", "model_element_postprocess",
            "annual_headcounts", "access")
    _find_chip(
            parsed_table,
            "Cattle_StaticNormals_ModelElement",
            "model_element_postprocess",
            "annual_headcounts", "define")


def test_html_table_define_highlights_accessors(parsed_table):
    define_chip = _find_chip(
            parsed_table,
            "Scale_Bovaer_2030", "model_element_prepare",
            "bovaer_budget", "define")
    access_chip = _find_chip(
            parsed_table,
            "Bovaer_Farm_Subsidy", "model_element_annual_step",
            "bovaer_budget", "access")
    _assert_related(parsed_table, define_chip, access_chip)


def test_html_table_access_highlights_definition(parsed_table):
    define_chip = _find_chip(
            parsed_table,
            "Bovaer_Farm_Subsidy", "model_element_annual_step",
            "annual_bovaer_fraction", "define")
    access_chip = _find_chip(
            parsed_table,
            "Cattle_StaticNormals_ModelElement",
            "model_element_postprocess",
            "annual_bovaer_fraction", "access")
    _assert_related(parsed_table, define_chip, access_chip)


def test_html_table_carry_relations(parsed_table):
    define_chip = _find_chip(
            parsed_table,
            "Bovaer_Farm_Subsidy", "model_element_prepare",
            "bovine_fraction_on_bovaer", "define")
    access_chip = _find_chip(
            parsed_table,
            "Bovaer_Adoption_Limit", "model_element_annual_step",
            "bovine_fraction_on_bovaer", "access")
    _assert_related(parsed_table, define_chip, access_chip)

    define_chip = _find_chip(
            parsed_table,
            "Bovaer_Adoption_Limit", "model_element_prepare",
            "max_possible_fraction_on_bovaer", "define")
    access_chip = _find_chip(
            parsed_table,
            "Bovaer_Farm_Subsidy", "model_element_annual_step",
            "max_fraction_on_bovaer", "access")
    _assert_related(parsed_table, define_chip, access_chip)


def test_html_table_dict_valued_relations(parsed_table):
    define_chip = _find_chip(
            parsed_table,
            "Cattle_StaticNormals_ModelElement",
            "model_element_postprocess",
            "ktCO2e_pt_keys", "define")
    access_chip = _find_chip(
            parsed_table,
            "Sector_GHG_Observation_IPCC_Sector.Enteric_Fermentation_GHG.CH4",
            "model_element_postprocess",
            "_contributors", "access")
    _assert_related(parsed_table, define_chip, access_chip)


def test_html_table_define_without_accessors_has_no_relations(parsed_table):
    define_chip = _find_chip(
            parsed_table,
            "Scale_Bovaer_2030", "model_element_prepare",
            "bovaer_program_launch", "define")
    assert define_chip["related"] == []


def test_html_table_same_name_on_different_elements_does_not_cross_link(
        parsed_table):
    cattle_define = _find_chip(
            parsed_table,
            "Cattle_StaticNormals_ModelElement",
            "model_element_postprocess",
            "ktCO2e_pt_keys", "define")
    other_define = _find_chip(
            parsed_table,
            "StaticNormals_ModelElement_Other Manufacturing_CO2",
            "model_element_postprocess",
            "ktCO2e_pt_keys", "define")
    assert cattle_define["id"] != other_define["id"]

    other_access = _find_chip(
            parsed_table,
            "Sector_GHG_Observation_IPCC_Sector.SCS__Manufacturing__Other_GHG.CO2",
            "model_element_postprocess",
            "_contributors", "access")
    assert other_define["id"] in other_access["related"]
    assert cattle_define["id"] not in other_access["related"]


def test_html_table_empty_cells_are_rendered(parsed_table):
    row = _row_for_element(parsed_table, "Scale_Bovaer_2030")
    # Scale_Bovaer_2030 has no annual-step or post-processing decorators.
    assert row[2]["chips"] == []
    assert row[3]["chips"] == []
    assert "—" in row[2]["text"]


def test_html_table_is_self_contained(html_table):
    assert html_table.startswith(
            '<div class="mcmc-dependency-table-container">')
    assert "<style>" in html_table
    assert "<script>" in html_table
    assert "mcmcHighlightRelated" in html_table


def test_unknown_mcmc_group_html_table_raises(scaling_model):
    with pytest.raises(ValueError, match="unknown mcmc_group"):
        render_mcmc_group_html_table(scaling_model, "not_a_group")
