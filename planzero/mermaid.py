"""Render ScenarioModel mcmc groups as Mermaid flowcharts.

The graph nodes are ``(model_element, method)`` pairs.  A method node is
included only when its decorators define or access at least one VarKey, so
the inherited no-op method implementations do not appear in the graph.
Edges run from the method that defines a VarKey to the method that reads it
and are labelled with the variable name.

``model_element_annual_step`` nodes are placed inside a single subgraph box
to show that they form the nested annual scan sub-computation.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from .scenario_model import ScenarioModel, VarKey
from .scenario_model.base import (
    ElementAnnualKey,
    ElementCarryKey,
    ElementGeneralKey,
    FinalCarry,
    InitialCarry,
    NextCarry,
    Subphase,
    ThisCarry,
)

__all__ = ["render_mcmc_group_mermaid"]


_METHOD_BY_SUBPHASE: dict[Subphase, str] = {
    Subphase.Prep: "model_element_prepare",
    Subphase.Step: "model_element_annual_step",
    Subphase.Proc: "model_element_postprocess",
}

_METHOD_ORDER = (
    "model_element_prepare",
    "model_element_annual_step",
    "model_element_postprocess",
)

# These attribute names mirror the decorator metadata attributes in
# scenario_model.scenario_model.  They are checked to distinguish a method
# that declares variables from an inherited no-op method implementation.
_DECLARATION_ATTRS = (
    "model_element_define",
    "model_element_define_annual",
    "model_element_define_carry",
    "model_element_access_val",
    "model_element_access_annual",
    "model_element_access_this_carry",
    "model_element_access_next_carry",
)


@dataclass(frozen=True)
class _MethodNode:
    elem_id: str
    method: str

    @property
    def label(self) -> str:
        return f"{self.elem_id} {self.short_method}"

    @property
    def short_method(self) -> str:
        return {
                "model_element_prepare": "[Prep]",
                "model_element_annual_step": "[Annual Step]",
                "model_element_postprocess": "[Post]",
        }[self.method]
        


def _escape_label(text: str) -> str:
    """Escape text for use inside a quoted Mermaid node or edge label."""
    return (
            text
            .replace("&", "&amp;")
            .replace('"', "&quot;")
            .replace("|", "&#124;")
            )


def _var_label(var_key: VarKey) -> str:
    """Human-readable variable label used on graph edges and step nodes."""
    if isinstance(var_key, ElementAnnualKey):
        return f"annual: {var_key.name}"
    if isinstance(var_key, ElementCarryKey):
        return f"carry: {var_key.name}"
    if isinstance(var_key, ElementGeneralKey):
        return f"general: {var_key.name}"
    if isinstance(var_key, InitialCarry):
        return f"initial carry: {var_key.carry_key.name}"
    if isinstance(var_key, ThisCarry):
        return f"this carry: {var_key.carry_key.name}"
    if isinstance(var_key, NextCarry):
        return f"next carry: {var_key.carry_key.name}"
    if isinstance(var_key, FinalCarry):
        return f"final carry: {var_key.carry_key.name}"
    return str(var_key)


def _method_declares_variables(element, method: str) -> bool:
    """True if method carries any define/access decorator metadata."""
    method_obj = getattr(element, method)
    return any(getattr(method_obj, attr, None) for attr in _DECLARATION_ATTRS)


def _group_element_ids(model: ScenarioModel, mcmc_group: str) -> list[str]:
    element_ids = [
            elem_id
            for elem_id in model.model_elements
            if model.mcmc_group_by_elemid.get(elem_id) == mcmc_group
            ]
    if not element_ids:
        available = sorted({
            group
            for group in model.mcmc_group_by_elemid.values()
            if group is not None
            })
        raise ValueError(
                f"unknown mcmc_group {mcmc_group!r};"
                f" available groups: {available}")
    return element_ids


def _collect_definitions(
        model: ScenarioModel,
        element_ids: Iterable[str],
        ) -> tuple[dict[_MethodNode, set[str]], dict[VarKey, _MethodNode]]:
    """Return per-node definitions and the definer node for each VarKey."""
    element_id_set = set(element_ids)
    definitions: dict[_MethodNode, set[str]] = defaultdict(set)
    definer_by_key: dict[VarKey, _MethodNode] = {}

    metadata_dicts = (
            model.mv.general_nd,
            model.mv.annual_nd,
            model.mv.carry_nd,
            )
    for metadata_d in metadata_dicts:
        for var_key, variable_metadata in metadata_d.items():
            elem_id = variable_metadata.defining_element_id
            if elem_id not in element_id_set:
                continue
            method = _METHOD_BY_SUBPHASE[
                    variable_metadata.defining_subphase]
            node = _MethodNode(elem_id, method)
            definitions[node].add(_var_label(var_key))
            definer_by_key[var_key] = node

    return definitions, definer_by_key


def _collect_accesses(
        model: ScenarioModel,
        element_ids: Iterable[str],
        ) -> tuple[dict[_MethodNode, set[str]], dict[_MethodNode, list]]:
    """Return per-node reads and the resolved accesses that produced them."""
    element_id_set = set(element_ids)
    reads: dict[_MethodNode, set[str]] = defaultdict(set)
    accesses_by_node: dict[_MethodNode, list] = defaultdict(list)

    for access in model.resolve_accesses():
        if access.elem_id not in element_id_set:
            continue
        node = _MethodNode(
                access.elem_id,
                _METHOD_BY_SUBPHASE[access.subphase])
        reads[node].add(_var_label(access.var_key))
        accesses_by_node[node].append(access)

    return reads, accesses_by_node


def _find_definer(
        dependency_key: VarKey,
        definer_by_key: dict[VarKey, _MethodNode],
        ) -> _MethodNode | None:
    if dependency_key in definer_by_key:
        return definer_by_key[dependency_key]
    # A this/next carry read depends on the per-step carry value, which is
    # registered under the underlying ElementCarryKey rather than under the
    # ThisCarry/NextCarry wrapper.
    if isinstance(dependency_key, (ThisCarry, NextCarry)):
        return definer_by_key.get(dependency_key.carry_key)
    return None


def _collect_edges(
        model: ScenarioModel,
        accesses_by_node: dict[_MethodNode, list],
        definer_by_key: dict[VarKey, _MethodNode],
        ) -> dict[tuple[_MethodNode, _MethodNode], set[str]]:
    edges: dict[tuple[_MethodNode, _MethodNode], set[str]] = defaultdict(set)

    for accessor_node, accesses in accesses_by_node.items():
        for access in accesses:
            dependency_key = model._dependency_key(access)
            if dependency_key is None:
                continue
            definer_node = _find_definer(dependency_key, definer_by_key)
            if definer_node is None or definer_node == accessor_node:
                continue
            edges[(definer_node, accessor_node)].add(
                    _var_label(access.var_key))

    return edges


def _node_label(
        node: _MethodNode,
        definitions: dict[_MethodNode, set[str]],
        reads: dict[_MethodNode, set[str]],
        ) -> str:
    label = node.label
    if node.method == "model_element_annual_step":
        if reads.get(node):
            label += "<br/><br/>reads:<br/>" + "<br/>".join(
                    sorted(reads[node]))
        if definitions.get(node):
            label += "<br/><br/>writes:<br/>" + "<br/>".join(
                    sorted(definitions[node]))
    return label


def render_mcmc_group_mermaid(
        model: ScenarioModel,
        mcmc_group: str,
        *,
        direction: Literal["LR", "RL", "TB", "BT"] = "LR",
        ) -> str:
    """Render one mcmc group of a ScenarioModel as a Mermaid flowchart.

    Parameters
    ----------
    model:
        The ScenarioModel whose elements and mcmc groups should be rendered.
    mcmc_group:
        The mcmc group to render.  Only model elements assigned to this group
        are included.
    direction:
        Mermaid flowchart direction.

    Returns
    -------
    str
        A Mermaid flowchart string suitable for a ``mermaid`` code block.
    """
    if direction not in ("LR", "RL", "TB", "BT"):
        raise ValueError(f"invalid mermaid direction {direction!r}")

    element_ids = _group_element_ids(model, mcmc_group)
    element_index = {
            elem_id: ii for ii, elem_id in enumerate(element_ids)}

    definitions, definer_by_key = _collect_definitions(model, element_ids)
    reads, accesses_by_node = _collect_accesses(model, element_ids)

    declared_nodes = {
            _MethodNode(elem_id, method)
            for elem_id in element_ids
            for method in _METHOD_ORDER
            if _method_declares_variables(
                model.model_elements[elem_id], method)
            }

    # Definitions are registered by subphase, which means that a carry
    # declared on prepare also registers a per-step definition on annual_step
    # and a final definition on postprocess.  Intersecting with the declared
    # nodes keeps inherited no-op methods out of the graph.
    nodes = (set(definitions) | set(reads)) & declared_nodes
    definer_by_key = {
            var_key: node
            for var_key, node in definer_by_key.items()
            if node in nodes
            }

    method_index = {
            method: ii for ii, method in enumerate(_METHOD_ORDER)}
    ordered_nodes = sorted(
            nodes,
            key=lambda node: (
                element_index[node.elem_id],
                method_index[node.method],
                ))

    node_ids = {
            node: f"n{ii}" for ii, node in enumerate(ordered_nodes)}
    node_order = {
            node: ii for ii, node in enumerate(ordered_nodes)}
    edges = _collect_edges(
            model, accesses_by_node, definer_by_key)

    lines = [
            f"flowchart {direction}",
            f'  subgraph group["mcmc_group:'
            f' {_escape_label(mcmc_group)}"]',
            "    direction TB",
            ]

    step_nodes = [
            node for node in ordered_nodes
            if node.method == "model_element_annual_step"]
    other_nodes = [
            node for node in ordered_nodes
            if node.method != "model_element_annual_step"]

    for node in other_nodes:
        label = _escape_label(_node_label(node, definitions, reads))
        lines.append(f'    {node_ids[node]}["{label}"]')

    if step_nodes:
        lines.append(
                '    subgraph scan["annual scan step'
                ' (nested sub-computation)"]')
        lines.append("      direction TB")
        for node in step_nodes:
            label = _escape_label(_node_label(node, definitions, reads))
            lines.append(f'        {node_ids[node]}["{label}"]')
        lines.append("    end")

    def edge_sort_key(item):
        (source, target), _ = item
        return (node_order[source], node_order[target])

    for (source, target), labels in sorted(
            edges.items(), key=edge_sort_key):
        label = _escape_label(", ".join(sorted(labels)))
        lines.append(
                f'    {node_ids[source]} -->|"{label}"|'
                f' {node_ids[target]}')

    lines.append("  end")
    return "\n".join(lines)
