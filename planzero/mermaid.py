"""Render ScenarioModel mcmc groups as Mermaid flowcharts and HTML tables.

The Mermaid renderer has two kinds of nodes:

* ``(model_element, method)`` pairs, included only when their decorators
  define or access at least one VarKey, and
* variable nodes, one per ``@define*`` decorator.

Edges run from a defining method node to its variable node, then from that
variable node to each method node that accesses the variable.  Access edges
are annotated with the property or string name used in the access decorator.
Loop-carried ``access_this_carry`` reads during the annual step are rendered
as dotted arrows to distinguish them from the normal forward flow.

``model_element_annual_step`` method nodes are placed inside a single subgraph
box to show that they form the nested annual scan sub-computation.

The HTML renderer presents the same define/access information as a table with
one row per model element and one column per method.  Hovering a property chip
highlights the chips it depends on.
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from typing import Iterable, Literal

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

__all__ = [
        "render_mcmc_group_html_table",
        "render_mcmc_group_mermaid",
        ]


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

_KIND_ORDER = {
    "general": 0,
    "annual": 1,
    "carry": 2,
}

# These attribute names mirror the decorator metadata attributes in
# scenario_model.scenario_model.  They are checked to distinguish a method
# that declares variables from an inherited no-op method implementation.
_DEFINE_ATTRS = (
    ("model_element_define", "general"),
    ("model_element_define_annual", "annual"),
    ("model_element_define_carry", "carry"),
)

_ACCESS_ATTRS = (
    ("model_element_access_val", "general", None),
    ("model_element_access_annual", "annual", None),
    ("model_element_access_this_carry", "carry", ThisCarry),
    ("model_element_access_next_carry", "carry", NextCarry),
)

_DECLARATION_ATTRS = tuple(
    attr for attr, _ in _DEFINE_ATTRS
) + tuple(
    attr for attr, _, _ in _ACCESS_ATTRS
)

_VARIABLE_KEY_CLASSES = {
    "general": ElementGeneralKey,
    "annual": ElementAnnualKey,
    "carry": ElementCarryKey,
}


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


@dataclass(frozen=True)
class _VariableNode:
    elem_id: str
    method: str
    kind: str
    name: str
    var_keys: tuple[VarKey, ...]
    definition_methods: tuple[str, ...]


@dataclass
class _AccessEdge:
    labels: set[str] = field(default_factory=set)
    backward: bool = False


@dataclass(frozen=True)
class _PropertyOccurrence:
    elem_id: str
    method: str
    direction: Literal["define", "access"]
    kind: str
    name: str
    lookup_keys: tuple[VarKey, ...]
    attr: str | None = None


def _escape_label(text: str) -> str:
    """Escape text for use inside a quoted Mermaid node or edge label."""
    return (
            text
            .replace("&", "&amp;")
            .replace('"', "&quot;")
            .replace("|", "&#124;")
            )


def _decorator_name(decorator_key: str | VarKey | property) -> str:
    """Return the property or string name used in a decorator."""
    if isinstance(decorator_key, property):
        if decorator_key.fget is None:
            return str(decorator_key)
        return decorator_key.fget.__name__
    if isinstance(decorator_key, str):
        return decorator_key
    return getattr(decorator_key, "name", str(decorator_key))


def _resolve_decorator_keys(
        element,
        decorator_key: str | VarKey | property,
        kind: str,
        wrapper=None,
        ) -> list[VarKey]:
    """Resolve a decorator key to the VarKeys that add_element registered.

    This mirrors the string/property/dict handling in
    ``ScenarioModel.add_element`` without changing any scenario_model data
    structures.
    """
    if isinstance(decorator_key, str):
        key_class = _VARIABLE_KEY_CLASSES[kind]
        decorator_key = key_class(
                elem_id=element.identifier,
                name=decorator_key)
    elif isinstance(decorator_key, property):
        decorator_key = decorator_key.__get__(element, type(element))

    if isinstance(decorator_key, dict):
        values = list(decorator_key.values())
    else:
        values = [decorator_key]

    rval = []
    for value in values:
        if not isinstance(value, VarKey):
            continue
        if wrapper is not None:
            if not isinstance(value, ElementCarryKey):
                continue
            value = wrapper(carry_key=value)
        rval.append(value)
    return rval


def _carry_family(carry_key: ElementCarryKey) -> tuple[VarKey, ...]:
    """All VarKeys that represent one carry variable's lifecycle."""
    return (
            carry_key,
            InitialCarry(carry_key=carry_key),
            ThisCarry(carry_key=carry_key),
            NextCarry(carry_key=carry_key),
            FinalCarry(carry_key=carry_key),
            )


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


def _collect_variable_nodes(
        model: ScenarioModel,
        element_ids: Iterable[str],
        ) -> tuple[list[_VariableNode], dict[VarKey, _VariableNode]]:
    """Create one variable node per @define* decorator.

    A decorator whose property resolves to a dict of VarKeys still creates a
    single variable node, named after the property, with all of those keys
    mapped to it.
    """
    variable_nodes_by_id: dict[tuple, _VariableNode] = {}
    variable_node_by_key: dict[VarKey, _VariableNode] = {}

    for elem_id in element_ids:
        element = model.model_elements[elem_id]
        for method in _METHOD_ORDER:
            method_obj = getattr(element, method)
            for attr, kind in _DEFINE_ATTRS:
                decorator_d = getattr(method_obj, attr, None)
                if not decorator_d:
                    continue
                for decorator_key in decorator_d:
                    name = _decorator_name(decorator_key)
                    resolved_keys = _resolve_decorator_keys(
                            element, decorator_key, kind)
                    if not resolved_keys:
                        continue

                    definition_methods = {method}
                    if kind == "carry":
                        for var_key in resolved_keys:
                            if not isinstance(var_key, ElementCarryKey):
                                continue
                            metadata = model.mv.carry_nd.get(var_key)
                            if metadata is None:
                                continue
                            if metadata.defining_element_id != elem_id:
                                continue
                            definition_methods.add(
                                    _METHOD_BY_SUBPHASE[
                                        metadata.defining_subphase])
                    definition_methods = tuple(
                            method_name
                            for method_name in _METHOD_ORDER
                            if method_name in definition_methods)

                    variable_node = _VariableNode(
                            elem_id=elem_id,
                            method=method,
                            kind=kind,
                            name=name,
                            var_keys=tuple(resolved_keys),
                            definition_methods=definition_methods)
                    variable_nodes_by_id.setdefault(
                            (
                                variable_node.elem_id,
                                variable_node.method,
                                variable_node.kind,
                                variable_node.name,
                                variable_node.var_keys,
                                variable_node.definition_methods,
                                ),
                            variable_node)

                    if kind == "carry":
                        mapped_keys = []
                        for var_key in resolved_keys:
                            if isinstance(var_key, ElementCarryKey):
                                mapped_keys.extend(_carry_family(var_key))
                            else:
                                mapped_keys.append(var_key)
                    else:
                        mapped_keys = resolved_keys

                    for var_key in mapped_keys:
                        variable_node_by_key[var_key] = variable_node

    return list(variable_nodes_by_id.values()), variable_node_by_key


def _collect_definition_edges(
        variable_nodes: Iterable[_VariableNode],
        ) -> set[tuple[_MethodNode, _VariableNode]]:
    return {
            (_MethodNode(node.elem_id, method), node)
            for node in variable_nodes
            for method in node.definition_methods
            }


def _collect_access_edges(
        model: ScenarioModel,
        element_ids: Iterable[str],
        variable_node_by_key: dict[VarKey, _VariableNode],
        ) -> dict[tuple[_VariableNode, _MethodNode], _AccessEdge]:
    """Build variable -> accessing-method edges and their decorator labels."""
    edges: dict[tuple[_VariableNode, _MethodNode], _AccessEdge] = {}

    for elem_id in element_ids:
        element = model.model_elements[elem_id]
        for method in _METHOD_ORDER:
            method_obj = getattr(element, method)
            method_node = _MethodNode(elem_id, method)
            for attr, kind, wrapper in _ACCESS_ATTRS:
                decorator_d = getattr(method_obj, attr, None)
                if not decorator_d:
                    continue
                is_backward = (
                        method == "model_element_annual_step"
                        and attr == "model_element_access_this_carry")
                for key_metadata in decorator_d.values():
                    for decorator_key in key_metadata:
                        name = _decorator_name(decorator_key)
                        resolved_keys = _resolve_decorator_keys(
                                element, decorator_key, kind, wrapper)
                        for var_key in resolved_keys:
                            variable_node = variable_node_by_key.get(var_key)
                            if variable_node is None:
                                continue
                            edge = edges.setdefault(
                                    (variable_node, method_node),
                                    _AccessEdge())
                            edge.labels.add(name)
                            edge.backward = edge.backward or is_backward

    return edges


def _variable_node_syntax(
        node_id: str,
        variable_node: _VariableNode,
        ) -> str:
    label = _escape_label(variable_node.name)
    if variable_node.kind == "general":
        return f'{node_id}("{label}")'
    if variable_node.kind == "carry":
        return f'{node_id}(["{label}"])'
    if variable_node.kind == "annual":
        return node_id + '{{"' + label + '"}}'
    raise NotImplementedError(variable_node.kind)


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

    variable_nodes, variable_node_by_key = _collect_variable_nodes(
            model, element_ids)
    definition_edges = _collect_definition_edges(variable_nodes)
    access_edges = _collect_access_edges(
            model, element_ids, variable_node_by_key)

    method_nodes = {
            _MethodNode(elem_id, method)
            for elem_id in element_ids
            for method in _METHOD_ORDER
            if _method_declares_variables(
                model.model_elements[elem_id], method)
            }
    # A carry decorator is declared on prepare, but its per-step next_carry
    # definition is registered in the annual_step subphase.  Include that
    # method node even when it has no decorators of its own.
    method_nodes |= {
            _MethodNode(node.elem_id, method)
            for node in variable_nodes
            for method in node.definition_methods
            }

    method_index = {
            method: ii for ii, method in enumerate(_METHOD_ORDER)}
    ordered_method_nodes = sorted(
            method_nodes,
            key=lambda node: (
                element_index[node.elem_id],
                method_index[node.method],
                ))
    ordered_variable_nodes = sorted(
            variable_nodes,
            key=lambda node: (
                element_index[node.elem_id],
                method_index[node.method],
                _KIND_ORDER[node.kind],
                node.name,
                ))

    method_node_ids = {
            node: f"m{ii}"
            for ii, node in enumerate(ordered_method_nodes)}
    variable_node_ids = {
            node: f"v{ii}"
            for ii, node in enumerate(ordered_variable_nodes)}

    method_node_order = {
            node: ii for ii, node in enumerate(ordered_method_nodes)}
    variable_node_order = {
            node: ii for ii, node in enumerate(ordered_variable_nodes)}

    lines = [
            f"flowchart {direction}",
            f'  subgraph group["mcmc_group:'
            f' {_escape_label(mcmc_group)}"]',
            "    direction TB",
            ]

    step_method_nodes = [
            node for node in ordered_method_nodes
            if node.method == "model_element_annual_step"]
    other_method_nodes = [
            node for node in ordered_method_nodes
            if node.method != "model_element_annual_step"]

    for node in other_method_nodes:
        label = _escape_label(node.label)
        lines.append(f'    {method_node_ids[node]}["{label}"]')

    for node in ordered_variable_nodes:
        lines.append(
                "    " + _variable_node_syntax(
                    variable_node_ids[node], node))

    if step_method_nodes:
        lines.append(
                '    subgraph scan["annual scan step'
                ' (nested sub-computation)"]')
        lines.append("      direction TB")
        for node in step_method_nodes:
            label = _escape_label(node.label)
            lines.append(f'        {method_node_ids[node]}["{label}"]')
        lines.append("    end")

    for source, target in sorted(
            definition_edges,
            key=lambda edge: (
                method_node_order[edge[0]],
                variable_node_order[edge[1]],
                )):
        lines.append(
                f'    {method_node_ids[source]} -->'
                f' {variable_node_ids[target]}')

    for (source, target), edge in sorted(
            access_edges.items(),
            key=lambda item: (
                variable_node_order[item[0][0]],
                method_node_order[item[0][1]],
                )):
        label = _escape_label(", ".join(sorted(edge.labels)))
        arrow = "-.->" if edge.backward else "-->"
        lines.append(
                f'    {variable_node_ids[source]} {arrow}|"{label}"|'
                f' {method_node_ids[target]}')

    lines.append("  end")
    return "\n".join(lines)


_HTML_TABLE_STYLE = """
.mcmc-dependency-table-container {
  margin: 1rem 0;
}
.mcmc-dependency-table {
  border-collapse: collapse;
  width: 100%;
  font-family: system-ui, sans-serif;
  font-size: 0.9rem;
}
.mcmc-dependency-table caption {
  caption-side: top;
  text-align: left;
  font-weight: 600;
  padding-bottom: 0.4rem;
}
.mcmc-dependency-table th,
.mcmc-dependency-table td {
  border: 1px solid #d0d7de;
  padding: 0.45rem 0.55rem;
  text-align: left;
  vertical-align: top;
}
.mcmc-dependency-table th {
  background: #f6f8fa;
  font-weight: 600;
}
.mcmc-dependency-table td.mcmc-element {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  white-space: normal;
  overflow-wrap: anywhere;
}
.mcmc-cell-group {
  margin: 0 0 0.25rem 0;
}
.mcmc-cell-group:last-child {
  margin-bottom: 0;
}
.mcmc-cell-label {
  display: block;
  margin-bottom: 0.15rem;
  color: #57606a;
  font-size: 0.72rem;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}
.mcmc-chip {
  display: inline-block;
  margin: 0.08rem 0.15rem 0.08rem 0;
  padding: 0.08rem 0.45rem;
  border: 1px solid transparent;
  border-radius: 999px;
  line-height: 1.35;
  cursor: default;
  transition: background-color 120ms ease, border-color 120ms ease,
              box-shadow 120ms ease;
}
.mcmc-chip.mcmc-define {
  background: #ddf4e4;
  border-color: #7ccf9a;
}
.mcmc-chip.mcmc-access {
  background: #e3eefc;
  border-color: #8ab6e8;
}
.mcmc-chip.mcmc-highlight {
  background: #ffe08a;
  border-color: #e09b00;
  box-shadow: 0 0 0 2px #e09b00;
}
.mcmc-chip:focus-visible {
  outline: 2px solid #0969da;
  outline-offset: 1px;
}
.mcmc-empty {
  color: #8c959f;
}
"""


_HTML_TABLE_SCRIPT = r"""
(function () {
  if (window.mcmcHighlightRelated) {
    return;
  }
  window.mcmcHighlightRelated = function (element, highlighted) {
    var table = element.closest('table.mcmc-dependency-table');
    if (!table) {
      return;
    }
    var related = (element.getAttribute('data-related') || '')
      .split(/\s+/)
      .filter(Boolean);
    related.forEach(function (chipId) {
      var target = table.querySelector(
        '[data-chip-id="' + chipId + '"]');
      if (target) {
        target.classList.toggle('mcmc-highlight', highlighted);
      }
    });
  };
})();
"""


def _collect_property_occurrences(
        model: ScenarioModel,
        element_ids: Iterable[str],
        ) -> list[_PropertyOccurrence]:
    """Collect define/access decorator occurrences for the HTML table."""
    occurrences = []
    for elem_id in element_ids:
        element = model.model_elements[elem_id]
        for method in _METHOD_ORDER:
            method_obj = getattr(element, method)
            for attr, kind in _DEFINE_ATTRS:
                decorator_d = getattr(method_obj, attr, None)
                if not decorator_d:
                    continue
                for decorator_key in decorator_d:
                    resolved_keys = _resolve_decorator_keys(
                            element, decorator_key, kind)
                    if not resolved_keys:
                        continue
                    lookup_keys = []
                    for var_key in resolved_keys:
                        if (kind == "carry"
                                and isinstance(var_key, ElementCarryKey)):
                            lookup_keys.extend(_carry_family(var_key))
                        else:
                            lookup_keys.append(var_key)
                    occurrences.append(_PropertyOccurrence(
                            elem_id=elem_id,
                            method=method,
                            direction="define",
                            kind=kind,
                            name=_decorator_name(decorator_key),
                            lookup_keys=tuple(lookup_keys),
                            attr=attr,
                            ))

            for attr, kind, wrapper in _ACCESS_ATTRS:
                decorator_d = getattr(method_obj, attr, None)
                if not decorator_d:
                    continue
                for key_metadata in decorator_d.values():
                    for decorator_key in key_metadata:
                        resolved_keys = _resolve_decorator_keys(
                                element, decorator_key, kind, wrapper)
                        if not resolved_keys:
                            continue
                        occurrences.append(_PropertyOccurrence(
                                elem_id=elem_id,
                                method=method,
                                direction="access",
                                kind=kind,
                                name=_decorator_name(decorator_key),
                                lookup_keys=tuple(resolved_keys),
                                attr=attr,
                                ))
    return occurrences


def _property_relations(
        occurrences: Iterable[_PropertyOccurrence],
        ) -> dict[int, list[int]]:
    """Map each occurrence to related occurrences of the opposite direction."""
    occurrences = list(occurrences)
    define_by_key: dict[VarKey, set[int]] = {}
    access_by_key: dict[VarKey, set[int]] = {}

    for ii, occurrence in enumerate(occurrences):
        by_key = (
                define_by_key
                if occurrence.direction == "define"
                else access_by_key)
        for var_key in occurrence.lookup_keys:
            by_key.setdefault(var_key, set()).add(ii)

    rval = {}
    for ii, occurrence in enumerate(occurrences):
        other = (
                access_by_key
                if occurrence.direction == "define"
                else define_by_key)
        related = set()
        for var_key in occurrence.lookup_keys:
            related.update(other.get(var_key, ()))
        rval[ii] = sorted(related)
    return rval


def _render_property_chip(
        chip_id: str,
        occurrence: _PropertyOccurrence,
        related_chip_ids: Iterable[str],
        ) -> str:
    classes = f"mcmc-chip mcmc-{occurrence.direction}"
    related = " ".join(related_chip_ids)
    title = f"{occurrence.direction} {occurrence.kind}: {occurrence.name}"
    return (
            f'<span class="{classes}"'
            f' data-chip-id="{chip_id}"'
            f' data-related="{related}"'
            f' title="{html.escape(title, quote=True)}"'
            ' tabindex="0"'
            ' onmouseenter="mcmcHighlightRelated(this, true)"'
            ' onmouseleave="mcmcHighlightRelated(this, false)"'
            ' onfocus="mcmcHighlightRelated(this, true)"'
            ' onblur="mcmcHighlightRelated(this, false)">'
            f'{html.escape(occurrence.name)}'
            '</span>'
            )


def _render_property_cell(
        occurrence_indices: Iterable[int],
        occurrences: list[_PropertyOccurrence],
        relations: dict[int, list[int]],
        ) -> str:
    define_chips = []
    access_chips = []
    for ii in occurrence_indices:
        occurrence = occurrences[ii]
        chip = _render_property_chip(
                f"c{ii}",
                occurrence,
                [f"c{jj}" for jj in relations[ii]],
                )
        if occurrence.direction == "define":
            define_chips.append(chip)
        else:
            access_chips.append(chip)

    if not define_chips and not access_chips:
        return '<span class="mcmc-empty">&mdash;</span>'

    parts = []
    if define_chips:
        parts.append(
                '<div class="mcmc-cell-group">'
                '<span class="mcmc-cell-label">defines</span>'
                + "".join(define_chips)
                + '</div>')
    if access_chips:
        parts.append(
                '<div class="mcmc-cell-group">'
                '<span class="mcmc-cell-label">accesses</span>'
                + "".join(access_chips)
                + '</div>')
    return "".join(parts)


def render_mcmc_group_html_table(
        model: ScenarioModel,
        mcmc_group: str,
        ) -> str:
    """Render one mcmc group as a self-contained HTML dependency table.

    The table has one row per model element and one column per method.  Define
    and access decorator names are shown as chips.  Hovering or focusing a
    define chip highlights the access chips that read its variables; hovering
    or focusing an access chip highlights the define chips that produce them.
    """
    element_ids = _group_element_ids(model, mcmc_group)
    occurrences = _collect_property_occurrences(model, element_ids)
    relations = _property_relations(occurrences)

    occurrences_by_cell: dict[tuple[str, str], list[int]] = {}
    for ii, occurrence in enumerate(occurrences):
        occurrences_by_cell.setdefault(
                (occurrence.elem_id, occurrence.method), []).append(ii)

    rows = []
    for elem_id in element_ids:
        display_id = elem_id.replace("_", " ")
        escaped_id = html.escape(elem_id, quote=True)
        cells = [
                '<td class="mcmc-element"'
                f' data-element-id="{escaped_id}"'
                f' title="{escaped_id}">'
                f'{html.escape(display_id)}</td>',
                ]
        for method in _METHOD_ORDER:
            occurrence_indices = occurrences_by_cell.get(
                    (elem_id, method), ())
            cells.append(
                    '<td>'
                    + _render_property_cell(
                        occurrence_indices, occurrences, relations)
                    + '</td>')
        rows.append('<tr>' + "".join(cells) + '</tr>')

    header = (
            '<thead><tr>'
            '<th scope="col">Model Element</th>'
            '<th scope="col">Preparation</th>'
            '<th scope="col">Annual Step</th>'
            '<th scope="col">Post-Processing</th>'
            '</tr></thead>'
            )
    table = (
            '<table class="mcmc-dependency-table">'
            f'<caption>mcmc_group: {html.escape(mcmc_group)}</caption>'
            f'{header}<tbody>{"".join(rows)}</tbody></table>'
            )
    return (
            '<div class="mcmc-dependency-table-container">'
            f'<style>{_HTML_TABLE_STYLE}</style>'
            f'{table}'
            f'<script>{_HTML_TABLE_SCRIPT}</script>'
            '</div>'
            )
