"""Build and render an interactive MCMC dependency explorer.

The explorer is a small, self-contained HTML widget backed by Cytoscape.js.
It exposes two linked views:

* an element-by-element dependency matrix for the overall structure, and
* a focused dependency graph for tracing individual variables and paths.

The JSON payload is generated from the same decorator metadata used by
``planzero.mermaid``.  No ``ScenarioModel`` data structures are modified.
"""

from __future__ import annotations

import html
import json
import re
from typing import Iterable

from .mermaid import (
    _METHOD_ORDER,
    _collect_property_occurrences,
    _group_element_ids,
)
from .scenario_model import ScenarioModel, VarKey
from .scenario_model.base import (
    FinalCarry,
    InitialCarry,
    NextCarry,
    ThisCarry,
)

__all__ = [
        "build_mcmc_group_dependency_data",
        "list_mcmc_groups",
        "render_mcmc_group_html_explorer",
        ]


_METHOD_KEYS = {
        "model_element_prepare": "prepare",
        "model_element_annual_step": "annual_step",
        "model_element_postprocess": "postprocess",
        }
_METHOD_LABELS = {
        "model_element_prepare": "Prep",
        "model_element_annual_step": "Annual Step",
        "model_element_postprocess": "Post",
        }
_METHOD_KEYS_BY_ORDER = tuple(_METHOD_KEYS[m] for m in _METHOD_ORDER)


def _canonical_var_key(var_key: VarKey) -> VarKey:
    """Collapse carry lifecycle wrappers onto their underlying carry key."""
    if isinstance(
            var_key,
            (InitialCarry, ThisCarry, NextCarry, FinalCarry)):
        return var_key.carry_key
    return var_key


def list_mcmc_groups(model: ScenarioModel) -> list[str]:
    """Return the sorted, non-empty inference-group names in a model."""
    return sorted({
        group
        for group in model.mcmc_group_by_elemid.values()
        if group is not None
        })


def _dedupe(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _plain_text(value: str | None) -> str | None:
    """Turn a model-element description into compact tooltip text."""
    if not value:
        return None
    text = re.sub(r"<[^>]+>", " ", value)
    text = html.unescape(text)
    text = " ".join(text.split())
    return text or None


def _site_model_element_metadata(
        model: ScenarioModel,
        elem_id: str,
        ) -> dict:
    """Return display metadata for elements that derive from SiteModelElement."""
    from .prob import SiteModelElement

    element = model.model_elements.get(elem_id)
    if element is None or not isinstance(element, SiteModelElement):
        return {
                "site_model_element": False,
                "pretty_name": None,
                "short_description": None,
                }
    try:
        pretty_name = element.pretty_name
    except Exception:
        pretty_name = None
    try:
        short_description = _plain_text(element.short_description)
    except Exception:
        short_description = None
    return {
            "site_model_element": True,
            "pretty_name": pretty_name,
            "short_description": short_description,
            }


def build_mcmc_group_dependency_data(
        model: ScenarioModel,
        mcmc_group: str,
        ) -> dict:
    """Build a JSON-serializable description of one inference group.

    The result contains elements, decorator occurrences, canonical variables,
    define/access dependency edges, and an element-level matrix.
    """
    element_ids = _group_element_ids(model, mcmc_group)
    occurrences = _collect_property_occurrences(model, element_ids)

    key_to_variable_id: dict[VarKey, str] = {}
    variable_id_to_key: dict[str, VarKey] = {}

    def variable_id_for(var_key: VarKey) -> str:
        var_key = _canonical_var_key(var_key)
        variable_id = key_to_variable_id.get(var_key)
        if variable_id is None:
            variable_id = f"v{len(key_to_variable_id)}"
            key_to_variable_id[var_key] = variable_id
            variable_id_to_key[variable_id] = var_key
        return variable_id

    element_data = {}
    for elem_id in element_ids:
        metadata = _site_model_element_metadata(model, elem_id)
        display = (
                metadata["pretty_name"]
                if metadata["site_model_element"] and metadata["pretty_name"]
                else elem_id.replace("_", " ")
                )
        element_data[elem_id] = {
                "id": elem_id,
                "display": display,
                "pretty_name": metadata["pretty_name"] or display,
                "site_model_element": metadata["site_model_element"],
                "short_description": metadata["short_description"],
                "methods": {
                    method_key: {"defines": [], "accesses": []}
                    for method_key in _METHOD_KEYS_BY_ORDER
                    },
                }

    occurrence_data = []
    variable_meta: dict[str, dict] = {}

    for occurrence in occurrences:
        method_key = _METHOD_KEYS[occurrence.method]
        variable_ids = sorted({
            variable_id_for(var_key)
            for var_key in occurrence.lookup_keys
            })
        occurrence_id = f"o{len(occurrence_data)}"
        backward = (
                occurrence.direction == "access"
                and occurrence.method == "model_element_annual_step"
                and occurrence.attr == "model_element_access_this_carry"
                )
        occurrence_data.append({
            "id": occurrence_id,
            "element": occurrence.elem_id,
            "method": method_key,
            "method_label": _METHOD_LABELS[occurrence.method],
            "direction": occurrence.direction,
            "kind": occurrence.kind,
            "name": occurrence.name,
            "variable_ids": variable_ids,
            "backward": backward,
            })

        method_data = element_data[occurrence.elem_id]["methods"][method_key]
        if occurrence.direction == "define":
            method_data["defines"].append(occurrence_id)
        else:
            method_data["accesses"].append(occurrence_id)

        for variable_id in variable_ids:
            meta = variable_meta.setdefault(variable_id, {
                "id": variable_id,
                "names": set(),
                "kinds": set(),
                "defines": [],
                "accesses": [],
                })
            meta["names"].add(occurrence.name)
            meta["kinds"].add(occurrence.kind)
            if occurrence.direction == "define":
                meta["defines"].append(occurrence_id)
            else:
                meta["accesses"].append(occurrence_id)

    occurrence_by_id = {o["id"]: o for o in occurrence_data}

    variables = []
    for variable_id, meta in variable_meta.items():
        defines = _dedupe(meta["defines"])
        accesses = _dedupe(meta["accesses"])
        names = sorted(meta["names"])

        primary_name = None
        for occurrence_id in defines + accesses:
            primary_name = occurrence_by_id[occurrence_id]["name"]
            break
        if primary_name is None:
            # This should not happen: every variable comes from an occurrence.
            primary_name = variable_id

        kinds = meta["kinds"]
        if "carry" in kinds:
            kind = "carry"
        elif "annual" in kinds:
            kind = "annual"
        else:
            kind = "general"

        define_elements = {
                occurrence_by_id[oid]["element"] for oid in defines
                }
        access_elements = {
                occurrence_by_id[oid]["element"] for oid in accesses
                }
        cross_element = any(
                define_elem != access_elem
                for define_elem in define_elements
                for access_elem in access_elements)
        self_element = any(
                define_elem == access_elem
                for define_elem in define_elements
                for access_elem in access_elements)

        variables.append({
            "id": variable_id,
            "name": primary_name,
            "display": primary_name.replace("_", " "),
            "aliases": names,
            "kind": kind,
            "defines": defines,
            "accesses": accesses,
            "define_elements": sorted(define_elements),
            "access_elements": sorted(access_elements),
            "cross_element": cross_element,
            "self_element": self_element,
            "fan_in": len(define_elements),
            "fan_out": len(access_elements),
            })

    for element in element_data.values():
        defines = []
        accesses = []
        for method in element["methods"].values():
            defines.extend(method["defines"])
            accesses.extend(method["accesses"])
        defines = _dedupe(defines)
        accesses = _dedupe(accesses)
        element["define_occurrences"] = defines
        element["access_occurrences"] = accesses
        element["define_variable_ids"] = sorted({
            variable_id
            for occurrence_id in defines
            for variable_id in occurrence_by_id[
                occurrence_id]["variable_ids"]
            })
        element["access_variable_ids"] = sorted({
            variable_id
            for occurrence_id in accesses
            for variable_id in occurrence_by_id[
                occurrence_id]["variable_ids"]
            })
        element["n_defines"] = len(defines)
        element["n_accesses"] = len(accesses)

    edges = []
    for variable in variables:
        for define_occurrence_id in variable["defines"]:
            for access_occurrence_id in variable["accesses"]:
                define_occurrence = occurrence_by_id[define_occurrence_id]
                access_occurrence = occurrence_by_id[access_occurrence_id]
                edges.append({
                    "id": f"e{len(edges)}",
                    "source": define_occurrence_id,
                    "target": access_occurrence_id,
                    "variable": variable["id"],
                    "source_element": define_occurrence["element"],
                    "source_method": define_occurrence["method"],
                    "target_element": access_occurrence["element"],
                    "target_method": access_occurrence["method"],
                    "backward": access_occurrence["backward"],
                    "cross_element": (
                        define_occurrence["element"]
                        != access_occurrence["element"]),
                    })

    matrix_by_pair: dict[tuple[str, str], dict] = {}
    for edge in edges:
        pair = (edge["source_element"], edge["target_element"])
        cell = matrix_by_pair.setdefault(pair, {
            "source": pair[0],
            "target": pair[1],
            "variable_ids": set(),
            "edge_ids": [],
            "source_methods": set(),
            "target_methods": set(),
            })
        cell["variable_ids"].add(edge["variable"])
        cell["edge_ids"].append(edge["id"])
        cell["source_methods"].add(edge["source_method"])
        cell["target_methods"].add(edge["target_method"])

    matrix = [
            {
                "source": cell["source"],
                "target": cell["target"],
                "count": len(cell["variable_ids"]),
                "variable_ids": sorted(cell["variable_ids"]),
                "edge_ids": cell["edge_ids"],
                "source_methods": sorted(cell["source_methods"]),
                "target_methods": sorted(cell["target_methods"]),
                }
            for cell in matrix_by_pair.values()
            ]

    element_order = {elem_id: ii for ii, elem_id in enumerate(element_ids)}
    matrix.sort(key=lambda cell: (
        element_order[cell["source"]],
        element_order[cell["target"]],
        ))

    top_fanout_variables = sorted(
            variables,
            key=lambda v: (-v["fan_out"], v["name"]))[:10]
    top_fanin_elements = sorted(
            element_data.values(),
            key=lambda e: (-len(e["access_variable_ids"]), e["id"]))[:10]
    top_fanout_elements = sorted(
            element_data.values(),
            key=lambda e: (-len(e["define_variable_ids"]), e["id"]))[:10]

    return {
        "group": mcmc_group,
        "elements": [element_data[elem_id] for elem_id in element_ids],
        "occurrences": occurrence_data,
        "variables": variables,
        "edges": edges,
        "matrix": matrix,
        "summary": {
            "n_elements": len(element_ids),
            "n_variables": len(variables),
            "n_occurrences": len(occurrence_data),
            "n_edges": len(edges),
            "n_cross_edges": sum(
                1 for edge in edges if edge["cross_element"]),
            "top_fanout_variables": [
                {
                    "id": variable["id"],
                    "name": variable["name"],
                    "fan_out": variable["fan_out"],
                    }
                for variable in top_fanout_variables
                ],
            "top_fanin_elements": [
                {
                    "id": element["id"],
                    "display": element["display"],
                    "fan_in": len(element["access_variable_ids"]),
                    }
                for element in top_fanin_elements
                ],
            "top_fanout_elements": [
                {
                    "id": element["id"],
                    "display": element["display"],
                    "fan_out": len(element["define_variable_ids"]),
                    }
                for element in top_fanout_elements
                ],
            },
        }


def render_mcmc_group_html_explorer(
        model: ScenarioModel,
        mcmc_group: str,
        *,
        model_name: str | None = None,
        ) -> str:
    """Render one inference group as an interactive HTML explorer fragment.

    The fragment is self-contained apart from the shared explorer CSS/JS
    assets and the Cytoscape.js CDN script.
    """
    data = build_mcmc_group_dependency_data(model, mcmc_group)
    if model_name is not None:
        data["model"] = model_name

    # Escape closing tags so the JSON can be embedded in a script element.
    data_json = json.dumps(
            data,
            separators=(",", ":"),
            ).replace("</", "<\\/")

    return (
            '<div class="mcmc-explorer" data-mcmc-explorer>'
            '<script type="application/json" class="mcmc-explorer-data">'
            f'{data_json}'
            '</script>'
            '<div class="mcmc-explorer-loading">'
            'Loading dependency explorer…'
            '</div>'
            '<noscript>'
            '<div class="mcmc-explorer-error">'
            'The interactive dependency explorer requires JavaScript.'
            '</div>'
            '</noscript>'
            '</div>'
            '<link rel="stylesheet" href="/assets/css/mcmc-explorer.css">'
            '<script src="https://cdn.jsdelivr.net/npm/cytoscape@3.30.2/'
            'dist/cytoscape.min.js"></script>'
            '<script src="/assets/js/mcmc-explorer.js"></script>'
            )
