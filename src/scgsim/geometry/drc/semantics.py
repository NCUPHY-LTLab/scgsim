"""Rule 2: the components declare their semantics the way ``docs/geometry-sgb.qmd`` documents.

Owns the checks of ``component_semantics`` (the record keys, the conductor regions, the
``ground_plane_contributions`` and the ports), of the levels they name, and of the registered
instances. Each record goes through the same parser ``GeometryPlan`` uses, so a record accepted here is
accepted there; this module only runs them one record at a time so that every defect is reported.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from scgsim.geometry.drc.report import (
    SEMANTIC_CONTRACT,
    Layer,
    LayoutFinding,
)
from scgsim.geometry.source.plan_assembly import (
    GROUND_CONTRIBUTION_KEYS,
    component_semantics,
    entity_declaration,
    ground_contribution,
    source_path,
    source_port,
)

_IDENTITY_TRANSFORM = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
_GROUND_CONTRIBUTIONS_HINT = (
    "give every ground_plane_contributions record exactly the keys "
    + ", ".join(sorted(GROUND_CONTRIBUTION_KEYS))
    + " (docs/geometry-sgb.qmd, 'Component-to-backend geometry contract')"
)


def semantic_findings(
    component: Any,
    *,
    root_id: str,
    instances: Mapping[str, Any],
    layer_stack: Any,
) -> list[LayoutFinding]:
    """Findings for the root component, each registered instance and the semantics of their cells."""
    findings: list[LayoutFinding] = []
    cells = {root_id: component}
    for path in sorted(instances, key=lambda item: (item.count("/"), item)):
        problem = _instance_problem(path, instances[path], cells)
        if problem is None:
            cells[path] = instances[path].cell
        else:
            findings.append(
                _finding(
                    path,
                    problem,
                    "register only occurrences that exist in the component, parent first",
                )
            )
    levels = _stack_levels(layer_stack)
    for path, cell in cells.items():
        findings.extend(_cell_findings(path, cell, levels))
    return findings


def entities_by_layer(cells: Mapping[str, Any]) -> dict[Layer, list[str]]:
    """The qualified Entity and port IDs declared on each GDS layer, for naming a polygon's owner."""
    owners: dict[Layer, list[str]] = {}
    for path, cell in cells.items():
        try:
            semantics = component_semantics(cell, path)
        except (ValueError, TypeError):
            continue  # rule 2 reports it
        declared = [
            (f"{path}/{record.get('semantic_id')}", record.get("gds_layer"))
            for record in _records(semantics, "conductor_regions")
        ] + [
            (f"{path}/{record.get('name')}", record.get("layer"))
            for record in _records(semantics, "ports")
        ]
        for entity, layer in declared:
            if _is_layer(layer):
                owners.setdefault((int(layer[0]), int(layer[1])), []).append(entity)
    return owners


def _instance_problem(
    path: str, reference: Any, cells: Mapping[str, Any]
) -> str | None:
    """Why ``reference`` cannot be the registered occurrence ``path``, or ``None``."""
    try:
        source_path(path)
    except ValueError as error:
        return str(error)
    parent_path = path.rpartition("/")[0]
    if parent_path not in cells:
        return f"instance {path!r} has no registered parent {parent_path!r}"
    parent = cells[parent_path]
    if getattr(reference, "parent_cell", None) != parent:
        return f"instance {path!r} is not an occurrence in its declared parent {parent_path!r}"
    matches = [item for item in parent.insts if item.instance == reference.instance]
    if len(matches) != 1:
        return f"instance {path!r} does not resolve to exactly one occurrence in {parent_path!r}"
    return None


def _stack_levels(layer_stack: Any) -> Mapping[str, Any] | None:
    levels = getattr(layer_stack, "layers", None)
    return levels if isinstance(levels, Mapping) else None


def _cell_findings(
    path: str, cell: Any, levels: Mapping[str, Any] | None
) -> Iterator[LayoutFinding]:
    try:
        semantics = component_semantics(cell, path)
    except (ValueError, TypeError) as error:
        yield _finding(
            path,
            str(error),
            "declare info['component_semantics'] with schema_version 2 and only the documented keys",
        )
        return
    for raw in _records(semantics, "conductor_regions"):
        yield from _conductor_region_findings(path, raw, levels)
    for raw in _records(semantics, "ground_plane_contributions"):
        yield from _contribution_findings(path, raw, levels)
    for raw in _records(semantics, "ports"):
        yield from _port_findings(path, cell, raw, levels)


def _conductor_region_findings(
    path: str, raw: Any, levels: Mapping[str, Any] | None
) -> Iterator[LayoutFinding]:
    entity = f"{path}/{raw.get('semantic_id')}" if isinstance(raw, Mapping) else path
    try:
        entity_declaration(path, raw, _IDENTITY_TRANSFORM, {})
    except (ValueError, TypeError) as error:
        yield _finding(
            entity,
            str(error),
            "each conductor region needs a one-segment semantic_id, a level and a gds_layer",
        )
        return
    yield from _unknown_level(entity, raw.get("level"), "conductor region", levels)


def _contribution_findings(
    path: str, raw: Any, levels: Mapping[str, Any] | None
) -> Iterator[LayoutFinding]:
    entity = f"{path}/{raw.get('local_id')}" if isinstance(raw, Mapping) else path
    try:
        ground_contribution(path, raw, _IDENTITY_TRANSFORM, set())
    except (ValueError, TypeError) as error:
        yield _finding(
            entity, _contribution_message(raw, error), _GROUND_CONTRIBUTIONS_HINT
        )
        return
    yield from _unknown_level(entity, raw["level"], "ground contribution", levels)


def _contribution_message(raw: Any, error: Exception) -> str:
    """The parser's message, with the missing and unexpected keys when the keys are what differ."""
    if not isinstance(raw, Mapping) or set(raw) == GROUND_CONTRIBUTION_KEYS:
        return str(error)
    missing = sorted(GROUND_CONTRIBUTION_KEYS - set(raw))
    unexpected = sorted(set(raw) - GROUND_CONTRIBUTION_KEYS)
    return f"{error}; missing keys {missing}, unexpected keys {unexpected}"


def _port_findings(
    path: str, cell: Any, raw: Any, levels: Mapping[str, Any] | None
) -> Iterator[LayoutFinding]:
    entity = f"{path}/{raw.get('name')}" if isinstance(raw, Mapping) else path
    try:
        source_port(path, cell, raw, _IDENTITY_TRANSFORM, 1)
    except (ValueError, TypeError, KeyError) as error:
        detail = (
            f"port {error} does not exist"
            if isinstance(error, KeyError)
            else str(error)
        )
        yield _finding(
            entity,
            detail,
            "declare each port with the name, GDS layer and target level of a port the cell has",
        )
        return
    yield from _unknown_level(entity, raw["target_layer"], "port target", levels)


def _unknown_level(
    entity: str, level: Any, what: str, levels: Mapping[str, Any] | None
) -> Iterator[LayoutFinding]:
    if levels is not None and level not in levels:
        yield _finding(
            entity,
            f"{what} level {level!r} is not a level of the layer stack",
            "use a level name that layer_stack.layers declares",
        )


def _records(semantics: Mapping[str, Any], key: str) -> tuple[Any, ...]:
    return tuple(semantics.get(key, ()))


def _is_layer(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == 2


def _finding(entity: str, message: str, hint: str) -> LayoutFinding:
    return LayoutFinding(SEMANTIC_CONTRACT, None, entity, None, message, hint)
