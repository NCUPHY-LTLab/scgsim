"""Rule 2: the components declare their semantics the way ``docs/geometry-sgb.qmd`` documents.

Owns the checks of ``component_semantics`` (the record keys, the conductor regions, the
``ground_plane_contributions`` and the ports), of the levels they name, and of the registered
instances. Each record goes through the same parser ``GeometryPlan`` uses, so a record accepted here is
accepted there; this module only runs them one record at a time so that every defect is reported.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterator, Mapping, Sequence
from typing import Any

from scgsim.geometry.drc.report import (
    SEMANTIC_CONTRACT,
    Layer,
    LayoutFinding,
    Location,
)
from scgsim.geometry.source.normalization import gds_layer
from scgsim.geometry.source.occurrence import selector_location
from scgsim.geometry.source.plan import compose_affine, instance_affine
from scgsim.geometry.source.plan_assembly import (
    GROUND_CONTRIBUTION_KEYS,
    component_semantics,
    entity_declaration,
    ground_contribution,
    source_path,
    source_port,
    transform_source_point,
)

Transform = tuple[float, float, float, float, float, float]
_IDENTITY_TRANSFORM: Transform = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
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
    placements = {root_id: _IDENTITY_TRANSFORM}
    for path in sorted(instances, key=lambda item: (item.count("/"), item)):
        problem = _instance_problem(path, instances[path], cells)
        if problem is None:
            cells[path] = instances[path].cell
            placements[path] = compose_affine(
                placements[path.rpartition("/")[0]], instance_affine(instances[path])
            )
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
        findings.extend(_cell_findings(path, cell, levels, placements[path]))
    return findings


def entities_by_layer(cells: Mapping[str, Any]) -> dict[Layer, list[str]]:
    """The qualified Entity and port IDs declared on each GDS layer, for naming a polygon's owner.

    Only usable records count: a malformed record is rule 2's to report, and never stops this lookup.
    """
    owners: dict[Layer, list[str]] = {}
    for path, cell in cells.items():
        try:
            semantics = component_semantics(cell, path)
        except (ValueError, TypeError):
            continue  # rule 2 reports it
        for key, name_key, layer_key in (
            ("conductor_regions", "semantic_id", "gds_layer"),
            ("ports", "name", "layer"),
        ):
            for record in _records(semantics, key):
                layer = _layer_of(record, layer_key)
                name = record.get(name_key) if isinstance(record, Mapping) else None
                if layer is not None and isinstance(name, str):
                    owners.setdefault(layer, []).append(f"{path}/{name}")
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
    try:
        instance_affine(reference)
    except ValueError as error:
        return f"instance {path!r}: {error}"
    return None


def _stack_levels(layer_stack: Any) -> Mapping[str, Any] | None:
    levels = getattr(layer_stack, "layers", None)
    return levels if isinstance(levels, Mapping) else None


def _cell_findings(
    path: str, cell: Any, levels: Mapping[str, Any] | None, placement: Transform
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
        yield from _conductor_region_findings(path, raw, levels, placement)
    for raw in _records(semantics, "ground_plane_contributions"):
        yield from _contribution_findings(path, raw, levels, placement)
    for raw in _records(semantics, "ports"):
        yield from _port_findings(path, cell, raw, levels, placement)


def _conductor_region_findings(
    path: str, raw: Any, levels: Mapping[str, Any] | None, placement: Transform
) -> Iterator[LayoutFinding]:
    entity = f"{path}/{raw.get('semantic_id')}" if isinstance(raw, Mapping) else path
    layer = _layer_of(raw, "gds_layer")
    where = (
        _selector_of(raw.get("geometry"), placement)
        if isinstance(raw, Mapping)
        else None
    )
    try:
        entity_declaration(path, raw, placement, {})
        gds_layer(raw.get("gds_layer"), entity)
    except (ValueError, TypeError) as error:
        yield _finding(
            entity,
            str(error),
            "each conductor region needs a one-segment semantic_id, a level and a gds_layer",
            layer,
            where,
        )
        return
    yield from _unknown_level(
        entity, raw.get("level"), "conductor region", levels, layer, where
    )


def _contribution_findings(
    path: str, raw: Any, levels: Mapping[str, Any] | None, placement: Transform
) -> Iterator[LayoutFinding]:
    entity = f"{path}/{raw.get('local_id')}" if isinstance(raw, Mapping) else path
    layer = _layer_of(raw, "layer")
    where = _selector_of(raw, placement)
    try:
        ground_contribution(path, raw, placement, set())
    except (ValueError, TypeError) as error:
        message = _contribution_message(raw, error, layer, where)
        yield _finding(entity, message, _GROUND_CONTRIBUTIONS_HINT, layer, where)
        return
    yield from _unknown_level(
        entity, raw["level"], "ground contribution", levels, layer, where
    )


def _contribution_message(
    raw: Any, error: Exception, layer: Layer | None, where: Location | None
) -> str:
    """The parser's message, the keys that differ, and which of layer and location the record lacks."""
    message = str(error)
    if isinstance(raw, Mapping) and set(raw) != GROUND_CONTRIBUTION_KEYS:
        missing = sorted(GROUND_CONTRIBUTION_KEYS - set(raw))
        unexpected = sorted(set(raw) - GROUND_CONTRIBUTION_KEYS)
        message += f"; missing keys {missing}, unexpected keys {unexpected}"
    unavailable = [
        name for name, value in (("layer", layer), ("location", where)) if value is None
    ]
    if unavailable:
        message += (
            f"; {' and '.join(unavailable)} unavailable "
            "(the record has no usable 'layer' or 'selector_point_um')"
        )
    return message


def _port_findings(
    path: str,
    cell: Any,
    raw: Any,
    levels: Mapping[str, Any] | None,
    placement: Transform,
) -> Iterator[LayoutFinding]:
    entity = f"{path}/{raw.get('name')}" if isinstance(raw, Mapping) else path
    layer = _layer_of(raw, "layer")
    try:
        record = source_port(path, cell, raw, placement, 1)
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
            layer,
        )
        return
    x, y, *_ = record["selector_point_um"]
    yield from _unknown_level(
        entity, raw["target_layer"], "port target", levels, layer, (x, y)
    )


def _unknown_level(  # noqa: PLR0913 - the finding's own fields
    entity: str,
    level: Any,
    what: str,
    levels: Mapping[str, Any] | None,
    layer: Layer | None,
    where: Location | None,
) -> Iterator[LayoutFinding]:
    if levels is None:
        return
    if not (isinstance(level, Hashable) and level in levels):
        yield _finding(
            entity,
            f"{what} level {level!r} is not a level of the layer stack",
            "use a level name that layer_stack.layers declares",
            layer,
            where,
        )


def _records(semantics: Mapping[str, Any], key: str) -> tuple[Any, ...]:
    return tuple(semantics.get(key, ()))


def _layer_of(record: Any, key: str) -> Layer | None:
    """The GDS layer and datatype a record names under ``key``, or ``None`` when it names none usable."""
    value = record.get(key) if isinstance(record, Mapping) else None
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 2
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    ):
        return value[0], value[1]
    return None


def _selector_of(geometry: Any, placement: Transform) -> Location | None:
    """The record's selector point in the frame of the checked component, or ``None`` without a usable one."""
    point = selector_location(geometry)
    if point is None:
        return None
    x, y, *_ = transform_source_point(placement, point)
    return x, y


def _finding(  # noqa: PLR0913 - the finding's own fields
    entity: str,
    message: str,
    hint: str,
    layer: Layer | None = None,
    location: Location | None = None,
) -> LayoutFinding:
    return LayoutFinding(SEMANTIC_CONTRACT, layer, entity, location, message, hint)
