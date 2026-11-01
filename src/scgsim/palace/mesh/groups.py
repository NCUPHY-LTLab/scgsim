"""Map SGB physical-group records onto live Gmsh physical groups.

SGB (semantic geometry builder) writes the XAO geometry plus one JSON record per
physical group. After Gmsh opens the XAO, this module joins each record to the
live group with the same (dimension, physical name) and files it under one of the
mesh ``groups`` sections that the Palace configuration is later written from.

It owns the record-to-section decisions. Record completeness is checked in
``record_checks.py``, lumped-port sheets in ``port_groups.py``, and anything that
touches the Gmsh mesh itself in ``lowering.py`` and ``checks_gmsh.py``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

from .._values import optional_stripped_string
from .identity import SGB_RUNTIME_AUTHORITY
from .port_groups import add_port_surface_group
from .record_checks import (
    STRUCTURED_INTERFACE_TYPES,
    has_structured_record_marker,
    is_solver_active_volume,
    structured_conductor_source_layer,
    validate_route_representation,
    validate_structured_domain_boundary_record,
    validate_structured_surface_record,
    validate_structured_volume_record,
    validate_surface_adjacency,
    validate_surface_contribution_ledger,
)

# Sections of the mesh ``groups`` mapping, filled in this order of creation:
# volumes (solution regions), boundary_surfaces (every classified interface or
# domain boundary), surface_groups (named but unclassified surfaces),
# pec_surfaces (perfect-electric-conductor metal faces), port_surfaces
# (lumped-port sheets keyed ``P<index>``).
MeshGroups = dict[str, dict[str, Any]]


class LivePhysicalGroup(NamedTuple):
    """A physical group as Gmsh currently holds it."""

    physical_group: int
    entity_tags: tuple[int, ...]


LiveGroups = Mapping[tuple[int, str], LivePhysicalGroup]


class SurfaceContext(NamedTuple):
    """Route-wide facts a surface record is judged against."""

    route: str
    solution_material_kinds: Mapping[str, str]


class _SurfaceRow(NamedTuple):
    """One boundary-surface entry to be written for a record."""

    name: str
    interface_type: str
    surface_epr: bool
    source_id: str | None = None
    face_kind: str | None = None


def empty_groups() -> MeshGroups:
    """Return the empty sections, in the order they appear in the artifacts."""
    return {
        "volumes": {},
        "boundary_surfaces": {},
        "surface_groups": {},
        "pec_surfaces": {},
        "port_surfaces": {},
    }


def augment_groups_from_live_records(
    *,
    records: Sequence[Any],
    groups: MeshGroups,
    live: LiveGroups,
    route: str,
    solution_material_kinds: Mapping[str, str],
) -> None:
    """Populate mesh groups by exact (dimension, physical_name) mapping.

    Solver-active records with no live Gmsh group are collected and reported
    together at the end, so one error lists every missing group.
    """
    route = route.upper()
    missing = _check_record_routes(records, route)
    context = SurfaceContext(route, solution_material_kinds)
    for record in records:
        _map_record(record, groups, live, context, missing)
    if missing:
        raise ValueError(
            "SGB XAO is missing solver-active physical groups from sidecar: "
            + ", ".join(sorted(missing))
        )


def _check_record_routes(records: Sequence[Any], route: str) -> list[str]:
    """Require every record to be an object of this route; flag unknown record kinds."""
    missing: list[str] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("SGB physical group records must be JSON objects.")
        if optional_stripped_string(record.get("route", "")) != route:
            raise ValueError(f"SGB physical-group record route must be {route}.")
        if (
            record.get("sgb_record")
            not in {None, "final_physical_group", "preliminary_physical_group"}
            and str(record.get("solver_use")) == "solver_active"
        ):
            missing.append("unknown_sgb_record")
    return missing


def _map_record(
    record: Any,
    groups: MeshGroups,
    live: LiveGroups,
    context: SurfaceContext,
    missing: list[str],
) -> None:
    """File one record under its section, or note that its live group is missing."""
    if not isinstance(record, Mapping):
        raise TypeError("SGB physical group records must be JSON objects.")
    target = _resolve_live_target(record, live, context.route, missing)
    if target is None:
        return
    dim, name, live_group = target

    if dim == 3 and not is_solver_active_volume(record):
        raise ValueError(
            f"SGB physical volume {name!r} is not an explicit solver-active "
            "material_volume and cannot enter the Palace solution."
        )
    if dim == 2 and record.get("role") == "material_volume":
        raise ValueError(f"SGB surface {name!r} cannot claim material_volume role.")

    if dim == 3 and is_solver_active_volume(record):
        _add_volume_group(groups, name, live_group, record)
    elif dim == 2:
        _add_surface_group_records(groups, name, live_group, record, context)


def _resolve_live_target(
    record: Mapping[str, Any],
    live: LiveGroups,
    route: str,
    missing: list[str],
) -> tuple[int, str, LivePhysicalGroup] | None:
    """Return (dimension, name, live group), or None when the record is skipped."""
    dim = int(record.get("dimension", 0) or 0)
    name = optional_stripped_string(record.get("physical_name"))
    if not name:
        if record.get("solver_use") == "solver_active":
            missing.append(f"{dim}:<missing physical_name>")
        return None

    if optional_stripped_string(record.get("route", "")) != route:
        raise ValueError(
            f"SGB physical-group route must be {route} for this mesh mode."
        )
    if record.get("solver_use") == "not_solver_active":
        return None

    if (dim, name) not in live:
        if str(record.get("solver_use", "")).lower() == "solver_active":
            missing.append(f"{dim}:{name}")
        return None
    return dim, name, live[(dim, name)]


def _add_volume_group(
    groups: MeshGroups,
    name: str,
    live_group: LivePhysicalGroup,
    record: Mapping[str, Any],
) -> None:
    """Add one solver-active solution volume."""
    validate_structured_volume_record(name, record)
    role = str(record.get("role", "")).lower()
    if "construction" in role or role == "mm_contact":
        raise ValueError(
            f"SGB volume {name!r} has construction-only/MM-contact role {role!r} "
            "and must not become a solver volume."
        )

    info = {
        "phys_group": live_group.physical_group,
        "tags": list(live_group.entity_tags),
        "dim": 3,
        "source": SGB_RUNTIME_AUTHORITY,
        "physical_name": name,
        "stack_layer": record.get("source_provenance", {}).get("stack_layer"),
        "source_layer": record.get("source_provenance", {}).get(
            "conductor_source_layer_name"
        ),
        "representation": record.get("representation"),
        "route": record.get("route"),
        "source_provenance": dict(record.get("source_provenance", {})),
        "physical_attribute": dict(record.get("physical_attribute", {})),
        "sgb_record": "final_physical_group",
        "solver_use": record.get("solver_use", "solver_active"),
        "role": record.get("role"),
        "owner_semantic_ids": tuple(record.get("owner_semantic_ids", ())),
        "adjacent_solution_volume_ids": tuple(
            record.get("adjacent_solution_volume_ids", ())
        ),
        "metadata": dict(record.get("metadata", {})),
    }

    groups["volumes"][name] = info


def _add_surface_group_records(
    groups: MeshGroups,
    name: str,
    live_group: LivePhysicalGroup,
    record: Mapping[str, Any],
    context: SurfaceContext,
) -> None:
    """Classify structured surface records into boundary + interface mappings."""
    if record.get("solver_use") == "not_solver_active":
        return

    role = str(record.get("role", "")).lower()
    if role == "domain_boundary":
        validate_structured_domain_boundary_record(name, record)
        groups["boundary_surfaces"][name] = _domain_boundary_info(
            name, live_group, record
        )
        return

    if not has_structured_record_marker(record):
        if str(record.get("solver_use", "")).lower() == "solver_active":
            raise ValueError(f"SGB surface {name!r} is missing structured metadata.")
        groups["surface_groups"][name] = {"physical_name": name, "dim": 2}
        return

    if str(record.get("solver_use")) == "not_solver_active":
        return

    interface_type = str(record.get("interface_type", "")).upper()
    if interface_type == "LUMPED_PORT" or role in {"lumped_port", "port_surface"}:
        add_port_surface_group(groups, name, live_group, record, context.route)
        return
    _add_interface_surface(groups, name, live_group, record, context)


def _domain_boundary_info(
    name: str, live_group: LivePhysicalGroup, record: Mapping[str, Any]
) -> dict[str, Any]:
    """Describe a domain-boundary surface, keeping every identity field of its record."""
    metadata = record.get("metadata")
    source_record_ids = (
        tuple(
            value
            for value in (
                metadata.get("source_record_ids", ())
                if isinstance(metadata, Mapping)
                else ()
            )
            if isinstance(value, str) and value
        )
        if isinstance(metadata, Mapping)
        else ()
    )
    return {
        "phys_group": live_group.physical_group,
        "tags": list(live_group.entity_tags),
        "dim": 2,
        "source": "domain_boundary",
        "surface_epr": False,
        "representation": str(record.get("representation", "")),
        "route": record.get("route"),
        "interface_type": record.get("interface_type"),
        "contact_kind": record.get("contact_kind"),
        "face_kind": optional_stripped_string(record.get("face_kind")),
        "source_id": optional_stripped_string(record.get("surface_id")) or name,
        "owner_semantic_ids": tuple(record.get("owner_semantic_ids", ())),
        "adjacent_solution_volume_ids": tuple(
            record.get("adjacent_solution_volume_ids", ())
        ),
        "surface_id": optional_stripped_string(record.get("surface_id")),
        "physical_attribute": dict(record.get("physical_attribute", {})),
        "source_provenance": dict(record.get("source_provenance", {})),
        "conductor_component_id": optional_stripped_string(
            record.get("conductor_component_id")
        ),
        "net_id": optional_stripped_string(record.get("net_id")),
        "equipotential_id": optional_stripped_string(record.get("equipotential_id")),
        "sgb_record": "final_physical_group",
        "solver_use": record.get("solver_use", "solver_active"),
        "sgb_metadata": dict(record.get("metadata", {})),
        "source_record_ids": source_record_ids,
    }


def _add_interface_surface(
    groups: MeshGroups,
    name: str,
    live_group: LivePhysicalGroup,
    record: Mapping[str, Any],
    context: SurfaceContext,
) -> None:
    """Validate an interface surface and register its boundary (and PEC) entries."""
    interface_type = str(record.get("interface_type", "")).upper()
    validate_structured_surface_record(name=name, record=record)
    validate_route_representation(name=name, record=record, route=context.route)
    if interface_type not in STRUCTURED_INTERFACE_TYPES:
        raise ValueError(
            f"SGB surface {name!r} has unsupported interface_type {interface_type!r}."
        )
    validate_surface_contribution_ledger(name, record, interface_type)

    component_id = optional_stripped_string(record.get("conductor_component_id"))
    net_id = optional_stripped_string(record.get("net_id"))
    if interface_type in {"MA", "MS", "MS_MA"} and (
        component_id is None or net_id is None
    ):
        raise ValueError(
            f"SGB conductor surface {name!r} must include conductor_component_id and net_id."
        )
    if interface_type == "SA" and net_id is not None:
        raise ValueError(
            f"SGB SA surface {name!r} must not carry conductor/net identity."
        )

    if interface_type in {"MM", "SS", "AA"}:
        groups["boundary_surfaces"][name] = _structured_surface_info(
            _SurfaceRow(name, interface_type, surface_epr=False), live_group, record
        )
        return
    validate_surface_adjacency(
        name, record, interface_type, context.solution_material_kinds
    )
    for row in _interface_rows(name, record, interface_type):
        if row.interface_type not in STRUCTURED_INTERFACE_TYPES:
            continue
        groups["boundary_surfaces"][row.name] = _structured_surface_info(
            row, live_group, record
        )
        if row.interface_type in {"MA", "MS"} and component_id is not None:
            groups["pec_surfaces"][row.name] = {
                "phys_group": live_group.physical_group,
                "tags": list(live_group.entity_tags),
                "dim": 2,
                "source": SGB_RUNTIME_AUTHORITY,
                "source_id": row.source_id,
                "layer": row.source_id,
                "representation": str(record.get("representation", "")).upper(),
                "owner_semantic_ids": tuple(record.get("owner_semantic_ids", ())),
            }


def _interface_rows(
    name: str, record: Mapping[str, Any], interface_type: str
) -> tuple[_SurfaceRow, ...]:
    """List the boundary entries one interface record produces (two for MS_MA)."""
    surface_id = str(record["surface_id"])
    if interface_type == "MS_MA":
        source_id = structured_conductor_source_layer(name=name, record=record)
        # One SGB physical attribute carries both semantic aliases.  Do not use
        # geometric orientation or material names to invent top/bottom meaning.
        return (
            _SurfaceRow(f"{surface_id}__MS", "MS", True, source_id, "interface"),
            _SurfaceRow(f"{surface_id}__MA", "MA", True, source_id, "interface"),
        )
    source_id = (
        surface_id
        if interface_type == "SA"
        else structured_conductor_source_layer(name=name, record=record)
    )
    return (
        _SurfaceRow(
            surface_id,
            interface_type,
            interface_type in {"MA", "MS", "SA"},
            source_id,
            str(record["face_kind"]),
        ),
    )


def _structured_surface_info(
    row: _SurfaceRow, live_group: LivePhysicalGroup, record: Mapping[str, Any]
) -> dict[str, Any]:
    """Keep every structured identity field when lowering a surface group."""
    return {
        "phys_group": live_group.physical_group,
        "tags": list(live_group.entity_tags),
        "dim": 2,
        "source": "volume_interface",
        "surface_epr": row.surface_epr,
        "interface_id": row.name,
        "interface_type": row.interface_type,
        "sgb_record": "final_physical_group",
        "solver_use": record.get("solver_use"),
        "conductor_component_id": optional_stripped_string(
            record.get("conductor_component_id")
        ),
        "net_id": optional_stripped_string(record.get("net_id")),
        "equipotential_id": optional_stripped_string(record.get("equipotential_id")),
        "physical_attribute": dict(record.get("physical_attribute", {})),
        "source_id": row.source_id
        or optional_stripped_string(record.get("surface_id"))
        or row.name,
        "face_kind": row.face_kind or optional_stripped_string(record.get("face_kind")),
        "contact_kind": optional_stripped_string(record.get("contact_kind")),
        "representation": optional_stripped_string(record.get("representation")),
        "route": optional_stripped_string(record.get("route")),
        "source_provenance": dict(record.get("source_provenance", {})),
        "source_record_ids": tuple(
            record.get("metadata", {}).get("source_record_ids", ())
        )
        if isinstance(record.get("metadata"), Mapping)
        else (),
        "surface_id": optional_stripped_string(record.get("surface_id")) or row.name,
        "owner_semantic_ids": tuple(record.get("owner_semantic_ids", ())),
        "adjacent_solution_volume_ids": tuple(
            record.get("adjacent_solution_volume_ids", ())
        ),
    }
