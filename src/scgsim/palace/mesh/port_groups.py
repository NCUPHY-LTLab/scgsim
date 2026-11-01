"""Validate and register SGB lumped-port sheets.

A lumped-port sheet is a 2D surface across the gap between two conductors, where
Palace applies a lumped port. SGB authors the sheet with its owners, direction and
embedded volume; this module checks that those fields agree with each other and
files the sheet under ``groups["port_surfaces"]["P<index>"]``.

It never reconstructs a port from geometry. A disagreement is an error, not
something to repair.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from .._values import optional_stripped_string
from .identity import SGB_RUNTIME_AUTHORITY
from .record_checks import (
    normalized_direction,
    required_distinct_strings,
    required_string,
)

if TYPE_CHECKING:
    from .groups import LivePhysicalGroup, MeshGroups


def add_port_surface_group(
    groups: MeshGroups,
    name: str,
    live_group: LivePhysicalGroup,
    record: Mapping[str, Any],
    route: str,
) -> None:
    """Preserve one SGB-authored layout sheet without reconstructing its meaning."""
    if record.get("solver_use") != "solver_active":
        raise ValueError(f"SGB lumped-port sheet {name!r} must be solver_active.")
    if record.get("representation") != "lumped_port_sheet":
        raise ValueError(
            f"SGB lumped-port sheet {name!r} requires lumped_port_sheet representation."
        )
    surface_id = required_string(record.get("surface_id"), f"{name} surface_id")
    owners = required_distinct_strings(
        record.get("owner_semantic_ids"), f"{name} owner_semantic_ids", count=2
    )
    # Route A expects two adjacent solution volumes; Route B only the one it is
    # embedded in (checked against ``embedded_volume_id`` below).
    adjacent = required_distinct_strings(
        record.get("adjacent_solution_volume_ids"),
        f"{name} adjacent_solution_volume_ids",
        count=2 if route == "A" else 1,
    )
    attribute = record.get("physical_attribute")
    provenance = record.get("source_provenance")
    if not isinstance(attribute, Mapping) or not isinstance(provenance, Mapping):
        raise TypeError(
            f"SGB lumped-port sheet {name!r} requires physical_attribute and source_provenance."
        )
    port_index = _port_index(name, attribute)
    port_name = required_string(attribute.get("port_name"), f"{name} port_name")
    source_layer = required_string(
        attribute.get("source_layer"), f"{name} source_layer"
    )
    target_layer = required_string(
        attribute.get("target_layer"), f"{name} target_layer"
    )
    embedded_volume_id = required_string(
        attribute.get("embedded_volume_id"), f"{name} embedded_volume_id"
    )
    direction = normalized_direction(attribute.get("direction"), name)
    attribute_owners = required_distinct_strings(
        attribute.get("owner_semantic_ids"),
        f"{name} physical_attribute.owner_semantic_ids",
        count=2,
    )
    if attribute_owners != owners:
        raise ValueError(f"SGB lumped-port sheet {name!r} owner identities disagree.")
    _check_owner_provenance(name, attribute, owners)
    if route == "A":
        if embedded_volume_id not in adjacent:
            raise ValueError(
                f"SGB Route-A lumped-port sheet {name!r} embedded volume must be in actual adjacency."
            )
    elif adjacent != (embedded_volume_id,):
        raise ValueError(
            f"SGB Route-B lumped-port sheet {name!r} requires one matching embedded volume."
        )
    if (
        provenance.get("source_name") != port_name
        or provenance.get("port_index") != port_index
        or provenance.get("source_layer") != source_layer
        or provenance.get("target_layer") != target_layer
        or normalized_direction(provenance.get("direction"), name) != direction
        or not optional_stripped_string(provenance.get("source_polygon_id"))
        or not optional_stripped_string(provenance.get("direction_sign_convention"))
    ):
        raise ValueError(
            f"SGB lumped-port sheet {name!r} source provenance is inconsistent."
        )
    key = f"P{port_index}"
    if key in groups["port_surfaces"]:
        raise ValueError(f"Duplicate SGB lumped-port sheet index {port_index}.")
    groups["port_surfaces"][key] = {
        "phys_group": live_group.physical_group,
        "tags": list(live_group.entity_tags),
        "dim": 2,
        "type": "lumped_sheet",
        "source": SGB_RUNTIME_AUTHORITY,
        "surface_epr": False,
        "sgb_record": "final_physical_group",
        "solver_use": record.get("solver_use"),
        "role": record.get("role"),
        "route": route,
        "representation": record.get("representation"),
        "surface_id": surface_id,
        "interface_type": "lumped_port",
        "face_kind": record.get("face_kind"),
        "owner_semantic_ids": owners,
        "adjacent_solution_volume_ids": adjacent,
        "interface_of": adjacent if route == "A" else (),
        "embedded_volume_id": embedded_volume_id,
        "direction": direction,
        "physical_name": name,
        "source_provenance": dict(provenance),
        "physical_attribute": dict(attribute),
        "conductor_component_id": record.get("conductor_component_id"),
        "net_id": record.get("net_id"),
        "equipotential_id": record.get("equipotential_id"),
    }


def _port_index(name: str, attribute: Mapping[str, Any]) -> int:
    """Return the 1-based port number; ``bool`` is rejected although it is an int."""
    port_index = attribute.get("port_index")
    if (
        isinstance(port_index, bool)
        or not isinstance(port_index, int)
        or port_index < 1
    ):
        raise ValueError(f"SGB lumped-port sheet {name!r} requires 1-based port_index.")
    return port_index


def _check_owner_provenance(
    name: str, attribute: Mapping[str, Any], owners: tuple[str, ...]
) -> None:
    """Require one provenance row per owner conductor, naming only known owners."""
    owner_provenance = attribute.get("owner_provenance")
    if not isinstance(owner_provenance, (list, tuple)) or len(owner_provenance) != 2:
        raise ValueError(
            f"SGB lumped-port sheet {name!r} requires two owner provenance rows."
        )
    for item in owner_provenance:
        if not isinstance(item, Mapping):
            raise TypeError(
                f"SGB lumped-port sheet {name!r} owner provenance must be mappings."
            )
        owner_id = required_string(item.get("semantic_id"), f"{name} owner semantic_id")
        if owner_id not in owners:
            raise ValueError(
                f"SGB lumped-port sheet {name!r} has foreign owner provenance."
            )
        for field in ("conductor_component_id", "net_id", "equipotential_id"):
            value = item.get(field)
            if value is not None:
                required_string(value, f"{name} owner {field}")
