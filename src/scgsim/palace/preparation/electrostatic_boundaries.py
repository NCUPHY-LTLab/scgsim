"""Synthesize the Electrostatic Terminal and Ground boundaries.

Palace Electrostatic needs each conducting net as a Terminal (a capacitance matrix
row) or as Ground. SGB (the geometry builder) labels conductor surfaces with a
``conductor_component_id`` and a ``net_id``; here every Terminal is matched to the
components on its net, and everything else is grounded or refused. Matching is exact:
no name is guessed from another.

Not here: assembling the whole config (``config.py``) and the domain rows
(``solution_volumes.py``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypedDict

from .._values import optional_stripped_string
from .solution_volumes import physical_group_values


@dataclass(frozen=True)
class TerminalBinding:
    """Exact net mapping record for a conductor terminal."""

    name: str
    net_id: str


class TerminalIndexEntry(TypedDict):
    """One ``Boundaries.Terminal`` entry of ``palace_index_map.json``."""

    section: str
    index: int
    entry_name: str
    role: str
    attributes: list[int]
    physical_names: list[str]
    metadata: dict[str, Any]
    terminal_name: str
    net_id: str
    conductor_component_ids: list[str]
    terminal_attributes: list[int]


class GroundBoundaryResolution(TypedDict):
    """The ``ground_boundary`` record of the Electrostatic index map.

    Physical Ground is conductor surface the caller named (or left unassigned);
    exterior Ground is the outer domain boundary. ``union_attributes`` is what
    ``Boundaries.Ground.Attributes`` holds in ``config.json``.
    """

    requested_net_ids: list[str]
    physical_net_ids: list[str]
    conductor_component_ids: list[str]
    physical_names: list[str]
    physical_attributes: list[int]
    exterior_physical_names: list[str]
    exterior_attributes: list[int]
    union_attributes: list[int]


@dataclass(frozen=True)
class _Conductors:
    """The solver-active conductor components SGB (the geometry builder) produced."""

    attributes: dict[str, set[int]]  # component id -> Palace boundary attributes
    nets: dict[str, str]  # component id -> net id
    physical_names: dict[str, set[str]]  # component id -> gmsh physical group names
    owner_of_attribute: dict[int, str]  # boundary attribute -> component id


@dataclass(frozen=True)
class _TerminalBoundaries:
    entries: list[dict[str, object]]  # config.json Boundaries.Terminal
    index_map: list[TerminalIndexEntry]
    assigned: set[str]  # component ids that belong to a Terminal


def electrostatic_boundaries(
    *,
    groups: Mapping[str, Mapping[str, Mapping[str, Any]]],
    terminals: Sequence[TerminalBinding],
    ground_net_ids: Sequence[str],
    unassigned_conductor_policy: str,
    exterior_boundary_policy: str,
) -> tuple[dict[str, object], list[TerminalIndexEntry], GroundBoundaryResolution]:
    """Map exact conductor_component_id/net_id pairs to terminal and ground boundaries."""
    conductors = _collect_conductors(groups)
    bound = _bind_terminals(groups, terminals, conductors)
    assigned = bound.assigned
    requested_ground_nets, physical_ground_components = _bind_physical_ground(
        ground_net_ids, conductors, assigned
    )

    unassigned = set(conductors.attributes).difference(assigned)
    if unassigned and unassigned_conductor_policy == "error":
        raise ValueError(
            "Unassigned conductor components: " + ", ".join(sorted(unassigned))
        )
    if unassigned_conductor_policy == "ground":
        physical_ground_components.update(unassigned)

    boundary_map: dict[str, object] = {"Terminal": bound.entries}
    physical_ground_attrs: set[int] = {
        int(attr)
        for component_id in physical_ground_components
        for attr in conductors.attributes[component_id]
    }
    exterior_ground_attrs, exterior_physical_names = _exterior_ground(
        groups, exterior_boundary_policy
    )

    if set(conductors.owner_of_attribute).intersection(exterior_ground_attrs):
        raise ValueError("conductor and exterior boundary attributes must be disjoint.")
    ground_attrs = physical_ground_attrs | exterior_ground_attrs
    if ground_attrs:
        boundary_map["Ground"] = {"Attributes": sorted(ground_attrs)}

    if sorted(
        (entry["index"], list(entry["attributes"])) for entry in bound.index_map
    ) != sorted((entry["Index"], list(entry["Attributes"])) for entry in bound.entries):
        raise ValueError("Terminal-to-boundary mapping mismatch after synthesis.")

    ground_components = sorted(physical_ground_components)
    return (
        boundary_map,
        bound.index_map,
        {
            "requested_net_ids": sorted(requested_ground_nets),
            "physical_net_ids": sorted(
                {conductors.nets[component_id] for component_id in ground_components}
            ),
            "conductor_component_ids": ground_components,
            "physical_names": sorted(
                {
                    name
                    for component_id in ground_components
                    for name in conductors.physical_names[component_id]
                }
            ),
            "physical_attributes": sorted(physical_ground_attrs),
            "exterior_physical_names": sorted(exterior_physical_names),
            "exterior_attributes": sorted(exterior_ground_attrs),
            "union_attributes": sorted(ground_attrs),
        },
    )


def _collect_conductors(
    groups: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> _Conductors:
    """Gather the metal surfaces (MA, MS interfaces) that Palace will see as PEC.

    MA and MS are the metal-air and metal-substrate interface types.
    """
    attributes: dict[str, set[int]] = {}
    nets: dict[str, str] = {}
    physical_names: dict[str, set[str]] = {}
    owner_of_attribute: dict[int, str] = {}

    for physical_name, info in groups.get("boundary_surfaces", {}).items():
        if not isinstance(info, Mapping):
            continue
        if (
            info.get("sgb_record") != "final_physical_group"
            or info.get("source") != "volume_interface"
            or info.get("solver_use") != "solver_active"
            or str(info.get("interface_type")).upper() not in {"MA", "MS"}
        ):
            continue

        component_id = optional_stripped_string(info.get("conductor_component_id"))
        net_id = optional_stripped_string(info.get("net_id"))
        if component_id is None:
            raise ValueError(
                "SGB solver-active conductor surface missing conductor_component_id."
            )
        if net_id is None:
            raise ValueError(
                f"SGB conductor component {component_id!r} missing exact net_id."
            )

        attrs = physical_group_values(info.get("phys_group"))
        if not attrs:
            raise ValueError(
                f"SGB conductor component {component_id!r} has no physical attributes."
            )

        attributes.setdefault(component_id, set()).update(attrs)
        if component_id in nets and nets[component_id] != net_id:
            raise ValueError(
                f"SGB conductor component {component_id!r} has conflicting net_id values."
            )
        nets[component_id] = net_id
        physical_names.setdefault(component_id, set()).add(str(physical_name))

        for attr in attrs:
            existing = owner_of_attribute.get(int(attr))
            if existing is None:
                owner_of_attribute[int(attr)] = component_id
            elif existing != component_id:
                raise ValueError(
                    f"SGB physical attribute {int(attr)} belongs to multiple conductor components."
                )

    if not attributes:
        raise ValueError(
            "SGB config needs at least one solver-active conductor component."
        )
    return _Conductors(attributes, nets, physical_names, owner_of_attribute)


def _bind_terminals(
    groups: Mapping[str, Mapping[str, Mapping[str, Any]]],
    terminals: Sequence[TerminalBinding],
    conductors: _Conductors,
) -> _TerminalBoundaries:
    """One Palace Terminal per binding, made of every component on its net."""
    entries: list[dict[str, object]] = []
    index_map: list[TerminalIndexEntry] = []
    assigned: set[str] = set()
    seen_nets: set[str] = set()

    for index, terminal in enumerate(terminals, start=1):
        net_id = optional_stripped_string(terminal.net_id)
        name = optional_stripped_string(terminal.name)
        if net_id is None or name is None:
            raise ValueError("terminal binding names and net ids must be non-empty.")
        if net_id in seen_nets:
            raise ValueError(f"SGB terminal net {net_id!r} is assigned more than once.")
        seen_nets.add(net_id)

        matching = sorted(
            component_id
            for component_id, value in conductors.nets.items()
            if value == net_id
        )
        if not matching:
            raise ValueError(f"SGB terminal {name!r} has no exact net match.")
        overlap = assigned.intersection(matching)
        if overlap:
            raise ValueError(
                f"SGB terminal {name!r} splits an already-assigned conductor component."
            )

        attrs = sorted(
            int(attr)
            for component_id in matching
            for attr in conductors.attributes[component_id]
        )
        entries.append({"Index": index, "Attributes": attrs})
        index_map.append(
            {
                "section": "Boundaries.Terminal",
                "index": index,
                "entry_name": name,
                "role": "terminal",
                "attributes": attrs,
                "physical_names": sorted(
                    name
                    for name, info in groups.get("boundary_surfaces", {}).items()
                    if isinstance(info, Mapping)
                    and optional_stripped_string(info.get("conductor_component_id"))
                    in matching
                ),
                "metadata": {
                    "net_id": net_id,
                    "conductor_component_ids": matching,
                },
                "terminal_name": name,
                "net_id": net_id,
                "conductor_component_ids": matching,
                "terminal_attributes": attrs,
            }
        )
        assigned.update(matching)
    return _TerminalBoundaries(entries, index_map, assigned)


def _bind_physical_ground(
    ground_net_ids: Sequence[str], conductors: _Conductors, assigned: set[str]
) -> tuple[list[str], set[str]]:
    """Components on the requested Ground nets. They are added to ``assigned``."""
    requested_nets: list[str] = []
    components: set[str] = set()
    for value in ground_net_ids:
        net_id = optional_stripped_string(value)
        if net_id is None:
            raise ValueError("physical Ground net ids must be non-empty.")
        if net_id in requested_nets:
            raise ValueError(f"SGB physical Ground net {net_id!r} is repeated.")
        matching = {
            component_id
            for component_id, candidate in conductors.nets.items()
            if candidate == net_id
        }
        if not matching:
            raise ValueError(f"SGB physical Ground net {net_id!r} has no exact match.")
        overlap = assigned.intersection(matching)
        if overlap:
            raise ValueError(
                f"SGB physical Ground net {net_id!r} overlaps a Terminal binding."
            )
        requested_nets.append(net_id)
        components.update(matching)
        assigned.update(matching)
    return requested_nets, components


def _exterior_ground(
    groups: Mapping[str, Mapping[str, Mapping[str, Any]]], policy: str
) -> tuple[set[int], set[str]]:
    """Attributes and physical names of the outer domain boundary, if it is grounded."""
    attributes: set[int] = set()
    names: set[str] = set()
    if policy != "ground":
        return attributes, names
    for physical_name, info in groups.get("boundary_surfaces", {}).items():
        if (
            isinstance(info, Mapping)
            and info.get("sgb_record") == "final_physical_group"
            and info.get("source") == "domain_boundary"
            and info.get("solver_use") == "solver_active"
        ):
            attributes.update(physical_group_values(info.get("phys_group")))
            names.add(str(physical_name))
    if not attributes:
        raise ValueError(
            "exterior_boundary_policy='ground' requires at least one exterior boundary."
        )
    return attributes, names
