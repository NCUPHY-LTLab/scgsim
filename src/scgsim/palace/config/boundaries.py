"""Synthesize electrostatic Terminal and Ground boundaries from the attributes of SGB groups.

A Terminal is the attributes of the conductor surfaces of one exact Net (the ``net`` field of the sidecar's
conductor surface groups), the exterior Ground is the attributes of the ``EXT`` groups, and a physical Ground
is chosen by Net.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from .config_models import ElectrostaticSettings, TerminalBinding, optional_string
from .index import TerminalEntry


@dataclass
class _BoundaryState:
    """Collected identities used while building one electrostatic boundary map."""

    groups: Mapping[str, Mapping[str, Mapping[str, Any]]]
    component_attrs: dict[str, set[int]] = field(default_factory=dict)
    component_nets: dict[str, str] = field(default_factory=dict)
    component_physical_names: dict[str, set[str]] = field(default_factory=dict)
    attribute_components: dict[int, str] = field(default_factory=dict)

    def collect_conductors(self) -> None:
        """Collect active conductor component, net and physical-attribute identities."""
        for physical_name, info in self.groups.get("pec_surfaces", {}).items():
            component_id, net_id, attrs = _conductor_identity(info)
            self._record_component(physical_name, component_id, net_id, attrs)
        if not self.component_attrs:
            raise ValueError(
                "SGB config needs at least one solver-active conductor component."
            )

    def _record_component(
        self, physical_name: str, component_id: str, net_id: str, attrs: tuple[int, ...]
    ) -> None:
        self.component_attrs.setdefault(component_id, set()).update(attrs)
        previous_net = self.component_nets.get(component_id)
        if previous_net is not None and previous_net != net_id:
            raise ValueError(
                f"SGB conductor component {component_id!r} has conflicting net_id values."
            )
        self.component_nets[component_id] = net_id
        self.component_physical_names.setdefault(component_id, set()).add(
            str(physical_name)
        )
        for attribute in attrs:
            previous_component = self.attribute_components.get(attribute)
            if previous_component is not None and previous_component != component_id:
                raise ValueError(
                    f"SGB physical attribute {attribute} belongs to multiple conductor components."
                )
            self.attribute_components[attribute] = component_id


@dataclass
class _BoundaryOutput:
    """Rows and identities assembled for the final solver boundary map."""

    terminal_rows: list[dict[str, object]]
    terminal_index: list[TerminalEntry]
    requested_grounds: list[str]
    physical_ground_components: set[str]
    exterior_attrs: set[int]
    exterior_names: set[str]


def build_electrostatic_boundaries(
    *,
    groups: Mapping[str, Mapping[str, Mapping[str, Any]]],
    terminals: Sequence[TerminalBinding],
    ground_net_ids: Sequence[str],
    electrostatic: ElectrostaticSettings,
) -> tuple[dict[str, object], list[TerminalEntry], dict[str, Any]]:
    """Build exact-net terminal and physical/exterior ground assignments, under the problem's boundary policies."""
    state = _BoundaryState(groups)
    state.collect_conductors()
    terminal_rows, terminal_index, assigned = _terminal_bindings(state, terminals)
    requested_grounds, physical_ground_components = _physical_grounds(
        state, ground_net_ids, assigned
    )
    _assign_unrequested_conductors(
        state,
        assigned,
        physical_ground_components,
        electrostatic.unassigned_conductor_policy,
    )
    exterior_attrs, exterior_names = _exterior_grounds(
        groups, electrostatic.exterior_boundary_policy
    )
    return _assemble_boundaries(
        state,
        _BoundaryOutput(
            terminal_rows,
            terminal_index,
            requested_grounds,
            physical_ground_components,
            exterior_attrs,
            exterior_names,
        ),
    )


def _conductor_identity(info: Mapping[str, Any]) -> tuple[str, str, tuple[int, ...]]:
    """The conductor component (the sidecar Entity id), its exact Net and its attributes."""
    component_id = optional_string(info.get("entity"))
    net_id = optional_string(info.get("net"))
    if component_id is None:
        raise ValueError("SGB conductor surface missing its Entity id.")
    if net_id is None:
        raise ValueError(
            f"SGB conductor component {component_id!r} missing exact net_id."
        )
    attrs = tuple(int(value) for value in info.get("attributes", ()))
    if not attrs:
        raise ValueError(
            f"SGB conductor component {component_id!r} has no physical attributes."
        )
    return component_id, net_id, attrs


def _terminal_bindings(
    state: _BoundaryState, terminals: Sequence[TerminalBinding]
) -> tuple[list[dict[str, object]], list[TerminalEntry], set[str]]:
    rows: list[dict[str, object]] = []
    index_rows: list[TerminalEntry] = []
    assigned: set[str] = set()
    seen_nets: set[str] = set()
    for index, terminal in enumerate(terminals, start=1):
        name, net_id = _terminal_identity(terminal)
        _require_new_terminal_net(net_id, seen_nets)
        matching = _components_on_net(state, name, net_id)
        _require_terminal_components(name, matching, assigned)
        attrs = _attributes_for_components(state, matching)
        rows.append({"Index": index, "Attributes": attrs})
        index_rows.append(
            _terminal_index_entry(state, terminal, index, matching, attrs)
        )
        assigned.update(matching)
    return rows, index_rows, assigned


def _terminal_identity(terminal: TerminalBinding) -> tuple[str, str]:
    net_id = optional_string(terminal.net_id)
    name = optional_string(terminal.name)
    if net_id is None or name is None:
        raise ValueError("terminal binding names and net ids must be non-empty.")
    return name, net_id


def _require_new_terminal_net(net_id: str, seen_nets: set[str]) -> None:
    if net_id in seen_nets:
        raise ValueError(f"SGB terminal net {net_id!r} is assigned more than once.")
    seen_nets.add(net_id)


def _components_on_net(
    state: _BoundaryState, terminal_name: str, net_id: str
) -> list[str]:
    matching = sorted(
        component_id
        for component_id, candidate in state.component_nets.items()
        if candidate == net_id
    )
    if not matching:
        raise ValueError(f"SGB terminal {terminal_name!r} has no exact net match.")
    return matching


def _require_terminal_components(
    name: str, matching: list[str], assigned: set[str]
) -> None:
    if assigned.intersection(matching):
        raise ValueError(
            f"SGB terminal {name!r} splits an already-assigned conductor component."
        )


def _attributes_for_components(
    state: _BoundaryState, component_ids: Sequence[str]
) -> list[int]:
    return sorted(
        int(attribute)
        for component_id in component_ids
        for attribute in state.component_attrs[component_id]
    )


def _terminal_index_entry(
    state: _BoundaryState,
    terminal: TerminalBinding,
    index: int,
    matching: list[str],
    attrs: list[int],
) -> TerminalEntry:
    physical_names = sorted(
        physical_name
        for physical_name, info in state.groups.get("pec_surfaces", {}).items()
        if optional_string(info.get("entity")) in matching
    )
    return TerminalEntry(
        index=index,
        name=str(terminal.name).strip(),
        net_id=str(terminal.net_id).strip(),
        conductor_component_ids=matching,
        attributes=attrs,
        physical_names=physical_names,
    )


def _physical_grounds(
    state: _BoundaryState, ground_net_ids: Sequence[str], assigned: set[str]
) -> tuple[list[str], set[str]]:
    requested: list[str] = []
    ground_components: set[str] = set()
    for value in ground_net_ids:
        net_id = optional_string(value)
        if net_id is None:
            raise ValueError("physical Ground net ids must be non-empty.")
        if net_id in requested:
            raise ValueError(f"SGB physical Ground net {net_id!r} is repeated.")
        matching = {
            component_id
            for component_id, candidate in state.component_nets.items()
            if candidate == net_id
        }
        if not matching:
            raise ValueError(f"SGB physical Ground net {net_id!r} has no exact match.")
        if assigned.intersection(matching):
            raise ValueError(
                f"SGB physical Ground net {net_id!r} overlaps a Terminal binding."
            )
        requested.append(net_id)
        ground_components.update(matching)
        assigned.update(matching)
    return requested, ground_components


def _assign_unrequested_conductors(
    state: _BoundaryState,
    assigned: set[str],
    ground_components: set[str],
    policy: str,
) -> None:
    unassigned = set(state.component_attrs).difference(assigned)
    if unassigned and policy == "error":
        raise ValueError(
            "Unassigned conductor components: " + ", ".join(sorted(unassigned))
        )
    if policy == "ground":
        ground_components.update(unassigned)


def _exterior_grounds(
    groups: Mapping[str, Mapping[str, Mapping[str, Any]]], policy: str
) -> tuple[set[int], set[str]]:
    attrs: set[int] = set()
    physical_names: set[str] = set()
    if policy == "ground":
        for physical_name, info in groups.get("boundary_surfaces", {}).items():
            if info.get("source") == "domain_boundary":
                attrs.update(int(value) for value in info["attributes"])
                physical_names.add(str(physical_name))
        if not attrs:
            raise ValueError(
                "exterior_boundary_policy='ground' requires at least one exterior boundary."
            )
    return attrs, physical_names


def _assemble_boundaries(
    state: _BoundaryState,
    output: _BoundaryOutput,
) -> tuple[dict[str, object], list[TerminalEntry], dict[str, Any]]:
    if set(state.attribute_components).intersection(output.exterior_attrs):
        raise ValueError("conductor and exterior boundary attributes must be disjoint.")
    boundary_map: dict[str, object] = {"Terminal": output.terminal_rows}
    physical_attrs = _attributes_for_components(
        state, sorted(output.physical_ground_components)
    )
    ground_attrs = set(physical_attrs) | output.exterior_attrs
    if ground_attrs:
        boundary_map["Ground"] = {"Attributes": sorted(ground_attrs)}
    _validate_terminal_map(output.terminal_index, output.terminal_rows)
    return (
        boundary_map,
        output.terminal_index,
        _ground_resolution(state, output, physical_attrs, ground_attrs),
    )


def _validate_terminal_map(
    terminal_index: list[TerminalEntry], terminal_rows: list[dict[str, object]]
) -> None:
    index_mapping = sorted(
        (entry.index, list(entry.attributes)) for entry in terminal_index
    )
    boundary_mapping = sorted(
        (entry["Index"], cast(list[int], entry["Attributes"]))
        for entry in terminal_rows
    )
    if index_mapping != boundary_mapping:
        raise ValueError("Terminal-to-boundary mapping mismatch after synthesis.")


def _ground_resolution(
    state: _BoundaryState,
    output: _BoundaryOutput,
    physical_attrs: list[int],
    union_attrs: set[int],
) -> dict[str, Any]:
    sorted_components = sorted(output.physical_ground_components)
    return {
        "requested_net_ids": sorted(output.requested_grounds),
        "physical_net_ids": sorted(
            {state.component_nets[component] for component in sorted_components}
        ),
        "conductor_component_ids": sorted_components,
        "physical_names": sorted(
            {
                name
                for component in sorted_components
                for name in state.component_physical_names[component]
            }
        ),
        "physical_attributes": sorted(physical_attrs),
        "exterior_physical_names": sorted(output.exterior_names),
        "exterior_attributes": sorted(output.exterior_attrs),
        "union_attributes": sorted(union_attrs),
    }


__all__ = ["build_electrostatic_boundaries"]
