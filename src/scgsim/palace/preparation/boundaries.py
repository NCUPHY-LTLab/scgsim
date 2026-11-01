"""Synthesize electrostatic Terminal and Ground boundaries from SGB groups."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from .config_models import TerminalBinding, optional_string, physical_group_values


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
        for physical_name, info in self.groups.get("boundary_surfaces", {}).items():
            if not _is_active_conductor_surface(info):
                continue
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
    terminal_index: list[dict[str, Any]]
    requested_grounds: list[str]
    physical_ground_components: set[str]
    exterior_attrs: set[int]
    exterior_names: set[str]


def build_electrostatic_boundaries(
    *,
    groups: Mapping[str, Mapping[str, Mapping[str, Any]]],
    terminals: Sequence[TerminalBinding],
    ground_net_ids: Sequence[str],
    unassigned_conductor_policy: str,
    exterior_boundary_policy: str,
) -> tuple[dict[str, object], list[dict[str, Any]], dict[str, Any]]:
    """Build exact-net terminal and physical/exterior ground assignments."""
    state = _BoundaryState(groups)
    state.collect_conductors()
    terminal_rows, terminal_index, assigned = _terminal_bindings(state, terminals)
    requested_grounds, physical_ground_components = _physical_grounds(
        state, ground_net_ids, assigned
    )
    _assign_unrequested_conductors(
        state, assigned, physical_ground_components, unassigned_conductor_policy
    )
    exterior_attrs, exterior_names = _exterior_grounds(groups, exterior_boundary_policy)
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


def _is_active_conductor_surface(info: Any) -> bool:
    if not isinstance(info, Mapping):
        return False
    return (
        info.get("sgb_record") == "final_physical_group"
        and info.get("source") == "volume_interface"
        and info.get("solver_use") == "solver_active"
        and str(info.get("interface_type")).upper() in {"MA", "MS"}
    )


def _conductor_identity(info: Mapping[str, Any]) -> tuple[str, str, tuple[int, ...]]:
    component_id = optional_string(info.get("conductor_component_id"))
    net_id = optional_string(info.get("net_id"))
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
    return component_id, net_id, attrs


def _terminal_bindings(
    state: _BoundaryState, terminals: Sequence[TerminalBinding]
) -> tuple[list[dict[str, object]], list[dict[str, Any]], set[str]]:
    rows: list[dict[str, object]] = []
    index_rows: list[dict[str, Any]] = []
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
) -> dict[str, Any]:
    name = str(terminal.name).strip()
    net_id = str(terminal.net_id).strip()
    physical_names = sorted(
        physical_name
        for physical_name, info in state.groups.get("boundary_surfaces", {}).items()
        if isinstance(info, Mapping)
        and optional_string(info.get("conductor_component_id")) in matching
    )
    return {
        "section": "Boundaries.Terminal",
        "index": index,
        "entry_name": name,
        "role": "terminal",
        "attributes": attrs,
        "physical_names": physical_names,
        "metadata": {"net_id": net_id, "conductor_component_ids": matching},
        "terminal_name": name,
        "net_id": net_id,
        "conductor_component_ids": matching,
        "terminal_attributes": attrs,
    }


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
            if _is_active_exterior_surface(info):
                attrs.update(physical_group_values(info.get("phys_group")))
                physical_names.add(str(physical_name))
        if not attrs:
            raise ValueError(
                "exterior_boundary_policy='ground' requires at least one exterior boundary."
            )
    return attrs, physical_names


def _is_active_exterior_surface(info: Any) -> bool:
    return (
        isinstance(info, Mapping)
        and info.get("sgb_record") == "final_physical_group"
        and info.get("source") == "domain_boundary"
        and info.get("solver_use") == "solver_active"
    )


def _assemble_boundaries(
    state: _BoundaryState,
    output: _BoundaryOutput,
) -> tuple[dict[str, object], list[dict[str, Any]], dict[str, Any]]:
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
    terminal_index: list[dict[str, Any]], terminal_rows: list[dict[str, object]]
) -> None:
    index_mapping = sorted(
        (entry["index"], list(entry["attributes"])) for entry in terminal_index
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
