"""Build the Palace ``config.json`` of each problem from the mesh's physical groups and the problem's settings.

``build_eigenmode_config`` and ``build_electrostatic_config`` take a ``ConfigInput`` (the mesh, materials,
Surface-EPR specs and numerical controls both problems share) and what the problem adds.  Each returns the
config with its index entries (``config.index``) and the records written beside it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any, NamedTuple

from scgsim.contracts.palace_run_folder import EIGENMODE, ELECTROSTATIC, OUTPUT_DIR
from scgsim.geometry.models.refusals import Declared, internal
from scgsim.refusal import found_text

from ..mesh.physical_groups import Groups
from ..source.ports import LayoutPortBinding
from .boundaries import build_electrostatic_boundaries
from .config_records import (
    MODEL_L0_M,
    ConfigInput,
    EigenmodeConfigBuildResult,
    EigenmodeSettings,
    ElectrostaticConfigBuildResult,
    ElectrostaticSettings,
    TerminalBinding,
    optional_string,
)
from .index import EnergyEntry, IndexedRows, LumpedPortRows, PortEntry, SurfaceEprEntry
from .numerical import build_output_formats, build_refinement, linear_solver
from .surface_epr import build_surface_epr_postprocessing

_WHERE_PLAN_CONDUCTORS = "the GeometryPlan's conductors"
_WHERE_PORTS_ON_LOSS = "the calls to add_port and set_surface_epr"
_WHERE_STACK_MATERIALS = "the stack file's materials"

__all__ = [
    "build_eigenmode_config",
    "build_electrostatic_config",
]


def build_electrostatic_config(
    common: ConfigInput,
    *,
    terminals: Sequence[TerminalBinding],
    ground_net_ids: Sequence[str],
    electrostatic: ElectrostaticSettings,
) -> ElectrostaticConfigBuildResult:
    """Build the Electrostatic config: terminals and grounds by Net, the domains, Surface EPR and the solver.

    Raises:
        ValueError: a terminal or ground Net does not match the mesh, a material is invalid, or a required
            Surface-EPR spec has no face.
    """
    boundaries = build_electrostatic_boundaries(
        groups=common.groups,
        terminals=terminals,
        ground_net_ids=ground_net_ids,
        electrostatic=electrostatic,
    )
    domain_volumes = _domains_from_solution_volumes(common.groups.get("volumes", {}))
    material_rows = _materials_from_solution_volumes(domain_volumes, common.materials)
    epr = _surface_epr_rows(common)
    energy = _domain_energy_rows(domain_volumes, common.materials)
    config = _config(
        common,
        solver={ELECTROSTATIC.name: {"Save": electrostatic.save_fields}},
        boundaries=boundaries.boundaries,
        domains=_Domains(material_rows, energy),
        epr=epr,
    )
    return ElectrostaticConfigBuildResult(
        config=config,
        index_entries=[*boundaries.terminals, *energy.entries, *epr.entries],
        material_resolution=_material_resolution(domain_volumes, common.materials),
        ground_boundary_resolution=boundaries.ground_resolution,
    )


def build_eigenmode_config(
    common: ConfigInput,
    *,
    ports: Sequence[LayoutPortBinding],
    eigenmode: EigenmodeSettings,
) -> EigenmodeConfigBuildResult:
    """Build one non-Floquet Eigenmode config: the PEC conductors, lumped ports, domains, Surface EPR and solver.

    Raises:
        ValueError: a port does not match its SGB group, a port is also a Surface-EPR loss group, the mesh has
            no PEC surface, a material is invalid, or a required Surface-EPR spec has no face.
    """
    groups = common.groups
    domain_volumes = _domains_from_solution_volumes(groups.get("volumes", {}))
    material_rows = _materials_from_solution_volumes(domain_volumes, common.materials)
    energy = _domain_energy_rows(domain_volumes, common.materials)
    epr = _surface_epr_rows(common)
    ports_rows = _lumped_ports(groups, ports)
    _refuse_ports_on_loss_groups(ports_rows.rows, epr.rows)
    boundaries: dict[str, Any] = {"PEC": {"Attributes": _pec_attributes(groups)}}
    if ports_rows.rows:
        boundaries["LumpedPort"] = ports_rows.rows
    config = _config(
        common,
        solver={EIGENMODE.name: _eigenmode_block(eigenmode)},
        boundaries=boundaries,
        domains=_Domains(material_rows, energy),
        epr=epr,
    )
    return EigenmodeConfigBuildResult(
        config=config,
        port_information=ports_rows.information,
        index_entries=[*ports_rows.entries, *energy.entries, *epr.entries],
        material_resolution=_material_resolution(domain_volumes, common.materials),
    )


def _surface_epr_rows(common: ConfigInput) -> IndexedRows[SurfaceEprEntry]:
    """The Surface-EPR ``Dielectric`` rows and their index entries; none when Surface EPR was not requested."""
    if common.surface_epr_specs is None:
        return IndexedRows([], [])
    return build_surface_epr_postprocessing(
        common.groups, common.surface_epr_specs, model_l0_m=MODEL_L0_M
    )


@dataclass(frozen=True)
class _Domains:
    """The ``Domains`` of a config: one material row per solution volume, and the energy rows and entries."""

    materials: list[dict[str, Any]]
    energy: IndexedRows[EnergyEntry]


def _config(
    common: ConfigInput,
    *,
    solver: dict[str, Any],
    boundaries: dict[str, Any],
    domains: _Domains,
    epr: IndexedRows[SurfaceEprEntry],
) -> dict[str, Any]:
    """The config both problems write, in Palace's section order, around the problem's solver block."""
    numerical = common.numerical
    if epr.rows:
        boundaries["Postprocessing"] = {"Dielectric": epr.rows}
    config: dict[str, Any] = {
        "Problem": {
            "Type": common.problem,
            "Verbose": 3,
            "Output": OUTPUT_DIR,
            **build_output_formats(numerical),
        },
        "Model": {
            "Mesh": common.mesh_path.name,
            "L0": MODEL_L0_M,
            "Refinement": build_refinement(numerical),
        },
        "Solver": {
            "Linear": linear_solver(numerical, problem=common.problem),
            "Order": numerical.order,
            "Device": numerical.device,
            **solver,
        },
        "Boundaries": boundaries,
        "Domains": {
            "Materials": domains.materials,
            "Postprocessing": {"Energy": domains.energy.rows, "Probe": []},
        },
    }
    if _requires_internal_boundary_cracking(epr.entries):
        config["Model"]["CrackInternalBoundaryElements"] = True
    return config


def _eigenmode_block(eigenmode: EigenmodeSettings) -> dict[str, Any]:
    """``Solver.Eigenmode``: the number of modes, the tolerance, the saved fields and the target in GHz."""
    block: dict[str, Any] = {
        "N": eigenmode.num_modes,
        "Tol": eigenmode.tolerance,
        "Save": eigenmode.save_fields,
    }
    if eigenmode.target_hz is not None:
        block["Target"] = eigenmode.target_hz / 1e9
    return block


def _pec_attributes(groups: Groups) -> list[int]:
    """The attributes of every SGB-authored PEC surface; an Eigenmode model has at least one."""
    attributes = sorted(
        {
            int(value)
            for info in groups.get("pec_surfaces", {}).values()
            for value in info["attributes"]
        }
    )
    if not attributes:
        raise ValueError(
            Declared(
                "Eigenmode config requires SGB-authored PEC surfaces; the plan's conductors produced none",
                _WHERE_PLAN_CONDUCTORS,
            ).refusal(found_text(sorted(groups.get("pec_surfaces", {}))))
        )
    return attributes


def _refuse_ports_on_loss_groups(
    port_rows: Sequence[Mapping[str, Any]], epr_rows: Sequence[Mapping[str, Any]]
) -> None:
    """Refuse a lumped port whose attributes are also a Surface-EPR ``Dielectric`` row's."""
    epr_attributes = {
        int(value) for row in epr_rows for value in row.get("Attributes", ())
    }
    port_attributes = {
        int(value) for row in port_rows for value in row.get("Attributes", ())
    }
    if epr_attributes.intersection(port_attributes):
        raise ValueError(
            Declared(
                "LumpedPort attributes must not also be Surface-EPR loss groups",
                _WHERE_PORTS_ON_LOSS,
            ).refusal(found_text(sorted(epr_attributes.intersection(port_attributes))))
        )


def _requires_internal_boundary_cracking(
    epr_entries: Sequence[SurfaceEprEntry],
) -> bool:
    """Whether a configured Surface-EPR row integrates one two-sided face from both sides.

    A row has a ``shared_face_group`` exactly when it belongs to a Route A conductor sheet, an internal face
    whose MA and MS rows integrate the same face once per side.  Palace keeps the two sides apart only when it
    cracks internal boundary elements (``Model.CrackInternalBoundaryElements``, which then duplicates the
    nodes of the sheet), so the key is written exactly when a configured row integrates such a face and left
    at Palace's default otherwise.  Sidecar rows no spec selects integrate nothing and do not count.
    """
    return any(
        row.shared_face_group is not None for entry in epr_entries for row in entry.rows
    )


def _domain_energy_rows(
    domain_volumes: Mapping[str, Mapping[str, Any]],
    materials: Mapping[str, Mapping[str, Any]],
) -> IndexedRows[EnergyEntry]:
    """The ``Domains.Postprocessing.Energy`` rows, one per solution volume, and their index entries."""
    rows = [
        {"Index": index, "Attributes": sorted(info["attributes"])}
        for index, info in enumerate(domain_volumes.values(), start=1)
    ]
    entries = [
        EnergyEntry(
            index=index,
            name=name,
            attributes=row["Attributes"],
            material=_material_identity(info, materials),
            entities=list(info["entities"]),
        )
        for index, ((name, info), row) in enumerate(
            zip(domain_volumes.items(), rows, strict=True), start=1
        )
    ]
    return IndexedRows(rows, entries)


def _material_resolution(
    domain_volumes: Mapping[str, Mapping[str, Any]],
    materials: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "solution_volume": name,
            "attributes": sorted(volume["attributes"]),
            "material": _material_identity(volume, materials),
        }
        for name, volume in domain_volumes.items()
    ]


def _lumped_ports(groups: Groups, ports: Sequence[LayoutPortBinding]) -> LumpedPortRows:
    """The ``LumpedPort`` rows, port information and index entries of ``ports``, each bound to its SGB group ``P<index>``.

    A row's circuit keys are the port's ``circuit``, written unchanged; no port is excited.
    """
    rows: list[dict[str, Any]] = []
    information: list[dict[str, Any]] = []
    entries: list[PortEntry] = []
    for port in ports:
        bound = _bound_port_group(groups, port)
        rows.append(
            {
                "Index": port.index,
                "Direction": list(bound.direction),
                "Excitation": False,
                "Attributes": bound.attributes,
                **port.circuit,
            }
        )
        information.append(
            {
                "portnumber": port.index,
                "name": port.name,
                "type": "lumped_sheet",
                "direction": list(bound.direction),
                "physical_name": bound.group["group"],
                "attributes": bound.attributes,
                "source_layer": port.source_layer,
                "target_layer": port.target_layer,
            }
        )
        entries.append(_port_entry(port, bound))
    if len(groups.get("port_surfaces", {})) != len(ports):
        raise ValueError(
            internal(
                "SGB produced unconfigured or missing layout-sheet port groups"
                + ": "
                + found_text(
                    sorted(groups.get("port_surfaces", {})), f"{len(ports)} port groups"
                )
            )
        )
    return LumpedPortRows(rows, entries, information)


class _BoundPortGroup(NamedTuple):
    """The SGB group a layout-sheet port binds to, with its checked direction and its attributes."""

    key: str
    group: Mapping[str, Any]
    direction: tuple[float, float, float]
    attributes: list[int]


def _bound_port_group(groups: Groups, port: LayoutPortBinding) -> _BoundPortGroup:
    """Find the port's group ``P<index>`` and require it to carry the port's name, layers and direction."""
    key = f"P{port.index}"
    group = groups.get("port_surfaces", {}).get(key)
    if group is None:
        raise ValueError(
            internal(
                f"layout_sheet port {port.name!r} has no SGB group {key}"
                + ": "
                + found_text(sorted(groups.get("port_surfaces", {})))
            )
        )
    direction = _direction_tuple(group.get("direction"), port.name)
    if (
        group["port_index"] != port.index
        or group["source_name"] != port.name
        or group["source_layer"] != port.source_layer
        or group["target_layer"] != port.target_layer
        or not all(
            math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
            for a, b in zip(direction, port.direction, strict=True)
        )
    ):
        raise ValueError(
            internal(
                f"layout_sheet port {port.name!r} does not match SGB-authored identity"
                + ": "
                + found_text(
                    {
                        "port_index": group["port_index"],
                        "source_name": group["source_name"],
                        "source_layer": group["source_layer"],
                        "target_layer": group["target_layer"],
                        "direction": direction,
                    },
                    {
                        "port_index": port.index,
                        "source_name": port.name,
                        "source_layer": port.source_layer,
                        "target_layer": port.target_layer,
                        "direction": port.direction,
                    },
                )
            )
        )
    attributes = sorted(int(value) for value in group["attributes"])
    return _BoundPortGroup(key, group, direction, attributes)


def _port_entry(port: LayoutPortBinding, bound: _BoundPortGroup) -> PortEntry:
    """The index-map entry of one lumped port: its row, its group and the sheet SGB cut for it."""
    return PortEntry(
        index=port.index,
        entry_name=bound.key,
        attributes=bound.attributes,
        physical_names=[bound.group["group"]],
        port_name=port.name,
        sheet={
            field: bound.group[field]
            for field in (
                "port_sheet_id",
                "host_entity_ids",
                "adjacent_volumes",
                "host_volumes",
            )
        },
    )


def _direction_tuple(value: Any, name: str) -> tuple[float, float, float]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, (list, tuple))
        or len(value) != 3
    ):
        raise ValueError(f"layout_sheet port {name!r} requires a 3D direction.")
    direction = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in direction):
        raise ValueError(f"layout_sheet port {name!r} direction must be finite.")
    return direction[0], direction[1], direction[2]


def _domains_from_solution_volumes(
    groups: Mapping[str, Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    """The solver volumes by group name, in name order: the order the domains are numbered in."""
    return {name: dict(groups[name]) for name in sorted(groups)}


def _materials_from_solution_volumes(
    domain_volumes: Mapping[str, Mapping[str, Any]],
    materials: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build Palace-domain material rows from explicit material records."""
    if not isinstance(materials, Mapping) or not materials:
        raise ValueError(
            Declared(
                "stack must include a non-empty top-level materials mapping for config generation",
                _WHERE_STACK_MATERIALS,
            ).refusal(
                found_text(
                    materials if not isinstance(materials, Mapping) else dict(materials)
                )
            )
        )

    material_rows: list[dict[str, Any]] = []
    seen_attributes: set[int] = set()
    for name, volume in domain_volumes.items():
        attrs = sorted(int(item) for item in volume["attributes"])
        material_id = optional_string(volume.get("material"))
        if not material_id:
            raise ValueError(
                Declared(
                    f"SGB solution volume {name!r} must define a non-empty material id",
                    _WHERE_STACK_MATERIALS,
                ).refusal(found_text(volume.get("material")))
            )
        constants = _dielectric_constants(materials, material_id, name)
        if seen_attributes.intersection(attrs):
            raise ValueError(
                f"SGB solution volume {name!r} duplicates physical attributes {attrs}."
            )
        seen_attributes.update(attrs)
        material_rows.append(
            {
                "Attributes": attrs,
                "Permittivity": constants.permittivity,
                "LossTan": constants.loss_tangent,
            }
        )
    return material_rows


class _DielectricConstants(NamedTuple):
    """The relative permittivity and loss tangent a solution volume's material gives Palace."""

    permittivity: float
    loss_tangent: float


def _dielectric_constants(
    materials: Mapping[str, Mapping[str, Any]], material_id: str, volume: str
) -> _DielectricConstants:
    """Read a solution material's constants, refusing an unknown, non-dielectric or out-of-range record."""
    material = materials.get(material_id)
    if material is None:
        raise ValueError(
            Declared(
                f"SGB solution volume {volume!r} references unknown material id {material_id!r}",
                _WHERE_STACK_MATERIALS,
            ).refusal(found_text(material_id, sorted(materials)))
        )
    if not isinstance(material, Mapping):
        raise TypeError(
            Declared(
                f"SGB material record {material_id!r} must be a mapping",
                _WHERE_STACK_MATERIALS,
            ).refusal(found_text(type(material).__name__))
        )
    if material.get("kind") not in {"vacuum", "dielectric"}:
        raise ValueError(
            Declared(
                f"SGB solution material {material_id!r} must have vacuum or dielectric kind",
                _WHERE_STACK_MATERIALS,
            ).refusal(found_text(material.get("kind")))
        )
    permittivity = material.get("permittivity")
    loss_tangent = material.get("loss_tangent")
    if (
        not isinstance(permittivity, Real)
        or not math.isfinite(float(permittivity))
        or float(permittivity) <= 0.0
    ):
        raise ValueError(
            Declared(
                f"Material {material_id!r} requires finite positive permittivity",
                _WHERE_STACK_MATERIALS,
            ).refusal(found_text(permittivity))
        )
    if (
        not isinstance(loss_tangent, Real)
        or not math.isfinite(float(loss_tangent))
        or float(loss_tangent) < 0.0
    ):
        raise ValueError(
            Declared(
                f"Material {material_id!r} requires finite non-negative loss_tangent",
                _WHERE_STACK_MATERIALS,
            ).refusal(found_text(loss_tangent))
        )
    return _DielectricConstants(float(permittivity), float(loss_tangent))


def _material_identity(
    volume: Mapping[str, Any], materials: Mapping[str, Mapping[str, Any]]
) -> dict[str, str]:
    material_id = volume.get("material")
    if not isinstance(material_id, str) or not material_id:
        raise ValueError(
            Declared(
                f"solution volume {volume.get('group')!r} must carry one explicit material id",
                _WHERE_STACK_MATERIALS,
            ).refusal(found_text(material_id))
        )
    material = materials.get(material_id)
    if not isinstance(material, Mapping) or material.get("kind") not in {
        "vacuum",
        "dielectric",
    }:
        raise ValueError(
            f"solution material {material_id!r} must carry vacuum or dielectric kind."
        )
    return {"material_id": material_id, "kind": str(material["kind"])}
