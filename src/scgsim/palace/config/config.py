"""Build the Palace ``config.json`` of each problem from the mesh's physical groups and the problem's settings.

``build_eigenmode_config`` and ``build_electrostatic_config`` take a ``ConfigInput`` (the mesh, materials,
Surface-EPR specs and numerical controls both problems share) and what the problem adds.  Each returns the
config with its index entries (``config.index``) and the records written beside it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

from ..mesh.physical_groups import Groups
from ..source.ports import LayoutPortBinding
from .boundaries import build_electrostatic_boundaries
from .config_models import (
    MODEL_L0_M,
    ConfigInput,
    EigenmodeConfigBuildResult,
    EigenmodeSettings,
    ElectrostaticConfigBuildResult,
    ElectrostaticSettings,
    TerminalBinding,
    optional_string,
)
from .index import EnergyEntry, PortEntry, SurfaceEprEntry
from .numerical import build_output_formats, build_refinement, linear_solver
from .surface_epr import build_surface_epr_postprocessing

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
    boundaries, terminal_entries, ground_resolution = build_electrostatic_boundaries(
        groups=common.groups,
        terminals=terminals,
        ground_net_ids=ground_net_ids,
        electrostatic=electrostatic,
    )
    domain_volumes = _domains_from_solution_volumes(common.groups.get("volumes", {}))
    material_rows = _materials_from_solution_volumes(domain_volumes, common.materials)
    epr_rows, epr_entries = build_surface_epr_postprocessing(
        common.groups, common.surface_epr_specs, model_l0_m=MODEL_L0_M
    )
    energy_rows, energy_entries = _domain_energy_rows(domain_volumes, common.materials)
    config = _config(
        common,
        solver={"Electrostatic": {"Save": electrostatic.save_fields}},
        boundaries=boundaries,
        domains=(material_rows, energy_rows),
        epr=(epr_rows, epr_entries),
    )
    return ElectrostaticConfigBuildResult(
        config=config,
        index_entries=[*terminal_entries, *energy_entries, *epr_entries],
        material_resolution=_material_resolution(domain_volumes, common.materials),
        ground_boundary_resolution=ground_resolution,
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
    energy_rows, energy_entries = _domain_energy_rows(domain_volumes, common.materials)
    epr_rows, epr_entries = build_surface_epr_postprocessing(
        groups, common.surface_epr_specs, model_l0_m=MODEL_L0_M
    )
    port_rows, port_info, port_entries = _lumped_ports(groups, ports)
    _refuse_ports_on_loss_groups(port_rows, epr_rows)
    boundaries: dict[str, Any] = {"PEC": {"Attributes": _pec_attributes(groups)}}
    if port_rows:
        boundaries["LumpedPort"] = port_rows
    config = _config(
        common,
        solver={"Eigenmode": _eigenmode_block(eigenmode)},
        boundaries=boundaries,
        domains=(material_rows, energy_rows),
        epr=(epr_rows, epr_entries),
    )
    return EigenmodeConfigBuildResult(
        config=config,
        port_information=port_info,
        index_entries=[*port_entries, *energy_entries, *epr_entries],
        material_resolution=_material_resolution(domain_volumes, common.materials),
    )


def _config(
    common: ConfigInput,
    *,
    solver: dict[str, Any],
    boundaries: dict[str, Any],
    domains: tuple[list[dict[str, Any]], list[dict[str, Any]]],
    epr: tuple[list[dict[str, Any]], list[SurfaceEprEntry]],
) -> dict[str, Any]:
    """The config both problems write, in Palace's section order, around the problem's solver block."""
    numerical = common.numerical
    material_rows, energy_rows = domains
    epr_rows, epr_entries = epr
    if epr_rows:
        boundaries["Postprocessing"] = {"Dielectric": epr_rows}
    config: dict[str, Any] = {
        "Problem": {
            "Type": common.problem,
            "Verbose": 3,
            "Output": "results/palace",
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
            "Materials": material_rows,
            "Postprocessing": {"Energy": energy_rows, "Probe": []},
        },
    }
    if _requires_internal_boundary_cracking(epr_entries):
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
        raise ValueError("Eigenmode config requires SGB-authored PEC surfaces.")
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
            "LumpedPort attributes must not also be Surface-EPR loss groups."
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
    domain_volumes: list[tuple[str, Mapping[str, Any]]],
    materials: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[EnergyEntry]]:
    """The ``Domains.Postprocessing.Energy`` rows, one per solution volume, and their index entries."""
    rows = [
        {"Index": index, "Attributes": sorted(info["attributes"])}
        for index, (_, info) in enumerate(domain_volumes, start=1)
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
            zip(domain_volumes, rows, strict=True), start=1
        )
    ]
    return rows, entries


def _material_resolution(
    domain_volumes: list[tuple[str, Mapping[str, Any]]],
    materials: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "solution_volume": name,
            "attributes": sorted(volume["attributes"]),
            "material": _material_identity(volume, materials),
        }
        for name, volume in domain_volumes
    ]


def _lumped_ports(
    groups: Groups, ports: Sequence[LayoutPortBinding]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[PortEntry]]:
    """The ``LumpedPort`` rows, port information and index entries of ``ports``, each bound to its SGB group ``P<index>``.

    A row's circuit keys are the port's ``circuit``, written unchanged; no port is excited.
    """
    rows: list[dict[str, Any]] = []
    information: list[dict[str, Any]] = []
    entries: list[PortEntry] = []
    for port in ports:
        key = f"P{port.index}"
        group = groups.get("port_surfaces", {}).get(key)
        if group is None:
            raise ValueError(f"layout_sheet port {port.name!r} has no SGB group {key}.")
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
                f"layout_sheet port {port.name!r} does not match SGB-authored identity."
            )
        attributes = sorted(int(value) for value in group["attributes"])
        rows.append(
            {
                "Index": port.index,
                "Direction": list(direction),
                "Excitation": False,
                "Attributes": attributes,
                **port.circuit,
            }
        )
        information.append(
            {
                "portnumber": port.index,
                "name": port.name,
                "type": "lumped_sheet",
                "direction": list(direction),
                "physical_name": group["group"],
                "attributes": attributes,
                "source_layer": port.source_layer,
                "target_layer": port.target_layer,
            }
        )
        entries.append(
            PortEntry(
                index=port.index,
                entry_name=key,
                attributes=attributes,
                physical_names=[group["group"]],
                port_name=port.name,
                sheet={
                    field: group[field]
                    for field in (
                        "port_sheet_id",
                        "host_entity_ids",
                        "adjacent_volumes",
                        "host_volumes",
                    )
                },
            )
        )
    if len(groups.get("port_surfaces", {})) != len(ports):
        raise ValueError(
            "SGB produced unconfigured or missing layout-sheet port groups."
        )
    return rows, information, entries


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
) -> list[tuple[str, Mapping[str, Any]]]:
    """Sort solver volumes as deterministic domains."""
    return sorted(
        ((name, dict(info)) for name, info in groups.items()), key=lambda item: item[0]
    )


def _materials_from_solution_volumes(
    domain_volumes: list[tuple[str, Mapping[str, Any]]],
    materials: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build Palace-domain material rows from explicit material records."""
    if not isinstance(materials, Mapping) or not materials:
        raise ValueError(
            "stack must include top-level materials mapping for config generation."
        )

    material_rows: list[dict[str, Any]] = []
    seen_attributes: set[int] = set()
    for name, volume in domain_volumes:
        attrs = sorted(int(item) for item in volume["attributes"])
        material_id = optional_string(volume.get("material"))
        if not material_id:
            raise ValueError(
                f"SGB solution volume {name!r} must define a non-empty material id."
            )
        material = materials.get(material_id)
        if material is None:
            raise ValueError(
                f"SGB solution volume {name!r} references unknown material id {material_id!r}."
            )
        if not isinstance(material, Mapping):
            raise TypeError(f"SGB material record {material_id!r} must be a mapping.")
        if material.get("kind") not in {"vacuum", "dielectric"}:
            raise ValueError(
                f"SGB solution material {material_id!r} must have vacuum or dielectric kind."
            )
        permittivity = material.get("permittivity")
        loss_tangent = material.get("loss_tangent")
        if (
            not isinstance(permittivity, Real)
            or not math.isfinite(float(permittivity))
            or float(permittivity) <= 0.0
        ):
            raise ValueError(
                f"Material {material_id!r} requires finite positive permittivity."
            )
        if (
            not isinstance(loss_tangent, Real)
            or not math.isfinite(float(loss_tangent))
            or float(loss_tangent) < 0.0
        ):
            raise ValueError(
                f"Material {material_id!r} requires finite non-negative loss_tangent."
            )
        if seen_attributes.intersection(attrs):
            raise ValueError(
                f"SGB solution volume {name!r} duplicates physical attributes {attrs}."
            )
        seen_attributes.update(attrs)
        material_rows.append(
            {
                "Attributes": attrs,
                "Permittivity": float(permittivity),
                "LossTan": float(loss_tangent),
            }
        )

    return material_rows


def _material_identity(
    volume: Mapping[str, Any], materials: Mapping[str, Mapping[str, Any]]
) -> dict[str, str]:
    material_id = volume.get("material")
    if not isinstance(material_id, str) or not material_id:
        raise ValueError("solution volume must carry one explicit material id.")
    material = materials.get(material_id)
    if not isinstance(material, Mapping) or material.get("kind") not in {
        "vacuum",
        "dielectric",
    }:
        raise ValueError(
            f"solution material {material_id!r} must carry vacuum or dielectric kind."
        )
    return {"material_id": material_id, "kind": str(material["kind"])}
