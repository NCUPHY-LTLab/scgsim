"""Build Palace configuration payloads from the solver attributes of prepared SGB groups."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any, Literal

from .boundaries import build_electrostatic_boundaries
from .config_models import (
    MODEL_L0_M,
    ConfigBuildResult,
    EigenmodeConfigBuildResult,
    LayoutPortBinding,
    TerminalBinding,
    optional_string,
    positive_finite,
)
from .numerical import (
    build_output_formats,
    build_refinement,
    configure_numerical_controls,
    linear_solver,
)
from .surface_epr import build_surface_epr_postprocessing

__all__ = [
    "MODEL_L0_M",
    "ConfigBuildResult",
    "EigenmodeConfigBuildResult",
    "LayoutPortBinding",
    "TerminalBinding",
    "build_eigenmode_config",
    "build_electrostatic_config",
    "configure_numerical_controls",
]


def build_electrostatic_config(  # noqa: PLR0913 - config builder preserves its named input API.
    *,
    groups: Mapping[str, Mapping[str, Mapping[str, Any]]],
    terminals: Sequence[TerminalBinding],
    ground_net_ids: Sequence[str] = (),
    materials: Mapping[str, Mapping[str, Any]],
    save_fields: int,
    unassigned_conductor_policy: Literal["ground", "error"],
    exterior_boundary_policy: Literal["none", "ground"],
    mesh_path: Path,
    surface_epr_specs: Mapping[str, Mapping[str, Any]],
    numerical: Mapping[str, Any],
) -> ConfigBuildResult:
    """Build a minimal Palace electrostatic config payload."""
    if (
        not isinstance(save_fields, int)
        or isinstance(save_fields, bool)
        or save_fields < 0
    ):
        raise ValueError("save_fields must be a non-bool int >= 0.")
    if exterior_boundary_policy == "ground" and unassigned_conductor_policy != "error":
        raise ValueError(
            "exterior ground requires unassigned_conductor_policy='error'."
        )

    boundary_map, terminal_index_map, ground_boundary_resolution = (
        build_electrostatic_boundaries(
            groups=groups,
            terminals=terminals,
            ground_net_ids=ground_net_ids,
            unassigned_conductor_policy=unassigned_conductor_policy,
            exterior_boundary_policy=exterior_boundary_policy,
        )
    )
    domain_volumes = _domains_from_solution_volumes(groups.get("volumes", {}))
    materials_payload = _materials_from_solution_volumes(
        domain_volumes=domain_volumes,
        materials=materials,
    )
    epr_rows, epr_index_map = build_surface_epr_postprocessing(
        groups, surface_epr_specs, model_l0_m=MODEL_L0_M
    )
    energy_rows, energy_index_map = _domain_energy_rows(domain_volumes, materials)

    linear = linear_solver(numerical, problem="Electrostatic")
    problem_block = build_output_formats(numerical)
    problem = {
        "Type": "Electrostatic",
        "Verbose": 3,
        "Output": "results/palace",
    }
    problem.update(problem_block)
    config: dict[str, Any] = {
        "Problem": problem,
        "Model": {
            "Mesh": str(mesh_path.name),
            "L0": MODEL_L0_M,
            "Refinement": build_refinement(numerical),
        },
        "Solver": {
            "Linear": linear,
            "Order": int(numerical["order"]),
            "Device": numerical["device"],
            "Electrostatic": {"Save": save_fields},
        },
        "Boundaries": boundary_map,
        "Domains": {
            "Materials": materials_payload,
            "Postprocessing": {"Energy": energy_rows, "Probe": []},
        },
    }
    if _requires_internal_boundary_cracking(groups):
        config["Model"]["CrackInternalBoundaryElements"] = True
    if epr_rows:
        config["Boundaries"]["Postprocessing"] = {"Dielectric": epr_rows}
    return ConfigBuildResult(
        config=config,
        domain_names=[name for name, _ in domain_volumes],
        terminal_index_map=terminal_index_map,
        surface_epr_index_map=epr_index_map,
        domain_energy_index_map=energy_index_map,
        material_resolution=_material_resolution(domain_volumes, materials),
        ground_boundary_resolution=ground_boundary_resolution,
    )


def build_eigenmode_config(  # noqa: PLR0913 - config builder preserves its named input API.
    *,
    groups: Mapping[str, Mapping[str, Mapping[str, Any]]],
    ports: Sequence[LayoutPortBinding],
    materials: Mapping[str, Mapping[str, Any]],
    surface_epr_specs: Mapping[str, Mapping[str, Any]],
    numerical: Mapping[str, Any],
    num_modes: int,
    target_hz: float | None,
    eigenmode_tolerance: float,
    save_fields: int,
    mesh_path: Path,
) -> EigenmodeConfigBuildResult:
    """Build one non-Floquet Eigenmode config from SGB-authored groups."""
    if numerical.get("amr_max_passes") is None:
        raise ValueError(
            "Eigenmode requires explicit amr_max_passes before config generation."
        )
    if numerical["amr_max_passes"] > 0 and (
        numerical.get("amr_tolerance") is None
        or numerical.get("amr_update_fraction") is None
    ):
        raise ValueError(
            "Enabled Eigenmode AMR requires explicit amr_tolerance and amr_update_fraction."
        )
    if not isinstance(num_modes, int) or isinstance(num_modes, bool) or num_modes < 1:
        raise ValueError("num_modes must be a positive integer.")
    if not positive_finite(eigenmode_tolerance):
        raise ValueError("Eigenmode tolerance must be finite and positive.")
    if target_hz is not None and not positive_finite(target_hz):
        raise ValueError("Eigenmode target must be finite positive Hz.")
    if (
        not isinstance(save_fields, int)
        or isinstance(save_fields, bool)
        or save_fields < 0
    ):
        raise ValueError("save_fields must be a non-negative integer.")

    domain_volumes = _domains_from_solution_volumes(groups.get("volumes", {}))
    material_rows = _materials_from_solution_volumes(
        domain_volumes=domain_volumes, materials=materials
    )
    energy_rows, energy_index = _domain_energy_rows(domain_volumes, materials)
    epr_rows, epr_index = build_surface_epr_postprocessing(
        groups, surface_epr_specs, model_l0_m=MODEL_L0_M
    )
    lumped_ports, port_info, port_index = _eigenmode_lumped_ports(groups, ports)
    epr_attributes = {
        int(value) for row in epr_rows for value in row.get("Attributes", ())
    }
    port_attributes = {
        int(value) for row in lumped_ports for value in row.get("Attributes", ())
    }
    if epr_attributes.intersection(port_attributes):
        raise ValueError(
            "LumpedPort attributes must not also be Surface-EPR loss groups."
        )

    pec_attributes = sorted(
        {
            int(value)
            for info in groups.get("pec_surfaces", {}).values()
            for value in info["attributes"]
        }
    )
    if not pec_attributes:
        raise ValueError("Eigenmode config requires SGB-authored PEC surfaces.")
    eigenmode: dict[str, Any] = {
        "N": num_modes,
        "Tol": float(eigenmode_tolerance),
        "Save": save_fields,
    }
    if target_hz is not None:
        eigenmode["Target"] = float(target_hz) / 1e9
    boundaries: dict[str, Any] = {"PEC": {"Attributes": pec_attributes}}
    if lumped_ports:
        boundaries["LumpedPort"] = lumped_ports
    if epr_rows:
        boundaries["Postprocessing"] = {"Dielectric": epr_rows}
    problem_block = build_output_formats(numerical)
    problem = {
        "Type": "Eigenmode",
        "Verbose": 3,
        "Output": "results/palace",
    }
    problem.update(problem_block)
    config = {
        "Problem": problem,
        "Model": {
            "Mesh": mesh_path.name,
            "L0": MODEL_L0_M,
            "Refinement": build_refinement(numerical),
        },
        "Solver": {
            "Linear": linear_solver(numerical),
            "Order": int(numerical["order"]),
            "Device": numerical["device"],
            "Eigenmode": eigenmode,
        },
        "Boundaries": boundaries,
        "Domains": {
            "Materials": material_rows,
            "Postprocessing": {"Energy": energy_rows, "Probe": []},
        },
    }
    if _requires_internal_boundary_cracking(groups):
        config["Model"]["CrackInternalBoundaryElements"] = True
    return EigenmodeConfigBuildResult(
        config=config,
        port_information=port_info,
        index_entries=[*port_index, *energy_index, *epr_index],
        material_resolution=_material_resolution(domain_volumes, materials),
    )


def _requires_internal_boundary_cracking(
    groups: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> bool:
    """Whether a Surface-EPR row integrates one two-sided face from both sides.

    A row has a ``shared_face_group`` exactly when it belongs to a Route A conductor sheet, an internal face
    whose MA and MS rows integrate the same face once per side.  Palace keeps the two sides apart only when it
    cracks internal boundary elements (``Model.CrackInternalBoundaryElements``, which then duplicates the
    nodes of the sheet), so the key is written exactly then and left at Palace's default otherwise.
    """
    return any(
        row["shared_face_group"] is not None
        for info in groups.get("boundary_surfaces", {}).values()
        for row in info["rows"].values()
    )


def _domain_energy_rows(
    domain_volumes: list[tuple[str, Mapping[str, Any]]],
    materials: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = [
        {"Index": index, "Attributes": sorted(info["attributes"])}
        for index, (_, info) in enumerate(domain_volumes, start=1)
    ]
    index_rows = [
        {
            "section": "Domains.Postprocessing.Energy",
            "index": index,
            "entry_name": name,
            "role": "domain_energy",
            "physical_name": name,
            "physical_names": [name],
            "attributes": row["Attributes"],
            "metadata": {
                "material": _material_identity(info, materials),
                "entities": list(info["entities"]),
            },
        }
        for index, ((name, info), row) in enumerate(
            zip(domain_volumes, rows, strict=True), start=1
        )
    ]
    return rows, index_rows


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


def _eigenmode_lumped_ports(
    groups: Mapping[str, Mapping[str, Mapping[str, Any]]],
    ports: Sequence[LayoutPortBinding],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    information: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
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
                "L": port.inductance,
            }
        )
        info = {
            "portnumber": port.index,
            "name": port.name,
            "type": "lumped_sheet",
            "direction": list(direction),
            "physical_name": group["group"],
            "attributes": attributes,
            "source_layer": port.source_layer,
            "target_layer": port.target_layer,
        }
        information.append(info)
        index_rows.append(
            {
                "section": "Boundaries.LumpedPort",
                "index": port.index,
                "entry_name": key,
                "role": "port_surface",
                "attributes": attributes,
                "physical_names": [group["group"]],
                "port_name": port.name,
                "metadata": {
                    field: group[field]
                    for field in (
                        "port_sheet_id",
                        "host_entity_ids",
                        "adjacent_volumes",
                        "host_volumes",
                    )
                },
            }
        )
    if len(groups.get("port_surfaces", {})) != len(ports):
        raise ValueError(
            "SGB produced unconfigured or missing layout-sheet port groups."
        )
    return rows, information, index_rows


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
    *,
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
