"""Build the Palace ``config.json`` from SGB (geometry builder) physical groups.

Palace reads boundaries and domains by integer attribute. This module turns the named
physical groups SGB wrote into mesh attributes, and records in an index map which
config entry stands for which named group, so a result column can be traced back.
Two builders: ``build_electrostatic_config`` (terminal capacitance) and
``build_eigenmode_config`` (lumped-port modes).

Not here: meshing (``palace/mesh``), the numerical controls (``numerical.py``), the
Terminal and Ground boundaries (``electrostatic_boundaries.py``), the domain and
material rows (``solution_volumes.py``), the Surface EPR rows (``surface_epr.py``), and
checking the finished config against the index map (``palace/results``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .electrostatic_boundaries import (
    GroundBoundaryResolution,
    TerminalBinding,
    TerminalIndexEntry,
    _electrostatic_boundaries,
)
from .numerical import _positive_finite
from .solution_volumes import (
    _domain_energy_rows,
    _domains_from_solution_volumes,
    _material_resolution,
    _materials_from_solution_volumes,
    _physical_group_values,
)
from .surface_epr import build_surface_epr_postprocessing

# Model.L0 in metres: the length unit of mesh coordinates and of lengths in the config
# (1e-6 is one micrometre).
_MODEL_L0_M = 1e-6


@dataclass(frozen=True)
class ConfigBuildResult:
    """An Electrostatic config and the records that explain its indices."""

    config: dict[str, Any]
    domain_names: list[str]
    terminal_index_map: list[TerminalIndexEntry]
    surface_epr_index_map: list[dict[str, Any]]
    domain_energy_index_map: list[dict[str, Any]]
    material_resolution: list[dict[str, Any]]
    ground_boundary_resolution: GroundBoundaryResolution


@dataclass(frozen=True)
class LayoutPortBinding:
    """One authored GDSFactory junction port resolved for SGB lowering."""

    index: int
    name: str
    target_layer: str
    source_layer: str
    direction: tuple[float, float, float]
    inductance: float


@dataclass(frozen=True)
class EigenmodeConfigBuildResult:
    """An Eigenmode config and the records that explain its indices."""

    config: dict[str, Any]
    port_information: list[dict[str, Any]]
    index_entries: list[dict[str, Any]]
    material_resolution: list[dict[str, Any]]


def build_electrostatic_config(  # noqa: PLR0913 - public keyword-only API, one argument per solver choice
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
        _electrostatic_boundaries(
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
        groups, surface_epr_specs, model_l0_m=_MODEL_L0_M
    )
    energy_rows, energy_index_map = _domain_energy_rows(domain_volumes, materials)

    config: dict[str, Any] = {
        "Problem": _problem_block("Electrostatic", numerical),
        "Model": _model_block(mesh_path, numerical),
        "Solver": {
            "Linear": _linear_solver(numerical),
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


def build_eigenmode_config(  # noqa: PLR0913 - public keyword-only API, one argument per solver choice
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
    _check_eigenmode_controls(num_modes, target_hz, eigenmode_tolerance, save_fields)

    domain_volumes = _domains_from_solution_volumes(groups.get("volumes", {}))
    material_rows = _materials_from_solution_volumes(
        domain_volumes=domain_volumes, materials=materials
    )
    energy_rows, energy_index = _domain_energy_rows(domain_volumes, materials)
    epr_rows, epr_index = build_surface_epr_postprocessing(
        groups, surface_epr_specs, model_l0_m=_MODEL_L0_M
    )
    lumped_ports, port_info, port_index = _eigenmode_lumped_ports(groups, ports)
    _check_ports_are_not_loss_groups(lumped_ports, epr_rows)

    pec_attributes = _pec_attributes(groups)
    eigenmode: dict[str, Any] = {
        "N": num_modes,
        "Tol": float(eigenmode_tolerance),
        "Save": save_fields,
    }
    if target_hz is not None:
        # Palace takes the target in GHz.
        eigenmode["Target"] = float(target_hz) / 1e9
    boundaries: dict[str, Any] = {"PEC": {"Attributes": pec_attributes}}
    if lumped_ports:
        boundaries["LumpedPort"] = lumped_ports
    if epr_rows:
        boundaries["Postprocessing"] = {"Dielectric": epr_rows}
    config = {
        "Problem": _problem_block("Eigenmode", numerical),
        "Model": _model_block(mesh_path, numerical),
        "Solver": {
            "Linear": _linear_solver(numerical),
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


def _check_eigenmode_controls(
    num_modes: int,
    target_hz: float | None,
    eigenmode_tolerance: float,
    save_fields: int,
) -> None:
    if not isinstance(num_modes, int) or isinstance(num_modes, bool) or num_modes < 1:
        raise ValueError("num_modes must be a positive integer.")
    if not _positive_finite(eigenmode_tolerance):
        raise ValueError("Eigenmode tolerance must be finite and positive.")
    if target_hz is not None and not _positive_finite(target_hz):
        raise ValueError("Eigenmode target must be finite positive Hz.")
    if (
        not isinstance(save_fields, int)
        or isinstance(save_fields, bool)
        or save_fields < 0
    ):
        raise ValueError("save_fields must be a non-negative integer.")


def _check_ports_are_not_loss_groups(
    lumped_ports: Sequence[Mapping[str, Any]], epr_rows: Sequence[Mapping[str, Any]]
) -> None:
    """A surface cannot be both a driven port and a lossy dielectric film."""
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


def _pec_attributes(
    groups: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> list[int]:
    """Mesh attributes of the perfect-electric-conductor (PEC) surfaces."""
    pec_attributes = sorted(
        {
            int(value)
            for info in groups.get("pec_surfaces", {}).values()
            for value in _physical_group_values(info.get("phys_group"))
        }
    )
    if not pec_attributes:
        raise ValueError("Eigenmode config requires SGB-authored PEC surfaces.")
    return pec_attributes


def _problem_block(problem_type: str, numerical: Mapping[str, Any]) -> dict[str, Any]:
    """``Problem``: the solver type and where Palace writes its tables."""
    return {
        "Type": problem_type,
        "Verbose": 3,
        "Output": "results/palace",
        **_build_output_formats(numerical),
    }


def _model_block(mesh_path: Path, numerical: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "Mesh": str(mesh_path.name),
        "L0": _MODEL_L0_M,
        "Refinement": _build_refinement(numerical),
    }


def _requires_internal_boundary_cracking(
    groups: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> bool:
    """Bind two-sided same-kind ledgers to Palace's cracked-interface semantics."""
    for info in groups.get("boundary_surfaces", {}).values():
        if not isinstance(info, Mapping):
            continue
        provenance = info.get("source_provenance")
        if not isinstance(provenance, Mapping):
            continue
        pending: list[Mapping[str, Any]] = [provenance]
        while pending:
            current = pending.pop()
            aggregation = current.get("native_aggregation")
            if (
                isinstance(aggregation, Mapping)
                and aggregation.get("policy") == "shared_native_surface_once"
                and aggregation.get("contribution_count") == 2
                and aggregation.get("native_row_count") == 1
            ):
                return True
            sources = current.get("sources", ())
            if isinstance(sources, (list, tuple)):
                pending.extend(
                    source for source in sources if isinstance(source, Mapping)
                )
    return False


# ----------------------------------------------- Electrostatic boundary synthesis


def _linear_solver(numerical: Mapping[str, Any]) -> dict[str, Any]:
    linear = {
        "Type": numerical["solver_type"],
        "KSPType": "GMRES",
        "Tol": float(numerical["tolerance"]),
        "MaxIts": int(numerical["max_iterations"]),
    }
    if numerical["solver_type"] == "MUMPS":
        linear.update(
            {
                "MaxIts": 1,
                "MGMaxLevels": 1,
                "PCMatReal": False,
                "ComplexCoarseSolve": True,
            }
        )
    if (
        numerical["solver_type"] == "Default"
        and numerical["preconditioner"] != "Default"
    ):
        linear["Type"] = numerical["preconditioner"]

    if numerical.get("estimator_mg") is not None:
        linear["EstimatorMG"] = bool(numerical["estimator_mg"])
    return linear


def _build_refinement(numerical: Mapping[str, Any]) -> dict[str, Any]:
    # Palace defaults Model.Refinement.Nonconformal to true if the key is omitted.
    # SCGSim always writes the key. The default is false (conformal tet AMR);
    # notebooks may set amr_nonconformal=True to opt into NC AMR.
    refinement = {
        "UniformLevels": 0,
        "Tol": float(numerical["amr_tolerance"]),
        "MaxIts": int(numerical["amr_max_passes"]),
        "Nonconformal": bool(numerical.get("amr_nonconformal", False)),
    }
    if numerical.get("save_adapt_iterations") is not None:
        refinement["SaveAdaptIterations"] = bool(numerical["save_adapt_iterations"])
    if numerical.get("save_adapt_mesh") is not None:
        refinement["SaveAdaptMesh"] = bool(numerical["save_adapt_mesh"])
    if numerical.get("amr_update_fraction") is not None:
        refinement["UpdateFraction"] = float(numerical["amr_update_fraction"])
    return refinement


def _build_output_formats(numerical: Mapping[str, Any]) -> dict[str, Any]:
    output_formats = {
        "Paraview": numerical.get("output_paraview"),
        "GridFunction": numerical.get("output_grid_function"),
    }
    outputs = {
        key: bool(value) for key, value in output_formats.items() if value is not None
    }
    if not outputs:
        return {}
    return {"OutputFormats": outputs}


def _eigenmode_lumped_ports(
    groups: Mapping[str, Mapping[str, Mapping[str, Any]]],
    ports: Sequence[LayoutPortBinding],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return ``(Palace LumpedPort rows, port information, index-map rows)``."""
    rows: list[dict[str, Any]] = []
    information: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    for port in ports:
        key = f"P{port.index}"
        group = _port_group(groups, port, key)
        direction = _direction_tuple(group.get("direction"), port.name)
        _check_port_identity(port, group, direction)
        attributes = sorted(_physical_group_values(group.get("phys_group")))
        rows.append(
            {
                "Index": port.index,
                "Direction": list(direction),
                "Excitation": False,
                "Attributes": attributes,
                "L": port.inductance,
            }
        )
        information.append(
            {
                "portnumber": port.index,
                "name": port.name,
                "type": "lumped_sheet",
                "direction": list(direction),
                "physical_name": group.get("physical_name"),
                "attributes": attributes,
                "source_layer": port.source_layer,
                "target_layer": port.target_layer,
            }
        )
        index_rows.append(_port_index_row(port, key, group, attributes))
    if len(groups.get("port_surfaces", {})) != len(ports):
        raise ValueError(
            "SGB produced unconfigured or missing layout-sheet port groups."
        )
    return rows, information, index_rows


def _port_group(
    groups: Mapping[str, Mapping[str, Mapping[str, Any]]],
    port: LayoutPortBinding,
    key: str,
) -> Mapping[str, Any]:
    group = groups.get("port_surfaces", {}).get(key)
    if group is None:
        raise ValueError(f"layout_sheet port {port.name!r} has no SGB group {key}.")
    if not isinstance(group, Mapping):
        raise TypeError(f"SGB layout_sheet group {key} must be a mapping.")
    if not isinstance(group.get("physical_attribute"), Mapping):
        raise TypeError(
            f"SGB layout_sheet port {port.name!r} lacks physical_attribute."
        )
    return group


def _check_port_identity(
    port: LayoutPortBinding,
    group: Mapping[str, Any],
    direction: tuple[float, float, float],
) -> None:
    """The SGB group must describe the same port the notebook authored."""
    attribute = group["physical_attribute"]
    if (
        attribute.get("port_index") != port.index
        or attribute.get("port_name") != port.name
        or attribute.get("source_layer") != port.source_layer
        or attribute.get("target_layer") != port.target_layer
        or not all(
            math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
            for a, b in zip(direction, port.direction, strict=True)
        )
    ):
        raise ValueError(
            f"layout_sheet port {port.name!r} does not match SGB-authored identity."
        )


def _port_index_row(
    port: LayoutPortBinding,
    key: str,
    group: Mapping[str, Any],
    attributes: list[int],
) -> dict[str, Any]:
    return {
        "section": "Boundaries.LumpedPort",
        "index": port.index,
        "entry_name": key,
        "role": "port_surface",
        "attributes": attributes,
        "physical_names": [group.get("physical_name")],
        "port_name": port.name,
        "metadata": {
            field: group[field]
            for field in (
                "surface_id",
                "owner_semantic_ids",
                "adjacent_solution_volume_ids",
                "embedded_volume_id",
                "interface_of",
                "source_provenance",
                "physical_attribute",
                "representation",
                "route",
            )
            if field in group
        },
    }


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
    return direction
