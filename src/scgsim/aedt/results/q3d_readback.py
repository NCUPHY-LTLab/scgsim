"""Strict offline validation for Q3D native bodies, region grounds, setup, and matrices."""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from scgsim.aedt.io import read_json
from scgsim.aedt.results.convergence.q3d import read_q3d_convergence
from scgsim.aedt.results.matrices import parse_matrix_export
from scgsim.aedt.results.readback_common import positive_unique_ids
from scgsim.aedt.results.receipt_validation import contained
from scgsim.aedt.specs.common import MatrixRunControl
from scgsim.aedt.specs.q3d import Q3dSpec
from scgsim.aedt.terminal_chain import q3d_terminal_segments

_Q3D_REGION_DIRECTIONS = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")

_Q3D_REGION_SHEET_NAMES = {
    "+X": ("SCGSimRegionGroundPX", "SCGSimRegionGroundPXThinConductor"),
    "-X": ("SCGSimRegionGroundNX", "SCGSimRegionGroundNXThinConductor"),
    "+Y": ("SCGSimRegionGroundPY", "SCGSimRegionGroundPYThinConductor"),
    "-Y": ("SCGSimRegionGroundNY", "SCGSimRegionGroundNYThinConductor"),
    "+Z": ("SCGSimRegionGroundPZ", "SCGSimRegionGroundPZThinConductor"),
    "-Z": ("SCGSimRegionGroundNZ", "SCGSimRegionGroundNZThinConductor"),
}


def validate_q3d_model_readback(
    root: Path, receipt: dict[str, Any], spec: Q3dSpec, *, prepared: bool = False
) -> None:
    """Bind model assignments separately from setup and matrix evidence."""
    if ("ports" in receipt or not prepared) and receipt.get("ports") != []:
        raise RuntimeError("Q3D receipt must not contain HFSS ports")
    nets = receipt.get("nets")
    if not isinstance(nets, list) or len(nets) != len(spec.nets):
        raise RuntimeError("Q3D receipt has invalid net evidence")
    for record, expected in zip(nets, spec.nets, strict=True):
        _validate_q3d_net_terminals(record, expected, spec)
        if expected.net_type == "Signal" and expected.source_object is not None:
            for kind, object_name, side in (
                ("source", expected.source_object, expected.source_side),
                ("sink", expected.sink_object, expected.sink_side),
            ):
                terminal = record.get(kind)
                if (
                    not isinstance(terminal, dict)
                    or terminal.get("name") != f"{expected.name}{kind.title()}"
                    or terminal.get("object_name") != object_name
                    or terminal.get("side") != side
                ):
                    raise RuntimeError("Q3D native terminal evidence is invalid")
                effective = (
                    {}
                    if spec.modeling is None
                    else {body["body_id"]: body for body in spec.effective_bodies}
                )
                if effective and effective[object_name]["representation"] == "sheet":
                    body = next(
                        body for body in spec.bodies if body.body_id == object_name
                    )
                    indices = q3d_terminal_segments(body, cast(str, side))
                    edges = terminal.get("native_edge_ids")
                    if (
                        not isinstance(edges, list)
                        or not positive_unique_ids(edges, len(indices))
                        or terminal.get("source_edges")
                        != [
                            {"source_segment_index": index, "native_edge_id": edge}
                            for index, edge in zip(indices, edges, strict=True)
                        ]
                        or terminal.get("terminal_type") != "ConstantVoltage"
                        or "native_face_ids" in terminal
                    ):
                        raise RuntimeError(
                            "Q3D Sheet terminal evidence differs from source exterior"
                        )
                else:
                    faces = terminal.get("native_face_ids")
                    if not isinstance(faces, list) or not positive_unique_ids(faces, 1):
                        raise RuntimeError(
                            "Q3D Solid terminal face evidence is invalid"
                        )
        elif "source" in record or "sink" in record:
            raise RuntimeError("Q3D net receipt contains undeclared terminals")
    net_object_ids = [
        object_id for record in nets for object_id in record["native_object_ids"]
    ]
    if len(set(net_object_ids)) != len(net_object_ids):
        raise RuntimeError("Q3D native object IDs must be unique across nets")
    _validate_q3d_body_bindings(receipt, spec, nets, root)
    _validate_q3d_region_ground(receipt, spec, nets)


def _validate_q3d_net_terminals(
    record: dict[str, Any], expected: Any, spec: Q3dSpec
) -> None:
    """Verify declared Q3D terminal identities and their native face or edge IDs."""
    if (
        not isinstance(record, dict)
        or record.get("name") != expected.name
        or record.get("net_type") != expected.net_type
        or record.get("object_names") != list(expected.object_names)
        or not isinstance(record.get("native_object_ids"), list)
        or not positive_unique_ids(
            record["native_object_ids"], len(expected.object_names)
        )
    ):
        raise RuntimeError("Q3D native net evidence does not match the spec")


def validate_q3d_readback(root: Path, receipt: dict[str, Any], spec: Q3dSpec) -> None:
    """Verify Q3D native model, setup, convergence, and exported matrices."""
    run_control = cast(MatrixRunControl, spec.run_control)
    validate_q3d_model_readback(root, receipt, spec)
    validate_q3d_setup(receipt, spec)
    if receipt.get("convergence") != read_q3d_convergence(root, spec):
        raise RuntimeError("Q3D native convergence evidence is invalid")
    readback = receipt.get("result_readback")
    matrices = readback.get("matrices") if isinstance(readback, dict) else None
    if spec.solve_ac_rl:
        if (
            not isinstance(matrices, dict)
            or matrices.get("frequency_ghz") != run_control.frequency_ghz
            or set(matrices.get("native", {})) != {"c", "ac_rl"}
            or not isinstance(matrices.get("normalized_rows"), int)
            or matrices["normalized_rows"] <= 0
        ):
            raise RuntimeError("Q3D matrix readback is invalid")
        _validate_normalized_matrices(
            contained(root, "results/q3d/matrices.csv"),
            matrices["normalized_rows"],
            {"C", "AC RL"},
        )
        _read_q3d_original_cg(root, spec)
        return
    rows, summary = _read_q3d_original_cg(root, spec)
    expected_matrices = {
        "path": "results/q3d/c_matrix.csv",
        "frequency_ghz": run_control.frequency_ghz,
        "native": summary,
        "primary_rows": len(rows),
    }
    if matrices != expected_matrices:
        raise RuntimeError("Q3D capacitance-only matrix readback is invalid")


def validate_q3d_setup(receipt: dict[str, Any], spec: Q3dSpec) -> None:
    """Bind the native Q3D setup readback to the requested setup controls."""
    run_control = cast(MatrixRunControl, spec.run_control)
    expected_setup = {
        "name": run_control.setup_name,
        "native": {
            "adaptive_frequency": f"{run_control.frequency_ghz:g}GHz",
            "capacitance_maximum_passes": run_control.maximum_passes,
            "capacitance_minimum_passes": run_control.minimum_passes,
            "capacitance_minimum_converged_passes": (
                run_control.minimum_converged_passes
            ),
            "capacitance_convergence_percent": run_control.convergence_percent,
            "capacitance_percent_refinement": run_control.percent_refinement,
            "dc_enabled": False,
        },
    }
    if spec.solve_ac_rl:
        expected_setup["native"].update(
            {
                "ac_rl_maximum_passes": run_control.maximum_passes,
                "ac_rl_minimum_passes": run_control.minimum_passes,
                "ac_rl_minimum_converged_passes": (
                    run_control.minimum_converged_passes
                ),
                "ac_rl_convergence_percent": run_control.convergence_percent,
                "ac_rl_percent_refinement": run_control.percent_refinement,
            }
        )
    if receipt.get("setup") != expected_setup:
        raise RuntimeError("Q3D native setup evidence is invalid")


def _read_q3d_original_cg(
    root: Path, spec: Q3dSpec
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run_control = cast(MatrixRunControl, spec.run_control)
    options = {
        "expected_label_set": {
            net.name for net in spec.nets if net.net_type == "Signal"
        }
    }
    return parse_matrix_export(
        contained(root, "results/q3d/c_matrix.csv"),
        "Q3D",
        "C",
        run_control.frequency_ghz,
        {"Capacitance Matrix": "C", "Conductance Matrix": "G"},
        expected_label_set=options["expected_label_set"],
    )


@dataclass
class Q3dBodyBindingContext:
    """Source and native inventories shared across Q3D body checks."""

    spec: Q3dSpec
    raw_bodies: dict[str, Any]
    effective_bodies: dict[str, Any]
    net_object_ids: dict[str, int]
    seen_native_ids: set[int]


def _validate_q3d_body_bindings(
    receipt: dict[str, Any], spec: Q3dSpec, nets: list[dict[str, Any]], root: Path
) -> None:
    """Bind every created body to its declared geometry and final native identity."""
    materials = receipt.get("materials")
    if not isinstance(materials, list) or len(materials) != len(spec.bodies):
        raise RuntimeError("Q3D body material readback is invalid")
    net_object_ids = {
        name: native_id
        for net_record, net_spec in zip(nets, spec.nets, strict=True)
        for name, native_id in zip(
            net_spec.object_names, net_record["native_object_ids"], strict=True
        )
    }
    records = {
        record.get("object_name"): record
        for record in materials
        if isinstance(record, dict)
    }
    if len(records) != len(materials) or set(records) != {
        body.body_id for body in spec.bodies
    }:
        raise RuntimeError("Q3D native body inventory differs from the spec")
    historical = spec.modeling is None
    raw_bodies = {
        body["body_id"]: body for body in read_json(root / "aedt_spec.json")["bodies"]
    }
    effective_bodies = (
        {} if historical else {body["body_id"]: body for body in spec.effective_bodies}
    )
    context = Q3dBodyBindingContext(
        spec, raw_bodies, effective_bodies, net_object_ids, set()
    )
    for body in spec.bodies:
        record = records[body.body_id]
        material = spec.materials[body.material_id]
        _validate_q3d_body_material(context, body, material, record)
        evidence = record.get("body_binding")
        if not isinstance(evidence, dict):
            raise RuntimeError("Q3D native body binding is absent")
        native_id, faces = _validate_q3d_body_identity(context, body, record, evidence)
        if not historical:
            _validate_q3d_effective_body(context, body, record, evidence)
        context.seen_native_ids.add(native_id)
        _validate_q3d_body_incidence(evidence, faces)


def _validate_q3d_body_material(
    context: Q3dBodyBindingContext, body: Any, material: Any, record: dict[str, Any]
) -> None:
    observed = record.get("observed")
    historical = context.spec.modeling is None
    sheet = (
        not historical
        and context.effective_bodies[body.body_id]["representation"] == "sheet"
    )
    observed_key = "native_boundary_material_name" if sheet else "native_material_name"
    expected_native_material = (
        "pec" if material.is_superconducting else material.library_name
    )
    if (
        record.get("material_id") != body.material_id
        or record.get("role") != body.physical_role
        or record.get("kind") != material.kind
        or record.get("is_superconducting") is not material.is_superconducting
        or record.get("requested_library_name") != material.library_name
        or not isinstance(observed, dict)
        or not isinstance(observed.get(observed_key), str)
        or observed[observed_key].casefold() != expected_native_material.casefold()
    ):
        raise RuntimeError("Q3D body material binding differs from the spec")
    if sheet and (
        observed.get("native_material_property") != "not_applicable_sheet"
        or "native_material_name" in observed
    ):
        raise RuntimeError("Q3D Sheet evidence must identify boundary-only material")


def _validate_q3d_body_identity(
    context: Q3dBodyBindingContext,
    body: Any,
    record: dict[str, Any],
    evidence: dict[str, Any],
) -> tuple[int, list[int]]:
    historical = context.spec.modeling is None
    effective = context.effective_bodies.get(body.body_id)
    native_id = evidence.get("native_object_id")
    faces = evidence.get("native_face_ids")
    if (
        evidence.get("body")
        != (context.raw_bodies[body.body_id] if historical else body.to_payload())
        or evidence.get("object_name") != body.body_id
        or evidence.get("native_object_type")
        != (
            "Sheet" if effective and effective["representation"] == "sheet" else "Solid"
        )
        or not positive_unique_ids([native_id], 1)
        or native_id in context.seen_native_ids
        or not isinstance(faces, list)
        or not faces
        or not positive_unique_ids(faces, len(faces))
        or (
            body.net_id is not None
            and context.net_object_ids.get(body.body_id) != native_id
        )
    ):
        raise RuntimeError(
            "Q3D body binding differs from the native identity/source declaration"
        )
    return cast(int, native_id), cast(list[int], faces)


def _validate_q3d_effective_body(
    context: Q3dBodyBindingContext,
    body: Any,
    record: dict[str, Any],
    evidence: dict[str, Any],
) -> None:
    effective = context.effective_bodies[body.body_id]
    native_id = evidence["native_object_id"]
    if evidence.get("effective_geometry") != effective:
        raise RuntimeError("Q3D effective geometry binding differs from the spec")
    thin = evidence.get("thin_conductor")
    if effective["representation"] == "sheet":
        _validate_q3d_thin_conductor(context, body, record, thin, native_id)
    elif thin is not None:
        raise RuntimeError("Q3D Solid body contains Sheet boundary evidence")


def _validate_q3d_thin_conductor(
    context: Q3dBodyBindingContext,
    body: Any,
    record: dict[str, Any],
    thin: Any,
    native_id: int,
) -> None:
    effective = context.effective_bodies[body.body_id]
    body_id = body.body_id
    if not isinstance(thin, dict):
        raise RuntimeError("Q3D ThinConductor evidence is not recorded")
    raw_material = thin.get("material_raw")
    if (
        isinstance(raw_material, list)
        and len(raw_material) == 2
        and raw_material[0] == "Material:="
    ):
        raw_material = raw_material[1]
    if not isinstance(raw_material, str):
        raise RuntimeError("Q3D ThinConductor Material has unknown native shape")
    material_name = raw_material.strip('"')
    raw_thickness = thin.get("thickness")
    length = re.fullmatch(
        r"\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)([A-Za-z]+)",
        raw_thickness if isinstance(raw_thickness, str) else "",
    )
    scales = {
        "fm": 1e-15,
        "pm": 1e-12,
        "nm": 1e-9,
        "um": 1e-6,
        "mm": 1e-3,
        "cm": 1e-2,
        "dm": 1e-1,
        "meter": 1.0,
        "km": 1e3,
        "uin": 0.0254 * 1e-6,
        "mil": 0.0254 * 1e-3,
        "in": 0.0254,
        "ft": 0.0254 * 12,
        "yd": 0.0254 * 36,
        "mile": 0.0254 * 63360,
    }
    if length is None or length[2].lower() not in scales:
        raise RuntimeError("Q3D ThinConductor Thickness is not a native length")
    thickness_um = float(length[1]) * scales[length[2].lower()] / 1e-6
    raw_ids = thin.get("native_assignment_ids_raw")
    if not isinstance(raw_ids, list) or any(
        not (
            type(value) is int
            or (isinstance(value, str) and re.fullmatch(r"[0-9]+", value))
        )
        for value in raw_ids
    ):
        raise RuntimeError("Q3D ThinConductor assignment readback is malformed")
    assigned_ids = [int(value) for value in raw_ids]
    expected_thin = {
        "name": f"{body_id}ThinConductor",
        "bound_type": "ThinConductor",
        "material": material_name,
        "material_raw": thin.get("material_raw"),
        "thickness": raw_thickness,
        "thickness_um": thickness_um,
        "native_assignment_ids_raw": raw_ids,
        "native_object_ids": assigned_ids,
    }
    observed = record.get("observed", {})
    observed_key = "native_boundary_material_name"
    if (
        thin != expected_thin
        or material_name.casefold() != "pec"
        or material_name != observed[observed_key]
        or thickness_um != effective["physical_thickness_um"]
        or assigned_ids != [native_id]
    ):
        raise RuntimeError("Q3D ThinConductor binding differs from physical source")


def _validate_q3d_body_incidence(evidence: dict[str, Any], faces: list[int]) -> None:
    incidence = evidence.get("incidence")
    if not isinstance(incidence, dict):
        raise RuntimeError("Q3D body incidence availability is not recorded")
    if incidence.get("status") == "unavailable":
        if (
            not isinstance(incidence.get("unavailable_reason"), str)
            or not incidence["unavailable_reason"]
        ):
            raise RuntimeError("Q3D unavailable incidence lacks a reason")
    elif incidence.get("status") == "observed":
        if (
            incidence.get("face_ids") != faces
            or not isinstance(incidence.get("face_edge_ids"), dict)
            or not isinstance(incidence.get("edge_vertex_ids"), dict)
        ):
            raise RuntimeError("Q3D observed native incidence is malformed")
    else:
        raise RuntimeError("Q3D native incidence status is invalid")


def _validate_q3d_region_ground(
    receipt: dict[str, Any], spec: Q3dSpec, nets: list[Any]
) -> None:
    region = receipt.get("region")
    if not isinstance(region, dict):
        raise TypeError("Q3D receipt has invalid Region evidence")
    requested_padding = list(spec.region_padding_um)
    if (
        region.get("padding_um") != list(spec.effective_region_padding_um)
        or region.get("requested_padding_um") != requested_padding
    ):
        raise RuntimeError("Q3D receipt Region padding does not match the spec")
    if (
        spec.effective_region is not None
        and region.get("source_effective_envelope") != spec.effective_region
    ):
        raise RuntimeError(
            "Q3D Region envelope differs from the bound neutral physical plan"
        )
    evidence = region.get("grounded_region")
    if spec.grounded_region_net is None:
        if evidence is not None:
            raise RuntimeError("open Q3D Region receipt contains grounding evidence")
        return
    if not isinstance(evidence, dict):
        raise TypeError("grounded Q3D Region receipt lacks evidence")
    source_id, source_bounds, source_faces = _q3d_region_evidence(
        evidence.get("source_region")
    )
    final_id, final_bounds, _final_faces = _q3d_region_evidence(
        evidence.get("final_region")
    )
    sheets = evidence.get("sheets")
    if (
        region.get("native_region_object_id") != final_id
        or region.get("native_bounding_box_um") != list(final_bounds)
        or evidence.get("target_net") != spec.grounded_region_net
        or evidence.get("target_net_origin") != "generated_region_enclosure"
        or evidence.get("native_final_region_padding_um") != [0.0] * 6
        or evidence.get("native_region_object_id") != final_id
        or evidence.get("native_bounding_box_um") != list(final_bounds)
        or not _q3d_values_close(list(source_bounds), list(final_bounds))
        or not isinstance(sheets, list)
        or len(sheets) != len(_Q3D_REGION_DIRECTIONS)
    ):
        raise RuntimeError("grounded Q3D Region receipt is invalid")
    sheet_ids: list[int] = []
    thin_boundaries: list[dict[str, Any]] = []
    source_face_ids: set[int] = set()
    for direction, record in zip(_Q3D_REGION_DIRECTIONS, sheets, strict=True):
        if not isinstance(record, dict):
            raise TypeError("grounded Q3D Region sheet receipt is invalid")
        sheet_name, boundary_name = _Q3D_REGION_SHEET_NAMES[direction]
        face_id = record.get("source_face_id")
        sheet_id = record.get("sheet_object_id")
        source = source_faces[direction]
        expected_bounds = _q3d_sheet_bounds(direction, final_bounds)
        if (
            record.get("direction") != direction
            or record.get("source_region_object_id") != source_id
            or face_id != source["native_face_id"]
            or face_id in source_face_ids
            or record.get("source_face_center_um") != source["native_face_center_um"]
            or record.get("source_face_normal") != source["native_face_normal"]
            or record.get("sheet_name") != sheet_name
            or not positive_unique_ids([sheet_id], 1)
            or not _q3d_values_close(
                record.get("sheet_bounding_box_um"), expected_bounds
            )
            or record.get("boundary_name") != boundary_name
            or record.get("native_thin_conductor_object_ids") != [sheet_id]
        ):
            raise RuntimeError("grounded Q3D Region sheet receipt is invalid")
        source_face_ids.add(cast(int, face_id))
        sheet_ids.append(cast(int, sheet_id))
        thin_boundaries.append(
            {
                "name": boundary_name,
                "bound_type": "ThinConductor",
                "object_ids": [sheet_id],
                "material": "pec",
                "thickness": "1um",
            }
        )
    declared_net_ids = {record["name"]: record["native_object_ids"] for record in nets}
    all_net_ids = {
        object_id for record in nets for object_id in record["native_object_ids"]
    }
    if (
        spec.grounded_region_net in declared_net_ids
        or evidence.get("native_declared_net_object_ids") != declared_net_ids
        or len(sheet_ids) != 6
        or len(set(sheet_ids)) != 6
        or set(sheet_ids) & all_net_ids
        or final_id in all_net_ids
        or final_id in sheet_ids
    ):
        raise RuntimeError("grounded Q3D Region object IDs are invalid")
    expected_ids = sheet_ids
    saved = evidence.get("native_saved_boundaries")
    if (
        evidence.get("native_target_net_object_ids") != expected_ids
        or evidence.get("native_design_validation")
        != {"method": "ValidateDesign", "ok": True}
        or not isinstance(saved, dict)
        or saved.get("target")
        != {
            "name": spec.grounded_region_net,
            "bound_type": "GroundNet",
            "origin": "generated_region_enclosure",
            "object_ids": expected_ids,
        }
        or saved.get("thin_conductors") != thin_boundaries
    ):
        raise RuntimeError("grounded Q3D Region native assignment receipt is invalid")


def _q3d_region_evidence(
    value: Any,
) -> tuple[
    int, tuple[float, float, float, float, float, float], dict[str, dict[str, Any]]
]:
    if not isinstance(value, dict):
        raise TypeError("Q3D Region evidence is invalid")
    region_id = value.get("native_object_id")
    bounds = _q3d_bounds(value.get("native_bounding_box_um"))
    faces = value.get("faces")
    if (
        not isinstance(region_id, int)
        or isinstance(region_id, bool)
        or region_id <= 0
        or not isinstance(faces, list)
        or len(faces) != len(_Q3D_REGION_DIRECTIONS)
    ):
        raise RuntimeError("Q3D Region evidence is invalid")
    by_direction: dict[str, dict[str, Any]] = {}
    face_ids: set[int] = set()
    for direction, record in zip(_Q3D_REGION_DIRECTIONS, faces, strict=True):
        if (
            not isinstance(record, dict)
            or record.get("direction") != direction
            or not positive_unique_ids([record.get("native_face_id")], 1)
            or record["native_face_id"] in face_ids
            or not _q3d_face_center_matches(
                direction, record.get("native_face_center_um"), bounds
            )
            or not _q3d_face_normal_matches(direction, record.get("native_face_normal"))
        ):
            raise RuntimeError("Q3D Region face evidence is invalid")
        face_ids.add(record["native_face_id"])
        by_direction[direction] = record
    return region_id, bounds, by_direction


def _q3d_bounds(value: Any) -> tuple[float, float, float, float, float, float]:
    if not isinstance(value, list) or len(value) != 6:
        raise RuntimeError("grounded Q3D Region bounds are invalid")
    try:
        bounds = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("grounded Q3D Region bounds are invalid") from exc
    if not all(math.isfinite(item) for item in bounds) or any(
        bounds[index] >= bounds[index + 3] for index in range(3)
    ):
        raise RuntimeError("grounded Q3D Region bounds are invalid")
    return cast(tuple[float, float, float, float, float, float], bounds)


def _q3d_face_center_matches(
    direction: str, value: Any, bounds: tuple[float, float, float, float, float, float]
) -> bool:
    if not isinstance(value, list) or len(value) != 3:
        return False
    try:
        center = [float(item) for item in value]
    except (TypeError, ValueError):
        return False
    if not all(math.isfinite(item) for item in center):
        return False
    x_min, y_min, z_min, x_max, y_max, z_max = bounds
    expected = {
        "+X": [x_max, (y_min + y_max) / 2, (z_min + z_max) / 2],
        "-X": [x_min, (y_min + y_max) / 2, (z_min + z_max) / 2],
        "+Y": [(x_min + x_max) / 2, y_max, (z_min + z_max) / 2],
        "-Y": [(x_min + x_max) / 2, y_min, (z_min + z_max) / 2],
        "+Z": [(x_min + x_max) / 2, (y_min + y_max) / 2, z_max],
        "-Z": [(x_min + x_max) / 2, (y_min + y_max) / 2, z_min],
    }[direction]
    return all(
        math.isclose(actual, required, rel_tol=0.0, abs_tol=1e-9)
        for actual, required in zip(center, expected, strict=True)
    )


def _q3d_sheet_bounds(
    direction: str, bounds: tuple[float, float, float, float, float, float]
) -> list[float]:
    x_min, y_min, z_min, x_max, y_max, z_max = bounds
    return {
        "+X": [x_max, y_min, z_min, x_max, y_max, z_max],
        "-X": [x_min, y_min, z_min, x_min, y_max, z_max],
        "+Y": [x_min, y_max, z_min, x_max, y_max, z_max],
        "-Y": [x_min, y_min, z_min, x_max, y_min, z_max],
        "+Z": [x_min, y_min, z_max, x_max, y_max, z_max],
        "-Z": [x_min, y_min, z_min, x_max, y_max, z_min],
    }[direction]


def _q3d_values_close(value: Any, expected: list[float]) -> bool:
    if not isinstance(value, list) or len(value) != len(expected):
        return False
    try:
        observed = [float(item) for item in value]
    except (TypeError, ValueError):
        return False
    return all(
        math.isclose(actual, required, rel_tol=0.0, abs_tol=1e-9)
        for actual, required in zip(observed, expected, strict=True)
    )


def _q3d_face_normal_matches(direction: str, value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 3:
        return False
    try:
        normal = [float(item) for item in value]
    except (TypeError, ValueError):
        return False
    if not all(math.isfinite(item) for item in normal):
        return False
    axis, sign = {
        "+X": (0, 1.0),
        "-X": (0, -1.0),
        "+Y": (1, 1.0),
        "-Y": (1, -1.0),
        "+Z": (2, 1.0),
        "-Z": (2, -1.0),
    }[direction]
    return math.isclose(normal[axis], sign, rel_tol=0.0, abs_tol=1e-9) and all(
        math.isclose(item, 0.0, rel_tol=0.0, abs_tol=1e-9)
        for index, item in enumerate(normal)
        if index != axis
    )


def _validate_normalized_matrices(
    path: Path, expected_rows: int, problem_types: set[str]
) -> None:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != expected_rows or not rows:
        raise RuntimeError("normalized matrix row count is invalid")
    expected_quantities = {"C", "G", "L", "R"}
    if {row.get("quantity") for row in rows} != expected_quantities:
        raise RuntimeError("normalized matrix quantities are incomplete")
    for row in rows:
        try:
            value = float(row["value"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("normalized matrix value is invalid") from exc
        if (
            not math.isfinite(value)
            or row.get("problem_type") not in problem_types
            or not row.get("row")
            or not row.get("column")
            or not row.get("unit")
        ):
            raise RuntimeError("normalized matrix row is invalid")
