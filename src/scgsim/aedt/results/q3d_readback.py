"""Strict offline validation of a Q3D run on an SGB model: Nets, Region, setup and the C/G matrix."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

from scgsim.aedt.results.convergence.q3d import read_q3d_convergence
from scgsim.aedt.results.matrices import parse_matrix_export
from scgsim.aedt.results.readback_common import (
    positive_unique_ids,
    validate_imported_sgb_model,
)
from scgsim.aedt.results.receipt_validation import contained
from scgsim.aedt.specs.common import MatrixRunControl
from scgsim.aedt.specs.q3d import Q3dSpec

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
    """Bind the receipt's model to the request: the imported SGB model, its Nets and Region."""
    if ("ports" in receipt or not prepared) and receipt.get("ports") != []:
        raise RuntimeError("Q3D receipt must not contain HFSS ports")
    validate_imported_sgb_model(receipt, spec.model)
    _validate_conductor_sheets(receipt["geometry"], spec)
    nets = receipt.get("nets")
    if not isinstance(nets, list) or len(nets) != len(spec.nets):
        raise RuntimeError("Q3D receipt has invalid net evidence")
    for record, expected in zip(nets, spec.nets, strict=True):
        _validate_q3d_net(record, expected)
    net_object_ids = [
        object_id for record in nets for object_id in record["native_object_ids"]
    ]
    if len(set(net_object_ids)) != len(net_object_ids):
        raise RuntimeError("Q3D native object IDs must be unique across nets")
    _validate_q3d_region_ground(receipt, spec, nets)


def _validate_conductor_sheets(geometry: dict[str, Any], spec: Q3dSpec) -> None:
    """Each Route A sheet group is one thin conductor of the model's thickness; Route C has none."""
    sheets = geometry.get("conductor_sheets")
    expected = dict(spec.model.sheet_thickness_um)
    if (
        not isinstance(sheets, list)
        or {record.get("group"): record.get("thickness_um") for record in sheets}
        != expected
        or any(str(record.get("material", "")).casefold() != "pec" for record in sheets)
    ):
        raise RuntimeError(
            "the Q3D conductor sheets are not the model's sheet groups at their thickness"
        )


def _validate_q3d_net(record: Any, expected: Any) -> None:
    """A Net record names the request's Net, type and objects, with one native id per object, and nothing else."""
    if (
        not isinstance(record, dict)
        or set(record) != {"name", "net_type", "object_names", "native_object_ids"}
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


def _validate_q3d_region_ground(
    receipt: dict[str, Any], spec: Q3dSpec, nets: list[Any]
) -> None:
    region = receipt.get("region")
    if not isinstance(region, dict):
        raise TypeError("Q3D receipt has invalid Region evidence")
    tolerance = receipt["geometry"].get("readback_tolerance_um")
    bounds = region.get("bounds_um")
    if (
        region.get("object_name") != "Region"
        or not isinstance(tolerance, float)
        or not isinstance(bounds, list)
        or len(bounds) != 6
        or any(
            abs(float(a) - b) > tolerance
            for a, b in zip(bounds, spec.model.vacuum_bounds_um, strict=True)
        )
    ):
        raise RuntimeError("the Q3D Region is not the model's VACUUM_REGION box")
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
