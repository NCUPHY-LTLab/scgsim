"""Check the Q3D Region receipt, including the grounded-Region enclosure.

The Region is the air box around the model. A grounded Region is closed by six
ThinConductor sheets (one per face, in the order +X, -X, +Y, -Y, +Z, -Z) that form
a ground net. Owns: Region padding and envelope evidence, and the face and sheet
geometry of the grounded enclosure. It does not own net or body evidence
(``q3d_readback``).
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

from scgsim.aedt.results.readback_common import positive_unique_ids
from scgsim.aedt.specs.q3d import Q3dSpec

Bounds = tuple[float, float, float, float, float, float]


class _RegionSnapshot(NamedTuple):
    """A Region as recorded: native object ID, bounding box and faces by direction."""

    object_id: int
    bounds: Bounds
    faces: dict[str, dict[str, Any]]


_REGION_DIRECTIONS = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")

# Native sheet name and ThinConductor boundary name for each Region face.
_REGION_SHEET_NAMES = {
    "+X": ("SCGSimRegionGroundPX", "SCGSimRegionGroundPXThinConductor"),
    "-X": ("SCGSimRegionGroundNX", "SCGSimRegionGroundNXThinConductor"),
    "+Y": ("SCGSimRegionGroundPY", "SCGSimRegionGroundPYThinConductor"),
    "-Y": ("SCGSimRegionGroundNY", "SCGSimRegionGroundNYThinConductor"),
    "+Z": ("SCGSimRegionGroundPZ", "SCGSimRegionGroundPZThinConductor"),
    "-Z": ("SCGSimRegionGroundNZ", "SCGSimRegionGroundNZThinConductor"),
}

# Outward face normal of each Region face as (axis index, sign).
_FACE_NORMAL_AXIS_SIGN = {
    "+X": (0, 1.0),
    "-X": (0, -1.0),
    "+Y": (1, 1.0),
    "-Y": (1, -1.0),
    "+Z": (2, 1.0),
    "-Z": (2, -1.0),
}

# Geometry readback is compared to this absolute tolerance in micrometres.
# TODO(human): why 1e-9 um is the right tolerance.
_GEOMETRY_ABS_TOL_UM = 1e-9


def validate_q3d_region_ground(
    receipt: dict[str, Any], spec: Q3dSpec, nets: list[Any]
) -> None:
    """Check Region padding; if the Region is grounded, check its enclosure too."""
    region = receipt.get("region")
    if not isinstance(region, dict):
        raise TypeError("Q3D receipt has invalid Region evidence")
    _check_region_padding(region, spec)
    evidence = region.get("grounded_region")
    if spec.grounded_region_net is None:
        if evidence is not None:
            raise RuntimeError("open Q3D Region receipt contains grounding evidence")
        return
    if not isinstance(evidence, dict):
        raise TypeError("grounded Q3D Region receipt lacks evidence")
    source = _region_evidence(evidence.get("source_region"))
    final = _region_evidence(evidence.get("final_region"))
    _check_grounded_region_header(region, evidence, spec, source, final)
    sheet_ids, thin_boundaries = _check_ground_sheets(
        evidence["sheets"], source, final.bounds
    )
    _check_ground_object_ids(evidence, spec, nets, final.object_id, sheet_ids)
    _check_ground_assignment(evidence, spec, sheet_ids, thin_boundaries)


def _check_region_padding(region: dict[str, Any], spec: Q3dSpec) -> None:
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


def _check_grounded_region_header(
    region: dict[str, Any],
    evidence: dict[str, Any],
    spec: Q3dSpec,
    source: _RegionSnapshot,
    final: _RegionSnapshot,
) -> None:
    """The final Region must be the source Region with no extra native padding."""
    final_id, final_bounds = final.object_id, final.bounds
    sheets = evidence.get("sheets")
    if (
        region.get("native_region_object_id") != final_id
        or region.get("native_bounding_box_um") != list(final_bounds)
        or evidence.get("target_net") != spec.grounded_region_net
        or evidence.get("target_net_origin") != "generated_region_enclosure"
        or evidence.get("native_final_region_padding_um") != [0.0] * 6
        or evidence.get("native_region_object_id") != final_id
        or evidence.get("native_bounding_box_um") != list(final_bounds)
        or not _values_close(list(source.bounds), list(final_bounds))
        or not isinstance(sheets, list)
        or len(sheets) != len(_REGION_DIRECTIONS)
    ):
        raise RuntimeError("grounded Q3D Region receipt is invalid")


def _check_ground_sheets(
    sheets: list[Any], source: _RegionSnapshot, final_bounds: Bounds
) -> tuple[list[int], list[dict[str, Any]]]:
    """Check one sheet per Region face; return sheet IDs and expected boundaries."""
    sheet_ids: list[int] = []
    thin_boundaries: list[dict[str, Any]] = []
    source_face_ids: set[int] = set()
    for direction, record in zip(_REGION_DIRECTIONS, sheets, strict=True):
        if not isinstance(record, dict):
            raise TypeError("grounded Q3D Region sheet receipt is invalid")
        sheet_name, boundary_name = _REGION_SHEET_NAMES[direction]
        face_id = record.get("source_face_id")
        sheet_id = record.get("sheet_object_id")
        source_face = source.faces[direction]
        expected_bounds = _sheet_bounds(direction, final_bounds)
        if (
            record.get("direction") != direction
            or record.get("source_region_object_id") != source.object_id
            or face_id != source_face["native_face_id"]
            or face_id in source_face_ids
            or record.get("source_face_center_um")
            != source_face["native_face_center_um"]
            or record.get("source_face_normal") != source_face["native_face_normal"]
            or record.get("sheet_name") != sheet_name
            or not positive_unique_ids([sheet_id], 1)
            or not _values_close(record.get("sheet_bounding_box_um"), expected_bounds)
            or record.get("boundary_name") != boundary_name
            or record.get("native_thin_conductor_object_ids") != [sheet_id]
        ):
            raise RuntimeError("grounded Q3D Region sheet receipt is invalid")
        source_face_ids.add(face_id)
        sheet_ids.append(sheet_id)
        thin_boundaries.append(
            {
                "name": boundary_name,
                "bound_type": "ThinConductor",
                "object_ids": [sheet_id],
                "material": "pec",
                # TODO(human): why the ground sheets are recorded with a nominal 1 um.
                "thickness": "1um",
            }
        )
    return sheet_ids, thin_boundaries


def _check_ground_object_ids(
    evidence: dict[str, Any],
    spec: Q3dSpec,
    nets: list[Any],
    final_id: int,
    sheet_ids: list[int],
) -> None:
    """Ground sheets and Region must not share native IDs with each other or any net."""
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


def _check_ground_assignment(
    evidence: dict[str, Any],
    spec: Q3dSpec,
    sheet_ids: list[int],
    thin_boundaries: list[dict[str, Any]],
) -> None:
    """The ground net must be assigned to exactly the six sheets, and validate."""
    saved = evidence.get("native_saved_boundaries")
    if (
        evidence.get("native_target_net_object_ids") != sheet_ids
        or evidence.get("native_design_validation")
        != {"method": "ValidateDesign", "ok": True}
        or not isinstance(saved, dict)
        or saved.get("target")
        != {
            "name": spec.grounded_region_net,
            "bound_type": "GroundNet",
            "origin": "generated_region_enclosure",
            "object_ids": sheet_ids,
        }
        or saved.get("thin_conductors") != thin_boundaries
    ):
        raise RuntimeError("grounded Q3D Region native assignment receipt is invalid")


def _region_evidence(value: Any) -> _RegionSnapshot:
    """Validate one recorded Region and return it as a snapshot."""
    if not isinstance(value, dict):
        raise TypeError("Q3D Region evidence is invalid")
    region_id = value.get("native_object_id")
    bounds = _bounds(value.get("native_bounding_box_um"))
    faces = value.get("faces")
    if (
        not isinstance(region_id, int)
        or isinstance(region_id, bool)
        or region_id <= 0
        or not isinstance(faces, list)
        or len(faces) != len(_REGION_DIRECTIONS)
    ):
        raise RuntimeError("Q3D Region evidence is invalid")
    by_direction: dict[str, dict[str, Any]] = {}
    face_ids: set[int] = set()
    for direction, record in zip(_REGION_DIRECTIONS, faces, strict=True):
        if (
            not isinstance(record, dict)
            or record.get("direction") != direction
            or not positive_unique_ids([record.get("native_face_id")], 1)
            or record["native_face_id"] in face_ids
            or not _face_center_matches(
                direction, record.get("native_face_center_um"), bounds
            )
            or not _face_normal_matches(direction, record.get("native_face_normal"))
        ):
            raise RuntimeError("Q3D Region face evidence is invalid")
        face_ids.add(record["native_face_id"])
        by_direction[direction] = record
    return _RegionSnapshot(region_id, bounds, by_direction)


def _bounds(value: Any) -> Bounds:
    """Parse ``[x_min, y_min, z_min, x_max, y_max, z_max]``; require a positive box."""
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
    return bounds


def _finite_floats(value: Any, count: int) -> list[float] | None:
    """``value`` as ``count`` finite floats, or ``None`` if it is not one."""
    if not isinstance(value, list) or len(value) != count:
        return None
    try:
        numbers = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in numbers):
        return None
    return numbers


def _face_center_matches(direction: str, value: Any, bounds: Bounds) -> bool:
    """A face centre lies on its bounding-box face, at the middle of the other axes."""
    center = _finite_floats(value, 3)
    if center is None:
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
        math.isclose(actual, required, rel_tol=0.0, abs_tol=_GEOMETRY_ABS_TOL_UM)
        for actual, required in zip(center, expected, strict=True)
    )


def _sheet_bounds(direction: str, bounds: Bounds) -> list[float]:
    """Bounding box of the zero-thickness sheet lying on one Region face."""
    x_min, y_min, z_min, x_max, y_max, z_max = bounds
    return {
        "+X": [x_max, y_min, z_min, x_max, y_max, z_max],
        "-X": [x_min, y_min, z_min, x_min, y_max, z_max],
        "+Y": [x_min, y_max, z_min, x_max, y_max, z_max],
        "-Y": [x_min, y_min, z_min, x_max, y_min, z_max],
        "+Z": [x_min, y_min, z_max, x_max, y_max, z_max],
        "-Z": [x_min, y_min, z_min, x_max, y_max, z_min],
    }[direction]


def _values_close(value: Any, expected: list[float]) -> bool:
    if not isinstance(value, list) or len(value) != len(expected):
        return False
    try:
        observed = [float(item) for item in value]
    except (TypeError, ValueError):
        return False
    return all(
        math.isclose(actual, required, rel_tol=0.0, abs_tol=_GEOMETRY_ABS_TOL_UM)
        for actual, required in zip(observed, expected, strict=True)
    )


def _face_normal_matches(direction: str, value: Any) -> bool:
    """The normal is the unit vector along the face's axis, signed outward."""
    normal = _finite_floats(value, 3)
    if normal is None:
        return False
    axis, sign = _FACE_NORMAL_AXIS_SIGN[direction]
    return math.isclose(
        normal[axis], sign, rel_tol=0.0, abs_tol=_GEOMETRY_ABS_TOL_UM
    ) and all(
        math.isclose(item, 0.0, rel_tol=0.0, abs_tol=_GEOMETRY_ABS_TOL_UM)
        for index, item in enumerate(normal)
        if index != axis
    )
