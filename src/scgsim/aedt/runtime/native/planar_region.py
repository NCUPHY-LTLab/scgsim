"""Create and verify the native solution Region and enclosure boundaries."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from scgsim.aedt.preparation.planar_source import plain
from scgsim.aedt.runtime.native.common import (
    native_object_boolean_property,
    native_object_evidence,
    native_object_property,
    resolve_native_assignment,
)


def _region_bounds(region: Any) -> tuple[float, ...]:
    bounds = tuple(float(value) for value in region.bounding_box)
    if (
        len(bounds) != 6
        or not all(math.isfinite(value) for value in bounds)
        or any(bounds[index] >= bounds[index + 3] for index in range(3))
    ):
        raise RuntimeError("native EPR Region bounds are invalid")
    return bounds


def verified_region_bounds(region: Any, plan: Mapping[str, Any]) -> tuple[float, ...]:
    """Compare native Region bounds with the prepared absolute-offset plan."""
    bounds = _region_bounds(region)
    loop = tuple(
        tuple(float(value) for value in point)
        for point in plan["envelope_outer_loop_um"]
    )
    z_min, z_max = plan["z_range_um"]
    expected = (
        min(point[0] for point in loop),
        min(point[1] for point in loop),
        float(z_min),
        max(point[0] for point in loop),
        max(point[1] for point in loop),
        float(z_max),
    )
    if any(
        not math.isclose(
            actual,
            wanted,
            rel_tol=0.0,
            abs_tol=max(1e-6, 1e-9 * max(abs(actual), abs(wanted))),
        )
        for actual, wanted in zip(bounds, expected, strict=False)
    ):
        raise RuntimeError("native EPR Region bounds differ from source envelope")
    return bounds


def create_epr_region(app: Any, source: Mapping[str, Any]) -> dict[str, Any]:
    """Create the single planned Region and verify its material and dimensions."""
    plan = source["native_region"]
    if app.modeler.get_object_from_name("Region") is not None:
        raise RuntimeError("new EPR model already has Region")
    region = app.modeler.create_region(
        pad_value=list(plan["padding_um"]),
        pad_type="Absolute Offset",
        name="Region",
    )
    if region is False or region is None or region.name != "Region":
        raise RuntimeError("native EPR Region creation failed")
    # HFSS partitions overlapping interior bodies from Region in solver and
    # Calculator volume support; its special Region object rejects CAD subtract.
    region.material_name = str(plan["material_id"])
    observed_material = native_object_property(region, "Material").strip('"')
    if observed_material.casefold() != str(plan["material_id"]).casefold():
        raise RuntimeError("native EPR Region vacuum material differs")
    if not native_object_boolean_property(region, "Solve Inside"):
        raise RuntimeError("native EPR Region must solve inside vacuum")
    evidence = native_object_evidence(app, "Region")
    if evidence["native_object_type"] != "Solid":
        raise RuntimeError("native EPR Region is not solid")
    return {
        "kind": "solution_domain",
        "semantic_id": "Region",
        "material_id": plan["material_id"],
        "logical_vacuum_ids": list(plan["logical_vacuum_ids"]),
        "object_name": "Region",
        "padding_um": list(plan["padding_um"]),
        "native_solve_inside": True,
        "native_bounding_box_um": list(verified_region_bounds(region, plan)),
        **evidence,
    }


def _enclosure_faces(app: Any, source: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the Region's six outer faces without assigning a boundary."""

    region = app.modeler.get_object_from_name("Region")
    if region is None:
        raise RuntimeError("native EPR Region is unavailable")
    bounds = verified_region_bounds(region, source["native_region"])
    z_min, z_max = source["native_region"]["z_range_um"]
    face_records: list[dict[str, Any]] = []
    plane_labels = ("x_minus", "y_minus", "bottom", "x_plus", "y_plus", "top")
    for face in region.faces:
        center = tuple(float(value) for value in face.center)
        matches = [
            plane_labels[index]
            for index in range(6)
            if math.isclose(center[index % 3], bounds[index], rel_tol=0.0, abs_tol=1e-6)
        ]
        if len(matches) == 1:
            face_records.append(
                {
                    "face_id": int(face.id),
                    "object_name": "Region",
                    "center_um": list(center),
                    "enclosure_plane": matches[0],
                }
            )
    planes = [item["enclosure_plane"] for item in face_records]
    if sorted(planes) != sorted(plane_labels):
        raise RuntimeError("native closed-enclosure face selection is incomplete")
    face_ids = [item["face_id"] for item in face_records]
    if len(face_ids) != len(set(face_ids)):
        raise RuntimeError("native closed-enclosure face selection repeats a face")
    return {
        "face_records": face_records,
        "envelope_outer_loop_um": plain(
            source["native_region"]["envelope_outer_loop_um"]
        ),
        "z_range_um": [z_min, z_max],
        "native_bounding_box_um": list(bounds),
    }


def _verify_no_explicit_region_boundary(app: Any) -> None:
    """Require native assignment evidence that leaves Region to HFSS defaults."""

    raw = [str(value) for value in app.oboundary.GetBoundaries()]
    if len(raw) % 2:
        raise RuntimeError("native HFSS boundary list is invalid")
    names = raw[::2]
    if len(names) != len(set(names)):
        raise RuntimeError("native HFSS boundary names are not unique")
    for name in names:
        assigned = app.oboundary.GetBoundaryAssignment(name)
        if assigned is None:
            raise RuntimeError(
                f"native boundary assignment is unavailable for {name!r}"
            )
        try:
            raw_ids = [int(value) for value in assigned]
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"native boundary assignment IDs are invalid for {name!r}"
            ) from exc
        if not raw_ids or len(raw_ids) != len(set(raw_ids)):
            raise RuntimeError(
                f"native boundary assignment IDs are invalid for {name!r}"
            )
        _, _, objects = resolve_native_assignment(app, raw_ids)
        if "Region" in objects:
            raise RuntimeError(f"Region has an explicit native boundary {name!r}")


def implicit_closed_enclosure(app: Any, source: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the implicit HFSS enclosure without adding an explicit boundary."""
    if source["native_region"].get("outer_boundary_policy") != "hfss_default.v1":
        raise ValueError("EPR Region boundary policy requires re-preparation")
    faces = _enclosure_faces(app, source)
    _verify_no_explicit_region_boundary(app)
    return {
        "boundary_policy": "hfss_default.v1",
        "explicit_assignment": False,
        **faces,
    }
