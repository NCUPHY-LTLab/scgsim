"""Own Q3D grounded Region construction and saved enclosure readback."""

from __future__ import annotations

import math
from typing import Any, NamedTuple, cast

from scgsim.aedt.runtime.native.common import (
    native_object_property as _native_object_property,
)
from scgsim.aedt.runtime.native.common import (
    q3d_region_bounds as _q3d_region_bounds,
)
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


def _q3d_region_faces(
    region: Any, bounds: tuple[float, float, float, float, float, float]
) -> dict[str, _RegionFace]:
    faces: dict[str, _RegionFace] = {}
    for face in region.faces:
        try:
            center = tuple(float(value) for value in face.center)
            normal = tuple(float(value) for value in face.normal)
            face_id = int(face.id)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Q3D Region native face readback is invalid") from exc
        if (
            face_id <= 0
            or len(center) != 3
            or len(normal) != 3
            or not all(math.isfinite(value) for value in [*center, *normal])
        ):
            raise RuntimeError("Q3D Region native face readback is invalid")
        matches = [
            direction
            for direction, axis, boundary in (
                ("+X", 0, bounds[3]),
                ("-X", 0, bounds[0]),
                ("+Y", 1, bounds[4]),
                ("-Y", 1, bounds[1]),
                ("+Z", 2, bounds[5]),
                ("-Z", 2, bounds[2]),
            )
            if math.isclose(center[axis], boundary, rel_tol=0.0, abs_tol=1e-9)
        ]
        if len(matches) != 1 or matches[0] in faces:
            raise RuntimeError("Q3D Region native faces do not identify six directions")
        direction = matches[0]
        expected_center = _q3d_face_center(direction, bounds)
        if not all(
            math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-9)
            for value, expected in zip(center, expected_center, strict=True)
        ):
            raise RuntimeError("Q3D Region native face center is invalid")
        axis, sign = {
            "+X": (0, 1.0),
            "-X": (0, -1.0),
            "+Y": (1, 1.0),
            "-Y": (1, -1.0),
            "+Z": (2, 1.0),
            "-Z": (2, -1.0),
        }[direction]
        if not math.isclose(normal[axis], sign, rel_tol=0.0, abs_tol=1e-9) or any(
            not math.isclose(value, 0.0, rel_tol=0.0, abs_tol=1e-9)
            for index, value in enumerate(normal)
            if index != axis
        ):
            raise RuntimeError("Q3D Region native face normal is invalid")
        faces[direction] = _RegionFace(
            face, (center[0], center[1], center[2]), (normal[0], normal[1], normal[2])
        )
    if set(faces) != set(_Q3D_REGION_DIRECTIONS):
        raise RuntimeError("Q3D Region must expose exactly one face in each direction")
    if len({int(face.id) for face, _, _ in faces.values()}) != len(faces):
        raise RuntimeError("Q3D Region native face IDs are not unique")
    return faces


def _q3d_face_center(
    direction: str, bounds: tuple[float, float, float, float, float, float]
) -> list[float]:
    x_min, y_min, z_min, x_max, y_max, z_max = bounds
    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2
    z_mid = (z_min + z_max) / 2
    return {
        "+X": [x_max, y_mid, z_mid],
        "-X": [x_min, y_mid, z_mid],
        "+Y": [x_mid, y_max, z_mid],
        "-Y": [x_mid, y_min, z_mid],
        "+Z": [x_mid, y_mid, z_max],
        "-Z": [x_mid, y_mid, z_min],
    }[direction]


def _q3d_region_evidence(
    region: Any,
    bounds: tuple[float, float, float, float, float, float],
    faces: dict[str, _RegionFace],
) -> dict[str, Any]:
    return {
        "native_object_id": int(region.id),
        "native_bounding_box_um": list(bounds),
        "faces": [
            {
                "direction": direction,
                "native_face_id": int(faces[direction][0].id),
                "native_face_center_um": list(faces[direction][1]),
                "native_face_normal": list(faces[direction][2]),
            }
            for direction in _Q3D_REGION_DIRECTIONS
        ],
    }


def _q3d_region_sheet_geometry(
    direction: str, bounds: tuple[float, float, float, float, float, float]
) -> tuple[str, list[float], list[float], list[float]]:
    x_min, y_min, z_min, x_max, y_max, z_max = bounds
    if direction == "+X":
        return (
            "YZ",
            [x_max, y_min, z_min],
            [y_max - y_min, z_max - z_min],
            [x_max, y_min, z_min, x_max, y_max, z_max],
        )
    if direction == "-X":
        return (
            "YZ",
            [x_min, y_min, z_min],
            [y_max - y_min, z_max - z_min],
            [x_min, y_min, z_min, x_min, y_max, z_max],
        )
    if direction == "+Y":
        return (
            "ZX",
            [x_min, y_max, z_min],
            [z_max - z_min, x_max - x_min],
            [x_min, y_max, z_min, x_max, y_max, z_max],
        )
    if direction == "-Y":
        return (
            "ZX",
            [x_min, y_min, z_min],
            [z_max - z_min, x_max - x_min],
            [x_min, y_min, z_min, x_max, y_min, z_max],
        )
    if direction == "+Z":
        return (
            "XY",
            [x_min, y_min, z_max],
            [x_max - x_min, y_max - y_min],
            [x_min, y_min, z_max, x_max, y_max, z_max],
        )
    if direction == "-Z":
        return (
            "XY",
            [x_min, y_min, z_min],
            [x_max - x_min, y_max - y_min],
            [x_min, y_min, z_min, x_max, y_max, z_min],
        )
    raise RuntimeError("Q3D Region direction is invalid")


def _q3d_assignment(app: Any, name: str) -> list[int]:
    try:
        assignment = [
            int(value) for value in app.oboundary.GetExcitationAssignment(name)
        ]
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Q3D native assignment is unavailable for {name!r}"
        ) from exc
    if not assignment:
        raise RuntimeError(f"Q3D native assignment is empty for {name!r}")
    return assignment


def _same_q3d_bounds(observed: Any, expected: list[float]) -> bool:
    try:
        values = [float(value) for value in observed]
    except (TypeError, ValueError):
        return False
    return len(values) == len(expected) and all(
        math.isclose(actual, required, rel_tol=0.0, abs_tol=1e-9)
        for actual, required in zip(values, expected, strict=True)
    )


def seal_q3d_region(app: Any, spec: Q3dSpec) -> dict[str, Any]:
    """Create the six grounded enclosure sheets while preserving declared net assignments."""
    initial_region = app.modeler.get_object_from_name("Region")
    if initial_region is None or initial_region.name != "Region":
        raise RuntimeError("Q3D Region is unavailable for grounding")
    intended_bounds = _q3d_region_bounds(initial_region)
    source_faces = _q3d_region_faces(initial_region, intended_bounds)
    source_region = _q3d_region_evidence(initial_region, intended_bounds, source_faces)
    declared_net_ids = {net.name: _q3d_assignment(app, net.name) for net in spec.nets}
    if spec.grounded_region_net in app.net_names:
        raise RuntimeError("Q3D grounded Region net already exists")
    sheet_names = [name for name, _ in _Q3D_REGION_SHEET_NAMES.values()]
    if set(sheet_names) & set(app.modeler.sheet_names):
        raise RuntimeError("Q3D Region grounding sheet name already exists")
    initial_sheets = set(app.modeler.sheet_names)
    if not app.modeler.delete("Region"):
        raise RuntimeError("Q3D initial Region could not be frozen for grounding")
    created: dict[str, _CreatedRegionSheet] = {}
    for direction in _Q3D_REGION_DIRECTIONS:
        sheet_name, _ = _Q3D_REGION_SHEET_NAMES[direction]
        orientation, origin, sizes, expected_bounds = _q3d_region_sheet_geometry(
            direction, intended_bounds
        )
        sheet = app.modeler.create_rectangle(
            orientation=orientation,
            origin=origin,
            sizes=sizes,
            name=sheet_name,
            material="pec",
        )
        if (
            sheet is None
            or sheet.name != sheet_name
            or sheet_name not in app.modeler.sheet_names
            or not _same_q3d_bounds(sheet.bounding_box, expected_bounds)
        ):
            raise RuntimeError(
                f"Q3D Region grounding sheet creation failed for {direction}"
            )
        created[direction] = _CreatedRegionSheet(sheet, expected_bounds)
    region = _create_grounded_region(app, spec, intended_bounds)
    bounds = _q3d_region_bounds(region)
    faces = _q3d_region_faces(region, bounds)
    records: list[dict[str, Any]] = []
    for direction in _Q3D_REGION_DIRECTIONS:
        records.append(
            _assign_region_sheet(
                app,
                initial_region,
                source_faces[direction],
                created[direction],
                direction,
            )
        )
    if set(app.modeler.sheet_names) - initial_sheets != set(sheet_names):
        raise RuntimeError("Q3D Region grounding did not create exactly six sheets")
    target = app.assign_net(sheet_names, spec.grounded_region_net, "Ground")
    if (
        target is None
        or target.name != spec.grounded_region_net
        or target.type != "GroundNet"
    ):
        raise RuntimeError("Q3D Region enclosure Ground net creation failed")
    native_target_ids = _q3d_assignment(app, cast(str, spec.grounded_region_net))
    if native_target_ids != [record["sheet_object_id"] for record in records]:
        raise RuntimeError("Q3D Region grounding target net readback failed")
    if any(
        _q3d_assignment(app, name) != object_ids
        for name, object_ids in declared_net_ids.items()
    ):
        raise RuntimeError(
            "Q3D declared net assignment changed during Region grounding"
        )
    if not _same_q3d_bounds(region.bounding_box, list(bounds)):
        raise RuntimeError("Q3D Region native geometry changed during grounding")
    return {
        "target_net": spec.grounded_region_net,
        "target_net_origin": "generated_region_enclosure",
        "native_region_object_id": int(region.id),
        "native_bounding_box_um": list(bounds),
        "native_final_region_padding_um": [0.0] * 6,
        "source_region": source_region,
        "final_region": _q3d_region_evidence(region, bounds, faces),
        "sheets": records,
        "native_declared_net_object_ids": declared_net_ids,
        "native_target_net_object_ids": native_target_ids,
    }


def read_q3d_region_ground(
    app: Any, spec: Q3dSpec, region_record: dict[str, Any]
) -> dict[str, Any]:
    """Verify the saved enclosure geometry, sheet boundaries and generated ground net."""
    evidence = region_record.get("grounded_region")
    region = app.modeler.get_object_from_name("Region")
    if not isinstance(evidence, dict) or region is None:
        raise RuntimeError("Q3D grounded Region evidence is unavailable after save")
    bounds = _q3d_region_bounds(region)
    faces = _q3d_region_faces(region, bounds)
    if (
        evidence.get("target_net") != spec.grounded_region_net
        or evidence.get("target_net_origin") != "generated_region_enclosure"
        or not _same_q3d_bounds(evidence.get("native_bounding_box_um"), list(bounds))
        or evidence.get("native_final_region_padding_um") != [0.0] * 6
        or _native_object_property(region, "Material").strip('"').casefold()
        != cast(str, spec.materials[spec.vacuum_material_id].library_name).casefold()
    ):
        raise RuntimeError("Q3D grounded Region geometry changed after save")
    boundary_setup = app.design_properties.get("BoundarySetup")
    boundaries = (
        boundary_setup.get("Boundaries") if isinstance(boundary_setup, dict) else None
    )
    if not isinstance(boundaries, dict):
        raise TypeError("Q3D saved boundary records are unavailable")
    sheets = evidence.get("sheets")
    source_region = evidence.get("source_region")
    source_faces = (
        source_region.get("faces") if isinstance(source_region, dict) else None
    )
    if (
        not isinstance(sheets, list)
        or len(sheets) != len(_Q3D_REGION_DIRECTIONS)
        or not isinstance(source_faces, list)
        or len(source_faces) != len(_Q3D_REGION_DIRECTIONS)
    ):
        raise RuntimeError("Q3D grounded Region sheet evidence is invalid")
    source_region = cast(dict[str, Any], source_region)
    native_thin_conductors: list[dict[str, Any]] = []
    sheet_ids: list[int] = []
    for direction, record, source in zip(
        _Q3D_REGION_DIRECTIONS, sheets, source_faces, strict=True
    ):
        if not isinstance(record, dict) or not isinstance(source, dict):
            raise TypeError("Q3D grounded Region sheet evidence is invalid")
        sheet_name, boundary_name = _Q3D_REGION_SHEET_NAMES[direction]
        sheet_id = record.get("sheet_object_id")
        _, _, _, expected_bounds = _q3d_region_sheet_geometry(direction, bounds)
        sheet = app.modeler.get_object_from_name(sheet_name)
        saved = boundaries.get(boundary_name)
        if (
            record.get("direction") != direction
            or record.get("source_region_object_id")
            != source_region.get("native_object_id")
            or record.get("source_face_id") != source.get("native_face_id")
            or record.get("source_face_center_um")
            != source.get("native_face_center_um")
            or record.get("source_face_normal") != source.get("native_face_normal")
            or record.get("sheet_name") != sheet_name
            or not isinstance(sheet_id, int)
            or not _same_q3d_bounds(
                record.get("sheet_bounding_box_um"), expected_bounds
            )
            or record.get("boundary_name") != boundary_name
            or record.get("native_thin_conductor_object_ids") != [sheet_id]
            or sheet is None
            or int(sheet.id) != sheet_id
            or not _same_q3d_bounds(sheet.bounding_box, expected_bounds)
            or not isinstance(saved, dict)
            or saved.get("BoundType") != "ThinConductor"
            or saved.get("Objects") != [sheet_id]
            or saved.get("Material") != "pec"
            or saved.get("Thickness") != "1um"
            or _q3d_assignment(app, boundary_name) != [sheet_id]
        ):
            raise RuntimeError("Q3D saved thin conductor readback is invalid")
        native_thin_conductors.append(
            {
                "name": boundary_name,
                "bound_type": "ThinConductor",
                "object_ids": [sheet_id],
                "material": "pec",
                "thickness": "1um",
            }
        )
        sheet_ids.append(sheet_id)
    target = boundaries.get(spec.grounded_region_net)
    expected_target = evidence.get("native_target_net_object_ids")
    declared_net_ids = {
        net.name: [
            int(app.modeler.get_object_from_name(name).id) for name in net.object_names
        ]
        for net in spec.nets
    }
    if (
        not isinstance(target, dict)
        or target.get("BoundType") != "GroundNet"
        or target.get("Objects") != expected_target
        or expected_target != sheet_ids
        or _q3d_assignment(app, cast(str, spec.grounded_region_net)) != expected_target
        or evidence.get("native_declared_net_object_ids") != declared_net_ids
        or any(
            _q3d_assignment(app, name) != object_ids
            for name, object_ids in declared_net_ids.items()
        )
    ):
        raise RuntimeError("Q3D saved grounded Region net readback is invalid")
    return {
        **evidence,
        "native_region_object_id": int(region.id),
        "native_bounding_box_um": list(bounds),
        "final_region": _q3d_region_evidence(region, bounds, faces),
        "native_saved_boundaries": {
            "target": {
                "name": spec.grounded_region_net,
                "bound_type": "GroundNet",
                "origin": "generated_region_enclosure",
                "object_ids": expected_target,
            },
            "thin_conductors": native_thin_conductors,
        },
    }


def _create_grounded_region(
    app: Any, spec: Q3dSpec, intended_bounds: tuple[float, ...]
) -> Any:
    """Recreate the final vacuum Region with zero offsets and verify its source bounds."""
    region = app.modeler.create_region(
        pad_value=[0.0] * 6,
        pad_type="Absolute Offset",
        name="Region",
    )
    vacuum = spec.materials[spec.vacuum_material_id]
    region.material_name = vacuum.library_name
    if (
        region.name != "Region"
        or _native_object_property(region, "Material").strip('"').casefold()
        != cast(str, vacuum.library_name).casefold()
        or not _same_q3d_bounds(region.bounding_box, list(intended_bounds))
    ):
        raise RuntimeError("Q3D final grounded Region readback failed")
    return region


def _assign_region_sheet(
    app: Any,
    initial_region: Any,
    source_face_record: _RegionFace,
    created_sheet: _CreatedRegionSheet,
    direction: str,
) -> dict[str, Any]:
    """Assign one enclosure sheet and verify the exact native object assignment."""
    source_face, source_center, source_normal = source_face_record
    sheet_name, boundary_name = _Q3D_REGION_SHEET_NAMES[direction]
    sheet, expected_bounds = created_sheet
    boundary = app.assign_thin_conductor(
        assignment=sheet,
        material="pec",
        thickness=1,
        name=boundary_name,
    )
    if not boundary or boundary.name != boundary_name:
        raise RuntimeError(
            f"Q3D Region thin conductor assignment failed for {direction}"
        )
    sheet_id = int(sheet.id)
    if _q3d_assignment(app, boundary_name) != [sheet_id]:
        raise RuntimeError(f"Q3D Region thin conductor readback failed for {direction}")
    return {
        "direction": direction,
        "source_region_object_id": int(initial_region.id),
        "source_face_id": int(source_face.id),
        "source_face_center_um": list(source_center),
        "source_face_normal": list(source_normal),
        "sheet_name": sheet_name,
        "sheet_object_id": sheet_id,
        "sheet_bounding_box_um": [float(value) for value in sheet.bounding_box],
        "boundary_name": boundary_name,
        "native_thin_conductor_object_ids": [sheet_id],
    }


class _RegionFace(NamedTuple):
    """One native Region face with its verified center and outward normal."""

    face: Any
    center: tuple[float, float, float]
    normal: tuple[float, float, float]


class _CreatedRegionSheet(NamedTuple):
    """One enclosure sheet with its expected native bounding box."""

    sheet: Any
    expected_bounds: list[float]
