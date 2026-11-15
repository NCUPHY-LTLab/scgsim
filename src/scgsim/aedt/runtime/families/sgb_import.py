"""Import the guarded SGB STEP into an empty HFSS or Q3D design and make the assignments its sidecar names.

AEDT models the geometry Palace meshes (issue #67).  :func:`import_sgb_model` runs, in order, and fails
before any assignment when what AEDT imported is not the guarded model:

1. check the prepared files again and read the guard (:meth:`PreparedSgbModel.verify_files`); plan every
   boundary and Net name from the sidecar and the guard (:mod:`.sgb_boundaries`), so a model the import
   cannot realize is refused before AEDT is touched;
2. import the STEP into the empty design; the objects must be exactly the guard's solids;
3. match every imported face to its planned face by geometry (:func:`.sgb_faces.match_solid_faces`);
4. rebuild the guard's free faces (sheets with no imported solid on either side) as sheet objects from
   their loops, the only faces STEP cannot carry;
5. realize the materials: a dielectric solid takes its stack material, a conductor solid (Route C) is PEC;
6. create the Region from the plan's ``VACUUM_REGION`` box with the vacuum's material.  HFSS gets Perfect H
   on the Region's faces: Palace assigns nothing to its exterior, which is its natural PMC condition, and
   the two realizations must have the same outer boundary condition;
7. HFSS: one Perfect E per Net over the Route A sheet faces, and a Lumped RLC over the port-sheet faces of
   each junction and of each generic ``LumpedRlc`` (issue #67 decision 8); in a Driven Terminal design, a
   lumped port over the port-sheet faces of each ``LumpedTerminalPort``, its integration line read from
   those faces' edges as the Lumped RLC's current line is.  Each assignment and its readback use the
   planned name.  Q3D assigns its Nets, with their request's types, and their sources and
   sinks after the import (:mod:`.q3d_terminals`); in Route A it first rebuilds each conductor-sheet group
   as one sheet object, a PEC thin conductor of its film's thickness, since Q3D puts a Net on objects only.

Every port sheet must carry a junction, an RLC or a lumped Terminal port.  Lumped Terminal ports are
assigned only in a Driven Terminal design, and junctions and RLCs only outside one.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from scgsim.aedt.preparation.sgb_model import PreparedSgbModel, port_group
from scgsim.aedt.preparation.sgb_q3d import sheet_object_names
from scgsim.aedt.runtime.families.sgb_boundaries import (
    HfssBoundaryPlan,
    plan_hfss_boundaries,
)
from scgsim.aedt.runtime.families.sgb_faces import (
    ImportedFace,
    SgbImportError,
    area_tolerance_um2,
    import_tolerance_um,
    match_solid_faces,
)
from scgsim.aedt.runtime.families.sgb_ports import assign_lumped_port, assign_port_rlc
from scgsim.aedt.runtime.native.common import (
    detached_data,
    native_boundary_type,
    native_object_boolean_property,
    native_object_property,
    realize_source_material,
)
from scgsim.contracts.sgb_sidecar import Sidecar, read_sidecar, views
from scgsim.geometry.sgb.output.exchange.guard import (
    FreeFace,
    LoopSegment,
    StepGuard,
    read_step_guard,
)
from scgsim.geometry.sgb.output.sidecar.aedt import aedt_names

Family = Literal["hfss", "q3d"]
Point = tuple[float, float, float]
REGION_NAME = "Region"
OUTER_BOUNDARY_NAME = "outer_perfect_h"


@dataclass(frozen=True)
class ImportedSgbModel:
    """Where each planned solid and face of the SGB model is in the AEDT design, and what was assigned.

    ``objects`` maps each planned solid to its AEDT object.  ``face_copies`` maps each planned face to its
    copies, AEDT object name to AEDT face id: a face two imported solids share imports once on each.  A
    rebuilt free face has one copy, the face of its sheet object (``sheets``).  ``record`` is the
    detached evidence of every assignment.
    """

    objects: Mapping[str, str]
    face_copies: Mapping[str, Mapping[str, int]]
    sheets: Mapping[str, str]
    record: dict[str, Any] = field(default_factory=dict)


def import_sgb_model(
    app: Any, model: PreparedSgbModel, *, family: Family
) -> ImportedSgbModel:
    """Import ``model`` into the empty design of ``app`` and assign what its sidecar names (see the module)."""
    _check_supported(model, family)
    model.verify_files()
    guard = read_step_guard(model.path("guard"))
    sidecar = read_sidecar(model.path("sidecar"))
    _check_ports(model, sidecar, family)
    hfss = (
        plan_hfss_boundaries(
            sidecar, guard, model.junctions, model.lumped_rlcs, model.lumped_ports
        )
        if family == "hfss"
        else None
    )
    # AEDT's readback is coarser than the sidecar's tolerance_um (decision 9), so every comparison with what
    # AEDT reports uses a thousandth of the bound model's own grid.
    tolerance_um = import_tolerance_um(float(sidecar["model"]["grid_um"]))
    _check_design(app, model, family)
    _import_step(app, model, guard)
    face_copies = _match_faces(app, guard, tolerance_um)
    sheets = _rebuild_free_faces(app, guard["free_faces"], tolerance_um)
    for face_id, name in sheets.items():
        face_copies[face_id] = {name: int(app.modeler[name].faces[0].id)}
    objects = {solid["solid"]: solid["aedt_name"] for solid in guard["solids"]}
    imported = ImportedSgbModel(objects, face_copies, sheets)
    imported.record["materials"] = _assign_materials(
        app, model, sidecar, objects, family
    )
    imported.record["region"] = _create_region(
        app, model, sidecar, family, tolerance_um
    )
    if hfss is not None:
        imported.record["perfect_e"] = assign_sheet_perfect_e(app, hfss, face_copies)
        imported.record["junctions"] = [
            assign_port_rlc(app, rlc, face_copies, tolerance_um)
            for rlc in hfss.junctions
        ]
        imported.record["lumped_rlcs"] = [
            assign_port_rlc(app, rlc, face_copies, tolerance_um)
            for rlc in hfss.lumped_rlcs
        ]
        if hfss.lumped_ports:
            port_faces = lumped_port_faces(guard, sheets)
            imported.record["lumped_ports"] = [
                assign_lumped_port(app, item, port_faces, tolerance_um)
                for item in hfss.lumped_ports
            ]
    if family == "q3d":
        imported.record["conductor_sheets"] = rebuild_conductor_sheets(
            app, model, sidecar, imported
        )
    imported.record["readback_tolerance_um"] = tolerance_um
    imported.record["model_identity_sha256"] = model.identity_sha256
    return imported


def lumped_port_faces(guard: StepGuard, sheets: Mapping[str, str]) -> dict[str, Any]:
    """Planned face -> the sheet object it was rebuilt as (a free face), or its guard record (on a solid)."""
    faces: dict[str, Any] = {
        face["face"]: face for solid in guard["solids"] for face in solid["faces"]
    }
    faces.update(sheets)
    return faces


def imported_geometry_record(imported: ImportedSgbModel) -> dict[str, Any]:
    """The detached record of an imported model: where its solids and faces are, and what was assigned."""
    return {
        "objects": dict(imported.objects),
        "face_copies": {
            face: dict(copies) for face, copies in imported.face_copies.items()
        },
        "sheets": dict(imported.sheets),
        **detached_data(imported.record),
    }


def _check_supported(model: PreparedSgbModel, family: Family) -> None:
    """Refuse what the import cannot realize yet rather than realize it partly."""
    if family not in ("hfss", "q3d"):
        raise ValueError(f"family must be 'hfss' or 'q3d', got {family!r}")
    if family == "q3d" and (model.junctions or model.lumped_rlcs):
        raise ValueError(
            "Q3D has no junctions or lumped RLCs; they belong to HFSS Eigenmode"
        )
    if family == "q3d" and model.lumped_ports:
        raise ValueError("Q3D has no lumped Terminal ports; they belong to HFSS Driven")
    if model.lumped_ports and (model.junctions or model.lumped_rlcs):
        raise ValueError(
            "a model with lumped Terminal ports is a Driven model; its junctions and lumped RLCs "
            "belong to HFSS Eigenmode"
        )


def _check_design(app: Any, model: PreparedSgbModel, family: Family) -> None:
    """Lumped Terminal ports need a Driven Terminal design, and a Driven Terminal design needs them."""
    if family != "hfss":
        return
    terminal = app.solution_type == "Terminal"
    if bool(model.lumped_ports) != terminal:
        raise SgbImportError(
            f"the design solves {app.solution_type!r} and the model has "
            f"{len(model.lumped_ports)} lumped Terminal ports; they go together"
        )


def _check_ports(model: PreparedSgbModel, sidecar: Sidecar, family: Family) -> None:
    """In HFSS every port sheet must carry a junction, an RLC or a lumped port; an untreated one has no AEDT meaning."""
    if family != "hfss":
        return
    bound = {
        *(item.port_group for item in model.junctions),
        *(port_group(item.port_sheet_id) for item in model.lumped_rlcs),
        *(port_group(item.port_sheet_id) for item in model.lumped_ports),
    }
    unbound = sorted(
        p["group"] for p in sidecar["port"].values() if p["group"] not in bound
    )
    if unbound:
        raise ValueError(
            f"port sheets {unbound} carry no junction, LumpedRlc or LumpedTerminalPort; give each one "
            "its treatment"
        )


# ---------------------------------------------------------------- import and face identity


def _import_step(app: Any, model: PreparedSgbModel, guard: StepGuard) -> None:
    """Import the STEP into an empty design; the objects AEDT creates must be exactly the guarded solids."""
    if list(app.modeler.object_names):
        raise SgbImportError(
            f"the design is not empty: {sorted(app.modeler.object_names)}"
        )
    app.modeler.model_units = "um"
    # Coplanar faces of one solid are distinct planned faces (bands, port sheets), so AEDT must not
    # merge them; healing would move them.
    imported = app.modeler.import_3d_cad(
        str(model.path("step")),
        healing=False,
        merge_planar_faces=False,
        import_free_surfaces=False,
    )
    if imported is not True:
        raise SgbImportError(f"AEDT did not import {model.path('step').name}")
    observed = sorted(app.modeler.object_names)
    planned = sorted(solid["aedt_name"] for solid in guard["solids"])
    if observed != planned:
        raise SgbImportError(
            f"AEDT imported objects {observed}, the guard names {planned}"
        )


def _match_faces(
    app: Any, guard: StepGuard, tolerance_um: float
) -> dict[str, dict[str, int]]:
    """Planned face id -> {AEDT object: AEDT face id} over every imported solid."""
    copies: dict[str, dict[str, int]] = {}
    for solid in guard["solids"]:
        name = solid["aedt_name"]
        faces = [imported_face(face) for face in app.modeler[name].faces]
        matched = match_solid_faces(name, solid["faces"], faces, tolerance_um)
        for planned, face_id in matched.items():
            copies.setdefault(planned, {})[name] = face_id
    return copies


def imported_face(face: Any) -> ImportedFace:
    """One PyAEDT face as the matcher reads it.

    A planar face's centre is AEDT's own ``GetFaceCenter``; a curved face has none (AEDT refuses it), so its
    vertex centroid stands there and the matcher compares its vertices instead.
    """
    vertices = tuple(_point(vertex.position) for vertex in face.vertices)
    planar = bool(face.is_planar)
    centre = face.center_from_aedt if planar else _mean(vertices)
    if centre is False or centre is None:
        raise SgbImportError(f"AEDT gives no centre for planar face {face.id}")
    return ImportedFace(
        face_id=int(face.id),
        area_um2=float(face.area),
        centre_um=_point(centre),
        vertices_um=vertices,
        perimeter_um=sum(float(edge.length) for edge in face.edges),
        planar=planar,
    )


def _rebuild_free_faces(
    app: Any, free_faces: Sequence[FreeFace], tolerance_um: float
) -> dict[str, str]:
    """Planned free face id -> the AEDT sheet rebuilt from its guarded loops, named by its group."""
    names = aedt_names(
        [(face["face"], face["group"] or "sheet") for face in free_faces]
    )
    sheets: dict[str, str] = {}
    for face in free_faces:
        outer, *holes = (
            _loop_sheet(app, loop, f"{names[face['face']]}__loop{k}")
            for k, loop in enumerate(face["loops"])
        )
        if holes:
            app.modeler.subtract(outer, holes, keep_originals=False)
        sheet = app.modeler[outer]
        sheet.name = names[face["face"]]
        perimeter = sum(segment_length_um(s) for loop in face["loops"] for s in loop)
        limit = area_tolerance_um2(face["area_um2"], perimeter, tolerance_um)
        area = float(sheet.faces[0].area)
        if len(sheet.faces) != 1 or abs(area - face["area_um2"]) > limit:
            raise SgbImportError(
                f"rebuilt sheet {sheet.name} has {len(sheet.faces)} faces of area {area} um2, "
                f"the guard planned one of {face['area_um2']} um2"
            )
        sheets[face["face"]] = sheet.name
    return sheets


def _loop_sheet(app: Any, loop: Sequence[LoopSegment], name: str) -> str:
    points, kinds = polyline_of_loop(loop)
    from ansys.aedt.core.modeler.cad.polylines import (  # noqa: PLC0415 -- PyAEDT is optional and loaded only inside a live AEDT session.
        PolylineSegment,
    )

    sheet = app.modeler.create_polyline(
        [list(point) for point in points],
        segment_type=[PolylineSegment(kind) for kind in kinds],
        cover_surface=True,
        close_surface=True,
        name=name,
    )
    if not sheet:
        raise SgbImportError(f"AEDT did not draw the free-face loop {name}")
    return sheet.name


def polyline_of_loop(
    loop: Sequence[LoopSegment],
) -> tuple[list[Point], list[Literal["Line", "Arc"]]]:
    """The points and segment kinds of a closed AEDT polyline drawing ``loop``.

    A segment starts where the previous one ended, so each adds its points but the first; an arc adds its
    mid and end point.  The last point is the first one again.
    """
    points = [_point(loop[0]["points_um"][0])]
    kinds: list[Literal["Line", "Arc"]] = []
    for segment in loop:
        if _point(segment["points_um"][0]) != points[-1]:
            raise ValueError("a free-face loop is not connected end to start")
        points += [_point(p) for p in segment["points_um"][1:]]
        kinds.append("Line" if segment["kind"] == "line" else "Arc")
    if points[-1] != points[0]:
        raise ValueError("a free-face loop does not close")
    return points, kinds


def segment_length_um(segment: LoopSegment) -> float:
    """The length of one loop segment: a line, or the circular arc through its start, mid and end point."""
    points = [_point(p) for p in segment["points_um"]]
    if segment["kind"] == "line":
        return math.dist(points[0], points[1])
    start, mid, end = points
    a, b, c = math.dist(start, mid), math.dist(mid, end), math.dist(start, end)
    radius = (
        a * b * c / math.sqrt((a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c))
    )
    # The inscribed angle at ``mid`` subtends the chord ``start``-``end``; the arc is the side holding mid.
    half = math.asin(min(1.0, c / (2 * radius)))
    obtuse = (
        a * a + b * b < c * c
    )  # mid sees the chord at more than 90 degrees: the minor arc
    return radius * (2 * half if obtuse else 2 * (math.pi - half))


# ---------------------------------------------------------------- materials and Region


def _assign_materials(
    app: Any,
    model: PreparedSgbModel,
    sidecar: Sidecar,
    objects: Mapping[str, str],
    family: Family,
) -> list[dict[str, Any]]:
    """Realize each imported solid's material: its stack dielectric, or PEC for a conductor (Route C)."""
    records = []
    for solid_id, name in objects.items():
        volume = sidecar["volume"][solid_id]
        obj = app.modeler[name]
        if volume["kind"] == "conductor":
            # A Route C conductor interior is the solid itself, PEC in both families (decision 7); HFSS
            # solves only outside it.
            obj.material_name = "pec"
            if family == "hfss":
                obj.solve_inside = False
            native = "pec"
            realized: dict[str, Any] = {"native_name": "pec"}
        elif volume["kind"] == "dielectric":
            realized = realize_source_material(
                app, volume["material"], model.materials[volume["material"]]
            )
            native = realized["native_name"]
            obj.material_name = native
        else:
            raise SgbImportError(f"{name} is a {volume['kind']} solid, not imported")
        observed = native_object_property(obj, "Material").strip('"')
        if observed.casefold() != native.casefold():
            raise SgbImportError(f"{name} has material {observed!r}, not {native!r}")
        records.append(
            {
                "object_name": name,
                "solid": solid_id,
                "kind": volume["kind"],
                "material_id": volume["material"],
                "net": volume["net"],
                "native_material": observed,
                "realization": realized,
            }
        )
    return records


def _create_region(
    app: Any,
    model: PreparedSgbModel,
    sidecar: Sidecar,
    family: Family,
    tolerance_um: float,
) -> dict[str, Any]:
    """The Region of the plan's ``VACUUM_REGION`` box, of the vacuum's material, and (HFSS) its outer boundary."""
    x_min, y_min, z_min, x_max, y_max, z_max = model.vacuum_bounds_um
    region = app.modeler.create_region(
        pad_value=[x_max, x_min, y_max, y_min, z_max, z_min],
        pad_type="Absolute Position",
        name=REGION_NAME,
    )
    if not region or region.name != REGION_NAME:
        raise SgbImportError("AEDT did not create the Region")
    observed = tuple(float(value) for value in region.bounding_box)
    if any(
        abs(a - b) > tolerance_um
        for a, b in zip(observed, model.vacuum_bounds_um, strict=True)
    ):
        raise SgbImportError(
            f"the Region spans {list(observed)} um, not the VACUUM_REGION box "
            f"{list(model.vacuum_bounds_um)} um"
        )
    # The vacuum may be several volumes (a substrate spanning the box splits it above and below); the Region
    # is one box, so they must share one material.
    vacuum = {
        v["material"] for v in sidecar["volume"].values() if v["kind"] == "vacuum"
    }
    if len(vacuum) != 1:
        raise SgbImportError(
            f"the model's vacuum volumes have materials {sorted(vacuum)}, not one"
        )
    (material_id,) = vacuum
    realized = realize_source_material(app, material_id, model.materials[material_id])
    region.material_name = realized["native_name"]
    material = native_object_property(region, "Material").strip('"')
    if material.casefold() != realized["native_name"].casefold():
        raise SgbImportError(f"the Region has material {material!r}")
    record: dict[str, Any] = {
        "object_name": REGION_NAME,
        "bounds_um": list(observed),
        "material_id": material_id,
        "native_material": material,
        "realization": realized,
    }
    if family == "hfss":
        if not native_object_boolean_property(region, "Solve Inside"):
            raise SgbImportError("the Region must solve inside")
        record["outer_boundary"] = _assign_outer_perfect_h(app, region)
    return record


def _assign_outer_perfect_h(app: Any, region: Any) -> dict[str, Any]:
    """Perfect H on the six Region faces: Palace's natural PMC exterior, made explicit in HFSS."""
    face_ids = sorted(int(face.id) for face in region.faces)
    if len(face_ids) != 6:
        raise SgbImportError(f"the Region has {len(face_ids)} faces, not six")
    boundary = app.assign_perfect_h(face_ids, name=OUTER_BOUNDARY_NAME)
    if not boundary or native_boundary_type(app, OUTER_BOUNDARY_NAME) != "Perfect H":
        raise SgbImportError("AEDT did not assign Perfect H on the Region faces")
    return {"boundary": OUTER_BOUNDARY_NAME, "type": "Perfect H", "faces": face_ids}


# ---------------------------------------------------------------- Q3D conductor sheets (Route A)


def rebuild_conductor_sheets(
    app: Any, model: PreparedSgbModel, sidecar: Sidecar, imported: ImportedSgbModel
) -> list[dict[str, Any]]:
    """Each ``SHEET`` group as one sheet object, a PEC thin conductor of its film's thickness; Route C has none.

    Q3D puts a Net on whole objects, not on faces of the substrate, so each group's faces are copied from the
    solid they lie on into one object named by the group (:func:`sheet_object_names`), and that object is a
    ``ThinConductor`` of the stack layer's thickness (``PreparedSgbModel.sheet_thickness_um``).
    """
    kinds = {
        name: sidecar["volume"][solid]["kind"]
        for solid, name in imported.objects.items()
    }
    members = views.group_members(sidecar)
    records = []
    for group, name in sheet_object_names(
        sidecar, set(imported.objects.values())
    ).items():
        faces = [
            _sheet_face_copy(group, planned, imported.face_copies[planned], kinds)
            for planned in members[group]
        ]
        copy_faces_to_sheet(app, f"sheet group {group}", faces, name, non_model=False)
        records.append(
            _assign_thin_conductor(app, group, name, model.sheet_thickness_um[group])
        )
    return records


def _sheet_face_copy(
    group: str, planned: str, copies: Mapping[str, int], kinds: Mapping[str, str]
) -> tuple[str, int]:
    """The copy of a sheet face the sheet is rebuilt from: its only copy, or the dielectric's under a body."""
    if len(copies) == 1:
        ((object_name, face_id),) = copies.items()
        return object_name, face_id
    dielectric = [(o, f) for o, f in copies.items() if kinds.get(o) == "dielectric"]
    if len(dielectric) != 1:
        raise SgbImportError(
            f"face {planned} of sheet group {group!r} has copies {dict(copies)}, not one dielectric copy"
        )
    return dielectric[0]


def _assign_thin_conductor(
    app: Any, group: str, name: str, thickness_um: float
) -> dict[str, Any]:
    boundary_name = f"{name}ThinConductor"
    thickness = f"{thickness_um:.17g}um"
    boundary = app.assign_thin_conductor(
        assignment=name, material="pec", thickness=thickness, name=boundary_name
    )
    if not boundary or boundary.name != boundary_name:
        raise SgbImportError(f"AEDT did not assign the thin conductor of {name!r}")
    child = boundary.child_object
    raw = child.GetPropValue("Material")
    # AEDT 2024.2 reports the thin conductor's material as ``["Material:=", "pec"]``.
    if isinstance(raw, (list, tuple)) and len(raw) == 2 and raw[0] == "Material:=":
        raw = raw[1]
    material = str(raw).strip('"')
    observed = _length_um(child.GetPropValue("Thickness"))
    assigned = [
        int(value) for value in app.oboundary.GetExcitationAssignment(boundary_name)
    ]
    if (
        material.casefold() != "pec"
        or abs(observed - thickness_um) > 1e-9 * thickness_um
        or assigned != [int(app.modeler[name].id)]
    ):
        raise SgbImportError(
            f"the thin conductor of {name!r} reads back as {material!r}, {observed} um on {assigned}"
        )
    return {
        "group": group,
        "object_name": name,
        "boundary": boundary_name,
        "material": material,
        "thickness_um": observed,
    }


def _length_um(value: Any) -> float:
    from ansys.aedt.core.generic.numbers_utils import (  # noqa: PLC0415 -- PyAEDT is optional and loaded only inside a live AEDT session.
        Quantity,
    )

    quantity = Quantity(value)
    if quantity.unit_system != "Length":
        raise SgbImportError(f"{value!r} is not a length")
    return float(quantity.to("um").value)


def copy_faces_to_sheet(
    app: Any,
    label: str,
    faces: Sequence[tuple[str, int]],
    name: str,
    *,
    non_model: bool,
) -> None:
    """Copy ``faces`` into one sheet named ``name``; its area must be theirs.  ``label`` names the owner."""
    areas = {
        (object_name, int(face.id)): float(face.area)
        for object_name in {object_name for object_name, _ in faces}
        for face in app.modeler[object_name].faces
    }
    expected = sum(areas[face] for face in faces)
    created = app.modeler.create_object_from_face(
        [face_id for _, face_id in faces], non_model=non_model
    )
    sheets = created if isinstance(created, list) else [created]
    names = [sheet.name for sheet in sheets if sheet]
    if len(names) != len(faces):
        raise SgbImportError(
            f"AEDT copied {len(names)} of the {len(faces)} faces of {label}"
        )
    if len(names) > 1 and not app.modeler.unite(names):
        raise SgbImportError(f"AEDT did not unite the faces of {label}")
    sheet = app.modeler[names[0]]
    sheet.name = name
    area = sum(float(face.area) for face in sheet.faces)
    # The copies are AEDT's own faces, so their areas agree to floating-point noise.
    if sheet.name != name or abs(area - expected) > 1e-9 * expected:
        raise SgbImportError(
            f"the sheet of {label} is {sheet.name!r} of {area} um2, not {name!r} of {expected} um2"
        )


# ---------------------------------------------------------------- HFSS conductors and junctions


def assign_sheet_perfect_e(
    app: Any, plan: HfssBoundaryPlan, face_copies: Mapping[str, Mapping[str, int]]
) -> list[dict[str, Any]]:
    """One Perfect E per Net over every copy of its Route A sheet faces, under the plan's name; Route C has none."""
    records = []
    for net, (name, planned_faces) in plan.perfect_e.items():
        faces = sorted(
            face_id
            for planned in planned_faces
            for face_id in face_copies[planned].values()
        )
        boundary = app.assign_perfect_e(faces, name=name)
        if not boundary or native_boundary_type(app, name) != "Perfect E":
            raise SgbImportError(f"AEDT did not assign Perfect E for Net {net!r}")
        records.append({"boundary": name, "net": net, "faces": faces})
    return records


# ---------------------------------------------------------------- small vector helpers


def _point(values: Sequence[Any]) -> Point:
    x, y, z = (float(value) for value in values)
    return (x, y, z)


def _mean(points: Sequence[Point]) -> Point:
    return _point([sum(p[k] for p in points) / len(points) for k in range(3)])
