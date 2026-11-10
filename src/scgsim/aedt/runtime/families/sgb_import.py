"""Import the guarded SGB STEP into an empty HFSS or Q3D design and make the assignments its sidecar names.

AEDT models the geometry Palace meshes (issue #67).  :func:`import_sgb_model` runs, in order, and fails
before any assignment when what AEDT imported is not the guarded model:

1. check the prepared files again and read the guard (:meth:`PreparedSgbModel.verify_files`);
2. import the STEP into the empty design; the objects must be exactly the guard's solids;
3. match every imported face to its planned face by geometry (:func:`.sgb_faces.match_solid_faces`);
4. rebuild the guard's free faces (sheets with no imported solid on either side) as sheet objects from
   their loops, the only faces STEP cannot carry;
5. realize the materials: a dielectric solid takes its stack material, a conductor solid (Route C) is PEC;
6. create the Region from the plan's ``VACUUM_REGION`` box with the vacuum's material.  HFSS gets Perfect H
   on the Region's faces: Palace assigns nothing to its exterior, which is its natural PMC condition, and
   the two realizations must have the same outer boundary condition;
7. HFSS: one Perfect E per Net over the Route A sheet faces, and a Lumped RLC on the port face of each
   junction and of each generic ``LumpedRlc`` (issue #67 decision 8).  Q3D: one Net per sidecar Net over its
   conductor solids.

Every port sheet must carry a junction or an RLC.  Lumped Terminal ports wait for the SGB flow's Driven
Terminal design, and a model that has them is refused, not approximated.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from scgsim.aedt.preparation.sgb_model import (
    PreparedSgbModel,
    SgbJunction,
    port_group,
)
from scgsim.aedt.rlc import rlc_native_topology
from scgsim.aedt.runtime.families.sgb_faces import (
    ImportedFace,
    area_tolerance_um2,
    import_tolerance_um,
    match_solid_faces,
)
from scgsim.aedt.runtime.native.common import (
    native_boundary_type,
    native_object_boolean_property,
    native_object_property,
    realize_source_material,
)
from scgsim.aedt.specs.common import LumpedRlc
from scgsim.geometry.sgb.exchange.guard import (
    FreeFace,
    LoopSegment,
    StepGuard,
    read_step_guard,
)
from scgsim.geometry.sgb.sidecar.aedt import aedt_names, sanitise_aedt_name
from scgsim.geometry.sgb.sidecar.files import read_sidecar

Family = Literal["hfss", "q3d"]
Point = tuple[float, float, float]
REGION_NAME = "Region"
OUTER_BOUNDARY_NAME = "outer_perfect_h"


class SgbImportError(RuntimeError):
    """The AEDT design does not hold the guarded SGB model, or an assignment did not take."""


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
    roles = q3d_net_roles(model, sidecar) if family == "q3d" else {}
    # AEDT's readback is coarser than the sidecar's tolerance_um (decision 9), so every comparison with what
    # AEDT reports uses a thousandth of the bound model's own grid.
    tolerance_um = import_tolerance_um(float(sidecar["grid_um"]))
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
    if family == "hfss":
        imported.record["perfect_e"] = _assign_sheet_perfect_e(
            app, sidecar, face_copies
        )
        imported.record["junctions"] = [
            _assign_junction_rlc(app, sidecar, face_copies, item, tolerance_um)
            for item in model.junctions
        ]
        imported.record["lumped_rlcs"] = [
            _assign_generic_rlc(app, sidecar, face_copies, item, tolerance_um)
            for item in model.lumped_rlcs
        ]
    else:
        imported.record["nets"] = _assign_q3d_nets(app, sidecar, objects, roles)
    imported.record["readback_tolerance_um"] = tolerance_um
    imported.record["model_identity_sha256"] = model.identity_sha256
    return imported


def _check_supported(model: PreparedSgbModel, family: Family) -> None:
    """Refuse what the import cannot realize yet rather than realize it partly."""
    if family not in ("hfss", "q3d"):
        raise ValueError(f"family must be 'hfss' or 'q3d', got {family!r}")
    if family == "q3d" and model.route == "A":
        # A Q3D sheet conductor needs its thin-conductor thickness, which the SGB model does not carry yet.
        raise NotImplementedError("Q3D imports Route C models only for now")
    if family == "q3d" and (model.junctions or model.lumped_rlcs):
        raise ValueError(
            "Q3D has no junctions or lumped RLCs; they belong to HFSS Eigenmode"
        )
    if model.lumped_ports:
        names = [port.name for port in model.lumped_ports]
        raise NotImplementedError(
            f"lumped Terminal ports {names} need the SGB flow's Driven Terminal design, "
            "which this import does not set up"
        )


def _check_ports(
    model: PreparedSgbModel, sidecar: Mapping[str, Any], family: Family
) -> None:
    """In HFSS every lumped-port sheet must carry a junction or an RLC; an untreated one has no AEDT meaning."""
    if family != "hfss":
        return
    bound = {item.port_group for item in model.junctions} | {
        port_group(item.port_sheet_id) for item in model.lumped_rlcs
    }
    unbound = sorted(p["group"] for p in sidecar["ports"] if p["group"] not in bound)
    if unbound:
        raise ValueError(
            f"port sheets {unbound} carry no junction or LumpedRlc; give each one its treatment"
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
    sidecar: Mapping[str, Any],
    objects: Mapping[str, str],
    family: Family,
) -> list[dict[str, Any]]:
    """Realize each imported solid's material: its stack dielectric, or PEC for a conductor (Route C)."""
    records = []
    for solid_id, name in objects.items():
        volume = sidecar["volumes"][solid_id]
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
    sidecar: Mapping[str, Any],
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
    vacuum = [v for v in sidecar["volumes"].values() if v["kind"] == "vacuum"]
    if len(vacuum) != 1:
        raise SgbImportError(f"the model has {len(vacuum)} vacuum volumes, not one")
    material_id = vacuum[0]["material"]
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


# ---------------------------------------------------------------- HFSS conductors and junctions


def _assign_sheet_perfect_e(
    app: Any, sidecar: Mapping[str, Any], face_copies: Mapping[str, Mapping[str, int]]
) -> list[dict[str, Any]]:
    """One Perfect E per Net over every copy of its Route A sheet faces; a Route C model has none."""
    groups = sidecar["physical_groups"]
    records = []
    for net, net_groups in sorted(sidecar["unions"]["nets"].items()):
        faces = sorted(
            face_id
            for group in net_groups
            if groups[group]["interface_type"] == "SHEET"
            for planned in groups[group]["ids"]
            for face_id in face_copies[planned].values()
        )
        if not faces:
            continue
        name = f"perfect_e_{sanitise_aedt_name(net)}"
        boundary = app.assign_perfect_e(faces, name=name)
        if not boundary or native_boundary_type(app, name) != "Perfect E":
            raise SgbImportError(f"AEDT did not assign Perfect E for Net {net!r}")
        records.append({"boundary": name, "net": net, "faces": faces})
    return records


def _assign_junction_rlc(
    app: Any,
    sidecar: Mapping[str, Any],
    face_copies: Mapping[str, Mapping[str, int]],
    item: SgbJunction,
    tolerance_um: float,
) -> dict[str, Any]:
    """The junction's parallel Lumped RLC (its inductance and capacitance) on its port face."""
    junction = item.junction
    port = _sidecar_port(sidecar, item.port_group)
    declared = _unit((*junction.direction_xy, 0.0))
    if math.dist(_unit(port["direction"]), declared) > 1e-9:
        raise SgbImportError(
            f"junction {junction.junction_id!r} runs along {junction.direction_xy}, its port sheet "
            f"along {port['direction']}"
        )
    branches: dict[str, Any] = {
        "RLC Type": "Parallel",
        "UseResist": False,
        "UseInduct": True,
        "Inductance": f"{junction.inductance_h}H",
        "UseCap": bool(junction.capacitance_f),
    }
    if junction.capacitance_f:
        branches["Capacitance"] = f"{junction.capacitance_f}F"
    name = f"junction_rlc_{sanitise_aedt_name(junction.junction_id)}"
    record = _assign_port_rlc(app, port, face_copies, name, branches, tolerance_um)
    return {
        "junction_id": junction.junction_id,
        **record,
        "inductance_h": junction.inductance_h,
        "capacitance_f": junction.capacitance_f,
    }


def _assign_generic_rlc(
    app: Any,
    sidecar: Mapping[str, Any],
    face_copies: Mapping[str, Mapping[str, int]],
    treatment: LumpedRlc,
    tolerance_um: float,
) -> dict[str, Any]:
    """A ``LumpedRlc`` on the port face it names: exactly its enabled branches, in its topology."""
    port = _sidecar_port(sidecar, port_group(treatment.port_sheet_id))
    name = f"rlc_{sanitise_aedt_name(treatment.port_sheet_id)}"
    branches = rlc_branches(treatment)
    record = _assign_port_rlc(app, port, face_copies, name, branches, tolerance_um)
    return {
        "port_sheet_id": treatment.port_sheet_id,
        **record,
        "requested": treatment.to_payload(),
    }


def rlc_branches(treatment: LumpedRlc) -> dict[str, Any]:
    """The native Lumped RLC properties of ``treatment``: its topology and exactly its enabled branches.

    ``None`` alone disables a branch, so an authored zero stays an enabled branch of value zero.
    """
    branches: dict[str, Any] = {"RLC Type": rlc_native_topology(treatment.topology)}
    for enabled, key, value, unit in (
        ("UseResist", "Resistance", treatment.resistance_ohm, "ohm"),
        ("UseInduct", "Inductance", treatment.inductance_h, "H"),
        ("UseCap", "Capacitance", treatment.capacitance_f, "F"),
    ):
        branches[enabled] = value is not None
        if value is not None:
            branches[key] = f"{value}{unit}"
    return branches


def _sidecar_port(sidecar: Mapping[str, Any], group: str) -> Mapping[str, Any]:
    ports = [p for p in sidecar["ports"] if p["group"] == group]
    if len(ports) != 1:
        raise SgbImportError(
            f"the sidecar has {len(ports)} lumped ports {group!r}, not one"
        )
    return ports[0]


def _assign_port_rlc(  # noqa: PLR0913 -- the port, its face copies, the boundary name and its branches
    app: Any,
    port: Mapping[str, Any],
    face_copies: Mapping[str, Mapping[str, int]],
    name: str,
    branches: Mapping[str, Any],
    tolerance_um: float,
) -> dict[str, Any]:
    """One Lumped RLC on a port's face, its current line read from that face's own edges along ``direction``.

    A face two solids share imports twice, about 4e-5 um apart on arcs (E4), so the line ends come from the
    imported face itself, never from the sidecar's coordinates.
    """
    (planned,) = port["faces"]
    copies = face_copies[planned]
    if len(copies) != 1:
        raise SgbImportError(
            f"port face {planned} has {len(copies)} imported copies, not one"
        )
    ((object_name, face_id),) = copies.items()
    face = next(f for f in app.modeler[object_name].faces if int(f.id) == face_id)
    edges = [
        tuple(_point(vertex.position) for vertex in edge.vertices)
        for edge in face.edges
    ]
    start, end = current_line_um(edges, _unit(port["direction"]), tolerance_um)
    props = {
        "Faces": [face_id],
        "CurrentLine": {
            "Start": [f"{value}um" for value in start],
            "End": [f"{value}um" for value in end],
        },
        **branches,
    }
    # PyAEDT's Lumped RLC helpers take sheet names only and refuse all-falsy values; the native writer takes
    # the face id (E4) and keeps an authored zero branch.
    boundary = app._create_boundary(name, props, "Lumped RLC")  # noqa: SLF001 -- the pinned native writer is the only face-id route.
    if not boundary or native_boundary_type(app, name) != "Lumped RLC":
        raise SgbImportError(f"AEDT did not assign the Lumped RLC {name!r}")
    return {
        "boundary": name,
        "port_group": port["group"],
        "object_name": object_name,
        "face_id": face_id,
        "host_entity_ids": list(port["host_entity_ids"]),
        "current_line_um": [list(start), list(end)],
        "branches": dict(branches),
    }


def current_line_um(
    edges: Sequence[Sequence[Point]], direction: Point, tolerance_um: float
) -> tuple[Point, Point]:
    """Start and end of the current line across a port face along ``direction``.

    At each end of the face's extent along ``direction`` its straight edges there must form one unbroken
    segment across it (SGB may split it where another face meets the port); the line joins the midpoints of
    the two segments, from the lower end to the upper.  The sidecar's ``direction`` points from the first host
    conductor to the second, so the line does too.
    """
    vertices = [point for edge in edges for point in edge]
    low = min(_dot(point, direction) for point in vertices)
    high = max(_dot(point, direction) for point in vertices)
    start, end = (
        _port_end_midpoint(edges, direction, extreme, tolerance_um)
        for extreme in (low, high)
    )
    return start, end


def _port_end_midpoint(
    edges: Sequence[Sequence[Point]],
    direction: Point,
    extreme: float,
    tolerance_um: float,
) -> Point:
    """The midpoint of the one unbroken segment the face's edges at ``extreme`` along ``direction`` form."""
    across = [
        edge
        for edge in edges
        if len(edge) == 2
        and all(abs(_dot(p, direction) - extreme) <= tolerance_um for p in edge)
    ]
    points = [p for edge in across for p in edge]
    if not across:
        raise SgbImportError(
            f"the port face has no edge across its direction at {extreme} um"
        )
    a, b = max(
        ((p, q) for p in points for q in points), key=lambda pair: math.dist(*pair)
    )
    covered = sum(math.dist(*edge) for edge in across)
    if abs(covered - math.dist(a, b)) > len(across) * tolerance_um:
        raise SgbImportError(
            f"the port face's edges at {extreme} um along its direction do not form one segment"
        )
    return _point([(a[k] + b[k]) / 2 for k in range(3)])


# ---------------------------------------------------------------- Q3D


def q3d_net_roles(
    model: PreparedSgbModel, sidecar: Mapping[str, Any]
) -> dict[str, Literal["Signal", "Ground"]]:
    """Each conductor Net's Q3D role from the model's ground declaration: Ground if declared, else Signal."""
    if model.ground_net_ids is None:
        raise ValueError(
            "the SGB model declares no ground Nets, so the Q3D Net roles are unknown; pass "
            "ground_net_ids to prepare_sgb_model (an empty sequence when no Net is ground)"
        )
    nets = {
        volume["net"]
        for volume in sidecar["volumes"].values()
        if volume["kind"] == "conductor"
    }
    missing = sorted(set(model.ground_net_ids) - nets)
    if missing:
        raise ValueError(f"ground Nets {missing} have no conductor solid in the model")
    return {
        net: "Ground" if net in model.ground_net_ids else "Signal"
        for net in sorted(nets)
        if net is not None
    }


def _assign_q3d_nets(
    app: Any,
    sidecar: Mapping[str, Any],
    objects: Mapping[str, str],
    roles: Mapping[str, Literal["Signal", "Ground"]],
) -> list[dict[str, Any]]:
    """One Net per sidecar Net over its conductor solids, in Net order, with its declared role."""
    by_net: dict[str, list[str]] = {}
    for solid_id, name in objects.items():
        volume = sidecar["volumes"][solid_id]
        if volume["kind"] == "conductor":
            if volume["net"] is None:
                raise SgbImportError(f"conductor {name} has no Net")
            by_net.setdefault(volume["net"], []).append(name)
    records = []
    for net, names in sorted(by_net.items()):
        net_name = sanitise_aedt_name(net)
        boundary = app.assign_net(names, net_name, roles[net])
        assigned = sorted(
            int(value) for value in app.oboundary.GetExcitationAssignment(net_name)
        )
        expected = sorted(int(app.modeler[name].id) for name in names)
        if not boundary or assigned != expected:
            raise SgbImportError(f"AEDT did not assign Net {net!r} to {names}")
        records.append(
            {"net": net, "name": net_name, "role": roles[net], "objects": names}
        )
    return records


# ---------------------------------------------------------------- small vector helpers


def _point(values: Sequence[Any]) -> Point:
    x, y, z = (float(value) for value in values)
    return (x, y, z)


def _mean(points: Sequence[Point]) -> Point:
    return _point([sum(p[k] for p in points) / len(points) for k in range(3)])


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _unit(values: Sequence[float]) -> Point:
    vector = _point(values)
    length = math.sqrt(_dot(vector, vector))
    if length == 0:
        raise ValueError("a direction must not be zero")
    return _point([value / length for value in vector])
