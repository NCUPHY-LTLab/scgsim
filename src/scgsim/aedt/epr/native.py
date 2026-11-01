"""Native AEDT selections that the EPR field integrals are taken over.

Owns: the non-model sheets and face lists on which surface integrals are
evaluated (inset sheets, whole-cap face references and per-binding analysis
sheets), the choice of field side for each, and the readback that proves a face
list was saved with the project.
Does not own: common model construction and saved geometry readback
(`runtime.native`), or the integrals themselves (`fields.py`, `cache.py`).

MA (metal-air), MS (metal-substrate) and SA (substrate-air) are defined in
docs/eigenmode-epr.qmd.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any

from scgsim.aedt._data import canonical_sha256, detached
from scgsim.aedt.epr.fields import create_verified_face_list
from scgsim.aedt.epr.geometry import base_sheet_plan
from scgsim.aedt.preparation.planar import (
    INSET_CIRCLE_POINTS,
    INSET_METHOD,
    entity_z_range,
    native_name,
)
from scgsim.aedt.runtime.native.common import (
    native_object_boolean_property,
    native_object_evidence,
)
from scgsim.aedt.runtime.native.hfss_bodies import (
    native_conductor_geometry as native_conductor_geometry,
)
from scgsim.aedt.runtime.native.hfss_junction_readback import (
    read_saved_hfss_design as read_saved_hfss_design,
)
from scgsim.aedt.specs.geometry import PreparedPlanarGeometry
from scgsim.geometry.primitives.spatial import geometry_ref_surface_z_um


def _analysis_surface_sheet(
    app: Any, geometry_ref: Mapping[str, Any], *, name: str
) -> tuple[Any, list[list[float]]]:
    """Create a non-model, hole-cut sheet for a planar loop; return it and its points."""

    local_region = geometry_ref.get("local_region")
    plane_basis = geometry_ref.get("plane_basis")
    outer = geometry_ref.get("outer_loop")
    plane = geometry_ref.get("plane")
    if isinstance(local_region, Mapping) and isinstance(plane_basis, Mapping):
        points = _lift_inset_ring(plane_basis, local_region["exterior"])
        holes = [
            _lift_inset_ring(plane_basis, ring)
            for ring in local_region.get("holes", ())
        ]
    elif isinstance(outer, Sequence) and not isinstance(outer, (str, bytes)):
        if (
            not isinstance(plane, Mapping) or plane.get("axis") != "z"
        ) and geometry_ref.get("shell_part") not in {"top", "bottom"}:
            raise ValueError("analysis surface loop requires an explicit Z plane")
        z_um = geometry_ref_surface_z_um(geometry_ref)
        points = [[float(x), float(y), z_um] for x, y in outer]
        holes = [
            [[float(x), float(y), z_um] for x, y in ring]
            for ring in geometry_ref.get("hole_loops", ())
        ]
    else:
        raise ValueError("horizontal analysis surface requires a planar loop")
    if len(points) < 3 or any(
        len(point) != 3 or not all(math.isfinite(value) for value in point)
        for point in points
    ):
        raise ValueError("analysis surface points are invalid")
    sheet = app.modeler.create_polyline(
        points,
        cover_surface=True,
        close_surface=True,
        name=name,
        non_model=True,
    )
    if sheet is False or sheet is None:
        raise RuntimeError(f"failed to create analysis surface {name!r}")
    for index, ring in enumerate(holes):
        hole_name = native_name("hole", name, index)
        hole = app.modeler.create_polyline(
            ring,
            cover_surface=True,
            close_surface=True,
            name=hole_name,
            non_model=True,
        )
        if hole is False or hole is None:
            raise RuntimeError(f"failed to create analysis hole {hole_name!r}")
        if not app.modeler.subtract(name, hole_name, keep_originals=False):
            raise RuntimeError(f"failed to subtract analysis hole {hole_name!r}")
    return app.modeler[name], points


def _native_sheet_side_binding(
    native_normal: Sequence[float],
    field_side: str,
    modeling: str | None,
    *,
    unknown_side_error: str,
    ambiguous_side_error: str,
) -> tuple[tuple[float, float, float], bool]:
    """Map a source side while limiting the empirical rule to ThinFilm."""

    if field_side == "top":
        desired = (0.0, 0.0, 1.0)
    elif field_side == "bottom":
        desired = (0.0, 0.0, -1.0)
    else:
        raise RuntimeError(unknown_side_error)
    orientation = sum(
        float(normal) * target
        for normal, target in zip(native_normal, desired, strict=False)
    )
    if abs(orientation) <= 1e-12:
        raise RuntimeError(ambiguous_side_error)
    if modeling == "thin_film":
        # Human-selected for pinned PyAEDT 1.3.0 / AEDT 2024.2: FacePrimitive.normal
        # points toward Adjacent. This is not a direct native arrow-vector accessor.
        adjacent_side = orientation > 0.0
    else:
        # Preserve the pre-existing Solid/historical convention; its physical
        # face-side correspondence has not been established by this change.
        adjacent_side = orientation < 0.0
    return desired, adjacent_side


def _lift_inset_ring(
    plane: Mapping[str, Any], ring: Sequence[Sequence[float]]
) -> list[list[float]]:
    """Map 2D (u, v) ring points onto the plane in 3D: origin + u*U + v*V."""

    origin = tuple(float(value) for value in plane["origin_um"])
    u = tuple(float(value) for value in plane["u"])
    v = tuple(float(value) for value in plane["v"])
    return [
        [
            origin[axis] + float(point[0]) * u[axis] + float(point[1]) * v[axis]
            for axis in range(3)
        ]
        for point in ring
    ]


@dataclass(slots=True)
class InsetSheetBook:
    """Planned inset sheets for one analysis, plus the facts read back so far.

    `facts` is filled as sheets are bound and reused within one authoring session;
    the field side is still decided per binding, never cached with the sheet.
    """

    plan: Mapping[tuple[str, float], Mapping[str, Any]]
    names: Mapping[str, Mapping[str, Any]]
    facts: dict[str, dict[str, Any]]
    modeling: str | None


def bind_inset_surface_selections(
    app: Any,
    binding: Mapping[str, Any],
    base_selection: Mapping[str, Any],
    margin_um: float,
    sheets: InsetSheetBook,
) -> list[dict[str, Any]]:
    """Create or rebind non-model inset sheets without changing solver CAD.

    An inset sheet is the part of a surface binding that lies `margin_um` away
    from the conductor edge (the planned region, computed with KLayout). One
    binding can yield several disjoint regions, so the result has one entry per
    region ("component").
    """

    plane = binding["mask_plane"]
    # Direction comes from this source contribution, never from the base Sheet's
    # orientation.
    field_side = str(binding["contribution"]["side"])
    planned = sheets.plan[(str(binding["binding_id"]), margin_um)]
    regions = planned["regions"]
    dbu_um = float(binding["mask_support"]["source_dbu_um"])
    result: list[dict[str, Any]] = []
    for index, region in enumerate(regions):
        contour_sha256 = canonical_sha256(
            {
                "plane": plane,
                "region": region,
                "method": INSET_METHOD,
                "source_dbu_um": dbu_um,
                "klayout_version": version("klayout"),
            }
        )
        planned_name = sheets.names[contour_sha256]
        name = str(planned_name["name"])
        facts = sheets.facts.get(name)
        if facts is None:
            facts = _read_inset_sheet(app, name, plane, region)
            facts["contour_sha256"] = contour_sha256
            facts["shared_sheet_members"] = planned_name["members"]
            sheets.facts[name] = facts
        if facts["contour_sha256"] != contour_sha256:
            raise RuntimeError(f"inset EPR selection name collision: {name!r}")
        native = facts["native"]
        native_normal = facts["native_normal"]
        desired, adjacent_side = _native_sheet_side_binding(
            native_normal,
            field_side,
            sheets.modeling,
            unknown_side_error=f"inset EPR selection {name!r} has unknown field side",
            ambiguous_side_error=(
                f"inset EPR selection {name!r} has ambiguous field side"
            ),
        )
        result.append(
            {
                "selection_name": name,
                "component_index": index,
                "contour_sha256": contour_sha256,
                "contour": region,
                "adjacent_side": adjacent_side,
                "native_normal": native_normal,
                "polygon_approximation": {
                    "method": INSET_METHOD,
                    "source_dbu_um": dbu_um,
                    "points_per_circle": INSET_CIRCLE_POINTS,
                    "requested_margin_um": margin_um,
                    "integer_margin_dbu": planned["integer_margin_dbu"],
                    "effective_margin_um": planned["effective_margin_um"],
                    "klayout_version": version("klayout"),
                },
                "reused_source_sheet": False,
                **native,
            }
        )
    return result


def _read_inset_sheet(
    app: Any, name: str, plane: Mapping[str, Any], region: Any
) -> dict[str, Any]:
    """Find or create one inset sheet and read back that it is a single sheet.

    The sheet must stay non-model geometry, so the solver CAD is unchanged.
    """

    sheet = app.modeler.get_object_from_name(name)
    if sheet is None:
        sheet, _ = _analysis_surface_sheet(
            app,
            {"plane_basis": plane, "local_region": region},
            name=name,
        )
    native = native_object_evidence(app, name)
    faces = sheet.faces
    if native["native_object_type"] != "Sheet" or len(faces) != 1:
        raise RuntimeError(f"inset EPR selection {name!r} is not one sheet")
    if native_object_boolean_property(sheet, "Model"):
        raise RuntimeError(f"inset EPR selection {name!r} became model geometry")
    normal = faces[0].normal
    if (
        normal is None
        or len(normal) != 3
        or not all(math.isfinite(float(value)) for value in normal)
    ):
        raise RuntimeError(f"inset EPR selection {name!r} has no native normal")
    return {
        "native": native,
        "native_normal": [float(value) for value in normal],
    }


def bind_native_surface_references(
    app: Any,
    prepared: PreparedPlanarGeometry,
    native_geometry: Mapping[str, Any],
    request: Any,
) -> list[dict[str, Any]]:
    """Bind whole Solid caps as references, separate from masked Sheet supports.

    A cap is the top or bottom face of a Solid conductor (MA, metal-air, and MS,
    metal-substrate) or of the substrate domain (SA, substrate-air); see
    docs/eigenmode-epr.qmd. Source ownership and constant-section Z construction
    select the cap. Native normals describe projection only; they do not establish
    an incident trace. Saved reuse changes selection metadata, never CAD or solved
    field topology.
    """
    if prepared.modeling != "solid":
        return []
    selected = (
        {item.contribution_id for item in prepared.contributions}
        if request.surface_contribution_ids is None
        else set(request.surface_contribution_ids)
    )
    specs = {item.contribution_id: item for item in prepared.contributions}
    references: dict[str, dict[str, Any]] = {}
    for binding in prepared.surface_bindings:
        contribution = binding["contribution"]
        contribution_id = contribution["contribution_id"]
        kind = contribution["classification"]
        if contribution_id not in selected or kind not in {"MA", "MS", "SA"}:
            continue
        z_um = geometry_ref_surface_z_um(binding["geometry_ref"])
        owner, entities, candidates = _source_owned_solids(
            prepared, native_geometry, binding, specs
        )
        if len(entities) != 1 or not candidates:
            raise RuntimeError(
                f"native reference {contribution_id!r} lacks source-owned Solid {owner!r}"
            )
        cap = _source_cap(prepared.source, entities[0], binding, z_um, owner)
        for native in candidates:
            face, center, normal = _native_cap_face(app, native, cap)
            identity = {
                "object_name": native["object_name"],
                "cap": cap,
                "projection_normal": normal,
            }
            reference_id = "native_cap_" + canonical_sha256(identity)[:24]
            record = references.get(reference_id)
            if record is None:
                record = _native_reference_record(
                    app, native, face, center, normal, reference_id, owner, cap, z_um
                )
                references[reference_id] = record
            _add_binding_support(record, binding)
    return detached([references[key] for key in sorted(references)])


def _source_owned_solids(
    prepared: PreparedPlanarGeometry,
    native_geometry: Mapping[str, Any],
    binding: Mapping[str, Any],
    specs: Mapping[str, Any],
) -> tuple[str, list[Any], list[Any]]:
    """Find the source entity that owns a binding and its native Solid objects.

    SA bindings belong to the substrate domain; MA and MS bindings to a conductor,
    possibly built from several source polygons (outer and hole polygons).
    """

    source = prepared.source
    objects = native_geometry["objects"]
    contribution = binding["contribution"]
    if contribution["classification"] == "SA":
        owner = binding["substrate_domain_id"]
        entities = [
            item for item in source["solution_regions"] if item["semantic_id"] == owner
        ]
        candidates = [
            item
            for item in objects
            if item["kind"] == "solution_domain" and item["semantic_id"] == owner
        ]
        return owner, entities, candidates
    owner = binding["owner_semantic_id"]
    entities = [item for item in source["conductors"] if item["semantic_id"] == owner]
    source_ids = {
        specs[contribution["contribution_id"]].source_polygon_id,
        *(contribution.get("outer_source_ids") or ()),
        *(contribution.get("hole_source_ids") or ()),
    }
    candidates = [
        item
        for item in objects
        if item["kind"] == "conductor"
        and item["semantic_id"] == owner
        and item["source_polygon_id"] in source_ids
    ]
    return owner, entities, candidates


def _source_cap(
    source: Mapping[str, Any],
    entity: Mapping[str, Any],
    binding: Mapping[str, Any],
    z_um: float,
    owner: str,
) -> str:
    """Return "top" or "bottom": which Z extreme of the owner the binding sits on."""

    contribution_id = binding["contribution"]["contribution_id"]
    if binding["contribution"]["classification"] == "SA":
        z_min, z_max = entity_z_range(entity)
    else:
        representation, z_min, z_max = native_conductor_geometry(source, entity)
        if representation != "solid":
            raise RuntimeError(
                f"native reference {contribution_id!r} owner is not a Solid"
            )
    if z_um == z_max and z_um != z_min:
        return "top"
    if z_um == z_min and z_um != z_max:
        return "bottom"
    raise RuntimeError(
        f"native reference {contribution_id!r} is not a source cap of {owner!r}"
    )


def _native_cap_face(
    app: Any, native: Mapping[str, Any], cap: str
) -> tuple[Any, list[float], list[float]]:
    """Pick the native face at the top or bottom extreme of a Solid.

    Returns the face, its center and its normal. The extreme face must be unique.
    """

    name = native["object_name"]
    obj = app.modeler.get_object_from_name(name)
    if obj is None or native["native_object_type"] != "Solid":
        raise RuntimeError(f"native reference Solid {name!r} is unavailable")
    centers = []
    for face in obj.faces:
        center = face.center_from_aedt
        if center is False or center is None or len(center) != 3:
            raise RuntimeError(
                f"native reference {name!r} face {face.id} center unavailable"
            )
        centers.append((float(center[2]), face, list(center)))
    if not centers:
        raise RuntimeError(f"native reference {name!r} has no faces")
    extreme = (max if cap == "top" else min)(item[0] for item in centers)
    matches = [item for item in centers if item[0] == extreme]
    if len(matches) != 1:
        raise RuntimeError(f"native reference {name!r} {cap} cap is ambiguous")
    _, face, center = matches[0]
    normal = face.normal
    if normal is None or len(normal) != 3:
        raise RuntimeError(f"native reference {name!r} cap projection unavailable")
    return face, center, [float(value) for value in normal]


def _face_edges(face: Any, name: str) -> list[dict[str, Any]]:
    """Describe a face's edges and vertices as evidence for the reference."""

    edges = []
    for edge in face.edges:
        vertices = []
        samples = []
        for vertex in edge.vertices:
            position = vertex.position
            if int(vertex.id) <= 0:
                # PyAEDT can synthesize an edge-parameter sample when
                # the native edge has no vertex IDs (e.g. a closed edge).
                samples.append(
                    {
                        "basis": "PyAEDT synthesized edge-parameter sample",
                        "position_um": list(position) if position is not None else None,
                        "native_vertex_id": None,
                    }
                )
                continue
            if position is None or len(position) != 3:
                raise RuntimeError(
                    f"native reference {name!r} vertex {vertex.id} unavailable"
                )
            vertices.append(
                {"vertex_id": int(vertex.id), "position_um": list(position)}
            )
        edges.append(
            {
                "edge_id": int(edge.id),
                "vertices": vertices,
                "native_vertex_identity_available": bool(vertices) and not samples,
                "parametric_samples": samples,
            }
        )
    return edges


def _native_reference_record(  # noqa: PLR0913 -- one argument per recorded fact.
    app: Any,
    native: Mapping[str, Any],
    face: Any,
    center: list[float],
    normal: list[float],
    reference_id: str,
    owner: str,
    cap: str,
    z_um: float,
) -> dict[str, Any]:
    """Create the face list for one cap and the reference record that cites it.

    Contribution, binding and support lists start empty and are filled by
    `_add_binding_support` for every binding that shares this cap.
    """

    name = native["object_name"]
    edges = _face_edges(face, name)
    area = face.area
    if area is None or area is False:
        raise RuntimeError(f"native reference {name!r} cap area unavailable")
    face_ids = [int(face.id)]
    face_list = create_verified_face_list(
        app,
        name=native_name("reference", reference_id),
        face_ids=face_ids,
        reuse_existing=True,
    )
    return {
        "reference_id": reference_id,
        "contribution_ids": [],
        "binding_ids": [],
        "interface_kinds": [],
        "selection_name": face_list["name"],
        "face_ids": face_ids,
        "face_list": face_list,
        "owner_semantic_id": owner,
        "source_polygon_id": native.get("source_polygon_id"),
        "object_name": name,
        "native_object_id": native["native_object_id"],
        "source_cap": cap,
        "source_z_um": z_um,
        "source_supports": [],
        "incident_domains": [],
        "native_normal": normal,
        "projection_normal": normal,
        "native_normal_basis": "PyAEDT FacePrimitive.normal geometric projection",
        # PyAEDT reports the face area in um^2.
        "native_area_m2": float(area) * 1e-12,
        "native_faces": [
            {
                "face_id": int(face.id),
                "center_um": center,
                "edges": edges,
                "area_um2": float(area),
            }
        ],
        "support_kind": "whole_native_cap",
        "sampling_basis": "native_solid_owner_face; incident_trace_unproven",
    }


def _add_binding_support(record: dict[str, Any], binding: Mapping[str, Any]) -> None:
    """Record that this binding's surface is supported by the reference's cap."""

    contribution = binding["contribution"]
    for key, value in (
        ("contribution_ids", contribution["contribution_id"]),
        ("binding_ids", binding["binding_id"]),
        ("interface_kinds", contribution["classification"]),
    ):
        if value not in record[key]:
            record[key].append(value)
    record["source_supports"].append(
        {
            "binding_id": binding["binding_id"],
            "geometry_ref": binding["geometry_ref"],
        }
    )
    incident = {
        "effective_domain_id": binding["effective_domain_id"],
        "effective_material_id": binding["effective_material_id"],
        "substrate_domain_id": binding["substrate_domain_id"],
    }
    if incident not in record["incident_domains"]:
        record["incident_domains"].append(incident)


def verify_saved_native_surface_references(
    project_path, design_name, references
) -> list[dict[str, Any]]:
    """Bind reference membership to the existing pre-Analyze native save.

    Created PyAEDT List properties hold the authored assignment. Only the saved
    GeometryEntityListOperation supplies native serialization membership here;
    this function neither saves nor changes the model or the reference support.
    """
    verified = detached(references)
    if not verified:
        return verified
    digest, design = read_saved_hfss_design(
        project_path, design_name, purpose="native surface reference"
    )
    setup = design.get("ModelSetup")
    core = setup.get("GeometryCore") if isinstance(setup, Mapping) else None
    operations = core.get("GeometryOperations") if isinstance(core, Mapping) else None
    lists = (
        operations.get("GeometryEntityLists")
        if isinstance(operations, Mapping)
        else None
    )
    records = (
        lists.get("GeometryEntityListOperation") if isinstance(lists, Mapping) else None
    )
    if isinstance(records, Mapping):
        records = [records]
    if not isinstance(records, list):
        raise RuntimeError(
            "saved native surface reference list records are unavailable"
        )
    for reference in verified:
        requested = reference["face_list"]
        name = requested["name"]
        matching = [
            record
            for record in records
            if isinstance(record, Mapping)
            and isinstance(record.get("Attributes"), Mapping)
            and record["Attributes"].get("Name") == name
        ]
        expected = {
            "name": name,
            "native_id": requested["native_id"],
            "entity_type": "Face",
            "face_ids": requested["face_ids"],
        }
        if len(matching) != 1:
            raise RuntimeError(
                f"saved native surface reference list missing or ambiguous: "
                f"requested={expected!r}; matching_records={matching!r}"
            )
        record = matching[0]
        parameters = record.get("GeometryEntityListParameters")
        actual = {
            "name": record["Attributes"]["Name"],
            "native_id": record.get("ID"),
            "entity_type": parameters.get("EntityType")
            if isinstance(parameters, Mapping)
            else None,
            "face_ids": parameters.get("EntityList")
            if isinstance(parameters, Mapping)
            else None,
        }
        if (
            actual != expected
            or type(actual["native_id"]) is not int
            or not isinstance(actual["face_ids"], list)
            or any(type(value) is not int for value in actual["face_ids"])
        ):
            raise RuntimeError(
                f"saved native surface reference membership differs: "
                f"requested={expected!r}; actual={actual!r}"
            )
        requested["saved_membership"] = {
            "basis": "saved_native_project_geometry_entity_list",
            "project_sha256": digest,
            "design_name": design_name,
            **actual,
        }
    return verified


def add_analysis_surfaces(
    app: Any, prepared: PreparedPlanarGeometry, geometry: dict[str, Any]
) -> None:
    """Add EPR selection sheets to the already constructed common native model."""
    phase_seconds: dict[str, float] = {}
    surface_selections: list[dict[str, Any]] = []
    started = time.perf_counter()
    base_names = base_sheet_plan(prepared.surface_bindings)
    base_points: dict[str, list[list[float]]] = {}
    for binding in prepared.surface_bindings:
        binding_id = str(binding["binding_id"])
        planned_name = base_names[binding_id]
        name = planned_name["name"]
        if name not in base_points:
            sheet, points = _analysis_surface_sheet(
                app, binding["geometry_ref"], name=name
            )
            base_points[name] = points
        else:
            sheet = app.modeler.get_object_from_name(name)
            if sheet is None:
                raise RuntimeError(f"shared analysis surface {name!r} vanished")
            points = base_points[name]
        native = native_object_evidence(app, sheet.name)
        if native["native_object_type"] != "Sheet" or len(sheet.faces) != 1:
            raise RuntimeError(f"analysis surface {name!r} is not one native sheet")
        normal = sheet.faces[0].normal
        if (
            normal is None
            or len(normal) != 3
            or not all(math.isfinite(float(value)) for value in normal)
        ):
            raise RuntimeError(f"analysis surface {name!r} normal is unavailable")
        unit_normal = tuple(float(value) for value in normal)
        centroid = tuple(
            sum(point[axis] for point in points) / len(points) for axis in range(3)
        )
        field_side = str(binding["contribution"]["side"])
        desired_normal, adjacent_side = _native_sheet_side_binding(
            unit_normal,
            field_side,
            prepared.modeling,
            unknown_side_error=f"analysis surface {name!r} has unknown field side",
            ambiguous_side_error=(
                f"analysis surface {name!r} has ambiguous effective-domain side"
            ),
        )
        surface_selections.append(
            {
                "binding_id": binding_id,
                "contribution_id": binding["contribution"]["contribution_id"],
                "selection_name": sheet.name,
                "shared_sheet_members": planned_name["members"],
                "effective_domain_id": binding["effective_domain_id"],
                "adjacent_side": adjacent_side,
                "field_side": field_side,
                "native_normal": list(unit_normal),
                "selection_centroid_um": list(centroid),
                "desired_field_side_normal": list(desired_normal),
                **native,
            }
        )
    phase_seconds["base_analysis_sheets_seconds"] = time.perf_counter() - started

    geometry["surface_selections"] = surface_selections
    geometry["geometry_phases"].update(
        {key: round(value, 6) for key, value in phase_seconds.items()}
    )


def bind_saved_analysis_surfaces(
    app: Any, prepared: PreparedPlanarGeometry, geometry: dict[str, Any]
) -> None:
    """Read EPR selection sheets from the saved common native model."""
    selections: list[dict[str, Any]] = []
    base_names = base_sheet_plan(prepared.surface_bindings)
    for binding in prepared.surface_bindings:
        binding_id = str(binding["binding_id"])
        planned_name = base_names[binding_id]
        name = planned_name["name"]
        obj = app.modeler.get_object_from_name(name)
        if obj is None or len(obj.faces) != 1:
            raise RuntimeError(f"saved analysis surface {name!r} is unavailable")
        evidence = native_object_evidence(app, name)
        if evidence["native_object_type"] != "Sheet":
            raise RuntimeError(f"saved analysis surface {name!r} is not a sheet")
        normal = obj.faces[0].normal
        if (
            normal is None
            or len(normal) != 3
            or not all(math.isfinite(float(value)) for value in normal)
        ):
            raise RuntimeError(f"saved analysis surface {name!r} normal is unavailable")
        native_normal = tuple(float(value) for value in normal)
        side = str(binding["contribution"]["side"])
        desired, adjacent_side = _native_sheet_side_binding(
            native_normal,
            side,
            prepared.modeling,
            unknown_side_error=f"saved analysis surface {name!r} has unknown side",
            ambiguous_side_error=(f"saved analysis surface {name!r} side is ambiguous"),
        )
        selections.append(
            {
                "binding_id": binding_id,
                "contribution_id": binding["contribution"]["contribution_id"],
                "selection_name": name,
                "shared_sheet_members": planned_name["members"],
                "effective_domain_id": binding["effective_domain_id"],
                "adjacent_side": adjacent_side,
                "field_side": side,
                "native_normal": list(native_normal),
                "desired_field_side_normal": list(desired),
                **evidence,
            }
        )
    geometry["surface_selections"] = selections
