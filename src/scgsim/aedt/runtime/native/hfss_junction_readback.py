"""Live and saved readback of declared Josephson-junction geometry.

Owns verifying junction sheets, partitions and terminal lines against the live
CAD state before authoring, and against the saved HFSS design afterwards. It does
not own body creation or the planar topology primitives."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

from scgsim.aedt._data import detached
from scgsim.aedt._io import file_sha256
from scgsim.aedt.preparation.planar import (
    _junction_terminal_line,
    _native_entity_name,
    _native_name,
)
from scgsim.aedt.runtime.native.common import (
    _native_boundary_type,
    _native_object_boolean_property,
    _resolve_native_assignment,
    native_object_property,
)
from scgsim.aedt.runtime.native.hfss_bodies import _native_conductor_geometry
from scgsim.aedt.runtime.native.hfss_planar_topology import (
    _NATIVE_JUNCTION_COORDINATE_ABS_TOL_UM,
    _native_boundary_contact,
    _native_closed_contact,
    _native_coordinates_close,
    _native_cross_sign,
    _native_edge_covered,
    _native_edge_section,
    _native_extrusion_sides,
    _native_planar_loops,
    _native_polygon_correspondence,
    _native_strict_inside,
    _native_z_contact,
    _native_z_may_contact,
)
from scgsim.aedt.specs.epr import (
    PlanarJunction,
)


# Length: over 150 lines on purpose. The steps share one growing set of live
# observations and coordinate deviations, and the order of the checks decides which
# readback failure a user sees first, so splitting needs a shared-state record.
# TODO(human): split into central face / per-net bodies / end contacts once the
# check order is confirmed to be free to group.
def _native_junction_live_readback(
    app: Any, source: Mapping[str, Any], junction: PlanarJunction
) -> dict[str, Any]:
    """Verify live CAD/electrical state before authoring; line awaits native save.

    Face edges supply complete planar footprints, rather than bounding-box
    contact guesses. Actual returned geometry is preserved. Coordinate identity,
    straightness, Z and contact comparisons use absolute 1e-5 um, rel_tol=0;
    correspondence never substitutes source vertices into contact geometry.
    Below-resolution gaps/curvature remain numerically unresolved. Missing live
    RLC properties or ambiguous topology are explicit readback failures. The
    transaction-local result is not a completed partition_readback; the owned
    runtime must verify the saved assigned line before publishing that record.
    """
    record = source["junction_partitions"][junction.junction_id]
    if app.modeler.model_units != "um":
        raise RuntimeError(
            "junction native readback requires source micrometre model units"
        )
    polygons = {p["polygon_id"]: p for p in source["polygons"]}
    z_um = float(record["z_um"])
    central_name = _native_name("junction", junction.junction_id)
    central = app.modeler.get_object_from_name(central_name)
    if central is None or len(central.faces) != 1 or len(central.faces[0].edges) != 4:
        raise RuntimeError("native central junction is not one four-edge face")
    observations, deviations = [], []
    central_loops = _native_planar_loops(central, z_um, observations)
    expected = polygons[record["central_polygon_id"]]
    central_matches = _native_polygon_correspondence(
        central_loops, expected, central_name, deviations, subdivisions=False
    )
    central_z = tuple(
        operation(p[2] for p in observations[0]["face_vertices_um"])
        for operation in (min, max)
    )

    def assignment(boundary_name, target, kind):
        if _native_boundary_type(app, boundary_name) != kind:
            raise RuntimeError(
                f"native junction boundary type differs: {boundary_name!r}"
            )
        raw = app.oboundary.GetBoundaryAssignment(boundary_name)
        if raw is None:
            raise RuntimeError(
                f"native junction boundary target unavailable: {boundary_name!r}"
            )
        ids = [int(v) for v in raw]
        if not ids or len(ids) != len(set(ids)):
            raise RuntimeError("native junction boundary assignment IDs are invalid")
        _, _, names = _resolve_native_assignment(app, ids)
        if names != {target}:
            raise RuntimeError(
                f"native junction boundary has wrong targets: {boundary_name!r}"
            )

    native_by_net: dict[str, list] = {}
    potential_by_net: dict[str, list] = {}
    native_by_polygon: dict[tuple[str, str], list] = {}
    native_z_by_polygon = {}
    entity_nets = {
        entity["semantic_id"]: entity["net_id"] for entity in source["conductors"]
    }
    authored_ids = {
        partition["authored_polygon_id"]
        for partition in source["junction_partitions"].values()
    }
    for entity in source["conductors"]:
        representation, effective_z_min_um, effective_z_max_um = (
            _native_conductor_geometry(source, entity)
        )
        plane = effective_z_min_um
        for pid in entity["polygon_ids"]:
            if pid in authored_ids:
                continue
            name = _native_entity_name(source, "conductor", entity["semantic_id"], pid)
            obj = app.modeler.get_object_from_name(name)
            if obj is None:
                raise RuntimeError(
                    f"native junction source conductor unavailable: {name!r}"
                )
            loops = _native_planar_loops(obj, plane, observations)
            wanted = polygons[pid]
            bottom_matches = _native_polygon_correspondence(
                loops, wanted, name, deviations
            )
            bottom_observation = observations[-1]
            bottom_xyz = {
                tuple(Fraction(str(v)) for v in p[:2]): p
                for p in bottom_observation["face_vertices_um"]
            }
            if representation == "sheet":
                assignment(
                    _native_name("pec_boundary", entity["semantic_id"], pid),
                    name,
                    "Perfect E",
                )
            else:
                if (
                    native_object_property(obj, "Material").strip('"').casefold()
                    != "pec"
                    or _native_object_boolean_property(obj, "Solve Inside") is not False
                ):
                    raise RuntimeError(
                        f"native junction conductor lacks finite PEC lowering: {name!r}"
                    )
                zs = [float(vertex.position[2]) for vertex in obj.vertices]
                if not zs or not _native_coordinates_close(
                    (min(zs), max(zs)),
                    (effective_z_min_um, effective_z_max_um),
                ):
                    raise RuntimeError(
                        f"native junction source material/Z lineage differs: {name!r}"
                    )
                top = _native_planar_loops(obj, max(zs), observations)
                top_matches = _native_polygon_correspondence(
                    top, wanted, name, deviations
                )
                try:
                    _native_polygon_correspondence(
                        top, {"exterior": loops[0][0], "holes": loops[0][1:]}, name, []
                    )
                except RuntimeError as exc:
                    raise RuntimeError(
                        "native junction actual top/bottom extrusion differs"
                    ) from exc
                top_xyz = {
                    tuple(Fraction(str(v)) for v in p[:2]): p
                    for p in observations[-1]["face_vertices_um"]
                }
                thickness = effective_z_max_um - effective_z_min_um
                if any(
                    not _native_coordinates_close(
                        bottom_matches[point], top_matches[point]
                    )
                    or not _native_coordinates_close(
                        (
                            top_xyz[top_matches[point]][2]
                            - bottom_xyz[bottom_matches[point]][2],
                        ),
                        (thickness,),
                    )
                    for point in bottom_matches
                ):
                    raise RuntimeError(
                        "native junction actual top/bottom extrusion differs"
                    )
                _native_extrusion_sides(
                    obj,
                    loops,
                    top,
                    (effective_z_min_um, effective_z_max_um),
                    thickness,
                    observations,
                )
            solid = representation != "sheet"
            native_z = (
                (min(zs), max(zs))
                if solid
                else tuple(
                    operation(p[2] for p in bottom_observation["face_vertices_um"])
                    for operation in (min, max)
                )
            )
            key = entity["semantic_id"], pid
            native_by_polygon[key] = loops
            native_z_by_polygon[key] = native_z, solid
            if _native_z_contact(central_z, False, native_z, solid):
                native_by_net.setdefault(entity["net_id"], []).extend(loops)
            if _native_z_may_contact(central_z, native_z):
                potential_by_net.setdefault(entity["net_id"], []).extend(loops)

    # Correspondence orders actual vertices in the authored A->B frame.
    points = [
        central_matches[tuple(Fraction(str(float(v))) for v in p)]
        for p in expected["exterior"]
    ]
    central_midpoint = tuple(sum(p[i] for p in points) / 4 for i in (0, 1))
    for bodies in potential_by_net.values():
        for rings in bodies:
            if any(_native_strict_inside(p, (points,)) for p in rings[0]) or any(
                _native_strict_inside(p, rings) for p in (*points, central_midpoint)
            ):
                raise RuntimeError("native PEC occupies central RLC interior")
            for ring in rings:
                for a, b in zip(ring, (*ring[1:], ring[0]), strict=False):
                    for c, d in zip(points, (*points[1:], points[0]), strict=False):
                        if (
                            _native_cross_sign(a, b, c) * _native_cross_sign(a, b, d)
                            < 0
                            and _native_cross_sign(c, d, a)
                            * _native_cross_sign(c, d, b)
                            < 0
                        ):
                            raise RuntimeError(
                                "native PEC crosses central RLC interior"
                            )
    derived_ids = {
        end["polygon_id"]
        for partition in source["junction_partitions"].values()
        for end in partition["ends"]
        if end["polygon_id"] is not None
    }

    for end in record["ends"]:
        if end["polygon_id"] is None:
            continue
        end_loops = native_by_polygon[end["source_entity_id"], end["polygon_id"]]
        end_z, end_solid = native_z_by_polygon[
            end["source_entity_id"], end["polygon_id"]
        ]
        arms = [
            rings
            for (eid, pid), loops in native_by_polygon.items()
            if eid == end["source_entity_id"]
            and pid not in derived_ids
            and _native_z_contact(end_z, end_solid, *native_z_by_polygon[eid, pid])
            for rings in loops
        ]
        if not _native_boundary_contact(end_loops, arms):
            raise RuntimeError(
                "native derived end lacks positive-length source-arm contact"
            )
        foreign = [
            rings
            for (eid, pid), loops in native_by_polygon.items()
            if entity_nets[eid] != end["net_id"]
            and _native_z_may_contact(end_z, native_z_by_polygon[eid, pid][0])
            for rings in loops
        ]
        if _native_closed_contact(end_loops, foreign):
            raise RuntimeError("native derived end contacts another net")
    edge_specs = (
        (points[3], points[0], junction.terminal_a_net),
        (points[1], points[2], junction.terminal_b_net),
        (points[0], points[1], None),
        (points[2], points[3], None),
    )
    edge_evidence = _verify_junction_edge_contacts(
        edge_specs, points, junction, native_by_net, potential_by_net
    )

    boundary_name = _native_name("junction_rlc", junction.junction_id)
    assignment(boundary_name, central_name, "Lumped RLC")
    _verify_junction_rlc_properties(app, boundary_name, junction)
    deviations.extend(
        observation["maximum_plane_coordinate_deviation_um"]
        for observation in observations
        if "maximum_plane_coordinate_deviation_um" in observation
    )
    return {
        "coordinate_comparison_policy": {
            "absolute_tolerance_um": _NATIVE_JUNCTION_COORDINATE_ABS_TOL_UM,
            "relative_tolerance": 0.0,
            "scope": "SCGSim native junction readback; not an AEDT precision guarantee",
            "geometry_moved": False,
        },
        "maximum_source_coordinate_deviation_um": max(deviations, default=0.0),
        "native_geometry_observations": observations,
        "edge_contacts": edge_evidence,
        "boundary_name": boundary_name,
    }


def _verify_junction_edge_contacts(
    edge_specs: Sequence[tuple[Any, Any, str | None]],
    points: Sequence[Any],
    junction: PlanarJunction,
    native_by_net: Mapping[str, list],
    potential_by_net: Mapping[str, list],
) -> list[dict[str, Any]]:
    """Check the junction's two end edges touch only their own net and its sides touch none.

    ``edge_specs`` is ``(start, end, net)`` per edge; ``net`` is None for a side edge.
    """
    edge_evidence = []
    for start, end, correct_net in edge_specs:
        if correct_net is not None and not _native_edge_covered(
            native_by_net.get(correct_net, []), start, end
        ):
            raise RuntimeError(
                "native designated junction end edge is not wholly covered"
            )
        for net, bodies in potential_by_net.items():
            for rings in bodies:
                intervals, contacts = _native_edge_section([rings], start, end)
                if correct_net is not None:
                    forbidden = net != correct_net and bool(intervals or contacts)
                else:
                    forbidden = bool(intervals) or any(0 < v < 1 for v in contacts)
                    # Open side corners belong to their respective end edge.
                    for v in contacts:
                        point = start if v == 0 else end
                        own_corner = (
                            junction.terminal_a_net
                            if point in (points[0], points[3])
                            else junction.terminal_b_net
                        )
                        forbidden |= net != own_corner
                if forbidden:
                    raise RuntimeError(
                        "native junction has forbidden side/wrong-net corner contact"
                    )
        edge_evidence.append(
            {
                "start_um": list(map(float, start)),
                "end_um": list(map(float, end)),
                "terminal_net": correct_net,
            }
        )
    return edge_evidence


def _verify_junction_rlc_properties(
    app: Any, boundary_name: str, junction: PlanarJunction
) -> None:
    """Compare the live lumped RLC boundary with the declared L (and optional C)."""
    from ansys.aedt.core.generic.constants import AEDT_UNITS
    from ansys.aedt.core.generic.numbers_utils import decompose_variable_value
    from ansys.aedt.core.modules.boundary.common import BoundaryObject

    live = BoundaryObject(app, boundary_name, auto_update=False)._child_object  # noqa: SLF001 - PyAEDT has no public API for this
    if live is None:
        raise RuntimeError("native live junction RLC property object unavailable")
    available = [str(v) for v in live.GetPropNames()]
    required = (
        "RLC Type",
        "Use Induct",
        "Inductance",
        "Use Cap",
        "Use Resist",
    )
    if any(name not in available for name in required):
        raise RuntimeError(
            f"native live RLC readback lacks required properties; available={available!r}"
        )
    values = {name: live.GetPropValue(name) for name in required}

    def enabled(value):
        if value in (True, "true", "True", 1):
            return True
        if value in (False, "false", "False", 0):
            return False
        raise RuntimeError("native RLC enabled property is invalid")

    def scalar(value, unit):
        # Exact decimal SI equality avoids binary unit-conversion roundoff;
        # the length comparison policy never applies to electrical parameters.
        number, observed_unit = decompose_variable_value(value)
        system = {"H": "Inductance", "F": "Capacitance"}[unit]
        if (
            observed_unit not in AEDT_UNITS[system]
            or not isinstance(number, (int, float))
            or isinstance(number, bool)
            or not math.isfinite(number)
        ):
            raise RuntimeError("native RLC scalar unit/value unavailable")
        return Decimal(str(number)) * Decimal(str(AEDT_UNITS[system][observed_unit]))

    if (
        values["RLC Type"] != "Parallel"
        or not enabled(values["Use Induct"])
        or enabled(values["Use Resist"])
        or scalar(values["Inductance"], "H") != Decimal(str(junction.inductance_h))
    ):
        raise RuntimeError("native junction parallel L/R parameters differ")
    if enabled(values["Use Cap"]) != bool(junction.capacitance_f):
        raise RuntimeError("native junction capacitance enable state differs")
    if junction.capacitance_f and (
        "Capacitance" not in available
        or scalar(live.GetPropValue("Capacitance"), "F")
        != Decimal(str(junction.capacitance_f))
    ):
        raise RuntimeError("native junction capacitance differs")


def _read_saved_hfss_design(project_path, design_name, *, purpose):
    """Read one saved HFSS design without a cached parser or project mutation."""
    from ansys.aedt.core.internal.load_aedt_file import load_entire_aedt_file

    path = Path(project_path).resolve(strict=True)
    digest = file_sha256(path)
    native = load_entire_aedt_file(path)
    if file_sha256(path) != digest:
        raise RuntimeError(f"saved {purpose} native project changed during readback")
    project = native.get("AnsoftProject")
    designs = project.get("HFSSModel") if isinstance(project, Mapping) else None
    if isinstance(designs, Mapping):
        designs = [designs]
    if not isinstance(designs, list):
        raise TypeError(f"saved {purpose} design records are unavailable")
    matching = [
        d for d in designs if isinstance(d, Mapping) and d.get("Name") == design_name
    ]
    if len(matching) != 1:
        raise RuntimeError(f"saved {purpose} design is missing or ambiguous")
    return digest, matching[0]


# Length: over 150 lines. One pass builds the saved-line evidence while it checks
# each junction against the saved project, and the checks run in a fixed order so
# the first failing check is the error reported.
# TODO(human): confirm the step boundaries before splitting.
def _read_saved_junction_lines(app, project_path, design_name, prepared, geometry):
    """Complete junction evidence from one uncached, identity-bound native save.

    PyAEDT1.3 configurations._update_boundaries interprets GeometryPosition XYZ
    in global model units. Attached EdgeCenter instead references actual edges;
    XYZ can be zero placeholders. Only these two position kinds are supported.
    IDs are resolved against freshly read central edges in the current object,
    with no cross-session ID persistence assumption or geometry guess.
    """
    source = detached(prepared.source)
    partitions = source.get("junction_partitions", {})
    if not partitions:
        return  # Historical geometry retains its existing binding contract.
    path = Path(project_path).resolve(strict=True)
    if (
        Path(app.project_file).resolve(strict=True) != path
        or app.design_name != design_name
        or app.design_type != "HFSS"
    ):
        raise RuntimeError("saved junction project/design identity differs")
    digest, design = _read_saved_hfss_design(path, design_name, purpose="junction")
    model_setup = design.get("ModelSetup")
    model = (
        model_setup.get("GeometryCore") if isinstance(model_setup, Mapping) else None
    )
    if (
        not isinstance(model, Mapping)
        or model.get("Units") != app.modeler.model_units
        or model.get("Units") != "um"
    ):
        raise RuntimeError(
            "saved junction model units differ from live micrometre geometry"
        )
    boundary_setup = design.get("BoundarySetup")
    boundaries = (
        boundary_setup.get("Boundaries")
        if isinstance(boundary_setup, Mapping)
        else None
    )
    if not isinstance(boundaries, Mapping):
        raise TypeError("saved junction boundary records are unavailable")
    polygons = {p["polygon_id"]: p for p in source["polygons"]}
    bindings = {j["junction_id"]: j for j in geometry["junctions"]}
    completed = []
    for junction in prepared.junctions:
        if junction.junction_id not in partitions:
            continue
        binding = bindings[junction.junction_id]
        evidence = binding["_partition_live_readback"]
        boundary = boundaries.get(binding["boundary_name"])
        if (
            not isinstance(boundary, Mapping)
            or boundary.get("BoundType") != "Lumped RLC"
        ):
            raise RuntimeError("saved junction boundary identity/type differs")
        raw_line = boundary.get("CurrentLine")
        positions = (
            raw_line.get("GeometryPosition") if isinstance(raw_line, Mapping) else None
        )
        if not isinstance(positions, list) or len(positions) != 2:
            raise RuntimeError(
                "saved junction line requires two ordered GeometryPosition records"
            )
        record = partitions[junction.junction_id]
        z_um = float(record["z_um"])
        expected = polygons[record["central_polygon_id"]]
        central = app.modeler.get_object_from_name(binding["object_name"])
        if (
            central is None
            or len(central.faces) != 1
            or len(central.faces[0].edges) != 4
        ):
            raise RuntimeError("saved junction central face/edge topology differs")
        central_id = int(app.modeler.oeditor.GetObjectIDByName(central.name))
        if boundary.get("Objects") != [central_id]:
            raise RuntimeError(
                "saved junction boundary target differs from actual central object"
            )
        observations, deviations = [], []
        loops = _native_planar_loops(central, z_um, observations)
        matches = _native_polygon_correspondence(
            loops, expected, central.name, deviations, subdivisions=False
        )
        points = [
            matches[tuple(Fraction(str(float(v))) for v in p)]
            for p in expected["exterior"]
        ]
        edge_ids = [edge["edge_id"] for edge in observations[0]["edges"]]
        if len(edge_ids) != len(set(edge_ids)):
            raise RuntimeError("saved junction native central edge IDs are ambiguous")
        terminal_edges = []
        for a, b in ((points[3], points[0]), (points[1], points[2])):
            found = [
                edge
                for edge in observations[0]["edges"]
                if {tuple(Fraction(str(v)) for v in p[:2]) for p in edge["vertices_um"]}
                == {a, b}
            ]
            if len(found) != 1:
                raise RuntimeError(
                    "saved junction actual terminal edge identity is unavailable"
                )
            if isinstance(found[0]["edge_id"], bool) or not isinstance(
                found[0]["edge_id"], int
            ):
                raise TypeError("saved junction actual terminal edge ID is invalid")
            terminal_edges.append(found[0])
        observed_line = []
        for position, edge in zip(positions, terminal_edges, strict=False):
            if not isinstance(position, Mapping):
                raise TypeError("saved junction line position is malformed")
            if (
                position.get("IsAttachedToEntity") is True
                and position.get("PositionType") == "EdgeCenter"
            ):
                entity_id = position.get("EntityID")
                if (
                    isinstance(entity_id, bool)
                    or not isinstance(entity_id, int)
                    or entity_id != edge["edge_id"]
                ):
                    raise RuntimeError(
                        "saved junction line attachment references wrong terminal edge"
                    )
                observed_line.append(list(edge["midpoint_um"]))
            elif (
                position.get("IsAttachedToEntity") is False
                and position.get("PositionType") == "AbsolutePosition"
            ):
                raw = [position.get(axis + "Position") for axis in "XYZ"]
                try:
                    coordinate = [float(v) for v in raw]
                except (TypeError, ValueError, OverflowError) as exc:
                    raise RuntimeError(
                        "saved junction absolute line coordinates are invalid"
                    ) from exc
                if any(
                    isinstance(v, bool) or not isinstance(v, (str, int, float))
                    for v in raw
                ) or not all(math.isfinite(v) for v in coordinate):
                    raise RuntimeError(
                        "saved junction absolute line coordinates are invalid"
                    )
                observed_line.append(coordinate)
            else:
                raise RuntimeError("saved junction line attachment kind is unsupported")
        line, _ = _junction_terminal_line(expected, junction, z_um)
        actual_midpoints = [edge["midpoint_um"] for edge in terminal_edges]
        if any(
            not _native_coordinates_close(a, b)
            for a, b in zip(observed_line, line, strict=False)
        ):
            raise RuntimeError("saved junction integration line differs")
        if any(
            not _native_coordinates_close(a, b)
            for a, b in zip(observed_line, actual_midpoints, strict=False)
        ):
            raise RuntimeError(
                "saved junction integration line differs from actual terminal edges"
            )
        deviations.extend(
            abs(a - b)
            for actual, wanted in zip(observed_line, line, strict=False)
            for a, b in zip(actual, wanted, strict=False)
        )
        completed.append(
            (
                binding,
                {
                    **evidence,
                    "method": "native_face_edges_live_electrical_and_saved_line.v3",
                    "maximum_source_coordinate_deviation_um": max(
                        evidence["maximum_source_coordinate_deviation_um"], *deviations
                    ),
                    "integration_line_um": observed_line,
                    "actual_terminal_edge_midpoints_um": actual_midpoints,
                    "raw_integration_line": detached(raw_line),
                    "saved_line_verification_project_sha256": digest,
                    "saved_line_native_geometry_observations": observations,
                },
            )
        )
    for binding, readback in completed:
        binding.pop("_partition_live_readback")
        binding["partition_readback"] = readback
