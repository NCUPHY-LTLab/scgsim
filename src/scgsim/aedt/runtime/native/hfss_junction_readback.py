"""Live and saved readback of declared Josephson-junction geometry.

Owns verifying junction sheets, partitions and terminal lines against the live
CAD state before authoring, and against the saved HFSS design afterwards. It does
not own body creation or the planar topology primitives."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

from scgsim.aedt._data import detached
from scgsim.aedt._io import file_sha256
from scgsim.aedt.preparation.planar import (
    junction_terminal_line,
    native_entity_name,
    native_name,
)
from scgsim.aedt.runtime.native.common import (
    native_boundary_type,
    native_object_boolean_property,
    native_object_property,
    resolve_native_assignment,
)
from scgsim.aedt.runtime.native.hfss_bodies import native_conductor_geometry
from scgsim.aedt.runtime.native.hfss_planar_topology import (
    NATIVE_JUNCTION_COORDINATE_ABS_TOL_UM,
    native_boundary_contact,
    native_closed_contact,
    native_coordinates_close,
    native_cross_sign,
    native_edge_covered,
    native_edge_section,
    native_extrusion_sides,
    native_planar_loops,
    native_polygon_correspondence,
    native_strict_inside,
    native_z_contact,
    native_z_may_contact,
)
from scgsim.aedt.specs.epr import (
    PlanarJunction,
)


@dataclass
class _Evidence:
    """Geometry observations and coordinate deviations collected by one readback."""

    observations: list[dict[str, Any]] = field(default_factory=list)
    deviations: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class _CentralFace:
    """The live central junction face and how it matches the authored polygon."""

    name: str
    expected: Mapping[str, Any]
    matches: Mapping[Any, Any]
    z_range_um: tuple[float, float]


@dataclass(frozen=True)
class _SourceBody:
    """One live source conductor polygon and the geometry it was authored with."""

    name: str
    obj: Any
    semantic_id: str
    polygon_id: str
    wanted: Mapping[str, Any]
    representation: str
    z_min_um: float
    z_max_um: float


@dataclass
class _LiveContacts:
    """Live source-conductor loops, grouped for the junction contact checks."""

    # Loops whose Z span touches the central face, grouped by net.
    touching_by_net: dict[str, list] = field(default_factory=dict)
    # Loops whose Z span may touch the central face, grouped by net.
    possible_by_net: dict[str, list] = field(default_factory=dict)
    loops_by_polygon: dict[tuple[str, str], list] = field(default_factory=dict)
    z_by_polygon: dict[tuple[str, str], tuple[tuple[float, float], bool]] = field(
        default_factory=dict
    )


def native_junction_live_readback(
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
    evidence = _Evidence()
    central = _read_central_face(app, source, record, junction.junction_id, evidence)
    contacts = _read_source_conductors(app, source, central.z_range_um, evidence)

    points = _central_corner_points(central.matches, central.expected)
    _verify_central_interior_clear(points, contacts.possible_by_net)
    _verify_derived_ends(source, record, contacts)
    # The first two edges are the terminal A and B end edges, which must lie wholly
    # on their own net; the last two are the open sides, which must touch nothing.
    edge_specs = (
        (points[3], points[0], junction.terminal_a_net),
        (points[1], points[2], junction.terminal_b_net),
        (points[0], points[1], None),
        (points[2], points[3], None),
    )
    edge_evidence = _verify_junction_edge_contacts(
        edge_specs,
        points,
        junction,
        contacts.touching_by_net,
        contacts.possible_by_net,
    )

    boundary_name = native_name("junction_rlc", junction.junction_id)
    _assert_boundary_target(app, boundary_name, central.name, "Lumped RLC")
    _verify_junction_rlc_properties(app, boundary_name, junction)
    evidence.deviations.extend(
        observation["maximum_plane_coordinate_deviation_um"]
        for observation in evidence.observations
        if "maximum_plane_coordinate_deviation_um" in observation
    )
    return {
        "coordinate_comparison_policy": {
            "absolute_tolerance_um": NATIVE_JUNCTION_COORDINATE_ABS_TOL_UM,
            "relative_tolerance": 0.0,
            "scope": "SCGSim native junction readback; not an AEDT precision guarantee",
            "geometry_moved": False,
        },
        "maximum_source_coordinate_deviation_um": max(evidence.deviations, default=0.0),
        "native_geometry_observations": evidence.observations,
        "edge_contacts": edge_evidence,
        "boundary_name": boundary_name,
    }


def _polygons_by_id(source: Mapping[str, Any]) -> dict[str, Any]:
    """Index the source polygons by ``polygon_id``."""
    return {p["polygon_id"]: p for p in source["polygons"]}


def _central_corner_points(
    matches: Mapping[Any, Any], expected: Mapping[str, Any]
) -> list[Any]:
    """Return the live central corners in the authored A->B order.

    The correspondence maps each authored vertex to the actual vertex, so the
    result is actual geometry ordered like the authored exterior.
    """
    return [
        matches[tuple(Fraction(str(float(v))) for v in p)] for p in expected["exterior"]
    ]


def _z_extent(vertices: Sequence[Sequence[float]]) -> tuple[float, float]:
    """Return ``(min, max)`` of the Z coordinate over ``vertices``."""
    return tuple(operation(p[2] for p in vertices) for operation in (min, max))


def _read_central_face(
    app: Any,
    source: Mapping[str, Any],
    record: Mapping[str, Any],
    junction_id: str,
    evidence: _Evidence,
) -> _CentralFace:
    """Find the one four-edge central face and match it to its authored polygon."""
    polygons = _polygons_by_id(source)
    z_um = float(record["z_um"])
    name = native_name("junction", junction_id)
    central = app.modeler.get_object_from_name(name)
    if central is None or len(central.faces) != 1 or len(central.faces[0].edges) != 4:
        raise RuntimeError("native central junction is not one four-edge face")
    loops = native_planar_loops(central, z_um, evidence.observations)
    expected = polygons[record["central_polygon_id"]]
    matches = native_polygon_correspondence(
        loops, expected, name, evidence.deviations, subdivisions=False
    )
    z_range = _z_extent(evidence.observations[0]["face_vertices_um"])
    return _CentralFace(name, expected, matches, z_range)


def _assert_boundary_target(
    app: Any, boundary_name: str, target: str, kind: str
) -> None:
    """Require boundary ``boundary_name`` to be a ``kind`` boundary on only ``target``."""
    if native_boundary_type(app, boundary_name) != kind:
        raise RuntimeError(f"native junction boundary type differs: {boundary_name!r}")
    raw = app.oboundary.GetBoundaryAssignment(boundary_name)
    if raw is None:
        raise RuntimeError(
            f"native junction boundary target unavailable: {boundary_name!r}"
        )
    ids = [int(v) for v in raw]
    if not ids or len(ids) != len(set(ids)):
        raise RuntimeError("native junction boundary assignment IDs are invalid")
    _, _, names = resolve_native_assignment(app, ids)
    if names != {target}:
        raise RuntimeError(
            f"native junction boundary has wrong targets: {boundary_name!r}"
        )


def _read_source_conductors(
    app: Any,
    source: Mapping[str, Any],
    central_z: tuple[float, float],
    evidence: _Evidence,
) -> _LiveContacts:
    """Read back each non-junction source polygon and sort it by Z contact with the junction."""
    polygons = _polygons_by_id(source)
    authored_ids = {
        partition["authored_polygon_id"]
        for partition in source["junction_partitions"].values()
    }
    contacts = _LiveContacts()
    for entity in source["conductors"]:
        representation, z_min_um, z_max_um = native_conductor_geometry(source, entity)
        for pid in entity["polygon_ids"]:
            if pid in authored_ids:
                continue
            name = native_entity_name(source, "conductor", entity["semantic_id"], pid)
            obj = app.modeler.get_object_from_name(name)
            if obj is None:
                raise RuntimeError(
                    f"native junction source conductor unavailable: {name!r}"
                )
            body = _SourceBody(
                name,
                obj,
                entity["semantic_id"],
                pid,
                polygons[pid],
                representation,
                z_min_um,
                z_max_um,
            )
            loops, native_z, solid = _read_source_polygon(app, body, evidence)
            key = entity["semantic_id"], pid
            contacts.loops_by_polygon[key] = loops
            contacts.z_by_polygon[key] = native_z, solid
            if native_z_contact(central_z, False, native_z, solid):
                contacts.touching_by_net.setdefault(entity["net_id"], []).extend(loops)
            if native_z_may_contact(central_z, native_z):
                contacts.possible_by_net.setdefault(entity["net_id"], []).extend(loops)
    return contacts


def _read_source_polygon(
    app: Any, body: _SourceBody, evidence: _Evidence
) -> tuple[list, tuple[float, float], bool]:
    """Match one live source polygon to its authored footprint.

    Returns the live loops, the live Z extent, and whether the body is a solid.
    A sheet is a zero-thickness face carrying a Perfect E boundary; anything else
    is a finite-thickness solid lowered as PEC.
    """
    loops = native_planar_loops(body.obj, body.z_min_um, evidence.observations)
    bottom_matches = native_polygon_correspondence(
        loops, body.wanted, body.name, evidence.deviations
    )
    bottom_vertices = evidence.observations[-1]["face_vertices_um"]
    if body.representation == "sheet":
        _assert_boundary_target(
            app,
            native_name("pec_boundary", body.semantic_id, body.polygon_id),
            body.name,
            "Perfect E",
        )
        return loops, _z_extent(bottom_vertices), False
    native_z = _verify_finite_pec_body(
        body, loops, bottom_matches, bottom_vertices, evidence
    )
    return loops, native_z, True


def _xy_to_vertex(vertices: Sequence[Sequence[float]]) -> dict[tuple, Any]:
    """Index 3D vertices by their exact ``(x, y)`` so top and bottom can be paired."""
    return {tuple(Fraction(str(v)) for v in p[:2]): p for p in vertices}


def _verify_finite_pec_body(
    body: _SourceBody,
    loops: list,
    bottom_matches: Mapping[Any, Any],
    bottom_vertices: Sequence[Sequence[float]],
    evidence: _Evidence,
) -> tuple[float, float]:
    """Check a finite conductor is PEC, spans its authored Z, and extrudes straight.

    Returns the live Z extent.
    """
    obj, name = body.obj, body.name
    bottom_xyz = _xy_to_vertex(bottom_vertices)
    if (
        native_object_property(obj, "Material").strip('"').casefold() != "pec"
        or native_object_boolean_property(obj, "Solve Inside") is not False
    ):
        raise RuntimeError(
            f"native junction conductor lacks finite PEC lowering: {name!r}"
        )
    zs = [float(vertex.position[2]) for vertex in obj.vertices]
    if not zs or not native_coordinates_close(
        (min(zs), max(zs)), (body.z_min_um, body.z_max_um)
    ):
        raise RuntimeError(
            f"native junction source material/Z lineage differs: {name!r}"
        )
    top = native_planar_loops(obj, max(zs), evidence.observations)
    top_matches = native_polygon_correspondence(
        top, body.wanted, name, evidence.deviations
    )
    try:
        native_polygon_correspondence(
            top, {"exterior": loops[0][0], "holes": loops[0][1:]}, name, []
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "native junction actual top/bottom extrusion differs"
        ) from exc
    top_xyz = _xy_to_vertex(evidence.observations[-1]["face_vertices_um"])
    thickness = body.z_max_um - body.z_min_um
    if any(
        not native_coordinates_close(bottom_matches[point], top_matches[point])
        or not native_coordinates_close(
            (top_xyz[top_matches[point]][2] - bottom_xyz[bottom_matches[point]][2],),
            (thickness,),
        )
        for point in bottom_matches
    ):
        raise RuntimeError("native junction actual top/bottom extrusion differs")
    native_extrusion_sides(
        obj,
        (body.z_min_um, body.z_max_um),
        thickness,
        evidence.observations,
    )
    return min(zs), max(zs)


def _verify_central_interior_clear(
    points: Sequence[Any], possible_by_net: Mapping[str, list]
) -> None:
    """Require that no conductor that could reach the central face enters or crosses it."""
    central_midpoint = tuple(sum(p[i] for p in points) / 4 for i in (0, 1))
    for bodies in possible_by_net.values():
        for rings in bodies:
            if any(native_strict_inside(p, (points,)) for p in rings[0]) or any(
                native_strict_inside(p, rings) for p in (*points, central_midpoint)
            ):
                raise RuntimeError("native PEC occupies central RLC interior")
            for ring in rings:
                for a, b in zip(ring, (*ring[1:], ring[0]), strict=False):
                    for c, d in zip(points, (*points[1:], points[0]), strict=False):
                        if (
                            native_cross_sign(a, b, c) * native_cross_sign(a, b, d) < 0
                            and native_cross_sign(c, d, a) * native_cross_sign(c, d, b)
                            < 0
                        ):
                            raise RuntimeError(
                                "native PEC crosses central RLC interior"
                            )


def _verify_derived_ends(
    source: Mapping[str, Any], record: Mapping[str, Any], contacts: _LiveContacts
) -> None:
    """Check each derived end polygon touches its own source arm and no other net."""
    entity_nets = {
        entity["semantic_id"]: entity["net_id"] for entity in source["conductors"]
    }
    derived_ids = {
        end["polygon_id"]
        for partition in source["junction_partitions"].values()
        for end in partition["ends"]
        if end["polygon_id"] is not None
    }
    loops_by_polygon = contacts.loops_by_polygon
    z_by_polygon = contacts.z_by_polygon
    for end in record["ends"]:
        if end["polygon_id"] is None:
            continue
        end_loops = loops_by_polygon[end["source_entity_id"], end["polygon_id"]]
        end_z, end_solid = z_by_polygon[end["source_entity_id"], end["polygon_id"]]
        arms = [
            rings
            for (eid, pid), loops in loops_by_polygon.items()
            if eid == end["source_entity_id"]
            and pid not in derived_ids
            and native_z_contact(end_z, end_solid, *z_by_polygon[eid, pid])
            for rings in loops
        ]
        if not native_boundary_contact(end_loops, arms):
            raise RuntimeError(
                "native derived end lacks positive-length source-arm contact"
            )
        foreign = [
            rings
            for (eid, pid), loops in loops_by_polygon.items()
            if entity_nets[eid] != end["net_id"]
            and native_z_may_contact(end_z, z_by_polygon[eid, pid][0])
            for rings in loops
        ]
        if native_closed_contact(end_loops, foreign):
            raise RuntimeError("native derived end contacts another net")


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
        if correct_net is not None and not native_edge_covered(
            native_by_net.get(correct_net, []), start, end
        ):
            raise RuntimeError(
                "native designated junction end edge is not wholly covered"
            )
        for net, bodies in potential_by_net.items():
            for rings in bodies:
                intervals, contacts = native_edge_section([rings], start, end)
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


def read_saved_hfss_design(project_path, design_name, *, purpose):
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


def read_saved_junction_lines(app, project_path, design_name, prepared, geometry):
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
    saved = _read_saved_boundaries(app, project_path, design_name)
    polygons = _polygons_by_id(source)
    bindings = {j["junction_id"]: j for j in geometry["junctions"]}
    completed = []
    for junction in prepared.junctions:
        if junction.junction_id not in partitions:
            continue
        binding = bindings[junction.junction_id]
        record = partitions[junction.junction_id]
        readback = _read_saved_junction_line(
            app,
            binding,
            junction,
            (record, polygons[record["central_polygon_id"]]),
            saved,
        )
        completed.append((binding, readback))
    # Publish only after every junction passed, so a failure leaves no partial records.
    for binding, readback in completed:
        binding.pop("_partition_live_readback")
        binding["partition_readback"] = readback


@dataclass(frozen=True)
class _SavedBoundaries:
    """Boundary records of a saved HFSS design and the digest of the file they came from."""

    project_sha256: str
    boundaries: Mapping[str, Any]


def _read_saved_boundaries(
    app: Any, project_path: Any, design_name: str
) -> _SavedBoundaries:
    """Read the boundary records of the saved HFSS design.

    Requires the live app to be that same saved project and design, with
    micrometre model units, so saved and live geometry share one frame.
    """
    path = Path(project_path).resolve(strict=True)
    if (
        Path(app.project_file).resolve(strict=True) != path
        or app.design_name != design_name
        or app.design_type != "HFSS"
    ):
        raise RuntimeError("saved junction project/design identity differs")
    digest, design = read_saved_hfss_design(path, design_name, purpose="junction")
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
    return _SavedBoundaries(digest, boundaries)


def _read_saved_junction_line(
    app: Any,
    binding: dict[str, Any],
    junction: PlanarJunction,
    partition: tuple[Mapping[str, Any], Mapping[str, Any]],
    saved: _SavedBoundaries,
) -> dict[str, Any]:
    """Verify one junction's saved integration line and return its completed readback.

    ``partition`` is the junction's partition record and its authored central polygon.
    """
    record, expected = partition
    boundary = saved.boundaries.get(binding["boundary_name"])
    if not isinstance(boundary, Mapping) or boundary.get("BoundType") != "Lumped RLC":
        raise RuntimeError("saved junction boundary identity/type differs")
    raw_line = boundary.get("CurrentLine")
    positions = (
        raw_line.get("GeometryPosition") if isinstance(raw_line, Mapping) else None
    )
    if not isinstance(positions, list) or len(positions) != 2:
        raise RuntimeError(
            "saved junction line requires two ordered GeometryPosition records"
        )
    z_um = float(record["z_um"])
    central = app.modeler.get_object_from_name(binding["object_name"])
    if central is None or len(central.faces) != 1 or len(central.faces[0].edges) != 4:
        raise RuntimeError("saved junction central face/edge topology differs")
    central_id = int(app.modeler.oeditor.GetObjectIDByName(central.name))
    if boundary.get("Objects") != [central_id]:
        raise RuntimeError(
            "saved junction boundary target differs from actual central object"
        )
    evidence = _Evidence()
    loops = native_planar_loops(central, z_um, evidence.observations)
    matches = native_polygon_correspondence(
        loops, expected, central.name, evidence.deviations, subdivisions=False
    )
    points = _central_corner_points(matches, expected)
    terminal_edges = _actual_terminal_edges(evidence.observations[0]["edges"], points)
    observed_line = [
        _saved_line_position(position, edge)
        for position, edge in zip(positions, terminal_edges, strict=False)
    ]
    line, _ = junction_terminal_line(expected, junction, z_um)
    actual_midpoints = [edge["midpoint_um"] for edge in terminal_edges]
    if any(
        not native_coordinates_close(a, b)
        for a, b in zip(observed_line, line, strict=False)
    ):
        raise RuntimeError("saved junction integration line differs")
    if any(
        not native_coordinates_close(a, b)
        for a, b in zip(observed_line, actual_midpoints, strict=False)
    ):
        raise RuntimeError(
            "saved junction integration line differs from actual terminal edges"
        )
    evidence.deviations.extend(
        abs(a - b)
        for actual, wanted in zip(observed_line, line, strict=False)
        for a, b in zip(actual, wanted, strict=False)
    )
    live = binding["_partition_live_readback"]
    return {
        **live,
        "method": "native_face_edges_live_electrical_and_saved_line.v3",
        "maximum_source_coordinate_deviation_um": max(
            live["maximum_source_coordinate_deviation_um"], *evidence.deviations
        ),
        "integration_line_um": observed_line,
        "actual_terminal_edge_midpoints_um": actual_midpoints,
        "raw_integration_line": detached(raw_line),
        "saved_line_verification_project_sha256": saved.project_sha256,
        "saved_line_native_geometry_observations": evidence.observations,
    }


def _actual_terminal_edges(
    edges: Sequence[Mapping[str, Any]], points: Sequence[Any]
) -> list[Mapping[str, Any]]:
    """Find the live central edges for terminal A (corners 3-0) and terminal B (1-2)."""
    edge_ids = [edge["edge_id"] for edge in edges]
    if len(edge_ids) != len(set(edge_ids)):
        raise RuntimeError("saved junction native central edge IDs are ambiguous")
    terminal_edges = []
    for a, b in ((points[3], points[0]), (points[1], points[2])):
        found = [
            edge
            for edge in edges
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
    return terminal_edges


def _saved_line_position(position: Any, edge: Mapping[str, Any]) -> list[float]:
    """Resolve one saved line endpoint to coordinates, given its expected terminal edge.

    An edge-attached ``EdgeCenter`` must reference the actual terminal edge and
    resolves to that edge's midpoint; an ``AbsolutePosition`` carries its own XYZ.
    """
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
        return list(edge["midpoint_um"])
    if (
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
            isinstance(v, bool) or not isinstance(v, (str, int, float)) for v in raw
        ) or not all(math.isfinite(v) for v in coordinate):
            raise RuntimeError("saved junction absolute line coordinates are invalid")
        return coordinate
    raise RuntimeError("saved junction line attachment kind is unsupported")
