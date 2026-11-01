"""Read saved HFSS designs and verify identity-bound junction integration lines."""

from __future__ import annotations

import math
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any

from scgsim.aedt.data import detached
from scgsim.aedt.io import file_sha256
from scgsim.aedt.runtime.native.junction_topology import (
    native_coordinates_close,
    native_planar_loops,
    native_polygon_correspondence,
)
from scgsim.aedt.runtime.native.planar_bodies import junction_geometry
from scgsim.geometry.planning.junction_partition import junction_terminal_line

_NATIVE_JUNCTION_COORDINATE_ABS_TOL_UM = 1e-5


def read_saved_hfss_design(project_path, design_name, *, purpose):
    """Read one saved HFSS design without a cached parser or project mutation."""
    from ansys.aedt.core.internal.load_aedt_file import load_entire_aedt_file  # noqa: PLC0415 -- optional PyAEDT loads only at the authorized native execution boundary.

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
    digest, boundaries = _read_junction_boundaries(app, project_path, design_name)
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
        loops = native_planar_loops(central, z_um, observations)
        matches = native_polygon_correspondence(
            loops, expected, central.name, deviations, subdivisions=False
        )
        points = [
            matches[tuple(Fraction(str(float(v))) for v in p)]
            for p in expected["exterior"]
        ]
        terminal_edges = _read_terminal_edges(observations, points)
        observed_line = _read_line_positions(positions, terminal_edges)
        line, _ = junction_terminal_line(expected, junction_geometry(junction), z_um)
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


def _read_line_positions(positions, terminal_edges):
    """Resolve ordered absolute or edge-attached native line positions."""
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
            raw: list[Any] = [position.get(axis + "Position") for axis in "XYZ"]
            try:
                coordinate = [float(v) for v in raw]
            except (TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError(
                    "saved junction absolute line coordinates are invalid"
                ) from exc
            if any(
                isinstance(v, bool) or not isinstance(v, (str, int, float)) for v in raw
            ) or not all(math.isfinite(v) for v in coordinate):
                raise RuntimeError(
                    "saved junction absolute line coordinates are invalid"
                )
            observed_line.append(coordinate)
        else:
            raise RuntimeError("saved junction line attachment kind is unsupported")
    return observed_line


def _read_terminal_edges(observations, points):
    """Match the two native terminal edges to the verified central corners."""
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
    return terminal_edges


def _read_junction_boundaries(app, project_path, design_name):
    """Read the saved design after verifying project identity and model units."""
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
    return digest, boundaries
