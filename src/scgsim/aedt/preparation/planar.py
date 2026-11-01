"""Prepare a planar HFSS model: the entry point that assembles `PreparedPlanarGeometry`.

Owns: validating the declared junctions, RLC treatments and Surface-EPR contributions
against the normalized geometry, partitioning junctions, binding surface masks, and
the hashes that identify the prepared model. Source and solver treatments stay
distinct: the source record is detached facts, while junctions and RLCs are
solver treatments recorded beside it.

Delegates: the detached source record to `planar_source.py`, surface contribution
bindings to `planar_surfaces.py`, junction partitioning to `junction_partition.py`.
Also owns the native object names and the KLayout helpers that the native AEDT
builders import from here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from fractions import Fraction
from importlib.metadata import version
from typing import Any

from scgsim.aedt._data import canonical_sha256
from scgsim.aedt.preparation.junction_exact import closed_contact
from scgsim.aedt.preparation.junction_partition import partition_junctions
from scgsim.aedt.preparation.modeling import effective_geometry_input
from scgsim.aedt.preparation.planar_source import (
    plain_geometry_source,
    source_payload,
)
from scgsim.aedt.preparation.planar_surfaces import (
    bind_mask_support,
    merged_contribution_catalog,
    plane_group_key,
    project_plane_region,
    resolve_contributions,
    surface_ledger,
)
from scgsim.aedt.specs._support import prepared_lumped_supports
from scgsim.aedt.specs.epr import (
    PlanarJunction,
    SurfaceEprSpec,
)
from scgsim.aedt.specs.geometry import PreparedPlanarGeometry
from scgsim.aedt.specs.modeling import Modeling, native_physical_record
from scgsim.geometry import GeometryBuildInput
from scgsim.geometry.planning.surfaces import plan_representation_contribution_geometry
from scgsim.geometry.source.validation import validate_geometry_input
from scgsim.semantics.route_a import geometry_z_range

INSET_CIRCLE_POINTS = 128
INSET_METHOD = "klayout_complement_minkowski.v1"

KLAYOUT_COORD_MIN = -(2**31)
KLAYOUT_COORD_MAX = 2**31 - 1


def prepare_planar_geometry_input(  # noqa: PLR0913 - public keyword-only API
    build_input: GeometryBuildInput,
    *,
    prepared_stack: Mapping[str, Any],
    modeling: Modeling,
    lumped_rlcs: Sequence[Any] = (),
    junctions: Sequence[PlanarJunction] = (),
    contributions: Sequence[SurfaceEprSpec] = (),
    source_dbu_um: float | None = None,
) -> PreparedPlanarGeometry:
    """Detach explicit source/effective Solid or Thin Film facts for HFSS EPR."""
    _check_planar_inputs(build_input, prepared_stack)
    source_dbu_um = _resolved_source_dbu_um(build_input, source_dbu_um)
    validate_geometry_input(build_input)
    build_input, prepared_stack, representation_context = effective_geometry_input(
        build_input, prepared_stack, modeling
    )
    rlc_tuple, junction_tuple, contribution_tuple = _declared_treatments(
        lumped_rlcs, junctions, contributions
    )
    _check_declared_treatments(build_input, junction_tuple, contribution_tuple)

    build_input = partition_junctions(build_input, junction_tuple, source_dbu_um)
    if junction_tuple:
        build_input = _with_junction_planes(build_input, junction_tuple)
    build_input, planned_surfaces = plan_representation_contribution_geometry(
        build_input, representation_context
    )
    surfaces = surface_ledger(planned_surfaces)
    contribution_catalog = merged_contribution_catalog(surfaces)

    source = source_payload(build_input, prepared_stack, source_dbu_um=source_dbu_um)
    _attach_lumped_supports(source, build_input, rlc_tuple)

    regions_by_id = {item["semantic_id"]: item for item in source["solution_regions"]}
    materials = source["materials"]
    support_bindings = bind_mask_support(
        surfaces, contribution_tuple, regions_by_id, materials
    )
    resolved_surfaces = resolve_contributions(
        contribution_tuple, surfaces, regions_by_id, materials
    )
    resolved_surfaces = _bind_physical_support_masks(
        resolved_surfaces, support_bindings, source_dbu_um=source_dbu_um
    )

    # The model hash covers what defines the solved model; the source hash also
    # covers which Surface-EPR contributions were requested and how they bind.
    model_input = {
        "modeling": modeling,
        "source": source,
        "junctions": [item.to_payload() for item in junction_tuple],
        **(
            {"lumped_rlcs": [item.to_payload() for item in rlc_tuple]}
            if rlc_tuple
            else {}
        ),
    }
    digest_input = {
        **model_input,
        "contribution_catalog": contribution_catalog,
        "contributions": [item.to_payload() for item in contribution_tuple],
        "surface_bindings": resolved_surfaces,
    }
    return PreparedPlanarGeometry(
        modeling=modeling,
        source=source,
        junctions=junction_tuple,
        lumped_rlcs=rlc_tuple,
        contribution_catalog=contribution_catalog,
        contributions=contribution_tuple,
        surface_bindings=tuple(resolved_surfaces),
        model_sha256=canonical_sha256(model_input),
        source_sha256=canonical_sha256(digest_input),
    )


def _attach_lumped_supports(source, build_input, rlc_tuple) -> None:
    """Add the neutral lumped supports to `source`; every RLC must use one of them."""
    if build_input.lumped_supports:
        source["lumped_supports"] = prepared_lumped_supports(
            build_input.lumped_supports, source
        )
    support_ids = {item["support_id"] for item in source.get("lumped_supports", ())}
    if any(item.support_id not in support_ids for item in rlc_tuple):
        raise ValueError("RLC treatment references an undeclared neutral support")


def _check_planar_inputs(build_input, prepared_stack) -> None:
    if not isinstance(build_input, GeometryBuildInput):
        raise TypeError("build_input must be GeometryBuildInput")
    if build_input.boundary_curves or build_input.boundary_reconstruction:
        raise NotImplementedError(
            "AEDT planar lowering does not support active source curve intent"
        )
    if not isinstance(prepared_stack, Mapping):
        raise TypeError("prepared_stack must be a mapping")


def _resolved_source_dbu_um(build_input, source_dbu_um) -> float:
    """Take the source database unit from the geometry, or check the explicit one."""
    recorded_dbu = build_input.metadata.get("source_dbu_um")
    if source_dbu_um is None:
        source_dbu_um = recorded_dbu
    elif recorded_dbu is not None and source_dbu_um != recorded_dbu:
        raise ValueError("explicit source_dbu_um differs from geometry source DBU")
    if (
        isinstance(source_dbu_um, bool)
        or not isinstance(source_dbu_um, (int, float))
        or not math.isfinite(float(source_dbu_um))
        or source_dbu_um <= 0
    ):
        raise ValueError("planar EPR requires a finite positive source_dbu_um")
    return float(source_dbu_um)


def _declared_treatments(lumped_rlcs, junctions, contributions):
    """Type-check the declared RLCs, junctions and contributions and freeze them."""
    from scgsim.aedt.specs.common import LumpedRlc

    rlc_tuple = tuple(lumped_rlcs)
    if any(not isinstance(item, LumpedRlc) for item in rlc_tuple):
        raise TypeError("lumped_rlcs must contain LumpedRlc records")
    if len({item.support_id for item in rlc_tuple}) != len(rlc_tuple):
        raise ValueError("one support cannot own multiple RLC treatments")
    junction_tuple = tuple(junctions)
    contribution_tuple = tuple(contributions)
    if any(not isinstance(item, PlanarJunction) for item in junction_tuple):
        raise TypeError("junctions must contain PlanarJunction records")
    if any(not isinstance(item, SurfaceEprSpec) for item in contribution_tuple):
        raise TypeError("contributions must contain SurfaceEprSpec records")
    if any(item.field_side == "sidewall" for item in contribution_tuple):
        raise ValueError(
            "sidewall Surface-EPR is excluded; reprepare with horizontal "
            "top/bottom contributions"
        )
    return rlc_tuple, junction_tuple, contribution_tuple


def _check_declared_treatments(build_input, junction_tuple, contribution_tuple) -> None:
    """Require unique ids, and junctions that sit on one authored rectangular sheet."""
    polygons = {item.polygon_id for item in build_input.polygons}
    junction_region_ids = {
        item.source_polygon_id for item in build_input.port_sheet_regions
    }
    ids = [item.junction_id for item in junction_tuple]
    if len(ids) != len(set(ids)):
        raise ValueError("junction ids must be unique")
    contribution_ids = [item.contribution_id for item in contribution_tuple]
    if len(contribution_ids) != len(set(contribution_ids)):
        raise ValueError("surface contribution ids must be unique")
    for item in junction_tuple:
        if item.source_polygon_id not in polygons | junction_region_ids:
            raise ValueError(
                f"junction {item.junction_id!r} references unknown polygon "
                f"{item.source_polygon_id!r}"
            )
        authored = next(
            (p for p in build_input.polygons if p.polygon_id == item.source_polygon_id),
            None,
        )
        if authored is None:
            authored = next(
                r
                for r in build_input.port_sheet_regions
                if r.source_polygon_id == item.source_polygon_id
            )
        junction_terminal_line(
            {"exterior": authored.exterior, "holes": authored.holes}, item, 0.0
        )


def _with_junction_planes(build_input, junction_tuple):
    """Record each partitioned junction's Z plane in its partition record.

    The plane comes from the source Entities that own the junction ends, after
    checking that the central polygon is still a valid terminal line.
    """
    final_polygons = {p.polygon_id: p for p in build_input.polygons}
    for junction in junction_tuple:
        record = build_input.metadata["aedt_junction_partitions"][junction.junction_id]
        polygon = final_polygons[record["central_polygon_id"]]
        junction_terminal_line(
            {"exterior": polygon.exterior, "holes": polygon.holes}, junction, 0.0
        )
    partition_source = {
        "conductors": [
            {
                "semantic_id": e.semantic_id,
                "source_semantic_id": e.metadata.get(
                    "source_semantic_id",
                    e.metadata.get("semantic_group_id", e.semantic_id),
                )
                if "split_polygon_index" in e.metadata
                else e.semantic_id,
                "representation": e.metadata["physical_representation"][
                    "representation"
                ],
                "physical_layer": native_physical_record(
                    e.metadata["physical_representation"]
                ),
                "geometry": e.geometry,
            }
            for e in build_input.entities
            if e.material_kind == "conductor"
        ],
    }
    records = plain_geometry_source(build_input.metadata["aedt_junction_partitions"])
    for record in records.values():
        record["z_um"] = _junction_partition_plane(partition_source, record)
    return replace(
        build_input,
        metadata={**build_input.metadata, "aedt_junction_partitions": records},
    )


def validate_geometry_workers(geometry_workers: int | None) -> None:
    """Require a positive integer worker count, or None for the default."""
    if geometry_workers is not None and (
        isinstance(geometry_workers, bool)
        or not isinstance(geometry_workers, int)
        or geometry_workers <= 0
    ):
        raise ValueError("geometry_workers must be a positive integer or None")


def _bind_physical_support_masks(
    bindings: list[dict[str, Any]],
    support_bindings: list[dict[str, Any]],
    *,
    source_dbu_um: float,
) -> list[dict[str, Any]]:
    if not bindings:
        return []
    import klayout.db as k

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for binding in support_bindings:
        groups.setdefault(plane_group_key(binding), []).append(binding)
    support_by_group: dict[
        tuple[Any, ...],
        tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]],
    ] = {}
    for key, group in groups.items():
        plane = group[0]["mask_plane"]
        projected = [project_plane_region(item["mask_plane"], plane) for item in group]
        union = klayout_region(projected, source_dbu_um, k)
        if union.is_empty():
            raise RuntimeError("physical EPR support union is empty")
        support_by_group[key] = (
            plane,
            regions_from_klayout(union, source_dbu_um),
            group,
        )
    result: list[dict[str, Any]] = []
    for binding in bindings:
        support = support_by_group.get(plane_group_key(binding))
        if support is None:
            raise RuntimeError("selected EPR surface lacks complete physical support")
        plane, support_regions, group = support
        result.append(
            {
                **binding,
                "mask_plane": {**plane, "exterior": [], "holes": []},
                "mask_support": {
                    "group_identity_sha256": canonical_sha256(
                        {
                            "plane": plane,
                            "classification": binding["contribution"]["classification"],
                            "side": binding["contribution"]["side"],
                            "material_id": binding["effective_material_id"],
                            "members": sorted(item["binding_id"] for item in group),
                        }
                    ),
                    "attribution_regions": [
                        project_plane_region(binding["mask_plane"], plane)
                    ],
                    # Offset the complete union before intersecting owner
                    # attribution; internal owner seams are not boundaries.
                    "support_regions": support_regions,
                    "source_dbu_um": source_dbu_um,
                    "inset_method": INSET_METHOD,
                    "circle_points": INSET_CIRCLE_POINTS,
                    "klayout_version": version("klayout"),
                },
            }
        )
    return result


def _klayout_point(point: Sequence[float], dbu_um: float, k: Any) -> Any:
    if len(point) != 2:
        raise ValueError("EPR mask point must have two local coordinates")
    values = [float(value) for value in point]
    if any(
        not math.isfinite(value)
        or not KLAYOUT_COORD_MIN < value / dbu_um < KLAYOUT_COORD_MAX
        for value in values
    ):
        raise OverflowError("EPR mask coordinate exceeds KLayout integer range")
    return k.DPoint(*values).to_itype(dbu_um)


def klayout_region(regions: Sequence[Mapping[str, Any]], dbu_um: float, k: Any) -> Any:
    """Merged KLayout Region built from exterior and hole rings in micrometres."""
    result = k.Region()
    for region in regions:
        exterior = [_klayout_point(point, dbu_um, k) for point in region["exterior"]]
        polygon = k.Polygon(exterior)
        for hole in region.get("holes", ()):
            polygon.insert_hole([_klayout_point(point, dbu_um, k) for point in hole])
        result.insert(polygon)
    return result.merged()


def regions_from_klayout(region: Any, dbu_um: float) -> list[dict[str, Any]]:
    # Split point contacts without changing the integer polygon set. Iterate
    # stored pieces so implicit merging cannot recreate a self-touching contour.
    """Convert a KLayout Region to exterior/hole dicts in micrometres, hash-sorted."""
    simple = region.merged(True, 1)

    def points(iterator: Any) -> list[list[float]]:
        return [[point.x * dbu_um, point.y * dbu_um] for point in iterator]

    result = [
        {
            "exterior": points(polygon.each_point_hull()),
            "holes": [
                points(polygon.each_point_hole(index))
                for index in range(polygon.holes())
            ],
        }
        for polygon in simple.each()
    ]
    return sorted(result, key=canonical_sha256)


def _junction_partition_plane(
    source: Mapping[str, Any], record: Mapping[str, Any]
) -> float:
    entities = {item["semantic_id"]: item for item in source["conductors"]}
    owners = [entities[end["source_entity_id"]] for end in record["ends"]]
    positions = [
        _route_a_sheet_z(source, owner)
        if owner["representation"] in {"surface_sheet", "sheet"}
        else entity_z_range(owner)[0]
        for owner in owners
    ]
    if len(set(positions)) != 1:
        raise ValueError(
            "junction source Entity planes differ; no thickness or plane fallback"
        )
    for owner in owners:
        if (
            owner["representation"] not in {"surface_sheet", "sheet"}
            and entity_z_range(owner)[1] <= positions[0]
        ):
            raise ValueError("junction Solid source Entity has no positive thickness")
    return positions[0]


def junction_polygon(
    source: Mapping[str, Any], junction: PlanarJunction, polygons: Mapping[str, Any]
):
    """Return the junction polygon and its partition record (None if unpartitioned)."""
    record = source.get("junction_partitions", {}).get(junction.junction_id)
    if record is None:
        # Historical prepared geometry retains its recorded authored sheet.
        return polygons[junction.source_polygon_id], None
    if record.get("method") != "scgsim.aedt.junction-partition.v1":
        raise ValueError("unknown junction partition method")
    if record["authored_polygon_id"] != junction.source_polygon_id:
        raise ValueError("junction partition authored source identity differs")
    return polygons[record["central_polygon_id"]], record


def junction_terminal_line(
    polygon: Mapping[str, Any], junction: PlanarJunction, z_um: float
) -> tuple[list[list[float]], float]:
    """Check a junction polygon is a rectangle along its direction; return its line."""
    exterior = tuple(
        (float(point[0]), float(point[1])) for point in polygon["exterior"]
    )
    if len(exterior) != 4 or polygon.get("holes"):
        raise ValueError(
            f"junction {junction.junction_id!r} requires one rectangular polygon without holes"
        )
    dx, dy = junction.direction_xy
    px, py = -dy, dx
    along = tuple(x * dx + y * dy for x, y in exterior)
    across = tuple(x * px + y * py for x, y in exterior)
    along_min, along_max = min(along), max(along)
    across_min, across_max = min(across), max(across)
    tolerance = 1e-9 * max(
        1.0, abs(along_min), abs(along_max), abs(across_min), abs(across_max)
    )
    if along_max - along_min <= tolerance or across_max - across_min <= tolerance:
        raise ValueError(
            f"junction {junction.junction_id!r} polygon has empty projected extent"
        )
    expected = {
        (along_min, across_min),
        (along_min, across_max),
        (along_max, across_min),
        (along_max, across_max),
    }
    if any(
        not any(
            abs(u - eu) <= tolerance and abs(v - ev) <= tolerance for eu, ev in expected
        )
        for u, v in zip(along, across, strict=False)
    ):
        raise ValueError(
            f"junction {junction.junction_id!r} polygon is not rectangular in its declared direction"
        )
    width = across_max - across_min
    if not math.isclose(width, junction.width_um, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(
            f"junction {junction.junction_id!r} width disagrees with its polygon"
        )
    across_mid = (across_min + across_max) / 2.0
    line = [
        [along_min * dx + across_mid * px, along_min * dy + across_mid * py, z_um],
        [along_max * dx + across_mid * px, along_max * dy + across_mid * py, z_um],
    ]
    return line, along_max - along_min


def planar_junction_from_port(  # noqa: PLR0913 - public keyword-only API
    build_input: GeometryBuildInput,
    *,
    port_name: str,
    junction_id: str,
    terminal_a_net: str,
    terminal_b_net: str,
    inductance_h: float,
    capacitance_f: float,
) -> PlanarJunction:
    """Bind an explicit circuit junction to one normalized authored port sheet."""

    if not isinstance(build_input, GeometryBuildInput):
        raise TypeError("build_input must be GeometryBuildInput")
    if not isinstance(port_name, str) or not port_name:
        raise ValueError("port_name must be non-empty text")
    validate_geometry_input(build_input)
    matches = tuple(
        region
        for region in build_input.port_sheet_regions
        if region.metadata.get("source_name") == port_name
    )
    if len(matches) != 1:
        raise ValueError(
            f"port {port_name!r} requires exactly one normalized source sheet; "
            f"found {len(matches)}"
        )
    region = matches[0]

    # Adapter overlap inventories contain positive area only. Binding also owns
    # complete terminal-edge contact, and therefore checks the closed XY source.
    def source_rings(exterior, holes):
        return tuple(
            tuple(tuple(Fraction(str(float(v))) for v in point) for point in ring)
            for ring in (exterior, *holes)
        )

    sheet = source_rings(region.exterior, region.holes)
    polygons = {polygon.polygon_id: polygon for polygon in build_input.polygons}
    overlap_nets = set()
    for entity in build_input.entities:
        if entity.material_kind != "conductor":
            continue
        for pid in entity.polygon_ids:
            if pid == region.source_polygon_id:
                continue
            polygon = polygons[pid]
            if closed_contact([sheet], [source_rings(polygon.exterior, polygon.holes)]):
                overlap_nets.add(entity.net_id)
    if overlap_nets != {terminal_a_net, terminal_b_net}:
        raise ValueError(
            f"port {port_name!r} overlaps must identify exactly the explicit "
            "terminal A and B Nets"
        )
    direction = region.metadata["direction"]
    dx, dy = float(direction[0]), float(direction[1])
    magnitude = math.hypot(dx, dy)
    dx, dy = dx / magnitude, dy / magnitude
    across = tuple(-dy * x + dx * y for x, y in region.exterior)
    width_um = max(across) - min(across)
    junction = PlanarJunction(
        junction_id=junction_id,
        source_polygon_id=region.source_polygon_id,
        terminal_a_net=terminal_a_net,
        terminal_b_net=terminal_b_net,
        direction_xy=(dx, dy),
        width_um=width_um,
        inductance_h=inductance_h,
        capacitance_f=capacitance_f,
        metadata={"source_port_name": port_name},
    )
    junction_terminal_line(
        {"exterior": region.exterior, "holes": region.holes}, junction, 0.0
    )
    return junction


def _route_a_sheet_z(source: Mapping[str, Any], entity: Mapping[str, Any]) -> float:
    effective = entity.get("physical_layer")
    if isinstance(effective, Mapping) and effective.get("representation") == "sheet":
        return float(effective["effective_z_min_um"])
    profile = source.get("route_a_thin_film")
    if not isinstance(profile, Mapping):
        raise ValueError("Route A conductor lacks thin-film profile provenance")
    ranges = profile.get("physical_face_metal_z_ranges_um")
    positions = profile.get("effective_sheet_z_um")
    if not isinstance(ranges, Mapping) or not isinstance(positions, Mapping):
        raise ValueError("Route A thin-film profile is incomplete")
    source_id = entity["source_semantic_id"]
    sides = tuple(
        str(side)
        for side, record in ranges.items()
        if isinstance(record, Mapping) and source_id in record.get("semantic_ids", ())
    )
    if len(sides) != 1:
        raise ValueError(
            f"Route A conductor {entity['semantic_id']!r} has no unique sheet side"
        )
    value = positions.get(sides[0])
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(
            f"Route A conductor {entity['semantic_id']!r} has no finite effective sheet Z"
        )
    return float(value)


def entity_z_range(entity: Mapping[str, Any]) -> tuple[float, float]:
    """Return the (min, max) Z range in micrometres of an entity's geometry."""
    return geometry_z_range(entity.get("geometry"), str(entity.get("semantic_id")))


def native_name(kind: str, *identity: Any) -> str:
    """Return a stable AEDT-safe name while retaining raw ids only as metadata."""

    if not isinstance(kind, str) or not kind.isidentifier():
        raise ValueError("native name kind must be an identifier")
    digest = canonical_sha256({"kind": kind, "identity": list(identity)})[:24]
    return f"scgsim_{kind}_{digest}"


def native_entity_name(source: Mapping[str, Any], kind: str, *identity: Any) -> str:
    """Keep historical hash names while giving new Plan entities readable names."""

    policy = source.get("native_entity_name_policy", "hash.v1")
    if policy == "hash.v1":
        return native_name(kind, *identity)
    if policy != "readable.v1":
        raise ValueError("unknown native entity name policy")
    raw = "__".join(str(value) for value in identity)
    label = "".join(
        character if character.isascii() and character.isalnum() else "_"
        for character in raw
    ).strip("_")
    if not label:
        raise ValueError("native entity name has no readable identity")
    digest = canonical_sha256({"kind": kind, "identity": list(identity)})[:10]
    prefix = f"SCGSim_{kind}_"
    label = label[: 60 - len(prefix) - len(digest) - 1].rstrip("_")
    return f"{prefix}{label}_{digest}"
