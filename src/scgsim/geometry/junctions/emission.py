"""Plan emitted junction geometry, realize it on the input grid, and check its source-bound support."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import replace
from fractions import Fraction
from typing import cast

from scgsim.geometry.grid.planar_loops import (
    cancel_reversed_planar_edges,
    simple_planar_loops_from_edges,
)
from scgsim.geometry.junctions.core import PRECISION_UM, canonical_sha256, plain_record
from scgsim.geometry.junctions.exact_geometry import (
    closed_contact,
    cross,
    inside_ring,
    region_edges,
)
from scgsim.geometry.junctions.model_checks import (
    emitted_partition_coverage,
    fraction_area_overlap,
    fraction_boundary_contact,
    model_edge_contacts,
    model_edge_covered,
    model_edge_section,
)
from scgsim.geometry.junctions.model_state import JunctionModel
from scgsim.geometry.junctions.terminal import junction_terminal_line
from scgsim.geometry.models.grid import geometry_grid_context
from scgsim.geometry.models.input import GeometryBuildInput, LayoutPolygonSpec
from scgsim.geometry.models.junctions import (
    JUNCTION_END_PIECE_PART,
    junction_net_id,
)
from scgsim.geometry.source.grid_projection import apply_geometry_grid


def initialize_registry(state: JunctionModel) -> None:
    """Initialize shared exact-to-float coordinate identity."""
    state.registry, state.reverse = ({}, {})


def rebuild_original_polygons(state: JunctionModel) -> None:
    """Reinsert registered source nodes into each affected authored polygon."""
    state.affected_polygons = state.authored | {
        pid
        for entity in state.build_input.entities
        if entity.semantic_id in state.affected_owners
        for pid in entity.polygon_ids
    }
    state.model = {}
    for pid, polygon in state.originals.items():
        if pid not in state.affected_polygons:
            state.model[pid] = polygon
            continue
        region = []
        for index, ring in enumerate(state.rings[pid]):
            loop = []
            for n in range(len(ring)):
                key = (pid, index, n)
                loop.extend(
                    (
                        state.emit(point)
                        for point, _ in sorted(
                            state.events[key].items(), key=lambda item: item[1]
                        )[:-1]
                    )
                )
            region.append(tuple(loop))
        state.model[pid] = replace(polygon, exterior=region[0], holes=tuple(region[1:]))


def emit_partitions(state: JunctionModel) -> None:
    """Emit central and terminal pieces with their exact source provenance.

    A landing end piece (its terminal is rebuilt with exact arcs) is the drawn sheet beyond
    its cut; any other end piece is the free part of the sheet there.  The free part is what
    rebuilds the terminal's support outline: the terminal joined with either is the same region.
    """
    state.additions, state.appended = ([], defaultdict(list))
    state.physical_ends = defaultdict(list)
    owners = {entity.semantic_id: entity for entity in state.build_input.entities}
    for jid, record in state.records.items():
        # Lowering registers emitted points, so the slices are lowered in this order.
        parts = [state.lower(state.refine(part)) for part in state.sliced[jid]]
        chosen = [
            state.sheet_sliced[jid][k] if end["landing"] else state.sliced[jid][k]
            for k, end in ((0, record["ends"][0]), (2, record["ends"][1]))
        ]
        ends = [
            state.lower(state.refine(chosen[0]), simplify=record["ends"][0]["landing"]),
            None,
            state.lower(state.refine(chosen[1]), simplify=record["ends"][1]["landing"]),
        ]
        corners = _central_corners(state, jid, record)
        sheet = state.originals[record["authored_polygon_id"]]
        state.additions.append(
            LayoutPolygonSpec(
                record["central_polygon_id"],
                sheet.layer,
                corners,
                metadata={
                    state.output_format.junction_metadata_key: jid,
                    "part": "central_rlc",
                },
            )
        )
        _emit_end_pieces(state, jid, record, (ends[0], ends[2]), owners)
        final_free, final_partition = _checked_partition_coverage(
            state,
            jid,
            chosen,
            [*parts[0], (corners,), *parts[2]],
            [*ends[0], (corners,), *ends[2]],
        )
        _record_emitted_partition(state, record, corners, final_free, final_partition)
    state.model.update({p.polygon_id: p for p in state.additions})


def _central_corners(state: JunctionModel, jid: str, record) -> tuple:
    """The central rectangle's four corners in model coordinates, in its local order along the junction."""
    central = state.lower(state.refine(state.sliced[jid][1]), simplify=True)
    if len(central) != 1 or len(central[0]) != 1 or len(central[0][0]) != 4:
        raise ValueError(
            "numerical junction final central remainder is not a four-corner rectangle"
        )
    dx, dy = record["direction_xy"]
    uv = [(x * dx + y * dy, -x * dy + y * dx) for x, y in central[0][0]]
    amin, amax = (min((u for u, _ in uv)), max((u for u, _ in uv)))
    vmin, vmax = (min((v for _, v in uv)), max((v for _, v in uv)))
    return tuple(
        (
            min(
                zip(central[0][0], uv, strict=False),
                key=lambda item: (item[1][0] - u) ** 2 + (item[1][1] - v) ** 2,
            )[0]
            for u, v in ((amin, vmin), (amax, vmin), (amax, vmax), (amin, vmax))
        )
    )


def _emit_end_pieces(state: JunctionModel, jid: str, record, ends, owners) -> None:
    """Add each nonempty end piece to its terminal's Entity, which must match the local selection."""
    sheet = state.originals[record["authored_polygon_id"]]
    for n, part in ((0, ends[0]), (1, ends[1])):
        end = record["ends"][n]
        if len(part) > 1:
            raise ValueError(
                "unsupported junction end topology in final model coordinates"
            )
        if bool(part) != (end["polygon_id"] is not None):
            raise ValueError(
                "numerical junction zero-end residual differs from selected local proof"
            )
        if part:
            state.additions.append(
                LayoutPolygonSpec(
                    end["polygon_id"],
                    sheet.layer,
                    part[0][0],
                    tuple(part[0][1:]),
                    net_name=owners[end["source_entity_id"]].net_id,
                    metadata={
                        state.output_format.junction_metadata_key: jid,
                        "source_entity_id": end["source_entity_id"],
                        "part": JUNCTION_END_PIECE_PART,
                    },
                )
            )
            state.appended[end["source_entity_id"]].append(end["polygon_id"])
            state.physical_ends[end["source_entity_id"]].extend(
                state.refine(state.sliced[jid][0 if n == 0 else 2])
            )
        end["model_regions"] = plain_record(part)


def _checked_partition_coverage(
    state: JunctionModel, jid: str, chosen, emitted_slices, emitted_pieces
) -> tuple:
    """The emitted free boundary and partition, each refused unless the emitted pieces cover it exactly.

    ``emitted_slices`` are the free part's slices with the central rectangle; ``emitted_pieces``
    the end pieces with it.
    """
    final_free = state.lower(state.refine(state.free_edges[jid]))
    if not emitted_partition_coverage(final_free, emitted_slices):
        raise ValueError(
            "numerical junction emitted partition does not cover source free boundary"
        )
    final_partition = state.lower(
        state.refine([*chosen[0], *state.sliced[jid][1], *chosen[1]])
    )
    if not emitted_partition_coverage(final_partition, emitted_pieces):
        raise ValueError(
            "numerical junction emitted end pieces and rectangle differ from their slices"
        )
    return final_free, final_partition


def _record_emitted_partition(
    state: JunctionModel, record, corners, final_free, final_partition
) -> None:
    """Write what the emission produced and how into the junction's partition record."""
    dx, dy = record["direction_xy"]
    sheet = state.originals[record["authored_polygon_id"]]
    record["free_model_regions"] = plain_record(final_free)
    record["partition_model_regions"] = plain_record(final_partition)
    record["central_model_regions"] = [[list(corners)]]
    record["proof_conditions"]["emitted_partition_coverage"] = True
    record["model_coordinate_method"] = "source_edge_incidence_and_two_cuts.v1"
    record["model_boolean_precision_um"] = PRECISION_UM
    record["maximum_model_source_coordinate_deviation_um"] = max(
        (
            abs(round(v / PRECISION_UM) * PRECISION_UM - v)
            for pid in state.needed
            for ring in (state.originals[pid].exterior, *state.originals[pid].holes)
            for point in ring
            for v in point
        )
    )
    record["maximum_model_intersection_coordinate_deviation_um"] = (
        state.intersection_deviation
    )
    record["authored_source_geometry"] = {
        "exterior": plain_record(sheet.exterior),
        "holes": plain_record(sheet.holes),
    }
    record["model_cut_projection_ranges_um"] = [
        [
            min((x * dx + y * dy for x, y in edge)),
            max((x * dx + y * dy for x, y in edge)),
        ]
        for edge in ((corners[3], corners[0]), (corners[1], corners[2]))
    ]


def rebuild_entity_support(state: JunctionModel) -> None:
    """Rebuild affected Entity supports from the exact emitted pieces."""
    state.entities = []
    for entity in state.build_input.entities:
        ids = tuple((pid for pid in entity.polygon_ids if pid not in state.authored))
        if not ids and entity.polygon_ids:
            continue
        if entity.material_kind != "conductor":
            state.entities.append(entity)
            continue
        if (
            entity.semantic_id not in state.affected_owners
            and ids == entity.polygon_ids
        ):
            state.entities.append(entity)
            continue
        loops = _support_loops(state, entity.semantic_id, ids)
        outer = [
            ring
            for ring in loops
            if not any(
                (
                    inside_ring(
                        tuple((Fraction(str(v)) for v in ring[0])),
                        tuple((tuple((Fraction(str(v)) for v in p)) for p in other)),
                    )
                    == 1
                    for other in loops
                    if other is not ring
                )
            )
        ]
        if len(outer) != 1:
            raise ValueError(
                "junction source Entity support cannot be represented as one connected normalized region"
            )
        geometry = {
            **entity.geometry,
            "outer_loop": outer[0],
            "hole_loops": tuple((ring for ring in loops if ring is not outer[0])),
        }
        state.entities.append(
            replace(
                entity,
                polygon_ids=(*ids, *state.appended[entity.semantic_id]),
                geometry=geometry,
            )
        )


def _support_loops(state: JunctionModel, entity_id: str, ids) -> tuple:
    """The boundary loops of an Entity's kept polygons joined with the physical ends it gained."""
    emitted_support = []
    for pid in ids:
        for index, ring in enumerate(
            (state.model[pid].exterior, *state.model[pid].holes)
        ):
            area = sum((a[0] * b[1] - b[0] * a[1] for a, b in region_edges((ring,))))
            oriented = ring if (area > 0) == (index == 0) else tuple(reversed(ring))
            emitted_support.extend(region_edges((oriented,)))
    emitted_support.extend(
        ((state.emit(a), state.emit(b)) for a, b, _ in state.physical_ends[entity_id])
    )
    boundary = cancel_reversed_planar_edges(emitted_support)
    return simple_planar_loops_from_edges(boundary)


def check_emitted_partitions(state: JunctionModel) -> None:
    """Check emitted coordinates against every terminal and open-side contract.

    For each junction: no conductor enters the central rectangle, each end piece meets
    its own arm and no other net and covers its end edge, and no conductor touches
    the rectangle's two open sides.
    """
    state.actual_rings = cast(
        dict[str, tuple[tuple[tuple[Fraction, Fraction], ...], ...]],
        {
            pid: tuple(
                (
                    tuple((tuple((Fraction(str(v)) for v in point)) for point in ring))
                    for ring in (polygon.exterior, *polygon.holes)
                )
            )
            for pid, polygon in state.model.items()
        },
    )
    for record in state.records.values():
        conductors = [
            (junction_net_id(e), pid, state.actual_rings[pid])
            for e in state.entities
            if e.material_kind == "conductor"
            and e.semantic_id in record["support_plane"]["eligible_entity_ids"]
            for pid in e.polygon_ids
        ]
        central = state.actual_rings[record["central_polygon_id"]][0]
        _check_central_clear(central, conductors)
        for end in record["ends"]:
            _check_emitted_end(state, end, central, conductors)
        for start, finish in ((central[0], central[1]), (central[2], central[3])):
            for _net, _, region in conductors:
                intervals, points = model_edge_section(region, start, finish)
                if intervals or any((0 < point < 1 for point in points)):
                    raise ValueError("final cross-junction open-side contact")
        record["proof_conditions"]["all_derived_junctions_checked"] = True


def _strictly_inside(point, region) -> bool:
    return inside_ring(point, region[0]) == 1 and all(
        (inside_ring(point, hole) == -1 for hole in region[1:])
    )


def _check_central_clear(central, conductors) -> None:
    """Refuse a conductor that reaches into the central rectangle or crosses its edges."""
    midpoint = tuple((sum((point[i] for point in central)) / 4 for i in (0, 1)))
    for _, _, region in conductors:
        if (
            any((_strictly_inside(point, (central,)) for point in region[0]))
            or any((_strictly_inside(point, region) for point in (*central, midpoint)))
            or any(
                (
                    cross(a, b, u) * cross(a, b, v) < 0
                    and cross(u, v, a) * cross(u, v, b) < 0
                    for a, b in region_edges(region)
                    for u, v in region_edges((central,))
                )
            )
        ):
            raise ValueError("final cross-junction PEC intersects RLC interior")


def _check_emitted_end(state: JunctionModel, end, central, conductors) -> None:
    """Refuse an end piece off its own arm or on another net, or an end edge its net leaves open."""
    own = [region for net, _, region in conductors if net == end["net_id"]]
    other = [region for net, _, region in conductors if net != end["net_id"]]
    if end["polygon_id"]:
        piece = state.actual_rings[end["polygon_id"]]
        arms = [
            state.actual_rings[pid]
            for e in state.entities
            if e.semantic_id == end["source_entity_id"]
            for pid in e.polygon_ids
            if pid
            not in {
                part["polygon_id"] for r in state.records.values() for part in r["ends"]
            }
        ]
        if not (
            fraction_boundary_contact([piece], arms)
            or fraction_area_overlap(state.scope, [piece], arms)
        ):
            raise ValueError("final model end lacks positive source-arm contact")
        if closed_contact([piece], other):
            raise ValueError("final cross-junction end contacts another net")
    n = 0 if end["side"] == "A" else 1
    start, finish = (central[3], central[0]) if n == 0 else (central[1], central[2])
    if not model_edge_covered(own, start, finish):
        raise ValueError("final model designated end edge is not wholly covered")
    if model_edge_contacts(other, start, finish):
        raise ValueError("final cross-junction wrong-net end/corner contact")


def realized_on_grid(build_input, result, junctions, records):
    """The emitted input realized on the input grid, each central rectangle and partition checked again.

    A failure keeps the junction and its grid in a note.
    """
    context = geometry_grid_context(build_input)
    # The emitted input was never read from a layout here, so it brings no source audits.
    result = _straight_end_pieces(
        apply_geometry_grid(result, context, source_audits=()), records
    )
    actual = {p.polygon_id: p for p in result.polygons}
    for junction in junctions:
        record = records[junction.junction_id]
        polygon = actual[record["central_polygon_id"]]
        try:
            junction_terminal_line(
                {"exterior": polygon.exterior, "holes": polygon.holes},
                junction,
                record["z_um"],
            )
            ends = [
                actual[end["polygon_id"]] for end in record["ends"] if end["polygon_id"]
            ]
            emitted = [(p.exterior, *p.holes) for p in ends]
            emitted.append((polygon.exterior, *polygon.holes))
            if not emitted_partition_coverage(
                record["partition_model_regions"], emitted
            ):
                raise ValueError("junction grid realization changes partition coverage")
        except ValueError as error:
            error.add_note(
                json.dumps(
                    {
                        "junction_id": junction.junction_id,
                        "geometry_grid": context.to_payload(),
                        "requested_direction_xy": list(junction.direction_xy),
                        "requested_width_um": junction.width_um,
                        "realization_status": "incompatible_with_existing_shape_contract",
                    },
                    sort_keys=True,
                )
            )
            raise
        record["geometry_grid"] = context.to_payload()
        record["requested_central_model_regions"] = record["central_model_regions"]
        record["central_model_regions"] = [
            [list(polygon.exterior), *map(list, polygon.holes)]
        ]
        record["realized_central_geometry_sha256"] = canonical_sha256(
            {"exterior": polygon.exterior, "holes": polygon.holes}
        )
        for end in record["ends"]:
            if end["polygon_id"]:
                piece = actual[end["polygon_id"]]
                end["requested_model_regions"] = end["model_regions"]
                end["model_regions"] = [[list(piece.exterior), *map(list, piece.holes)]]
    return result


def _straight_end_pieces(result: GeometryBuildInput, records) -> GeometryBuildInput:
    """The grid-realized input with every landing end piece cleared of collinear vertices.

    The grid realization nodes a polygon wherever another polygon's vertex lies on its edge.
    A landing end piece overlaps its terminal, so the drawn terminal's vertices on the sheet
    sides land on the end piece's side edges.  SGB would keep each as a corner between two
    carriers and split the side wall standing there; without them the region is the same.
    """
    ids = {
        end["polygon_id"]
        for record in records.values()
        for end in record["ends"]
        if end["polygon_id"] and end["landing"]
    }
    return replace(
        result,
        polygons=tuple(
            replace(
                polygon,
                exterior=_without_collinear(polygon.exterior),
                holes=tuple(_without_collinear(hole) for hole in polygon.holes),
            )
            if polygon.polygon_id in ids
            else polygon
            for polygon in result.polygons
        ),
    )


def _without_collinear(ring):
    exact = [tuple(Fraction(str(v)) for v in point) for point in ring]
    count = len(ring)
    return tuple(
        ring[n]
        for n in range(count)
        if cross(exact[n - 1], exact[n], exact[(n + 1) % count]) != 0
    )
