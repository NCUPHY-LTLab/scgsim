"""Create live polygon distance partitions before shared topology and volumes."""

from dataclasses import replace
from math import hypot

from scgsim.geometry.models.construction import SurfacePartitionRecord
from scgsim.geometry.models.input import GeometryBuildInput
from scgsim.geometry.models.surface_bands import (
    SourceBoundarySegment,
    SurfaceDistanceBandSpec,
)
from scgsim.geometry.models.topology import SurfacePlanRecord


def _simple_boundary_loops(points):
    """Split exact grid-collinear cutlines and point-contact boundary cycles."""
    from decimal import Decimal
    from itertools import pairwise

    vertices = tuple(dict.fromkeys(tuple(point) for point in points))

    def decimal(p):
        return tuple(Decimal(str(v)) for v in p)

    edges = []
    for a, b in zip(points, (*points[1:], points[0])):
        a, b = tuple(a), tuple(b)
        ax, ay = decimal(a)
        bx, by = decimal(b)
        along = []
        for p in vertices:
            px, py = decimal(p)
            if (
                (px - ax) * (by - ay) == (py - ay) * (bx - ax)
                and min(ax, bx) <= px <= max(ax, bx)
                and min(ay, by) <= py <= max(ay, by)
            ):
                along.append(p)
        axis = 0 if a[0] != b[0] else 1
        along.sort(key=lambda p: p[axis], reverse=b[axis] < a[axis])
        edges.extend(pairwise(along))
    retained = []
    for edge in edges:
        reverse = edge[::-1]
        if reverse in retained:
            retained.remove(reverse)
        else:
            retained.append(edge)
    from math import atan2, pi

    loops = []
    while retained:
        a, b = retained.pop(0)
        loop = [a]
        while b != loop[0]:
            loop.append(b)
            options = [edge for edge in retained if edge[0] == b]
            if not options:
                raise ValueError("distance band has an open Boolean boundary")
            angle = atan2(b[1] - a[1], b[0] - a[0])
            edge = max(
                options,
                key=lambda edge: (
                    (atan2(edge[1][1] - b[1], edge[1][0] - b[0]) - angle) % (2 * pi)
                ),
            )
            retained.remove(edge)
            a, b = edge
        if len(loop) >= 3:
            loops.append(tuple(loop))
    return tuple(loops)


def _restore_parent_boundary(ring, parent_rings, precision):
    """Intersect generated contour lines with uniquely attributable source edges.

    The only association bound is the caller-declared Boolean grid. It is not
    a nearest-edge selector: multiple eligible source segments fail explicitly.
    """
    from math import sqrt

    from fractions import Fraction

    source_edges = []
    source_edge_vertices = []
    for source in parent_rings:
        # Boolean output may cross several collinear original segments without
        # retaining their contact vertices. Associate the complete exact run,
        # then restore every original vertex below; the parent is unchanged.
        corners = []
        for index, point in enumerate(source):
            previous, following = source[index - 1], source[(index + 1) % len(source)]
            incoming = tuple(Fraction(b) - Fraction(a) for a, b in zip(previous, point))
            outgoing = tuple(Fraction(b) - Fraction(a) for a, b in zip(point, following))
            if (
                incoming[0] * outgoing[1] != incoming[1] * outgoing[0]
                or sum(a * b for a, b in zip(incoming, outgoing)) <= 0
            ):
                corners.append(index)
        if not corners:
            raise ValueError("original parent boundary has no source segment turns")
        for index, start in enumerate(corners):
            stop = corners[(index + 1) % len(corners)]
            vertices = [source[start]]
            cursor = (start + 1) % len(source)
            while cursor != stop:
                vertices.append(source[cursor])
                cursor = (cursor + 1) % len(source)
            vertices.append(source[stop])
            source_edges.append((vertices[0], vertices[-1]))
            source_edge_vertices.append(tuple(vertices))
    bound = precision / sqrt(2)

    def candidates(a, b):
        found = []
        for index, (c, d) in enumerate(source_edges):
            dx, dy = d[0] - c[0], d[1] - c[1]
            length = hypot(dx, dy)

            def incident(p):
                cross = (p[0] - c[0]) * dy - (p[1] - c[1]) * dx
                projection = ((p[0] - c[0]) * dx + (p[1] - c[1]) * dy) / length
                return (
                    abs(cross) / length <= bound
                    and -bound <= projection <= length + bound
                )

            if incident(a) and incident(b):
                found.append(index)
        if len(found) > 1:
            raise ValueError(
                "Boolean boundary lineage is ambiguous at declared precision"
            )
        return found[0] if found else None

    owners = [candidates(a, b) for a, b in zip(ring, (*ring[1:], ring[0]))]
    restored = []
    for index, p in enumerate(ring):
        before, after = owners[index - 1], owners[index]
        if before is None and after is None:
            restored.append(p)
            continue
        if before is not None and before == after:
            # Collinear Boolean vertex on an already identified source segment.
            # Its exact source-line position is retained; no contour intersection.
            a, b = source_edges[before]
            axis = 0 if abs(b[0] - a[0]) >= abs(b[1] - a[1]) else 1
            parameter = (p[axis] - a[axis]) / (b[axis] - a[axis])
            restored.append(
                (a[0] + parameter * (b[0] - a[0]), a[1] + parameter * (b[1] - a[1]))
            )
            continue
        if before is not None and after is not None and before != after:
            common = set(source_edges[before]) & set(source_edges[after])
            if len(common) != 1:
                raise ValueError(
                    "Boolean source boundary junction has no unique original vertex"
                )
            restored.append(common.pop())
            continue
        owner = after if after is not None else before
        a, b = source_edges[owner]
        # The adjacent non-source contour defines the analytic clipping line.
        q = ring[index - 1] if before is None else ring[(index + 1) % len(ring)]
        dx, dy = b[0] - a[0], b[1] - a[1]
        ex, ey = q[0] - p[0], q[1] - p[1]
        determinant = dx * ey - dy * ex
        if determinant == 0:
            if p in (a, b):
                restored.append(p)
                continue
            raise ValueError(
                "parallel Boolean contour cannot establish exact source intersection"
            )
        parameter = ((p[0] - a[0]) * ey - (p[1] - a[1]) * ex) / determinant
        restored.append((a[0] + parameter * dx, a[1] + parameter * dy))
    # Keep original collinear vertices: they may carry external contact topology.
    output = []
    for index, a in enumerate(restored):
        output.append(a)
        owner = owners[index]
        if owner is None:
            continue
        b = restored[(index + 1) % len(restored)]
        axis = 0 if a[0] != b[0] else 1
        for point in sorted(
            {
                point
                for point in source_edge_vertices[owner]
                if min(a[axis], b[axis]) < point[axis] < max(a[axis], b[axis])
            },
            key=lambda p: p[axis],
            reverse=b[axis] < a[axis],
        ):
            output.append(point)
    return tuple(output)


def _rings3d(parent):
    from scgsim.geometry._primitives.spatial import _geometry_ref_surface_z_um

    ref = parent.geometry_ref
    if "planar_loops_3d" in ref:
        return tuple(tuple(map(tuple, ring)) for ring in ref["planar_loops_3d"])
    if "quad_points" in ref:
        return (tuple(map(tuple, ref["quad_points"])),)
    if "outer_loop" not in ref:
        raise NotImplementedError("distance partition requires explicit polygon loops")
    z = _geometry_ref_surface_z_um(ref)
    return tuple(
        tuple((x, y, z) for x, y in ring)
        for ring in (ref["outer_loop"], *ref.get("hole_loops", ()))
    )


def _local_frame(rings):
    from math import sqrt

    origin = rings[0][0]

    def sub(a, b):
        return tuple(x - y for x, y in zip(a, b))

    def dot(a, b):
        return sum(x * y for x, y in zip(a, b))

    def cross(a, b):
        return (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )

    direction = sub(rings[0][1], origin)
    length = sqrt(dot(direction, direction))
    u = tuple(x / length for x in direction)
    normal = next(
        (
            cross(direction, sub(point, origin))
            for point in rings[0][2:]
            if any(cross(direction, sub(point, origin)))
        ),
        None,
    )
    if normal is None:
        raise ValueError("parent polygon is degenerate")
    # Planarity is an input structural property, not an approximation request.
    if any(dot(normal, sub(point, origin)) != 0 for ring in rings for point in ring):
        raise NotImplementedError("nonplanar parent distance partition is unsupported")
    nlength = sqrt(dot(normal, normal))
    n = tuple(x / nlength for x in normal)
    v = cross(n, u)
    local = tuple(
        tuple(
            (dot(sub(point, origin), u), dot(sub(point, origin), v)) for point in ring
        )
        for ring in rings
    )

    def lift(point):
        return tuple(origin[i] + point[0] * u[i] + point[1] * v[i] for i in range(3))

    def project(point):
        return dot(sub(point, origin), u), dot(sub(point, origin), v)

    return local, lift, project


def _parent_source_identity(surface, build_input=None):
    owners = tuple(
        surface.metadata.get("owner_semantic_ids", (surface.owner_semantic_id,))
    )
    polygon_ids = tuple(surface.geometry_ref.get("source_polygon_ids", ()))
    if not polygon_ids and build_input is not None:
        polygon_ids = tuple(
            dict.fromkeys(
                polygon_id
                for entity in build_input.entities
                if entity.semantic_id in owners
                for polygon_id in entity.polygon_ids
            )
        )
    return owners, polygon_ids


def partition_source_boundary(geometry_ref, parent_surface_id):
    """Preserve original interface rings before explicit child replacement."""
    if not any(
        key in geometry_ref for key in ("outer_loop", "quad_points", "planar_loops_3d")
    ):
        return None
    source = SurfacePlanRecord(parent_surface_id, "", "partition_origin", geometry_ref)
    return {"parent_surface_id": parent_surface_id, "rings_3d_um": _rings3d(source)}


def _distance_origin(surface):
    if "surface_distance_band" in surface.metadata:
        return None, "distance partition children cannot become new distance origins"
    inherited = (
        surface.parent_surface_id is not None
        or surface.partition_label is not None
        or "partition_id" in surface.geometry_ref
    )
    if inherited:
        lineage = surface.metadata.get("surface_partition_origin")
        if lineage is None:
            return (
                None,
                "partition has no original source-boundary lineage; generated seams are not distance origins",
            )
        return lineage, None
    return {
        "parent_surface_id": surface.surface_id,
        "rings_3d_um": _rings3d(surface),
    }, None


def synchronize_distance_partitions(existing, generated):
    """Flatten replaced child records without losing their original ownership."""
    replaced = {
        record.metadata["surface_distance_band"]["parent_surface_id"]
        for record in generated
    }
    retained = [
        record for record in existing if record.child_surface_id not in replaced
    ]
    for record in generated:
        selected = record.metadata["surface_distance_band"]["parent_surface_id"]
        ancestors = tuple(
            ancestor for ancestor in existing if ancestor.child_surface_id == selected
        )
        if ancestors:
            interfaces = {ancestor.parent_interface_id for ancestor in ancestors}
            if len(interfaces) != 1:
                raise ValueError(
                    "replaced partition child has ambiguous interface ownership"
                )
            lineage = tuple(
                {
                    "partition_id": ancestor.partition_id,
                    "parent_interface_id": ancestor.parent_interface_id,
                    "child_surface_id": ancestor.child_surface_id,
                    "label": ancestor.label,
                    "valid_routes": ancestor.valid_routes,
                    "metadata": dict(ancestor.metadata),
                }
                for ancestor in ancestors
            )
            record = replace(
                record,
                parent_interface_id=ancestors[0].parent_interface_id,
                metadata={**record.metadata, "ancestor_surface_partitions": lineage},
            )
        retained.append(record)
    return tuple(retained)


def surface_distance_band_catalogue(plan, *, build_input=None):
    """Expose original rings, source provenance and origin support explicitly.

    A nested partition needs retained original interface-boundary lineage.
    Its child edges are never silently promoted to new distance origins.
    """
    rows = []
    for surface in plan.surfaces:
        origin, reason = _distance_origin(surface)
        owners, polygon_ids = _parent_source_identity(surface, build_input)
        bindings = []
        if build_input is not None:
            for entity in build_input.entities:
                if polygon_ids:
                    bindings.extend(
                        {"source_entity_id": entity.semantic_id,
                         "source_polygon_id": polygon_id}
                        for polygon_id in entity.polygon_ids if polygon_id in polygon_ids
                    )
                elif entity.semantic_id in owners:
                    bindings.append({"source_entity_id": entity.semantic_id,
                                     "source_polygon_id": None})
        rows.append(
            {
                "parent_surface_id": surface.surface_id,
                "source_entity_ids": owners,
                "source_polygon_ids": polygon_ids,
                "source_bindings": tuple(bindings),
                "source_provenance": dict(
                    surface.metadata.get("source_provenance", {})
                ),
                "rings_3d_um": _rings3d(surface),
                "already_partitioned": "surface_distance_band" in surface.metadata,
                "distance_origin_status": "supported"
                if origin is not None
                else "unsupported",
                "distance_origin_reason": reason,
                "distance_origin_parent_surface_id": None
                if origin is None
                else origin["parent_surface_id"],
                "distance_origin_rings_3d_um": None
                if origin is None
                else origin["rings_3d_um"],
            }
        )
    return tuple(rows)


def partition_surface_distance_bands(
    build_input: GeometryBuildInput,
    *,
    surfaces: tuple[SurfacePlanRecord, ...],
):
    """Return live children, partition records and explicit empty-band summaries.

    Planar polygon faces are supported. Origins are the selected original parent
    ring edges, including holes; generated partition seams never enter this set.
    """
    import gdstk

    raw = build_input.metadata.get("surface_distance_bands", ())
    if not raw:
        return surfaces, (), ()
    specs = tuple(SurfaceDistanceBandSpec.from_payload(value) for value in raw)
    requests = {spec.parent_surface_id: spec for spec in specs}
    if len(requests) != len(specs):
        raise ValueError("duplicate distance partition parent")
    missing = requests.keys() - {surface.surface_id for surface in surfaces}
    if missing:
        raise ValueError(
            f"unknown distance partition parent surfaces: {sorted(missing)}"
        )
    entities = {entity.semantic_id: entity for entity in build_input.entities}
    polygon_ids = {polygon.polygon_id for polygon in build_input.polygons}
    result, partitions, summaries = [], [], []
    for parent in surfaces:
        spec = requests.get(parent.surface_id)
        if spec is None:
            result.append(parent)
            continue
        entity = entities[spec.source_entity_id]
        if spec.source_polygon_id is not None and (
            spec.source_polygon_id not in polygon_ids
            or spec.source_polygon_id not in entity.polygon_ids
        ):
            raise ValueError(
                "distance origin polygon does not belong to the explicit source entity"
            )
        if any(
            chain.entity_id == entity.semantic_id
            for chain in (
                *build_input.boundary_curves,
                *build_input.boundary_reconstruction,
            )
        ):
            raise NotImplementedError(
                "exact-curve distance partitions require an explicit curved-distance contract"
            )
        ref = parent.geometry_ref
        origin, reason = _distance_origin(parent)
        if origin is None:
            raise ValueError(reason)
        owners, source_ids = _parent_source_identity(parent, build_input)
        if spec.source_polygon_id is None:
            if source_ids:
                raise ValueError("polygon-backed parent requires an explicit source polygon anchor")
            if entity.semantic_id not in owners:
                raise ValueError("entity-only source anchor must be a declared parent owner")
        elif spec.source_polygon_id not in source_ids:
            raise ValueError(
                "parent surface provenance does not contain requested source polygon anchor"
            )
        if (
            not parent.geometry_ref.get("source_polygon_ids")
            and entity.semantic_id not in owners
        ):
            raise ValueError(
                "generated parent source anchor must belong to a declared parent owner"
            )
        original_rings = _rings3d(parent)
        rings, lift, project = _local_frame(original_rings)
        origin_rings = tuple(
            tuple(project(point) for point in ring) for ring in origin["rings_3d_um"]
        )
        origins = spec.boundary_segments
        if origins is None:
            origins = tuple(
                SourceBoundarySegment("exterior", i)
                for i in range(len(origin_rings[0]))
            ) + tuple(
                SourceBoundarySegment("hole", i, hole_index)
                for hole_index, hole in enumerate(origin_rings[1:])
                for i in range(len(hole))
            )
        edges = []
        for boundary in origins:
            if boundary.ring == "hole" and boundary.hole_index >= len(origin_rings) - 1:
                raise ValueError(
                    "source-bound parent hole index is outside its recorded rings"
                )
            ring = (
                origin_rings[0]
                if boundary.ring == "exterior"
                else origin_rings[boundary.hole_index + 1]
            )
            if boundary.segment_index >= len(ring):
                raise ValueError(
                    "parent boundary segment index is outside its recorded ring"
                )
            a, b = (
                ring[boundary.segment_index],
                ring[(boundary.segment_index + 1) % len(ring)],
            )
            if a == b:
                raise ValueError("parent boundary contains a zero-length selected edge")
            edges.append((a, b))
        precision = spec.precision_um

        def boolean(a, b, operation):
            return (
                tuple(gdstk.boolean(a, b, operation, precision=precision)) if a else ()
            )

        parent_region = (gdstk.Polygon(rings[0]),)
        if len(rings) > 1:
            parent_region = boolean(
                parent_region, [gdstk.Polygon(ring) for ring in rings[1:]], "not"
            )
        remaining = parent_region
        for index, upper in enumerate((*spec.distances_um, None)):
            lower = 0.0 if index == 0 else spec.distances_um[index - 1]
            if upper is None:
                pieces = remaining
            else:
                tubes = []
                for a, b in edges:
                    length = hypot(b[0] - a[0], b[1] - a[1])
                    nx, ny = (
                        -(b[1] - a[1]) * upper / length,
                        (b[0] - a[0]) * upper / length,
                    )
                    tubes.append(
                        gdstk.Polygon(
                            (
                                (a[0] + nx, a[1] + ny),
                                (b[0] + nx, b[1] + ny),
                                (b[0] - nx, b[1] - ny),
                                (a[0] - nx, a[1] - ny),
                            )
                        )
                    )
                    tubes.extend(
                        gdstk.ellipse(point, upper, tolerance=precision)
                        for point in (a, b)
                    )
                region = boolean(tubes, (), "or")
                pieces = boolean(remaining, region, "and")
                remaining = boolean(remaining, region, "not")
            pieces = sorted(
                pieces, key=lambda piece: tuple(sorted(map(tuple, piece.points)))
            )
            component_count = len(pieces)
            summary = {
                **spec.to_payload(),
                "boundary_segments": [
                    {
                        "ring": edge.ring,
                        "segment_index": edge.segment_index,
                        "hole_index": edge.hole_index,
                    }
                    for edge in origins
                ],
                "distance_origin_parent_surface_id": origin["parent_surface_id"],
                "distance_origin_rings_3d_um": origin["rings_3d_um"],
                "parent_source_polygon_ids": source_ids,
                "parent_owner_semantic_ids": owners,
                "source_anchor_role": "parent_provenance_group_anchor",
                "band_index": index,
                "lower_um": lower,
                "upper_um": upper,
                "child_surface_ids": [],
                "component_count": component_count,
            }
            simple_pieces = []
            for piece in pieces:
                loops = _simple_boundary_loops(tuple(map(tuple, piece.points)))

                def signed_area(loop):
                    return (
                        sum(
                            a[0] * b[1] - a[1] * b[0]
                            for a, b in zip(loop, (*loop[1:], loop[0]))
                        )
                        / 2
                    )

                outer_loops = [loop for loop in loops if signed_area(loop) > 0]
                hole_loops = [loop for loop in loops if signed_area(loop) < 0]
                for outer in outer_loops:
                    holes = tuple(
                        hole
                        for hole in hole_loops
                        if gdstk.inside((hole[0],), (gdstk.Polygon(outer),))[0]
                    )
                    simple_pieces.append((outer, *holes))
            summary["component_count"] = len(simple_pieces)
            for component, piece in enumerate(simple_pieces):
                # Canonical SGB holes use the same loop direction as the outer
                # wire; native surface construction assigns the hole role.
                local_rings = tuple(
                    _restore_parent_boundary(
                        ring if index == 0 else tuple(reversed(ring)), rings, precision
                    )
                    for index, ring in enumerate(piece)
                )
                original_points = {
                    local: original
                    for local_ring, original_ring in zip(rings, original_rings)
                    for local, original in zip(local_ring, original_ring)
                }

                def preserved_lift(point):
                    return (
                        original_points[point]
                        if point in original_points
                        else lift(point)
                    )

                child_ref = {
                    **ref,
                    "planar_loops_3d": tuple(
                        tuple(preserved_lift(point) for point in ring)
                        for ring in local_rings
                    ),
                }
                for obsolete in ("quad_points", "outer_loop", "hole_loops"):
                    child_ref.pop(obsolete, None)
                child_id = f"{parent.surface_id}__DIST_{index:04d}_{component:04d}"
                identity = {
                    **summary,
                    "component_index": component,
                    "boundary_construction": "unique_source_segment_analytic_intersection",
                    "distance_contour_precision_um": precision,
                }
                identity.pop("child_surface_ids")
                child = replace(
                    parent,
                    surface_id=child_id,
                    geometry_ref=child_ref,
                    outer_loop_ref=None,
                    hole_loop_refs=(),
                    parent_surface_id=parent.surface_id,
                    partition_label=f"DIST_{index:04d}_{component:04d}",
                    metadata={
                        **parent.metadata,
                        "surface_distance_band": identity,
                        "physical_name": child_id,
                    },
                )
                result.append(child)
                partitions.append(
                    SurfacePartitionRecord(
                        partition_id=f"PART__{child_id}",
                        parent_interface_id=parent.interface_id or parent.surface_id,
                        child_surface_id=child_id,
                        label=child.partition_label,
                        valid_routes=parent.valid_routes,
                        metadata={"surface_distance_band": identity},
                    )
                )
                summary["child_surface_ids"].append(child_id)
            summaries.append(summary)

    return tuple(result), tuple(partitions), tuple(summaries)
