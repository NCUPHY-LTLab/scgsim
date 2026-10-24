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


def _restore_parent_boundary(ring, parent_rings, precision, *, diagnostic_context=None):
    """Intersect generated contour lines with uniquely attributable source edges.

    The only association bound is the caller-declared Boolean grid. It is not
    a nearest-edge selector: multiple eligible source segments fail explicitly.
    """
    from math import sqrt

    from fractions import Fraction

    source_edges = []
    source_edge_vertices = []
    source_edge_identities = []
    for ring_index, source in enumerate(parent_rings):
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
            segments = [start]
            cursor = (start + 1) % len(source)
            while cursor != stop:
                vertices.append(source[cursor])
                segments.append(cursor)
                cursor = (cursor + 1) % len(source)
            vertices.append(source[stop])
            source_edges.append((vertices[0], vertices[-1]))
            source_edge_vertices.append(tuple(vertices))
            source_edge_identities.append({
                "source_run_index": len(source_edges) - 1,
                "source_ring_index": ring_index,
                "source_ring_role": "exterior" if ring_index == 0 else "hole",
                "source_hole_index": None if ring_index == 0 else ring_index - 1,
                "start_vertex_index": start,
                "end_vertex_index": stop,
                "source_segment_indices": segments,
            })
    bound = precision / sqrt(2)

    def candidates(a, b, edge_index):
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
            error = ValueError(
                "Boolean boundary lineage is ambiguous at declared precision"
            )
            # Enrich only this unchanged failure; retain indices and exact
            # relations rather than transporting source/generated coordinates.
            try:
                import json
                from itertools import combinations

                context = diagnostic_context or {}
                candidates_detail = []
                for index in found:
                    endpoints = source_edges[index]
                    candidates_detail.append({
                        **source_edge_identities[index],
                        "zero_length_source_run": endpoints[0] == endpoints[1],
                        "generated_start_is_run_endpoint": a in endpoints,
                        "generated_end_is_run_endpoint": b in endpoints,
                        "generated_start_is_source_run_vertex": a in source_edge_vertices[index],
                        "generated_end_is_source_run_vertex": b in source_edge_vertices[index],
                    })
                relations = []
                for first, second in combinations(found, 2):
                    left, right = source_edge_identities[first], source_edge_identities[second]
                    same_ring = left["source_ring_index"] == right["source_ring_index"]
                    relations.append({
                        "source_run_indices": [first, second],
                        "same_source_ring": same_ring,
                        "adjacent_source_runs": same_ring and (
                            left["end_vertex_index"] == right["start_vertex_index"]
                            or right["end_vertex_index"] == left["start_vertex_index"]
                        ),
                        "shared_endpoint_count": len(set(source_edges[first]) & set(source_edges[second])),
                        "shared_source_vertex_count": len(set(source_edge_vertices[first]) & set(source_edge_vertices[second])),
                    })
                note = {
                    "diagnostic": "surface_band_boundary_lineage",
                    "stage": "restore_parent_boundary.source_run_candidates",
                    **{key: context.get(key) for key in (
                        "parent_surface_id", "band_index", "lower_um", "upper_um",
                        "component_index", "ring_role", "hole_index",
                    )},
                    "precision_um": precision,
                    "generated_edge_index": edge_index,
                    "zero_length_generated_edge": a == b,
                    "generated_start_is_original_vertex": any(a in source for source in parent_rings),
                    "generated_end_is_original_vertex": any(b in source for source in parent_rings),
                    "candidate_count": len(found),
                    "candidates": candidates_detail,
                    "candidate_relations": relations,
                }
                error.add_note(json.dumps(note, sort_keys=True, separators=(",", ":")))
            except Exception:
                # Diagnostics must not replace the original geometric failure.
                error.add_note('{"diagnostic":"surface_band_boundary_lineage","status":"unavailable"}')
            raise error
        return found[0] if found else None

    owners = [
        candidates(a, b, edge_index)
        for edge_index, (a, b) in enumerate(zip(ring, (*ring[1:], ring[0])))
    ]
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


def _local_frame(rings, *, vertical_construction=False):
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
    if vertical_construction and normal[2] == 0:
        # Stack extrusion owns Z independently of layout normalization. Its
        # declared vertical axis is exact; deriving it from two normalized
        # cross products can move generated boundary points off the source Z.
        along = next(sub(point, origin) for ring in rings for point in ring
                     if point[:2] != origin[:2])
        span = sqrt(along[0]**2 + along[1]**2)
        u = (along[0]/span, along[1]/span)
        levels = {point[2]-origin[2]: point[2] for ring in rings for point in ring}
        local = tuple(tuple(((point[0]-origin[0])*u[0]+(point[1]-origin[1])*u[1],
                              point[2]-origin[2]) for point in ring) for ring in rings)
        def lift(point):
            return (origin[0]+point[0]*u[0], origin[1]+point[0]*u[1],
                    levels[point[1]] if point[1] in levels else origin[2]+point[1])
        def project(point):
            return ((point[0]-origin[0])*u[0]+(point[1]-origin[1])*u[1], point[2]-origin[2])
        return local, lift, project
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
        (record.metadata.get("surface_distance_band") or record.metadata["layout_grid_realization"])["parent_surface_id"]
        for record in generated
    }
    retained = [
        record for record in existing if record.child_surface_id not in replaced
    ]
    for record in generated:
        selected = (record.metadata.get("surface_distance_band") or record.metadata["layout_grid_realization"])["parent_surface_id"]
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
    from scgsim.geometry.models.grid import geometry_grid_context
    grid_context = geometry_grid_context(build_input)
    layout = _layout_band_construction(surfaces, requests, grid_context) if grid_context is not None else None
    entities = {entity.semantic_id: entity for entity in build_input.entities}
    polygon_ids = {polygon.polygon_id for polygon in build_input.polygons}
    result, partitions, summaries = [], [], []
    for parent in surfaces:
        spec = requests.get(parent.surface_id)
        if spec is None:
            result.append(_layout_surface(parent, layout))
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
        from scgsim.geometry.models.grid import geometry_grid_context
        grid_context = geometry_grid_context(build_input)
        horizontal = len({p[2] for r in original_rings for p in r}) == 1
        if grid_context is not None and horizontal:
            rings = tuple(tuple((p[0], p[1]) for p in r) for r in original_rings)
            z = original_rings[0][0][2]
            def lift(point):
                return (point[0], point[1], z)
            def project(point):
                return (point[0], point[1])
        else:
            rings, lift, project = _local_frame(original_rings, vertical_construction=grid_context is not None)
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
        if grid_context is not None:
            if precision != grid_context.grid_size_um:
                raise ValueError(f"SurfaceDistanceBandSpec.precision_um differs from geometry_grid for {parent.surface_id}")
            children, records, band_summary = _partition_grid_parent(
                parent, spec, rings, original_rings, lift, edges, origin,
                owners, source_ids, grid_context, horizontal,
                selected_boundaries=origins,
                shared=None if layout is None else layout["parents"].get(parent.surface_id),
            )
            result.extend(children)
            partitions.extend(records)
            summaries.extend(band_summary)
            continue

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
                        gdstk.ellipse(point, upper, tolerance=spec.chord_error_um or precision)
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
                        ring if index == 0 else tuple(reversed(ring)), rings, precision,
                        diagnostic_context={
                            "parent_surface_id": parent.surface_id,
                            "band_index": summary["band_index"],
                            "lower_um": lower,
                            "upper_um": upper,
                            "component_index": component,
                            "ring_role": "exterior" if index == 0 else "hole",
                            "hole_index": None if index == 0 else index - 1,
                        },
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

    from scgsim.geometry.models.grid import geometry_grid_context
    context = geometry_grid_context(build_input)
    if context is not None:
        result, partitions, summaries = _node_band_planes(result, summaries, context)
        if layout is not None:
            result, partitions, summaries = _emit_layout_wall_facets(result, summaries, layout)
    return tuple(result), tuple(partitions), tuple(summaries)



def _parent_oriented_rings(outer, holes, parent_outer):
    """Keep the parent's wire direction through GEOS region reconstruction.

    SGB assigns hole roles separately, with hole wires in the outer direction.
    Reversing a wire preserves all shared incident vertices and coordinates.
    """
    def area2(ring):
        return sum(a[0] * b[1] - b[0] * a[1]
                   for a, b in zip(ring, (*ring[1:], ring[0])))

    parent_direction = area2(parent_outer)
    return tuple(ring if (area2(ring) > 0) == (parent_direction > 0)
                 else tuple(reversed(ring)) for ring in (outer, *holes))


def _partition_grid_parent(parent, spec, rings, original_rings, lift, edges,
                           origin, owners, source_ids, context, horizontal, *,
                           selected_boundaries, shared=None):
    """Construct actual shared band faces; no source-edge ownership guessing."""
    from math import acos, ceil, pi
    from shapely import LineString, union_all
    from scgsim.geometry._primitives.grid import GridOperand, build_grid_arrangement

    parent_key = "parent"
    if shared is None:
        operands = [GridOperand("parent", rings[0], rings[1:], physical=True, contact=False)]
        tube_ids = []
        chord_error = spec.precision_um if spec.chord_error_um is None else spec.chord_error_um
        for index, upper in enumerate(spec.distances_um):
            quad_segs = 1 if chord_error >= upper else ceil(pi / (4 * acos(1 - chord_error / upper)))
            tubes = [LineString((a, b)).buffer(upper, quad_segs=quad_segs) for a,b in edges]
            # The control selects a distance-to-boundary UNION, not individual caps.
            merged = union_all(tubes, grid_size=context.grid_size_um if horizontal else None)
            from scgsim.geometry._primitives.grid import _polygons
            ids = []
            for component, tube in enumerate(_polygons(merged)):
                identity = f"distance_{index}_component_{component}"
                operands.append(GridOperand(identity, tuple(tube.exterior.coords)[:-1],
                    tuple(tuple(h.coords)[:-1] for h in tube.interiors), physical=False, contact=False))
                ids.append(identity)
            tube_ids.append(set(ids))
        arrangement = build_grid_arrangement(operands, context, apply_grid=horizontal)
    else:
        arrangement, parent_key, tube_ids = shared
        chord_error = spec.precision_um if spec.chord_error_um is None else spec.chord_error_um
    selected = [[] for _ in range(len(spec.distances_um) + 1)]
    for face in arrangement.faces:
        if parent_key not in face.operand_ids:
            continue
        membership = set(face.operand_ids)
        index = next((i for i, ids in enumerate(tube_ids) if membership & ids), len(tube_ids))
        selected[index].append(face.face_id)
    original_points = {local: original for local_ring, original_ring in zip(rings, original_rings)
                       for local, original in zip(local_ring, original_ring)}
    children, partitions, summaries = [], [], []
    for index, face_ids in enumerate(selected):
        regions = arrangement.face_regions(set(face_ids))
        summary = {**spec.to_payload(),
            "boundary_segments": [dict(ring=e.ring, segment_index=e.segment_index, hole_index=e.hole_index)
                                  for e in selected_boundaries],
            "distance_origin_parent_surface_id": origin["parent_surface_id"],
            "distance_origin_rings_3d_um": origin["rings_3d_um"],
            "parent_source_polygon_ids": source_ids, "parent_owner_semantic_ids": owners,
            "source_anchor_role": "parent_provenance_group_anchor", "band_index": index,
            "lower_um": 0.0 if index == 0 else spec.distances_um[index-1],
            "upper_um": spec.distances_um[index] if index < len(spec.distances_um) else None,
            "component_count": len(regions), "child_surface_ids": [],
            "geometry_grid": context.to_payload(),
            "grid_scope": "assembly_xy" if horizontal else "parent_plane_unquantized",
            "chord_error_um": chord_error,
        }
        for component, (outer, holes) in enumerate(regions):
            local_rings = _parent_oriented_rings(outer, holes, rings[0])
            child_id = f"{parent.surface_id}__DIST_{index:04d}_{component:04d}"
            identity = {**summary, "component_index": component,
                "boundary_construction": "geos_labelled_polygon_overlay",
                "distance_contour_precision_um": spec.precision_um}
            identity.pop("child_surface_ids")
            ref = {k: v for k, v in parent.geometry_ref.items()
                   if k not in {"quad_points", "outer_loop", "hole_loops"}}
            ref["planar_loops_3d"] = tuple(tuple(original_points[p] if p in original_points
                else lift(p) for p in ring) for ring in local_rings)
            child = replace(parent, surface_id=child_id, geometry_ref=ref, outer_loop_ref=None,
                hole_loop_refs=(), parent_surface_id=parent.surface_id,
                partition_label=f"DIST_{index:04d}_{component:04d}",
                metadata={**parent.metadata, "surface_distance_band": identity,
                          "physical_name": child_id})
            children.append(child)
            partitions.append(SurfacePartitionRecord(f"PART__{child_id}",
                parent.interface_id or parent.surface_id, child_id, child.partition_label,
                valid_routes=parent.valid_routes, metadata={"surface_distance_band": identity}))
            summary["child_surface_ids"].append(child_id)
        summaries.append(summary)
    return tuple(children), tuple(partitions), tuple(summaries)


def _node_band_planes(surfaces, summaries, context):
    """Share incident vertices among all actual coplanar band/neighbor faces."""
    from fractions import Fraction
    from collections import defaultdict
    from scgsim.geometry._primitives.grid import GridOperand, build_grid_arrangement, _canonical_grid_point

    def plane_key(rings):
        points = [tuple(Fraction(str(v)) for v in p) for r in rings for p in r]
        a = points[0]
        for b in points[1:]:
            ab = tuple(b[i]-a[i] for i in range(3))
            for c in points[2:]:
                ac = tuple(c[i]-a[i] for i in range(3))
                n = (ab[1]*ac[2]-ab[2]*ac[1], ab[2]*ac[0]-ab[0]*ac[2],
                     ab[0]*ac[1]-ab[1]*ac[0])
                pivot = next((v for v in n if v), None)
                if pivot is not None:
                    n = tuple(v/pivot for v in n)
                    return (*n, sum(n[i]*a[i] for i in range(3)))
        raise ValueError("distance-band surface has no nondegenerate plane")

    groups = defaultdict(list)
    for surface in surfaces:
        rings = _rings3d(surface)
        groups[plane_key(rings)].append((surface, rings))
    replacements = {}
    for group in groups.values():
        if not any("surface_distance_band" in surface.metadata for surface, _ in group):
            continue
        first = group[0][1]
        horizontal = len({p[2] for r in first for p in r}) == 1
        if horizontal:
            z = first[0][0][2]
            def project(point):
                return point[0], point[1]
            def lift(point):
                return point[0], point[1], z
        else:
            _, lift, project = _local_frame(first, vertical_construction=True)
        original = {}
        operands = []
        for surface, rings in group:
            local = tuple(tuple(project(p) for p in ring) for ring in rings)
            for source, projected in zip(rings, local):
                for world, point in zip(source, projected):
                    # Raw projected operands below remain the physical baseline.
                    # Only the post-arrangement lookup/emission shares canonical XY.
                    key = _canonical_grid_point(point, context.grid_size_um) if horizontal else point
                    emitted = lift(key) if horizontal else world
                    if key in original and original[key] != emitted:
                        raise ValueError("shared parent-plane vertices disagree on source coordinates")
                    original[key] = emitted
            operands.append(GridOperand(surface.surface_id, local[0], local[1:],
                physical="surface_distance_band" not in surface.metadata, contact=False))
        arrangement = build_grid_arrangement(operands, context, apply_grid=horizontal)
        for surface, source_rings in group:
            parent_outer = tuple(project(p) for p in source_rings[0])
            pieces = []
            for outer, holes in arrangement.operand_regions(surface.surface_id):
                ref = {k:v for k,v in surface.geometry_ref.items()
                       if k not in {"outer_loop", "hole_loops", "quad_points"}}
                ref["planar_loops_3d"] = tuple(tuple(original[p] if p in original else lift(p)
                    for p in ring) for ring in _parent_oriented_rings(outer, holes, parent_outer))
                pieces.append(replace(surface, geometry_ref=ref, outer_loop_ref=None,
                                      hole_loop_refs=()))
            replacements[surface.surface_id] = pieces
    actual = [piece for surface in surfaces for piece in replacements.get(surface.surface_id, (surface,))]
    by_band = defaultdict(list)
    for surface in actual:
        identity = surface.metadata.get("surface_distance_band")
        if identity is not None:
            by_band[(identity["parent_surface_id"], identity["band_index"])].append(surface)
    updated, partitions = [], []
    for surface in actual:
        identity = surface.metadata.get("surface_distance_band")
        if identity is None:
            updated.append(surface)
            continue
        key = identity["parent_surface_id"], identity["band_index"]
        siblings = by_band[key]
        component = next(i for i,piece in enumerate(siblings) if piece is surface)
        child_id = f"{key[0]}__DIST_{key[1]:04d}_{component:04d}"
        identity = {**identity, "component_index": component, "component_count": len(siblings),
                    "shared_plane_construction": "geos_labelled_polygon_overlay"}
        child = replace(surface, surface_id=child_id,
            partition_label=f"DIST_{key[1]:04d}_{component:04d}",
            metadata={**surface.metadata, "surface_distance_band": identity, "physical_name": child_id})
        updated.append(child)
        partitions.append(SurfacePartitionRecord(partition_id=f"PART__{child_id}",
            parent_interface_id=child.interface_id or child.parent_surface_id,
            child_surface_id=child_id, label=child.partition_label,
            valid_routes=child.valid_routes, metadata={"surface_distance_band": identity}))
    rows = []
    for summary in summaries:
        matches = [s.surface_id for s in updated if s.metadata.get("surface_distance_band", {}).get(
            "parent_surface_id") == summary["parent_surface_id"] and s.metadata.get(
            "surface_distance_band", {}).get("band_index") == summary["band_index"]]
        rows.append({**summary, "child_surface_ids": matches, "component_count": len(matches),
                     "shared_plane_construction": "geos_labelled_polygon_overlay"})
    return updated, partitions, rows


def _layout_band_construction(surfaces, requests, context):
    """Realize shared XY boundaries before emitting any incident wall facets."""
    from math import acos, ceil, pi
    from shapely import LineString, Point, intersection, union_all
    from scgsim.geometry._primitives.grid import GridOperand, build_grid_arrangement, _polygons, _canonical_grid_point
    horizontal = [(surface, _rings3d(surface)) for surface in surfaces
                  if len({p[2] for r in _rings3d(surface) for p in r}) == 1]
    if not any(surface.surface_id in requests for surface, _ in horizontal):
        return None
    operands, parent_keys, cut_keys = [], {}, {}
    for surface, rings in horizontal:
        key = "layout_parent::"+surface.surface_id
        parent_keys[surface.surface_id] = key
        operands.append(GridOperand(key, tuple(p[:2] for p in rings[0]),
            tuple(tuple(p[:2] for p in r) for r in rings[1:]), contact=False))
        spec = requests.get(surface.surface_id)
        if spec is None:
            continue
        origin, reason = _distance_origin(surface)
        if origin is None:
            raise ValueError(reason)
        source = origin["rings_3d_um"]
        selected = spec.boundary_segments
        if selected is None:
            selected = tuple(SourceBoundarySegment("exterior", i) for i in range(len(source[0])))+tuple(
                SourceBoundarySegment("hole", i, h) for h,r in enumerate(source[1:]) for i in range(len(r)))
        edges = []
        for segment in selected:
            index = 0 if segment.ring == "exterior" else segment.hole_index+1
            if index >= len(source) or segment.segment_index >= len(source[index]):
                raise ValueError("distance origin boundary segment is outside the original ring")
            ring = source[index]
            edges.append((ring[segment.segment_index][:2], ring[(segment.segment_index+1)%len(ring)][:2]))
        cuts = []
        error = spec.precision_um if spec.chord_error_um is None else spec.chord_error_um
        for band, distance in enumerate(spec.distances_um):
            count = 1 if error >= distance else ceil(pi/(4*acos(1-error/distance)))
            tube = union_all([LineString(edge).buffer(distance, quad_segs=count) for edge in edges],
                             grid_size=context.grid_size_um)
            identities = set()
            for component, polygon in enumerate(_polygons(tube)):
                identity = f"layout_cut::{surface.surface_id}::{band}::{component}"
                operands.append(GridOperand(identity, tuple(polygon.exterior.coords)[:-1],
                    tuple(tuple(r.coords)[:-1] for r in polygon.interiors), physical=False, contact=False))
                identities.add(identity)
            cuts.append(identities)
        cut_keys[surface.surface_id] = cuts
    arrangement = build_grid_arrangement(operands, context)
    parents, realized, chains, ledger = {}, {}, {}, []
    def normalized(point):
        value = intersection(Point(point[:2]), Point(point[:2]), grid_size=context.grid_size_um)
        return _canonical_grid_point(value.coords[0], context.grid_size_um)
    for surface, source in horizontal:
        regions = arrangement.operand_regions(parent_keys[surface.surface_id])
        if len(regions) != 1:
            raise ValueError(f"layout parent component identity changed: {surface.surface_id}")
        outer, holes = regions[0]
        realized[surface.surface_id] = (outer, holes)
        if surface.surface_id in requests:
            parents[surface.surface_id] = arrangement, parent_keys[surface.surface_id], cut_keys[surface.surface_id]
        available = [outer, *holes]
        for ring_index, ring in enumerate(source):
            anchors = tuple(normalized(p) for p in ring)
            matches = []
            for candidate in available:
                try:
                    mapped = _boundary_edge_chains(anchors, candidate)
                except ValueError:
                    continue
                matches.append(mapped)
            if len(matches) != 1:
                raise ValueError(f"layout original ring has missing/ambiguous construction incidence: {surface.surface_id}, ring {ring_index}")
            for edge_index, chain in enumerate(matches[0]):
                a,b = anchors[edge_index], anchors[(edge_index+1)%len(anchors)]
                if a == b:
                    continue
                key = (a,b) if a < b else (b,a)
                ordered = chain if a < b else tuple(reversed(chain))
                if key in chains and chains[key] != ordered:
                    raise ValueError(f"incident layout parents disagree on realized boundary chain: {surface.surface_id}, ring {ring_index}, edge {edge_index}")
                chains[key] = ordered
                ledger.append({"parent_surface_id": surface.surface_id, "source_ring_index": ring_index,
                    "source_edge_index": edge_index, "source_endpoints": (ring[edge_index],ring[(edge_index+1)%len(ring)]),
                    "realized_xy_chain": chain})
    return {"parents": parents, "realized": realized, "chains": chains,
            "ledger": tuple(ledger), "context": context, "normalize": normalized}


def _boundary_edge_chains(anchors, ring):
    """Use exact retained endpoint incidence/order, never a nearest edge owner."""
    from fractions import Fraction
    points = list(ring)
    def on(p,a,b):
        p,a,b = (tuple(Fraction(str(v)) for v in q) for q in (p,a,b))
        return ((p[0]-a[0])*(b[1]-a[1]) == (p[1]-a[1])*(b[0]-a[0]) and
                min(a[0],b[0]) <= p[0] <= max(a[0],b[0]) and min(a[1],b[1]) <= p[1] <= max(a[1],b[1]))
    for anchor in anchors:
        if anchor in points:
            continue
        incident = [i for i,a in enumerate(points) if on(anchor,a,points[(i+1)%len(points)])]
        if len(incident) != 1:
            raise ValueError("source vertex has no unique exact boundary incidence")
        points.insert(incident[0]+1, anchor)
    def signed(values):
        return sum(a[0]*b[1]-a[1]*b[0] for a,b in zip(values,(*values[1:],values[0])))
    if signed(points)*signed(anchors) < 0:
        points.reverse()
    result = []
    for a,b in zip(anchors,(*anchors[1:],anchors[0])):
        if a == b:
            result.append((a,));continue
        i=points.index(a);path=[a]
        for step in range(1,len(points)+1):
            point=points[(i+step)%len(points)];path.append(point)
            if point == b:
                break
            if point in anchors:
                raise ValueError("source boundary endpoint order changed")
        else:
            raise ValueError("source boundary chain is not closed")
        result.append(tuple(path))
    return tuple(result)


def _layout_surface(surface, layout):
    if layout is None or surface.surface_id not in layout["realized"]:
        return surface
    outer, holes = layout["realized"][surface.surface_id]
    source_rings = _rings3d(surface)
    z = source_rings[0][0][2]
    local_rings = _parent_oriented_rings(outer, holes, tuple(p[:2] for p in source_rings[0]))
    ref = {key:value for key,value in surface.geometry_ref.items()
           if key not in {"quad_points","outer_loop","hole_loops"}}
    ref["planar_loops_3d"] = tuple(tuple((x,y,z) for x,y in ring) for ring in local_rings)
    return replace(surface, geometry_ref=ref, outer_loop_ref=None,hole_loop_refs=())


def _wall_construction_chain(source_xy, layout, surface_id):
    """Join exact original-segment incidences, retaining collinear anchors."""
    from collections import defaultdict, deque
    from fractions import Fraction

    def exact(point):
        return tuple(Fraction(str(value)) for value in point)

    points = tuple(map(exact, source_xy))
    spans = tuple(max(p[i] for p in points) - min(p[i] for p in points)
                  for i in range(2))
    axis = 0 if spans[0] >= spans[1] else 1
    a = min(source_xy, key=lambda p: p[axis])
    b = max(source_xy, key=lambda p: p[axis])
    ea, eb = exact(a), exact(b)
    if ea == eb:
        raise ValueError(f"wall construction incidence is degenerate: {surface_id}")

    def collinear(point):
        p = exact(point)
        return ((p[0] - ea[0]) * (eb[1] - ea[1]) ==
                (p[1] - ea[1]) * (eb[0] - ea[0]))

    if not all(collinear(point) for point in source_xy):
        return None
    chains = []
    for record in layout["ledger"]:
        left, right = tuple(p[:2] for p in record["source_endpoints"])
        if not collinear(left) or not collinear(right):
            continue
        lo, hi = sorted((exact(left)[axis], exact(right)[axis]))
        if max(lo, ea[axis]) < min(hi, eb[axis]):
            chains.append(record["realized_xy_chain"])
    if not chains:
        raise ValueError(f"wall has missing original-segment construction incidence: {surface_id}")
    anchors = tuple(layout["normalize"](point) for point in source_xy)
    vertices = set(anchors)
    vertices.update(point for chain in chains for point in chain)
    graph = defaultdict(set)
    for chain in chains:
        for left, right in zip(chain, chain[1:]):
            el, er = exact(left), exact(right)
            direction = 0 if el[0] != er[0] else 1
            along = [point for point in vertices
                     if (exact(point)[0]-el[0])*(er[1]-el[1]) ==
                        (exact(point)[1]-el[1])*(er[0]-el[0])
                     and all(min(el[i], er[i]) <= exact(point)[i] <= max(el[i], er[i])
                             for i in range(2))]
            along.sort(key=lambda point: point[direction])
            for first, second in zip(along, along[1:]):
                if first != second:
                    graph[first].add(second)
                    graph[second].add(first)
    start, end = layout["normalize"](a), layout["normalize"](b)
    parents = {start: None}
    queue = deque((start,))
    while queue:
        point = queue.popleft()
        for neighbor in sorted(graph[point]):
            if neighbor == parents[point]:
                continue
            if neighbor in parents:
                raise ValueError(f"wall has ambiguous original-segment construction incidence: {surface_id}")
            parents[neighbor] = point
            queue.append(neighbor)
    if end not in parents or any(anchor not in parents for anchor in anchors):
        raise ValueError(f"wall has missing retained boundary-anchor incidence: {surface_id}")
    chain = [end]
    while chain[-1] != start:
        chain.append(parents[chain[-1]])
    chain.reverse()
    if any(anchor not in chain for anchor in anchors):
        raise ValueError(f"wall retained anchors are outside its construction chain: {surface_id}")
    return a, b, tuple(chain)


def _emit_layout_wall_facets(surfaces, summaries, layout):
    """Carry preclassified planar cells through the extrusion construction map."""
    from fractions import Fraction
    from shapely import Polygon, box, intersection
    from scgsim.geometry._primitives.grid import _regions
    output=[];extra=[]
    for surface in surfaces:
        rings=_rings3d(surface)
        if len({p[2] for r in rings for p in r}) == 1:
            output.append(surface);continue
        # Match the owning wall's original extrusion edge through its retained
        # parent construction, not the possibly inset child boundary.
        origin=surface.metadata.get("surface_partition_origin")
        band=surface.metadata.get("surface_distance_band")
        source = band["distance_origin_rings_3d_um"] if band else origin["rings_3d_um"] if origin else rings
        source_xy=tuple(dict.fromkeys(p[:2] for r in source for p in r))
        construction = _wall_construction_chain(source_xy, layout, surface.surface_id)
        if construction is None:
            output.append(surface);continue
        a,b,chain=construction
        axis=0 if abs(b[0]-a[0]) >= abs(b[1]-a[1]) else 1
        denominator=Fraction(str(b[axis]))-Fraction(str(a[axis]))
        values=[Fraction(0),*( (Fraction(str(p[axis]))-Fraction(str(a[axis])))/denominator for p in chain[1:-1]),Fraction(1)]
        if any(x >= y for x,y in zip(values,values[1:])):
            raise ValueError(f"wall construction incidence is noninvertible: {surface.surface_id}")
        def parameter(p):
            return float((Fraction(str(p[axis]))-Fraction(str(a[axis])))/denominator),p[2]
        requested=Polygon(tuple(parameter(p) for p in rings[0]),tuple(tuple(parameter(p) for p in r) for r in rings[1:]))
        lower=min(p[2] for r in source for p in r);upper=max(p[2] for r in source for p in r)
        for facet,(t0,t1,c0,c1) in enumerate(zip(values,values[1:],chain,chain[1:])):
            part=intersection(requested,box(float(t0),lower,float(t1),upper))
            for component,(outer,holes) in enumerate(_regions(part)):
                def emit(p):
                    t,z=p
                    if t == float(t0):xy=c0
                    elif t == float(t1):xy=c1
                    else:
                        ratio=(Fraction(str(t))-t0)/(t1-t0)
                        xy=tuple(float(Fraction(str(c0[i]))+ratio*(Fraction(str(c1[i]))-Fraction(str(c0[i])))) for i in range(2))
                    return (*xy,z)
                ref={k:v for k,v in surface.geometry_ref.items() if k not in {"quad_points","outer_loop","hole_loops"}}
                ref["planar_loops_3d"]=tuple(tuple(emit(p) for p in ring) for ring in _parent_oriented_rings(
                    outer, holes, tuple(parameter(p) for p in rings[0])))
                identity={"parent_surface_id":surface.surface_id,"facet_index":facet,"piece_index":component,
                    "source_xy_edge":(a,b),"source_xy_anchors":source_xy,"realized_xy_edge":(c0,c1),"construction_interval":(str(t0),str(t1)),
                    "method":"planar_classification_then_affine_extrusion_realization"}
                child_id=f"{surface.surface_id}__GRID_{facet:04d}_{component:04d}"
                child=replace(surface,surface_id=child_id,geometry_ref=ref,outer_loop_ref=None,hole_loop_refs=(),
                    parent_surface_id=surface.parent_surface_id or surface.surface_id,
                    partition_label=surface.partition_label or f"GRID_{facet:04d}_{component:04d}",
                    metadata={**surface.metadata,"layout_grid_realization":identity,"physical_name":child_id,
                              "surface_partition_origin":origin or {"parent_surface_id":surface.surface_id,"rings_3d_um":source}})
                output.append(child)
                if band is None:
                    extra.append(SurfacePartitionRecord(partition_id="PART__"+child_id,
                        parent_interface_id=surface.interface_id or surface.surface_id,child_surface_id=child_id,
                        label=child.partition_label,valid_routes=surface.valid_routes,metadata={"layout_grid_realization":identity}))
    # Re-index the final physical surface fragments without re-evaluating the
    # planar distance metric on their realized facets.
    from collections import defaultdict
    groups=defaultdict(list)
    for surface in output:
        band=surface.metadata.get("surface_distance_band")
        if band is not None:groups[(band["parent_surface_id"],band["band_index"])].append(surface)
    actual=[];partitions=list(extra)
    for surface in output:
        band=surface.metadata.get("surface_distance_band")
        if band is None:actual.append(surface);continue
        key=band["parent_surface_id"],band["band_index"]
        index=next(i for i,p in enumerate(groups[key]) if p is surface)
        child_id=f"{key[0]}__DIST_{key[1]:04d}_{index:04d}"
        identity={**band,"construction_component_index":band["component_index"],"component_index":index,"component_count":len(groups[key])}
        child=replace(surface,surface_id=child_id,metadata={**surface.metadata,"surface_distance_band":identity,"physical_name":child_id})
        actual.append(child)
        partitions.append(SurfacePartitionRecord(partition_id="PART__"+child_id,parent_interface_id=child.interface_id or key[0],
            child_surface_id=child_id,label=child.partition_label,valid_routes=child.valid_routes,metadata={"surface_distance_band":identity}))
    rows=[]
    for row in summaries:
        ids=[s.surface_id for s in actual if s.metadata.get("surface_distance_band",{}).get("parent_surface_id")==row["parent_surface_id"]
             and s.metadata.get("surface_distance_band",{}).get("band_index")==row["band_index"]]
        rows.append({**row,"child_surface_ids":ids,"component_count":len(ids),"layout_boundary_realizations":layout["ledger"],
                     "distance_classification":"original_planar_parent_before_wall_facet_emission"})
    return actual,partitions,rows
