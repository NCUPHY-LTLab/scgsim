"""Apply the source-bound assembled XY policy without rewriting source GDS."""

from dataclasses import asdict, replace
from decimal import Decimal
import hashlib
import json

from scgsim.geometry.models.grid import GeometryGridContext
from scgsim.geometry._primitives.grid import GridOperand, build_grid_arrangement


def source_grid_context(source_dbu_um, requested):
    context = GeometryGridContext(source_dbu_um if requested is None else requested)
    if Decimal(str(context.grid_size_um)) != Decimal(str(source_dbu_um)):
        raise ValueError(
            f"grid_size_um={context.grid_size_um} differs from emitted source_dbu_um={source_dbu_um}"
        )
    return context


def _oriented_like(ring, source):
    def signed(loop):
        return sum(
            a[0] * b[1] - a[1] * b[0] for a, b in zip(loop, (*loop[1:], loop[0]))
        )

    return tuple(reversed(ring)) if signed(ring) * signed(source) < 0 else ring


def apply_geometry_grid(build_input, context):
    from scgsim.geometry._primitives.grid import geometry_grid_scope

    with geometry_grid_scope(context):
        return _apply_geometry_grid(build_input, context)


def _apply_geometry_grid(build_input, context):
    """Node all placed layout operands once; keep raw source beside effective data."""
    from scgsim.geometry.models.grid import geometry_grid_context

    bound = geometry_grid_context(build_input)
    if bound is not None and bound != context:
        raise ValueError("geometry grid context differs from the bound input policy")
    from scgsim.geometry._primitives.grid import source_grid_audits

    fresh_audits = source_grid_audits()
    audits = (
        *fresh_audits,
        *build_input.metadata.get("geometry_grid_source_audits", ()),
    )
    use_source_preimage = bound is None or bool(fresh_audits)
    earliest = {}
    for audit in audits:
        for polygon in audit.get("requested_polygons", ()):
            earliest.setdefault(
                (audit["source_entity_id"], polygon["source_polygon_id"]), polygon
            )
    owners = {
        pid: entity for entity in build_input.entities for pid in entity.polygon_ids
    }
    reference = {}
    for polygon in build_input.polygons:
        identity = polygon.metadata.get("geometry_grid_source_identity")
        owner = owners.get(polygon.polygon_id)
        if identity is None and "source_polygon_id" in polygon.metadata:
            identity = {
                "source_entity_id": polygon.metadata["source_semantic_id"],
                "source_polygon_id": polygon.metadata["source_polygon_id"],
            }
        if identity is None and owner is not None:
            identity = {
                "source_entity_id": owner.semantic_id,
                "source_polygon_id": polygon.polygon_id,
            }
        saved = (
            None
            if not use_source_preimage or identity is None
            else earliest.get(
                (identity["source_entity_id"], identity["source_polygon_id"])
            )
        )
        if (
            use_source_preimage
            and "geometry_grid_source_identity" in polygon.metadata
            and saved is None
        ):
            raise ValueError(
                f"declared physical source preimage is unavailable: {polygon.polygon_id}"
            )
        reference[polygon.polygon_id] = saved or {
            "exterior": polygon.exterior,
            "holes": polygon.holes,
        }
    operands = [
        GridOperand(p.polygon_id, p.exterior, p.holes, contact=False)
        for p in build_input.polygons
    ]
    operands.extend(
        GridOperand("PORT::" + p.port_sheet_id, p.exterior, p.holes, contact=False)
        for p in build_input.port_sheet_regions
    )
    arrangement = build_grid_arrangement(operands, context)
    polygons = []
    for polygon in build_input.polygons:
        regions = arrangement.operand_regions(polygon.polygon_id)
        if len(regions) != 1:
            raise ValueError(
                f"source polygon component identity changed at grid: {polygon.polygon_id}"
            )
        outer, holes = regions[0]
        outer = _oriented_like(outer, polygon.exterior)
        holes = tuple(_oriented_like(r, tuple(reversed(outer))) for r in holes)
        polygons.append(
            replace(
                polygon,
                exterior=outer,
                holes=holes,
                metadata={
                    **polygon.metadata,
                    "unquantized_source_geometry": polygon.metadata.get(
                        "unquantized_source_geometry",
                        {
                            "exterior": reference[polygon.polygon_id]["exterior"],
                            "holes": reference[polygon.polygon_id]["holes"],
                        },
                    ),
                },
            )
        )
    # Preserve existing source conductor contact, across exact physically incident layers.
    from shapely import Polygon, union_all
    from scgsim.semantics.route_a import geometry_z_range
    from itertools import combinations

    conductors = [e for e in build_input.entities if e.material_kind == "conductor"]
    raw = {
        e.semantic_id: union_all(
            [
                Polygon(reference[pid]["exterior"], reference[pid]["holes"])
                for pid in e.polygon_ids
            ]
        )
        for e in conductors
    }
    final = {
        e.semantic_id: union_all(
            [
                Polygon(p.exterior, p.holes)
                for p in polygons
                if p.polygon_id in e.polygon_ids
            ],
            grid_size=context.grid_size_um,
        )
        for e in conductors
    }
    contact_relations = []
    for first, second in combinations(conductors, 2):
        a, b = (
            geometry_z_range(first.geometry, first.semantic_id),
            geometry_z_range(second.geometry, second.semantic_id),
        )
        if max(a[0], b[0]) > min(a[1], b[1]):
            continue
        before = raw[first.semantic_id].relate(raw[second.semantic_id])
        after = final[first.semantic_id].relate(final[second.semantic_id])
        if before != after:
            raise ValueError(
                f"conductor contact changed at grid: {first.semantic_id}, {second.semantic_id}"
            )
        contact_relations.append(
            {
                "entity_ids": (first.semantic_id, second.semantic_id),
                "requested_de9im": before,
                "realized_de9im": after,
            }
        )
    by_id = {p.polygon_id: p for p in polygons}
    ports = []
    from scgsim.geometry.source.adapter import (
        _port_sheet_overlaps,
        _is_port_sheet_host_entity,
        _port_sheet_target_layer_matches,
    )
    from scgsim.geometry.models.input import LayoutPolygonSpec

    for port in build_input.port_sheet_regions:
        regions = arrangement.operand_regions("PORT::" + port.port_sheet_id)
        if len(regions) != 1:
            raise ValueError(
                f"port sheet component identity changed at grid: {port.port_sheet_id}"
            )
        outer, holes = regions[0]
        outer = _oriented_like(outer, port.exterior)
        holes = tuple(_oriented_like(r, tuple(reversed(outer))) for r in holes)
        # Source overlap identity binds both Entity AND original polygon. New
        # terminal-owned JJ ends are checked by the partition contract; they
        # must not silently become additional authored overlap carriers.
        host_bindings = {(r.host_semantic_id, r.host_polygon_id) for r in port.overlaps}
        eligible = {
            pid: entity
            for pid, entity in owners.items()
            if _is_port_sheet_host_entity(entity)
            and _port_sheet_target_layer_matches(entity, port.metadata["target_layer"])
        }
        raw_port, final_port = Polygon(port.exterior, port.holes), Polygon(outer, holes)
        port_contacts = []
        for pid, host in eligible.items():
            before_host, after_host = reference[pid], by_id[pid]
            before = raw_port.relate(
                Polygon(before_host["exterior"], before_host["holes"])
            )
            after = final_port.relate(Polygon(after_host.exterior, after_host.holes))
            if before != after:
                raise ValueError(
                    f"port contact changed at grid: {port.port_sheet_id}, {host.semantic_id}, {pid}"
                )
            port_contacts.append(
                {
                    "host_entity_id": host.semantic_id,
                    "host_polygon_id": pid,
                    "requested_de9im": before,
                    "realized_de9im": after,
                }
            )
        host_by_polygon = {
            pid: entity
            for pid, entity in eligible.items()
            if (entity.semantic_id, pid) in host_bindings
        }
        overlaps = _port_sheet_overlaps(
            port_sheet_id=port.port_sheet_id,
            port_polygon=LayoutPolygonSpec(
                port.source_polygon_id, port.source_layer, outer, holes
            ),
            target_layer=port.metadata["target_layer"],
            host_entity_by_polygon_id=host_by_polygon,
            host_polygons_by_id={pid: by_id[pid] for pid in host_by_polygon},
        )
        if {(r.host_semantic_id, r.host_polygon_id) for r in overlaps} != {
            (r.host_semantic_id, r.host_polygon_id) for r in port.overlaps
        }:
            raise ValueError(
                f"port terminal overlap changed at grid: {port.port_sheet_id}"
            )
        ports.append(
            replace(
                port,
                exterior=outer,
                holes=holes,
                overlaps=overlaps,
                metadata={
                    **port.metadata,
                    "geometry_grid_contact_relations": port_contacts,
                    "unquantized_source_geometry": {
                        "exterior": port.exterior,
                        "holes": port.holes,
                    },
                },
            )
        )
    entities = []
    for entity in build_input.entities:
        owned = [by_id[pid] for pid in entity.polygon_ids]
        geometry = dict(entity.geometry)
        if owned and "outer_loop" in geometry:
            combined = union_all(
                [Polygon(p.exterior, p.holes) for p in owned],
                grid_size=context.grid_size_um,
            )
            if combined.geom_type != "Polygon":
                raise ValueError(
                    f"source Entity support changed component identity at grid: {entity.semantic_id}"
                )
            from scgsim.geometry._primitives.grid import _regions

            outer, holes = _regions(
                combined, arrangement.vertices, context.grid_size_um
            )[0]
            geometry.update(outer_loop=outer, hole_loops=holes)
        entities.append(replace(entity, geometry=geometry))
    source = {
        "polygons": [
            {
                "polygon_id": p.polygon_id,
                "layer": p.layer,
                "exterior": reference[p.polygon_id]["exterior"],
                "holes": reference[p.polygon_id]["holes"],
            }
            for p in build_input.polygons
        ],
        "port_sheet_regions": [asdict(p) for p in build_input.port_sheet_regions],
    }
    return replace(
        build_input,
        polygons=tuple(polygons),
        entities=tuple(entities),
        port_sheet_regions=tuple(ports),
        metadata={
            **build_input.metadata,
            "geometry_grid": context.to_payload(),
            "unquantized_source_sha256": hashlib.sha256(
                json.dumps(
                    source, sort_keys=True, separators=(",", ":"), default=list
                ).encode()
            ).hexdigest(),
            "geometry_grid_arrangement": arrangement.metadata,
            "geometry_grid_source_audits": source_grid_audits()
            or build_input.metadata.get("geometry_grid_source_audits", ()),
            "geometry_grid_contact_relations": contact_relations,
        },
    )


def with_source_grid(function):
    """Bind actual source DBU before occurrence-local recipe operations."""
    from functools import wraps
    from scgsim.geometry._primitives.grid import geometry_grid_scope

    @wraps(function)
    def prepare(plan, *args, **kwargs):
        context = source_grid_context(
            float(plan._component.kcl.dbu), plan._grid_size_um
        )
        with geometry_grid_scope(context):
            return function(plan, *args, **kwargs)

    return prepare
