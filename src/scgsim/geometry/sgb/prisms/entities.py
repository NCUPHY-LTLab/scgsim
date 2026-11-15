"""Effective source entities as prisms (stage S2).

Owns the rules that give each entity of an effective ``GeometryBuildInput`` its Z
interval, Net, material and priority, and builds its prisms:

* **Z interval.**  The prepared representation (``RepresentationContext.entities``)
  carries the effective interval and whether the conductor is a *sheet* (a zero
  thickness interval) or a *solid*; solution regions fall back to their effective
  geometry (``z_min_um``/``z_max_um``, else ``z_um`` + ``thickness_um``), the fields
  ``primitives.entities.entity_z_range_um`` reads.  The automatic vacuum
  envelope takes the Z of its prepared enclosure.  Levels are exactly these values:
  no Z threshold merges or drops any of them.
* **Net.**  A conductor without a declared Net is its own Net (a floating
  conductor): its entity id.  Two touching or overlapping entities without a shared
  declared Net therefore fail closed instead of being assumed connected.
* **Polygons.**  One prism per connected source footprint; the polygon index is the
  footprint's position, so lineage names the polygon each edge came from.

``host_void_semantic_id`` and the route-specific ``route_representations`` are not read:
the host of a conductor emerges from the arrangement, and the route is applied after
the prisms exist.
"""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from typing import cast

from scgsim.geometry.exact.boundaries import RingKey
from scgsim.geometry.exact.carriers import ArcCarrier
from scgsim.geometry.exact.errors import InputError
from scgsim.geometry.exact.footprints import SourceFootprint, is_auto_vacuum_region
from scgsim.geometry.exact.grid import exact_z
from scgsim.geometry.exact.plane import Ring
from scgsim.geometry.models.input import GeometryBuildInput, SemanticEntitySpec
from scgsim.geometry.models.physical import PHYSICAL_LAYER_FIELD
from scgsim.geometry.models.representation import RepresentationContext
from scgsim.geometry.sgb.records.prisms import (
    Footprint,
    LineEdge,
    MaterialKind,
    PrismRecord,
    RingPrimitive,
)

ZInterval = tuple[Fraction, Fraction]
RingPrimitives = dict[RingKey, tuple[RingPrimitive, ...]]
SOURCE_KINDS = ("vacuum", "dielectric", "conductor")


def check_entity_kind(entity: SemanticEntitySpec) -> None:
    """Fail closed on an entity whose role and material kind do not agree."""
    if entity.material_kind not in SOURCE_KINDS:
        raise InputError(
            f"{entity.semantic_id} material_kind must be vacuum, dielectric or conductor, "
            f"not {entity.material_kind!r}"
        )
    if entity.role == "solution_region" and entity.material_kind == "conductor":
        raise InputError(
            f"{entity.semantic_id} solution_region must be vacuum or dielectric"
        )
    if entity.role != "solution_region" and entity.material_kind != "conductor":
        raise InputError(
            f"{entity.semantic_id} non-solution entity must be a conductor"
        )


def entity_interval(
    entity: SemanticEntitySpec, representation: RepresentationContext
) -> ZInterval:
    """The effective Z interval of an entity; a sheet has ``z0 == z1``."""
    enclosure = representation.enclosures.get(entity.semantic_id)
    record = representation.entities.get(entity.semantic_id)
    if is_auto_vacuum_region(entity) and enclosure is not None:
        bounds = enclosure.effective_bounds_um
        low, high = exact_z(bounds["z_min_um"]), exact_z(bounds["z_max_um"])
    elif record is not None:
        low, high = (exact_z(v) for v in record.effective_z_interval_um)
    elif entity.material_kind == "conductor":
        raise InputError(
            f"{entity.semantic_id} is a conductor missing from the representation "
            "context; interpret the physical geometry first"
        )
    else:
        low, high = _geometry_interval(entity)
    is_sheet = record is not None and record.representation == "sheet"
    if is_sheet and low != high:
        raise InputError(
            f"{entity.semantic_id} is a sheet but its effective Z interval is [{low}, {high}] um"
        )
    if not is_sheet and not low < high:
        raise InputError(
            f"{entity.semantic_id} has a non-positive Z extent [{low}, {high}] um"
        )
    return low, high


def _source_occurrence(entity: SemanticEntitySpec) -> str | None:
    path = entity.metadata.get("source_occurrence_path")
    return str(path) if path is not None else None


def _source_layer(entity: SemanticEntitySpec) -> str | None:
    logical = entity.metadata.get("logical_layer_id")
    if logical is not None:
        return str(logical)
    layer, datatype = (
        entity.geometry.get("gds_layer"),
        entity.geometry.get("gds_datatype"),
    )
    return f"{layer}/{datatype}" if layer is not None else None


def _geometry_interval(entity: SemanticEntitySpec) -> ZInterval:
    geometry = entity.geometry
    try:
        if "z_min_um" in geometry or "z_max_um" in geometry:
            return exact_z(geometry["z_min_um"]), exact_z(geometry["z_max_um"])
        low = exact_z(geometry["z_um"])
        return low, low + exact_z(geometry["thickness_um"])
    except (KeyError, TypeError, ValueError) as error:
        raise InputError(
            f"{entity.semantic_id} has no finite Z interval (z_min_um/z_max_um or z_um/thickness_um)"
        ) from error


def physical_feature_kind(
    entity: SemanticEntitySpec,
    build_input: GeometryBuildInput,
    representation: RepresentationContext,
) -> str | None:
    """The feature kind of the entity's physical declaration, read from the prepared plan (never guessed).

    The effective input carries the plan payload (``metadata["physical_representation"]``), whose
    declarations are keyed by ``physical_layer_id``; the representation context says which layer an
    entity belongs to.  An entity authored with its own ``physical_layer`` payload is read the same
    way when the plan does not list it.  ``None`` means the entity has no declaration.
    """
    record = representation.entities.get(entity.semantic_id)
    layer_id = record.physical_layer_id if record is not None else None
    plan = build_input.metadata.get("physical_representation")
    declarations = plan.get("declarations", ()) if isinstance(plan, Mapping) else ()
    for declaration in declarations:
        if layer_id is not None and declaration.get("physical_layer_id") == layer_id:
            return str(declaration["feature_kind"])
    authored = entity.metadata.get(PHYSICAL_LAYER_FIELD)
    if isinstance(authored, Mapping) and "feature_kind" in authored:
        return str(authored["feature_kind"])
    return None


def entity_prisms(  # noqa: PLR0913 - an entity, its footprints, interval, bound rings and declaration
    entity: SemanticEntitySpec,
    footprints: list[SourceFootprint],
    interval: ZInterval,
    ring_primitives: RingPrimitives,
    feature_kind: str | None = None,
) -> list[PrismRecord]:
    """One prism per connected source footprint of an entity."""
    kind = cast(MaterialKind, entity.material_kind)
    net = (entity.net_id or entity.semantic_id) if kind == "conductor" else None
    prisms = []
    for source in footprints:
        footprint, arcs = _footprint(source, ring_primitives)
        prisms.append(
            PrismRecord(
                entity_id=entity.semantic_id,
                footprint=footprint,
                z0=interval[0],
                z1=interval[1],
                material=entity.material_id,
                material_kind=kind,
                net_id=net,
                priority=int(entity.priority),
                polygon=source.polygon,
                arcs=arcs,
                feature_kind=feature_kind,
                occurrence=_source_occurrence(entity),
                layer=_source_layer(entity),
            )
        )
    return prisms


def _footprint(
    source: SourceFootprint, ring_primitives: RingPrimitives
) -> tuple[Footprint, tuple[ArcCarrier, ...]]:
    """The oriented footprint of a source polygon; rings with bound curves go through their primitives."""
    keys = [
        RingKey(entity_id=source.entity_id, polygon=source.polygon, ring=i)
        for i in range(len(source.rings()))
    ]
    if not any(key in ring_primitives for key in keys):
        return Footprint.from_rings(source.outer, source.holes), ()
    rings = [
        ring_primitives.get(key) or _line_primitives(ring)
        for key, ring in zip(keys, source.rings(), strict=False)
    ]
    oriented = [
        _oriented_primitives(primitives, ccw=index == 0)
        for index, primitives in enumerate(rings)
    ]
    return Footprint.from_primitives(oriented[0], oriented[1:])


def _line_primitives(ring: Ring) -> tuple[RingPrimitive, ...]:
    count = len(ring)
    return tuple(
        LineEdge(ring[i], ring[(i + 1) % count], i)
        for i in range(count)
        if ring[i] != ring[(i + 1) % count]
    )


def _oriented_primitives(
    primitives: tuple[RingPrimitive, ...], *, ccw: bool
) -> list[RingPrimitive]:
    """Primitives in travel order with the requested orientation (reversed together when it is wrong)."""
    points = []
    for primitive in primitives:
        if isinstance(primitive, LineEdge):
            points.append(primitive.p)
        else:
            points.extend(primitive.witness_points()[:-1])
    twice_area = sum(
        x0 * y1 - x1 * y0
        for (x0, y0), (x1, y1) in zip(points, [*points[1:], points[0]], strict=False)
    )
    if (twice_area > 0) == ccw:
        return list(primitives)
    return [primitive.reversed() for primitive in reversed(primitives)]
