"""Lumped-port sheet regions as sheet prisms (stage S2).

Owns turning each adapter-authored ``PortSheetRegionRecord`` into a sheet prism that
carries the port identity.  The semantics follow the compiler's port lowering
(``planning.port_planning.lowering`` and the curved arrangement's ``port_planes``):

* **Plane.**  The port sheet lies at the middle of its host conductors' effective Z
  interval.  A Route-A host is a sheet (a zero-thickness interval), so the middle is
  the sheet plane; a Route-B host is a finite body and the middle is its mid-plane.
  Both hosts must share one interval.  One rule for both, because the prism catalogue
  does not know the route.
* **Footprint.**  The *active remainder*: the authored region minus each host overlap
  (the landing of the port on its host).  The overlaps enter as extra hole rings, which
  the even-odd footprint test reads as removed area.  A region bound to a rectangular
  junction partition is already the junction's central rectangle and keeps its full
  footprint (``lowering._PortSheetLowerer.read_geometry``).
* **Identity.**  Port sheet id, source name, 1-based port index, normalised XY
  direction, host entity ids and source/target layers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
from math import hypot, isfinite

from scgsim.geometry.models.input import GeometryBuildInput
from scgsim.geometry.models.regions import PortSheetRegionRecord
from scgsim.geometry.sgb.errors import InputError
from scgsim.geometry.sgb.prisms.footprints import lattice_ring
from scgsim.geometry.sgb.records.grid import Grid
from scgsim.geometry.sgb.records.prisms import Footprint, PortIdentity, PrismRecord

PORT_MATERIAL = "lumped_port"
JUNCTION_PARTITION_KEY = "rectangular_junction_partition"


def port_prisms(
    build_input: GeometryBuildInput,
    grid: Grid,
    host_intervals: Mapping[str, tuple[Fraction, Fraction]],
) -> list[PrismRecord]:
    """One sheet prism per port sheet region, in source order.

    ``host_intervals`` maps a conductor entity id to its effective ``(z0, z1)``.
    """
    return [
        _port_prism(region, grid, host_intervals)
        for region in build_input.port_sheet_regions
    ]


def _port_prism(
    region: PortSheetRegionRecord,
    grid: Grid,
    host_intervals: Mapping[str, tuple[Fraction, Fraction]],
) -> PrismRecord:
    what = f"port sheet {region.port_sheet_id!r}"
    hosts = tuple(dict.fromkeys(o.host_semantic_id for o in region.overlaps))
    plane = _plane(what, hosts, host_intervals)
    exterior = lattice_ring(region.exterior, grid, f"{what} exterior")
    holes = [
        lattice_ring(hole, grid, f"{what} hole {k}")
        for k, hole in enumerate(region.holes)
    ]
    if JUNCTION_PARTITION_KEY not in region.metadata:
        holes.extend(
            lattice_ring(o.overlap_loop, grid, f"{what} overlap {o.overlap_id!r}")
            for o in region.overlaps
        )
    footprint = Footprint.from_rings(exterior, holes)
    if footprint.twice_area() <= 0:
        raise InputError(
            f"{what} has no active remainder: its host overlaps cover it entirely"
        )
    return PrismRecord(
        entity_id=region.port_sheet_id,
        footprint=footprint,
        z0=plane,
        z1=plane,
        material=PORT_MATERIAL,
        material_kind="port",
        port=PortIdentity(
            port_sheet_id=region.port_sheet_id,
            source_name=str(region.metadata.get("source_name", region.port_sheet_id)),
            port_index=int(region.metadata.get("port_index", 0)),
            direction=port_direction(region.metadata.get("direction"), what),
            host_entity_ids=hosts,
            target_layer=str(region.metadata.get("target_layer", "")),
            source_layer=region.source_layer,
        ),
    )


def _plane(
    what: str, hosts: Sequence[str], intervals: Mapping[str, tuple[Fraction, Fraction]]
) -> Fraction:
    if not hosts:
        raise InputError(f"{what} has no host overlap, so its plane is undefined")
    unknown = [h for h in hosts if h not in intervals]
    if unknown:
        raise InputError(
            f"{what} references host(s) that are not conductor entities: {unknown}"
        )
    distinct = {intervals[h] for h in hosts}
    if len(distinct) != 1:
        raise InputError(
            f"{what} hosts {list(hosts)} do not share one effective Z interval: "
            f"{sorted((str(z0), str(z1)) for z0, z1 in distinct)}"
        )
    z0, z1 = distinct.pop()
    return (z0 + z1) / 2


def port_direction(raw: object, what: str) -> tuple[float, float, float]:
    """The authored XY direction, normalised; a defect fails closed."""
    if isinstance(raw, str | bytes) or not isinstance(raw, Sequence) or len(raw) != 3:
        raise InputError(f"{what} requires a 3D direction")
    try:
        values = tuple(float(v) for v in raw)
    except (TypeError, ValueError) as error:
        raise InputError(f"{what} direction must be numeric") from error
    if (
        not all(isfinite(v) for v in values)
        or values[2] != 0.0
        or values[0] == values[1] == 0.0
    ):
        raise InputError(f"{what} direction must be finite, XY, and nonzero")
    length = hypot(values[0], values[1])
    return (values[0] / length, values[1] / length, 0.0)
