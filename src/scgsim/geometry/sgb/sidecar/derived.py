"""The derived layer of the sidecar: what SGB computed from the sources (design brief sections 5 to 7, #170).

Owns ``band_family`` (one per banded face: its margins, support area and band groups), ``support_ring`` and
``support_primitive`` (the exact boundary of each family's support region: per primitive its curve, length,
whether it is a distance origin or on the die edge, and the source curves it lies on), and ``carrier`` (every
carrier on an emitted horizontal or refinement edge: the source curve it lies on, the support primitive a band
curve follows with its role, margin and piece, or the refinement request it draws).  A band member names its
primitive, which names its ring, which names its family: each link is stated once.
"""

from __future__ import annotations

from dataclasses import dataclass

from scgsim.contracts.sgb_sidecar.records import (
    BandFamily,
    BandMemberCarrier,
    Carrier,
    RefinementCarrier,
    SourceCarrier,
    SupportPrimitive,
    SupportRing,
    UnknownCarrier,
)
from scgsim.geometry.sgb.emission.measures import rounded
from scgsim.geometry.sgb.emission.plan import BrepPlan
from scgsim.geometry.sgb.errors import SgbError
from scgsim.geometry.sgb.model import ConformalModel
from scgsim.geometry.sgb.records.carriers import LineEdge, RingPrimitive
from scgsim.geometry.sgb.records.support import SupportRegion
from scgsim.geometry.sgb.sidecar.groups import band_groups
from scgsim.geometry.sgb.sidecar.lineage import (
    BAND_FAMILY_ROLE,
    REFINEMENT_FAMILY_ROLE,
    CarrierLineage,
)
from scgsim.geometry.sgb.sidecar.sources import (
    arc_curve,
    arc_length_um,
    concrete_support,
    line_curve,
    line_length_um,
)


def band_family_records(model: ConformalModel) -> dict[str, BandFamily]:
    """Every band family: its face (the request every request on the face agrees with), margins, area and bands."""
    grid = model.topology.grid_um
    bands = band_groups(model.topology, model.routed.face_group)
    regions = support_regions(model)
    out: dict[str, BandFamily] = {}
    for request in model.inset_plan.requests:
        region = regions.get(request.request_id)
        out[request.request_id] = BandFamily(
            support=concrete_support(request.support),
            interface=request.interface,
            margins_um=list(request.margins_um),
            margins_grid=list(region.margins) if region else [],
            band_count=len(request.margins_um) + 1,
            support_area_um2=rounded(region.area() * grid * grid) if region else None,
            bands={
                str(index): names
                for index, names in bands.get(request.request_id, {}).items()
            },
        )
    return dict(sorted(out.items()))


def support_regions(model: ConformalModel) -> dict[str, SupportRegion]:
    """The support region of every band family that has one, by family id."""
    return {
        f.family_id: f.support
        for f in model.labelled.catalogue.families
        if f.support is not None
    }


def ring_id(family: str, ring: int) -> str:
    """The id of ring ``ring`` of a family's support region."""
    return f"{family}/ring{ring}"


def primitive_id(family: str, ring: int, index: int) -> str:
    """The id of primitive ``index`` of ring ``ring`` of a family's support region."""
    return f"{family}/ring{ring}/primitive{index}"


def support_records(
    model: ConformalModel, lineage: CarrierLineage
) -> tuple[dict[str, SupportRing], dict[str, SupportPrimitive]]:
    """The rings and primitives of every family's support region."""
    grid = model.topology.grid_um
    rings: dict[str, SupportRing] = {}
    primitives: dict[str, SupportPrimitive] = {}
    for family, region in sorted(support_regions(model).items()):
        for r, ring in enumerate(region.rings):
            rid = ring_id(family, r)
            ids = [primitive_id(family, r, k) for k in range(len(ring))]
            rings[rid] = SupportRing(family=family, index=r, primitives=ids)
            for k, primitive in enumerate(ring):
                primitives[ids[k]] = _primitive(
                    rid, primitive, _place(region, lineage, r, k), grid
                )
    return dict(sorted(rings.items())), dict(sorted(primitives.items()))


@dataclass(frozen=True)
class _Place:
    """What a support region records about one primitive: origin, die edge and the source curves along it."""

    origin: bool
    die_edge: bool
    sources: list[str]


def _place(
    region: SupportRegion, lineage: CarrierLineage, ring: int, index: int
) -> _Place:
    # An empty mask means every primitive is an origin (origin_mask) or none is on the die edge (die_edge_mask).
    origin = region.origin_mask[ring][index] if region.origin_mask else True
    die_edge = region.die_edge_mask[ring][index] if region.die_edge_mask else False
    carriers = region.sources[ring][index] if region.sources else ()
    # A primitive lies on source carriers; a witness chord of a source arc is that arc's curve.
    sources = list(dict.fromkeys(lineage.source_curve(c) for c in carriers))
    return _Place(origin, die_edge, sources)


def _primitive(
    ring: str, primitive: RingPrimitive, place: _Place, grid_um: float
) -> SupportPrimitive:
    if isinstance(primitive, LineEdge):
        curve = line_curve(primitive.p, primitive.q, grid_um)
        length = line_length_um(primitive.p, primitive.q, grid_um)
    else:
        curve = arc_curve(primitive, grid_um)
        length = arc_length_um(primitive, grid_um)
    return SupportPrimitive(
        ring=ring,
        curve=curve,
        length_um=length,
        origin=place.origin,
        die_edge=place.die_edge,
        sources=place.sources,
    )


def carrier_records(
    model: ConformalModel, plan: BrepPlan, lineage: CarrierLineage
) -> dict[str, Carrier]:
    """Every carrier that an emitted horizontal or refinement edge lies on, and what it comes from."""
    topology = model.topology
    on_edges = {
        carrier
        for edge_id in plan.edge_ids
        if topology.edges[edge_id].kind != "vertical"
        for carrier in edge_sources(model, edge_id)
    }
    return {cid: _carrier(cid, lineage) for cid in sorted(on_edges)}


def edge_sources(model: ConformalModel, edge_id: str) -> tuple[str, ...]:
    """The carrier ids of a horizontal or refinement edge, or the Entities of a wall column."""
    topology = model.topology
    return tuple(topology.lineage.get(edge_id) or topology.edges[edge_id].sources)


def _carrier(carrier_id: str, lineage: CarrierLineage) -> Carrier:
    found = lineage.lineage(carrier_id)
    if found is None:
        return UnknownCarrier(kind="unknown")
    if found.kind == "polygon_edge":
        return SourceCarrier(
            kind="source", source_curve=lineage.source_curve(carrier_id)
        )
    role = lineage.role(found.family_id)
    if role == REFINEMENT_FAMILY_ROLE:
        return RefinementCarrier(kind="refinement", request=found.family_id)
    member = found.band_member
    if role != BAND_FAMILY_ROLE or member is None:
        raise SgbError(
            f"internal: the carrier {carrier_id!r} of family {found.family_id!r} ({role}) has no band member"
        )
    return BandMemberCarrier(
        kind="band_member",
        primitive=primitive_id(found.family_id, member.ring, member.index),
        role=member.role,
        margin_grid=member.margin,
        piece=member.piece,
    )
