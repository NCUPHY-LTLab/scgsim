"""The sources layer of the sidecar: what the layout drew and asked for (design brief section 7, #170).

Owns the tables a consumer follows lineage back to: ``entity`` (each Entity's prisms, each with its kind,
material, Net, layer, z range and feature kind), ``source_curve`` (each straight edge and arc of each Entity outline, with its exact
curve and length), ``inset_request`` (the inset requests as given, with the band families each became) and
``port`` (each lumped-port sheet with the faces and volumes that carry it).  Lengths and coordinates are micrometres, rounded
like every value of the sidecar (``output.emission.measures.rounded``).
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import cast

from scgsim.contracts.sgb_sidecar.records import (
    ArcCurve,
    BandedFaceKind,
    Entity,
    EntityPrism,
    FaceSupport,
    InsetRequest,
    InterfaceSupport,
    LineCurve,
    Port,
    SkippedFace,
    SourceCurve,
    SurfaceEprType,
    TypeSupport,
)
from scgsim.geometry.exact.carriers import ArcCarrier, Lineage
from scgsim.geometry.exact.errors import SgbError
from scgsim.geometry.exact.plane import Point
from scgsim.geometry.sgb.bands.inset_plan import TypeExpansion
from scgsim.geometry.sgb.bands.requests import (
    ConductorSurface,
    InterfaceSurfaces,
    SolutionInterface,
)
from scgsim.geometry.sgb.bands.requests import (
    InsetRequest as InsetRequestRecord,
)
from scgsim.geometry.sgb.model import ConformalModel
from scgsim.geometry.sgb.output.emission.measures import rounded
from scgsim.geometry.sgb.output.emission.plan import BrepPlan
from scgsim.geometry.sgb.records.prisms import PrismRecord
from scgsim.geometry.sgb.topology.routes.naming import port_group
from scgsim.refusal import internal


def entity_records(model: ConformalModel) -> dict[str, Entity]:
    """Every Entity of the catalogue, with each of its prisms (polygons) as the catalogue gives it."""
    out: dict[str, Entity] = {}
    for prism in model.labelled.catalogue.prisms:
        out.setdefault(prism.entity_id, Entity(prisms=[]))["prisms"].append(
            _entity_prism(prism)
        )
    return dict(sorted(out.items()))


def _entity_prism(prism: PrismRecord) -> EntityPrism:
    return EntityPrism(
        polygon=prism.polygon,
        kind=prism.material_kind,
        material=prism.material,
        net=prism.net_id or None,
        layer=prism.layer,
        z_um=[float(prism.z0), float(prism.z1)],
        feature_kind=prism.feature_kind,
    )


def source_curve_records(model: ConformalModel) -> dict[str, SourceCurve]:
    """Every straight edge and arc of every Entity outline, by carrier id: where it lies, its curve and length.

    The witness chords an arc is noded through are not source curves: they are carriers of the arc's curve.
    """
    grid = model.topology.grid_um
    out: dict[str, SourceCurve] = {}
    for prism in model.labelled.catalogue.prisms:
        for carrier in prism.carriers():
            if carrier.role == "source_edge":
                curve = line_curve(carrier.p, carrier.q, grid)
                out[carrier.id] = _source_curve(
                    carrier.lineage, curve, line_length_um(carrier.p, carrier.q, grid)
                )
        for arc in prism.arcs:
            out[arc.id] = _source_curve(
                arc.lineage, arc_curve(arc, grid), arc_length_um(arc, grid)
            )
    return dict(sorted(out.items()))


def _source_curve(
    lineage: Lineage, curve: LineCurve | ArcCurve, length_um: float
) -> SourceCurve:
    return SourceCurve(
        entity=lineage.entity_id,
        polygon=lineage.polygon,
        ring=lineage.ring,
        edge=lineage.edge,
        curve=curve,
        length_um=length_um,
    )


def line_curve(p: Point, q: Point, grid_um: float) -> LineCurve:
    """The straight curve from lattice point ``p`` to ``q``, in micrometres."""
    return LineCurve(
        kind="line", start_um=_point_um(p, grid_um), end_um=_point_um(q, grid_um)
    )


def line_length_um(p: Point, q: Point, grid_um: float) -> float:
    """The length of the straight curve from ``p`` to ``q``, in micrometres."""
    return rounded(math.hypot(q[0] - p[0], q[1] - p[1]) * grid_um)


def arc_curve(arc: ArcCarrier, grid_um: float) -> ArcCurve:
    """The exact arc of a carrier, in micrometres."""
    return ArcCurve(
        kind="arc",
        centre_um=_point_um(arc.centre, grid_um),
        radius_um=rounded(float(arc.radius) * grid_um),
        ccw=arc.ccw,
        start_um=_point_um(arc.start, grid_um),
        end_um=_point_um(arc.end, grid_um),
    )


def arc_length_um(arc: ArcCarrier, grid_um: float) -> float:
    """The length of a carrier arc: its radius times its sweep, in micrometres."""
    _, sweep = arc.sweep()
    return rounded(abs(float(sweep)) * float(arc.radius) * grid_um)


def _point_um(point: tuple[Fraction, Fraction] | Point, grid_um: float) -> list[float]:
    return [rounded(float(point[0]) * grid_um), rounded(float(point[1]) * grid_um)]


# --------------------------------------------------------------------------- requests


def inset_request_records(model: ConformalModel) -> dict[str, InsetRequest]:
    """The inset requests as given, with the band families each became and the faces it skipped."""
    expansions = {e.request.request_id: e for e in model.inset_plan.expansions}
    out: dict[str, InsetRequest] = {}
    for request in sorted(model.insets, key=lambda r: r.request_id):
        if isinstance(request.support, InterfaceSurfaces):
            out[request.request_id] = _type_request(
                request, request.support, expansions[request.request_id]
            )
        else:
            out[request.request_id] = InsetRequest(
                scope="face",
                support=concrete_support(request.support),
                interface=request.interface,
                margins_um=list(request.margins_um),
                families=[request.request_id],  # a concrete request is its own family
                skipped=[],
            )
    return out


def _type_request(
    request: InsetRequestRecord, support: InterfaceSurfaces, expansion: TypeExpansion
) -> InsetRequest:
    return InsetRequest(
        scope="type",
        support=TypeSupport(
            # plan_insets accepted the type and the face kinds, so they are among the contract's values.
            interface_type=cast(SurfaceEprType, support.interface_type),
            face_kinds=None
            if support.face_kinds is None
            else cast(list[BandedFaceKind], list(support.face_kinds)),
            exclude_entity_ids=list(support.exclude_entity_ids),
        ),
        interface=None,
        margins_um=list(request.margins_um),
        families=list(expansion.families),
        skipped=[
            SkippedFace(support=concrete_support(s.support), reason="no_gap_edge")
            for s in expansion.skipped
        ],
    )


def concrete_support(
    support: ConductorSurface | SolutionInterface | InterfaceSurfaces,
) -> FaceSupport | InterfaceSupport:
    """The record of the one face a concrete request names."""
    if isinstance(support, ConductorSurface):
        return FaceSupport(entity=support.entity_id, side=support.side)
    if isinstance(support, SolutionInterface):
        return InterfaceSupport(
            first_entity=support.first_entity_id,
            second_entity=support.second_entity_id,
            z_um=support.z_um,
        )
    raise SgbError(internal(f"{support!r} names every face of a type, not one face"))


def port_records(model: ConformalModel, plan: BrepPlan) -> dict[str, Port]:
    """Lumped-port sheets: the authored identity, the sheet's Entity, and the faces and volumes that carry it."""
    topology = model.topology
    host_of = {
        face: volume_id
        for volume_id in plan.solid_ids
        for face in topology.volumes[volume_id].embedded_faces
    }
    out: dict[str, Port] = {}
    for prism in model.labelled.catalogue.prisms:
        identity = prism.port
        if identity is None:
            continue
        group = port_group(prism.entity_id)
        faces = sorted(
            fid for fid in plan.face_ids if model.routed.face_group.get(fid) == group
        )
        out[identity.port_sheet_id] = Port(
            entity=prism.entity_id,
            group=group,
            source_name=identity.source_name,
            port_index=identity.port_index,
            direction=list(identity.direction),
            host_entity_ids=list(identity.host_entity_ids),
            source_layer=identity.source_layer,
            target_layer=identity.target_layer,
            z_um=float(prism.z0),
            faces=faces,
            host_volumes=sorted({host_of[f] for f in faces if f in host_of}),
            adjacent_volumes=sorted(
                {
                    v
                    for f in faces
                    for v in (
                        topology.faces[f].neg_volume,
                        topology.faces[f].pos_volume,
                    )
                    if v is not None and v in model.routed.emitted_volumes
                }
            ),
        )
    return dict(sorted(out.items()))
