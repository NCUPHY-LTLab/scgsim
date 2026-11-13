"""Assembling the sidecar of a built model and its emitted shapes (design brief sections 7 and 12, #170).

Owns :func:`build_sidecar`: the ``scgsim.sgb.sidecar.v4`` document that :mod:`scgsim.geometry.sgb.output.sidecar`
documents and ``scgsim.contracts.sgb_sidecar`` defines, from the validated model, the numbered plan and the
requests that shaped the model.  It reads the plan and the model only; it never touches geometry or a file.
The tables come in three layers, each from its own module: sources (:mod:`.sources`), derived
(:mod:`.derived`) and emitted (:mod:`.emitted`, :mod:`.groups`, :mod:`.features`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

from scgsim.contracts.sgb_sidecar import SIDECAR_SCHEMA
from scgsim.contracts.sgb_sidecar.records import (
    Counts,
    Face,
    Model,
    Route,
    Sidecar,
    SidecarFile,
    SliverSummary,
    Summary,
    Volume,
)
from scgsim.geometry.sgb.errors import SgbError
from scgsim.geometry.sgb.model import ConformalModel
from scgsim.geometry.sgb.output.emission.plan import BrepPlan
from scgsim.geometry.sgb.output.sidecar.derived import (
    band_family_records,
    carrier_records,
    cut_family_records,
    support_records,
)
from scgsim.geometry.sgb.output.sidecar.emitted import (
    edge_records,
    face_records,
    vertex_records,
    volume_records,
)
from scgsim.geometry.sgb.output.sidecar.features import mesh_features
from scgsim.geometry.sgb.output.sidecar.groups import (
    group_members,
    surface_epr_rows,
    surface_group_records,
    volume_group_records,
)
from scgsim.geometry.sgb.output.sidecar.lineage import CarrierLineage
from scgsim.geometry.sgb.output.sidecar.sources import (
    entity_records,
    inset_request_records,
    port_records,
    source_curve_records,
)

BINDING_RULE = (
    "planned id -> native tag: tag = 1 + position in ids_by_tag[dim] "
    "(TopExp::MapShapes pre-order of the written compound); never by geometry"
)


@dataclass(frozen=True)
class SidecarContext:
    """What the sidecar records besides the model and the plan: the file names and the sliver report summary."""

    sliver_summary: (
        SliverSummary  # ``checks.slivers.report.sliver_report(model)["summary"]``
    )
    files: dict[str, str] = field(default_factory=dict)
    name: str = "model"


def build_sidecar(
    model: ConformalModel, plan: BrepPlan, context: SidecarContext
) -> Sidecar:
    """The sidecar of a model whose emitted shapes are numbered in ``plan``."""
    lineage = CarrierLineage.of(model)
    surfaces = surface_group_records(model, plan)
    volume_groups = volume_group_records(model, plan)
    members = group_members(plan)
    ports = port_records(model, plan)
    rings, primitives = support_records(model, lineage)
    carriers = carrier_records(model, plan, lineage, primitives)
    faces = face_records(model, plan, lineage, carriers)
    volumes = volume_records(model, plan, volume_groups)
    _require_named_members(members, faces, volumes)
    return Sidecar(
        schema=SIDECAR_SCHEMA,
        model=_model(model, plan, context),
        binding={
            "rule": BINDING_RULE,
            "ids_by_tag": {
                "0": plan.vertex_ids,
                "1": plan.edge_ids,
                "2": plan.face_ids,
                "3": plan.solid_ids,
            },
        },
        summary=_summary(model, plan, context),
        entity=entity_records(model),
        source_curve=source_curve_records(model),
        inset_request=inset_request_records(model),
        port=ports,
        cut_family=cut_family_records(model, carriers),
        band_family=band_family_records(model),
        support_ring=rings,
        support_primitive=primitives,
        carrier=carriers,
        vertex=vertex_records(plan),
        edge=edge_records(model, plan),
        face=faces,
        volume=volumes,
        group={**surfaces, **volume_groups},
        surface_epr_row=surface_epr_rows(model, surfaces, members),
        mesh_feature=mesh_features(model, plan, lineage, ports),
    )


def _model(model: ConformalModel, plan: BrepPlan, context: SidecarContext) -> Model:
    policy = model.policy
    return Model(
        name=context.name,
        identity_digest=model.identity_digest(),
        route=cast(Route, policy.name),
        route_policy={
            "conductor_volumes": "excluded"
            if policy.exclude_conductor_volumes
            else "retained",
            "conductor_sheets": policy.allows_sheets,
        },
        grid_um=model.topology.grid_um,
        tolerance_um=plan.tolerance,
        # emit_model names exactly the files of the contract.
        files=cast(dict[SidecarFile, str], dict(context.files)),
    )


def _summary(model: ConformalModel, plan: BrepPlan, context: SidecarContext) -> Summary:
    topology = model.topology
    return Summary(
        counts=cast(Counts, {**plan.counts, "volumes_total": len(topology.volumes)}),
        slivers=context.sliver_summary,
        arc_edges=sum(1 for e in topology.edges.values() if e.curve is not None),
    )


def _require_named_members(
    members: dict[str, list[str]], faces: dict[str, Face], volumes: dict[str, Volume]
) -> None:
    """The plan's members of each group are exactly the written faces and solids that name it.

    A reader computes a group's members from the faces and volumes (``views.group_members``), so the two must
    agree; they do by construction, and this keeps it so.
    """
    named: dict[str, list[str]] = {}
    for face_id, face in sorted(faces.items()):
        if face["group"] is not None:
            named.setdefault(face["group"], []).append(face_id)
    for volume_id, volume in sorted(volumes.items()):
        if volume["emitted"] and volume["group"] is not None:
            named.setdefault(volume["group"], []).append(volume_id)
    differ = sorted(g for g in {*named, *members} if named.get(g) != members.get(g))
    if differ:
        raise SgbError(
            f"internal: the members of groups {differ[:3]} are not the faces and volumes that name them"
        )
