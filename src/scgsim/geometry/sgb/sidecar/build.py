"""Assembling the sidecar of a built model and its emitted shapes (design brief sections 7 and 12).

Owns :func:`build_sidecar`: the dictionary that :mod:`scgsim.geometry.sgb.sidecar` documents and
``sidecar.schema.json`` constrains, from the validated model, the numbered plan and the requests that
shaped the model.  It reads the plan and the model only; it never touches geometry or a file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from scgsim.geometry.sgb.cutfamilies.inset_plan import TypeExpansion
from scgsim.geometry.sgb.cutfamilies.requests import (
    ConductorSurface,
    InsetRequest,
    InterfaceSurfaces,
    RefinementRequest,
    SolutionInterface,
)
from scgsim.geometry.sgb.emission.measures import planned_volume_um3, rounded
from scgsim.geometry.sgb.emission.plan import BrepPlan
from scgsim.geometry.sgb.emission.surfaces import CYLINDER
from scgsim.geometry.sgb.model import ConformalModel
from scgsim.geometry.sgb.records.support import SupportRegion
from scgsim.geometry.sgb.sidecar.aedt import aedt_names
from scgsim.geometry.sgb.sidecar.expected import expected_record
from scgsim.geometry.sgb.sidecar.features import mesh_features
from scgsim.geometry.sgb.sidecar.groups import (
    band_groups,
    face_kind,
    surface_epr_rows,
    surface_group_records,
    unions,
    volume_group_records,
)
from scgsim.geometry.sgb.sidecar.lineage import edge_family, lineage_record

SCHEMA = "scgsim.sgb.sidecar.v2"
BINDING_RULE = (
    "planned id -> native tag: tag = 1 + position in ids_by_tag[dim] "
    "(TopExp::MapShapes pre-order of the written compound); never by geometry"
)


@dataclass(frozen=True)
class SidecarContext:
    """What the sidecar records besides the model and the plan: the file names and the sliver report summary."""

    sliver_summary: dict[str, Any]  # ``slivers.report.sliver_report(model)["summary"]``
    files: dict[str, str] = field(default_factory=dict)
    name: str = "model"


def build_sidecar(
    model: ConformalModel, plan: BrepPlan, context: SidecarContext
) -> dict[str, Any]:
    """The sidecar of a model whose emitted shapes are numbered in ``plan``."""
    topology = model.topology
    surfaces = surface_group_records(model, plan)
    volume_groups = volume_group_records(model, plan)
    lineage = lineage_record(model, plan)
    embedded = _embedded_faces(model, plan)
    refinement_edges = _refinement_edges(model, plan, lineage["carriers"])
    ports = _ports(model, plan, embedded)
    policy = model.policy
    return {
        "schema": SCHEMA,
        "route": policy.name,
        "route_policy": {
            "conductor_volumes": "excluded"
            if policy.exclude_conductor_volumes
            else "retained",
            "conductor_sheets": policy.allows_sheets,
        },
        "grid_um": topology.grid_um,
        "tolerance_um": plan.tolerance,
        "model": {"name": context.name, "identity_digest": model.identity_digest()},
        "files": dict(context.files),
        "binding": {
            "rule": BINDING_RULE,
            "ids_by_tag": {
                "0": plan.vertex_ids,
                "1": plan.edge_ids,
                "2": plan.face_ids,
                "3": plan.solid_ids,
            },
        },
        "counts": {**plan.counts, "volumes_total": len(topology.volumes)},
        "physical_groups": {**surfaces, **volume_groups},
        "unions": unions(model),
        "surface_epr_rows": surface_epr_rows(model, surfaces),
        "volumes": _volumes(model, plan, volume_groups),
        "faces": _faces(model, plan),
        "embedded_faces": embedded,
        "refinement_edges": refinement_edges,
        "mesh_features": mesh_features(model, plan, lineage["carriers"], ports),
        "ports": ports,
        "lineage": lineage,
        "insets": _insets(model),
        "refinements": _refinements(model.refinements, refinement_edges),
        "slivers": context.sliver_summary,
        "arrangement": {
            "max_snap_drift_grid": context.sliver_summary.get("max_snap_drift_grid"),
            "max_arc_deviation_grid": context.sliver_summary.get(
                "max_arc_deviation_grid"
            ),
            "arc_edges": sum(1 for e in topology.edges.values() if e.curve is not None),
        },
        "curved_wall_faces": [f for f in plan.face_ids if topology.faces[f].curved],
        "surface_kinds": {
            "plane": int((plan.surf_kind != CYLINDER).sum()),
            "cylinder": int((plan.surf_kind == CYLINDER).sum()),
        },
        "expected": expected_record(topology, plan),
    }


def _volumes(
    model: ConformalModel, plan: BrepPlan, volume_groups: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Every planned volume with its kind, and for emitted solids the group, planned volume and AEDT name."""
    topology = model.topology
    group_of = {vid: name for name, g in volume_groups.items() for vid in g["ids"]}
    names = aedt_names([(vid, group_of[vid]) for vid in plan.solid_ids])
    out: dict[str, dict[str, Any]] = {}
    for vid, volume in sorted(topology.volumes.items()):
        emitted = vid in model.routed.emitted_volumes
        out[vid] = {
            "entity": volume.domain.name,
            "kind": volume.domain.kind,
            "material": volume.domain.material,
            "net": volume.domain.net or None,
            "emitted": emitted,
            "group": group_of.get(vid),
            "shells": len(volume.shells),
            "volume_um3": rounded(planned_volume_um3(topology, volume)),
            "aedt_name": names.get(vid),
        }
    return out


def _faces(model: ConformalModel, plan: BrepPlan) -> dict[str, dict[str, Any]]:
    topology = model.topology
    return {
        fid: {
            "group": model.routed.face_group.get(fid),
            "kind": face.kind,
            "face_kind": face_kind(face),
            "z_um": [float(face.z_range[0]), float(face.z_range[1])],
            "negative_volume": face.neg_volume,
            "positive_volume": face.pos_volume,
        }
        for fid in plan.face_ids
        for face in (topology.faces[fid],)
    }


def _embedded_faces(model: ConformalModel, plan: BrepPlan) -> list[dict[str, Any]]:
    """(face, host volume) pairs: sheets inside an emitted volume that bound no shell."""
    topology = model.topology
    return [
        {"face": fid, "host_volume": vid, "group": model.routed.face_group.get(fid)}
        for vid in plan.solid_ids
        for fid in topology.volumes[vid].embedded_faces
    ]


def _refinement_edges(
    model: ConformalModel, plan: BrepPlan, carriers: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """(edge, host face) pairs: refinement-only curves embedded in the face that contains them."""
    topology = model.topology
    free = {plan.edge_ids[i] for i in plan.found.free_edges}
    return [
        {
            "edge": eid,
            "host_face": fid,
            "family": edge_family(topology, eid, carriers),
            "written_as_free_edge": eid in free,
        }
        for fid in plan.face_ids
        for eid in topology.faces[fid].refinement_edges
    ]


def _ports(
    model: ConformalModel, plan: BrepPlan, embedded: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Lumped-port sheets: the authored identity plus the faces that carry them in the model."""
    topology = model.topology
    host_of = {e["face"]: e["host_volume"] for e in embedded}
    out = []
    for prism in model.labelled.catalogue.prisms:
        identity = prism.port
        if identity is None:
            continue
        group = f"PORT::{prism.entity_id}"
        faces = sorted(
            fid for fid in plan.face_ids if model.routed.face_group.get(fid) == group
        )
        out.append(
            {
                "port_sheet_id": identity.port_sheet_id,
                "group": group,
                "source_name": identity.source_name,
                "port_index": identity.port_index,
                "direction": list(identity.direction),
                "host_entity_ids": list(identity.host_entity_ids),
                "source_layer": identity.source_layer,
                "target_layer": identity.target_layer,
                "z_um": float(prism.z0),
                "faces": faces,
                "host_volumes": sorted({host_of[f] for f in faces if f in host_of}),
                "adjacent_volumes": sorted(
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
            }
        )
    return sorted(out, key=lambda p: (p["port_index"], p["port_sheet_id"]))


def _insets(model: ConformalModel) -> list[dict[str, Any]]:
    """The inset requests as given.

    A concrete request is the record of its face (its family).  A type-level request lists the families that carry
    its bands, each as the record of its face, and the faces it skipped with the reason.
    """
    facts = _FamilyFacts(
        model.topology.grid_um,
        band_groups(model.topology, model.routed.face_group),
        {
            f.family_id: f.support
            for f in model.labelled.catalogue.families
            if f.support is not None
        },
    )
    families = {r.request_id: r for r in model.inset_plan.requests}
    expansions = {e.request.request_id: e for e in model.inset_plan.expansions}
    out = []
    for request in sorted(model.insets, key=lambda r: r.request_id):
        if isinstance(request.support, InterfaceSurfaces):
            out.append(
                _type_record(request, expansions[request.request_id], families, facts)
            )
        else:
            out.append(_face_record(families[request.request_id], facts))
    return out


@dataclass(frozen=True)
class _FamilyFacts:
    """What the model knows about its band families: the grid, each family's band groups and support region."""

    grid_um: float
    bands: dict[str, dict[int, list[str]]]
    regions: dict[str, SupportRegion]


def _face_record(request: InsetRequest, facts: _FamilyFacts) -> dict[str, Any]:
    """One band family: its request (margins are the union of every request on the face), margins in grid units, area and band groups."""
    region = facts.regions.get(request.request_id)
    return {
        "request_id": request.request_id,
        "support": _support_record(request.support),
        "interface": request.interface,
        "margins_um": list(request.margins_um),
        "margins_grid": list(region.margins) if region else [],
        "band_count": len(request.margins_um) + 1,
        "support_area_um2": rounded(region.area() * facts.grid_um * facts.grid_um)
        if region
        else None,
        "bands": {
            str(index): names
            for index, names in facts.bands.get(request.request_id, {}).items()
        },
    }


def _type_record(
    request: InsetRequest,
    expansion: TypeExpansion,
    families: dict[str, InsetRequest],
    facts: _FamilyFacts,
) -> dict[str, Any]:
    """A type-level request as given, and the faces it was expanded into or skipped."""
    return {
        "request_id": request.request_id,
        "support": _support_record(request.support),
        "interface": None,
        "margins_um": list(request.margins_um),
        "expansion": {
            "banded": [_face_record(families[i], facts) for i in expansion.banded],
            "skipped": [
                {"support": _support_record(s.support), "reason": s.reason}
                for s in expansion.skipped
            ],
        },
    }


def _support_record(
    support: ConductorSurface | SolutionInterface | InterfaceSurfaces,
) -> dict[str, Any]:
    if isinstance(support, ConductorSurface):
        return {"entity": support.entity_id, "side": support.side}
    if isinstance(support, SolutionInterface):
        return {
            "first_entity": support.first_entity_id,
            "second_entity": support.second_entity_id,
            "z_um": support.z_um,
        }
    return {
        "interface_type": support.interface_type,
        "face_kinds": None if support.face_kinds is None else list(support.face_kinds),
        "exclude_entity_ids": list(support.exclude_entity_ids),
    }


def _refinements(
    requests: tuple[RefinementRequest, ...], refinement_edges: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    return [
        {
            "request_id": r.request_id,
            "z_um": r.z_um,
            "points_um": [list(p) for p in r.points_um],
            "edges": sorted(
                e["edge"] for e in refinement_edges if e["family"] == r.request_id
            ),
        }
        for r in sorted(requests, key=lambda r: r.request_id)
    ]
