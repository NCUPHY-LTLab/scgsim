"""Select ground-bump lattice sites in the established source order."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import ceil
from typing import Any

from scgsim.geometry.primitives.grid import polygon_boolean
from scgsim.geometry.source.ground_bump_geometry.layout import (
    intersecting_blocker_ids,
    translate_polygon,
)


@dataclass(frozen=True)
class LatticeSelectionContext:
    """Hold PDK geometry and semantic facts for ordered lattice selection."""

    gdstk: Any
    origin: tuple[float, float]
    pitch: float
    clearance: float
    body_polygon: Sequence[tuple[float, float]]
    contact_polygon: Sequence[tuple[float, float]]
    fill_prefix: str
    geometry_digest: str
    semantic_contract: Mapping[str, Any]
    ground: Sequence[Any]
    blockers: list[tuple[str, Sequence[Any]]]


def select_lattice_sites(
    context: LatticeSelectionContext, bounds: Mapping[str, float]
) -> tuple[list[tuple[int, int, float, float]], list[dict[str, Any]]]:
    """Return accepted and rejected sites using the existing nested lattice order."""
    accepted: list[tuple[int, int, float, float]] = []
    rejected: list[dict[str, Any]] = []
    x_indices = range(
        ceil((bounds["x_min_um"] - context.origin[0]) / context.pitch),
        ceil((bounds["x_max_um"] - context.origin[0]) / context.pitch) + 1,
    )
    y_indices = range(
        ceil((bounds["y_min_um"] - context.origin[1]) / context.pitch),
        ceil((bounds["y_max_um"] - context.origin[1]) / context.pitch) + 1,
    )
    for i in x_indices:
        for j in y_indices:
            site = (
                i,
                j,
                context.origin[0] + i * context.pitch,
                context.origin[1] + j * context.pitch,
            )
            candidate, rejection = _evaluate_lattice_site(context, site)
            if rejection is not None:
                rejected.append(rejection)
                continue
            accepted.append(site)
            # Accepted geometry blocks later lattice sites in this same order.
            context.blockers.append(
                (f"generated_site:{fill_id(context.fill_prefix, i, j)}", candidate)
            )
    return accepted, rejected


def _evaluate_lattice_site(
    context: LatticeSelectionContext, site: tuple[int, int, float, float]
) -> tuple[tuple[Any, ...], dict[str, Any] | None]:
    i, j, x, y = site
    candidate = (
        context.gdstk.Polygon(translate_polygon(context.body_polygon, x, y)),
        context.gdstk.Polygon(translate_polygon(context.contact_polygon, x, y)),
    )
    expanded = (
        context.gdstk.offset(candidate, context.clearance, join="round", tolerance=1e-6)
        if context.clearance
        else list(candidate)
    )
    candidate_id = fill_id(context.fill_prefix, i, j)
    blocker_ids: tuple[str, ...] = ()
    if polygon_boolean(expanded, context.ground, "not", precision=1e-9):
        reason = "outside_d0_d1_ground_intersection"
        blocker_ids = ("D0_D1_GROUND_INTERSECTION",)
    else:
        blocker_ids = intersecting_blocker_ids(expanded, context.blockers)
        reason = "intersects_blocking_footprint" if blocker_ids else None
    if not reason:
        return candidate, None
    semantic_contract = context.semantic_contract
    rejection = {
        "semantic_id": candidate_id,
        "lattice_index": [i, j],
        "center_um": [x, y],
        "geometry_placement_sha256": placement_digest(
            semantic_id=candidate_id,
            lattice_index=(i, j),
            center_um=(x, y),
            geometry_digest=context.geometry_digest,
        ),
        "owner_semantic_ids": semantic_contract["owner_semantic_ids"],
        "net_id": semantic_contract["net_id"],
        "equipotential_id": semantic_contract["equipotential_id"],
        "host_solution_volume_id": semantic_contract["host_solution_volume_id"],
        "source_kind": "fill",
        "source_semantic_id": candidate_id,
        "disposition": "rejected",
        "reason": reason,
        "blocker_source_ids": list(blocker_ids),
    }
    return candidate, rejection


def accepted_site_records(
    context: LatticeSelectionContext,
    accepted: Sequence[tuple[int, int, float, float]],
) -> list[dict[str, Any]]:
    """Build stable receipt records for accepted lattice sites."""
    contract = context.semantic_contract
    records = []
    for i, j, x, y in accepted:
        semantic_id = fill_id(context.fill_prefix, i, j)
        records.append(
            {
                "semantic_id": semantic_id,
                "lattice_index": [i, j],
                "center_um": [x, y],
                "geometry_placement_sha256": placement_digest(
                    semantic_id=semantic_id,
                    lattice_index=(i, j),
                    center_um=(x, y),
                    geometry_digest=context.geometry_digest,
                ),
                "owner_semantic_ids": contract["owner_semantic_ids"],
                "net_id": contract["net_id"],
                "equipotential_id": contract["equipotential_id"],
                "host_solution_volume_id": contract["host_solution_volume_id"],
                "source_kind": "fill",
                "source_semantic_id": semantic_id,
                "disposition": "accepted",
                "source_provenance": "public_pdk_indium_ground_fill",
            }
        )
    return records


def placement_digest(
    *,
    semantic_id: str,
    lattice_index: tuple[int, int],
    center_um: tuple[float, float],
    geometry_digest: str,
) -> str:
    """Hash one site's identity and placement with its source geometry."""
    payload = {
        "semantic_id": semantic_id,
        "lattice_index": list(lattice_index),
        "center_um": list(center_um),
        "pdk_geometry_sha256": geometry_digest,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def fill_id(prefix: str, i: int, j: int) -> str:
    """Build the stable generated-site identifier from lattice indices."""
    return f"{prefix}__I{i:+05d}__J{j:+05d}"
