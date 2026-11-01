"""Partition each Route A sheet into patches by the solution entities below and above it.

Route A draws a thin conductor as a zero-thickness sheet (``docs/geometry-sgb.qmd``).
Support is local: part of a sheet may rest on a substrate while another part is
exposed, so evidence is attached per patch. Does not build the surfaces; see
``sheet_surfaces``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any, NamedTuple

from scgsim.geometry.models.common import RouteLiteral
from scgsim.geometry.models.input import GeometryBuildInput, SemanticEntitySpec
from scgsim.geometry.models.representation import RepresentationContext
from scgsim.geometry.models.topology import InterfacePlanRecord
from scgsim.geometry.planning.conductors.faces import planar_side_solution_regions
from scgsim.geometry.planning.context.evidence import conductor_solution_evidence
from scgsim.geometry.planning.solutions.adjacency import route_a_sheet_plane_z_um
from scgsim.geometry.primitives.entities import (
    entity_by_id,
    geometry_ref_from_metadata,
)
from scgsim.geometry.primitives.loops import (
    boolean_gdstk_region,
    gdstk_surface_region,
    geometry_refs_from_gdstk_region,
    loop_signature,
)
from scgsim.geometry.primitives.surface_records import (
    RouteASheetPatch,
    is_route_a_sheet_interface,
)
from scgsim.semantics import SemanticEvidenceFacade


def plan_route_a_sheet_patches(
    build_input: GeometryBuildInput,
    *,
    interfaces: Sequence[InterfacePlanRecord],
    semantic_facts: SemanticEvidenceFacade,
    route: RouteLiteral | None = "A",
    representation_context: RepresentationContext | None = None,
) -> tuple[RouteASheetPatch, ...]:
    """Partition each Route-A sheet by its exact ordered local domains.

    Route A support is local, not one label per sheet: part of a sheet may rest on a
    substrate while another part is exposed (see ``docs/geometry-sgb.qmd``). Each
    stretch with its own (bottom solution, top solution) pair becomes one patch.
    """
    import gdstk

    raw_patches: list[_RawSheetPatch] = []
    for interface in interfaces:
        if not is_route_a_sheet_interface(
            route, interface, representation_context=representation_context
        ):
            continue
        raw_patches.extend(
            _raw_sheet_patches(
                gdstk,
                build_input,
                interface,
                representation_context=representation_context,
            )
        )
    return _numbered_sheet_patches(raw_patches, semantic_facts, route)


class _RawSheetPatch(NamedTuple):
    """A sheet stretch over one (bottom solution, top solution) pair, before numbering."""

    interface: InterfacePlanRecord
    sheet: SemanticEntitySpec
    bottom_id: str
    top_id: str
    geometry_ref: dict[str, Any]


def _raw_sheet_patches(
    gdstk: Any,
    build_input: GeometryBuildInput,
    interface: InterfacePlanRecord,
    *,
    representation_context: RepresentationContext | None,
) -> list[_RawSheetPatch]:
    """Split one sheet interface by the solutions below and above it.

    Raises when the patches do not cover the sheet or when every patch has the same
    solution on both sides, because then nothing physically supports the sheet.
    """
    raw_patches: list[_RawSheetPatch] = []
    sheet = entity_by_id(build_input, interface.owner_semantic_ids[0])
    plane_z_um = route_a_sheet_plane_z_um(
        build_input, sheet, representation_context=representation_context
    )
    parent_geometry_ref = {
        "from_interface_id": interface.interface_id,
        "source_polygon_ids": interface.source_polygon_ids,
        **geometry_ref_from_metadata(interface.metadata),
        "plane": {"axis": "z", "value_um": plane_z_um},
        "representation": "surface_sheet",
    }
    sheet_region = gdstk_surface_region(parent_geometry_ref)
    if not sheet_region:
        raise ValueError(f"{sheet.semantic_id} Route A sheet has no occupied region")
    bottom_regions = _route_a_sheet_side_solution_regions(
        build_input,
        sheet=sheet,
        sheet_region=sheet_region,
        plane_z_um=plane_z_um,
        side="bottom",
    )
    top_regions = _route_a_sheet_side_solution_regions(
        build_input,
        sheet=sheet,
        sheet_region=sheet_region,
        plane_z_um=plane_z_um,
        side="top",
    )
    pair_region: tuple[Any, ...] = ()
    has_distinct_side_domains = False
    for bottom_id, bottom_region in bottom_regions:
        for top_id, top_region in top_regions:
            overlap = boolean_gdstk_region(
                gdstk,
                bottom_region,
                top_region,
                "and",
            )
            if not overlap:
                continue
            has_distinct_side_domains |= bottom_id != top_id
            refs = geometry_refs_from_gdstk_region(parent_geometry_ref, overlap)
            for geometry_ref in refs:
                raw_patches.append(
                    _RawSheetPatch(interface, sheet, bottom_id, top_id, geometry_ref)
                )
            pair_region = (
                overlap
                if not pair_region
                else boolean_gdstk_region(gdstk, pair_region, overlap, "or")
            )
    uncovered = boolean_gdstk_region(
        gdstk,
        sheet_region,
        pair_region,
        "not",
    )
    if uncovered:
        raise ValueError(
            f"{sheet.semantic_id} local Route A patches do not cover the full sheet."
        )
    if not has_distinct_side_domains:
        raise ValueError(
            f"{sheet.semantic_id} has no positive-area physical support from "
            "distinct side domains."
        )
    return raw_patches


def _numbered_sheet_patches(
    raw_patches: Sequence[_RawSheetPatch],
    semantic_facts: SemanticEvidenceFacade,
    route: RouteLiteral | None,
) -> tuple[RouteASheetPatch, ...]:
    """Order patches deterministically, give them ids, and attach their evidence.

    The sort key (interface id, bottom id, top id, outer loop signature) decides patch
    numbering, so it must not change. A sheet with one patch gets no numeric suffix.
    """
    ordered = sorted(
        raw_patches,
        key=lambda item: (
            item.interface.interface_id,
            item.bottom_id,
            item.top_id,
            loop_signature(item.geometry_ref["outer_loop"]),
        ),
    )
    counts: Counter[str] = Counter(item.interface.interface_id for item in ordered)
    indexes: Counter[str] = Counter()
    records: list[RouteASheetPatch] = []
    for interface, sheet, bottom_id, top_id, geometry_ref in ordered:
        component_index = indexes[interface.interface_id]
        indexes[interface.interface_id] += 1
        patch_suffix = (
            "" if counts[interface.interface_id] == 1 else f":{component_index:04d}"
        )
        patch_prefix = "representation-local" if route is None else "route-a-local"
        patch_id = f"{patch_prefix}:{interface.interface_id}{patch_suffix}"
        bottom = conductor_solution_evidence(
            semantic_facts,
            contribution_id=f"{patch_id}:bottom:{bottom_id}",
            patch_id=f"planned:{patch_id}:bottom",
            conductor_id=sheet.semantic_id,
            solution_id=bottom_id,
            side="bottom",
        )
        top = conductor_solution_evidence(
            semantic_facts,
            contribution_id=f"{patch_id}:top:{top_id}",
            patch_id=f"planned:{patch_id}:top",
            conductor_id=sheet.semantic_id,
            solution_id=top_id,
            side="top",
        )
        records.append(
            RouteASheetPatch(
                parent_interface_id=interface.interface_id,
                sheet_entity_id=sheet.semantic_id,
                patch_id=patch_id,
                geometry_ref=geometry_ref,
                bottom=bottom,
                top=top,
            )
        )
    return tuple(records)


def _route_a_sheet_side_solution_regions(
    build_input: GeometryBuildInput,
    *,
    sheet: SemanticEntitySpec,
    sheet_region: tuple[Any, ...],
    plane_z_um: float,
    side: str,
) -> tuple[tuple[str, tuple[Any, ...]], ...]:
    """Return nonoverlapping exact solution coverage for one sheet side."""
    return planar_side_solution_regions(
        build_input,
        owner_id=sheet.semantic_id,
        occupied_region=sheet_region,
        plane_z_um=plane_z_um,
        side=side,
    )
