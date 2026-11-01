"""Shared value preparation and ordered persistence for Palace workflows.

Simulation classes retain lifecycle and mutable-state ownership.  This module
only composes the existing preparation transforms and writes caller-provided
payloads in their declared order.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from scgsim.geometry.models.input import GeometryBuildInput, VacuumRegionSpec
from scgsim.geometry.models.physical import (
    ApproximationProfile,
    PhysicalRepresentationPlan,
)
from scgsim.geometry.planning.physical import interpret_physical_geometry
from scgsim.geometry.source.adapter import solution_region_entity_from_record
from scgsim.geometry.source.ground_bumps import prepare_indium_ground_bump_fill
from scgsim.geometry.source.vacuum import apply_vacuum_region_to_stack
from scgsim.semantics.snapshot import canonical_sha256, plain_semantic_facts

from ..mesh.build import with_port_sheet_metadata
from .transforms import (
    RouteAThinFilm,
    apply_airbox_to_stack,
    normalized_component_input,
)


@dataclass(frozen=True)
class PreparedMeshInput:
    """Values prepared before a simulation applies its airbox and builds a mesh."""

    component: Any
    stack: Mapping[str, Any]
    indium_ground_bump_fill: Mapping[str, Any] | None
    materials: dict[str, Mapping[str, Any]] | None
    geometry_input: GeometryBuildInput
    physical_plan: PhysicalRepresentationPlan


def prepare_mesh_input(  # noqa: PLR0913 - public keyword-only API, one argument per input
    *,
    component: Any,
    stack: Mapping[str, Any],
    route: str,
    route_a_thin_film: RouteAThinFilm | None,
    vacuum_region: VacuumRegionSpec | None,
    indium_ground_bumps: Mapping[str, Any] | None,
    build_input: GeometryBuildInput | None = None,
    approximation_profile: ApproximationProfile | None = None,
    airbox: Mapping[str, Any] | None = None,
    port_sheet_source_layers: Sequence[Mapping[str, Any]] | None = None,
) -> PreparedMeshInput:
    """Apply the common transforms while preserving their established order."""
    prepared_stack, prepared_component, indium_fill = _transform_stack(
        component=component,
        stack=stack,
        vacuum_region=vacuum_region,
        indium_ground_bumps=indium_ground_bumps,
        airbox=airbox,
        port_sheet_source_layers=port_sheet_source_layers,
    )
    if build_input is None:
        build_input = normalized_component_input(prepared_component, prepared_stack)
    else:
        # Synchronize caller source selections before the one physical interpretation.
        build_input = _sync_solution_regions(build_input, prepared_stack)
        if port_sheet_source_layers is not None:
            build_input = _select_port_sheet_regions(
                build_input, port_sheet_source_layers
            )
    if indium_fill is not None:
        build_input = replace(
            build_input,
            polygons=(*build_input.polygons, *indium_fill.get("injected_polygons", ())),
            entities=(*build_input.entities, *indium_fill.get("injected_entities", ())),
        )
    approximation_profile = _checked_approximation_profile(
        approximation_profile, route, route_a_thin_film
    )
    effective_input, effective_stack, plan = interpret_physical_geometry(
        build_input, prepared_stack, approximation_profile
    )
    prepared_stack = effective_stack
    if route == "A":
        prepared_stack["metadata"]["route_a_thin_film"] = _route_a_metadata(
            stack, approximation_profile
        )
    return PreparedMeshInput(
        component=prepared_component,
        stack=prepared_stack,
        indium_ground_bump_fill=indium_fill,
        materials=_material_records(prepared_stack),
        geometry_input=effective_input,
        physical_plan=plan,
    )


def _transform_stack(  # noqa: PLR0913 - one argument per transform, passed by keyword
    *,
    component: Any,
    stack: Mapping[str, Any],
    vacuum_region: VacuumRegionSpec | None,
    indium_ground_bumps: Mapping[str, Any] | None,
    airbox: Mapping[str, Any] | None,
    port_sheet_source_layers: Sequence[Mapping[str, Any]] | None,
) -> tuple[Mapping[str, Any], Any, Mapping[str, Any] | None]:
    """Vacuum region, indium bump fill, airbox, then port sheets, in that order.

    Returns the stack, the component (the bump fill replaces it) and the fill.
    """
    prepared_stack = stack
    if vacuum_region is not None:
        prepared_stack = apply_vacuum_region_to_stack(stack, vacuum_region)
    indium_fill = None
    prepared_component = component
    if indium_ground_bumps is not None:
        indium_fill = prepare_indium_ground_bump_fill(
            component=component,
            stack=prepared_stack,
            **indium_ground_bumps,
        )
        prepared_component = indium_fill["component"]
        prepared_stack = indium_fill["stack"]
    prepared_stack = apply_airbox_to_stack(prepared_stack, airbox or {})
    if port_sheet_source_layers is not None:
        prepared_stack = {
            **prepared_stack,
            "metadata": {
                **prepared_stack.get("metadata", {}),
                "port_sheet_source_layers": [],
            },
        }
        prepared_stack = with_port_sheet_metadata(
            prepared_stack, port_sheet_source_layers
        )
    return prepared_stack, prepared_component, indium_fill


def _sync_solution_regions(
    build_input: GeometryBuildInput, prepared_stack: Mapping[str, Any]
) -> GeometryBuildInput:
    """Make the caller's source entities agree with the stack's solution regions."""
    if prepared_stack["solution_regions"] == build_input.solution_regions:
        return build_input
    regions = tuple(
        solution_region_entity_from_record(
            key,
            record,
            materials=prepared_stack["materials"],
            cell_bounds_um=build_input.metadata["cell_bounds_um"],
        )
        for key, record in prepared_stack["solution_regions"].items()
    )
    by_id = {entity.semantic_id: entity for entity in regions}
    old_ids = set(build_input.solution_regions)
    retained = tuple(
        by_id.pop(entity.semantic_id) if entity.semantic_id in by_id else entity
        for entity in build_input.entities
        if entity.semantic_id not in old_ids or entity.semantic_id in by_id
    )
    return replace(
        build_input,
        entities=(*retained, *by_id.values()),
        solution_regions=prepared_stack["solution_regions"],
    )


def _select_port_sheet_regions(
    build_input: GeometryBuildInput,
    port_sheet_source_layers: Sequence[Mapping[str, Any]],
) -> GeometryBuildInput:
    """Keep only the authored port regions the caller named, numbered in that order."""
    selected = []
    for source_index, record in enumerate(port_sheet_source_layers):
        matches = tuple(
            region
            for region in build_input.port_sheet_regions
            if region.metadata.get("source_name") == record["name"]
        )
        if not matches:
            raise ValueError(
                f"normalized source has no authored port {record['name']!r}"
            )
        selected.extend(
            replace(
                region,
                metadata={
                    **region.metadata,
                    "source_index": source_index,
                    "port_index": record["port_index"],
                },
            )
            for region in matches
        )
    return replace(
        build_input,
        port_sheet_regions=tuple(selected),
        metadata={
            **build_input.metadata,
            "port_sheet_source_layers": list(port_sheet_source_layers),
        },
    )


def _checked_approximation_profile(
    approximation_profile: ApproximationProfile | None,
    route: str,
    route_a_thin_film: RouteAThinFilm | None,
) -> ApproximationProfile:
    """Route A must say how its thin film is lowered; Route B needs no such choice."""
    if approximation_profile is None:
        if route == "A":
            raise ValueError(
                "fresh Palace Route A requires approximation_profile=ApproximationProfile(name, 'v1', {'film_bindings': explicit_bindings}); route_a_thin_film does not author supporting Entities"
            )
        approximation_profile = ApproximationProfile("route_b", "v1", {})
    if not isinstance(approximation_profile, ApproximationProfile):
        raise TypeError("approximation_profile must be ApproximationProfile")
    if (
        route_a_thin_film is not None
        and approximation_profile.name != route_a_thin_film
    ):
        raise ValueError(
            "route_a_thin_film selector differs from the explicit approximation profile"
        )
    return approximation_profile


def _route_a_metadata(
    stack: Mapping[str, Any], approximation_profile: ApproximationProfile
) -> dict[str, Any]:
    """The thin-film identity recorded for Route A, bound to the source stack's hash."""
    source_hash = canonical_sha256(plain_semantic_facts(stack))
    return {
        "schema_version": 1,
        "variant": approximation_profile.name,
        "display_label": "A_PRIME"
        if approximation_profile.name == "substrate_face"
        else "A",
        "source_stack": {
            "revision": f"sha256:{source_hash}",
            "sha256": source_hash,
        },
    }


def _material_records(stack: Mapping[str, Any]) -> dict[str, Mapping[str, Any]] | None:
    materials = stack.get("materials")
    if not isinstance(materials, Mapping):
        return None
    return {
        str(material_id): dict(material)
        for material_id, material in materials.items()
        if isinstance(material, Mapping)
    }


def persist_problem_files(
    *,
    metadata_files: Sequence[tuple[Path, Mapping[str, Any]]],
    config_path: Path,
    config: Mapping[str, Any],
) -> Path:
    """Replace ordered metadata files independently, then replace config last."""
    for path, payload in metadata_files:
        _write_json_atomic_creating_parent(path, payload)
    _write_json_atomic_creating_parent(config_path, config)
    return config_path


def _write_json_atomic_creating_parent(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


__all__ = ["PreparedMeshInput", "persist_problem_files", "prepare_mesh_input"]
