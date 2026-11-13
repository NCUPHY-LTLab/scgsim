"""The preparation of a plan's geometry input under the simulation's geometry settings.

The simulation holds the settings and the lifecycle; this module takes them as ``GeometrySettings`` and
holds no state.  It applies them in a fixed order: vacuum, port sheets, solution regions, authored ports,
the physical interpretation of the route.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from scgsim.geometry.interpretation.physical import interpret_physical_geometry
from scgsim.geometry.models.input import GeometryBuildInput, VacuumRegionSpec
from scgsim.geometry.models.physical import (
    ApproximationProfile,
    PhysicalRepresentationPlan,
)
from scgsim.geometry.source.adapter_records import solution_region_entity
from scgsim.geometry.source.port_sheets import with_port_sheet_metadata
from scgsim.geometry.source.vacuum import apply_vacuum_region_to_stack
from scgsim.geometry.source.validation import has_vacuum_solution_region
from scgsim.semantics.snapshot import canonical_sha256, to_plain

from .ports import with_rectangular_ports
from .surface_epr import SurfaceEprSpecs


@dataclass(frozen=True)
class GeometrySettings:
    """The simulation settings the geometry preparation and build read, as the simulation holds them.

    ``route`` is how the conductors are modelled (``"A"`` thin-film sheets, ``"B"`` 3D conductors), with
    the ``approximation_profile`` given to ``set_route``, None for the route's default profile;
    ``vacuum_region`` is the simulation's own vacuum, None when the plan has one or none was set.
    """

    route: str
    approximation_profile: ApproximationProfile | None
    vacuum_region: VacuumRegionSpec | None
    surface_epr_specs: SurfaceEprSpecs | None


@dataclass(frozen=True)
class PreparedMeshInput:
    """Values prepared before a simulation builds a mesh."""

    stack: Mapping[str, Any]
    materials: dict[str, Mapping[str, Any]] | None
    geometry_input: GeometryBuildInput
    physical_plan: PhysicalRepresentationPlan


def prepare_mesh_input(
    *,
    stack: Mapping[str, Any],
    build_input: GeometryBuildInput,
    settings: GeometrySettings,
    port_sheet_source_layers: Sequence[Mapping[str, Any]] | None = None,
) -> PreparedMeshInput:
    """Prepare a stack and geometry input for the build: vacuum, ports and the route's interpretation, in that order.

    ``port_sheet_source_layers`` lists the authored sheets cut as lumped ports, None for a problem with none.
    """
    route = settings.route
    vacuum_region = settings.vacuum_region
    prepared_stack = stack
    if vacuum_region is not None:
        prepared_stack = apply_vacuum_region_to_stack(stack, vacuum_region)
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
    # Synchronize source selections before the single physical interpretation.
    build_input = _synchronize_solution_regions(build_input, prepared_stack)
    build_input = _select_authored_ports(build_input, port_sheet_source_layers)
    if port_sheet_source_layers is not None:
        # Every selected sheet is a Lumped Port or RLC; the physical interpretation cuts its rectangle.
        build_input = with_rectangular_ports(build_input)
    approximation_profile = route_profile(route, settings.approximation_profile)
    effective_input, effective_stack, plan = interpret_physical_geometry(
        build_input, prepared_stack, approximation_profile
    )
    prepared_stack = effective_stack
    if not has_vacuum_solution_region(effective_input):
        raise ValueError(
            "the model has no vacuum region: the plan declares none and none was set on the simulation; "
            "call plan.set_vacuum_region(...) before prepare() or sim.set_vacuum_region(...) before mesh()"
        )
    if route == "A":
        source_hash = canonical_sha256(to_plain(stack))
        prepared_stack["metadata"]["route_a_thin_film"] = {
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
    prepared_materials = prepared_stack.get("materials")
    materials: dict[str, Mapping[str, Any]] | None = None
    if isinstance(prepared_materials, Mapping):
        materials = {
            str(material_id): dict(material)
            for material_id, material in prepared_materials.items()
            if isinstance(material, Mapping)
        }
    return PreparedMeshInput(
        stack=prepared_stack,
        materials=materials,
        geometry_input=effective_input,
        physical_plan=plan,
    )


def _synchronize_solution_regions(
    build_input: GeometryBuildInput, prepared_stack: Mapping[str, Any]
) -> GeometryBuildInput:
    """Replace changed solution regions without disturbing retained entity order."""
    if prepared_stack["solution_regions"] != build_input.solution_regions:
        regions = tuple(
            solution_region_entity(
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
        build_input = replace(
            build_input,
            entities=(*retained, *by_id.values()),
            solution_regions=prepared_stack["solution_regions"],
        )
    return build_input


def _select_authored_ports(
    build_input: GeometryBuildInput,
    port_sheet_source_layers: Sequence[Mapping[str, Any]] | None,
) -> GeometryBuildInput:
    """Keep exact source identities while assigning request-order port indices."""
    if port_sheet_source_layers is not None:
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
        build_input = replace(
            build_input,
            port_sheet_regions=tuple(selected),
            metadata={
                **build_input.metadata,
                "port_sheet_source_layers": list(port_sheet_source_layers),
            },
        )
    return build_input


def route_profile(
    route: str, approximation_profile: ApproximationProfile | None
) -> ApproximationProfile:
    """The approximation profile the preparation interprets the plan with.

    The given profile is used as given.  Without one, each route has its default: Route A lowers the films
    with ``thin_film`` and Route B keeps the conductors finite with the identity profile ``route_b``.
    ``set_route`` calls this when the setting is made, and the preparation and ``input_summary()`` again for
    the profile they use, so the three agree.

    Raises:
        TypeError: ``approximation_profile`` is neither None nor an ``ApproximationProfile``.
    """
    if approximation_profile is None:
        name = "thin_film" if route == "A" else "route_b"
        return ApproximationProfile(name, "v1", {})
    if not isinstance(approximation_profile, ApproximationProfile):
        raise TypeError(
            "approximation_profile must be an ApproximationProfile: pass "
            "set_route(..., approximation_profile=ApproximationProfile(name, 'v1', parameters))."
        )
    return approximation_profile


__all__ = [
    "GeometrySettings",
    "PreparedMeshInput",
    "prepare_mesh_input",
    "route_profile",
]
