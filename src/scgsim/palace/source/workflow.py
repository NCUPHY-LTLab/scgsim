"""The preparation of a plan's geometry input under the simulation's geometry settings.

The simulation holds the settings and the lifecycle; this module takes them as ``GeometrySettings`` and
holds no state.  It applies them in a fixed order: vacuum, port sheets, solution regions, authored ports,
the physical interpretation of the route.  The route rule is here too: ``set_route`` and ``simulation_drc``
both state a route setting, and ``checked_route_setting`` checks it for either, naming the call it was given in.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

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
from scgsim.refusal import Declared, found_text, internal
from scgsim.semantics.snapshot import canonical_sha256, to_plain

from .ports import with_rectangular_ports
from .surface_epr import SurfaceEprSpecs

_WHERE_VACUUM = "the GeometryPlan, or the call to set_vacuum_region"


@dataclass(frozen=True)
class RouteCall:
    """The call that states a route setting, which a refusal of that setting names.

    ``name`` is the call (``set_route`` or ``simulation_drc``); ``route_argument`` spells how it takes the
    route, with ``{route}`` standing for the route as Python writes it.
    """

    name: str
    route_argument: str

    @property
    def where(self) -> str:
        """Where a refused route setting is changed: this call."""
        return f"the call to {self.name}"

    def with_route(self, route: str) -> str:
        """This call, spelled with ``route``."""
        return f"{self.name}({self.route_argument.format(route=repr(route))})"


SET_ROUTE = RouteCall("set_route", "{route}")
SIMULATION_DRC = RouteCall("simulation_drc", "..., route={route}")


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
    prepared_stack = _stack_with_vacuum_and_ports(
        stack, settings.vacuum_region, port_sheet_source_layers
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
            Declared("the model", _WHERE_VACUUM).refusal(
                "has no vacuum region: the plan declares none and none was set on the simulation; call plan.set_vacuum_region(...) before prepare() or sim.set_vacuum_region(...) before mesh()"
            )
        )
    if route == "A":
        prepared_stack["metadata"]["route_a_thin_film"] = _route_a_thin_film(
            stack, approximation_profile
        )
    materials = _stack_material_records(prepared_stack)
    return PreparedMeshInput(
        stack=prepared_stack,
        materials=materials,
        geometry_input=effective_input,
        physical_plan=plan,
    )


def _stack_with_vacuum_and_ports(
    stack: Mapping[str, Any],
    vacuum_region: VacuumRegionSpec | None,
    port_sheet_source_layers: Sequence[Mapping[str, Any]] | None,
) -> Mapping[str, Any]:
    """The stack with the simulation's vacuum region applied and its port sheets recorded in the metadata."""
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
    return prepared_stack


def _route_a_thin_film(
    stack: Mapping[str, Any], approximation_profile: ApproximationProfile
) -> dict[str, Any]:
    """The Route A thin-film identity: the approximation's variant and the hash of the authored stack."""
    source_hash = canonical_sha256(to_plain(stack))
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


def _stack_material_records(
    prepared_stack: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]] | None:
    """The prepared stack's material records by id, None when the stack has no materials mapping."""
    prepared_materials = prepared_stack.get("materials")
    if not isinstance(prepared_materials, Mapping):
        return None
    return {
        str(material_id): dict(material)
        for material_id, material in prepared_materials.items()
        if isinstance(material, Mapping)
    }


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
                    internal(
                        f"normalized source has no authored port {record['name']!r}"
                    )
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


def checked_route_setting(
    route: Any, approximation_profile: Any, call: RouteCall
) -> tuple[Literal["A", "B"], ApproximationProfile | None]:
    """The route setting ``call`` states, checked: ``route`` as ``"A"`` or ``"B"``, and the profile as given.

    Whether a profile suits the route depends on the plan's films, so SGB's route check refuses a
    mismatch when the geometry is built; this check refuses only what no plan can use.

    Raises:
        ValueError: ``route`` is not ``"A"`` or ``"B"``.
        TypeError: ``approximation_profile`` is neither None nor an ``ApproximationProfile``.
    """
    chosen = _checked_route(route, call)
    if approximation_profile is not None and not isinstance(
        approximation_profile, ApproximationProfile
    ):
        raise TypeError(
            Declared(
                "approximation_profile must be an ApproximationProfile: pass "
                f"{call.name}(..., approximation_profile=ApproximationProfile(name, 'v1', parameters))",
                call.where,
            ).refusal(found_text(type(approximation_profile).__name__))
        )
    return chosen, approximation_profile


def _checked_route(route: Any, call: RouteCall) -> Literal["A", "B"]:
    """``route`` as ``"A"`` or ``"B"``; any other route is refused with the two a Palace simulation takes."""
    chosen = route.strip().upper() if isinstance(route, str) else route
    if chosen == "A" or chosen == "B":
        return chosen
    if chosen == "C":
        raise ValueError(
            Declared("Route C", call.where).refusal(
                "is not a Palace route: SGB builds Route C for backends that keep conductor volumes. "
                f"Call {call.with_route('A')} or {call.with_route('B')}"
            )
        )
    raise ValueError(
        Declared(f"Unknown route {route!r}", call.where).refusal(
            f"call {call.with_route('A')} for films as zero-thickness sheets or "
            f"{call.with_route('B')} for 3D conductors"
        )
    )


def route_profile(
    route: str, approximation_profile: ApproximationProfile | None
) -> ApproximationProfile:
    """The approximation profile the preparation interprets the plan with.

    The given profile is used as given.  Without one, each route has its default: Route A lowers the films
    with ``thin_film`` and Route B keeps the conductors finite with the identity profile ``route_b``.
    The preparation and ``input_summary()`` both call this for the profile they use, so the two agree.
    """
    if approximation_profile is None:
        name = "thin_film" if route == "A" else "route_b"
        return ApproximationProfile(name, "v1", {})
    return approximation_profile


__all__ = [
    "SET_ROUTE",
    "SIMULATION_DRC",
    "GeometrySettings",
    "PreparedMeshInput",
    "RouteCall",
    "checked_route_setting",
    "prepare_mesh_input",
    "route_profile",
]
