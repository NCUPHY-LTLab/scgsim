"""Stage S2: the effective input as a prism catalogue.

Owns the entry point from the stable S0-S1 outputs -- the effective
``GeometryBuildInput``, the ``RepresentationContext`` and the grid declared in the
input's metadata -- to the prisms, carriers and Z structure the arrangement stages
consume.  The catalogue is route-independent: Route A, B and C are labelling
policies applied after the arrangement exists.

Design decisions (each with the compiler code it follows) are in the modules of this
package: ``footprints`` (where an entity's rings come from), ``entities`` (Z interval,
Net, priority), ``curves`` (authored boundaries and reconstruction), ``ports``
(lumped-port sheets), ``arcs`` (circles on the grid).
"""

from __future__ import annotations

from scgsim.geometry.models.grid import geometry_grid_context
from scgsim.geometry.models.input import GeometryBuildInput
from scgsim.geometry.models.representation import RepresentationContext
from scgsim.geometry.sgb.errors import InputError
from scgsim.geometry.sgb.prisms.curves import bind_boundaries
from scgsim.geometry.sgb.prisms.entities import (
    check_entity_kind,
    entity_interval,
    entity_prisms,
    physical_feature_kind,
)
from scgsim.geometry.sgb.prisms.footprints import SourceFootprint, source_footprints
from scgsim.geometry.sgb.prisms.ports import port_prisms
from scgsim.geometry.sgb.records.grid import Grid
from scgsim.geometry.sgb.records.prisms import PrismCatalogue, PrismRecord


def build_prism_catalogue(
    build_input: GeometryBuildInput, representation: RepresentationContext
) -> PrismCatalogue:
    """Turn an effective input into prisms on its declared grid (fail closed on any defect)."""
    grid = declared_grid(build_input)
    entities = build_input.entities
    for entity in entities:
        check_entity_kind(entity)
    footprints: dict[str, list[SourceFootprint]] = {
        entity.semantic_id: source_footprints(entity, build_input, grid, representation)
        for entity in entities
    }
    ring_primitives = bind_boundaries(build_input, grid, footprints)
    prisms: list[PrismRecord] = []
    for entity in entities:
        interval = entity_interval(entity, representation)
        prisms.extend(
            entity_prisms(
                entity,
                footprints[entity.semantic_id],
                interval,
                ring_primitives,
                physical_feature_kind(entity, build_input, representation),
            )
        )
    prisms.extend(port_prisms(build_input, grid, representation))
    if not prisms:
        raise InputError("the effective input has no entities")
    return PrismCatalogue(grid, tuple(prisms))


def declared_grid(build_input: GeometryBuildInput) -> Grid:
    """The model grid the input declares; an input without one is rejected, never defaulted."""
    context = geometry_grid_context(build_input)
    if context is None:
        raise InputError(
            "the input declares no geometry grid (metadata 'geometry_grid'); build it with "
            "GeometryPlan or build_gds_stack_geometry_input so XY coordinates sit on a declared grid"
        )
    return Grid(context.grid_size_um)
