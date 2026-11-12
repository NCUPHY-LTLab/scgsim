"""The bound rings of chosen conductors, as SGB builds them (stage S2).

Owns one read of SGB's boundary binding for a caller outside SGB: the junction partition
reads where a lumped port's terminal is built from exact lines and circular arcs.  It runs
the footprints and the binder of ``catalogue`` on the chosen Entities and their boundary
records alone, so a record is bound, and refused, by the one rule and wording the full
build uses.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

from scgsim.geometry.models.input import GeometryBuildInput
from scgsim.geometry.models.representation import RepresentationContext
from scgsim.geometry.sgb.prisms.catalogue import declared_grid
from scgsim.geometry.sgb.prisms.curves import RingKey, bind_boundaries
from scgsim.geometry.sgb.prisms.footprints import source_footprints
from scgsim.geometry.sgb.records.grid import Grid
from scgsim.geometry.sgb.records.prisms import RingPrimitive


def bound_ring_primitives(
    build_input: GeometryBuildInput, entity_ids: Iterable[str]
) -> tuple[Grid, dict[RingKey, tuple[RingPrimitive, ...]]]:
    """The declared grid, and the ring primitives of every ring a record of these conductors binds.

    The input is narrowed to the chosen Entities and the records that name them; a ring no
    record binds is absent, as in ``bind_boundaries``.  A key is ``(Entity, footprint index,
    ring index)``: the footprint index counts the Entity's ``polygon_ids``, the ring index is
    0 for the outer ring and ``1 + k`` for hole ``k``.  The Entities are conductors, whose
    footprints read no prepared enclosure.  An input that declares no grid is refused with
    ``InputError``, as the full build refuses it.
    """
    selected = set(entity_ids)
    narrowed = replace(
        build_input,
        entities=tuple(e for e in build_input.entities if e.semantic_id in selected),
        boundary_curves=tuple(
            r for r in build_input.boundary_curves if r.entity_id in selected
        ),
        boundary_reconstruction=tuple(
            r for r in build_input.boundary_reconstruction if r.entity_id in selected
        ),
    )
    grid = declared_grid(narrowed)
    no_enclosures = RepresentationContext(entities={})
    footprints = {
        entity.semantic_id: source_footprints(entity, narrowed, grid, no_enclosures)
        for entity in narrowed.entities
    }
    return grid, bind_boundaries(narrowed, grid, footprints)
