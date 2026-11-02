"""Compose exact source incidence, model emission and final junction checks."""

from __future__ import annotations

from scgsim.geometry.planning.junction_planning.core import DEFAULT_FORMAT
from scgsim.geometry.planning.junction_planning.emission import (
    check_emitted_partitions,
    emit_partitions,
    initialize_registry,
    rebuild_entity_support,
    rebuild_original_polygons,
)
from scgsim.geometry.planning.junction_planning.incidence import (
    initialize_source,
    register_crossings,
    register_owned_seams,
    register_source_edges,
    slice_free_boundaries,
    trace_free_boundaries,
)
from scgsim.geometry.planning.junction_planning.model_state import JunctionModel


def model_geometry(build_input, originals, records, output_format=DEFAULT_FORMAT):
    """Trace source incidences and owned seams before topology, then slice at two cuts."""
    state = JunctionModel(build_input, originals, records, output_format)
    initialize_source(state)
    register_source_edges(state)
    register_crossings(state)
    register_owned_seams(state)
    trace_free_boundaries(state)
    slice_free_boundaries(state)
    initialize_registry(state)
    rebuild_original_polygons(state)
    emit_partitions(state)
    rebuild_entity_support(state)
    check_emitted_partitions(state)
    return (state.model, tuple(state.entities))
