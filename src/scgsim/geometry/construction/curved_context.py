"""Named state shared between the stages of a curved construction build.

Each record is produced by one stage of `build_curved_construction_plan` and read
by the later ones, so a stage signature says which earlier results it depends on.
Owns the shapes only; it creates no native entities.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BuildSession:
    """The live gmsh session and the validated build input one build runs in."""

    gmsh: Any
    occ: Any
    kernel: Any
    route: str
    prepared: Any
    # Entities keyed by semantic id.
    entities: dict


@dataclass(frozen=True)
class PlanarArrangement:
    """Planar cells produced by fragmenting every source footprint together."""

    # Native 2D cell tags.
    cells: set
    # Cell tag -> semantic ids (or "PORT::<id>") of the footprints covering it.
    membership: dict
    # Cell tag -> ordered native edge loops (signed tags), outer loop first.
    cell_loops: dict
    # Native edge tag -> cells on either side of it.
    edge_cells: dict
    # Edges kept after scratch source entities are removed.
    retained_edges: set
    # Edge tag -> native type, source-curve history and parameter bounds.
    curve_descriptors: dict
    # Cell tag -> stable content hash.
    cell_ids: dict
    # Native source-edge tag -> history entries, kept for edge provenance.
    native_edges: dict
    source_footprints: list


@dataclass(frozen=True)
class SlabOwnership:
    """Which entity owns each (slab, cell) and how cells join into volumes."""

    # Semantic ids whose footprint is a single sheet (Route A face metal).
    sheet_ids: set
    # Semantic id -> (z_min, z_max) in um.
    ranges: dict
    # Port sheet id -> Z of the port plane in um.
    port_planes: dict
    # Sorted distinct Z levels in um; slab i spans z_values[i] to z_values[i+1].
    z_values: list
    # (slab, cell) -> sorted conductor semantic ids filling it.
    occupied: dict
    # (slab, cell) -> semantic id of the solution region owning it.
    owners: dict
    # Union-find root -> member (slab, cell) keys.
    components: dict
    # Union-find root -> volume plan id.
    volume_ids: dict
    # (slab, cell) -> volume plan id.
    volume_for: dict


@dataclass(frozen=True)
class SweptComplex:
    """Native edges and walls produced by sweeping the curve complex along Z."""

    # (level, source edge) -> edge copy at that Z level.
    lifted: dict
    # (slab, source vertex) -> shared vertical edge.
    vertical: dict
    # (slab, source edge) -> swept wall face.
    swept_walls: dict


@dataclass(frozen=True)
class CurvedContext:
    """Everything a surface or record stage needs from the earlier stages."""

    session: BuildSession
    arrangement: PlanarArrangement
    ownership: SlabOwnership
    sweep: SweptComplex


@dataclass
class SurfaceBook:
    """Native surfaces and contacts registered while walls and caps are created."""

    # Surface id -> native face tag.
    native_surfaces: dict = field(default_factory=dict)
    # Surface id -> owner, interface id, metadata, normal, role and face kind.
    surface_data: dict = field(default_factory=dict)
    # Volume plan id -> surface references on its boundary.
    volume_refs: dict = field(default_factory=lambda: defaultdict(list))
    native_contacts: list = field(default_factory=list)
    contact_keys: set = field(default_factory=set)


@dataclass(frozen=True)
class EdgeLookup:
    """Maps a native edge back to the planar curve or vertical edge it came from."""

    lifted_inverse: dict
    vertical_inverse: dict
    curve_descriptors: dict
    native_edges: dict
    z_values: list
