"""The definition of the SGB sidecar ``scgsim.sgb.sidecar.v2``: every section and record, as a ``TypedDict``.

Owns the shape of the file: its sections, the fields of each record, their types, enums and bounds, and
which records accept more keys.  It is the one definition the JSON Schema is derived from (:mod:`.schema`)
and the shared reader checks against (:mod:`.read`).  The writer is ``scgsim.geometry.sgb.sidecar``; what
each section means for SGB and for its readers is documented there.

Every record is closed (a key it does not declare makes the sidecar invalid) except the two marked
``@open_record``: ``counts`` and ``slivers``.  A record's docstring is its description in the schema, and
``Doc`` describes a field.  The module uses no ``Required``: Python does not see it in a postponed annotation,
so a record with optional fields declares its required ones in a base and the rest in a ``total=False``
subclass (``Carrier``).

Terms.  A *planned id* is SGB's id of a vertex, edge, face or volume; its *native tag* is the Gmsh tag of that
shape in the written XAO, 1 + the id's position in ``binding.ids_by_tag``.  Interface types: ``MA``
metal-air, ``MS`` metal-substrate, ``MM`` metal-metal, ``SA`` substrate-air, ``AA`` air-air, ``SS``
substrate-substrate, ``SHEET`` a Route A conductor sheet, ``PORT`` a lumped-port sheet, ``EXT`` the model
boundary.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from .markers import (
    Doc,
    ExclusiveMinimum,
    Length,
    Minimum,
    MinProperties,
    Pattern,
    open_record,
)
from .version import SidecarVersion

# --------------------------------------------------------------------------- values

IdList = list[str]  # planned ids, or group or Entity names
PositiveInt = Annotated[int, Minimum(1)]
Tag = PositiveInt  # a native tag (or a physical-group tag): never 0
TagList = list[Tag]
Count = Annotated[int, Minimum(0)]
Positive = Annotated[float, ExclusiveMinimum(0)]
NonNegative = Annotated[float, Minimum(0)]
PlanePoint = Annotated[list[float], Length(2, 2)]  # (x, y) um
SpacePoint = Annotated[list[float], Length(3, 3)]  # (x, y, z) um
ZRange = Annotated[list[float], Length(2, 2)]  # (lowest, highest) z um
GroupLists = dict[str, IdList]  # a key -> primary group names

Route = Literal["A", "B", "C"]
VolumeKind = Literal["vacuum", "dielectric", "conductor"]
SolutionKind = Literal["vacuum", "dielectric"]
FaceKind = Literal["top", "bottom", "sidewall", "sheet", "port", "interface"]
InterfaceType = Literal["MA", "MS", "MM", "SA", "AA", "SS", "SHEET", "PORT", "EXT"]
SurfaceEprType = Literal["MA", "MS", "SA"]
SheetSide = Literal["above", "below", "both"]
RowSide = Literal["above", "below", "both", "mixed"]
SidecarFile = Literal["xao", "brep", "sidecar", "sliver_report", "validation_report"]
CarrierKind = Literal["polygon_edge", "family_member", "unknown"]
BandedFaceKind = Literal["sheet", "top", "bottom", "interface"]


# --------------------------------------------------------------------------- the document


class Sidecar(TypedDict):
    """Sidecar of the SGB XAO/BREP: planned ids bound to native tags by construction order, primary physical
    groups with their semantic facts, unions, embeddings, lineage and the load-side expectations."""

    schema: SidecarVersion
    route: Route
    route_policy: RoutePolicy
    grid_um: Positive
    tolerance_um: Positive
    model: ModelIdentity
    files: Annotated[dict[SidecarFile, str], Doc("File names relative to the sidecar.")]
    binding: Binding
    counts: Counts
    physical_groups: Annotated[
        dict[str, SurfaceGroup | VolumeGroup],
        Doc("Primary physical groups by name; they partition each dimension."),
    ]
    unions: Unions
    surface_epr_rows: dict[str, SurfaceEprRow]
    volumes: dict[str, Volume]
    faces: dict[str, Face]
    embedded_faces: Annotated[
        list[EmbeddedFace],
        Doc(
            "Sheets inside an emitted volume; the mesher embeds each face in its host volume."
        ),
    ]
    refinement_edges: Annotated[
        list[RefinementEdge],
        Doc("Refinement-only curves; the mesher embeds each edge in its host face."),
    ]
    mesh_features: Annotated[
        list[MeshFeature],
        Doc(
            "Facts about the geometry for the mesher (no sizes): film sidewalls, band boundaries, arcs, refinement"
            " curves, port edges. Sorted by kind, source and canonical JSON of the facts."
        ),
    ]
    ports: list[Port]
    lineage: Lineage
    insets: Annotated[
        list[InsetFace | InsetType],
        Doc(
            "The inset requests as given, by request id.  A concrete request is the record of its face; a"
            " type-level request lists the faces it was expanded into and the faces it skipped."
        ),
    ]
    refinements: list[Refinement]
    slivers: SliverSummary
    arrangement: Arrangement
    curved_wall_faces: IdList
    surface_kinds: SurfaceKinds
    expected: Expected


# --------------------------------------------------------------------------- provenance and binding


class RoutePolicy(TypedDict):
    """Whether the route excludes or retains conductor volumes, and whether it allows conductor sheets."""

    conductor_volumes: Literal["excluded", "retained"]
    conductor_sheets: bool


class ModelIdentity(TypedDict):
    """The model's name and identity digest."""

    name: str
    identity_digest: Annotated[str, Pattern("^model:[0-9a-f]{16}$")]


class Binding(TypedDict):
    """Planned id <-> native tag per dimension, and the rule that relates them."""

    rule: str
    ids_by_tag: Annotated[
        IdsByTag,
        Doc("Planned ids per dimension in native-tag order: tag = 1 + position."),
    ]


IdsByTag = TypedDict("IdsByTag", {"0": IdList, "1": IdList, "2": IdList, "3": IdList})


@open_record(Count)
class Counts(TypedDict):
    """Shapes of the written file by kind, and the number of planned volumes (volumes_total)."""

    vertices: Count
    edges: Count
    faces: Count
    solids: Count
    wires: Count
    shells: Count
    surfaces: Count
    cylinders: Count
    arcs: Count
    pcurves: Count
    free_faces: Count
    free_edges: Count
    volumes_total: Count


# --------------------------------------------------------------------------- groups and Surface EPR


class SurfaceGroup(TypedDict):
    """One primary surface group (dimension 2) and the facts derived from its member faces."""

    dim: Literal[2]
    tag: Tag
    interface_type: Annotated[
        InterfaceType,
        Doc(
            "Derived from the face structure, never parsed from the group name; SHEET is a Route A conductor"
            " sheet, whose group is named by the bare entity id."
        ),
    ]
    ids: IdList
    native_tags: TagList
    entities: IdList
    net: str | None
    face_kind: Annotated[
        FaceKind,
        Doc(
            "The one face kind every face of the group has; it is part of the group name (PORT groups: port)."
        ),
    ]
    adjacent_domains: list[AdjacentDomain]
    adjacent_volumes: IdList
    band: BandLabel | None
    area_um2: NonNegative


class AdjacentDomain(TypedDict):
    """A solution region (vacuum or dielectric) on a side of the group's faces."""

    name: str
    kind: SolutionKind
    material: str


class BandLabel(TypedDict):
    """The inset band a group's faces lie in: the band family and the band index."""

    family: str
    index: Count


class VolumeGroup(TypedDict):
    """One primary volume group (dimension 3), named by the bare entity id."""

    dim: Literal[3]
    tag: Tag
    interface_type: None
    ids: IdList
    native_tags: TagList
    entities: IdList
    kind: VolumeKind
    material: str
    net: str | None


class Unions(TypedDict):
    """Unions of primary groups as lists of group names: by interface type, and by Net (exposed conductor surfaces)."""

    interface_types: GroupLists
    nets: GroupLists


class SurfaceEprRow(TypedDict):
    """One Surface-EPR row and the primary group that carries it."""

    group: str
    interface_type: SurfaceEprType
    entities: IdList
    band: Count | None
    side: Annotated[
        RowSide | None,
        Doc(
            "A sheet row: the side every face in face_sides agrees on, or mixed when they disagree; null for any"
            " other row."
        ),
    ]
    face_sides: Annotated[
        Annotated[dict[str, SheetSide], MinProperties(1)] | None,
        Doc(
            "A sheet row: face id -> the side of that face whose neighbour (vacuum for MA, dielectric for MS) the"
            " row integrates, listing exactly the faces that have such a neighbour; null for any other row."
        ),
    ]
    shared_face_group: Annotated[
        str | None,
        Doc(
            "The sheet group for a sheet row (its MA and MS rows integrate the same two-sided face), else null."
        ),
    ]


# --------------------------------------------------------------------------- planned volumes and faces


class Volume(TypedDict):
    """One planned volume: its domain, whether it is emitted, its group, shells, planned volume and AEDT name."""

    entity: str
    kind: VolumeKind
    material: str
    net: str | None
    emitted: bool
    group: str | None
    shells: PositiveInt
    volume_um3: float
    aedt_name: Annotated[str, Pattern("^[A-Za-z0-9_]+$")] | None


class Face(TypedDict):
    """One planned face: its group, kind, face kind, z range and the volumes on its two sides."""

    group: str | None
    kind: Literal["horizontal", "wall"]
    face_kind: FaceKind
    z_um: ZRange
    negative_volume: str | None
    positive_volume: str | None


class EmbeddedFace(TypedDict):
    """A sheet face inside an emitted volume, and its group."""

    face: str
    host_volume: str
    group: str | None


class RefinementEdge(TypedDict):
    """A refinement-only edge in its host face, its cut family, and whether it is written as a free edge."""

    edge: str
    host_face: str
    family: str | None
    written_as_free_edge: bool


# --------------------------------------------------------------------------- mesh features


class _MeshFeatureEdges(TypedDict):
    """The edges every mesh feature covers, by planned id and by native tag."""

    edges: Annotated[IdList, Length(min_items=1)]
    native_tags: Annotated[TagList, Length(min_items=1)]


class FilmSidewall(_MeshFeatureEdges):
    """Every edge of the sidewall faces of a finite body; thickness_um is the z extent of the sidewall column."""

    kind: Literal["film_sidewall"]
    source: Annotated[str, Doc("The entity id of the body.")]
    facts: FilmSidewallFacts


class FilmSidewallFacts(TypedDict):
    """The facts of a film sidewall."""

    thickness_um: Positive


class BandBoundary(_MeshFeatureEdges):
    """The edges of an inset family; margins_um are the request's margins, ascending, and narrowest_band_um the
    smallest difference between consecutive margins with 0 prepended."""

    kind: Literal["band_boundary"]
    source: Annotated[str, Doc("The inset request id.")]
    facts: BandBoundaryFacts


class BandBoundaryFacts(TypedDict):
    """The facts of a band boundary."""

    margins_um: Annotated[list[Positive], Length(min_items=1)]
    narrowest_band_um: Positive


class ArcFeature(_MeshFeatureEdges):
    """The circular-arc edges of one carrier radius."""

    kind: Literal["arc"]
    source: Annotated[
        str | None,
        Doc("The one entity that owns the arcs; null when no single entity does."),
    ]
    facts: ArcFacts


class ArcFacts(TypedDict):
    """The facts of an arc feature."""

    radius_um: Positive


class RefinementCurve(_MeshFeatureEdges):
    """The refinement-only curves of a refinement request."""

    kind: Literal["refinement_curve"]
    source: Annotated[str, Doc("The refinement request id.")]
    facts: NoFacts


class PortEdge(_MeshFeatureEdges):
    """The boundary edges of the faces of a lumped-port sheet."""

    kind: Literal["port_edge"]
    source: Annotated[str, Doc("The port sheet id.")]
    facts: NoFacts


class NoFacts(TypedDict):
    """The facts of a feature kind that has none: an empty object."""


MeshFeature = FilmSidewall | BandBoundary | ArcFeature | RefinementCurve | PortEdge


# --------------------------------------------------------------------------- ports


class Port(TypedDict):
    """A lumped-port sheet: the authored identity, and the faces and volumes that carry it in the model."""

    port_sheet_id: str
    group: str
    source_name: str
    port_index: int
    direction: list[float]
    host_entity_ids: IdList
    source_layer: str
    target_layer: str
    z_um: float
    faces: IdList
    host_volumes: IdList
    adjacent_volumes: IdList


# --------------------------------------------------------------------------- lineage


class Lineage(TypedDict):
    """Every edge and face back to its source: edge carriers, carrier descriptions, wall columns, face sources
    and the area each Entity covers."""

    edges: dict[str, IdList]
    carriers: dict[str, Carrier]
    vertical_edges: dict[str, IdList]
    faces: dict[str, IdList]
    membership: Annotated[
        dict[str, dict[str, NonNegative]],
        Doc(
            "Face -> {Entity: covered fraction of the face area}; non-exclusive, may sum to more than 1."
        ),
    ]


class _CarrierKind(TypedDict):
    """The one field every carrier has."""

    kind: CarrierKind


class Carrier(_CarrierKind, total=False):
    """A carrier: a source polygon edge (entity, polygon, ring, edge), a cut-family member (family, member), or
    unknown."""

    entity: str
    polygon: int
    ring: int
    edge: int
    family: str
    member: str


# --------------------------------------------------------------------------- requests


class InsetFace(TypedDict):
    """One band family: the request of a face.  Every request that landed on the face asks these margins."""

    request_id: str
    support: dict[str, Any]
    interface: str | None
    margins_um: list[Positive]
    margins_grid: list[PositiveInt]
    band_count: PositiveInt
    support_area_um2: float | None
    bands: Annotated[dict[str, IdList], Doc("Band index -> primary groups.")]


class InsetType(TypedDict):
    """A type-level request (every face of a Surface-EPR type) with the families that carry its bands and the
    faces it left unbanded."""

    request_id: str
    support: TypeSupport
    interface: None
    margins_um: list[Positive]
    expansion: TypeExpansion


class TypeSupport(TypedDict):
    """The faces a type-level request asks for: a Surface-EPR type, optionally only some face kinds, without
    the excluded Entities."""

    interface_type: SurfaceEprType
    face_kinds: list[BandedFaceKind] | None
    exclude_entity_ids: IdList


class TypeExpansion(TypedDict):
    """The families a type-level request was expanded into, and the faces it skipped."""

    banded: Annotated[
        list[InsetFace],
        Doc(
            "The families the physical pre-pass created, each as the record of its family (ids <request"
            " id>::<entity>::<side>, or of the first request on a shared face).  A port placed later can cover a"
            " whole face: read whether a family has bands from its own bands and from surface_epr_rows."
        ),
    ]
    skipped: list[SkippedFace]


class SkippedFace(TypedDict):
    """A face a type-level request left unbanded, and why."""

    support: dict[str, Any]
    reason: Literal["no_gap_edge"]


class Refinement(TypedDict):
    """A refinement request as given, and the edges it produced."""

    request_id: str
    z_um: float
    points_um: list[PlanePoint]
    edges: IdList


# --------------------------------------------------------------------------- reports and expectations


@open_record()
class SliverSummary(TypedDict, total=False):
    """Summary of the sliver report (measurements only; decision D2)."""

    shortest_edge_um: float | None
    narrowest_face_um: float | None
    smallest_angle_deg: float | None
    thin_slab_count: int
    max_snap_drift_grid: float | None
    max_arc_deviation_grid: float | None


class Arrangement(TypedDict):
    """The measured snap drift and arc deviation of the arrangement (grid units), and the number of arc edges."""

    max_snap_drift_grid: float | None
    max_arc_deviation_grid: float | None
    arc_edges: Count


class SurfaceKinds(TypedDict):
    """The number of planar and of cylindrical surfaces written."""

    plane: Count
    cylinder: Count


class Expected(TypedDict):
    """What verify_loaded compares, by native tag (tag = 1 + position)."""

    vertices_um: list[SpacePoint]
    edge_vertices: list[Annotated[TagList, Length(2, 2)]]
    face_edges: list[TagList]
    solid_faces: list[TagList]
    face_area_um2: list[Positive]
    solid_volume_um3: list[Positive]
