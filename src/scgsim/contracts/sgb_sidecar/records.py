"""The definition of the SGB sidecar ``scgsim.sgb.sidecar.v3``: every table and record, as a ``TypedDict``.

Owns the shape of the file: its tables, the fields of each record, their types, enums and bounds, the table
each id names, and which records accept more keys.  The value types and the ids of each table are in :mod:`.values`, and the
mesh-feature records in :mod:`.features`.  It is the one definition the JSON Schema is derived from
(:mod:`.schema`) and the shared reader checks against (:mod:`.read`, :mod:`.references`).  The writer is
``scgsim.geometry.sgb.sidecar``; what each table means for SGB and for its readers is documented there.

The sidecar has three layers, each a set of tables ``{id: record}``.  *Sources* are what the layout drew and
asked for (``entity``, ``source_curve``, ``inset_request``, ``refinement_request``, ``port``).  *Derived* are
what SGB computed from them (``cut_family``, ``band_family``, ``support_ring``, ``support_primitive``,
``carrier``).  *Emitted*
are the shapes that entered the model and what is said about them (``vertex``, ``edge``, ``face``, ``volume``,
``group``, ``surface_epr_row``, ``mesh_feature``).  A record names a record of another table by its id, and the
field's type says which table (``Ref``, :mod:`.markers`); the shared reader checks that every such id is a key
of that table.  Ids are opaque: a reader compares them and looks them up, and never reads anything out of one.
Each fact is stored once; orderings and copies a reader needs, such as anything by native tag, are computed
from the tables by :mod:`.views`.

Every record is closed (a key it does not declare makes the sidecar invalid) except the two marked
``@open_record``: ``counts`` and ``slivers``.  A record's docstring is its description in the schema, and
``Doc`` describes a field.  The module uses no ``Required``: Python does not see it in a postponed annotation,
so a record with optional fields declares its required ones in a base and the rest in a ``total=False``
subclass.

Terms.  A *planned id* is SGB's id of a vertex, edge, face or volume; its *native tag* is the Gmsh tag of that
shape in the written XAO, 1 + the id's position in ``binding.ids_by_tag``.  An *Entity* is one object of the
layout (a conductor, a dielectric, the vacuum, a lumped-port sheet).  Interface types: ``MA`` metal-air, ``MS``
metal-substrate, ``MM`` metal-metal, ``SA`` substrate-air, ``AA`` air-air, ``SS`` substrate-substrate, ``SHEET``
a Route A conductor sheet, ``PORT`` a lumped-port sheet, ``EXT`` the model boundary.
"""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from .features import MeshFeature
from .markers import (
    Doc,
    Length,
    MinProperties,
    Pattern,
    open_record,
)
from .values import (
    BandedFaceKind,
    BandFamilyRef,
    BandMemberRole,
    BoxSide,
    CarrierRef,
    ConductorSide,
    Count,
    CutFamilyRef,
    EdgeRef,
    EntityKind,
    EntityRef,
    FaceKind,
    FaceRef,
    IdList,
    InterfaceType,
    NonNegative,
    PlanePoint,
    Positive,
    PositiveInt,
    RefinementRequestRef,
    Route,
    RowSide,
    Segment,
    SheetSide,
    SidecarFile,
    SolutionKind,
    SourceCurveRef,
    SpacePoint,
    SupportPrimitiveRef,
    SupportRingRef,
    SurfaceEprType,
    SurfaceGroupRef,
    Tag,
    VertexRef,
    VolumeGroupRef,
    VolumeKind,
    VolumeRef,
    ZRange,
)
from .version import SidecarVersion

# --------------------------------------------------------------------------- the document


class Sidecar(TypedDict):
    """Sidecar of the SGB XAO/BREP: the model, the binding of planned ids to native tags, a summary, and the
    tables of sources, derived curves and emitted shapes, linked by typed ids."""

    schema: SidecarVersion
    model: Model
    binding: Binding
    summary: Summary
    entity: Annotated[dict[str, Entity], Doc("Sources: every Entity, by Entity id.")]
    source_curve: Annotated[
        dict[str, SourceCurve],
        Doc(
            "Sources: every straight edge and circular arc of every Entity outline, by carrier id."
        ),
    ]
    inset_request: Annotated[
        dict[str, InsetRequest],
        Doc("Sources: the inset requests as given, by request id."),
    ]
    refinement_request: Annotated[
        dict[str, RefinementRequest],
        Doc("Sources: the refinement requests as given, by request id."),
    ]
    port: Annotated[
        dict[str, Port], Doc("Sources: the lumped-port sheets, by port sheet id.")
    ]
    cut_family: Annotated[
        dict[str, CutFamily],
        Doc(
            "Derived: every cut family by id (band, refinement and any other role), with every family a carrier"
            " names."
        ),
    ]
    band_family: Annotated[
        dict[str, BandFamily],
        Doc("Derived: one band family per banded face, by family id."),
    ]
    support_ring: Annotated[
        dict[str, SupportRing],
        Doc("Derived: the boundary rings of each band family's support region."),
    ]
    support_primitive: Annotated[
        dict[str, SupportPrimitive],
        Doc("Derived: the exact primitives (lines and arcs) of each support ring."),
    ]
    carrier: Annotated[
        dict[str, Carrier],
        Doc(
            "Derived: every carrier an emitted horizontal or refinement edge lies on, and what it comes from."
        ),
    ]
    vertex: Annotated[dict[str, Vertex], Doc("Emitted: every written vertex.")]
    edge: Annotated[dict[str, Edge], Doc("Emitted: every written edge.")]
    face: Annotated[dict[str, Face], Doc("Emitted: every written face.")]
    volume: Annotated[
        dict[str, Volume],
        Doc("Emitted: every planned volume, the written solids marked emitted."),
    ]
    group: Annotated[
        dict[str, SurfaceGroup | VolumeGroup],
        Doc(
            "Emitted: the primary physical groups by name; they partition each dimension.  A face or volume"
            " names its group; the members of a group are computed from them."
        ),
    ]
    surface_epr_row: Annotated[
        dict[str, SurfaceEprRow],
        Doc("Emitted: every Surface-EPR row and the group that carries it, by row id."),
    ]
    mesh_feature: Annotated[
        dict[str, MeshFeature],
        Doc(
            "Emitted: facts about the geometry for the mesher (no sizes): film sidewalls, band boundaries, arcs,"
            " refinement curves, port edges.  Their order is computed (by kind, source and facts)."
        ),
    ]


# --------------------------------------------------------------------------- the model, binding and summary


class Model(TypedDict):
    """The model's name and identity digest, its route, grid and tolerance, and the files written beside it."""

    name: str
    identity_digest: Annotated[str, Pattern("^model:[0-9a-f]{16}$")]
    route: Route
    route_policy: RoutePolicy
    grid_um: Positive
    tolerance_um: Positive
    files: Annotated[dict[SidecarFile, str], Doc("File names relative to the sidecar.")]


class RoutePolicy(TypedDict):
    """Whether the route excludes or retains conductor volumes, and whether it allows conductor sheets."""

    conductor_volumes: Literal["excluded", "retained"]
    conductor_sheets: bool


class Binding(TypedDict):
    """Planned id <-> native tag per dimension, and the rule that relates them."""

    rule: str
    ids_by_tag: Annotated[
        IdsByTag,
        Doc(
            "Planned ids per dimension in native-tag order: tag = 1 + position.  They are exactly the ids of"
            " the vertex, edge and face tables and of the emitted volumes."
        ),
    ]


IdsByTag = TypedDict(
    "IdsByTag",
    {
        "0": list[VertexRef],
        "1": list[EdgeRef],
        "2": list[FaceRef],
        "3": list[VolumeRef],
    },
)


class Summary(TypedDict):
    """Counts of the written file, the sliver report's summary and the number of arc edges, each once."""

    counts: Counts
    slivers: SliverSummary
    arc_edges: Annotated[
        Count, Doc("The arc edges of the planned topology, written or not.")
    ]


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


@open_record()
class SliverSummary(TypedDict, total=False):
    """Summary of the sliver report (measurements only; decision D2), with the snap drift and arc deviation of
    the arrangement in grid units."""

    shortest_edge_um: float | None
    narrowest_face_um: float | None
    smallest_angle_deg: float | None
    thin_slab_count: int
    max_snap_drift_grid: float | None
    max_arc_deviation_grid: float | None


# --------------------------------------------------------------------------- sources


class Entity(TypedDict):
    """One Entity of the layout: the prisms it is drawn as, one per polygon, in catalogue order."""

    prisms: Annotated[list[EntityPrism], Length(min_items=1)]


class EntityPrism(TypedDict):
    """One prism (polygon) of an Entity: what it is made of, its Net, the layer it was drawn on, its z range and
    feature kind.  The prisms of one Entity may differ in any of these."""

    polygon: Count
    kind: EntityKind
    material: str
    net: str | None
    layer: Annotated[
        str | None,
        Doc(
            "The source layer: its logical level id, else its GDS layer/datatype; null when the prism has"
            " none (the vacuum, a substrate, a scene drawn directly as SGB records)."
        ),
    ]
    z_um: ZRange
    feature_kind: Annotated[
        str | None,
        Doc(
            "The physical declaration's feature kind (stack_film, local_film, finite_volume); null without one."
        ),
    ]


class SourceCurve(TypedDict):
    """One straight edge or circular arc of an Entity outline as the layout drew it: where it lies in the
    outline (polygon, ring, edge number), its exact curve and its length."""

    entity: EntityRef
    polygon: Count
    ring: Count
    edge: Count
    curve: LineCurve | ArcCurve
    length_um: NonNegative


class LineCurve(TypedDict):
    """A straight curve from start to end."""

    kind: Literal["line"]
    start_um: PlanePoint
    end_um: PlanePoint


class ArcCurve(TypedDict):
    """A circular arc from start to end about a centre, counter-clockwise or not."""

    kind: Literal["arc"]
    centre_um: PlanePoint
    radius_um: Positive
    ccw: bool
    start_um: PlanePoint
    end_um: PlanePoint


class InsetRequest(TypedDict):
    """An inset request as given: the face it names (scope face) or the Surface-EPR type it asks for (scope
    type), its margins, the band families it became and the faces it left unbanded."""

    scope: Literal["face", "type"]
    support: FaceSupport | InterfaceSupport | TypeSupport
    interface: str | None
    margins_um: list[Positive]
    families: Annotated[
        list[BandFamilyRef],
        Doc(
            "The band families the request became: a face request's own family, or the families the physical"
            " pre-pass created for a type-level request, in face order.  A port placed later can cover a whole"
            " face: read whether a family has bands from its own bands and from surface_epr_row."
        ),
    ]
    skipped: Annotated[
        list[SkippedFace],
        Doc("The faces a type-level request left unbanded (empty for a face request)."),
    ]


class FaceSupport(TypedDict):
    """A conductor face: the Entity and its side."""

    entity: EntityRef
    side: ConductorSide


class InterfaceSupport(TypedDict):
    """The interface between two solution regions, at one height or (null) wherever they meet."""

    first_entity: EntityRef
    second_entity: EntityRef
    z_um: float | None


class TypeSupport(TypedDict):
    """The faces a type-level request asks for: a Surface-EPR type, optionally only some face kinds, without
    the excluded Entities."""

    interface_type: SurfaceEprType
    face_kinds: list[BandedFaceKind] | None
    exclude_entity_ids: IdList


class SkippedFace(TypedDict):
    """A face a type-level request left unbanded, and why."""

    support: FaceSupport | InterfaceSupport
    reason: Literal["no_gap_edge"]


class RefinementRequest(TypedDict):
    """A refinement family: the request as given (a polyline along which the mesher refines) and the segments SGB
    drew from it, which are its carriers.  The heights they are drawn at are the family's levels (``cut_family``)."""

    points_um: Annotated[
        list[PlanePoint] | None,
        Doc(
            "The polyline as the request gave it; null for a family a catalogue declares directly, with no request."
        ),
    ]
    segments_um: Annotated[
        list[Segment],
        Doc(
            "Each segment of the family, start and end, in carrier order: a repeated point draws none, and"
            " segments a catalogue declares are kept as they are, joined or not."
        ),
    ]


class Port(TypedDict):
    """A lumped-port sheet: the authored identity, its Entity, and the faces and volumes that carry it."""

    entity: EntityRef
    group: SurfaceGroupRef
    source_name: str
    port_index: int
    direction: list[float]
    host_entity_ids: list[EntityRef]
    source_layer: str
    target_layer: str
    z_um: float
    faces: list[FaceRef]
    host_volumes: list[VolumeRef]
    adjacent_volumes: list[VolumeRef]


# --------------------------------------------------------------------------- derived


class CutFamily(TypedDict):
    """A named set of curves SGB adds to the arrangement: its semantic role, whether its curves split cells
    (``required``) or are only embedded (``refinement_only``), and every level they are drawn at.  A family that a
    carrier names but the catalogue does not hold has no role, policy or levels."""

    role: str | None
    edge_policy: Literal["required", "refinement_only"] | None
    levels_um: Annotated[
        list[float],
        Doc("Every height the family's curves are drawn at, in the family's order."),
    ]


class BandFamily(TypedDict):
    """One band family: the face an inset request bands, its margins (also in model-grid units, the identity a
    margin is looked up by), the area of its support region and the groups of each band."""

    support: FaceSupport | InterfaceSupport
    interface: str | None
    margins_um: list[Positive]
    margins_grid: list[PositiveInt]
    band_count: PositiveInt
    support_area_um2: float | None
    bands: Annotated[
        dict[str, IdList],
        Doc(
            "Band index -> the primary groups of the faces in that band.  In Route B a family also names the"
            " MM group of a metal-metal contact, which is not a physical group, so these names are not checked."
        ),
    ]


class SupportRing(TypedDict):
    """One boundary ring of a band family's support region (interior on the left): its primitives in order, so
    their count is the ring's size."""

    family: BandFamilyRef
    index: Annotated[Count, Doc("The ring's position among the region's rings.")]
    primitives: list[SupportPrimitiveRef]


class SupportPrimitive(TypedDict):
    """One exact primitive of a support ring: its curve and length, whether bands are measured from it (a
    distance origin), whether it lies on the die edge, and the source curves it lies on."""

    ring: SupportRingRef
    curve: LineCurve | ArcCurve
    length_um: NonNegative
    origin: bool
    die_edge: bool
    sources: Annotated[
        list[SourceCurveRef],
        Doc(
            "The source curves drawn along the primitive, one per Entity whose outline runs there (a ground"
            " edge on the die outline also carries the substrate and vacuum edges)."
        ),
    ]


class SourceCarrier(TypedDict):
    """A carrier that is a source curve, or a witness chord of one."""

    kind: Literal["source"]
    source_curve: SourceCurveRef


class BandMemberCarrier(TypedDict):
    """A band curve of an inset family: the support primitive it follows, how, and at which margin.

    ``edge`` offsets the primitive; ``corner`` turns round the reflex vertex after it (to the next primitive of
    its ring); ``capstart`` and ``capend`` close round its start or end.  ``piece`` numbers the parts of a cap or
    corner cut into several, and is null for a member in one piece."""

    kind: Literal["band_member"]
    primitive: SupportPrimitiveRef
    role: BandMemberRole
    margin_grid: PositiveInt
    piece: Count | None


class RefinementCarrier(TypedDict):
    """A segment of a refinement request's polyline."""

    kind: Literal["refinement"]
    request: RefinementRequestRef


class FamilyMemberCarrier(TypedDict):
    """A curve of a cut family that is neither a band curve on a support primitive nor a refinement segment: the
    family its lineage names and the member name it was given (an opaque label)."""

    kind: Literal["family_member"]
    family: CutFamilyRef
    member: str


class UnknownCarrier(TypedDict):
    """A carrier whose origin SGB does not know."""

    kind: Literal["unknown"]


Carrier = (
    SourceCarrier
    | BandMemberCarrier
    | RefinementCarrier
    | FamilyMemberCarrier
    | UnknownCarrier
)


# --------------------------------------------------------------------------- emitted


class Vertex(TypedDict):
    """One written vertex."""

    position_um: SpacePoint


class Edge(TypedDict):
    """One written edge: its kind, its two vertices, its arc (null for a line), its length, and what it comes
    from: the carriers of a horizontal or refinement edge, the Entities whose walls stand on a vertical one."""

    kind: Literal["horizontal", "vertical", "refinement"]
    vertices: Annotated[list[VertexRef], Length(2, 2)]
    arc: EdgeArc | None
    length_um: NonNegative
    carriers: list[CarrierRef]
    entities: list[EntityRef]
    written_as_free_edge: bool


class EdgeArc(TypedDict):
    """The circle an arc edge was written on."""

    centre_um: PlanePoint
    radius_um: Positive
    ccw: Annotated[
        bool, Doc("The direction of travel from the first vertex to the second.")
    ]


class Face(TypedDict):
    """One written face: its group, kind, face kind, z range, the volumes on its two sides, its written loops and
    area, and what it comes from."""

    group: SurfaceGroupRef | None
    kind: Literal["horizontal", "wall"]
    face_kind: FaceKind
    z_um: ZRange
    negative_volume: VolumeRef | None
    positive_volume: VolumeRef | None
    loops: Annotated[
        list[Loop], Length(min_items=1), Doc("The written wires of the face.")
    ]
    area_um2: Positive
    curved: Annotated[bool, Doc("A wall on a cylinder.")]
    entities: Annotated[
        list[EntityRef],
        Doc(
            "The Entities whose outlines bound the face, from the source carriers of its edges."
        ),
    ]
    families: Annotated[
        list[BandFamilyRef],
        Doc(
            "The band families whose curves bound the face, from its band-member carriers."
        ),
    ]
    refinements: Annotated[
        list[RefinementRequestRef],
        Doc("The refinement requests whose segments lie on its boundary edges."),
    ]
    cut_families: Annotated[
        list[CutFamilyRef],
        Doc("The other cut families whose members lie on its boundary edges."),
    ]
    membership: Annotated[
        dict[EntityRef, NonNegative],
        Doc(
            "Entity -> covered fraction of the face area; non-exclusive, may sum to more than 1 (walls use the"
            " share of their bottom boundary length)."
        ),
    ]
    bands: Annotated[
        list[BandLabel], Doc("The inset band labels of the face, in label order.")
    ]
    embedded_edges: Annotated[
        list[EdgeRef],
        Doc("Refinement-only curves the mesher embeds in this face, in order."),
    ]
    outer_boundary: Annotated[
        OuterBoundary | None,
        Doc(
            "For a face on the model's outer boundary (one side volume is null): where it lies; null for every"
            " other face."
        ),
    ]


class OuterBoundary(TypedDict):
    """Where a face on the model's outer boundary lies: the side of the model's box and the domain it bounds.

    The box is the extent of the written vertices.  A planar face whose vertices all lie on one of its six
    planes is on that side: ``-x`` and ``+x`` the lowest and highest x, ``-y`` and ``+y`` likewise, ``bottom``
    and ``top`` the lowest and highest z.  A face on none of them (a curved wall, or a die edge that is not
    parallel to an axis) has no side."""

    side: BoxSide | None
    domain: Annotated[
        EntityRef,
        Doc("The Entity of the volume on the face's other side, inside the model."),
    ]


class Loop(TypedDict):
    """One written wire: its edges in order, and whether each is travelled against its own direction."""

    edges: Annotated[list[EdgeRef], Length(min_items=1)]
    reversed: list[bool]


class BandLabel(TypedDict):
    """An inset band: the band family and the band index."""

    family: BandFamilyRef
    index: Count


class Volume(TypedDict):
    """One planned volume: its Entity and domain, whether it is emitted, its group, shells, planned volume and
    AEDT name, the faces of the written solid and the sheets embedded in it."""

    entity: EntityRef
    kind: VolumeKind
    material: str
    net: str | None
    emitted: bool
    group: VolumeGroupRef | None
    shells: PositiveInt
    volume_um3: float
    aedt_name: Annotated[str, Pattern("^[A-Za-z0-9_]+$")] | None
    faces: Annotated[
        list[FaceRef],
        Doc("The faces of the written solid in tag order; empty when not emitted."),
    ]
    embedded_faces: Annotated[
        list[FaceRef],
        Doc(
            "Sheets inside the emitted volume that bound no shell; the mesher embeds each, in order."
        ),
    ]


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
    entities: list[EntityRef]
    net: str | None
    face_kind: Annotated[
        FaceKind,
        Doc(
            "The one face kind every face of the group has; it is part of the group name (PORT groups: port)."
        ),
    ]
    adjacent_domains: list[AdjacentDomain]
    adjacent_volumes: list[VolumeRef]
    band: GroupBand | None
    area_um2: NonNegative


class AdjacentDomain(TypedDict):
    """A solution region (vacuum or dielectric) on a side of the group's faces."""

    name: EntityRef
    kind: SolutionKind
    material: str


class GroupBand(TypedDict):
    """The inset band a group's faces lie in: the band family and the band index."""

    family: BandFamilyRef
    index: Count


class VolumeGroup(TypedDict):
    """One primary volume group (dimension 3), named by the bare entity id."""

    dim: Literal[3]
    tag: Tag
    interface_type: None
    entities: list[EntityRef]
    kind: VolumeKind
    material: str
    net: str | None


class SurfaceEprRow(TypedDict):
    """One Surface-EPR row and the primary group that carries it."""

    group: SurfaceGroupRef
    interface_type: SurfaceEprType
    entities: list[EntityRef]
    band: Count | None
    side: Annotated[
        RowSide | None,
        Doc(
            "A sheet row: the side every face in face_sides agrees on, or mixed when they disagree; null for any"
            " other row."
        ),
    ]
    face_sides: Annotated[
        Annotated[dict[FaceRef, SheetSide], MinProperties(1)] | None,
        Doc(
            "A sheet row: face id -> the side of that face whose neighbour (vacuum for MA, dielectric for MS) the"
            " row integrates, listing exactly the faces that have such a neighbour; null for any other row."
        ),
    ]
    shared_face_group: Annotated[
        SurfaceGroupRef | None,
        Doc(
            "The sheet group for a sheet row (its MA and MS rows integrate the same two-sided face), else null."
        ),
    ]
