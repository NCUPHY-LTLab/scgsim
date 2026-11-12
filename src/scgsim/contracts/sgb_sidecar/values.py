"""The values the sidecar records are made of: bounded numbers, points, the enums, and the ids of each table.

Owns the type aliases :mod:`.records` and :mod:`.features` share: a ``Ref`` alias says which table an id names
(:mod:`.markers`), and a ``Literal`` lists the values a field may take.
"""

from typing import Annotated, Literal

from .markers import ExclusiveMinimum, Length, Minimum, Ref

IdList = list[
    str
]  # names the sidecar keeps no table of (requests as a caller gave them)
PositiveInt = Annotated[int, Minimum(1)]
Tag = PositiveInt  # a native tag (or a physical-group tag): never 0
Count = Annotated[int, Minimum(0)]
Positive = Annotated[float, ExclusiveMinimum(0)]
NonNegative = Annotated[float, Minimum(0)]
PlanePoint = Annotated[list[float], Length(2, 2)]  # (x, y) um
SpacePoint = Annotated[list[float], Length(3, 3)]  # (x, y, z) um
ZRange = Annotated[list[float], Length(2, 2)]  # (lowest, highest) z um
Segment = Annotated[list[PlanePoint], Length(2, 2)]  # (start, end) um

EntityRef = Annotated[str, Ref("entity")]
SourceCurveRef = Annotated[str, Ref("source_curve")]
BandFamilyRef = Annotated[str, Ref("band_family")]
RefinementRequestRef = Annotated[str, Ref("refinement_request")]
CutFamilyRef = Annotated[str, Ref("cut_family")]
PortRef = Annotated[str, Ref("port")]
SupportRingRef = Annotated[str, Ref("support_ring")]
SupportPrimitiveRef = Annotated[str, Ref("support_primitive")]
CarrierRef = Annotated[str, Ref("carrier")]
VertexRef = Annotated[str, Ref("vertex")]
EdgeRef = Annotated[str, Ref("edge")]
FaceRef = Annotated[str, Ref("face")]
VolumeRef = Annotated[str, Ref("volume")]
SurfaceGroupRef = Annotated[str, Ref("group", dimension=2)]
VolumeGroupRef = Annotated[str, Ref("group", dimension=3)]

Route = Literal["A", "B", "C"]
EntityKind = Literal["conductor", "dielectric", "vacuum", "port"]
VolumeKind = Literal["vacuum", "dielectric", "conductor"]
SolutionKind = Literal["vacuum", "dielectric"]
FaceKind = Literal["top", "bottom", "sidewall", "sheet", "port", "interface"]
InterfaceType = Literal["MA", "MS", "MM", "SA", "AA", "SS", "SHEET", "PORT", "EXT"]
SurfaceEprType = Literal["MA", "MS", "SA"]
SheetSide = Literal["above", "below", "both"]
RowSide = Literal["above", "below", "both", "mixed"]
SidecarFile = Literal["xao", "brep", "sidecar", "sliver_report", "validation_report"]
BandedFaceKind = Literal["sheet", "top", "bottom", "interface"]
ConductorSide = Literal["sheet", "top", "bottom"]
BandMemberRole = Literal["edge", "corner", "capstart", "capend"]
BoxSide = Literal["-x", "+x", "-y", "+y", "bottom", "top"]
