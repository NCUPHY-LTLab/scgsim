"""The meshing stage of ``simulation_drc``: fine edges and faces of a built model, where they are and who owns them.

Owns the check that runs on a built SGB model before meshing.  It reads only the sidecar
(``built_model``), measures every wall face, horizontal face and free horizontal edge, keeps those below the
limits of :class:`FineStructureLimits`, and attributes each to its owner by lineage:

* ``layout`` -- bounded only by source curves (source polygon edges, or curves a caller asked for), or a band
  curve within the band's widest margin of such a fine structure, which the band reproduces.  The layout
  author fixes it; SCGSim reports it and never repairs it.
* ``erosion`` -- band curves offset from two source edges that are not neighbours meet here: the exact
  erosion has this structure, because the band margin meets the conductor's local width.  The fix is the
  margin or the layout width.  Also a short band edge that offsets a straight source edge shorter than its
  margin: the exact erosion of that edge is this short (#138 (c)).  The message names that source edge, and the
  fix is to declare the arc it is a chord of exactly or to draw longer chords.
* ``sgb`` -- a band curve where the source has no fine structure: an SGB defect, fixed in SCGSim.

The check reports; it never edits geometry.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

# The one geometric constant the meshing stage shares with the geometry layer, which owns it.
from scgsim.geometry.models.grid import WALL_COLUMN_LIMIT_STEPS

from .built_model import (
    Bounds,
    BuiltModel,
    Carrier,
    Measured,
    Point3,
    closest_far_edges,
    face_widths,
    free_edges,
    read_built_model,
    wall_gaps,
    wedge_angles,
)
from .features import checked, load_checked

Action = Literal["refuse", "warn"]
Owner = Literal["layout", "erosion", "sgb"]
_ACTIONS = ("refuse", "warn")
STAGE = "meshing"
_BAND_MEMBER = "band_member"


@dataclass(frozen=True)
class FineStructureLimits:
    """The sizes below which an edge or face is a meshing-stage finding, and whether each class refuses or warns.

    The default values and actions are the Human's decision.  They are a guard against fine structure, not
    a validated criterion for a failed mesh: no meshing stop of the measured baseline cases had a finding of
    these classes.  Lengths are grid steps, so a limit scales with the model grid; findings give micrometres
    beside them.

    * Class A (``wall_column_gap_steps``): a wall face whose column gap -- the length of the outline edge
      it stands on, along the edge (an arc's length for a curved wall), not its film height -- is below
      the limit.  Measured: the 1-step TC jogs of the flip-chip Disk Qubit, and nothing else between 1 and
      65 steps.
    * Class B (``face_width_steps``, ``wedge_angle_deg``): a horizontal face whose width ``2 A / P`` is below
      the limit, or one of whose corners is sharper than the angle.  Measured: the band sliver of
      ``junction_straight``; every other face had 2A/P of 20 nm or more and corners of 5.6 degrees or more.
      Palace refuses a requested Surface-EPR band narrower than ``face_width_steps`` before it builds.
    * Class C (``free_edge_steps``): a horizontal edge that bounds no wall and is shorter than the limit:
      the arc a 0.5 um band leaves at a vertex that turns 2.8 degrees.  The repair loop cleans most of
      these at a cost, so the class warns.
    """

    wall_column_gap_steps: float = float(WALL_COLUMN_LIMIT_STEPS)
    wall_action: Action = "refuse"
    face_width_steps: float = 10.0
    wedge_angle_deg: float = 1.0
    face_action: Action = "refuse"
    free_edge_steps: float = 25.0
    free_edge_action: Action = "warn"

    def __post_init__(self) -> None:
        """Reject a limit that is not a positive number and an action other than refuse or warn."""
        for name in (
            "wall_column_gap_steps",
            "face_width_steps",
            "wedge_angle_deg",
            "free_edge_steps",
        ):
            value = getattr(self, name)
            if not (
                isinstance(value, (int, float)) and math.isfinite(value) and value > 0
            ):
                raise ValueError(f"{name} must be a positive number, got {value!r}.")
        for name in ("wall_action", "face_action", "free_edge_action"):
            if getattr(self, name) not in _ACTIONS:
                raise ValueError(
                    f"{name} must be 'refuse' or 'warn', got {getattr(self, name)!r}."
                )


@dataclass(frozen=True)
class FineFeature:
    """One fine edge or face of the built model: what, where, how small, who owns it and what to do.

    ``rule`` is the class (``A``, ``B`` or ``C``) and ``measure`` the size it was found by; ``value`` is in
    micrometres, or degrees for a wedge angle, and ``value_steps`` the same length in grid steps.
    ``entity`` is the SGB planned id (``dimension`` 1 for an edge, 2 for a face) and ``native_tag`` its tag
    in the XAO.  ``groups`` are the physical groups of the faces it belongs to, ``entities`` the layout
    objects and band requests those faces come from, ``layers`` the layers those objects were drawn on (the
    sidecar's ``entity`` table; an object drawn on no layer, such as the vacuum, adds none), and ``sources``
    the carriers of its edges (a source polygon edge ``<entity>/polygon<p>/ring<r>/edge<e>``, or a band
    offset).  ``bounds_um`` is its box.
    """

    rule: str
    measure: str
    action: Action
    owner: Owner
    entity: str
    dimension: int
    native_tag: int
    groups: tuple[str, ...]
    entities: tuple[str, ...]
    layers: tuple[str, ...]
    sources: tuple[str, ...]
    location_um: Point3
    bounds_um: Bounds
    value: float
    value_steps: float | None
    limit: float
    limit_unit: str
    cause: str
    fix: str
    stage: str = STAGE

    def describe(self) -> str:
        """The finding as one paragraph of text: stage, rule, place, layer, size, source, owner, cause and fix."""
        x, y, z = self.location_um
        size = (
            f"{self.value:.4g} deg"
            if self.value_steps is None
            else f"{self.value:.6g} um ({self.value_steps:.3g} grid steps)"
        )
        groups = ", ".join(self.groups) or "-"
        return (
            f"{self.stage} stage, {self.action} {self.rule}: {self.measure} {size} < {self.limit:g} "
            f"{self.limit_unit} at ({x:.4f}, {y:.4f}, {z:.4f}) um, {self.entity} in {groups}"
            f"{_on_layers(self.layers)}; "
            f"from {', '.join(self.sources) or '-'}; owner {self.owner}. {self.cause} Fix: {self.fix}"
        )


@dataclass(frozen=True)
class FineStructureReport:
    """The meshing-stage findings of one built model and the limits they were found with."""

    grid_um: float
    limits: FineStructureLimits
    findings: tuple[FineFeature, ...] = field(default_factory=tuple)

    @property
    def passed(self) -> bool:
        """Whether no finding refuses the model (warnings do not)."""
        return not self.refusals

    @property
    def refusals(self) -> tuple[FineFeature, ...]:
        """The findings that refuse the model."""
        return tuple(f for f in self.findings if f.action == "refuse")

    @property
    def warnings(self) -> tuple[FineFeature, ...]:
        """The findings that only warn."""
        return tuple(f for f in self.findings if f.action == "warn")

    def to_json(self) -> dict[str, Any]:
        """The report as a JSON-serialisable mapping."""
        return {
            "stage": STAGE,
            "passed": self.passed,
            "grid_um": self.grid_um,
            "limits": asdict(self.limits),
            "findings": [asdict(finding) for finding in self.findings],
        }

    def render_text(self) -> str:
        """The findings as text, refusals first."""
        if not self.findings:
            return f"{STAGE} stage: passed, no fine edge or face below the limits.\n"
        verdict = "refused" if not self.passed else "passed with warnings"
        lines = [
            f"{STAGE} stage: {verdict} ({len(self.refusals)} refusing, {len(self.warnings)} warning)"
        ]
        lines.extend(f"  {finding.describe()}" for finding in self.findings)
        return "\n".join(lines) + "\n"


def check_fine_structure(
    sidecar: Path | str | Mapping[str, Any],
    limits: FineStructureLimits = FineStructureLimits(),  # noqa: B008 -- a frozen dataclass is a safe default
) -> FineStructureReport:
    """The meshing-stage findings of the SGB model a sidecar describes.

    ``sidecar`` is the ``scgsim.sgb.sidecar.v5`` file of one build, or its decoded content; the shared reader
    checks it as the mesher does.  Every wall face, horizontal face and free horizontal edge of the built model
    is measured; those below ``limits`` are found, located in micrometres and attributed to the layout, to the
    exact erosion or to SGB (see the module docstring).  The model is not changed.
    """
    document = (
        checked(sidecar) if isinstance(sidecar, Mapping) else load_checked(sidecar)
    )
    model = read_built_model(document)
    found = _below_limits(model, limits)
    lineage = _Lineage(model)
    layout = [item for item in found if not lineage.band_carriers(item.measured.edges)]
    findings = [
        _finding(model, item, _attribution(lineage, item, layout)) for item in found
    ]
    findings.sort(
        key=lambda f: (f.action != "refuse", f.rule, f.value, f.entity, f.native_tag)
    )
    return FineStructureReport(model.grid_um, limits, tuple(findings))


@dataclass(frozen=True)
class _Found:
    """A measured edge or face below its limit, with the class it falls in."""

    measured: Measured
    rule: str
    action: Action
    limit: float
    limit_unit: str


# Lengths are read from micrometre coordinates, so a length of exactly N grid steps can come out a hair
# under N (100.005 - 99.995 = 0.009999999999990905).  Comparing in steps, below the limit by more than this
# margin, keeps "below N steps" exact; the margin is float noise, far below the one step a lattice resolves.
_FLOAT_NOISE_STEPS = 1e-6


def below_steps(value_um: float, limit_steps: float, grid_um: float) -> bool:
    """Whether a length is below a limit given in grid steps, ignoring float noise at the limit itself.

    The meshing stage compares every length this way; Palace refuses a Surface-EPR band narrower than the
    face-width limit with the same comparison, so the two agree on a band of exactly the limit.
    """
    return value_um / grid_um < limit_steps - _FLOAT_NOISE_STEPS


def _below_limits(model: BuiltModel, limits: FineStructureLimits) -> list[_Found]:
    g = model.grid_um
    found = [
        _Found(m, "A", limits.wall_action, limits.wall_column_gap_steps, "grid steps")
        for m in wall_gaps(model)
        if below_steps(m.value, limits.wall_column_gap_steps, g)
    ]
    found += [
        _Found(m, "B", limits.face_action, limits.face_width_steps, "grid steps")
        for m in face_widths(model)
        if below_steps(m.value, limits.face_width_steps, g)
    ]
    found += [
        _Found(m, "B", limits.face_action, limits.wedge_angle_deg, "deg")
        for m in wedge_angles(model)
        if m.value < limits.wedge_angle_deg
    ]
    found += [
        _Found(m, "C", limits.free_edge_action, limits.free_edge_steps, "grid steps")
        for m in free_edges(model)
        if below_steps(m.value, limits.free_edge_steps, g)
    ]
    return found


class _Lineage:
    """Where the edges of a measured feature come from: their carriers and the band edges they meet."""

    def __init__(self, model: BuiltModel) -> None:
        self.model = model
        self.edges_at: dict[int, list[int]] = {}
        for edge, ends in enumerate(model.edge_vertices, start=1):
            for vertex in ends:
                self.edges_at.setdefault(vertex, []).append(edge)

    def carriers(self, edges: tuple[int, ...]) -> list[Carrier]:
        """The carriers of ``edges``, in edge order."""
        model = self.model
        return [
            model.carriers.get(cid) or Carrier(cid, "unknown")
            for edge in edges
            for cid in model.edge_carriers.get(model.edge_id(edge), ())
        ]

    def band_carriers(self, edges: tuple[int, ...]) -> list[Carrier]:
        """The carriers of ``edges`` that are band curves (members of an inset family)."""
        return [
            c
            for c in self.carriers(edges)
            if c.family is not None and c.family in self.model.band_margins
        ]

    def meeting_edges(self, measured: Measured) -> tuple[int, ...]:
        """The edges whose offsets meet in the feature.

        For a short edge, the edge and every edge that meets it at an end; for a corner, its two edges; for
        a narrow face, the two boundary edges that come closest across it.
        """
        if measured.measure == "edge length":
            return tuple(
                sorted(
                    {
                        other
                        for edge in measured.edges
                        for vertex in self.model.edge_vertices[edge - 1]
                        for other in self.edges_at.get(vertex, ())
                    }
                )
            )
        if measured.measure == "face width 2A/P":
            return closest_far_edges(self.model, measured.edges) or ()
        return measured.edges

    def offsets_of_far_edges(self, carriers: list[Carrier]) -> bool:
        """Whether two band curves of one family are offsets of source primitives that are not neighbours.

        A member names the primitives it touches on a ring of the band's support (see :meth:`place`).  Two
        members on different rings are apart.  On one ring of ``N`` primitives, as many as its ``support_ring``
        record lists, primitives ``a`` and ``b`` are neighbours exactly when ``(a - b) mod N`` is 0, 1 or
        ``N - 1``: the same primitive, or two that share a vertex, the ring's last and first included.  Two
        members are apart when no primitive one touches is a neighbour of one the other touches.
        """
        members = [p for c in carriers if (p := self.place(c)) is not None]
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                if a.family != b.family:
                    continue
                if a.ring != b.ring:
                    return True
                if all(
                    _steps_around(p, q, a.ring_size) > 1
                    for p in a.touched
                    for q in b.touched
                ):
                    return True
        return False

    def short_sources(self, measured: Measured) -> tuple[_ShortSource, ...]:
        """The straight source edges shorter than their band margin that a short band edge offsets (#138 (c)).

        Only an edge-length finding has them.  Each of its band curves of role ``edge`` offsets one support
        primitive by that curve's own margin, and the primitive lies on source curves; a straight one shorter
        than the margin is such an edge.  An arc source is left out: the rule and its fix are about chords.
        """
        if measured.measure != "edge length":
            return ()
        model = self.model
        found = {
            _ShortSource(source, carrier.margin_grid)
            for carrier in self.band_carriers(measured.edges)
            if carrier.role == "edge" and carrier.margin_grid is not None
            for source in carrier.primitive_sources
            if (curve := model.source_curves[source]).kind == "line"
            and below_steps(curve.length_um, carrier.margin_grid, model.grid_um)
        }
        return tuple(sorted(found, key=lambda s: (s.source, s.margin_grid)))

    def place(self, carrier: Carrier) -> _MemberPlace | None:
        """The ring primitives a band member offsets or turns round, or ``None`` for another kind of carrier.

        An ``edge`` member is the offset of primitive ``k`` of its ring, a ``corner`` the arc about the vertex
        between primitives ``k`` and ``k + 1`` (the ring's first primitive after its last), and a ``capstart``
        or ``capend`` the cap about an end of primitive ``k``.  A cap or corner cut into pieces (each at most a
        quarter turn) is placed as the whole member: every piece is an arc of the member's margin about the
        same vertex, so it touches the same primitives.
        """
        if (
            carrier.kind != _BAND_MEMBER
            or carrier.family is None
            or carrier.ring is None
            or carrier.index is None
        ):
            return None
        size = self.model.ring_sizes[carrier.ring]
        index = carrier.index
        touched = (index, (index + 1) % size) if carrier.role == "corner" else (index,)
        return _MemberPlace(carrier.family, carrier.ring, size, touched)


@dataclass(frozen=True)
class _MemberPlace:
    """Where on its support a band member comes from: the family, the ring, its size and the primitives touched."""

    family: str
    ring: str
    ring_size: int
    touched: tuple[int, ...]


def _steps_around(a: int, b: int, ring_size: int) -> int:
    """How many primitives apart places ``a`` and ``b`` lie on a closed ring, the shorter way round."""
    forward = (a - b) % ring_size
    return min(forward, ring_size - forward)


@dataclass(frozen=True)
class _ShortSource:
    """A source curve a band curve offsets, shorter than that curve's margin (grid steps)."""

    source: str
    margin_grid: int


@dataclass(frozen=True)
class _Attribution:
    """Who owns a found feature; for the erosion of a source edge shorter than its margin, those source edges."""

    owner: Owner
    short_sources: tuple[_ShortSource, ...] = ()


def _attribution(lineage: _Lineage, item: _Found, layout: list[_Found]) -> _Attribution:
    """Who owns a found feature, by lineage (see the module docstring).

    The two erosion rules are tried in this order: where offsets of far source edges meet, the margin meets the
    conductor's width whatever its edges' lengths, so that finding keeps its fix; only a short edge no such
    meeting explains is laid to a source edge shorter than its margin.
    """
    band = lineage.band_carriers(item.measured.edges)
    if not band:
        return _Attribution("layout")
    reach = max(
        max(lineage.model.band_margins[c.family or ""], default=0.0) for c in band
    )
    if any(
        _box_distance(item.measured.bounds, other.measured.bounds) <= reach
        for other in layout
    ):
        # The band's offset of a fine structure the layout drew.
        return _Attribution("layout")
    meeting = lineage.band_carriers(lineage.meeting_edges(item.measured))
    if lineage.offsets_of_far_edges(meeting):
        return _Attribution("erosion")
    short = lineage.short_sources(item.measured)
    if short:
        return _Attribution("erosion", short)
    return _Attribution("sgb")


def _finding(model: BuiltModel, item: _Found, attribution: _Attribution) -> FineFeature:
    measured = item.measured
    faces = _faces_of(model, measured)
    face_ids = [model.face_id(face) for face in faces]
    is_angle = measured.measure == "wedge angle"
    carriers = sorted(
        {
            cid
            for edge in measured.edges
            for cid in model.edge_carriers.get(model.edge_id(edge), ())
        }
    )
    entity = (
        model.edge_id(measured.tag)
        if measured.dimension == 1
        else model.face_id(measured.tag)
    )
    owner = attribution.owner
    cause, fix = _cause_and_fix(model, measured.measure, attribution)
    entities = sorted({s for f in face_ids for s in model.face_sources.get(f, ())})
    return FineFeature(
        rule=item.rule,
        measure=measured.measure,
        action=item.action,
        owner=owner,
        entity=entity,
        dimension=measured.dimension,
        native_tag=measured.tag,
        groups=tuple(sorted({g for f in face_ids if (g := model.face_group.get(f))})),
        entities=tuple(entities),
        layers=_layers(model, face_ids),
        sources=tuple(carriers),
        location_um=_rounded_point(measured.location),
        bounds_um=(
            _rounded_point(measured.bounds[0]),
            _rounded_point(measured.bounds[1]),
        ),
        value=round(measured.value, 9),
        value_steps=None if is_angle else round(measured.value / model.grid_um, 3),
        limit=item.limit,
        limit_unit=item.limit_unit,
        cause=cause,
        fix=fix,
    )


def _faces_of(model: BuiltModel, measured: Measured) -> list[int]:
    """The faces a feature belongs to: the face itself, or every face an edge bounds."""
    if measured.dimension == 2:  # noqa: PLR2004 -- a face
        return [measured.tag]
    return [
        face
        for face, edges in enumerate(model.face_edges, start=1)
        if measured.tag in edges
    ]


# Why the mesher fails on each measure, and the fix each owner can make.
_CAUSE = {
    "wall column gap": (
        "A wall face this narrow forces elements as narrow as it across the whole film height; "
        "the repair loop refines around it and new bad elements appear."
    ),
    "face width 2A/P": (
        "A face this narrow forces flat elements along its whole length; refining it does not "
        "make them better shaped."
    ),
    "wedge angle": "A corner this sharp forces flat elements into its tip.",
    "edge length": (
        "An edge this short needs elements of its own length around it, which costs repair "
        "rounds and elements."
    ),
}
_LAYOUT_FIX = {
    "wall column gap": (
        "remove the step from the polygon outline (an out-and-back spike, a jog or a slit leaves an "
        "outline edge this short)."
    ),
    "face width 2A/P": "widen or remove the sliver between these polygons.",
    "wedge angle": "open or remove the sharp corner where these outline edges meet.",
    "edge length": "join the short outline edge into its neighbours.",
}
_EROSION_FIX = (
    "the band margin meets the conductor's local width here; change the margins in "
    "surface_epr_specs or the width of the conductor."
)
_SGB_FIX = (
    "an SGB defect: the band offsets leave this where the layout has no fine structure. "
    "Report it to SCGSim with this finding; the layout needs no change."
)


def _cause_and_fix(
    model: BuiltModel, measure: str, attribution: _Attribution
) -> tuple[str, str]:
    if attribution.owner == "layout":
        return _CAUSE[measure], _LAYOUT_FIX[measure]
    if attribution.short_sources:
        return _CAUSE[measure], _short_source_fix(model, attribution.short_sources)
    if attribution.owner == "erosion":
        return _CAUSE[measure], _EROSION_FIX
    return _CAUSE[measure], _SGB_FIX


def _short_source_fix(model: BuiltModel, short: tuple[_ShortSource, ...]) -> str:
    """The fix for a band edge the exact erosion of a short source edge leaves: name each edge, where and how long."""
    named = []
    for item in short:
        curve = model.source_curves[item.source]
        (x0, y0), (x1, y1) = curve.start_um, curve.end_um
        named.append(
            f"the source edge {item.source}, {curve.length_um:.6g} um long from ({x0:.4f}, {y0:.4f}) to "
            f"({x1:.4f}, {y1:.4f}) um, is shorter than the {item.margin_grid * model.grid_um:g} um band "
            "margin it is offset by"
        )
    return (
        f"{'; '.join(named)}, so the exact erosion leaves an edge this short. Declare the arc that edge "
        "is a chord of exactly, or draw it with longer chords."
    )


def _layers(model: BuiltModel, face_ids: list[str]) -> tuple[str, ...]:
    """The layers of the faces' Entities, sorted; a band family or request named like an Entity draws none."""
    return tuple(
        sorted(
            {
                layer
                for face in face_ids
                for entity in model.face_entities.get(face, ())
                for layer in model.entity_layers.get(entity, ())
            }
        )
    )


def _on_layers(layers: tuple[str, ...]) -> str:
    """`` on layer L`` or `` on layers L1, L2``, as every located refusal names its layers; nothing for none."""
    if not layers:
        return ""
    return f" on layer{'s' if len(layers) > 1 else ''} {', '.join(layers)}"


def _box_distance(a: Bounds, b: Bounds) -> float:
    """Distance between two axis-aligned boxes (0 when they touch or overlap)."""
    return math.sqrt(
        sum(max(a[0][k] - b[1][k], b[0][k] - a[1][k], 0.0) ** 2 for k in range(3))
    )


def _rounded_point(point: Point3) -> Point3:
    return (round(point[0], 6), round(point[1], 6), round(point[2], 6))
