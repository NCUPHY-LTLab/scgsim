"""Compose authored junction cuts, model emission and optional grid realization."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import Any

from scgsim.geometry.grid.grid_regions import grid_change_message
from scgsim.geometry.grid.planar_loops import PolygonCoordinates
from scgsim.geometry.grid.scope import UNGRIDDED, GridScope
from scgsim.geometry.junctions.boolean_geometry import (
    boolean,
    build_polygon_region,
    read_boolean_rings,
    rectangle,
)
from scgsim.geometry.junctions.contact import (
    JunctionContactRequest,
    junction_contact_inventory,
)
from scgsim.geometry.junctions.core import (
    DEFAULT_FORMAT,
    PRECISION_UM,
    JunctionPartitionFormat,
    canonical_sha256,
    plain_record,
)
from scgsim.geometry.junctions.emission import realized_on_grid
from scgsim.geometry.junctions.exact_geometry import covered, section
from scgsim.geometry.junctions.lattice import (
    CutLattice,
    LatticeRectangle,
    drawn_exterior,
    lattice_rectangle,
)
from scgsim.geometry.junctions.local_frame import JunctionLocalFrame, JunctionPlaces
from scgsim.geometry.junctions.model import assembled_input, model_geometry
from scgsim.geometry.junctions.selection import (
    END_REFUSALS,
    BuiltOutline,
    EndSection,
    built_outline,
    contact_points,
    contacts,
    end_candidate,
    end_region,
    nearest_meetings,
    own_contact,
    side_steps,
)
from scgsim.geometry.junctions.terminal import junction_terminal_line
from scgsim.geometry.models.grid import (
    WALL_COLUMN_LIMIT_STEPS,
    GeometryGridContext,
    format_extent,
    format_length,
    geometry_grid_context,
)
from scgsim.geometry.models.input import (
    GeometryBuildInput,
    LayoutPolygonSpec,
    SemanticEntitySpec,
)
from scgsim.geometry.models.junctions import JunctionGeometry, junction_net_id
from scgsim.geometry.models.refusals import Declared, internal


def partition_junctions(
    build_input: GeometryBuildInput,
    junctions: Sequence[JunctionGeometry],
    step: float,
    *,
    output_format: JunctionPartitionFormat = DEFAULT_FORMAT,
):
    """Derive terminal-owned ends and a complete central rectangle once, on the input's declared grid."""
    # An input without a declared grid is refused before anything else.
    geometry_grid_context(build_input)
    if not junctions:
        return build_input
    if output_format.metadata_key in build_input.metadata:
        raise ValueError(
            Declared(
                "the junction input",
                "the call: pass the authored input, before any partition",
            ).refusal(
                "junction input is already partitioned; use immutable authored input"
            )
        )
    return _JunctionPartitionPlanner(
        build_input, junctions, step, output_format
    ).build()


@dataclass(frozen=True)
class _EndSide:
    """One end of a junction: its side, contact bound, own and foreign metal and side meetings."""

    left: bool
    bound: int | Fraction
    own: list[tuple]
    foreign: list[tuple]
    outline: BuiltOutline


@dataclass
class _JunctionPartitionPlanner:
    """Carry authored source, local cut state and ordered partition records."""

    build_input: GeometryBuildInput
    junctions: Sequence[JunctionGeometry]
    step: float
    output_format: JunctionPartitionFormat
    grid: GeometryGridContext = field(init=False)
    grid_um: float | None = field(init=False)
    selection: GridScope = field(init=False)
    originals: dict[str, LayoutPolygonSpec] = field(init=False)
    authored_ids: set[str] = field(init=False)
    entities: dict[str, SemanticEntitySpec] = field(init=False)
    records: dict[str, dict[str, Any]] = field(init=False)
    junction: JunctionGeometry = field(init=False)
    support_plane: dict[str, Any] = field(init=False)
    sheet: LayoutPolygonSpec = field(init=False)
    lattice: LatticeRectangle = field(init=False)
    cuts: CutLattice = field(init=False)
    modelled_sheets: dict[str, LayoutPolygonSpec] = field(init=False)
    dx: float = field(init=False)
    dy: float = field(init=False)
    px: float = field(init=False)
    py: float = field(init=False)
    origin: tuple[float, float] = field(init=False)
    span: int = field(init=False)
    width: int = field(init=False)
    deviations: list[float] = field(init=False)
    frame: JunctionLocalFrame = field(init=False)
    where: JunctionPlaces = field(init=False)
    by_entity: dict[str, list[tuple]] = field(init=False)
    own_a: list[tuple] = field(init=False)
    own_b: list[tuple] = field(init=False)
    other: list[tuple] = field(init=False)
    outline_a: BuiltOutline = field(init=False)
    outline_b: BuiltOutline = field(init=False)
    lower: int | Fraction = field(init=False)
    upper: int | Fraction = field(init=False)
    sheet_region: Sequence[PolygonCoordinates] = field(init=False)
    free: Sequence[PolygonCoordinates] = field(init=False)
    choices: list[tuple] = field(init=False)
    checks: list[dict[str, Any]] = field(init=False)
    a: int = field(init=False)
    b: int = field(init=False)
    end_a: list[tuple] = field(init=False)
    end_b: list[tuple] = field(init=False)
    ka: int = field(init=False)
    kb: int = field(init=False)
    central: tuple[tuple[tuple[tuple[int, int], ...], ...], ...] = field(init=False)
    end_records: list[dict[str, Any]] = field(init=False)
    model: dict[str, LayoutPolygonSpec] = field(init=False)
    derived_entities: tuple[SemanticEntitySpec, ...] = field(init=False)
    result: GeometryBuildInput = field(init=False)

    def build(self) -> GeometryBuildInput:
        """Select every junction before shared model emission and grid realization."""
        self.initialize()
        for junction in self.junctions:
            self.junction = junction
            self.plan_one()
        self.emit_model()
        self.result = assembled_input(
            self.build_input,
            self.model,
            self.derived_entities,
            self.records,
            self.output_format,
        )
        self.realize_grid()
        return self.result

    def plan_one(self) -> None:
        """Select one junction's cuts in its local frame, then record them."""
        self.read_source_frame()
        self.frame = JunctionLocalFrame(
            self.dx, self.dy, self.px, self.py, self.origin, self.deviations
        )
        self.cuts = CutLattice(self.lattice, self.frame, self.step)
        self.where = JunctionPlaces(
            self.frame,
            self.grid_um,
            self.width,
            self.junction.junction_id,
            self.sheet.metadata.get("source_name"),
            self.sheet.layer,
        )
        self.selection = self.selection_scope()
        self.read_conductors()
        self.read_outline_meetings()
        self.find_corridor()
        self.check_cut_room()
        self.choose_ends()
        self.check_partition()
        self.record_ends()
        self.record_junction()
        self.model_sheet()

    def selection_scope(self) -> GridScope:
        """The grid the local selection's Booleans run on.

        Along x or y the world grid lines are local ones, so the Booleans run in the polygon
        world frame and snap as SGB will.  In any other direction no local line is a grid line:
        snapping there would move every cut edge and every side crossing off its line, so the
        Booleans run at the local Boolean precision instead, and the lattice keeps the cut ends
        and the central corners on grid points.
        """
        if not self.lattice.unit:
            return UNGRIDDED
        a, b, c, d = self.dx, self.px, self.dy, self.py
        x, y = self.origin
        return GridScope(self.grid).placed((a, b, c, d, x * a + y * b, x * c + y * d))

    def initialize(self):
        """Index immutable authored polygons, source Entities and junction identities."""
        self.grid = geometry_grid_context(self.build_input)
        self.grid_um = self.grid.grid_size_um
        self.originals = {p.polygon_id: p for p in self.build_input.polygons}
        # A port sheet keeps its metadata, where the grid realization records it as drawn.
        self.originals.update(
            {
                r.source_polygon_id: LayoutPolygonSpec(
                    r.source_polygon_id,
                    r.source_layer,
                    r.exterior,
                    r.holes,
                    metadata=r.metadata,
                )
                for r in self.build_input.port_sheet_regions
            }
        )
        self.authored_ids = {j.source_polygon_id for j in self.junctions}
        self.entities = {e.semantic_id: e for e in self.build_input.entities}
        self.records = {}
        self.modelled_sheets = {}

    def read_source_frame(self):
        """Resolve support, direction and the sheet's lattice rectangle and extent.

        The extent is that of the lattice rectangle's four corners; along x or y on the grid
        they are the drawn sheet's extremes, as the extent always was.
        """
        self.support_plane = junction_contact_inventory(
            self.build_input,
            JunctionContactRequest(
                self.junction.source_polygon_id,
                self.junction.terminal_a_net,
                self.junction.terminal_b_net,
                self.junction.metadata.get("junction_support_plane"),
                support_schema=self.output_format.support_schema,
            ),
        )
        self.sheet = self.originals[self.junction.source_polygon_id]
        self.lattice = lattice_rectangle(
            self.sheet.exterior,
            self.step,
            self.junction.direction_xy,
            drawn=drawn_exterior(self.sheet),
            port=f"junction {self.junction.junction_id!r}",
            layer=self.sheet.layer,
        )
        corners = self.lattice.corners_um
        # The declared direction must be the lattice rectangle's, within the shared rectangle check.
        junction_terminal_line(
            {"exterior": corners, "holes": self.sheet.holes}, self.junction, 0.0
        )
        self.dx, self.dy = self.junction.direction_xy
        self.px, self.py = (-self.dy, self.dx)
        along = [x * self.dx + y * self.dy for x, y in corners]
        across = [x * self.px + y * self.py for x, y in corners]
        self.origin = (min(along), min(across))
        self.span = round((max(along) - self.origin[0]) / PRECISION_UM)
        self.width = round((max(across) - self.origin[1]) / PRECISION_UM)
        if self.span <= 0 or self.width <= 0:
            raise ValueError(
                self.junction_refusal(
                    "junction authored geometry collapses at Boolean precision"
                )
            )
        self.deviations = []

    def read_conductors(self):
        """Convert eligible source conductors and resolve both explicit terminal nets."""
        self.by_entity = {
            e.semantic_id: [
                self.frame.convert(self.originals[p])
                for p in e.polygon_ids
                if p not in self.authored_ids
            ]
            for e in self.entities.values()
            if e.semantic_id in self.support_plane["eligible_entity_ids"]
        }
        by_net = defaultdict(list)
        for eid, polygons in self.by_entity.items():
            by_net[junction_net_id(self.entities[eid])].extend(polygons)
        self.own_a, self.own_b = (
            by_net[self.junction.terminal_a_net],
            by_net[self.junction.terminal_b_net],
        )
        if not self.own_a or not self.own_b:
            raise ValueError(
                self.junction_refusal(
                    "junction requires both explicit terminal conductor geometries: a terminal Net draws no conductor in the sheet's support plane"
                )
            )
        self.other = [
            p
            for net, polygons in by_net.items()
            if net not in {self.junction.terminal_a_net, self.junction.terminal_b_net}
            for p in polygons
        ]

    def read_outline_meetings(self):
        """Find where each terminal's built outline meets the two sheet sides."""
        self.outline_a, self.outline_b = (
            built_outline(
                self.build_input,
                (
                    eid
                    for eid in self.by_entity
                    if junction_net_id(self.entities[eid]) == net
                ),
                self.frame,
                self.span,
                self.width,
            )
            for net in (self.junction.terminal_a_net, self.junction.terminal_b_net)
        )

    def find_corridor(self):
        """Verify terminal ordering and complete clear-corridor containment."""
        terminals = (self.junction.terminal_a_net, self.junction.terminal_b_net)
        if contacts(self.selection, self.other, self.span, self.width) is not None:
            raise ValueError(
                self.where.refusal(
                    "third-net closed contact with authored junction sheet",
                    self.conductor_contacts(lambda net: net not in terminals),
                    "Fix the layout: keep the sheet clear of every conductor but the two "
                    "it lands on.",
                )
            )
        ca, cb = (
            contacts(self.selection, self.own_a, self.span, self.width),
            contacts(self.selection, self.own_b, self.span, self.width),
        )
        if ca is None or cb is None:
            raise ValueError(
                self.junction_refusal(
                    "unsupported junction contact ordering or clear corridor: each terminal must meet the sheet from its own end"
                )
            )
        # An exact arc can reach farther into the sheet than the drawn polygon's chords; the
        # cuts lie beyond it, so the central rectangle stays clear of the conductor SGB builds.
        lower, upper = ca[1], cb[0]
        if self.outline_a.arc_reach is not None:
            lower = max(lower, self.outline_a.arc_reach[1])
        if self.outline_b.arc_reach is not None:
            upper = min(upper, self.outline_b.arc_reach[0])
        if lower >= upper:
            raise ValueError(self.no_gap_refusal(lower, upper))
        self.lower, self.upper = (lower, upper)
        self.check_corridor_sides()
        self.check_corridor_clear()

    def check_corridor_sides(self) -> None:
        """Refuse metal that touches the sheet's long sides between the two cut bounds."""
        for polygons in (self.own_a, self.own_b, self.other):
            for rings in polygons:
                for y in (0, self.width):
                    intervals, points = section(rings, 1, y, self.lower, self.upper)
                    if any((a < b for a, b in intervals)) or any(
                        (self.lower < v < self.upper for v in points)
                    ):
                        raise ValueError(
                            self.junction_refusal(
                                "unsupported corridor side-boundary metal contact: metal touches the sheet's long sides"
                            )
                        )

    def check_corridor_clear(self) -> None:
        """Refuse a corridor between the bounds that is not wholly free of both terminals."""
        self.sheet_region = build_polygon_region(
            self.selection, rectangle(0, 0, self.span, self.width)
        )
        self.free = boolean(
            self.selection,
            self.sheet_region,
            build_polygon_region(self.selection, [*self.own_a, *self.own_b]),
            "not",
        )
        corridor = build_polygon_region(
            self.selection, rectangle(self.lower, 0, self.upper, self.width)
        )
        if boolean(self.selection, corridor, self.free, "not"):
            raise ValueError(
                self.junction_refusal(
                    "unsupported corridor containment; no general search performed: the terminals leave no clear corridor across the sheet"
                )
            )

    def check_cut_room(self) -> None:
        """Refuse, located, a corridor that holds fewer than two cut positions.

        A cut lies only every |v| grid steps along the port, so the clear gap between the two
        conductors must hold two of them for the central rectangle to have a length.
        """
        first = self.cuts.candidates(self.lower, left=True)[0]
        if self.cuts.position(first + 1)[1] <= self.upper:
            return
        grid_um = self.grid_um
        p, q = self.lattice.direction
        raise ValueError(
            grid_change_message(
                f"junction {self.junction.junction_id!r} has no room for a rectangle "
                "between its conductors",
                [self.sheet.layer],
                grid_um,
                [
                    f"its sheet runs along v = ({p}, {q}), where a cut lies only every "
                    f"{format_length(self.lattice.spacing_um, grid_um)} along it; "
                    f"{self.where.gap(self.lower, self.upper)}, and holds fewer than two such cuts"
                ],
                "Fix the layout: lengthen the gap the sheet crosses, or draw the sheet "
                "along x or y or along a direction of fewer grid steps, such as 45 degrees.",
            )
        )

    def choose_ends(self):
        """Select each end's grid cut and the central span between them.

        On each side the cut starts at the first verified grid candidate beside the
        terminal contact (the largest rectangle) and moves one grid step into the gap until
        the edge its end piece adds along each sheet side is zero or at least the class A
        wall-column limit, so the terminal the end piece joins keeps no wall shorter than
        that limit.
        """
        self.choices = []
        self.checks = []
        limit = round(WALL_COLUMN_LIMIT_STEPS * self.step / PRECISION_UM)
        for side in (
            _EndSide(
                True, self.lower, self.own_a, self.own_b + self.other, self.outline_a
            ),
            _EndSide(
                False, self.upper, self.own_b, self.own_a + self.other, self.outline_b
            ),
        ):
            first = self.first_valid_cut(side)
            self.choices.append(self.step_clear_cut(side, first, limit))
        (self.a, self.end_a, self.ka), (self.b, self.end_b, self.kb) = self.choices
        if self.a >= self.b:
            raise ValueError(self.no_span_refusal())
        self.central = rectangle(self.a, 0, self.b, self.width)

    def first_valid_cut(self, side: _EndSide):
        """The first grid candidate beside ``bound`` whose end piece passes every topology check."""
        left, bound = side.left, side.bound
        refused = []
        for index in self.cuts.candidates(bound, left=left):
            _, cut = self.cuts.position(index)
            interior = self.cuts.beyond(index, bound, left=left)
            if interior and cut == bound:
                raise ValueError(
                    internal(
                        "numerical junction geometry failure: interior grid cut collapses to boundary"
                    )
                )
            if (
                cut < self.lower
                or cut > self.upper
                or (interior and (cut <= self.lower or cut >= self.upper))
            ):
                continue
            end = self.check_end(side, index)
            if end is not None:
                return (cut, end, index)
            refused.append((cut, self.checks[-1]["reason"]))
        raise ValueError(
            self.where.refusal(
                "unsupported junction end topology on verified corridor/grid",
                [self.end_finding(side, cut, reason) for cut, reason in refused],
                "Fix the layout: draw the sheet's end so that beyond the cut it is one "
                "piece that meets its own conductor and no other.",
            )
        )

    def step_clear_cut(self, side: _EndSide, first, limit):
        """Move a valid cut into the gap, one grid step at a time, until its side steps pass.

        Inside the verified corridor the end piece only gains a clear rectangle, so it keeps
        the topology ``first`` was accepted with; each step is checked again all the same.
        """
        cut, end, index = first
        left = side.left
        while not all(
            step == 0 or step >= limit
            for step in side_steps(side.outline.meetings, cut, self.span, left=left)
        ):
            index += 1 if left else -1
            cut = self.cuts.position(index)[1]
            if (cut >= self.upper) if left else (cut <= self.lower):
                raise ValueError(self.short_steps_refusal(side, first[0]))
            end = self.check_end(side, index)
            if end is None:
                raise ValueError(
                    self.junction_refusal(
                        "unsupported junction end topology on verified corridor/grid"
                    )
                )
        return (cut, end, index)

    def check_end(self, side: _EndSide, index):
        """Check the end piece of one grid cut and keep the evidence, with its side steps."""
        requested, cut = self.cuts.position(index)
        left = side.left
        end, evidence = end_candidate(
            self.selection,
            self.end_source(side),
            side.own,
            side.foreign,
            EndSection(cut, self.span, self.width, left=left),
        )
        if end is not None:
            evidence["side_steps_um"] = [
                step * PRECISION_UM
                for step in side_steps(side.outline.meetings, cut, self.span, left=left)
            ]
        self.checks.append(
            {
                "side": "A" if left else "B",
                "index": index,
                "requested_s_um": requested,
                "actual_s_um": cut * PRECISION_UM,
                "deviation_um": cut * PRECISION_UM - requested,
                **evidence,
            }
        )
        return end

    def check_partition(self):
        """Check closed endpoint contacts, open sides and that the three pieces partition the sheet.

        The two end pieces and the central rectangle have disjoint interiors, cover the free
        part of the sheet between the two terminals, and lie within the drawn sheet.
        """
        for y in (0, self.width):
            for rings in [
                *self.own_a,
                *self.own_b,
                *self.other,
                *self.end_a,
                *self.end_b,
            ]:
                intervals, points = section(rings, 1, y, self.a, self.b)
                if any((x < z for x, z in intervals)) or any(
                    (self.a < v < self.b for v in points)
                ):
                    raise ValueError(
                        self.junction_refusal(
                            "RLC open side has forbidden line or point contact"
                        )
                    )
        for cut, wrong in (
            (self.a, self.own_b + self.other + self.end_b),
            (self.b, self.own_a + self.other + self.end_a),
        ):
            if any((any(section(rings, 0, cut, 0, self.width)) for rings in wrong)):
                raise ValueError(
                    self.junction_refusal("wrong-net RLC end/corner contact")
                )
        parts = [
            build_polygon_region(self.selection, self.end_a),
            build_polygon_region(self.selection, self.central),
            build_polygon_region(self.selection, self.end_b),
        ]
        union = boolean(self.selection, [p for part in parts for p in part], [], "or")
        if (
            boolean(self.selection, self.free, union, "not")
            or boolean(self.selection, union, self.sheet_region, "not")
        ) or any(
            (
                boolean(self.selection, parts[i], parts[j], "and")
                for i, j in ((0, 1), (0, 2), (1, 2))
            )
        ):
            raise ValueError(
                internal(
                    "numerical junction geometry failure: partition coverage/interiors differ"
                )
            )

    def record_ends(self):
        """Bind each end piece to its unique source Entity/material lineage."""
        self.end_records = []
        for label, end, outline, cut, net in (
            ("A", self.end_a, self.outline_a, self.a, self.junction.terminal_a_net),
            ("B", self.end_b, self.outline_b, self.b, self.junction.terminal_b_net),
        ):
            owners = [
                eid
                for eid, polygons in self.by_entity.items()
                if junction_net_id(self.entities[eid]) == net
                and (
                    own_contact(self.selection, end, polygons)
                    if end
                    else covered(polygons, 0, cut, 0, self.width)
                )
            ]
            if len(owners) != 1:
                raise ValueError(self.lineage_refusal(end, cut, net, owners))
            owner = self.entities[owners[0]]
            polygon_id = (
                f"{self.output_format.id_prefix}_{canonical_sha256([self.junction.junction_id, label, owner.semantic_id])[:24]}"
                if end
                else None
            )
            self.end_records.append(
                {
                    "side": label,
                    "source_entity_id": owner.semantic_id,
                    "net_id": net,
                    "material_id": owner.material_id,
                    "geometry": plain_record(owner.geometry),
                    "representation": dict(owner.route_representations),
                    "polygon_id": polygon_id,
                    "empty_residual": not end,
                    "landing": outline.rebuilt_arcs,
                    "local_regions": plain_record(end),
                }
            )

    def record_junction(self):
        """Store the selected source-bound geometry and existing proof records."""
        cid = f"{self.output_format.id_prefix}_{canonical_sha256([self.junction.junction_id, 'central'])[:24]}"
        self.records[self.junction.junction_id] = {
            "method": self.output_format.method,
            "support_plane": self.support_plane,
            "z_um": self.support_plane["effective_z_um"],
            "authored_polygon_id": self.junction.source_polygon_id,
            "central_polygon_id": cid,
            "direction_xy": [self.dx, self.dy],
            "local_origin_projection_um": list(self.origin),
            "source_dbu_um": self.step,
            "boolean_precision_um": PRECISION_UM,
            "span_um": self.span * PRECISION_UM,
            "width_um": self.width * PRECISION_UM,
            "corridor_um": [self.lower * PRECISION_UM, self.upper * PRECISION_UM],
            "cut_indices": [self.ka, self.kb],
            "terminal_step_limit_um": WALL_COLUMN_LIMIT_STEPS * self.step,
            "actual_cuts_um": [self.a * PRECISION_UM, self.b * PRECISION_UM],
            "candidate_checks": self.checks,
            "candidate_count": len(self.checks),
            "proof_conditions": {
                "full_width_corridor_containment": True,
                "closed_corridor_side_clearance": True,
                "terminal_contact_ordering": "A_left_B_right",
                "partition_coverage_and_disjoint_interiors": True,
            },
            "maximum_input_coordinate_deviation_um": max(self.deviations, default=0.0),
            "ends": self.end_records,
            "free_local_regions": plain_record(read_boolean_rings(self.free)),
            "central_local_regions": plain_record(self.central),
            "lattice": {
                **self.lattice.record(),
                "cuts": [
                    self.lattice.cut_record(index) for index in (self.ka, self.kb)
                ],
            },
        }

    def model_sheet(self) -> None:
        """Keep the sheet the model partitions: straight along its lattice lines between the cuts."""
        exterior = self.lattice.modelled_sheet((self.ka, self.kb))
        if exterior is not None:
            self.modelled_sheets[self.sheet.polygon_id] = replace(
                self.sheet, exterior=exterior
            )

    def emit_model(self):
        """Emit registered model coordinates only after all local selections.

        The model partitions each sheet as ``model_sheet`` kept it; the records keep the sheet
        as drawn, with the modelled one beside it.
        """
        self.model, self.derived_entities = model_geometry(
            self.build_input,
            {**self.originals, **self.modelled_sheets},
            self.records,
            self.output_format,
        )
        for record in self.records.values():
            sheet_id = record["authored_polygon_id"]
            if sheet_id in self.modelled_sheets:
                drawn = self.originals[sheet_id]
                record["lattice"]["modelled_sheet_um"] = plain_record(
                    self.modelled_sheets[sheet_id].exterior
                )
                record["authored_source_geometry"] = {
                    "exterior": plain_record(drawn.exterior),
                    "holes": plain_record(drawn.holes),
                }

    def realize_grid(self):
        """Realize an existing input grid and retain exact shape failure context."""
        self.result = realized_on_grid(
            self.build_input, self.result, self.junctions, self.records
        )

    def junction_refusal(self, problem: str) -> str:
        """The refusal of the junction being planned: its id, where its sheet lies, and who changes it."""
        sheet = getattr(self, "sheet", None)
        where = (
            ""
            if sheet is None
            else f" (its sheet lies at {format_extent(sheet.exterior, self.grid_um)})"
        )
        return Declared(
            f"junction {self.junction.junction_id!r}{where}",
            "the layout: draw the sheet and its two terminals as the rectangular "
            "junction rule needs",
        ).refusal(problem)

    def net_of(self, side: _EndSide) -> str:
        """The Net of the terminal an end is cut beside."""
        return (
            self.junction.terminal_a_net if side.left else self.junction.terminal_b_net
        )

    def end_source(self, side: _EndSide):
        """What an end piece is cut from: the drawn sheet where SGB rebuilds the terminal's arcs, else its free part."""
        return self.sheet_region if side.outline.rebuilt_arcs else self.free

    # -- located refusals: where in the layout (``self.where``) and who acts --

    def conductor(self, entity_id: str) -> str:
        """A conductor Entity with the source layers of its polygons."""
        polygons = self.entities[entity_id].polygon_ids
        layers = sorted(
            {self.originals[p].layer for p in polygons if p in self.originals}
        )
        return f"{entity_id} on layer {', '.join(layers)}" if layers else entity_id

    def conductor_contacts(self, of_net) -> list[str]:
        """Where each eligible conductor of a Net that ``of_net`` accepts touches or overlaps the closed sheet."""
        found = []
        for entity_id, polygons in self.by_entity.items():
            net = junction_net_id(self.entities[entity_id])
            points = (
                contact_points(self.selection, polygons, self.span, self.width)
                if of_net(net)
                else []
            )
            if points:
                found.append(
                    f"{self.conductor(entity_id)}, of Net {net}, meets the sheet at "
                    f"{self.where.place(points)}"
                )
        return found

    def no_gap_refusal(self, lower, upper) -> str:
        """The terminals' contacts with the sheet overlap along the port, so no clear gap lies between them."""
        a, b = self.junction.terminal_a_net, self.junction.terminal_b_net
        arcs = (
            self.outline_a.arc_reach is not None or self.outline_b.arc_reach is not None
        )
        return self.where.refusal(
            "unsupported junction contact ordering or clear corridor",
            [
                *self.conductor_contacts(lambda net: net in (a, b)),
                f"the conductors of Net {a}{', with the exact arcs SGB builds,' if arcs else ''} "
                f"reach along the sheet to {self.where.centre(lower)}, past where those "
                f"of Net {b} begin, {self.where.centre(upper)}",
            ],
            "Fix the layout: draw the sheet so that it crosses one clear gap from one of its "
            "conductors to the other, and touches each only at its own end.",
        )

    def short_steps_refusal(self, side: _EndSide, first) -> str:
        """No cut beside one terminal leaves its edges along the sheet sides zero or at least the limit."""
        starts = nearest_meetings(
            side.outline.meetings, first, self.span, left=side.left
        )
        return self.where.refusal(
            "junction corridor has no cut whose terminal steps are zero or at least "
            f"{WALL_COLUMN_LIMIT_STEPS} grid steps",
            [
                f"{self.where.gap(self.lower, self.upper)}; beside Net {self.net_of(side)} the end piece runs along the sheet "
                f"sides from {', '.join(self.where.place([start]) for start in starts)}, where "
                "the conductor's outline meets them, and every cut from "
                f"{self.where.centre(first)} across the gap leaves such an edge "
                f"shorter than {WALL_COLUMN_LIMIT_STEPS} grid steps"
            ],
            "Fix the layout: lengthen the gap the sheet crosses, or give the conductor a "
            "straight edge across the sheet.",
        )

    def no_span_refusal(self) -> str:
        """Each end's first cut whose steps pass lies past the other's, so no rectangle is left."""
        a, b = self.junction.terminal_a_net, self.junction.terminal_b_net
        return self.where.refusal(
            "junction DBU grid has no positive central span",
            [
                f"{self.where.gap(self.lower, self.upper)}; the first cut beside Net {a} whose edges along the sheet "
                f"sides are zero or at least {WALL_COLUMN_LIMIT_STEPS} grid steps long lies "
                f"at {self.where.centre(self.a)}, and the first beside Net {b} at "
                f"{self.where.centre(self.b)}, so no rectangle is left between them"
            ],
            "Fix the layout: lengthen the gap the sheet crosses, or give the conductors "
            "straight edges across the sheet.",
        )

    def end_finding(self, side: _EndSide, cut, reason: str) -> str:
        """Why the end piece beyond one candidate cut was refused, and where its pieces lie."""
        section = EndSection(cut, self.span, self.width, left=side.left)
        pieces = [
            self.where.place(
                [(x / PRECISION_UM, y / PRECISION_UM) for x, y in piece.points]
            )
            for piece in end_region(self.selection, self.end_source(side), section)
        ]
        words = END_REFUSALS.get(reason, "has parts that meet only at a point")
        where = f", at {', '.join(pieces)}" if pieces else ""
        return (
            f"beside Net {self.net_of(side)}, the end piece beyond the cut at "
            f"{self.where.centre(cut)} {words}{where}"
        )

    def lineage_refusal(self, end, cut, net: str, owners: list[str]) -> str:
        """An end piece that no one conductor of its Net owns, or that several do."""
        names = ", ".join(map(self.conductor, owners)) or "none of them alone"
        # An empty end piece is the cut itself, which its conductor must cover alone.
        points = [p for rings in end for ring in rings for p in ring]
        what = "its end piece at {} meets" if end else "its cut at {} is covered by"
        where = what.format(self.where.place(points or [(cut, 0), (cut, self.width)]))
        return self.where.refusal(
            "junction end lacks an unambiguous source Entity/material/Z lineage",
            [f"{where} these conductors of Net {net}: {names}"],
            "Fix the layout: land each end of the sheet on one conductor, or draw the "
            "conductors it lands on there as one.",
        )
