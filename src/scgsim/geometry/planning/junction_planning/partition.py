"""Compose authored junction cuts, model emission and optional grid realization."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import Any

from scgsim.geometry.models.grid import (
    WALL_COLUMN_LIMIT_STEPS,
    format_length,
    format_point,
    geometry_grid_context,
)
from scgsim.geometry.models.input import (
    GeometryBuildInput,
    LayoutPolygonSpec,
    SemanticEntitySpec,
)
from scgsim.geometry.models.junctions import JunctionGeometry, junction_net_id
from scgsim.geometry.planning.junction_planning.boolean_geometry import (
    boolean,
    build_polygon_region,
    read_boolean_rings,
    rectangle,
)
from scgsim.geometry.planning.junction_planning.contact import (
    JunctionContactRequest,
    junction_contact_inventory,
)
from scgsim.geometry.planning.junction_planning.core import (
    DEFAULT_FORMAT,
    PRECISION_UM,
    JunctionPartitionFormat,
    canonical_sha256,
    plain_record,
)
from scgsim.geometry.planning.junction_planning.exact_geometry import (
    covered,
    cross,
    section,
)
from scgsim.geometry.planning.junction_planning.lattice import (
    CutLattice,
    LatticeRectangle,
    drawn_exterior,
    lattice_rectangle,
)
from scgsim.geometry.planning.junction_planning.local_frame import JunctionLocalFrame
from scgsim.geometry.planning.junction_planning.model import model_geometry
from scgsim.geometry.planning.junction_planning.model_checks import (
    emitted_partition_coverage,
)
from scgsim.geometry.planning.junction_planning.selection import (
    BuiltOutline,
    EndSection,
    built_outline,
    contacts,
    end_candidate,
    own_contact,
    side_steps,
)
from scgsim.geometry.planning.junction_planning.terminal import junction_terminal_line
from scgsim.geometry.primitives.grid import (
    geometry_grid_scope,
    polygon_world_frame,
    with_input_grid,
)
from scgsim.geometry.primitives.grid_regions import grid_change_message
from scgsim.geometry.primitives.planar_loops import PolygonCoordinates


@with_input_grid
def partition_junctions(
    build_input: GeometryBuildInput,
    junctions: Sequence[JunctionGeometry],
    step: float,
    *,
    output_format: JunctionPartitionFormat = DEFAULT_FORMAT,
):
    """Derive terminal-owned ends and a complete central rectangle once."""
    if not junctions:
        return build_input
    if output_format.metadata_key in build_input.metadata:
        raise ValueError(
            "junction input is already partitioned; use immutable authored input"
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
        self.assemble_output()
        self.realize_grid()
        return self.result

    def plan_one(self) -> None:
        """Select one junction's cuts in its local frame, then record them."""
        self.read_source_frame()
        self.frame = JunctionLocalFrame(
            self.dx, self.dy, self.px, self.py, self.origin, self.deviations
        )
        self.cuts = CutLattice(self.lattice, self.frame, self.step)
        with self.selection_grid():
            self.read_conductors()
            self.read_outline_meetings()
            self.find_corridor()
            self.check_cut_room()
            self.choose_ends()
            self.check_partition()
            self.record_ends()
            self.record_junction()
        self.model_sheet()

    def selection_grid(self):
        """The grid the local selection's Booleans run on.

        Along x or y the world grid lines are local ones, so the Booleans run in the polygon
        world frame and snap as SGB will.  In any other direction no local line is a grid line:
        snapping there would move every cut edge and every side crossing off its line, so the
        Booleans run at the local Boolean precision instead, and the lattice keeps the cut ends
        and the central corners on grid points.
        """
        if not self.lattice.unit:
            return geometry_grid_scope(None)
        return polygon_world_frame(
            (
                self.dx,
                self.px,
                self.dy,
                self.py,
                self.origin[0] * self.dx + self.origin[1] * self.px,
                self.origin[0] * self.dy + self.origin[1] * self.py,
            )
        )

    def initialize(self):
        """Index immutable authored polygons, source Entities and junction identities."""
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
                "junction authored geometry collapses at Boolean precision"
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
                "junction requires both explicit terminal conductor geometries"
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
        if contacts(self.other, self.span, self.width) is not None:
            raise ValueError("third-net closed contact with authored junction sheet")
        ca, cb = (
            contacts(self.own_a, self.span, self.width),
            contacts(self.own_b, self.span, self.width),
        )
        if ca is None or cb is None:
            raise ValueError("unsupported junction contact ordering or clear corridor")
        # An exact arc can reach farther into the sheet than the drawn polygon's chords; the
        # cuts lie beyond it, so the central rectangle stays clear of the conductor SGB builds.
        lower, upper = ca[1], cb[0]
        if self.outline_a.arc_reach is not None:
            lower = max(lower, self.outline_a.arc_reach[1])
        if self.outline_b.arc_reach is not None:
            upper = min(upper, self.outline_b.arc_reach[0])
        if lower >= upper:
            raise ValueError("unsupported junction contact ordering or clear corridor")
        self.lower, self.upper = (lower, upper)
        for polygons in (self.own_a, self.own_b, self.other):
            for rings in polygons:
                for y in (0, self.width):
                    intervals, points = section(rings, 1, y, self.lower, self.upper)
                    if any((a < b for a, b in intervals)) or any(
                        (self.lower < v < self.upper for v in points)
                    ):
                        raise ValueError(
                            "unsupported corridor side-boundary metal contact"
                        )
        self.sheet_region = build_polygon_region(rectangle(0, 0, self.span, self.width))
        self.free = boolean(
            self.sheet_region, build_polygon_region([*self.own_a, *self.own_b]), "not"
        )
        corridor = build_polygon_region(
            rectangle(self.lower, 0, self.upper, self.width)
        )
        if boolean(corridor, self.free, "not"):
            raise ValueError(
                "unsupported corridor containment; no general search performed"
            )

    def check_cut_room(self) -> None:
        """Refuse, located, a corridor that holds fewer than two cut positions.

        A cut lies only every |v| grid steps along the port, so the clear gap between the two
        conductors must hold two of them for the central rectangle to have a length.
        """
        first = self.cuts.candidates(self.lower, left=True)[0]
        if self.cuts.position(first + 1)[1] <= self.upper:
            return
        grid = geometry_grid_context(self.build_input)
        grid_um = None if grid is None else grid.grid_size_um
        p, q = self.lattice.direction
        start, end = (self.centre_line_point(u) for u in (self.lower, self.upper))
        raise ValueError(
            grid_change_message(
                f"junction {self.junction.junction_id!r} has no room for a rectangle "
                "between its conductors",
                [self.sheet.layer],
                grid_um,
                [
                    f"its sheet runs along v = ({p}, {q}), where a cut lies only every "
                    f"{format_length(self.lattice.spacing_um, grid_um)} along it; the "
                    f"clear gap between its conductors runs from {start} to {end}, "
                    f"{format_length((self.upper - self.lower) * PRECISION_UM, grid_um)} "
                    "long, and holds fewer than two such cuts"
                ],
                "Fix the layout: lengthen the gap the sheet crosses, or draw the sheet "
                "along x or y or along a direction of fewer grid steps, such as 45 degrees.",
            )
        )

    def centre_line_point(self, u) -> str:
        """Where the sheet's centre line lies ``u`` Boolean units along the port, in the layout."""
        along = self.origin[0] + float(u) * PRECISION_UM
        across = self.origin[1] + self.width * PRECISION_UM / 2
        grid = geometry_grid_context(self.build_input)
        return format_point(
            along * self.dx + across * self.px,
            along * self.dy + across * self.py,
            None if grid is None else grid.grid_size_um,
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
            raise ValueError("junction DBU grid has no positive central span")
        self.central = rectangle(self.a, 0, self.b, self.width)

    def first_valid_cut(self, side: _EndSide):
        """The first grid candidate beside ``bound`` whose end piece passes every topology check."""
        left, bound = side.left, side.bound
        for index in self.cuts.candidates(bound, left=left):
            _, cut = self.cuts.position(index)
            interior = self.cuts.beyond(index, bound, left=left)
            if interior and cut == bound:
                raise ValueError(
                    "numerical junction geometry failure: interior grid cut collapses to boundary"
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
        raise ValueError("unsupported junction end topology on verified corridor/grid")

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
                raise ValueError(
                    "junction corridor has no cut whose terminal steps are zero or at "
                    f"least {WALL_COLUMN_LIMIT_STEPS} grid steps"
                )
            end = self.check_end(side, index)
            if end is None:
                raise ValueError(
                    "unsupported junction end topology on verified corridor/grid"
                )
        return (cut, end, index)

    def check_end(self, side: _EndSide, index):
        """Check the end piece of one grid cut and keep the evidence, with its side steps."""
        requested, cut = self.cuts.position(index)
        left = side.left
        end, evidence = end_candidate(
            self.sheet_region if side.outline.rebuilt_arcs else self.free,
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
                        "RLC open side has forbidden line or point contact"
                    )
        for cut, wrong in (
            (self.a, self.own_b + self.other + self.end_b),
            (self.b, self.own_a + self.other + self.end_a),
        ):
            if any((any(section(rings, 0, cut, 0, self.width)) for rings in wrong)):
                raise ValueError("wrong-net RLC end/corner contact")
        parts = [
            build_polygon_region(self.end_a),
            build_polygon_region(self.central),
            build_polygon_region(self.end_b),
        ]
        union = boolean([p for part in parts for p in part], [], "or")
        if (
            boolean(self.free, union, "not") or boolean(union, self.sheet_region, "not")
        ) or any(
            (boolean(parts[i], parts[j], "and") for i, j in ((0, 1), (0, 2), (1, 2)))
        ):
            raise ValueError(
                "numerical junction geometry failure: partition coverage/interiors differ"
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
                    own_contact(end, polygons)
                    if end
                    else covered(polygons, 0, cut, 0, self.width)
                )
            ]
            if len(owners) != 1:
                raise ValueError(
                    "junction end lacks an unambiguous source Entity/material/Z lineage"
                )
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

    def assemble_output(self):
        """Assemble derived source records and the authored-input hash."""
        metadata = dict(self.build_input.metadata)
        metadata[self.output_format.source_hash_key] = canonical_sha256(
            plain_record(self.build_input)
        )
        metadata[self.output_format.metadata_key] = self.records
        original_ids = {p.polygon_id for p in self.build_input.polygons}
        derived_polygons = tuple(
            (self.model[p.polygon_id] for p in self.build_input.polygons)
        ) + tuple(
            (
                p
                for pid, p in self.model.items()
                if pid not in original_ids and pid not in self.authored_ids
            )
        )
        regions = tuple(
            (
                replace(
                    r,
                    exterior=self.model[r.source_polygon_id].exterior,
                    holes=self.model[r.source_polygon_id].holes,
                )
                if r.source_polygon_id in self.model
                else r
                for r in self.build_input.port_sheet_regions
            )
        )
        self.result = replace(
            self.build_input,
            polygons=derived_polygons,
            entities=self.derived_entities,
            port_sheet_regions=regions,
            metadata=metadata,
        )

    def realize_grid(self):
        """Realize an existing input grid and retain exact shape failure context."""
        context = geometry_grid_context(self.build_input)
        if context is not None:
            from scgsim.geometry.source.grid import apply_geometry_grid  # noqa: PLC0415 -- retain the conditional grid-realization dependency and its failure boundary.

            self.result = _straight_end_pieces(
                apply_geometry_grid(self.result, context), self.records
            )
            actual = {p.polygon_id: p for p in self.result.polygons}
            for junction in self.junctions:
                record = self.records[junction.junction_id]
                polygon = actual[record["central_polygon_id"]]
                try:
                    junction_terminal_line(
                        {"exterior": polygon.exterior, "holes": polygon.holes},
                        junction,
                        record["z_um"],
                    )
                    ends = [
                        actual[end["polygon_id"]]
                        for end in record["ends"]
                        if end["polygon_id"]
                    ]
                    emitted = [(p.exterior, *p.holes) for p in ends]
                    emitted.append((polygon.exterior, *polygon.holes))
                    if not emitted_partition_coverage(
                        record["partition_model_regions"], emitted
                    ):
                        raise ValueError(
                            "junction grid realization changes partition coverage"
                        )
                except ValueError as error:
                    error.add_note(
                        json.dumps(
                            {
                                "junction_id": junction.junction_id,
                                "geometry_grid": context.to_payload(),
                                "requested_direction_xy": list(junction.direction_xy),
                                "requested_width_um": junction.width_um,
                                "realization_status": "incompatible_with_existing_shape_contract",
                            },
                            sort_keys=True,
                        )
                    )
                    raise
                record["geometry_grid"] = context.to_payload()
                record["requested_central_model_regions"] = record[
                    "central_model_regions"
                ]
                record["central_model_regions"] = [
                    [list(polygon.exterior), *map(list, polygon.holes)]
                ]
                record["realized_central_geometry_sha256"] = canonical_sha256(
                    {"exterior": polygon.exterior, "holes": polygon.holes}
                )
                for end in record["ends"]:
                    if end["polygon_id"]:
                        piece = actual[end["polygon_id"]]
                        end["requested_model_regions"] = end["model_regions"]
                        end["model_regions"] = [
                            [list(piece.exterior), *map(list, piece.holes)]
                        ]


def _straight_end_pieces(result: GeometryBuildInput, records) -> GeometryBuildInput:
    """The grid-realized input with every landing end piece cleared of collinear vertices.

    The grid realization nodes a polygon wherever another polygon's vertex lies on its edge.
    A landing end piece overlaps its terminal, so the drawn terminal's vertices on the sheet
    sides land on the end piece's side edges.  SGB would keep each as a corner between two
    carriers and split the side wall standing there; without them the region is the same.
    """
    ids = {
        end["polygon_id"]
        for record in records.values()
        for end in record["ends"]
        if end["polygon_id"] and end["landing"]
    }
    return replace(
        result,
        polygons=tuple(
            replace(
                polygon,
                exterior=_without_collinear(polygon.exterior),
                holes=tuple(_without_collinear(hole) for hole in polygon.holes),
            )
            if polygon.polygon_id in ids
            else polygon
            for polygon in result.polygons
        ),
    )


def _without_collinear(ring):
    exact = [tuple(Fraction(str(v)) for v in point) for point in ring]
    count = len(ring)
    return tuple(
        ring[n]
        for n in range(count)
        if cross(exact[n - 1], exact[n], exact[(n + 1) % count]) != 0
    )
