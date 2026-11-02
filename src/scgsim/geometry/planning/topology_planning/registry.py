"""Register canonical points, curves, loops and their retained surface claims."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import pairwise
from typing import cast

from scgsim.geometry.models.common import SurfaceLoopRoleLiteral
from scgsim.geometry.models.topology import (
    CurvePlanRecord,
    CurveRefRecord,
    PointPlanRecord,
    SurfaceLoopRecord,
    SurfacePlanRecord,
)
from scgsim.geometry.planning.topology_planning.indexing import (
    axis_aligned_point_index,
    index_points_by_axis,
    segment_candidate_points,
    segment_parameter,
)
from scgsim.geometry.planning.topology_planning.ownership import surface_owner_ids
from scgsim.geometry.planning.topology_planning.rings import (
    ring3d_edges,
    surface_ring3d_specs,
)
from scgsim.geometry.primitives.loops import ordered_loop_signature
from scgsim.geometry.primitives.spatial import coordinate_key


def plan_canonical_topology(
    *, surfaces: tuple[SurfacePlanRecord, ...]
) -> tuple[
    tuple[PointPlanRecord, ...],
    tuple[CurvePlanRecord, ...],
    tuple[SurfaceLoopRecord, ...],
    tuple[SurfacePlanRecord, ...],
]:
    """Canonicalize live surface rings through one ordered topology registry."""
    return _TopologyRegistry(surfaces).build()


@dataclass
class _TopologyRegistry:
    """Carry one ordered registry of canonical geometry and retained ownership."""

    surfaces: tuple[SurfacePlanRecord, ...]
    point_ids: dict[tuple[float, float, float], str] = field(init=False)
    point_coordinates: dict[str, tuple[float, float, float]] = field(init=False)
    point_curve_ids: dict[str, set[str]] = field(init=False)
    curve_ids: dict[tuple[str, str], str] = field(init=False)
    curve_owner_ids: dict[str, set[str]] = field(init=False)
    curve_interface_ids: dict[str, set[str]] = field(init=False)
    curve_surface_ids: dict[str, set[str]] = field(init=False)
    curve_volume_ids: dict[str, set[str]] = field(init=False)
    loop_ids: dict[tuple[str, tuple[tuple[str, int], ...]], str] = field(init=False)
    loops: dict[str, SurfaceLoopRecord] = field(init=False)
    canonical_surfaces: list[SurfacePlanRecord] = field(init=False)
    surface_specs: list[
        tuple[
            SurfacePlanRecord,
            tuple[tuple[str, int, tuple[tuple[float, float, float], ...]], ...],
        ]
    ] = field(init=False)
    point_axis_index: tuple[
        tuple[tuple[float, ...], tuple[tuple[float, float, float], ...]], ...
    ] = field(init=False)
    point_line_index: dict[
        tuple[int, tuple[float, float]],
        tuple[tuple[float, ...], tuple[tuple[float, float, float], ...]],
    ] = field(init=False)
    retained_curve_surface_ids: dict[str, set[str]] = field(init=False)
    retained_curve_ids: set[str] = field(init=False)
    retained_surfaces: dict[str, SurfacePlanRecord] = field(init=False)

    def build(self):
        """Register complete rings, then rebuild claims from retained topology."""
        self.initialize()
        self.collect_points()
        self.index_points()
        self.register_loops()
        self.retain_surfaces()
        # Provisional claims can include canceled Boolean residuals. Published
        # curve ownership is rebuilt entirely from the retained surface loops.
        self.resolve_claims()
        self.assert_retained_claims()
        return self.assemble()

    def initialize(self) -> None:
        """Allocate canonical identity ledgers in their existing order."""
        self.point_ids: dict[tuple[float, float, float], str] = {}
        self.point_coordinates: dict[str, tuple[float, float, float]] = {}
        self.point_curve_ids: dict[str, set[str]] = {}
        self.curve_ids: dict[tuple[str, str], str] = {}
        self.curve_owner_ids: dict[str, set[str]] = {}
        self.curve_interface_ids: dict[str, set[str]] = {}
        self.curve_surface_ids: dict[str, set[str]] = {}
        self.curve_volume_ids: dict[str, set[str]] = {}
        self.loop_ids: dict[tuple[str, tuple[tuple[str, int], ...]], str] = {}
        self.loops: dict[str, SurfaceLoopRecord] = {}
        self.canonical_surfaces: list[SurfacePlanRecord] = []
        self.surface_specs: list[
            tuple[
                SurfacePlanRecord,
                tuple[tuple[str, int, tuple[tuple[float, float, float], ...]], ...],
            ]
        ] = []

    def collect_points(self) -> None:
        """Collect all live surface coordinates before splitting any edge."""
        for surface in self.surfaces:
            if surface.construction_only:
                self.canonical_surfaces.append(surface)
                continue
            specs = surface_ring3d_specs(surface)
            self.surface_specs.append((surface, specs))
            for _, _, ring in specs:
                for coordinate in ring:
                    self.point_id(coordinate)

    def index_points(self) -> None:
        """Index the complete point set along axes and aligned lines."""
        self.point_axis_index = index_points_by_axis(self.point_coordinates.values())
        self.point_line_index = axis_aligned_point_index(
            self.point_coordinates.values()
        )

    def register_loops(self) -> None:
        """Split rings into canonical curves and retain loop orientation."""
        for surface, specs in self.surface_specs:
            loop_refs: list[str] = []
            for role, index, ring in specs:
                raw_curve_refs = tuple(
                    (
                        self.curve_ref(segment_start, segment_end, surface)
                        for start, end in ring3d_edges(ring)
                        for split_points in (self.split_edge_points(start, end),)
                        for segment_start, segment_end in pairwise(split_points)
                    )
                )
                curve_refs = _cancel_backtracking_curve_refs(raw_curve_refs)
                if not curve_refs:
                    raise ValueError(f"{surface.surface_id} has empty canonical loop")
                loop_key = (role, ordered_loop_signature(curve_refs))
                loop_id = self.loop_ids.get(loop_key)
                if loop_id is None:
                    suffix = "OUTER" if role == "outer" else f"HOLE_{index:04d}"
                    loop_id = f"LOOP__{surface.surface_id}__{suffix}"
                    self.loop_ids[loop_key] = loop_id
                    self.loops[loop_id] = SurfaceLoopRecord(
                        loop_id=loop_id,
                        curve_refs=curve_refs,
                        role=cast(SurfaceLoopRoleLiteral, role),
                        surface_id=surface.surface_id,
                    )
                loop_refs.append(loop_id)
            if not loop_refs:
                raise ValueError(f"{surface.surface_id} has no planned loops")
            self.canonical_surfaces.append(
                replace(
                    surface,
                    outer_loop_ref=loop_refs[0],
                    hole_loop_refs=tuple(loop_refs[1:]),
                )
            )

    def retain_surfaces(self) -> None:
        """Resolve the surfaces and curves that survive residual cancellation."""
        retained_loop_surface_ids: dict[str, set[str]] = {}
        for surface in self.canonical_surfaces:
            if surface.construction_only:
                continue
            for loop_id in (surface.outer_loop_ref, *surface.hole_loop_refs):
                if loop_id is not None:
                    retained_loop_surface_ids.setdefault(loop_id, set()).add(
                        surface.surface_id
                    )
        self.retained_curve_surface_ids: dict[str, set[str]] = {}
        for loop_id, loop in self.loops.items():
            for ref in loop.curve_refs:
                self.retained_curve_surface_ids.setdefault(ref.curve_id, set()).update(
                    retained_loop_surface_ids.get(loop_id, ())
                )
        self.retained_curve_ids = set(self.retained_curve_surface_ids)
        self.retained_surfaces = {
            surface.surface_id: surface for surface in self.canonical_surfaces
        }

    def resolve_claims(self) -> None:
        """Rebuild public ownership from the retained surface loops."""
        self.curve_owner_ids = {
            curve_id: {
                owner_id
                for surface_id in surface_ids
                for owner_id in surface_owner_ids(self.retained_surfaces[surface_id])
            }
            for curve_id, surface_ids in self.retained_curve_surface_ids.items()
        }
        self.curve_interface_ids = {
            curve_id: {
                cast(str, self.retained_surfaces[surface_id].interface_id)
                for surface_id in surface_ids
                if self.retained_surfaces[surface_id].interface_id is not None
            }
            for curve_id, surface_ids in self.retained_curve_surface_ids.items()
        }
        self.curve_volume_ids = {
            curve_id: {
                str(volume_id)
                for surface_id in surface_ids
                for volume_id in self.retained_surfaces[surface_id].metadata.get(
                    "boundary_volume_ids", ()
                )
            }
            for curve_id, surface_ids in self.retained_curve_surface_ids.items()
        }

    def assert_retained_claims(self) -> None:
        """Check each curve claim against its retained source surfaces."""
        for curve_id, surface_ids in self.retained_curve_surface_ids.items():
            retained = tuple(
                (self.retained_surfaces[surface_id] for surface_id in surface_ids)
            )
            expected_owners = {
                owner_id
                for surface in retained
                for owner_id in surface_owner_ids(surface)
            }
            expected_interfaces = {
                surface.interface_id
                for surface in retained
                if surface.interface_id is not None
            }
            expected_volumes = {
                str(volume_id)
                for surface in retained
                for volume_id in surface.metadata.get("boundary_volume_ids", ())
            }
            if (
                self.curve_owner_ids[curve_id] != expected_owners
                or self.curve_interface_ids[curve_id] != expected_interfaces
                or self.curve_volume_ids[curve_id] != expected_volumes
            ):
                raise AssertionError(
                    f"{curve_id} retains claims absent from its retained surfaces"
                )

    def assemble(
        self,
    ) -> tuple[
        tuple[PointPlanRecord, ...],
        tuple[CurvePlanRecord, ...],
        tuple[SurfaceLoopRecord, ...],
        tuple[SurfacePlanRecord, ...],
    ]:
        """Emit canonical points, retained curves, loops and surfaces."""
        self.point_curve_ids = {
            point_id_: {
                curve_id
                for curve_id in curve_ids_
                if curve_id in self.retained_curve_ids
            }
            for point_id_, curve_ids_ in self.point_curve_ids.items()
        }
        points = tuple(
            (
                PointPlanRecord(
                    point_id=point_id_,
                    coordinate=self.point_coordinates[point_id_],
                    used_by_curve_ids=tuple(sorted(self.point_curve_ids[point_id_])),
                )
                for point_id_ in sorted(self.point_coordinates)
            )
        )
        curves = tuple(
            (
                CurvePlanRecord(
                    curve_id=curve_id,
                    curve_kind="line_segment",
                    start_point_id=start_id,
                    end_point_id=end_id,
                    owner_semantic_ids=tuple(
                        sorted(self.curve_owner_ids.get(curve_id, ()))
                    ),
                    interface_ids=tuple(
                        sorted(self.curve_interface_ids.get(curve_id, ()))
                    ),
                    used_by_surface_ids=tuple(
                        sorted(self.retained_curve_surface_ids.get(curve_id, ()))
                    ),
                    boundary_volume_ids=tuple(
                        sorted(self.curve_volume_ids.get(curve_id, ()))
                    ),
                )
                for (start_id, end_id), curve_id in sorted(
                    self.curve_ids.items(), key=lambda item: item[1]
                )
                if curve_id in self.retained_curve_ids
            )
        )
        return (
            points,
            curves,
            tuple(self.loops.values()),
            tuple(self.canonical_surfaces),
        )

    def point_id(self, coordinate: tuple[float, float, float]) -> str:
        """Resolve point id within the shared registry."""
        key = coordinate_key(coordinate)
        existing = self.point_ids.get(key)
        if existing is not None:
            return existing
        new_id = f"P__{len(self.point_ids):06d}"
        self.point_ids[key] = new_id
        self.point_coordinates[new_id] = key
        self.point_curve_ids[new_id] = set()
        return new_id

    def split_edge_points(
        self, start: tuple[float, float, float], end: tuple[float, float, float]
    ) -> tuple[tuple[float, float, float], ...]:
        """Resolve split edge points within the shared registry."""
        points_on_edge = [
            (parameter, coordinate)
            for coordinate in segment_candidate_points(
                start, end, self.point_axis_index, self.point_line_index
            )
            for parameter in (segment_parameter(coordinate, start, end),)
            if parameter is not None
        ]
        return tuple((coordinate for _, coordinate in sorted(points_on_edge)))

    def curve_ref(
        self,
        start: tuple[float, float, float],
        end: tuple[float, float, float],
        surface: SurfacePlanRecord,
    ) -> CurveRefRecord:
        """Resolve curve ref within the shared registry."""
        start_id = self.point_id(start)
        end_id = self.point_id(end)
        if start_id == end_id:
            raise ValueError(f"{surface.surface_id} has zero-length curve")
        key = cast(tuple[str, str], tuple(sorted((start_id, end_id))))
        curve_id = self.curve_ids.get(key)
        if curve_id is None:
            curve_id = f"C__{len(self.curve_ids):06d}"
            self.curve_ids[key] = curve_id
        self.point_curve_ids[start_id].add(curve_id)
        self.point_curve_ids[end_id].add(curve_id)
        self.curve_owner_ids.setdefault(curve_id, set()).update(
            (str(owner_id) for owner_id in surface_owner_ids(surface))
        )
        if surface.interface_id is not None:
            self.curve_interface_ids.setdefault(curve_id, set()).add(
                surface.interface_id
            )
        self.curve_surface_ids.setdefault(curve_id, set()).add(surface.surface_id)
        self.curve_volume_ids.setdefault(curve_id, set()).update(
            (
                str(volume_id)
                for volume_id in surface.metadata.get("boundary_volume_ids", ())
            )
        )
        return CurveRefRecord(
            curve_id=curve_id,
            orientation=1 if key == (start_id, end_id) else -1,
            role="boundary",
        )


def _cancel_backtracking_curve_refs(
    refs: tuple[CurveRefRecord, ...],
) -> tuple[CurveRefRecord, ...]:
    """Remove immediate A→B→A artifacts from Boolean residual splitting."""
    result: list[CurveRefRecord] = []
    for ref in refs:
        if (
            result
            and result[-1].curve_id == ref.curve_id
            and result[-1].orientation == -ref.orientation
        ):
            result.pop()
        else:
            result.append(ref)
    if (
        len(result) > 1
        and result[0].curve_id == result[-1].curve_id
        and result[0].orientation == -result[-1].orientation
    ):
        result = result[1:-1]
    return tuple(result)
