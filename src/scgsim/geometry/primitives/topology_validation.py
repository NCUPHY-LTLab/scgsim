"""Own canonical point/curve/loop connectivity, surface overlap and volume shells.

The checks preserve existing error aggregation, geometry arithmetic and tolerances.
Native polygon operations remain optional and load only at their original call sites.
"""

from __future__ import annotations

import json
from bisect import bisect_left, bisect_right
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import import_module
from typing import TYPE_CHECKING, Any, NamedTuple

from scgsim.geometry.models.topology import (
    CurvePlanRecord,
    PointPlanRecord,
    SurfaceLoopRecord,
    SurfacePlanRecord,
    SurfaceRefRecord,
    VolumePlanRecord,
)
from scgsim.geometry.primitives.entity_validation import (
    BOOLEAN_AREA_EPS_UM2,
    BOOLEAN_RELATIVE_AREA_EPS,
)

if TYPE_CHECKING:
    import gdstk

type _CoordinateKey = tuple[float, ...]
type _CurveSignature = tuple[str, ...] | tuple[str, str, tuple[float, float] | None]
type _AxisPointIndex = tuple[tuple[float, ...], tuple[str, ...]]
type _PointLineKey = tuple[int, _CoordinateKey]
type _Region = tuple[gdstk.Polygon, ...]
type _Loop2D = tuple[tuple[float, float], ...]
type _Bounds2D = tuple[float, float, float, float]


class _HorizontalSurface(NamedTuple):
    surface: SurfacePlanRecord
    z_um: float
    region: _Region
    bounds: _Bounds2D


def validate_curve_plan_coverage(
    *,
    points: tuple[PointPlanRecord, ...],
    curves: tuple[CurvePlanRecord, ...],
    surface_loops: tuple[SurfaceLoopRecord, ...],
    surfaces: tuple[SurfacePlanRecord, ...],
) -> None:
    """Validate compiler-owned point, curve, and surface-loop topology.

    Each `SurfaceLoopRecord` must be one simple directed cycle: holes are
    separate loops, and shared curves are legal across different surfaces or
    loops but may not repeat inside one loop. This is the pre-OCC contract that
    keeps self-touching boolean output from reaching backend curve-loop
    lowering.
    """
    _CurvePlanValidation(points, curves, surface_loops, surfaces).validate()


def validate_surface_deduplication(
    *,
    surfaces: tuple[SurfacePlanRecord, ...],
) -> None:
    """Reject two backend-live surfaces with the same canonical signature."""
    signatures: dict[str, str] = {}
    errors: list[str] = []
    for surface in surfaces:
        if surface.construction_only:
            continue
        signature = _surface_signature(surface)
        existing = signatures.get(signature)
        if existing is not None and existing != surface.surface_id:
            errors.append(
                f"duplicate live surface geometry: {existing} and {surface.surface_id}"
            )
        signatures[signature] = surface.surface_id
    if errors:
        raise ValueError("Invalid surface deduplication: " + "; ".join(errors))


def validate_no_surface_overlap(
    *,
    surfaces: tuple[SurfacePlanRecord, ...],
) -> None:
    """Reject horizontal same-plane partial overlap for all solver-live surfaces.

    This guard is intentionally not allowed to skip high-count conductors by
    semantic owner. Physical groups may collapse many instances into one solver
    label, but topology validation must still see every live surface. General
    sidewall/arbitrary-plane overlap validation is a separate contract and must
    fail fast before those cases are advertised as mesh-safe.
    """
    horizontal_surfaces = tuple(
        _HorizontalSurface(surface, z_um, region, _region_bounds(region))
        for surface in surfaces
        if not surface.construction_only
        for z_um, region in (_horizontal_surface_region(surface),)
        if region
    )
    outer_loop_signatures = {
        item.surface.surface_id: _loop2d_signature(
            item.surface.geometry_ref.get("outer_loop")
        )
        for item in horizontal_surfaces
    }
    known_hole_signatures = {
        item.surface.surface_id: _known_hole_signatures(item.surface)
        for item in horizontal_surfaces
    }
    surfaces_by_z: dict[float, list[_HorizontalSurface]] = {}
    for item in horizontal_surfaces:
        surfaces_by_z.setdefault(round(item.z_um, 9), []).append(item)
    error = _first_surface_overlap(
        surfaces_by_z, outer_loop_signatures, known_hole_signatures
    )
    if error is not None:
        raise ValueError("Invalid surface overlap: " + error)


def validate_volume_surface_closure(
    *,
    volumes: tuple[VolumePlanRecord, ...],
    surfaces: tuple[SurfacePlanRecord, ...],
    surface_loops: tuple[SurfaceLoopRecord, ...],
) -> None:
    """Validate volume surface references before backend lowering.

    Full closed-shell validation belongs to the canonical curve/loop plan. This
    guard rejects missing references and then checks curve incidence from the
    planned loops, before OCC is allowed to construct a volume.
    """
    checker = _VolumeShellValidation(
        {surface.surface_id: surface for surface in surfaces},
        {loop.loop_id: loop for loop in surface_loops},
    )
    for volume in volumes:
        if volume.construction_only:
            continue
        checker.validate_shell(
            volume_id=volume.volume_id,
            shell_id="exterior",
            surface_refs=(
                getattr(volume, "exterior_surface_refs", ()) or volume.surface_refs
            ),
        )
        for void in getattr(volume, "inner_pec_void_shells", ()):
            checker.validate_shell(
                volume_id=volume.volume_id,
                shell_id=void.shell_id,
                surface_refs=void.surface_refs,
            )
    if checker.errors:
        raise ValueError("Invalid volume surface closure: " + "; ".join(checker.errors))


def validate_surface_use_counts(
    *,
    volumes: tuple[VolumePlanRecord, ...],
    surfaces: tuple[SurfacePlanRecord, ...],
) -> None:
    """Validate internal/exterior surface incidence across live volumes."""
    live_surface_ids = {
        surface.surface_id for surface in surfaces if not surface.construction_only
    }
    use_counts: Counter[str] = Counter(
        surface_ref.surface_id
        for volume in volumes
        if not volume.construction_only
        for surface_ref in volume.surface_refs
    )
    errors: list[str] = []
    for surface in surfaces:
        if surface.construction_only:
            continue
        if surface.surface_role == "lumped_port" and not surface.metadata.get(
            "route_a_boundary_port"
        ):
            continue
        if surface.metadata.get("embedded_surface_sheet") and not surface.metadata.get(
            "sheet_contact_cap"
        ):
            continue
        count = use_counts[surface.surface_id]
        if surface.surface_id not in live_surface_ids:
            errors.append(f"{surface.surface_id} is not live")
        if surface.metadata.get("route_a_boundary_port"):
            if count != 2:
                errors.append(
                    f"{surface.surface_id} Route A port use-count must be 2, "
                    f"got {count}"
                )
        elif surface.interface_id is not None and count not in {1, 2}:
            errors.append(
                f"{surface.surface_id} interface use-count must be 1 or 2, got {count}"
            )
        elif surface.interface_id is None and count != 1:
            errors.append(
                f"{surface.surface_id} exterior use-count must be 1, got {count}"
            )
    if errors:
        raise ValueError("Invalid surface use counts: " + "; ".join(errors))


@dataclass
class _CurvePlanValidation:
    points: tuple[PointPlanRecord, ...]
    curves: tuple[CurvePlanRecord, ...]
    surface_loops: tuple[SurfaceLoopRecord, ...]
    surfaces: tuple[SurfacePlanRecord, ...]
    errors: list[str] = field(default_factory=list)
    point_ids: set[str] = field(default_factory=set)
    points_by_id: dict[str, PointPlanRecord] = field(default_factory=dict)

    def validate(self) -> None:
        """Accumulate the existing failures in their original phase order."""
        if any((not surface.construction_only for surface in self.surfaces)) and (
            not self.points or not self.curves or (not self.surface_loops)
        ):
            self.errors.append(
                "backend-live surfaces require canonical points, curves, and loops"
            )
        self.index_points()
        self.check_curve_records()
        self.check_curve_interiors()
        self.check_surface_loops()
        self.check_surface_loop_refs()
        if self.errors:
            raise ValueError("Invalid curve plan coverage: " + "; ".join(self.errors))

    def index_points(self) -> None:
        """Retain the final point per ID while reporting duplicate IDs/coordinates."""
        point_signatures: dict[_CoordinateKey, str] = {}
        for point in self.points:
            if point.point_id in self.point_ids:
                self.errors.append(f"duplicate point_id: {point.point_id}")
            self.point_ids.add(point.point_id)
            self.points_by_id[point.point_id] = point
            signature = _coordinate_signature(point.coordinate)
            existing = point_signatures.get(signature)
            if existing is not None:
                self.errors.append(
                    f"duplicate point coordinate: {existing} and {point.point_id}"
                )
            point_signatures[signature] = point.point_id

    def check_curve_records(self) -> None:
        """Check curve identities, endpoints and original geometry signatures."""
        curve_ids: set[str] = set()
        curve_signatures: dict[_CurveSignature, str] = {}
        for curve in self.curves:
            if curve.curve_id in curve_ids:
                self.errors.append(f"duplicate curve_id: {curve.curve_id}")
            curve_ids.add(curve.curve_id)
            if curve.start_point_id not in self.point_ids:
                self.errors.append(
                    f"{curve.curve_id} references unknown point {curve.start_point_id}"
                )
            if curve.end_point_id not in self.point_ids:
                self.errors.append(
                    f"{curve.curve_id} references unknown point {curve.end_point_id}"
                )
            if (
                curve.start_point_id == curve.end_point_id
                and curve.curve_kind == "line_segment"
            ):
                self.errors.append(f"{curve.curve_id} has identical start/end points")
                continue
            signature = (
                tuple(sorted((curve.start_point_id, curve.end_point_id)))
                if curve.curve_kind == "line_segment"
                else (
                    curve.curve_kind,
                    json.dumps(curve.geometry, sort_keys=True),
                    curve.parameter_interval,
                )
            )
            existing = curve_signatures.get(signature)
            if existing is not None:
                self.errors.append(
                    f"duplicate curve geometry: {existing} and {curve.curve_id}"
                )
            curve_signatures[signature] = curve.curve_id

    def check_curve_interiors(self) -> None:
        """Check un-noded straight segments with the unchanged point indexes."""
        point_axis_index = _point_axis_index(self.points_by_id)
        point_line_index = _axis_aligned_point_index(self.points_by_id)
        for curve in self.curves:
            if curve.curve_kind != "line_segment":
                continue
            start = self.points_by_id.get(curve.start_point_id)
            end = self.points_by_id.get(curve.end_point_id)
            if start is None or end is None:
                continue
            interior_points = [
                point_id
                for point_id in _segment_candidate_point_ids(
                    start.coordinate, end.coordinate, point_axis_index, point_line_index
                )
                if point_id not in {curve.start_point_id, curve.end_point_id}
                and _point_on_segment_interior(
                    self.points_by_id[point_id].coordinate,
                    start.coordinate,
                    end.coordinate,
                )
            ]
            if interior_points:
                self.errors.append(
                    f"{curve.curve_id} contains planned points in its interior: {interior_points!r}"
                )

    def check_surface_loops(self) -> None:
        """Check each loop's size, curve references and directed cycle in order."""
        curves_by_id = {curve.curve_id: curve for curve in self.curves}
        for loop in self.surface_loops:
            if len(loop.curve_refs) < 3 and (
                not any(
                    (
                        curves_by_id.get(ref.curve_id) is not None
                        and curves_by_id[ref.curve_id].curve_kind != "line_segment"
                        for ref in loop.curve_refs
                    )
                )
            ):
                self.errors.append(
                    f"{loop.loop_id} requires at least three straight curves"
                )
                continue
            directed_edges = self.directed_edges(loop, curves_by_id)
            self.check_loop_cycle(loop, directed_edges)

    def directed_edges(
        self, loop: SurfaceLoopRecord, curves_by_id: Mapping[str, CurvePlanRecord]
    ) -> list[tuple[str, str]]:
        """Report repeated/unknown refs before deriving valid oriented edges."""
        repeated_curve_refs = sorted(
            (
                curve_id
                for curve_id, count in Counter(
                    (curve_ref.curve_id for curve_ref in loop.curve_refs)
                ).items()
                if count > 1
            )
        )
        if repeated_curve_refs:
            self.errors.append(
                f"{loop.loop_id} repeats curve refs inside one surface loop: {repeated_curve_refs!r}"
            )
        directed_edges: list[tuple[str, str]] = []
        for curve_ref in loop.curve_refs:
            curve = curves_by_id.get(curve_ref.curve_id)
            if curve is None:
                self.errors.append(
                    f"{loop.loop_id} references unknown curve {curve_ref.curve_id}"
                )
                continue
            if curve_ref.orientation not in {-1, 1}:
                self.errors.append(
                    f"{loop.loop_id} curve {curve_ref.curve_id} has invalid orientation {curve_ref.orientation!r}"
                )
                continue
            if curve_ref.orientation == 1:
                directed_edges.append((curve.start_point_id, curve.end_point_id))
            else:
                directed_edges.append((curve.end_point_id, curve.start_point_id))
        return directed_edges

    def check_loop_cycle(
        self, loop: SurfaceLoopRecord, directed_edges: list[tuple[str, str]]
    ) -> None:
        """Check closure before the original simple-cycle incidence requirement."""
        for index, (_, end_point_id) in enumerate(directed_edges):
            next_start_id, _ = directed_edges[(index + 1) % len(directed_edges)]
            if end_point_id != next_start_id:
                self.errors.append(
                    f"{loop.loop_id} curve refs do not form a closed loop"
                )
                break
        if len(directed_edges) == len(loop.curve_refs):
            point_use_counts: Counter[str] = Counter()
            for start_point_id, end_point_id in directed_edges:
                point_use_counts.update((start_point_id, end_point_id))
            bad_point_use_counts = {
                point_id: count
                for point_id, count in point_use_counts.items()
                if count != 2
            }
            if bad_point_use_counts:
                self.errors.append(
                    f"{loop.loop_id} is not a simple surface loop; point use counts {bad_point_use_counts!r}"
                )

    def check_surface_loop_refs(self) -> None:
        """Check live surface loop references after all loop diagnostics."""
        loop_ids = {loop.loop_id for loop in self.surface_loops}
        for surface in self.surfaces:
            if surface.construction_only:
                continue
            if surface.outer_loop_ref is None:
                self.errors.append(f"{surface.surface_id} lacks outer_loop_ref")
            refs = (
                *(
                    (surface.outer_loop_ref,)
                    if surface.outer_loop_ref is not None
                    else ()
                ),
                *surface.hole_loop_refs,
            )
            for loop_ref in refs:
                if loop_ref not in loop_ids:
                    self.errors.append(
                        f"{surface.surface_id} references unknown loop {loop_ref}"
                    )


@dataclass
class _VolumeShellValidation:
    surfaces_by_id: Mapping[str, SurfacePlanRecord]
    loops_by_id: Mapping[str, SurfaceLoopRecord]
    errors: list[str] = field(default_factory=list)

    def validate_shell(
        self,
        *,
        volume_id: str,
        shell_id: str,
        surface_refs: tuple[SurfaceRefRecord, ...],
    ) -> None:
        """Accumulate one shell's reference and curve-incidence errors."""
        curve_use_count: Counter[str] = Counter()
        for surface_ref in surface_refs:
            surface = self.surfaces_by_id.get(surface_ref.surface_id)
            if surface is None:
                self.errors.append(
                    f"{volume_id}/{shell_id} references unknown surface {surface_ref.surface_id}"
                )
                continue
            loop_ids = (
                *((surface.outer_loop_ref,) if surface.outer_loop_ref else ()),
                *surface.hole_loop_refs,
            )
            if not loop_ids:
                self.errors.append(
                    f"{volume_id}/{shell_id} uses non-canonical surface {surface.surface_id}"
                )
            for loop_id in loop_ids:
                loop = self.loops_by_id.get(loop_id)
                if loop is None:
                    self.errors.append(
                        f"{volume_id}/{shell_id} references unknown loop {loop_id}"
                    )
                    continue
                curve_use_count.update(
                    (curve_ref.curve_id for curve_ref in loop.curve_refs)
                )
        bad_counts = {
            curve_id: count for curve_id, count in curve_use_count.items() if count != 2
        }
        if bad_counts:
            self.errors.append(
                f"{volume_id}/{shell_id} shell is not closed; curve incidence {bad_counts!r}"
            )


def _first_surface_overlap(
    surfaces_by_z: Mapping[float, list[_HorizontalSurface]],
    outer_loop_signatures: Mapping[str, _Loop2D | None],
    known_hole_signatures: Mapping[str, set[_Loop2D]],
) -> str | None:
    for z_um, z_surfaces in surfaces_by_z.items():
        ordered = sorted(z_surfaces, key=lambda item: item.bounds[0])
        for index, left in enumerate(ordered):
            for right in ordered[index + 1 :]:
                if right.bounds[0] > left.bounds[2] + 1e-9:
                    break
                if not _bounds_overlap(left.bounds, right.bounds):
                    continue
                left_outer = outer_loop_signatures[left.surface.surface_id]
                right_outer = outer_loop_signatures[right.surface.surface_id]
                if (
                    left_outer is not None
                    and left_outer in known_hole_signatures[right.surface.surface_id]
                ) or (
                    right_outer is not None
                    and right_outer in known_hole_signatures[left.surface.surface_id]
                ):
                    continue
                if not _regions_overlap(left.region, right.region):
                    continue
                return f"{left.surface.surface_id} overlaps {right.surface.surface_id} on z={z_um}"
    return None


def _horizontal_surface_region(
    surface: SurfacePlanRecord,
) -> tuple[float, _Region]:
    geometry_ref = surface.geometry_ref
    if "outer_loop" not in geometry_ref or "quad_points" in geometry_ref:
        return 0.0, ()
    # Polygon operations are optional; identifier-only checks never load gdstk.
    gdstk = import_module("gdstk")

    outer = gdstk.Polygon(_clean_loop2d(geometry_ref["outer_loop"]))
    holes = tuple(
        gdstk.Polygon(_clean_loop2d(hole_loop))
        for hole_loop in geometry_ref.get("hole_loops", ())
    )
    if holes:
        region = tuple(gdstk.boolean((outer,), holes, "not", precision=1e-9) or ())
    else:
        region = (outer,)
    return _geometry_ref_z_um(geometry_ref), tuple(
        polygon
        for polygon in region
        if abs(float(polygon.area())) > BOOLEAN_AREA_EPS_UM2
    )


def _regions_overlap(left: _Region, right: _Region) -> bool:
    # Polygon operations are optional; identifier-only checks never load gdstk.
    gdstk = import_module("gdstk")

    overlap = gdstk.boolean(left, right, "and", precision=1e-9) or ()
    overlap_area = _region_area(overlap)
    area_limit = max(
        BOOLEAN_AREA_EPS_UM2,
        BOOLEAN_RELATIVE_AREA_EPS * min(_region_area(left), _region_area(right)),
    )
    return overlap_area > area_limit


def _region_area(region: Sequence[gdstk.Polygon]) -> float:
    return sum(abs(float(polygon.area())) for polygon in region)


def _known_hole_signatures(
    surface: SurfacePlanRecord,
) -> set[tuple[tuple[float, float], ...]]:
    geometry_ref = surface.geometry_ref
    return {
        signature
        for key in ("hole_loops", "contact_hole_loops")
        for loop in geometry_ref.get(key, ())
        for signature in (_loop2d_signature(loop),)
        if signature is not None
    }


def _loop2d_signature(
    loop: Iterable[Sequence[float]] | None,
) -> tuple[tuple[float, float], ...] | None:
    if loop is None:
        return None
    return tuple(sorted(_coordinate_2d_key(point) for point in _clean_loop2d(loop)))


def _coordinate_2d_key(point: tuple[float, float]) -> tuple[float, float]:
    return (round(float(point[0]), 9), round(float(point[1]), 9))


def _region_bounds(
    region: Sequence[gdstk.Polygon],
) -> tuple[float, float, float, float]:
    boxes = [polygon.bounding_box() for polygon in region]
    return (
        min(float(box[0][0]) for box in boxes),
        min(float(box[0][1]) for box in boxes),
        max(float(box[1][0]) for box in boxes),
        max(float(box[1][1]) for box in boxes),
    )


def _bounds_overlap(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> bool:
    return not (
        left[2] <= right[0] + 1e-9
        or right[2] <= left[0] + 1e-9
        or left[3] <= right[1] + 1e-9
        or right[3] <= left[1] + 1e-9
    )


def _clean_loop2d(loop: Iterable[Sequence[float]]) -> tuple[tuple[float, float], ...]:
    points = tuple((float(point[0]), float(point[1])) for point in loop)
    if len(points) > 1 and points[0] == points[-1]:
        points = points[:-1]
    if len(points) < 3:
        raise ValueError("surface loop requires at least 3 points")
    return points


def _geometry_ref_z_um(geometry_ref: Mapping[str, Any]) -> float:
    z_min_um = float(geometry_ref.get("z_min_um", geometry_ref.get("z_um", 0.0)))
    if geometry_ref.get("shell_part") == "top":
        return z_min_um + float(geometry_ref.get("thickness_um", 0.0))
    if geometry_ref.get("shell_part") == "bottom":
        return z_min_um
    plane = geometry_ref.get("plane") or geometry_ref.get("contact_plane")
    if isinstance(plane, Mapping) and plane.get("axis") == "z":
        return float(plane["value_um"])
    return z_min_um


def _surface_signature(surface: SurfacePlanRecord) -> str:
    if surface.outer_loop_ref is not None:
        return json.dumps(
            {
                "outer": surface.outer_loop_ref,
                "holes": surface.hole_loop_refs,
            },
            sort_keys=True,
        )
    return json.dumps(surface.geometry_ref, sort_keys=True, default=str)


def _coordinate_signature(
    coordinate: tuple[float, float, float],
) -> _CoordinateKey:
    return tuple(round(float(value), 9) for value in coordinate)


def _point_axis_index(
    points_by_id: Mapping[str, PointPlanRecord],
) -> tuple[_AxisPointIndex, ...]:
    indexes: list[_AxisPointIndex] = []
    for axis in range(3):
        items = sorted(
            (
                _coordinate_signature(point.coordinate)[axis],
                point_id,
            )
            for point_id, point in points_by_id.items()
        )
        indexes.append(
            (
                tuple(value for value, _ in items),
                tuple(point_id for _, point_id in items),
            )
        )
    return tuple(indexes)


def _axis_aligned_point_index(
    points_by_id: Mapping[str, PointPlanRecord],
) -> dict[_PointLineKey, _AxisPointIndex]:
    records: dict[
        _PointLineKey,
        list[tuple[float, str]],
    ] = {}
    for point_id, point in points_by_id.items():
        key = _coordinate_signature(point.coordinate)
        for varying_axis in range(3):
            fixed_key = tuple(key[axis] for axis in range(3) if axis != varying_axis)
            records.setdefault((varying_axis, fixed_key), []).append(
                (key[varying_axis], point_id)
            )
    return {
        line_key: (
            tuple(value for value, _ in sorted_items),
            tuple(point_id for _, point_id in sorted_items),
        )
        for line_key, items in records.items()
        for sorted_items in (tuple(sorted(items)),)
    }


def _segment_candidate_point_ids(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    point_axis_index: tuple[_AxisPointIndex, ...],
    point_line_index: Mapping[
        _PointLineKey,
        _AxisPointIndex,
    ],
) -> tuple[str, ...]:
    start_key = _coordinate_signature(start)
    end_key = _coordinate_signature(end)
    varying_axes = tuple(
        axis for axis in range(3) if abs(start_key[axis] - end_key[axis]) > 1e-9
    )
    if len(varying_axes) == 1:
        varying_axis = varying_axes[0]
        fixed_key = tuple(start_key[axis] for axis in range(3) if axis != varying_axis)
        values, point_ids = point_line_index.get(
            (varying_axis, fixed_key),
            ((), ()),
        )
        lower = min(start_key[varying_axis], end_key[varying_axis]) - 1e-9
        upper = max(start_key[varying_axis], end_key[varying_axis]) + 1e-9
        left = bisect_left(values, lower)
        right = bisect_right(values, upper)
        return point_ids[left:right]
    bounds = tuple(
        (
            min(start_key[axis], end_key[axis]) - 1e-9,
            max(start_key[axis], end_key[axis]) + 1e-9,
        )
        for axis in range(3)
    )
    ranges: list[tuple[int, int, int]] = []
    for axis, (values, _) in enumerate(point_axis_index):
        lower, upper = bounds[axis]
        left = bisect_left(values, lower)
        right = bisect_right(values, upper)
        ranges.append((right - left, left, axis))
    count, left, axis = min(ranges)
    return point_axis_index[axis][1][left : left + count]


def _point_on_segment_interior(
    point: tuple[float, float, float],
    start: tuple[float, float, float],
    end: tuple[float, float, float],
) -> bool:
    vector = tuple(float(end[index]) - float(start[index]) for index in range(3))
    offset = tuple(float(point[index]) - float(start[index]) for index in range(3))
    length_sq = sum(value * value for value in vector)
    if length_sq <= 1e-18:
        return False
    parameter = sum(offset[index] * vector[index] for index in range(3)) / length_sq
    if parameter <= 1e-9 or parameter >= 1.0 - 1e-9:
        return False
    closest = tuple(
        float(start[index]) + parameter * vector[index] for index in range(3)
    )
    distance_sq = sum((float(point[index]) - closest[index]) ** 2 for index in range(3))
    return distance_sq <= 1e-18
