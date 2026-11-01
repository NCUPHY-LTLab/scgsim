"""Range-safe tetrahedron orientation, volume, center, and condition metrics.

For vertices p0..p3, B=[p1-p0 p2-p0 p3-p0], D=det(B), signed
volume is D/6, and volume magnitude is abs(D)/6.  The reference edge matrix
W has rows (1,1/2,1/2), (0,sqrt(3)/2,sqrt(3)/6), and
(0,0,sqrt(2/3)); J=B inv(W) and kappa(J)=smax/smin.

The floating orientation filter assumes IEEE-754 binary64 round-to-nearest.
One exactly reversible, element-wide power-of-two scaling occurs before edge
subtraction.  The explicit determinant expansion uses the matching bound
E=(7+56u)uP with u=2**-53.  This certifies sign only, not relative volume
accuracy. Unsafe scaling, subnormal intermediates, inconclusive signs, and
representation-boundary ambiguity use exact dyadic arithmetic from the twelve
original parsed binary64 coordinates.
"""

from __future__ import annotations

import dataclasses
import math
import sys
from dataclasses import dataclass
from enum import IntEnum
from fractions import Fraction
from types import MappingProxyType
from typing import Mapping, NamedTuple

import numpy as np

from .reader import MeshData, _readonly

_U = 2.0**-53
_EPS = 2.0**-52
_MIN_NORMAL = sys.float_info.min
_W = np.asarray(
    (
        (1.0, 0.5, 0.5),
        (0.0, math.sqrt(3.0) / 2.0, math.sqrt(3.0) / 6.0),
        (0.0, 0.0, math.sqrt(2.0 / 3.0)),
    ),
    dtype=np.float64,
)
_INV_W = np.linalg.inv(_W)
_INV_W.flags.writeable = False


class _StatusCode(IntEnum):
    """Byte codes stored in the status and method arrays, decoded by ``_STATUS_LABELS``."""

    UNAVAILABLE = 0
    FLOATING_FILTER = 1
    EXACT_DYADIC = 2
    MISSING_NODES = 3
    NONFINITE_COORDINATES = 4
    NOT_REQUESTED = 5
    EXACT_ZERO = 6
    OVERFLOW = 7
    UNDERFLOW = 8
    AVAILABLE = 9
    AMBIGUOUS_REPRESENTATION = 10
    CENTER_OVERFLOW = 11
    CENTER_UNDERFLOW = 12
    SCALING_SUBNORMAL = 13
    SCALING_NOT_REVERSIBLE = 14
    UNSAFE_SUBTRACTION = 15
    EXACT_SINGULAR = 16
    SVD_FAILED = 17
    NUMERICALLY_UNRESOLVED = 18
    EXACT_ROUNDED = 19
    FLOATING_ESTIMATE = 20


_STATUS_LABELS: Mapping[_StatusCode, str] = MappingProxyType(
    {
        _StatusCode.UNAVAILABLE: "unavailable",
        _StatusCode.FLOATING_FILTER: "floating_filter",
        _StatusCode.EXACT_DYADIC: "exact_dyadic",
        _StatusCode.MISSING_NODES: "missing_nodes",
        _StatusCode.NONFINITE_COORDINATES: "nonfinite_coordinates",
        _StatusCode.NOT_REQUESTED: "not_requested",
        _StatusCode.EXACT_ZERO: "exact_zero",
        _StatusCode.OVERFLOW: "overflow",
        _StatusCode.UNDERFLOW: "underflow",
        _StatusCode.AVAILABLE: "available",
        _StatusCode.AMBIGUOUS_REPRESENTATION: "ambiguous_representation",
        _StatusCode.CENTER_OVERFLOW: "center_overflow",
        _StatusCode.CENTER_UNDERFLOW: "center_underflow",
        _StatusCode.SCALING_SUBNORMAL: "scaling_subnormal",
        _StatusCode.SCALING_NOT_REVERSIBLE: "scaling_not_reversible",
        _StatusCode.UNSAFE_SUBTRACTION: "unsafe_subtraction",
        _StatusCode.EXACT_SINGULAR: "exact_singular",
        _StatusCode.SVD_FAILED: "svd_failed",
        _StatusCode.NUMERICALLY_UNRESOLVED: "numerically_unresolved",
        _StatusCode.EXACT_ROUNDED: "exact_rounded",
        _StatusCode.FLOATING_ESTIMATE: "floating_estimate",
    }
)


def _decode_label(value: int | np.integer) -> str:
    """Decode one internal byte code, rejecting corrupt or unknown values."""

    try:
        code = _StatusCode(int(value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"unknown mesh-quality status code {value!r}") from error
    try:
        return _STATUS_LABELS[code]
    except KeyError as error:  # Defensive if the enum and immutable map diverge.
        raise ValueError(f"unknown mesh-quality status code {int(code)}") from error


@dataclass(frozen=True, slots=True)
class Measurements:
    """Per-tetrahedron results; unset entries are NaN with an explanatory status code.

    ``*_method`` and ``*_status`` arrays hold ``_StatusCode`` values and are
    decoded to labels by ``_decode_label``. Volumes ending ``_m3`` are in cubic
    metres and exist only when a length scale was given.
    """

    orientation: np.ndarray
    orientation_method: np.ndarray
    determinant_estimate: np.ndarray
    determinant_error_bound: np.ndarray
    determinant_scale_exponent: np.ndarray
    signed_volume: np.ndarray
    volume: np.ndarray
    volume_status: np.ndarray
    volume_method: np.ndarray
    volume_error_bound: np.ndarray
    signed_volume_m3: np.ndarray
    volume_m3: np.ndarray
    volume_m3_status: np.ndarray
    volume_m3_method: np.ndarray
    volume_m3_error_bound: np.ndarray
    kappa_j: np.ndarray
    condition_status: np.ndarray
    centers: np.ndarray
    center_status: np.ndarray
    fallback_used: np.ndarray


def _exact_determinant(points: np.ndarray) -> Fraction:
    exact = [
        [Fraction.from_float(float(points[row, axis])) for axis in range(3)]
        for row in range(4)
    ]
    a = [exact[1][axis] - exact[0][axis] for axis in range(3)]
    b = [exact[2][axis] - exact[0][axis] for axis in range(3)]
    c = [exact[3][axis] - exact[0][axis] for axis in range(3)]
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - b[0] * (a[1] * c[2] - a[2] * c[1])
        + c[0] * (a[1] * b[2] - a[2] * b[1])
    )


def _normal_rows(values: np.ndarray) -> np.ndarray:
    """Return rows whose nonzero binary64 values are finite and normal."""

    axes = tuple(range(1, values.ndim))
    valid = np.isfinite(values) & ((values == 0.0) | (np.abs(values) >= _MIN_NORMAL))
    return np.all(valid, axis=axes)


def _filtered_determinants(
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorize the explicit determinant and matching sign bound for one chunk."""

    a = edges[:, :, 0]
    b = edges[:, :, 1]
    c = edges[:, :, 2]
    products = np.column_stack(
        (
            b[:, 1] * c[:, 2],
            b[:, 2] * c[:, 1],
            a[:, 1] * c[:, 2],
            a[:, 2] * c[:, 1],
            a[:, 1] * b[:, 2],
            a[:, 2] * b[:, 1],
        )
    )
    differences = np.column_stack(
        (
            products[:, 0] - products[:, 1],
            products[:, 2] - products[:, 3],
            products[:, 4] - products[:, 5],
        )
    )
    terms = np.column_stack(
        (
            a[:, 0] * differences[:, 0],
            -b[:, 0] * differences[:, 1],
            c[:, 0] * differences[:, 2],
        )
    )
    determinant = (terms[:, 0] + terms[:, 1]) + terms[:, 2]
    permanent = (
        np.abs(a[:, 0]) * (np.abs(products[:, 0]) + np.abs(products[:, 1]))
        + np.abs(b[:, 0]) * (np.abs(products[:, 2]) + np.abs(products[:, 3]))
        + np.abs(c[:, 0]) * (np.abs(products[:, 4]) + np.abs(products[:, 5]))
    )
    bound = (7.0 + 56.0 * _U) * _U * permanent
    intermediates = np.column_stack(
        (products, differences, terms, determinant, permanent, bound)
    )
    accepted = _normal_rows(intermediates) & (np.abs(determinant) > bound)
    return determinant, bound, accepted


def _fraction_float(value: Fraction) -> tuple[float, _StatusCode]:
    if value == 0:
        return 0.0, _StatusCode.EXACT_ZERO
    try:
        converted = float(value)
    except OverflowError:
        return math.nan, _StatusCode.OVERFLOW
    if not math.isfinite(converted):
        return math.nan, _StatusCode.OVERFLOW
    if converted == 0.0:
        return math.nan, _StatusCode.UNDERFLOW
    return converted, _StatusCode.AVAILABLE


def _restore_interval(
    estimate: float,
    bound: float,
    exponent: int,
    *,
    scale_m: float | None,
) -> tuple[float, float, _StatusCode]:
    factor = 1.0 / 6.0
    restore_exponent = 3 * exponent
    if scale_m is not None:
        mantissa, scale_exponent = math.frexp(scale_m)
        factor *= mantissa**3
        restore_exponent += 3 * scale_exponent
    low = max(0.0, abs(estimate) - bound) * factor
    middle = abs(estimate) * factor
    high = (abs(estimate) + bound) * factor
    try:
        restored = math.ldexp(middle, restore_exponent)
        restored_low = math.ldexp(low, restore_exponent)
        restored_high = math.ldexp(high, restore_exponent)
    except OverflowError:
        return math.nan, math.nan, _StatusCode.AMBIGUOUS_REPRESENTATION
    if (
        not math.isfinite(restored)
        or not math.isfinite(restored_high)
        or (low > 0.0 and restored_low == 0.0)
        or (middle > 0.0 and restored == 0.0)
    ):
        return math.nan, math.nan, _StatusCode.AMBIGUOUS_REPRESENTATION
    try:
        restored_bound = math.ldexp(bound * factor, restore_exponent)
    except OverflowError:
        return math.nan, math.nan, _StatusCode.AMBIGUOUS_REPRESENTATION
    if not math.isfinite(restored_bound):
        return math.nan, math.nan, _StatusCode.AMBIGUOUS_REPRESENTATION
    restored_bound += 16.0 * _U * (abs(restored) + abs(restored_bound)) + math.ulp(
        restored
    )
    if restored <= restored_bound:
        return math.nan, math.nan, _StatusCode.AMBIGUOUS_REPRESENTATION
    return restored, restored_bound, _StatusCode.AVAILABLE


def _exact_center(points: np.ndarray) -> tuple[np.ndarray, _StatusCode]:
    values = np.empty(3, dtype=np.float64)
    for axis in range(3):
        exact = (
            sum(
                (Fraction.from_float(float(points[row, axis])) for row in range(4)),
                Fraction(),
            )
            / 4
        )
        value, status = _fraction_float(exact)
        if status not in {_StatusCode.AVAILABLE, _StatusCode.EXACT_ZERO}:
            center_status = (
                _StatusCode.CENTER_OVERFLOW
                if status == _StatusCode.OVERFLOW
                else _StatusCode.CENTER_UNDERFLOW
            )
            return np.full(3, np.nan), center_status
        values[axis] = value
    return values, _StatusCode.AVAILABLE


def _scaled_center(
    scaled_points: np.ndarray, original_points: np.ndarray, exponent: int
) -> tuple[np.ndarray, _StatusCode]:
    totals = np.asarray(
        [
            math.fsum(float(value) for value in scaled_points[:, axis])
            for axis in range(3)
        ],
        dtype=np.float64,
    )
    try:
        restored = np.asarray(
            [math.ldexp(float(value), exponent - 2) for value in totals],
            dtype=np.float64,
        )
    except OverflowError:
        return _exact_center(original_points)
    if not np.all(np.isfinite(restored)):
        return _exact_center(original_points)
    if np.any((totals != 0.0) & (restored == 0.0)):
        return _exact_center(original_points)
    if np.any((restored != 0.0) & (np.abs(restored) < _MIN_NORMAL)):
        return _exact_center(original_points)
    return restored, _StatusCode.AVAILABLE


def _condition_chunk(
    edges: np.ndarray,
    rows: np.ndarray,
    condition_status: np.ndarray,
    kappa_j: np.ndarray,
) -> None:
    """Measure one bounded SVD chunk and isolate a failing LAPACK element."""

    batch = edges @ _INV_W
    try:
        singular_values = np.linalg.svd(batch, compute_uv=False)
        results = [
            (int(index), values, None)
            for index, values in zip(rows, singular_values, strict=True)
        ]
    except np.linalg.LinAlgError:
        results = []
        for index, matrix in zip(rows, batch, strict=True):
            try:
                values = np.linalg.svd(matrix, compute_uv=False)
                results.append((int(index), values, None))
            except np.linalg.LinAlgError as error:
                results.append((int(index), None, error))
    for index, values, error in results:
        if condition_status[index] == _StatusCode.EXACT_SINGULAR:
            continue
        if error is not None or values is None or not np.all(np.isfinite(values)):
            condition_status[index] = _StatusCode.SVD_FAILED
            continue
        maximum = float(values[0])
        minimum = float(values[-1])
        if maximum == 0.0 or minimum == 0.0 or minimum <= 3.0 * _EPS * maximum:
            condition_status[index] = _StatusCode.NUMERICALLY_UNRESOLVED
            continue
        kappa_j[index] = maximum / minimum
        condition_status[index] = _StatusCode.AVAILABLE


@dataclass
class _Workspace:
    """Writable arrays for every ``Measurements`` field, filled chunk by chunk."""

    orientation: np.ndarray
    orientation_method: np.ndarray
    determinant_estimate: np.ndarray
    determinant_error_bound: np.ndarray
    determinant_scale_exponent: np.ndarray
    signed_volume: np.ndarray
    volume: np.ndarray
    volume_status: np.ndarray
    volume_method: np.ndarray
    volume_error_bound: np.ndarray
    signed_volume_m3: np.ndarray
    volume_m3: np.ndarray
    volume_m3_status: np.ndarray
    volume_m3_method: np.ndarray
    volume_m3_error_bound: np.ndarray
    kappa_j: np.ndarray
    condition_status: np.ndarray
    centers: np.ndarray
    center_status: np.ndarray
    fallback_used: np.ndarray


def _allocate_workspace(count: int, length_scale_m: float | None) -> _Workspace:
    """Create every array unset: NaN values, UNAVAILABLE statuses."""
    m3_status = (
        _StatusCode.NOT_REQUESTED if length_scale_m is None else _StatusCode.UNAVAILABLE
    )
    return _Workspace(
        orientation=np.zeros(count, dtype=np.int8),
        orientation_method=np.full(count, _StatusCode.UNAVAILABLE, dtype=np.uint8),
        determinant_estimate=np.full(count, np.nan, dtype=np.float64),
        determinant_error_bound=np.full(count, np.nan, dtype=np.float64),
        determinant_scale_exponent=np.zeros(count, dtype=np.int32),
        signed_volume=np.full(count, np.nan, dtype=np.float64),
        volume=np.full(count, np.nan, dtype=np.float64),
        volume_status=np.full(count, _StatusCode.UNAVAILABLE, dtype=np.uint8),
        volume_method=np.full(count, _StatusCode.UNAVAILABLE, dtype=np.uint8),
        volume_error_bound=np.full(count, np.nan, dtype=np.float64),
        signed_volume_m3=np.full(count, np.nan, dtype=np.float64),
        volume_m3=np.full(count, np.nan, dtype=np.float64),
        volume_m3_status=np.full(count, m3_status, dtype=np.uint8),
        volume_m3_method=np.full(count, m3_status, dtype=np.uint8),
        volume_m3_error_bound=np.full(count, np.nan, dtype=np.float64),
        kappa_j=np.full(count, np.nan, dtype=np.float64),
        condition_status=np.full(count, _StatusCode.UNAVAILABLE, dtype=np.uint8),
        centers=np.full((count, 3), np.nan, dtype=np.float64),
        center_status=np.full(count, _StatusCode.UNAVAILABLE, dtype=np.uint8),
        fallback_used=np.zeros(count, dtype=np.bool_),
    )


def measure(mesh: MeshData, length_scale_m: float | None) -> Measurements:
    """Measure all locatable type-4 elements without changing their orientation.

    Elements are processed in chunks of 4096 to bound memory. Each chunk first uses
    fast floating-point filters whose error bounds certify the result, and falls
    back to exact rational arithmetic for any element the filters cannot decide.
    """
    count = len(mesh.element_ids)
    work = _allocate_workspace(count, length_scale_m)
    chunk_size = 4096
    for start in range(0, count, chunk_size):
        rows = np.arange(start, min(start + chunk_size, count), dtype=np.int64)
        _measure_chunk(mesh, work, rows, length_scale_m)
    return Measurements(
        **{
            item.name: _readonly(getattr(work, item.name))
            for item in dataclasses.fields(work)
        }
    )


def _mark_unmeasurable(
    work: _Workspace,
    rows: np.ndarray,
    status: _StatusCode,
    length_scale_m: float | None,
) -> None:
    """Record why these elements have no measurements (missing or non-finite input)."""
    work.orientation_method[rows] = status
    work.volume_status[rows] = status
    work.volume_method[rows] = status
    work.condition_status[rows] = status
    work.center_status[rows] = status
    if length_scale_m is not None:
        work.volume_m3_status[rows] = status
        work.volume_m3_method[rows] = status


def _measure_chunk(
    mesh: MeshData,
    work: _Workspace,
    rows: np.ndarray,
    length_scale_m: float | None,
) -> None:
    """Measure one chunk of elements into ``work``."""
    indices = mesh.node_indices[rows]
    missing = np.any(indices < 0, axis=1)
    _mark_unmeasurable(work, rows[missing], _StatusCode.MISSING_NODES, length_scale_m)

    present_rows = rows[~missing]
    if not len(present_rows):
        return
    present_points = mesh.coordinates[mesh.node_indices[present_rows]]
    finite = np.all(np.isfinite(present_points), axis=(1, 2))
    _mark_unmeasurable(
        work, present_rows[~finite], _StatusCode.NONFINITE_COORDINATES, length_scale_m
    )

    valid_rows = present_rows[finite]
    points = present_points[finite]
    if not len(valid_rows):
        return
    scaling = _scale_to_unit_range(work, valid_rows, points)
    safe = _classify_scaling(work, valid_rows, points, scaling)
    _assign_centers(work, valid_rows, safe, points, scaling)

    safe_rows = valid_rows[safe]
    safe_edges = scaling.edges[safe]
    accepted_rows = _filter_orientation(work, safe_rows, safe_edges)

    exact = _ExactFallback(mesh, work)
    rejected_rows = np.concatenate(
        (
            valid_rows[~safe],
            safe_rows[
                work.orientation_method[safe_rows] != _StatusCode.FLOATING_FILTER
            ],
        )
    )
    for row_value in rejected_rows:
        exact.apply_to_volume(int(row_value))
    for row_value in accepted_rows:
        _restore_filtered_volume(work, exact, int(row_value))

    if length_scale_m is not None:
        _fill_si_volumes(work, exact, valid_rows, length_scale_m)

    if len(safe_rows):
        _condition_chunk(safe_edges, safe_rows, work.condition_status, work.kappa_j)


class _Scaling(NamedTuple):
    """One chunk after power-of-two scaling."""

    exponents: np.ndarray
    scaled: np.ndarray
    restored: np.ndarray
    edges: np.ndarray


def _scale_to_unit_range(
    work: _Workspace, valid_rows: np.ndarray, points: np.ndarray
) -> _Scaling:
    """Scale each element by a power of two and form its edge vectors.

    Power-of-two scaling is exact, so it can be undone bit for bit; ``restored``
    is the round trip that ``_classify_scaling`` compares with the input.
    """
    maximum = np.max(np.abs(points), axis=(1, 2))
    exponents = np.frexp(maximum)[1].astype(np.int32)
    exponents[maximum == 0.0] = 0
    work.determinant_scale_exponent[valid_rows] = exponents
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        scaled = np.ldexp(points, -exponents[:, None, None])
        restored = np.ldexp(scaled, exponents[:, None, None])
        edges = np.stack(
            (
                scaled[:, 1] - scaled[:, 0],
                scaled[:, 2] - scaled[:, 0],
                scaled[:, 3] - scaled[:, 0],
            ),
            axis=2,
        )
    return _Scaling(exponents, scaled, restored, edges)


def _classify_scaling(
    work: _Workspace,
    valid_rows: np.ndarray,
    points: np.ndarray,
    scaling: _Scaling,
) -> np.ndarray:
    """Return which elements may use the floating filter; record why others may not."""
    scaled_normal = _normal_rows(scaling.scaled)
    reversible = scaled_normal & np.all(scaling.restored == points, axis=(1, 2))
    reversible &= np.all((points == 0.0) | (scaling.scaled != 0.0), axis=(1, 2))
    edge_normal = _normal_rows(scaling.edges)
    safe = reversible & edge_normal

    work.condition_status[valid_rows[~scaled_normal]] = _StatusCode.SCALING_SUBNORMAL
    work.condition_status[valid_rows[scaled_normal & ~reversible]] = (
        _StatusCode.SCALING_NOT_REVERSIBLE
    )
    work.condition_status[valid_rows[reversible & ~edge_normal]] = (
        _StatusCode.UNSAFE_SUBTRACTION
    )
    return safe


def _assign_centers(
    work: _Workspace,
    valid_rows: np.ndarray,
    safe: np.ndarray,
    points: np.ndarray,
    scaling: _Scaling,
) -> None:
    """Compute element centers from the scaled points, or exactly when unsafe."""
    for local, row in enumerate(valid_rows):
        if safe[local]:
            work.centers[row], work.center_status[row] = _scaled_center(
                scaling.scaled[local], points[local], int(scaling.exponents[local])
            )
        else:
            work.centers[row], work.center_status[row] = _exact_center(points[local])


def _filter_orientation(
    work: _Workspace, safe_rows: np.ndarray, safe_edges: np.ndarray
) -> np.ndarray:
    """Decide orientation by the floating filter; return the rows it certified."""
    accepted_rows = np.empty(0, dtype=np.int64)
    if len(safe_rows):
        estimates, bounds, accepted = _filtered_determinants(safe_edges)
        work.determinant_estimate[safe_rows] = estimates
        work.determinant_error_bound[safe_rows] = bounds
        accepted_rows = safe_rows[accepted]
        work.orientation[accepted_rows] = np.where(estimates[accepted] > 0.0, 1, -1)
        work.orientation_method[accepted_rows] = _StatusCode.FLOATING_FILTER
    return accepted_rows


class _ExactFallback:
    """Exact-arithmetic determinants of one chunk, computed once per element."""

    def __init__(self, mesh: MeshData, work: _Workspace) -> None:
        self._mesh = mesh
        self._work = work
        self.cache: dict[int, Fraction] = {}

    def determinant(self, row: int) -> Fraction:
        """Return the exact determinant of ``row``, computing it on first use."""
        if row not in self.cache:
            self.cache[row] = _exact_determinant(
                self._mesh.coordinates[self._mesh.node_indices[row]]
            )
        return self.cache[row]

    def apply_to_volume(self, row: int) -> None:
        """Set orientation and volume of ``row`` from the exact determinant."""
        work = self._work
        exact_determinant = self.determinant(row)
        work.fallback_used[row] = True
        sign = 1 if exact_determinant > 0 else -1 if exact_determinant < 0 else 0
        work.orientation[row] = sign
        work.orientation_method[row] = _StatusCode.EXACT_DYADIC
        exact_signed = exact_determinant / 6
        magnitude, status = _fraction_float(abs(exact_signed))
        signed, signed_status = _fraction_float(exact_signed)
        work.volume_status[row] = status
        work.volume_method[row] = _StatusCode.EXACT_ROUNDED
        if status in {_StatusCode.AVAILABLE, _StatusCode.EXACT_ZERO}:
            work.volume[row] = magnitude
        if signed_status in {_StatusCode.AVAILABLE, _StatusCode.EXACT_ZERO}:
            work.signed_volume[row] = signed
        if sign == 0:
            work.condition_status[row] = _StatusCode.EXACT_SINGULAR
            work.kappa_j[row] = math.inf


def _restore_filtered_volume(work: _Workspace, exact: _ExactFallback, row: int) -> None:
    """Undo the scaling on a filter-certified volume; go exact if that is ambiguous."""
    restored_volume, restored_bound, restored_status = _restore_interval(
        float(work.determinant_estimate[row]),
        float(work.determinant_error_bound[row]),
        int(work.determinant_scale_exponent[row]),
        scale_m=None,
    )
    if restored_status == _StatusCode.AVAILABLE:
        work.volume[row] = restored_volume
        work.signed_volume[row] = math.copysign(restored_volume, work.orientation[row])
        work.volume_status[row] = _StatusCode.AVAILABLE
        work.volume_method[row] = _StatusCode.FLOATING_ESTIMATE
        work.volume_error_bound[row] = restored_bound
    else:
        exact.apply_to_volume(row)


def _fill_si_volumes(
    work: _Workspace,
    exact: _ExactFallback,
    valid_rows: np.ndarray,
    length_scale_m: float,
) -> None:
    """Convert volumes to cubic metres with the caller's length scale in metres."""
    exact_scale = Fraction.from_float(length_scale_m) ** 3
    for row_value in valid_rows:
        row = int(row_value)
        if row not in exact.cache:
            restored_si, restored_si_bound, status_si = _restore_interval(
                float(work.determinant_estimate[row]),
                float(work.determinant_error_bound[row]),
                int(work.determinant_scale_exponent[row]),
                scale_m=length_scale_m,
            )
            if status_si == _StatusCode.AVAILABLE:
                work.volume_m3[row] = restored_si
                work.signed_volume_m3[row] = math.copysign(
                    restored_si, work.orientation[row]
                )
                work.volume_m3_status[row] = _StatusCode.AVAILABLE
                work.volume_m3_method[row] = _StatusCode.FLOATING_ESTIMATE
                work.volume_m3_error_bound[row] = restored_si_bound
                continue
            exact.determinant(row)
            work.fallback_used[row] = True
        exact_signed_si = exact.cache[row] * exact_scale / 6
        value_si, status_si = _fraction_float(abs(exact_signed_si))
        signed_si, signed_si_status = _fraction_float(exact_signed_si)
        work.volume_m3_status[row] = status_si
        work.volume_m3_method[row] = _StatusCode.EXACT_ROUNDED
        if status_si in {_StatusCode.AVAILABLE, _StatusCode.EXACT_ZERO}:
            work.volume_m3[row] = value_si
        if signed_si_status in {_StatusCode.AVAILABLE, _StatusCode.EXACT_ZERO}:
            work.signed_volume_m3[row] = signed_si
