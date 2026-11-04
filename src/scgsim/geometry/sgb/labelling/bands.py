"""The distance band of a raw face (design brief sections 5 and 6, stage S5 labelling).

A raw face lies wholly inside one band of a support region: the offset curves that bound the
bands are arrangement edges, so no face crosses one.  What a single sample point cannot tell is
which band a *zero-area* locus belongs to: where two offsets coincide (a gap exactly twice the
margin wide, the medial axis of a thin strip) the distance equals the margin on a line, and
``d >= m`` puts that line in the upper band while the area around it is in the lower one.

So the band is decided from the face's area.  The face is cut into trapezoids whose exact areas
add up to the face; each votes with the first of its sample points that is clear of every band
boundary, and the face takes the band that holds its area.  Two bands with area each mean a cut
is missing, which stops the build.

A lumped-port face is the exception: band seams stop at the port boundary, so the one face of a port may
lie across several bands.  It is no band's surface (its area is left out of every band group), yet it lies
in the support region and the coverage check counts it, so it carries band 0 of every family whose support
holds it (``nominal``).
"""

from __future__ import annotations

from fractions import Fraction

from scgsim.geometry.sgb.arrangement.dcel import (
    Arrangement,
    interior_point,
    strip_cells,
)
from scgsim.geometry.sgb.errors import ValidationError
from scgsim.geometry.sgb.records.support import DISTANCE_SLACK, SupportRegion

CLEARANCE = DISTANCE_SLACK
"""A sample counts only if its distance to the origins differs from every margin by more than the
slack of a cut curve: closer than that the exact distance does not tell which side of the cut it is on."""

OUTSIDE = -1
"""Verdict of a sample outside the support region (band indices are non-negative)."""


def face_band(
    arr: Arrangement,
    face_index: int,
    support: SupportRegion,
    *,
    nominal: bool = False,
) -> int | None:
    """Band of a raw face in the support region (0 = closest to the origins); ``None`` outside it.

    ``nominal`` is for a port face, which may straddle bands: see :func:`_nominal_band`.
    """
    if nominal:
        return _nominal_band(arr, face_index, support)
    area_by_verdict: dict[int, Fraction] = {}
    for cell in strip_cells(arr, face_index):
        verdicts = {v for p in cell.points if (v := _verdict(p, support)) is not None}
        if len(verdicts) > 1:
            raise _straddle(arr, face_index, verdicts)
        for verdict in verdicts:
            area_by_verdict[verdict] = (
                area_by_verdict.get(verdict, Fraction(0)) + cell.area
            )
    if len(area_by_verdict) > 1:
        raise _straddle(arr, face_index, set(area_by_verdict))
    if not area_by_verdict:
        # No sample clear of every margin: the face is as thin as the slack of a cut, so either
        # neighbouring band is as exact as the cut itself.
        return support.band_of(interior_point(arr, face_index))
    (verdict,) = area_by_verdict
    return None if verdict == OUTSIDE else verdict


def _nominal_band(
    arr: Arrangement, face_index: int, support: SupportRegion
) -> int | None:
    """Band 0 for a face inside the support region, ``None`` outside.

    A port face is no band's surface and the seams stop at its boundary, so it may lie across several bands.
    One fixed label keeps its pieces alike wherever a user curve cuts it, and the support boundary is an
    arrangement edge, so the face lies wholly inside or outside the region.
    """
    return 0 if support.contains(interior_point(arr, face_index)) else None


def _straddle(arr: Arrangement, face_index: int, verdicts: set[int]) -> ValidationError:
    return ValidationError(
        f"raw face at {arr.face_anchor(face_index)} holds area in several bands "
        f"{sorted(v for v in verdicts if v != OUTSIDE)}"
        f"{' and outside the region' if OUTSIDE in verdicts else ''} "
        "of one support region: a cut is missing"
    )


def _verdict(point: tuple[int, int, int], support: SupportRegion) -> int | None:
    """Band of a sample, ``OUTSIDE`` outside the region, ``None`` when it is too close to a band boundary."""
    if not support.contains(point):
        return OUTSIDE
    distance = support.distance((point[0] / point[2], point[1] / point[2]))
    if any(abs(distance - margin) <= CLEARANCE for margin in support.margins):
        return None
    return sum(1 for margin in support.margins if distance > margin)
