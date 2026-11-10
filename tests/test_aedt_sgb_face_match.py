"""The imported-face matcher: one to one by geometry, never by order, and fail closed otherwise."""

from __future__ import annotations

import pytest

from scgsim.aedt.runtime.families.sgb_faces import (
    FaceMatchError,
    ImportedFace,
    area_tolerance_um2,
    match_solid_faces,
)

TOL = 1e-3  # um, the sidecar tolerance of these cases


def _guard(face: str, centre, area=100.0, vertices=(), curved=False):
    return {
        "face": face,
        "area_um2": area,
        "centre_um": list(centre),
        "vertex_centroid_um": list(centre),
        "vertices_um": [list(v) for v in vertices],
        "curved": curved,
    }


def _imported(face_id, centre, area=100.0, vertices=(), *, planar=True):
    return ImportedFace(face_id, area, tuple(centre), tuple(vertices), 40.0, planar)


# The six faces of a 10 um cube, planned in one order and reported by AEDT in another.
CENTRES = {
    "F1": (5, 5, 0),
    "F2": (5, 5, 10),
    "F3": (0, 5, 5),
    "F4": (10, 5, 5),
    "F5": (5, 0, 5),
    "F6": (5, 10, 5),
}


def test_faces_match_by_geometry_whatever_order_aedt_reports() -> None:
    """Each planned face gets the AEDT face at its place, whatever order AEDT lists them in."""
    guard = [_guard(name, centre) for name, centre in CENTRES.items()]
    reported = [
        _imported(10 + k, CENTRES[name])
        for k, name in enumerate(["F4", "F1", "F6", "F2", "F5", "F3"])
    ]
    assert match_solid_faces("cube", guard, reported, TOL) == {
        "F4": 10,
        "F1": 11,
        "F6": 12,
        "F2": 13,
        "F5": 14,
        "F3": 15,
    }


def test_a_different_face_count_fails() -> None:
    """A solid with a face more or less than planned is refused before any matching."""
    guard = [_guard(name, centre) for name, centre in CENTRES.items()]
    reported = [_imported(k, centre) for k, centre in enumerate(CENTRES.values())][:5]
    with pytest.raises(FaceMatchError, match="reports 5 faces, the guard planned 6"):
        match_solid_faces("cube", guard, reported, TOL)


def test_a_face_moved_beyond_the_tolerance_has_no_candidate() -> None:
    """A centre farther than tolerance_um from the planned one matches nothing."""
    guard = [_guard("F1", (5, 5, 0)), _guard("F2", (5, 5, 10))]
    reported = [_imported(1, (5, 5, 0)), _imported(2, (5, 5, 10 + 2 * TOL))]
    with pytest.raises(FaceMatchError, match="no AEDT face .* planned face F2"):
        match_solid_faces("cube", guard, reported, TOL)


def test_an_area_beyond_the_perimeter_tolerance_has_no_candidate() -> None:
    """An area just inside the decision-3 tolerance matches; just outside it does not."""
    limit = area_tolerance_um2(100.0, 40.0, TOL)
    guard = [_guard("F1", (5, 5, 0))]
    assert match_solid_faces(
        "s", guard, [_imported(1, (5, 5, 0), 100.0 + 0.9 * limit)], TOL
    )
    with pytest.raises(FaceMatchError, match="no AEDT face"):
        match_solid_faces(
            "s", guard, [_imported(1, (5, 5, 0), 100.0 + 1.1 * limit)], TOL
        )


def test_two_candidates_inside_tolerance_are_an_ambiguity() -> None:
    """Two AEDT faces within tolerance of one planned face are refused, not picked between."""
    # Two coincident-looking faces: the matcher must refuse rather than pick one.
    guard = [_guard("F1", (5, 5, 0)), _guard("F2", (5, 5, 0.0005))]
    reported = [_imported(1, (5, 5, 0)), _imported(2, (5, 5, 0.0005))]
    with pytest.raises(FaceMatchError, match=r"AEDT faces \[1, 2\]"):
        match_solid_faces("s", guard, reported, TOL)


def test_a_curved_face_matches_by_its_vertices_not_its_reported_centre() -> None:
    """A curved face matches by its vertex set, though AEDT reports its vertex centroid."""
    arc_vertices = ((0, 0, 0), (0, 0, 1), (10, 0, 0), (10, 0, 1))
    guard = [_guard("C1", (5, 3.2, 0.5), 31.4, arc_vertices, curved=True)]
    # PyAEDT gives the vertex centroid of a curved face, far from its centre of mass.
    reported = [
        _imported(7, (5, 0, 0.5), 31.4, tuple(reversed(arc_vertices)), planar=False)
    ]
    assert match_solid_faces("pad", guard, reported, TOL) == {"C1": 7}


def test_a_curved_face_does_not_match_a_planar_one() -> None:
    """A planar AEDT face never stands in for a planned curved face."""
    vertices = ((0, 0, 0), (10, 0, 0))
    guard = [_guard("C1", (5, 0, 0), 31.4, vertices, curved=True)]
    reported = [_imported(7, (5, 0, 0), 31.4, vertices, planar=True)]
    with pytest.raises(FaceMatchError, match="curved"):
        match_solid_faces("pad", guard, reported, TOL)
