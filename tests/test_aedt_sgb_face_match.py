"""The imported-face matcher: one to one by geometry, never by order, and fail closed otherwise."""

from __future__ import annotations

import pytest

from scgsim.aedt.runtime.families.sgb_faces import (
    FaceMatchError,
    ImportedFace,
    area_tolerance_um2,
    import_tolerance_um,
    match_solid_faces,
)

TOL = 1e-3  # um, the readback tolerance of these cases


def _square(centre):
    """The corners of the 10 um cube face centred at ``centre`` (one coordinate 0 or 10, the others 5)."""
    fixed = next(k for k in range(3) if centre[k] in (0, 10))
    u, v = (k for k in range(3) if k != fixed)
    corners = []
    for a, b in ((0, 0), (10, 0), (10, 10), (0, 10)):
        point = list(centre)
        point[u], point[v] = a, b
        corners.append(tuple(float(x) for x in point))
    return tuple(corners)


def _guard(face: str, centre, area=100.0, vertices=None, curved=False):
    vertices = _square(centre) if vertices is None else vertices
    return {
        "face": face,
        "area_um2": area,
        "centre_um": list(centre),
        "vertex_centroid_um": list(centre),
        "vertices_um": [list(v) for v in vertices],
        "curved": curved,
    }


def _imported(face_id, centre, area=100.0, vertices=None, *, planar=True):
    vertices = _square(centre) if vertices is None else vertices
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
    moved = (5, 5, 10 + 2 * TOL)
    reported = [
        _imported(1, (5, 5, 0)),
        _imported(2, moved, vertices=_square((5, 5, 10))),
    ]
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
    near = ((0, 0, 0.0005), (10, 0, 0.0005), (10, 10, 0.0005), (0, 10, 0.0005))
    guard = [_guard("F1", (5, 5, 0)), _guard("F2", (5, 5, 0.0005), vertices=near)]
    reported = [_imported(1, (5, 5, 0)), _imported(2, (5, 5, 0.0005), vertices=near)]
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


def test_swapped_centres_of_two_equal_planar_faces_are_refused() -> None:
    """Two opposite faces of equal area whose guarded centres were swapped match nothing (#126 round 3 B1).

    Area and centre alone would bind each id to the other face; the vertices keep each record on its face.
    """
    bottom, top = _guard("F1", CENTRES["F1"]), _guard("F2", CENTRES["F2"])
    bottom["centre_um"], top["centre_um"] = top["centre_um"], bottom["centre_um"]
    reported = [_imported(1, CENTRES["F1"]), _imported(2, CENTRES["F2"])]
    with pytest.raises(FaceMatchError, match="no AEDT face .* planned face F1"):
        match_solid_faces("cube", [bottom, top], reported, TOL)


def test_a_planar_face_whose_vertices_moved_beyond_the_tolerance_has_no_candidate() -> (
    None
):
    """A planar face at the planned centre and area but with a vertex elsewhere is not the planned face."""
    guard = [_guard("F1", CENTRES["F1"])]
    corners = list(_square(CENTRES["F1"]))
    corners[0] = (corners[0][0] + 2 * TOL, corners[0][1], corners[0][2])
    with pytest.raises(FaceMatchError, match="no AEDT face"):
        match_solid_faces(
            "cube", guard, [_imported(1, CENTRES["F1"], vertices=tuple(corners))], TOL
        )


def test_the_readback_tolerance_is_a_thousandth_of_the_grid() -> None:
    """Decision 9: 5e-4 um on a 0.5 um grid; a non-positive grid is refused."""
    assert import_tolerance_um(0.5) == pytest.approx(5e-4)
    with pytest.raises(ValueError, match="grid_um"):
        import_tolerance_um(0.0)
