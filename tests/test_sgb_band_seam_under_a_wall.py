"""A band seam under a wall draws the wall's rim but names no Entity (Issue #255; needs Gmsh).

The thin-film pair with a bump: the left film is the square 0 .. 4 um, every edge a gap edge, and the bump
stands on the square 1 .. 3 um.  With margins (0.5, 1.0) um the exact erosion of the film at 1.0 um is the
bump's footprint, so the band seam at 1.0 um runs under all four bump walls.  The seam belongs to the band
family ``epr_MA::chip/left::<side>``, which is no Entity: the walls and their corner columns belong to the
bump alone, and the seam is recorded as a band family on the walls' boundary.  The scene meshes in about two
seconds per route.
"""

from __future__ import annotations

import pytest
from palace_sgb_support import mesh_run, surface_epr_specs
from sgb_inputs import BUMP_LOOP, thin_film_pair
from shapely.geometry import Polygon, box

from scgsim.contracts.sgb_sidecar import read_sidecar

MARGINS = [0.5, 1.0]
SPECS = surface_epr_specs(MA=MARGINS, MS=MARGINS, SA=[0.5])
FILM = box(0.0, 0.0, 4.0, 4.0)
BUMP = Polygon(BUMP_LOOP)


def erosion_bands(support, exposed):
    """The exposed area of each band of ``support`` eroded at ``MARGINS``: outermost band first, core last."""
    cores = [support, *(support.buffer(-m, join_style="mitre") for m in MARGINS)]
    rings = [
        outer.difference(inner) for outer, inner in zip(cores, cores[1:], strict=False)
    ]
    return [band.intersection(exposed).area for band in (*rings, cores[-1])]


def band_areas(sidecar, family):
    """The area of each band of ``family`` the sidecar's faces lie in, by band index."""
    areas = [0.0] * sidecar["band_family"][family]["band_count"]
    for face in sidecar["face"].values():
        for band in face["bands"]:
            if band["family"] == family:
                areas[band["index"]] += face["area_um2"]
    return areas


@pytest.fixture(scope="module", params=["A", "B"])
def built(request, tmp_path_factory):
    """The route, the family of the left film's MA bands, and the checked sidecar of the meshed run.

    Route B needs the films finite (profile ``solid``); its GDS bytes are stand-ins, as Route A's are.
    """
    route = request.param
    directory = tmp_path_factory.mktemp(f"seam_{route}")
    if route == "A":
        effective, family = thin_film_pair(bump=True), "epr_MA::chip/left::sheet"
    else:
        effective, family = (
            thin_film_pair(profile="solid", bump=True),
            "epr_MA::chip/left::top",
        )
    gds = directory / "stand_in.gds"
    gds.write_bytes(b"x")
    result = mesh_run(effective, route, directory, specs=SPECS, gds=gds)
    return route, family, read_sidecar(result.sidecar_path)


def test_the_bump_footprint_is_the_exact_erosion_of_the_film_at_the_outer_margin():
    """The scene's premise: the 1.0 um core of the left film is the bump's footprint."""
    assert (
        FILM.buffer(-MARGINS[-1], join_style="mitre").symmetric_difference(BUMP).area
        == 0
    )


def test_the_walls_on_the_seam_belong_to_the_bump_and_record_the_seam_as_a_band_family(
    built,
):
    """Every wall and wall column of the bump names Entities only; the seam is the walls' band family."""
    _, family, sidecar = built
    walls = [
        face
        for face in sidecar["face"].values()
        if face["group"] == "MA::chip/bump__sidewall"
    ]
    assert len(walls) == 4
    for wall in walls:
        assert wall["entity_coverage"] == {"chip/bump": 1.0}
        assert wall["band_families"] == [family]
    columns = [
        edge
        for edge in sidecar["edge"].values()
        if edge["kind"] == "vertical" and "chip/bump" in edge["entities"]
    ]
    assert columns and all(edge["entities"] == ["chip/bump"] for edge in columns)


def test_the_left_film_has_the_bands_of_its_exact_erosion(built):
    """Route A's sheet keeps its core under the bump; Route B's top face hides it as a contact with the bump."""
    route, family, sidecar = built
    exposed = FILM if route == "A" else FILM.difference(BUMP)
    assert band_areas(sidecar, family) == pytest.approx(
        erosion_bands(FILM, exposed), abs=1e-9
    )
    hidden = sidecar["band_family"][family]["hidden_contacts"]
    if route == "A":
        assert hidden == {}
    else:
        assert hidden == {
            "2": [{"entities": ["chip/bump", "chip/left"], "face_kind": "interface"}]
        }
