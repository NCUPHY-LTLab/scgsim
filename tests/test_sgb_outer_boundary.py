"""Every face on the model's outer boundary names the side of the model's box it lies on and the domain it bounds.

A wave port (#187) takes the faces of one outer wall, while the groups put every die-edge wall of a domain into one
``EXT::<domain>__<face kind>`` group, so the side is a field of each face (#170).  The checks are against the
written vertices: a face on side ``-x`` has every vertex at the lowest x of the model, and so on.
"""

from __future__ import annotations

import copy

import pytest
from sgb_supported_kit import Conductor, Scene, catalogue, lattice

from scgsim.contracts.sgb_sidecar import SidecarError, check_sidecar, read_sidecar
from scgsim.geometry.sgb.build import emit_model
from scgsim.geometry.sgb.model import build_model_from_catalogue
from scgsim.geometry.sgb.records.grid import exact_z
from scgsim.geometry.sgb.records.prisms import Footprint, PrismRecord

HALF = 50.0
PAD = [
    (lattice(x), lattice(y)) for x, y in ((-10, -10), (10, -10), (10, 10), (-10, 10))
]
# An L-shaped die: the box (-50, -50)-(50, 50) without its (0, 0)-(50, 50) quarter.
L_DIE = [
    (lattice(x), lattice(y))
    for x, y in ((-50, -50), (50, -50), (50, 0), (0, 0), (0, 50), (-50, 50))
]
AXIS = {"-x": (0, min), "+x": (0, max), "-y": (1, min), "+y": (1, max)}


def sidecar_of(scene: Scene, tmp_path):
    """The sidecar of a scene built on Route B, as the shared reader reads it."""
    model = build_model_from_catalogue(catalogue(scene), "B")
    return read_sidecar(emit_model(model, tmp_path).sidecar)


def pad_scene(prisms: list[PrismRecord] | None = None) -> Scene:
    """A 20 um pad on a 100 um die (or on the die the prisms draw)."""
    return Scene(
        "pad",
        (),
        (-HALF, -HALF, HALF, HALF),
        [Conductor("pad", "N", PAD)],
        prisms=prisms,
    )


def l_die_prisms() -> list[PrismRecord]:
    """Substrate below z = 0 and vacuum above, both on the L-shaped die."""
    die = Footprint.from_rings(L_DIE)
    return [
        PrismRecord(
            "substrate", die, exact_z(-100), exact_z(0), "substrate", "dielectric"
        ),
        PrismRecord("vacuum", die, exact_z(0), exact_z(100), "vacuum", "vacuum"),
    ]


def vertices(sidecar, face) -> list[list[float]]:
    """The written positions of a face's vertices."""
    edges, places = sidecar["edge"], sidecar["vertex"]
    return [
        places[v]["position_um"]
        for loop in face["loops"]
        for e in loop["edges"]
        for v in edges[e]["vertices"]
    ]


def test_each_outer_face_names_its_side_of_the_box_and_its_domain(tmp_path):
    """Four walls per domain and the bottom and top, each on the plane its side names; the groups are as before."""
    sidecar = sidecar_of(pad_scene(), tmp_path)
    positions = [v["position_um"] for v in sidecar["vertex"].values()]
    low = [min(p[k] for p in positions) for k in range(3)]
    high = [max(p[k] for p in positions) for k in range(3)]
    found = {}
    for face_id, face in sidecar["face"].items():
        outer = face["outer_boundary"]
        assert (outer is not None) == (
            None in (face["negative_volume"], face["positive_volume"])
        )
        if outer is None:
            continue
        found[face_id] = (outer["side"], outer["domain"])
        points = vertices(sidecar, face)
        if outer["side"] in AXIS:
            axis, end = AXIS[outer["side"]]
            assert all(p[axis] == end(low[axis], high[axis]) for p in points)
            assert face["kind"] == "wall"
        else:
            assert all(
                p[2] == (low[2] if outer["side"] == "bottom" else high[2])
                for p in points
            )
            assert face["kind"] == "horizontal"
        assert sidecar["group"][face["group"]]["interface_type"] == "EXT"
    assert sorted(found.values()) == sorted(
        [(side, domain) for side in AXIS for domain in ("substrate", "vacuum")]
        + [("bottom", "substrate"), ("top", "vacuum")]
    )


def test_a_wall_on_no_plane_of_the_box_has_no_side(tmp_path):
    """The two walls of the L-shaped die's inner corner lie inside the box: an outer face with no side."""
    sidecar = sidecar_of(pad_scene(l_die_prisms()), tmp_path)
    sideless = [
        face
        for face in sidecar["face"].values()
        if face["outer_boundary"] and face["outer_boundary"]["side"] is None
    ]
    # The inner walls of the substrate and of the vacuum (split where a level crosses them), on x = 0 or y = 0.
    assert {f["outer_boundary"]["domain"] for f in sideless} == {"substrate", "vacuum"}
    for face in sideless:
        points = vertices(sidecar, face)
        assert {p[0] for p in points} == {0.0} or {p[1] for p in points} == {0.0}
    sides = {
        f["outer_boundary"]["side"]
        for f in sidecar["face"].values()
        if f["outer_boundary"]
    }
    assert sides == {"-x", "+x", "-y", "+y", "bottom", "top", None}


def test_the_reader_refuses_an_outer_face_without_its_record_or_with_another_domain(
    tmp_path,
):
    """The record is on exactly the faces with one side volume null, and names the Entity of the other side."""
    sidecar = sidecar_of(pad_scene(), tmp_path)
    face_id, face = next(
        (i, f) for i, f in sidecar["face"].items() if f["outer_boundary"]
    )
    missing = copy.deepcopy(sidecar)
    missing["face"][face_id]["outer_boundary"] = None
    with pytest.raises(SidecarError, match="has outer_boundary None"):
        check_sidecar(missing)
    other = copy.deepcopy(sidecar)
    domain = face["outer_boundary"]["domain"]
    other["face"][face_id]["outer_boundary"]["domain"] = (
        "vacuum" if domain == "substrate" else "substrate"
    )
    with pytest.raises(SidecarError, match="names the domain"):
        check_sidecar(other)
