"""What a face comes from, and what a refinement family draws, are written from typed facts, never from ids.

A face's sources are the Entities, band families and refinement requests of the carriers that bound it.  Ids of
different kinds may be equal strings (an Entity and a request may be called alike), so the sidecar classifies each
source by the carrier's lineage, not by looking its id up in a table.  A refinement family a catalogue declares
directly is recorded by its segments exactly as they are: disjoint or listed out of order, nothing is joined.
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import pytest
from sgb_extreme_cases import catalogue_of, chip, metal, rect_um

from scgsim.contracts.sgb_sidecar import read_sidecar, views
from scgsim.geometry.sgb.build import emit_model
from scgsim.geometry.sgb.bands.requests import (
    ConductorSurface,
    InsetRequest,
    RefinementRequest,
)
from scgsim.geometry.sgb.model import build_model_from_catalogue
from scgsim.geometry.sgb.exact.carriers import Lineage, LineCarrier
from scgsim.geometry.sgb.exact.families import CurveFamilyRecord
from scgsim.geometry.sgb.exact.prisms import PrismCatalogue
from scgsim.meshing.built_model import read_built_model

GRID_UM = 0.001


def two_sheets(other: str):
    """Route A: the sheet ``pad`` and a sheet called ``other``, side by side on the chip."""
    return [
        *chip(),
        metal("pad", "N", rect_um(-15, -5, -2, 5), "A"),
        metal(other, "M", rect_um(2, -5, 15, 5), "A"),
    ]


def sidecar_of(model, tmp_path):
    """The sidecar SGB writes for ``model``, read by the shared reader."""
    return read_sidecar(emit_model(model, tmp_path).sidecar)


def outline_faces(sidecar, entity):
    """The written faces one of whose boundary edges lies on ``entity``'s outline."""
    curves, carriers = sidecar["source_curve"], sidecar["carrier"]
    on_outline = {
        cid
        for cid, c in carriers.items()
        if c["kind"] == "source" and curves[c["source_curve"]]["entity"] == entity
    }
    return [
        fid
        for fid, face in sidecar["face"].items()
        if any(
            set(sidecar["edge"][e]["carriers"]) & on_outline
            for loop in face["loops"]
            for e in loop["edges"]
        )
    ]


def test_an_entity_named_like_a_band_family_stays_an_entity_of_the_faces_it_bounds(
    tmp_path,
):
    """The Entity ``bands`` and the band family ``bands`` of the pad: each face names what really bounds it."""
    request = InsetRequest("bands", ConductorSurface("pad", "sheet"), (0.5,))
    model = build_model_from_catalogue(
        catalogue_of("collide", "A", two_sheets("bands")), "A", insets=(request,)
    )
    sidecar = sidecar_of(model, tmp_path)
    bounded = outline_faces(sidecar, "bands")
    assert bounded
    for fid in bounded:
        assert "bands" in sidecar["face"][fid]["entities"], fid
    banded = [f for f, face in sidecar["face"].items() if "bands" in face["families"]]
    assert banded and all(
        any(label["family"] == "bands" for label in sidecar["face"][f]["bands"])
        or sidecar["face"][f]["kind"] == "wall"
        or any(
            sidecar["carrier"][c]["kind"] == "band_member"
            for loop in sidecar["face"][f]["loops"]
            for e in loop["edges"]
            for c in sidecar["edge"][e]["carriers"]
        )
        for f in banded
    )
    built = read_built_model(sidecar)
    assert all("bands" in built.face_entities[fid] for fid in bounded)


def test_an_entity_named_like_a_refinement_request_writes_a_sidecar_the_reader_accepts(
    tmp_path,
):
    """The Entity ``ref`` and the refinement request ``ref``: the sheet's faces name the Entity, and every id resolves."""
    refinement = RefinementRequest("ref", 0.0, ((-12.0, 0.0), (-4.0, 0.0)))
    model = build_model_from_catalogue(
        catalogue_of("collide", "A", two_sheets("ref")), "A", refinements=(refinement,)
    )
    sidecar = sidecar_of(model, tmp_path)  # the shared reader checks every reference
    bounded = outline_faces(sidecar, "ref")
    assert bounded and all("ref" in sidecar["face"][f]["entities"] for f in bounded)
    assert all("ref" not in face["families"] for face in sidecar["face"].values())
    ((edge, _),) = views.embedded_edges(sidecar)
    assert {
        sidecar["carrier"][c]["request"] for c in sidecar["edge"][edge]["carriers"]
    } == {"ref"}


def segment(name: str, k: int, p: tuple[int, int], q: tuple[int, int]) -> LineCarrier:
    """Segment ``k`` of the refinement family ``name``, from lattice point ``p`` to ``q``."""
    lineage = Lineage("family_member", family_id=name, member=f"{name}/seg{k}")
    return LineCarrier(f"{name}/seg{k}", p, q, lineage)


@pytest.mark.parametrize(
    "carriers",
    [
        # Two segments that do not touch.
        [((-12_000, 2_000), (-8_000, 2_000)), ((-6_000, -2_000), (-3_000, -2_000))],
        # One chain A-B-C, listed as B-C before A-B.
        [((-8_000, 0), (-4_000, 0)), ((-12_000, 0), (-8_000, 0))],
    ],
    ids=["disjoint", "out_of_order"],
)
def test_a_declared_refinement_family_is_recorded_segment_by_segment(
    carriers, tmp_path
):
    """A family with no request: no polyline is invented, and each carrier is one segment, in carrier order."""
    family = CurveFamilyRecord(
        "drawn",
        "refinement",
        tuple(segment("drawn", k, p, q) for k, (p, q) in enumerate(carriers)),
        (Fraction(0),),
        "refinement_only",
    )
    model = build_model_from_catalogue(
        catalogue_of("declared", "A", two_sheets("other"), (family,)), "A"
    )
    sidecar = sidecar_of(model, tmp_path)
    record = sidecar["refinement_request"]["drawn"]
    assert record["points_um"] is None
    assert record["segments_um"] == [
        [[p[0] * GRID_UM, p[1] * GRID_UM], [q[0] * GRID_UM, q[1] * GRID_UM]]
        for p, q in carriers
    ]
    assert sidecar["cut_family"]["drawn"]["levels_um"] == [0.0]


def test_a_declared_refinement_family_on_two_levels_records_both_heights(tmp_path):
    """The builder draws a family at every one of its target levels, so the sidecar lists every height."""
    family = CurveFamilyRecord(
        "drawn",
        "refinement",
        (segment("drawn", 0, (-12_000, 2_000), (-8_000, 2_000)),),
        (Fraction(0), Fraction(-10)),
        "refinement_only",
    )
    model = build_model_from_catalogue(
        catalogue_of("two_levels", "A", two_sheets("other"), (family,)), "A"
    )
    sidecar = sidecar_of(model, tmp_path)
    assert sidecar["cut_family"]["drawn"]["levels_um"] == [0.0, -10.0]
    drawn_at = sorted(
        sidecar["vertex"][sidecar["edge"][edge]["vertices"][0]]["position_um"][2]
        for edge, _ in views.embedded_edges(sidecar)
    )
    assert drawn_at == [-10.0, 0.0]  # one refinement edge on each level


def test_a_requested_refinement_keeps_its_polyline_and_its_segments(tmp_path):
    """A request as given: its polyline, and the segments SGB drew from it (a repeated point draws none)."""
    points = ((-12.0, 0.0), (-8.0, 0.0), (-8.0, 0.0), (-8.0, 3.0))
    model = build_model_from_catalogue(
        catalogue_of("requested", "A", two_sheets("other")),
        "A",
        refinements=(RefinementRequest("ref", 0.0, points),),
    )
    record = sidecar_of(model, tmp_path)["refinement_request"]["ref"]
    assert record["points_um"] == [list(p) for p in points]
    assert record["segments_um"] == [
        [[-12.0, 0.0], [-8.0, 0.0]],
        [[-8.0, 0.0], [-8.0, 3.0]],
    ]


# --------------------------------------------------------------------------- cut families of any role


def family_segment(
    family: str,
    k: int,
    p: tuple[int, int],
    q: tuple[int, int],
    named: str | None = None,
) -> LineCarrier:
    """Segment ``k`` of a family; its lineage names the family ``named`` (by default the family itself)."""
    lineage = Lineage("family_member", family_id=named or family, member=f"{family}{k}")
    return LineCarrier(f"{family}{k}", p, q, lineage)


def ring_family(name: str, role: str, policy: str, named: str | None = None):
    """A closed square of four segments in the gap beside the pad sheet, at z = 0."""
    ring = [
        (-75_000, -20_000),
        (-55_000, -20_000),
        (-55_000, 20_000),
        (-75_000, 20_000),
    ]
    return CurveFamilyRecord(
        name,
        role,
        tuple(
            family_segment(name, i, ring[i], ring[(i + 1) % 4], named) for i in range(4)
        ),
        (Fraction(0),),
        policy,  # type: ignore[arg-type]
    )


def test_a_required_family_of_another_role_is_written_as_its_members(tmp_path):
    """A ``port_footprint`` family (any role the catalogue gives) cuts the model; each carrier is a member of it.

    Its carriers are typed ``family_member`` records naming the family and the member, the family is a
    ``cut_family`` record with its role, edge policy and levels, and the faces it bounds list it.
    """
    from sgb_scenes import pad_in_ground  # noqa: PLC0415 -- the scene of test_curve_families_required_and_refinement_only

    base = pad_in_ground("A")
    family = ring_family("cap", "port_footprint", "required")
    model = build_model_from_catalogue(
        PrismCatalogue(base.grid, base.prisms, (family,)), "A"
    )
    sidecar = sidecar_of(model, tmp_path)
    members = {
        cid: c for cid, c in sidecar["carrier"].items() if c["kind"] == "family_member"
    }
    assert members and {c["family"] for c in members.values()} == {"cap"}
    assert {c["member"] for c in members.values()} <= {f"cap{i}" for i in range(4)}
    assert sidecar["cut_family"]["cap"] == {
        "role": "port_footprint",
        "edge_policy": "required",
        "levels_um": [0.0],
    }
    assert any("cap" in face["cut_families"] for face in sidecar["face"].values())
    built = read_built_model(sidecar)
    assert all(built.carriers[cid].family == "cap" for cid in members)


def test_a_member_naming_a_family_the_catalogue_lacks_is_kept_with_that_name(tmp_path):
    """The carriers of family ``cap`` name ``fam`` in their lineage (as test_sgb_model draws them): written as members
    of ``fam``, a cut family with no role, policy or levels, rather than refused."""
    from sgb_scenes import pad_in_ground  # noqa: PLC0415 -- the scene of test_curve_families_required_and_refinement_only

    base = pad_in_ground("A")
    family = ring_family("cap", "port_footprint", "required", named="fam")
    model = build_model_from_catalogue(
        PrismCatalogue(base.grid, base.prisms, (family,)), "A"
    )
    sidecar = sidecar_of(model, tmp_path)
    assert {
        c["family"] for c in sidecar["carrier"].values() if c["kind"] == "family_member"
    } == {"fam"}
    assert sidecar["cut_family"]["fam"] == {
        "role": None,
        "edge_policy": None,
        "levels_um": [],
    }
    assert sidecar["cut_family"]["cap"]["role"] == "port_footprint"


def test_a_refinement_only_family_of_another_role_is_embedded_and_written_as_its_members(
    tmp_path,
):
    """A refinement-only family whose role is not ``refinement`` is embedded like one, and written as members."""
    family = CurveFamilyRecord(
        "guide",
        "guide_curve",
        (family_segment("guide", 0, (-12_000, 2_000), (-8_000, 2_000)),),
        (Fraction(0),),
        "refinement_only",
    )
    model = build_model_from_catalogue(
        catalogue_of("guide", "A", two_sheets("other"), (family,)), "A"
    )
    sidecar = sidecar_of(model, tmp_path)
    ((edge, _),) = views.embedded_edges(sidecar)
    assert [sidecar["carrier"][c] for c in sidecar["edge"][edge]["carriers"]] == [
        {"kind": "family_member", "family": "guide", "member": "guide0"}
    ]
    assert sidecar["cut_family"]["guide"]["edge_policy"] == "refinement_only"
    assert "guide" not in sidecar["refinement_request"]
    assert not [f for f in sidecar["mesh_feature"].values() if f["source"] == "guide"]


def test_an_entity_whose_prisms_differ_is_written_prism_by_prism(tmp_path):
    """Two polygons of one Entity on different layers and of different materials: each prism keeps its own facts,
    and the meshing stage reads every layer of the Entity."""
    first = replace(metal("pad", "N", rect_um(-15, -5, -2, 5), "A"), layer="M1")
    second = replace(
        metal("pad", "N", rect_um(2, -5, 15, 5), "A"),
        polygon=1,
        layer="M2",
        material="Nb",
    )
    model = build_model_from_catalogue(
        catalogue_of("two_layers", "A", [*chip(), first, second]), "A"
    )
    sidecar = sidecar_of(model, tmp_path)
    prisms = sidecar["entity"]["pad"]["prisms"]
    assert [(p["polygon"], p["layer"], p["material"]) for p in prisms] == [
        (0, "M1", first.material),
        (1, "M2", "Nb"),
    ]
    assert read_built_model(sidecar).entity_layers["pad"] == ("M1", "M2")
