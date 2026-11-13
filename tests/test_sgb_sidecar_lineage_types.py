"""What a face comes from, and what a cut family draws, are written from typed facts, never from ids.

A face's sources are the Entities, band families and other cut families of the carriers that bound it.  Ids of
different kinds may be equal strings (an Entity and a band family may be called alike), so the sidecar classifies
each source by the carrier's lineage, not by looking its id up in a table.
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

from sgb_extreme_cases import catalogue_of, chip, metal, rect_um

from scgsim.contracts.sgb_sidecar import read_sidecar
from scgsim.geometry.sgb.bands.requests import ConductorSurface, InsetRequest
from scgsim.geometry.sgb.build import emit_model
from scgsim.geometry.sgb.exact.carriers import Lineage, LineCarrier
from scgsim.geometry.sgb.exact.families import CurveFamilyRecord
from scgsim.geometry.sgb.exact.prisms import PrismCatalogue
from scgsim.geometry.sgb.model import build_model_from_catalogue
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
    banded = [
        f for f, face in sidecar["face"].items() if "bands" in face["band_families"]
    ]
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
    return LineCarrier(f"{family}{k}", p, q, lineage, "family_line")


def ring_family(
    name: str,
    role: str,
    named: str | None = None,
    levels: tuple[Fraction, ...] = (Fraction(0),),
):
    """A closed square of four segments in the gap beside the pad sheet, at z = 0 unless ``levels`` says."""
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
        levels,
    )


def test_a_required_family_of_another_role_is_written_as_its_members(tmp_path):
    """A ``port_footprint`` family (any role the catalogue gives) cuts the model; each carrier is a member of it.

    Its carriers are typed ``family_member`` records naming the family and the member, the family is a
    ``cut_family`` record with its role and levels, and the faces it bounds list it.
    """
    from sgb_scenes import pad_in_ground  # noqa: PLC0415 -- the scene of test_a_curve_family_splits_a_cell_into_faces_of_equal_label

    base = pad_in_ground("A")
    family = ring_family("cap", "port_footprint")
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
        "levels_um": [0.0],
    }
    assert any("cap" in face["other_cut_families"] for face in sidecar["face"].values())
    built = read_built_model(sidecar)
    assert all(built.carriers[cid].family == "cap" for cid in members)


def test_a_member_naming_a_family_the_catalogue_lacks_is_kept_with_that_name(tmp_path):
    """The carriers of family ``cap`` name ``fam`` in their lineage (as test_sgb_model draws them): written as members
    of ``fam``, a cut family with no role or levels, rather than refused."""
    from sgb_scenes import pad_in_ground  # noqa: PLC0415 -- the scene of test_a_curve_family_splits_a_cell_into_faces_of_equal_label

    base = pad_in_ground("A")
    family = ring_family("cap", "port_footprint", named="fam")
    model = build_model_from_catalogue(
        PrismCatalogue(base.grid, base.prisms, (family,)), "A"
    )
    sidecar = sidecar_of(model, tmp_path)
    assert {
        c["family"] for c in sidecar["carrier"].values() if c["kind"] == "family_member"
    } == {"fam"}
    assert sidecar["cut_family"]["fam"] == {
        "role": None,
        "levels_um": [],
    }
    assert sidecar["cut_family"]["cap"]["role"] == "port_footprint"


def test_a_family_on_two_levels_records_both_heights(tmp_path):
    """The builder draws a family at every one of its target levels, so the sidecar lists every height."""
    gap = [(-1_000, -2_000), (1_000, -2_000), (1_000, 2_000), (-1_000, 2_000)]
    family = CurveFamilyRecord(
        "cap",
        "port_footprint",
        tuple(family_segment("cap", i, gap[i], gap[(i + 1) % 4]) for i in range(4)),
        (Fraction(0), Fraction(-10)),
    )
    model = build_model_from_catalogue(
        catalogue_of("two_levels", "A", two_sheets("other"), (family,)), "A"
    )
    sidecar = sidecar_of(model, tmp_path)
    assert sidecar["cut_family"]["cap"]["levels_um"] == [0.0, -10.0]
    drawn_at = {
        sidecar["vertex"][edge["vertices"][0]]["position_um"][2]
        for edge in sidecar["edge"].values()
        if any(
            sidecar["carrier"][c]["kind"] == "family_member" for c in edge["carriers"]
        )
    }
    assert drawn_at == {-10.0, 0.0}


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
