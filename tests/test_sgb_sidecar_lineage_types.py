"""What a face comes from, and what a refinement family draws, are written from typed facts, never from ids.

A face's sources are the Entities, band families and refinement requests of the carriers that bound it.  Ids of
different kinds may be equal strings (an Entity and a request may be called alike), so the sidecar classifies each
source by the carrier's lineage, not by looking its id up in a table.  A refinement family a catalogue declares
directly is recorded by its segments exactly as they are: disjoint or listed out of order, nothing is joined.
"""

from __future__ import annotations

from fractions import Fraction

import pytest
from sgb_extreme_cases import catalogue_of, chip, metal, rect_um

from scgsim.contracts.sgb_sidecar import read_sidecar, views
from scgsim.geometry.sgb.build import emit_model
from scgsim.geometry.sgb.cutfamilies.requests import (
    ConductorSurface,
    InsetRequest,
    RefinementRequest,
)
from scgsim.geometry.sgb.model import build_model_from_catalogue
from scgsim.geometry.sgb.records.carriers import Lineage, LineCarrier
from scgsim.geometry.sgb.records.families import CurveFamilyRecord
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
    record = sidecar_of(model, tmp_path)["refinement_request"]["drawn"]
    assert record["points_um"] is None
    assert record["segments_um"] == [
        [[p[0] * GRID_UM, p[1] * GRID_UM], [q[0] * GRID_UM, q[1] * GRID_UM]]
        for p, q in carriers
    ]
    assert record["z_um"] == 0.0


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
