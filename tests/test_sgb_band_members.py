"""What SGB keeps in memory of each band curve and each support primitive, as typed fields (Issue #170, slice B).

A band curve's lineage records the support primitive it comes from -- its ring, its position on the ring, its
role (``edge``, ``corner``, ``capstart``, ``capend``) -- and its margin, as a ``BandMember``.  A support region
records, per primitive, the source curves the primitive lies on.  Each fact is checked here against the geometry
it describes, never by reading an id.

The sidecar does not carry these facts yet: slice C writes them.  Keeping them must not change the file, so the
sidecar bytes of a few banded models are pinned below.
"""

from __future__ import annotations

import hashlib
import math
from fractions import Fraction

import pytest
import sgb_emit_cases
import sgb_supported_cells as disk_qubit
import sgb_supported_pdk_cells as pdk
from sgb_supported_kit import Conductor, Scene, catalogue, to_lattice, type_bands

from scgsim.contracts.sgb_sidecar import read_sidecar
from scgsim.geometry.sgb.build import emit_model
from scgsim.geometry.sgb.exact.carriers import (
    ArcCarrier,
    LineCarrier,
    LineEdge,
    round_to_lattice,
)
from scgsim.geometry.sgb.exact.families import CurveFamilyRecord
from scgsim.geometry.sgb.exact.support import (
    DISTANCE_SLACK,
    SupportRegion,
    distance_to_primitive,
)
from scgsim.geometry.sgb.model import ConformalModel, build_model_from_catalogue
from scgsim.geometry.sgb.output.sidecar.lineage import CarrierLineage

# A source edge is noded by snap rounding: the arrangement edges along it end within half a grid diagonal of it.
SOURCE_SNAP_STEPS = math.sqrt(2) / 2


def strip_into_pad() -> Scene:
    """A strip from the die edge into a pad of its own Net: the contact is no origin, so the strip's bands cap there.

    The scene of ``test_sgb_die_edge_cut``: the only model here whose band curves have end caps.
    """
    strip = [(0.0, -25.0), (20.0, 15.0), (20.0, 25.0), (0.0, -15.0)]
    pad = [(20.0, -20.0), (30.0, -20.0), (30.0, 28.0), (20.0, 28.0)]
    return Scene(
        "strip_into_pad",
        (),
        (0.0, -30.0, 40.0, 30.0),
        [
            Conductor("strip", "N", to_lattice(strip)),
            Conductor("pad", "N", to_lattice(pad)),
        ],
    )


def scene_model(scene: Scene) -> ConformalModel:
    """A scene as the sample set builds it: Route B, MA, MS and SA bands asked for by type."""
    return build_model_from_catalogue(catalogue(scene), "B", insets=type_bands())


# Lines, arcs offset from source arcs, reflex corners, end caps, die cuts and a port, on both routes.
MODELS = {
    "pad_a_bands": sgb_emit_cases.CASES["pad_a_bands"],
    "pad_b_bands": sgb_emit_cases.CASES["pad_b_bands"],
    "junction_fan": lambda: scene_model(disk_qubit.SCENES["junction_fan"]()),
    "cell13_circular_bend": lambda: scene_model(pdk.cell13_circular_bend()),
    "strip_into_pad": lambda: scene_model(strip_into_pad()),
}


@pytest.fixture(scope="module", params=sorted(MODELS))
def model(request) -> ConformalModel:
    """Each banded model, built once."""
    return MODELS[request.param]()


def band_families(model: ConformalModel) -> list[CurveFamilyRecord]:
    """The inset band families of a model."""
    return [f for f in model.labelled.catalogue.families if f.support is not None]


# --------------------------------------------------------------------------- band curves


def test_every_band_curve_records_the_primitive_it_comes_from_and_its_margin(model):
    """Each line and arc of a band family names a ring position of its support, a role and one of its margins.

    The role agrees with the distance origins around that position, and the margin with the band its seam bounds.
    """
    roles = set()
    for family in band_families(model):
        support = family.support
        assert support is not None and family.regions is not None
        bands = {seam.carrier_id: seam.band for seam in family.regions.seams}
        for carrier in (*family.carriers, *family.arcs):
            member = carrier.lineage.band_member
            assert member is not None, carrier.id
            assert carrier.lineage.family_id == family.family_id
            assert support.margins[bands[carrier.id] - 1] == member.margin
            count = len(support.rings[member.ring])
            assert 0 <= member.index < count
            before, after = (member.index - 1) % count, (member.index + 1) % count
            assert support.is_origin(member.ring, member.index), carrier.id
            if member.role == "corner":
                assert support.is_origin(member.ring, after)
            if member.role == "capend":
                assert not support.is_origin(member.ring, after)
            if member.role == "capstart":
                assert not support.is_origin(member.ring, before)
            roles.add(member.role)
    assert roles >= {"edge", "corner"}


def test_every_band_curve_lies_at_its_margin_from_the_primitive_it_records(model):
    """The recorded place is the curve's geometry: an offset by the margin, or an arc of that radius about a vertex.

    A straight offset lies ``margin`` to the left of its primitive's line, within the snap of its end points; an
    offset of an arc is concentric with it; a corner or cap is an arc of radius ``margin`` about the vertex
    between two primitives, or about the end of the primitive.
    """
    for family in band_families(model):
        support = family.support
        assert support is not None
        for carrier in (*family.carriers, *family.arcs):
            member = carrier.lineage.band_member
            assert member is not None
            primitive = support.rings[member.ring][member.index]
            if member.role != "edge":
                assert isinstance(carrier, ArcCarrier)
                assert carrier.radius == member.margin
                at_end = member.role in ("corner", "capend")
                assert carrier.centre == _lattice_end(primitive, at_end=at_end)
            elif isinstance(carrier, LineCarrier):
                assert isinstance(primitive, LineEdge)
                for point in (carrier.p, carrier.q):
                    offset = _left_of(primitive, point)
                    assert abs(offset - member.margin) <= DISTANCE_SLACK, carrier.id
            else:
                assert isinstance(primitive, ArcCarrier)
                inward = -1 if primitive.ccw else 1
                assert carrier.centre == primitive.centre
                assert carrier.radius == primitive.radius + inward * member.margin


def test_end_caps_record_their_role_at_the_strips_corner_on_the_pad():
    """Where the strip meets the pad, its MA and MS bands close round the corner at (20, 15) um with caps."""
    model = scene_model(strip_into_pad())
    for family_id in ("ma::strip::top", "ms::strip::bottom"):
        (family,) = [f for f in band_families(model) if f.family_id == family_id]
        caps = [
            arc
            for arc in family.arcs
            if arc.lineage.band_member is not None
            and arc.lineage.band_member.role in ("capstart", "capend")
        ]
        assert {(float(a.centre[0]), float(a.centre[1])) for a in caps} == {
            (20_000.0, 15_000.0)
        }


def _left_of(line: LineEdge, point: tuple[int, int]) -> float:
    """Signed distance of a point from the line through an edge, positive on its left (the support's interior)."""
    (px, py), (qx, qy) = line.p, line.q
    dx, dy = qx - px, qy - py
    return (dx * (point[1] - py) - dy * (point[0] - px)) / math.hypot(dx, dy)


def _lattice_end(
    primitive: LineEdge | ArcCarrier, *, at_end: bool
) -> tuple[Fraction, Fraction]:
    """The lattice point of a primitive's end or start, where a corner or cap arc is centred."""
    if isinstance(primitive, LineEdge):
        x, y = primitive.q if at_end else primitive.p
        return Fraction(x), Fraction(y)
    x, y = round_to_lattice(primitive.end if at_end else primitive.start)
    return Fraction(x), Fraction(y)


# --------------------------------------------------------------------------- support primitives


def test_every_support_primitive_records_the_source_curves_it_lies_on(model):
    """Each primitive of a support region names one or more source edges or arcs, and lies on each of them.

    A straight primitive lies along the source edge, within the snap of its end points; an arc primitive lies on
    the circle of its source arc.  Several sources are the outlines of several Entities drawn along one edge.
    """
    lines = {
        c.id: c for prism in model.labelled.catalogue.prisms for c in prism.carriers()
    }
    arcs = {a.id: a for prism in model.labelled.catalogue.prisms for a in prism.arcs}
    for family in band_families(model):
        support = family.support
        assert support is not None
        assert [len(ring) for ring in support.sources] == [
            len(ring) for ring in support.rings
        ]
        for ring, sources in zip(support.rings, support.sources, strict=True):
            for primitive, named in zip(ring, sources, strict=True):
                assert named, f"{family.family_id}: a primitive without a source"
                for source in named:
                    _assert_lies_on(primitive, source, lines, arcs)


def _assert_lies_on(
    primitive: LineEdge | ArcCarrier,
    source: str,
    lines: dict[str, LineCarrier],
    arcs: dict[str, ArcCarrier],
) -> None:
    if isinstance(primitive, ArcCarrier):
        # On a level between two slabs the arc is the realised circle of the source arc, equal to it up to rounding.
        arc = arcs[source]
        assert math.dist(arc.centre, primitive.centre) < 1e-6, source
        assert float(arc.radius) == pytest.approx(float(primitive.radius)), source
        return
    edge = LineEdge(lines[source].p, lines[source].q)
    for x, y in (primitive.p, primitive.q):
        assert distance_to_primitive(edge, x, y) <= SOURCE_SNAP_STEPS, source


def test_a_short_band_edge_leads_to_its_source_edge_through_recorded_fields():
    """The example of Issue #138 (c): from an emitted edge to the source edge its band offsets, no id read.

    On ``junction_fan`` the edge ``e:9509e27bdf7e2d28`` is 15.6 grid steps long.  Its one carrier is a band curve
    of the SA face offset by 1 um from primitive 13 of ring 0; that primitive is a 0.1954 um line, which lies on
    the ground's source edge ``ground/polygon0/ring1/edge29``: a source edge shorter than the margin.
    """
    model = scene_model(disk_qubit.SCENES["junction_fan"]())
    (carrier_id,) = model.topology.lineage["e:9509e27bdf7e2d28"]
    ((family, carrier),) = [
        (f, c)
        for f in band_families(model)
        for c in (*f.carriers, *f.arcs)
        if c.id == carrier_id
    ]
    member = carrier.lineage.band_member
    assert family.family_id == "sa::substrate|vacuum::z=0.0"
    assert member is not None
    assert (member.ring, member.index, member.role, member.margin) == (
        0,
        13,
        "edge",
        1000,
    )
    support = family.support
    assert isinstance(support, SupportRegion)
    primitive = support.rings[member.ring][member.index]
    assert primitive == LineEdge((164402, -171473), (164278, -171322), 13)
    assert math.dist(primitive.p, primitive.q) == pytest.approx(195.4, abs=0.05)
    assert support.sources[member.ring][member.index] == (
        "ground/polygon0/ring1/edge29",
    )


# --------------------------------------------------------------------------- the sidecar writes them as fields


def test_the_sidecar_leads_a_short_band_edge_to_the_source_edge_it_offsets(tmp_path):
    """The #138 (c) chain read from ``scgsim.sgb.sidecar.v3`` alone, by following typed ids and reading fields.

    Edge ``e:9509e27bdf7e2d28`` of ``junction_fan`` -> its carrier, a band member of role ``edge`` at margin 1000
    grid steps -> the support primitive it offsets, place 13 on its ring, a 0.1954 um line -> the source curve it
    lies on, the ground's 0.1954 um edge, shorter than the margin.
    """
    sidecar = read_sidecar(emit_model(MODELS["junction_fan"](), tmp_path).sidecar)
    (carrier_id,) = sidecar["edge"]["e:9509e27bdf7e2d28"]["carriers"]
    member = sidecar["carrier"][carrier_id]
    assert (member["kind"], member["role"], member["margin_grid"], member["piece"]) == (
        "band_member",
        "edge",
        1000,
        None,
    )
    primitive = sidecar["support_primitive"][member["primitive"]]
    ring = sidecar["support_ring"][primitive["ring"]]
    assert ring["primitives"].index(member["primitive"]) == 13 and ring["index"] == 0
    assert primitive["curve"]["kind"] == "line"
    assert primitive["length_um"] == pytest.approx(0.1954, abs=5e-5)
    (source,) = primitive["sources"]
    curve = sidecar["source_curve"][source]
    assert (curve["entity"], curve["polygon"], curve["ring"], curve["edge"]) == (
        "ground",
        0,
        1,
        29,
    )
    assert curve["length_um"] == pytest.approx(primitive["length_um"], abs=1e-9)
    margin_um = member["margin_grid"] * sidecar["model"]["grid_um"]
    assert curve["length_um"] < margin_um
    assert sidecar["band_family"][ring["family"]]["margins_grid"][-1] == 1000


@pytest.mark.parametrize("name", sorted(MODELS))
def test_every_band_member_of_the_sidecar_names_its_place_as_the_model_does(
    name, tmp_path
):
    """Each band-member carrier names the primitive, role, margin and piece of its in-memory ``BandMember``."""
    model = MODELS[name]()
    sidecar = read_sidecar(emit_model(model, tmp_path).sidecar)
    lineage = CarrierLineage.of(model)
    members = 0
    for carrier_id, record in sidecar["carrier"].items():
        member = lineage.lineage(carrier_id).band_member
        if record["kind"] != "band_member":
            assert member is None, carrier_id
            continue
        members += 1
        primitive = sidecar["support_primitive"][record["primitive"]]
        ring = sidecar["support_ring"][primitive["ring"]]
        assert (ring["index"], ring["primitives"].index(record["primitive"])) == (
            member.ring,
            member.index,
        )
        assert (record["role"], record["margin_grid"], record["piece"]) == (
            member.role,
            member.margin,
            member.piece,
        )
    assert members


# --------------------------------------------------------------------------- the file is pinned

# SHA-256 of each model's ``scgsim.sgb.sidecar.v3`` file.  Slice C re-pinned these (v2 became v3); a change meant
# to change the file re-pins the digests it changes and says why.
SIDECAR_SHA256 = {
    "pad_a_bands": "7fe9bcc29bc8ee09f6ea146a3cd37deb2e74b8b55435aef11d407ed7580d816f",
    "pad_b_bands": "131c0e45eb0c27cc51573babe0b01ecc8ac8c6b6e3d7133ce27724496dacf177",
    "junction_fan": "947a9858c07e66a0da7331ab8e24031f3b1ddee381be0b24c811b030691bdac5",
    "cell13_circular_bend": "9b30c52d9165cbb9de8e4a351dcf26254325ba450ee4b9a8b4badc4581e5928c",
    "strip_into_pad": "f8575530493caf605ce8956754e96b056a274ab6b748c51a5a32f1409ca70a8f",
}


@pytest.mark.parametrize("name", sorted(SIDECAR_SHA256))
def test_the_sidecar_of_each_banded_model_is_the_pinned_file(name, tmp_path):
    """The sidecar of each banded model is the pinned file, byte for byte."""
    artifacts = emit_model(MODELS[name](), tmp_path)
    digest = hashlib.sha256(artifacts.sidecar.read_bytes()).hexdigest()
    assert digest == SIDECAR_SHA256[name]
