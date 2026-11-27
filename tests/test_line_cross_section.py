"""The line cross-section of a 2D model: the classification of a plane section, and its advanced entry (issue 188).

Synthetic catalogues in grid units (``sgb_scenes``) are built in 3D on Route B and on Route C, cut across at
x = 0 and classified.  A coplanar waveguide (CPW) is checked exactly: its pieces with their roles, its gaps and
its dielectrics; both routes give one record and one digest.  Each refusal of the step-1 scope is reached once,
and the pieces it lists are read back out of it.  ``cross_section_at`` runs on a stand-in plan drawn with
gdsfactory, with no solver; the PDK's finite-ground coupon is in ``test_line_cross_section_pdk``.
"""

from __future__ import annotations

import pytest
from sgb_scenes import GRID, conductor, footprint, rect_um
from test_located_refusals import places
from test_sgb_section import by_type
from test_simulation_drc import BANDED, LAYER_STACK, NO_VACUUM_STACK, layout_plan

from scgsim.geometry.exact.errors import InputError
from scgsim.geometry.exact.grid import exact_z
from scgsim.geometry.models.section import (
    CutLine,
    LineConductor,
    LineCrossSection,
    LineDielectric,
    LineGap,
)
from scgsim.geometry.sections import build_plan_model, cross_section_at
from scgsim.geometry.sgb.model import build_model_from_catalogue
from scgsim.geometry.sgb.records.prisms import PrismCatalogue, PrismRecord
from scgsim.geometry.sgb.section import build_section_model, line_cross_section

CUT = CutLine((0.0, -91.0), (0.0, 91.0))
# Strips along x as (Entity, y0, y1, Net) in um: grounds 80 um, gaps 6 um, signal 10 um.
CPW = (
    ("gnd_minus", -91, -11, "ground"),
    ("signal", -5, 5, "signal"),
    ("gnd_plus", 11, 91, "ground"),
)
# The digest of the CPW's record: the same in every run while the record's fields do not change.
CPW_DIGEST = "2d17d6a307c93a36c25fc717a75b81e6f3ad61f4a0b7d8e246faa3ce043ea7b0"


def strips(lines=CPW, *, route="B", film="0.1", extra=()):
    """Conductor strips 100 um long along x, each ``(Entity, y0, y1, Net)``, on 500 um of substrate.

    The substrate and the vacuum (100 um) span y = -91 to 91 um, so the cut ``CUT`` runs from one edge of the
    model to the other.  ``film`` is the strips' thickness (0 for Route A sheets), ``extra`` more prisms.
    MA, MS and SA are banded on Routes B and C.
    """
    outline = footprint(rect_um(-50, -91, 50, 91))
    prisms = [
        PrismRecord(
            "substrate", outline, exact_z(-500), exact_z(0), "si", "dielectric"
        ),
        PrismRecord("vacuum", outline, exact_z(0), exact_z(100), "vacuum", "vacuum"),
        *(
            conductor(entity, rect_um(-50, y0, 50, y1), 0, film, net)
            for entity, y0, y1, net in lines
        ),
        *extra,
    ]
    insets = by_type("MA", "MS", "SA") if route != "A" else ()
    catalogue = PrismCatalogue(GRID, tuple(prisms), name="line")
    return build_model_from_catalogue(catalogue, route, insets=insets)


def classify(model, cut: CutLine = CUT) -> LineCrossSection:
    """The line cross-section of ``model`` along ``cut``."""
    return line_cross_section(build_section_model(model, cut))


def refusal(model, cut: CutLine = CUT) -> str:
    """The refusal of a section that is no line, as text."""
    with pytest.raises(InputError) as error:
        classify(model, cut)
    return str(error.value)


def piece(x: float, y0: float, y1: float) -> tuple[float, float, float, float]:
    """The place of a piece on the cut x = ``x``, as a box."""
    return (x, y0, x, y1)


# --------------------------------------------------------------------------- the CPW, exactly


def test_a_cpw_is_classified_exactly():
    """Ground [0, 80], gap, signal [86, 96], gap, ground [102, 182] um along the cut, in the 0.1 um film."""
    line = classify(strips())
    assert line.cut == CUT
    film = (0.0, 0.1)
    assert line.conductors == (
        LineConductor("gnd_minus", "ground", "reference", "Al", (0.0, 80.0), film),
        LineConductor("signal", "signal", "signal", "Al", (86.0, 96.0), film),
        LineConductor("gnd_plus", "ground", "reference", "Al", (102.0, 182.0), film),
    )
    assert line.gaps == (
        LineGap((80.0, 86.0), film, ("ground", "signal")),
        LineGap((96.0, 102.0), film, ("signal", "ground")),
    )
    assert line.dielectrics == (
        LineDielectric("substrate", "dielectric", "si", (0.0, 182.0), (-500.0, 0.0)),
        LineDielectric("vacuum", "vacuum", "vacuum", (0.0, 182.0), (0.0, 100.0)),
    )
    assert line.describe() == (
        "ground 80 | 6 | signal 10 | 6 | ground 80 um; Al 0.1 um on si 500 um; cut at x = 0 um"
    )


def test_route_b_and_route_c_give_one_record_and_one_digest():
    """Route B excludes the conductors and Route C keeps them as faces; the pieces are the same regions."""
    on_b, on_c = (build_section_model(strips(route=r), CUT) for r in ("B", "C"))
    assert {c.face for c in on_b.conductors} == {None}
    assert None not in {c.face for c in on_c.conductors}
    assert [(c.domains, c.u, c.z) for c in on_b.conductors] == [
        (c.domains, c.u, c.z) for c in on_c.conductors
    ]
    line_b, line_c = line_cross_section(on_b), line_cross_section(on_c)
    assert line_b == line_c
    assert line_b.digest == line_c.digest


def test_the_digest_is_stable_and_follows_the_line():
    """Two builds of one layout give one digest; a wider gap gives another, and an asymmetric CPW is a line."""
    first, again = classify(strips()), classify(strips())
    assert first.digest == again.digest == CPW_DIGEST
    wider = classify(strips((("gnd_minus", -91, -13, "ground"), *CPW[1:])))
    assert [gap.u_um for gap in wider.gaps] == [(78.0, 86.0), (96.0, 102.0)]
    assert wider.digest != first.digest


def test_a_cut_drawn_the_other_way_reads_the_line_from_its_other_end():
    """``u`` runs from the cut's start: the pieces come in the other order, with the same roles."""
    line = classify(strips(), CutLine((0.0, 91.0), (0.0, -91.0)))
    assert [(c.entity, c.role) for c in line.conductors] == [
        ("gnd_plus", "reference"),
        ("signal", "signal"),
        ("gnd_minus", "reference"),
    ]
    assert [gap.u_um for gap in line.gaps] == [(80.0, 86.0), (96.0, 102.0)]
    assert line.digest != classify(strips()).digest  # the cut is part of the record


# --------------------------------------------------------------------------- refusals, located


def test_an_extra_conductor_on_the_cut_is_refused_with_every_piece():
    """A floating island over the signal is a fourth piece: the refusal lists all four, in order along the cut."""
    island = conductor("island", rect_um(-50, -3, 50, 3), 5, "5.2", "floating")
    message = refusal(strips(extra=[island]))
    assert "crosses 4 conductor pieces" in message
    assert (
        "island (Net floating) at (0.000, -3.000)..(0.000, 3.000) um, z 5..5.2 um"
        in message
    )
    assert "That is 1 more than a line's three" in message
    assert "Fix the cut line" in message
    assert places(message)[1:] == [
        piece(0, -91, -11),
        piece(0, -5, 5),
        piece(0, -3, 3),
        piece(0, 11, 91),
    ]


def test_a_single_gap_is_refused():
    """A ground and a signal show one gap: a slot line, not a CPW."""
    message = refusal(strips((CPW[0], ("signal", -5, 91, "signal"))))
    assert "crosses 2 conductor pieces" in message
    assert "Two conductors show a single gap" in message
    assert places(message)[1:] == [piece(0, -91, -11), piece(0, -5, 91)]


def test_an_unknown_reference_is_refused():
    """The outer pieces are of two Nets, so the structure does not say which is the reference."""
    grounds = (
        ("gnd_minus", -91, -11, "ground_a"),
        CPW[1],
        ("gnd_plus", 11, 91, "ground_b"),
    )
    message = refusal(strips(grounds))
    assert "The outer pieces are of different Nets (ground_a, ground_b)" in message
    assert "does not say which Net is the reference" in message
    assert "Fix the plan's Nets with GeometryPlan.set_nets" in message


@pytest.mark.parametrize(
    ("lines", "extra", "reason"),
    [
        (
            (CPW[0], ("signal", -5, 5, "ground"), CPW[2]),
            (),
            "All three pieces are of Net ground",
        ),
        (
            (CPW[0], CPW[2]),
            (conductor("signal", rect_um(-50, -5, 50, 5), 0, "0.2", "signal"),),
            "The pieces do not lie side by side in one film",
        ),
        ((("plane", -91, 91, "ground"),), (), "One conductor shows no gap"),
        ((), (), "crosses no conductor piece. A line needs conductors on the cut"),
    ],
    ids=["no signal", "two films", "one plane", "nothing"],
)
def test_a_section_that_is_no_cpw_is_refused(lines, extra, reason):
    """No signal Net, a signal thicker than its grounds, one plane across the whole cut, no conductor."""
    message = refusal(strips(lines, extra=extra))
    assert reason in message
    assert "A line cross-section in this version is one signal piece" in message


@pytest.mark.parametrize("route", ["B", "C"])
def test_entities_of_one_net_that_touch_are_refused_on_both_routes(route):
    """Route B joins them into one excluded region and Route C gives two faces; both have no gap between them."""
    joined = (("gnd_minus", -91, -5, "signal"), *CPW[1:])
    message = refusal(strips(joined, route=route))
    assert "gnd_minus and signal touch, with no gap between them" in message


def test_a_route_a_sheet_on_the_cut_is_refused():
    """A zero-thickness sheet is no piece: Route A is a later step."""
    message = refusal(strips(route="A", film=0))
    assert (
        "crosses the conductor sheet gnd_minus at (0.000, -91.000) um on level z = 0 um"
        in message
    )
    assert "build the model on Route B or C" in message


# --------------------------------------------------------------------------- the entry, on a stand-in plan


def cpw_plan(monkeypatch, *, layer_stack=LAYER_STACK, surface_epr_specs=BANDED):
    """The CPW drawn with gdsfactory on the stand-in stack (PEC film 0.2 um on 5 um of Si), MA banded.

    The stack has its own vacuum unless ``layer_stack`` is :data:`NO_VACUUM_STACK`, and the plan sets none.
    """
    import gdsfactory as gf  # noqa: PLC0415 -- optional layout stack, imported only by the tests that need it

    # Restore process-local PDK state after this public generic fixture.
    monkeypatch.setattr(
        gf.pdk,
        "_ACTIVE_PDK",
        gf.pdk._ACTIVE_PDK,  # noqa: SLF001 -- preserve the active PDK during test setup
    )
    gf.gpdk.PDK.activate()
    component = gf.Component()
    regions = []
    for layer, (entity, y0, y1, _net) in enumerate(CPW, start=1):
        component.add_polygon(
            ((-50, y0), (50, y0), (50, y1), (-50, y1)), layer=(layer, 0)
        )
        regions.append(
            {"semantic_id": entity, "level": "metal", "gds_layer": [layer, 0]}
        )
    component.info["component_semantics"] = {
        "schema_version": 2,
        "conductor_regions": regions,
    }
    nets = {"signal": ("chip/signal",), "ground": ("chip/gnd_minus", "chip/gnd_plus")}
    plan = layout_plan(
        component,
        layer_stack=layer_stack,
        nets=nets,
        vacuum_padding_um=None,
        surface_epr_specs=surface_epr_specs,
    )
    return plan.prepare()


def test_cross_section_at_classifies_the_plan_on_both_routes(monkeypatch):
    """The plan's own 3D model, cut and classified: equal records on Route B and Route C."""
    snapshot = cpw_plan(monkeypatch)
    on_b = cross_section_at(snapshot, CUT)
    on_c = cross_section_at(snapshot, CUT, route="C")
    assert on_b == on_c and on_b.digest == on_c.digest
    film = (0.0, 0.2)
    assert on_b.conductors == (
        LineConductor(
            "chip/gnd_minus_0000", "ground", "reference", "PEC", (0.0, 80.0), film
        ),
        LineConductor(
            "chip/signal_0000", "signal", "signal", "PEC", (86.0, 96.0), film
        ),
        LineConductor(
            "chip/gnd_plus_0000", "ground", "reference", "PEC", (102.0, 182.0), film
        ),
    )
    assert [gap.u_um for gap in on_b.gaps] == [(80.0, 86.0), (96.0, 102.0)]
    assert on_b.describe().startswith(
        "ground 80 | 6 | signal 10 | 6 | ground 80 um; PEC 0.2 um on Si 5 um"
    )
    assert cross_section_at(snapshot, CUT).digest == on_b.digest


def test_the_plan_model_carries_the_plans_bands(monkeypatch):
    """The in-memory build bands what the specs ask, a conductor customised by name included.

    MA by type bands each conductor's top; a customised conductor's faces come from an unbanded first build,
    which only that path makes, in a temporary folder.
    """
    model = build_plan_model(cpw_plan(monkeypatch))
    assert sorted(r.request_id for r in model.inset_plan.requests) == [
        f"epr_MA::chip/{entity}_0000::top"
        for entity in ("gnd_minus", "gnd_plus", "signal")
    ]
    by_name = {"chip/signal_0000": [0.25]}
    custom = {"MA": {**BANDED["MA"], "inset_margins_by_entity_um": by_name}}
    model = build_plan_model(cpw_plan(monkeypatch, surface_epr_specs=custom))
    assert {r.request_id: r.margins_um for r in model.insets} == {
        "epr::chip/signal_0000::top": (0.25,),
        "epr_MA": (0.5, 1.0),
    }


@pytest.mark.parametrize(
    ("route", "error", "text"),
    [
        (
            "A",
            ValueError,
            "takes route 'B' or 'C': Route A models conductors as sheets",
        ),
        ("D", ValueError, "route must be 'B' or 'C', not 'D'"),
    ],
)
def test_cross_section_at_takes_route_b_or_c(monkeypatch, route, error, text):
    """Route A has no 2D model yet; any other route is refused, naming the call it was given to."""
    snapshot = cpw_plan(monkeypatch)
    with pytest.raises(error, match=text) as refused:
        cross_section_at(snapshot, CUT, route=route)
    assert str(refused.value).endswith("Change it in the call to cross_section_at.")
    with pytest.raises(error, match=text) as refused:
        build_plan_model(snapshot, route=route)
    assert str(refused.value).endswith("Change it in the call to build_plan_model.")


def test_cross_section_at_refuses_what_is_no_snapshot_or_cut(monkeypatch):
    """Both arguments are checked before anything is built."""
    with pytest.raises(TypeError, match="cut must be a CutLine"):
        cross_section_at(cpw_plan(monkeypatch), ((0, -91), (0, 91)))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="snapshot must be a GeometryPlanSnapshot"):
        cross_section_at(object(), CUT)  # type: ignore[arg-type]


def test_cross_section_at_refuses_a_plan_without_vacuum(monkeypatch):
    """A stack with no vacuum and a plan that sets none have no model to cut."""
    snapshot = cpw_plan(monkeypatch, layer_stack=NO_VACUUM_STACK)
    with pytest.raises(ValueError, match="has no vacuum region"):
        cross_section_at(snapshot, CUT)
