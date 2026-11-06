"""Surface-EPR bands asked for by type: ``InterfaceSurfaces`` expanded by SGB into one family per face.

A type-level request names no face.  SGB finds the faces that have a Surface-EPR row of the type, bands each one
that has a gap edge and records the others as skipped.  The concrete requests keep working exactly as before.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest
from sgb_extreme_cases import FIXTURES, chip
from sgb_scenes import GRID, conductor, pad_in_ground, rect_um, slab_model
from test_sgb_sidecar import sidecar_of

from scgsim.geometry.sgb.build import emit_model
from scgsim.geometry.sgb.cutfamilies.requests import (
    ConductorSurface,
    InsetRequest,
    InterfaceSurfaces,
    SolutionInterface,
)
from scgsim.geometry.sgb.errors import InputError, NoGapEdgeError
from scgsim.geometry.sgb.model import build_model_from_catalogue
from scgsim.geometry.sgb.records.grid import exact_z
from scgsim.geometry.sgb.records.prisms import PrismCatalogue, PrismRecord
from scgsim.geometry.sgb.sidecar.files import sidecar_schema

TESTS = Path(__file__).resolve().parent
MARGINS = (0.5, 1.0)
TWO_PADS = [("a", "N", (-12, -10, -2, 10)), ("b", "M", (2, -10, 12, 10))]
GAP = "substrate|vacuum::z=0.0"


def by_type(*types: str, margins=MARGINS, **filters) -> tuple[InsetRequest, ...]:
    """One type-level request per Surface-EPR type, id = the lower-case type."""
    return tuple(
        InsetRequest(t.lower(), InterfaceSurfaces(t, **filters), margins) for t in types
    )


def build(catalogue, route, *requests, **options):
    """The model of a catalogue under a route with the given inset requests."""
    return build_model_from_catalogue(catalogue, route, insets=requests, **options)


def expansion_of(model, request_id):
    """The expansion of one type-level request: banded family ids and skipped (entity or pair, side) pairs."""
    (found,) = (
        e for e in model.inset_plan.expansions if e.request.request_id == request_id
    )
    return list(found.banded), [s.support for s in found.skipped]


def family_margins(model):
    """Family id -> its margins, for every band family of a model."""
    return {r.request_id: r.margins_um for r in model.inset_plan.requests}


def band_groups_of(model):
    """The sorted names of the band groups of a model."""
    return sorted(
        name for name in model.routed.groups if "::band=" in name or "band=" in name
    )


# --------------------------------------------------------------------------- expansion per route


def test_route_a_bands_every_sheet_for_both_ma_and_ms_with_one_family():
    """A sheet with vacuum above and a dielectric below carries MA and MS rows: one family serves both."""
    model = build(slab_model("A", TWO_PADS, half=20), "A", *by_type("MA", "MS", "SA"))
    assert expansion_of(model, "ma") == (["ma::a::sheet", "ma::b::sheet"], [])
    assert expansion_of(model, "ms") == (["ma::a::sheet", "ma::b::sheet"], [])
    assert expansion_of(model, "sa") == ([f"sa::{GAP}"], [])
    assert set(family_margins(model)) == {"ma::a::sheet", "ma::b::sheet", f"sa::{GAP}"}
    assert {"a::band=0", "a::band=1", "a::band=2", "b::band=2"} <= set(
        model.routed.groups
    )
    assert {"MA::a__sheet::band=1", "MS::a__sheet::band=1"} <= set(
        model.routed.surface_epr_rows
    )


@pytest.mark.parametrize("route", ["B", "C"])
def test_finite_conductors_band_the_top_for_ma_and_the_bottom_for_ms(route):
    """A finite pad on a substrate: its top faces vacuum (MA), its bottom the substrate (MS); sidewalls never."""
    model = build(slab_model("B", TWO_PADS, half=20), route, *by_type("MA", "MS", "SA"))
    assert expansion_of(model, "ma") == (["ma::a::top", "ma::b::top"], [])
    assert expansion_of(model, "ms") == (["ms::a::bottom", "ms::b::bottom"], [])
    assert expansion_of(model, "sa") == ([f"sa::{GAP}"], [])
    assert not [g for g in band_groups_of(model) if "sidewall" in g]
    assert {"MA::a__top::band=2", "MS::a__bottom::band=2"} <= set(model.routed.groups)


def test_a_floating_sheet_in_vacuum_has_an_ma_row_but_no_gap_edge_and_no_ms_row():
    """Vacuum on both sides gives MA only, and the exposed vacuum round it is no gap: skipped, recorded."""
    catalogue = FIXTURES["floating_conductor"].build("A")
    model = build(catalogue, "A", *by_type("MA", "MS"))
    assert expansion_of(model, "ma") == ([], [ConductorSurface("float", "sheet")])
    assert expansion_of(model, "ms") == ([], [])
    assert model.inset_plan.supports == ()


def test_a_floating_finite_conductor_skips_its_top_and_bottom():
    """Route B: both faces of a conductor enclosed by vacuum are MA and neither has a gap edge."""
    catalogue = FIXTURES["floating_conductor"].build("B")
    model = build(catalogue, "B", *by_type("MA", "MS"))
    assert expansion_of(model, "ma") == (
        [],
        [ConductorSurface("float", "bottom"), ConductorSurface("float", "top")],
    )
    assert expansion_of(model, "ms") == ([], [])


def vacuum_below_substrate() -> PrismCatalogue:
    """A substrate with a vacuum above (a pad on it) and another vacuum under it."""
    box = chip(20.0)[0].footprint
    under = PrismRecord(
        "vacuum_below", box, exact_z(-20), exact_z(-10), "vacuum", "vacuum"
    )
    pad = conductor("pad", rect_um(-5, -5, 5, 5), 0, 0, "N")
    return PrismCatalogue(GRID, (*chip(20.0), under, pad), name="underside")


def test_sa_is_banded_where_a_conductor_is_attached_and_skipped_on_the_underside():
    """The substrate-vacuum interface under the pad's plane has a gap edge; the one under the substrate has none."""
    model = build(vacuum_below_substrate(), "A", *by_type("SA"))
    banded, skipped = expansion_of(model, "sa")
    assert banded == [f"sa::{GAP}"]
    assert skipped == [SolutionInterface("substrate", "vacuum_below", -10.0)]


# --------------------------------------------------------------------------- filters


def test_face_kinds_keep_only_the_named_kinds():
    """``face_kinds`` selects sheet, top, bottom or interface faces; a kind the type has no face of selects none."""
    catalogue = slab_model("B", TWO_PADS, half=20)
    only_top = build(catalogue, "B", *by_type("MA", "MS", face_kinds=("top",)))
    assert expansion_of(only_top, "ma")[0] == ["ma::a::top", "ma::b::top"]
    assert expansion_of(only_top, "ms")[0] == []
    bottoms = build(
        catalogue, "B", *by_type("MA", "MS", face_kinds=("bottom", "interface"))
    )
    assert expansion_of(bottoms, "ma")[0] == []
    assert expansion_of(bottoms, "ms")[0] == ["ms::a::bottom", "ms::b::bottom"]
    gap_only = build(catalogue, "B", *by_type("MA", "SA", face_kinds=("interface",)))
    assert expansion_of(gap_only, "ma")[0] == []
    assert expansion_of(gap_only, "sa")[0] == [f"sa::{GAP}"]
    no_gap = build(catalogue, "B", *by_type("SA", face_kinds=("top", "sheet")))
    assert expansion_of(no_gap, "sa")[0] == []


def test_excluded_entities_are_left_out_and_their_own_requests_band_them():
    """A conductor face is left out by its entity, an interface by either side; a concrete request bands the excluded one."""
    catalogue = slab_model("B", TWO_PADS, half=20)
    own = InsetRequest("a_top", ConductorSurface("a", "top"), (0.25,))
    model = build(catalogue, "B", own, *by_type("MA", exclude_entity_ids=("a",)))
    assert expansion_of(model, "ma")[0] == ["ma::b::top"]
    assert family_margins(model) == {"a_top": (0.25,), "ma::b::top": MARGINS}
    assert "MA::a__top::band=1" in model.routed.groups
    assert "MA::a__top::band=2" not in model.routed.groups
    record = {r["request_id"]: r for r in sidecar_of(model)["insets"]}
    assert record["ma"]["support"]["exclude_entity_ids"] == ["a"]
    assert record["a_top"]["margins_um"] == [0.25]
    for side in ("substrate", "vacuum"):
        solution = build(catalogue, "B", *by_type("SA", exclude_entity_ids=(side,)))
        assert expansion_of(solution, "sa") == ([], [])


@pytest.mark.parametrize(
    ("support", "message"),
    [
        (InterfaceSurfaces("MM"), "not 'MM'"),
        (InterfaceSurfaces("SS"), "not 'SS'"),
        (InterfaceSurfaces("MA", face_kinds=("sidewall",)), "sidewalls are not banded"),
        (InterfaceSurfaces("MA", face_kinds=()), "must name some of"),
        (InterfaceSurfaces("MA", exclude_entity_ids=("nope",)), "names no entity"),
    ],
)
def test_a_type_request_that_cannot_be_read_fails_closed(support, message):
    """Unsupported types, sidewalls, empty kind lists and unknown excluded entities are input errors."""
    with pytest.raises(InputError, match=message):
        build(
            slab_model("B", TWO_PADS, half=20), "B", InsetRequest("t", support, MARGINS)
        )


def test_a_type_request_takes_no_expected_interface_and_validates_its_margins():
    """The type is in the support; margins follow the concrete rules."""
    catalogue = slab_model("A", TWO_PADS, half=20)
    with pytest.raises(InputError, match="leave interface unset"):
        build(catalogue, "A", InsetRequest("t", InterfaceSurfaces("MA"), MARGINS, "MA"))
    with pytest.raises(InputError, match="strictly increasing"):
        build(catalogue, "A", InsetRequest("t", InterfaceSurfaces("MA"), (1.0, 0.5)))
    with pytest.raises(InputError, match="needs at least one margin"):
        build(catalogue, "A", InsetRequest("t", InterfaceSurfaces("MA"), ()))


# --------------------------------------------------------------------------- one family per face


def test_requests_on_one_face_asking_the_same_margins_make_one_family():
    """MA and MS requests on a Route A sheet; two requests of one type; one family each, with their margins."""
    catalogue = slab_model("A", TWO_PADS, half=20)
    split = build(
        catalogue,
        "A",
        InsetRequest("ma", InterfaceSurfaces("MA"), (0.5, 2.0)),
        InsetRequest("ms", InterfaceSurfaces("MS"), (0.5, 2.0)),
    )
    assert family_margins(split) == {
        "ma::a::sheet": (0.5, 2.0),
        "ma::b::sheet": (0.5, 2.0),
    }
    assert expansion_of(split, "ms")[0] == ["ma::a::sheet", "ma::b::sheet"]
    assert "a::band=2" in split.routed.groups and "a::band=3" not in split.routed.groups
    again = build(
        catalogue,
        "A",
        InsetRequest("z", InterfaceSurfaces("MA", face_kinds=("sheet",)), (0.5, 1.0)),
        InsetRequest("y", InterfaceSurfaces("MA"), (0.5, 1.0)),
    )
    assert family_margins(again)["y::a::sheet"] == (
        0.5,
        1.0,
    )  # the id of the first request in id order


def test_margins_equal_on_the_grid_are_the_same_margins_however_they_are_written():
    """0.3 and 0.1 + 0.2 are one multiple of the model grid: one family, not two values."""
    catalogue = slab_model("A", TWO_PADS, half=20)
    model = build(
        catalogue,
        "A",
        InsetRequest("ma", InterfaceSurfaces("MA"), (0.3,)),
        InsetRequest("ms", InterfaceSurfaces("MS"), (0.1 + 0.2,)),
    )
    assert family_margins(model) == {"ma::a::sheet": (0.3,), "ma::b::sheet": (0.3,)}
    record = sidecar_of(model)["insets"]
    assert all(i["margins_grid"] == [300] for i in record[0]["expansion"]["banded"])


def test_requests_on_one_face_asking_different_margins_are_refused():
    """One face has one band family and so one set of margins: MA and MS may not differ on a Route A sheet."""
    catalogue = slab_model("A", TWO_PADS, half=20)
    with pytest.raises(InputError, match="ask different margins") as error:
        build(
            catalogue,
            "A",
            InsetRequest("ma", InterfaceSurfaces("MA"), (0.5, 2.0)),
            InsetRequest("ms", InterfaceSurfaces("MS"), (1.0, 2.0)),
        )
    message = str(error.value)
    assert "'ma'" in message and "'ms'" in message
    assert "[0.5, 2.0]" in message and "[1.0, 2.0]" in message
    assert "same margins" in message


def test_a_concrete_request_keeps_its_id_and_must_ask_the_margins_of_a_type_request_on_its_face():
    """The concrete family is the one family of the face and keeps its id; the type request must agree."""
    catalogue = slab_model("A", TWO_PADS, half=20)
    named = InsetRequest("a_sheet", ConductorSurface("a", "sheet"), (0.5, 1.0))
    model = build(catalogue, "A", named, *by_type("MA", margins=(0.5, 1.0)))
    assert family_margins(model) == {
        "a_sheet": (0.5, 1.0),
        "ma::b::sheet": (0.5, 1.0),
    }
    assert expansion_of(model, "ma")[0] == ["a_sheet", "ma::b::sheet"]
    record = {r["request_id"]: r for r in sidecar_of(model)["insets"]}
    assert record["a_sheet"]["margins_um"] == [0.5, 1.0]
    differing = InsetRequest("a_sheet", ConductorSurface("a", "sheet"), (0.25, 0.5))
    with pytest.raises(InputError, match="'a_sheet' and 'ma' ask different margins"):
        build(catalogue, "A", differing, *by_type("MA", margins=(0.5, 1.0)))


def test_two_concrete_requests_on_one_face_are_refused_with_or_without_a_type_request():
    """Both families would share the band groups of the face: refused at planning (see test_sgb_inset_band_groups)."""
    catalogue = slab_model("A", TWO_PADS, half=20)
    pair = (
        InsetRequest("first", ConductorSurface("a", "sheet"), (0.5,)),
        InsetRequest("second", ConductorSurface("a", "sheet"), (1.0,)),
    )
    for requests in (pair, (*pair, *by_type("MA"))):
        with pytest.raises(InputError, match="'first' and 'second' name the same face"):
            build(catalogue, "A", *requests)


def test_generated_ids_must_not_repeat_a_request_id():
    """A request named like an expanded family is refused, not silently merged."""
    catalogue = slab_model("A", TWO_PADS, half=20)
    clash = InsetRequest("ma::a::sheet", ConductorSurface("b", "sheet"), MARGINS)
    with pytest.raises(InputError, match="repeated: ma::a::sheet"):
        build(catalogue, "A", clash, *by_type("MA", face_kinds=("sheet",)))


# --------------------------------------------------------------------------- no gap edge


def covered_chip(route: str) -> PrismCatalogue:
    """A conductor over the whole die: its edges lie on the die edge, so no exposed dielectric is next to it."""
    return slab_model(route, [("full", "N", (-40, -40, 40, 40))])


def test_a_concrete_request_still_fails_closed_without_a_gap_edge_and_a_type_request_skips():
    """The same face: the concrete request raises, the type-level request records it as skipped."""
    request = InsetRequest("full_top", ConductorSurface("full", "top"), MARGINS)
    with pytest.raises(NoGapEdgeError, match="no gap edge") as error:
        build(covered_chip("B"), "B", request)
    assert isinstance(error.value, InputError)
    model = build(covered_chip("B"), "B", *by_type("MA", "MS"))
    assert expansion_of(model, "ma") == ([], [ConductorSurface("full", "top")])
    assert expansion_of(model, "ms") == ([], [ConductorSurface("full", "bottom")])
    assert model.inset_plan.supports == ()


def test_a_concrete_request_on_a_face_without_gap_edge_still_fails_when_a_type_request_shares_it():
    """Merging with a type-level request never hides the concrete request's failure."""
    request = InsetRequest("full_top", ConductorSurface("full", "top"), MARGINS)
    with pytest.raises(NoGapEdgeError):
        build(covered_chip("B"), "B", request, *by_type("MA"))


# --------------------------------------------------------------------------- equivalence with concrete requests


def concrete_equivalent(
    route: str, margins, *, expanded_ids: bool
) -> tuple[InsetRequest, ...]:
    """The concrete requests a caller would write for MA, MS and SA bands of the pad-in-ground scene.

    With ``expanded_ids`` they carry the ids SGB gives the expanded faces, which makes the two models identical.
    """

    def named(kind: str, entity: str, side: str) -> str:
        return f"{kind}::{entity}::{side}" if expanded_ids else f"{entity}_{side}"

    if route == "A":
        faces = [("ma", e, "sheet") for e in ("ground", "pad")]
    else:
        faces = [
            (k, e, s)
            for k, s in (("ma", "top"), ("ms", "bottom"))
            for e in ("ground", "pad")
        ]
    gap_id = f"sa::{GAP}" if expanded_ids else "gap"
    return (
        *(
            InsetRequest(named(k, e, s), ConductorSurface(e, s), margins)
            for k, e, s in faces
        ),
        InsetRequest(gap_id, SolutionInterface("substrate", "vacuum", 0.0), margins),
    )


def face_records(side):
    """The band families of a sidecar by support, dropping ids: the facts a solver reads."""
    families = {r["request_id"]: r for r in side["insets"] if "expansion" not in r}
    for r in side["insets"]:
        families.update(
            {f["request_id"]: f for f in r.get("expansion", {}).get("banded", [])}
        )
    return {
        json.dumps(f["support"], sort_keys=True): {
            k: v for k, v in f.items() if k not in ("request_id", "support")
        }
        for f in families.values()
    }, {fid: json.dumps(f["support"], sort_keys=True) for fid, f in families.items()}


@pytest.mark.parametrize(("route", "scene"), [("A", "A"), ("B", "B"), ("C", "B")])
def test_a_type_request_gives_the_model_of_the_equivalent_concrete_requests(
    route, scene
):
    """With the ids SGB gives the faces the two models are identical, down to the sidecar apart from its insets."""
    margins = (0.1, 0.5, 2.0)
    typed = build(
        pad_in_ground(scene), route, *by_type("MA", "MS", "SA", margins=margins)
    )
    written = build(
        pad_in_ground(scene),
        route,
        *concrete_equivalent(route, margins, expanded_ids=True),
    )
    assert typed.identity_digest() == written.identity_digest()
    side_typed, side_written = sidecar_of(typed), sidecar_of(written)
    for key in side_typed:
        if key != "insets":
            assert side_typed[key] == side_written[key], key
    records_typed, _ = face_records(side_typed)
    records_written, _ = face_records(side_written)
    assert records_typed == records_written
    assert len(records_typed) == len(typed.inset_plan.supports) > 0
    assert [r["request_id"] for r in side_typed["insets"]] == ["ma", "ms", "sa"]


@pytest.mark.parametrize(("route", "scene"), [("A", "A"), ("B", "B"), ("C", "B")])
def test_the_band_groups_do_not_depend_on_the_ids_of_the_requests(route, scene):
    """Whatever the caller calls the concrete requests, the groups, rows and bands are those of the type request."""
    margins = (0.1, 0.5, 2.0)
    typed = build(
        pad_in_ground(scene), route, *by_type("MA", "MS", "SA", margins=margins)
    )
    written = build(
        pad_in_ground(scene),
        route,
        *concrete_equivalent(route, margins, expanded_ids=False),
    )
    assert set(typed.routed.groups) == set(written.routed.groups)
    assert typed.routed.surface_epr_rows == written.routed.surface_epr_rows
    side_typed, side_written = sidecar_of(typed), sidecar_of(written)
    assert side_typed["unions"] == side_written["unions"]
    records_typed, ids_typed = face_records(side_typed)
    records_written, ids_written = face_records(side_written)
    assert records_typed == records_written
    assert _features_by_support(side_typed, ids_typed) == _features_by_support(
        side_written, ids_written
    )


def _features_by_support(side, support_of):
    """The band boundaries of a sidecar with the family replaced by its support (facts only; the edge ids follow the ids)."""
    return sorted(
        (support_of[f["source"]], len(f["edges"]), f["facts"])
        for f in side["mesh_features"]
        if f["kind"] == "band_boundary"
    )


# --------------------------------------------------------------------------- sidecar


def test_the_sidecar_records_the_request_as_given_its_banded_faces_and_its_skipped_faces():
    """The type-level record carries the request, the faces banded with their ids and why the others were left out."""
    catalogue = vacuum_below_substrate()
    side = sidecar_of(build(catalogue, "A", *by_type("SA", "MA", face_kinds=None)))
    assert not list(jsonschema.Draft202012Validator(sidecar_schema()).iter_errors(side))
    sa = next(r for r in side["insets"] if r["request_id"] == "sa")
    assert sa["support"] == {
        "interface_type": "SA",
        "face_kinds": None,
        "exclude_entity_ids": [],
    }
    assert sa["interface"] is None and sa["margins_um"] == [0.5, 1.0]
    assert [f["request_id"] for f in sa["expansion"]["banded"]] == [f"sa::{GAP}"]
    assert sa["expansion"]["banded"][0]["bands"]["0"] == [
        "SA::substrate|vacuum__interface::band=0"
    ]
    assert sa["expansion"]["skipped"] == [
        {
            "support": {
                "first_entity": "substrate",
                "second_entity": "vacuum_below",
                "z_um": -10.0,
            },
            "reason": "no_gap_edge",
        }
    ]
    sources = {
        f["source"] for f in side["mesh_features"] if f["kind"] == "band_boundary"
    }
    assert f"sa::{GAP}" in sources and "ma::pad::sheet" in sources
    assert {
        g["band"]["family"]
        for g in side["physical_groups"].values()
        if g["dim"] == 2 and g["band"]
    } <= sources


def test_a_build_without_type_requests_writes_the_records_it_wrote_before():
    """Concrete requests are recorded as today, and every record validates against the schema."""
    model = build(
        pad_in_ground("B"), "B", *concrete_equivalent("B", MARGINS, expanded_ids=False)
    )
    side = sidecar_of(model)
    assert not list(jsonschema.Draft202012Validator(sidecar_schema()).iter_errors(side))
    assert all("expansion" not in r for r in side["insets"])
    assert set(side["insets"][0]) == {
        "request_id", "support", "interface", "margins_um", "margins_grid", "band_count",
        "support_area_um2", "bands",
    }  # fmt: skip


def test_the_schema_refuses_a_type_record_without_its_expansion():
    """Negative control: a record that is neither a face record nor a complete type record."""
    side = sidecar_of(build(slab_model("A", TWO_PADS, half=20), "A", *by_type("MA")))
    validator = jsonschema.Draft202012Validator(sidecar_schema())
    assert not list(validator.iter_errors(side))
    broken = {
        **side,
        "insets": [{k: v for k, v in side["insets"][0].items() if k != "expansion"}],
    }
    assert list(validator.iter_errors(broken))


# --------------------------------------------------------------------------- determinism


def written_bytes(model, folder: Path) -> dict[str, bytes]:
    """Every file of a model written to a folder, by name."""
    artifacts = emit_model(model, folder)
    return {
        path.name: path.read_bytes()
        for path in (
            artifacts.xao,
            artifacts.brep,
            artifacts.sidecar,
            artifacts.sliver_report,
            artifacts.validation_report,
        )
    }


@pytest.mark.slow
def test_a_type_request_gives_byte_identical_files_for_every_worker_and_tile_count(
    tmp_path,
):
    """Ids, lineage, band groups and records do not depend on how the model was derived."""
    requests = by_type("MA", "MS", "SA", margins=(0.1, 0.5, 2.0))
    reference = written_bytes(
        build(pad_in_ground("B"), "B", *requests), tmp_path / "one"
    )
    many = written_bytes(
        build(pad_in_ground("B"), "B", *requests, tiles=3, workers=2), tmp_path / "many"
    )
    assert reference == many
    assert b"ma::pad::top" in reference["semantic_geometry_route_b.sidecar.json"]


@pytest.mark.slow
def test_a_type_request_gives_one_sidecar_for_every_hash_seed():
    """Set and dict ordering never reaches a family id or a record."""
    script = (
        f"import sys; sys.path.insert(0, {str(TESTS)!r})\n"
        "from test_sgb_inset_types import build, by_type, sidecar_of, slab_model, TWO_PADS\n"
        "import json\n"
        "m = build(slab_model('B', TWO_PADS, half=20), 'B', *by_type('MA', 'MS', 'SA'))\n"
        "print(json.dumps(sidecar_of(m)['insets'], sort_keys=True))\n"
    )
    outputs = {
        subprocess.run(
            [sys.executable, "-c", script],
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for seed in ("0", "7")
    }
    assert len(outputs) == 1


@pytest.mark.occt
def test_a_type_request_model_passes_the_official_occt_checks(tmp_path):
    """Strict emission of a model banded by type: the written XAO is read back and checked by OpenCASCADE."""
    pytest.importorskip("OCP")
    model = build(
        pad_in_ground("B"), "B", *by_type("MA", "MS", "SA", margins=(0.5, 2.0))
    )
    assert emit_model(model, tmp_path, strict=True).occt is not None
