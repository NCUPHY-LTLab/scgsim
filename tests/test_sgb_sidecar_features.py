"""The ``mesh_features`` of the sidecar ``scgsim.sgb.sidecar.v2``: facts about the geometry, never a mesh size."""

from __future__ import annotations

import json
from collections import Counter

import jsonschema
import pytest
from sgb_emit_cases import CASES
from sgb_extreme_cases import catalogue_of, chip, disk
from sgb_scenes import GRID, chip_prisms, conductor, footprint, pad_in_ground, rect_um

from scgsim.contracts.sgb_sidecar import sidecar_schema, views
from scgsim.geometry.sgb.bands.requests import ConductorSurface, InsetRequest
from scgsim.geometry.sgb.checks.slivers.report import sliver_report
from scgsim.geometry.sgb.exact.grid import exact_z
from scgsim.geometry.sgb.exact.prisms import PrismCatalogue, PrismRecord
from scgsim.geometry.sgb.model import ConformalModel, build_model_from_catalogue
from scgsim.geometry.sgb.output.emission.plan import plan_brep
from scgsim.geometry.sgb.output.sidecar.build import SidecarContext, build_sidecar
from scgsim.geometry.sgb.output.sidecar.features import FEATURE_KINDS

FACTS_OF_KIND = {
    "film_sidewall": {"thickness_um"},
    "band_boundary": {"margins_um", "narrowest_band_um"},
    "arc": {"radius_um"},
    "port_edge": set(),
}


def sidecar_of(model: ConformalModel) -> dict:
    """The sidecar of a model (not written)."""
    return build_sidecar(
        model,
        plan_brep(model.topology, model.routed),
        SidecarContext(
            sliver_report(model.labelled, model.topology, model.routed)["summary"]
        ),
    )


def features(side: dict) -> list[dict]:
    """The mesh features in reading order, each with the native tags of its edges."""
    return [
        {**f, "native_tags": views.native_tags(side, 1, f["edges"])}
        for f in views.mesh_features(side)
    ]


def features_of(side: dict, kind: str) -> list[dict]:
    """The features of one kind, in reading order."""
    return [f for f in features(side) if f["kind"] == kind]


def edge_tags_of_faces(side: dict, face_ids: list[str]) -> set[int]:
    """The native tags of every edge on the boundary of the given faces (from the sidecar alone)."""
    face_tag = {fid: i for i, fid in enumerate(side["binding"]["ids_by_tag"]["2"])}
    face_edges = views.expected(side)["face_edges"]
    return {t for fid in face_ids for t in face_edges[face_tag[fid]]}


def band_family(side: dict, carrier: str) -> str | None:
    """The band family a carrier is a band curve of, through the primitive and ring it names; else ``None``."""
    record = side["carrier"][carrier]
    if record["kind"] != "band_member":
        return None
    ring = side["support_primitive"][record["primitive"]]["ring"]
    return side["support_ring"][ring]["family"]


def sidewall_faces_of(side: dict, body: str) -> list[str]:
    """The sidewall faces that have the conductor ``body`` on one side (or on the model boundary), read from the groups."""
    members = views.group_members(side)
    return sorted(
        fid
        for name, group in side["group"].items()
        if group["dim"] == 2
        and group["face_kind"] == "sidewall"
        and group["interface_type"] in ("MA", "MS", "MM", "EXT")
        and body in group["entities"]
        for fid in members[name]
    )


def disc_model(route: str) -> ConformalModel:
    """One round pad of radius 5 um on a small chip: four quarter arcs, no polygon."""
    prisms = [*chip(), disk("disc", "N", (0, 0), 5000, route)]
    return build_model_from_catalogue(catalogue_of("disc", route, prisms), route)


def oxide_model(route: str) -> ConformalModel:
    """A 0.1 um dielectric film (SiO2) on the substrate, away from the chip boundary."""
    prisms = chip_prisms(GRID, 40.0)
    outline = footprint(rect_um(-10, -10, 10, 10))
    prisms.append(
        PrismRecord(
            "oxide",
            outline,
            exact_z(0),
            exact_z("0.1"),
            "SiO2",
            "dielectric",
            priority=1,
        )
    )
    return build_model_from_catalogue(
        PrismCatalogue(GRID, tuple(prisms), name="film"), route
    )


@pytest.fixture(scope="module")
def sidecars(emit_models) -> dict[str, dict]:
    """The sidecar of every named case, built once."""
    return {name: sidecar_of(model) for name, model in emit_models.items()}


# --------------------------------------------------------------------------- shape, order, schema


@pytest.mark.parametrize("name", sorted(CASES))
def test_every_feature_has_a_known_kind_the_facts_of_that_kind_and_no_size(
    sidecars, name
):
    """Each feature names a kind of the list, only the facts of that kind, and its edges in tag order."""
    side = sidecars[name]
    for feature in side["mesh_feature"].values():
        assert set(feature) == {"kind", "source", "edges", "facts"}
        assert feature["kind"] in FEATURE_KINDS
        assert set(feature["facts"]) == FACTS_OF_KIND[feature["kind"]]
        tags = views.native_tags(side, 1, feature["edges"])
        assert feature["edges"] and tags == sorted(set(tags))


@pytest.mark.parametrize("name", sorted(CASES))
def test_features_are_sorted_by_kind_source_and_canonical_facts(sidecars, name):
    """The features read in the order of (kind, source or "", canonical JSON of the facts), with no duplicate key."""
    keys = [
        (
            f["kind"],
            f["source"] or "",
            json.dumps(f["facts"], sort_keys=True, separators=(",", ":")),
        )
        for f in features(sidecars[name])
    ]
    assert keys == sorted(keys) and len(set(keys)) == len(keys)


def test_the_schema_accepts_every_kind_and_rejects_an_unknown_kind_a_size_and_a_missing_fact(
    sidecars,
):
    """A kind outside the list, a stowaway size, a missing fact and a wrong source type are schema errors."""
    validator = jsonschema.Draft202012Validator(sidecar_schema())
    good = sidecars["pad_b_bands"]
    assert {f["kind"] for f in good["mesh_feature"].values()} == set(FEATURE_KINDS)
    assert not list(validator.iter_errors(good))
    by_kind = {f["kind"]: f for f in good["mesh_feature"].values()}
    broken_features = [
        {**by_kind["arc"], "kind": "gap"},
        {**by_kind["arc"], "size": 1.0},
        {**by_kind["arc"], "facts": {"radius_um": 1.0, "size_um": 0.2}},
        {**by_kind["film_sidewall"], "facts": {}},
        {**by_kind["film_sidewall"], "facts": {"thickness_um": 0.0}},
        {**by_kind["band_boundary"], "facts": {"margins_um": [0.1]}},
        {**by_kind["band_boundary"], "source": None},
        {**by_kind["port_edge"], "facts": {"thickness_um": 0.2}},
        {**by_kind["port_edge"], "edges": []},
    ]
    for broken in broken_features:
        side = {**good, "mesh_feature": {**good["mesh_feature"], "mf:broken": broken}}
        assert list(validator.iter_errors(side)), broken["kind"]


# --------------------------------------------------------------------------- film sidewalls


def test_film_sidewall_covers_every_edge_of_a_conductors_sidewall_faces_with_its_thickness(
    sidecars,
):
    """Route B: pad and ground are 0.2 um films, and each feature is exactly the edge set of its sidewall faces."""
    side = sidecars["pad_b"]
    films = {f["source"]: f for f in features_of(side, "film_sidewall")}
    assert set(films) == {"pad", "ground"}
    for body, feature in films.items():
        assert feature["facts"] == {"thickness_um": 0.2}
        assert set(feature["native_tags"]) == edge_tags_of_faces(
            side, sidewall_faces_of(side, body)
        )


def test_a_level_that_splits_a_wall_does_not_make_it_a_thinner_film(sidecars):
    """The mid-plane port of the stage-level scene cuts some walls at z = 0.1; the column is still 0.2 um."""
    side = sidecars["pad_b_bands"]
    halves = [
        f
        for f in side["face"].values()
        if f["face_kind"] == "sidewall" and f["z_um"] == [0.0, 0.1]
    ]
    assert halves  # walls were split by the port plane
    assert {f["facts"]["thickness_um"] for f in features_of(side, "film_sidewall")} == {
        0.2
    }
    assert Counter(f["source"] for f in features_of(side, "film_sidewall")) == {
        "pad": 1,
        "ground": 1,
    }


def test_route_c_adds_the_walls_of_the_retained_conductor_on_the_model_boundary(
    sidecars,
):
    """Route C emits the ground walls on the chip edge too, so its ground feature has more edges than Route B's."""
    b, c = sidecars["pad_b"], sidecars["pad_c"]
    ground_b = next(
        f for f in features_of(b, "film_sidewall") if f["source"] == "ground"
    )
    ground_c = next(
        f for f in features_of(c, "film_sidewall") if f["source"] == "ground"
    )
    assert ground_b["facts"] == ground_c["facts"] == {"thickness_um": 0.2}
    assert set(ground_b["edges"]) < set(ground_c["edges"])
    assert set(ground_c["native_tags"]) == edge_tags_of_faces(
        c, sidewall_faces_of(c, "ground")
    )


def test_a_sheet_has_no_film_sidewall(sidecars):
    """Route A conductors are sheets of zero thickness; the substrate and vacuum walls lie on the model boundary."""
    for name in ("pad_a", "pad_a_bands"):
        assert not features_of(sidecars[name], "film_sidewall"), name


def test_stacked_bodies_have_the_thickness_of_their_own_slab(sidecars):
    """The airbridge: 0.2 um ground, 2.8 um posts and a 0.3 um deck, one feature each."""
    thickness = {
        f["source"]: f["facts"]["thickness_um"]
        for f in features_of(sidecars["bridge_b"], "film_sidewall")
    }
    assert thickness == {
        "ground_left": 0.2,
        "ground_right": 0.2,
        "post_left": 2.8,
        "post_right": 2.8,
        "deck": 0.3,
    }


@pytest.mark.parametrize("route", ["B", "C"])
def test_a_dielectric_film_off_the_model_boundary_is_a_film(route):
    """A 0.1 um oxide is a finite body: twelve edges (top, bottom, vertical); the substrate's chip-edge walls are not."""
    side = sidecar_of(oxide_model(route))
    ((film,),) = [features_of(side, "film_sidewall")]
    assert (film["source"], film["facts"], len(film["edges"])) == (
        "oxide",
        {"thickness_um": 0.1},
        12,
    )


@pytest.mark.parametrize(
    ("route", "thick"),
    [("B", [0.1, 0.2]), ("C", [0.2])],
)
def test_the_thickness_counts_only_emitted_walls_so_it_follows_the_route(route, thick):
    """Two touching conductors of one Net, 0.2 and 0.1 um: Route B does not emit the wall between them."""
    prisms = [
        *chip_prisms(GRID, 40.0),
        conductor("thick", rect_um(-10, -5, 0, 5), 0, "0.2", "N"),
        conductor("thin", rect_um(0, -5, 10, 5), 0, "0.1", "N"),
    ]
    model = build_model_from_catalogue(
        PrismCatalogue(GRID, tuple(prisms), name="touching"), route
    )
    films = features_of(sidecar_of(model), "film_sidewall")
    assert [
        f["facts"]["thickness_um"] for f in films if f["source"] == "thick"
    ] == thick
    assert [f["facts"]["thickness_um"] for f in films if f["source"] == "thin"] == [0.1]


@pytest.mark.parametrize("route", ["B", "C"])
def test_a_wall_shared_with_a_conductor_stays_part_of_the_dielectric_film(route):
    """A 0.05 um conductor along the oxide's whole right wall: the oxide keeps its 0.1 um column, the conductor its own."""
    prisms = [
        *oxide_model(route).labelled.catalogue.prisms,
        conductor("strip", rect_um(10, -10, 20, 10), 0, "0.05", "S"),
    ]
    model = build_model_from_catalogue(
        PrismCatalogue(GRID, tuple(prisms), name="film_and_strip"), route
    )
    films = features_of(sidecar_of(model), "film_sidewall")
    assert [(f["source"], f["facts"]["thickness_um"]) for f in films] == [
        ("oxide", 0.1),
        ("strip", 0.05),
    ]
    oxide, strip = films
    shared = set(oxide["native_tags"]) & set(strip["native_tags"])
    assert shared, "the edges of the shared wall belong to both bodies"


def test_band_boundary_keeps_the_edges_of_the_inset_family_and_adds_its_margins(
    sidecars,
):
    """pad_b_bands: three families, margins 0.1, 0.5 and 2 um, the narrowest band 0.1 um wide."""
    side = sidecars["pad_b_bands"]
    bands = {f["source"]: f for f in features_of(side, "band_boundary")}
    assert set(bands) == {"band_pad_top", "band_pad_bottom", "band_gap"}
    for family, feature in bands.items():
        assert feature["facts"] == {
            "margins_um": [0.1, 0.5, 2.0],
            "narrowest_band_um": 0.1,
        }
        assert all(
            any(band_family(side, c) == family for c in side["edge"][e]["carriers"])
            for e in feature["edges"]
        )


@pytest.mark.parametrize(
    ("margins", "narrowest"),
    [
        ((0.5, 1.0), 0.5),
        ((0.5, 0.6, 2.0), 0.1),
        ((0.3,), 0.3),
        ((0.25, 1.0, 1.125), 0.125),
    ],
)
def test_narrowest_band_is_the_smallest_difference_between_margins_with_zero_prepended(
    margins, narrowest
):
    """The first band starts at 0, so the first margin counts as a width; the number is a clean decimal."""
    request = InsetRequest("bands", ConductorSurface("pad", "top"), margins)
    model = build_model_from_catalogue(pad_in_ground("B"), "B", insets=(request,))
    (feature,) = features_of(sidecar_of(model), "band_boundary")
    assert feature["facts"] == {
        "margins_um": list(margins),
        "narrowest_band_um": narrowest,
    }
    assert (
        json.dumps(feature["facts"]).count("0000") == 0
    )  # no float noise such as 0.09999999999999998


# --------------------------------------------------------------------------- arcs


def test_arc_covers_every_arc_edge_once_grouped_by_owner_and_exact_radius(sidecars):
    """pad_b: the 5 um rounded corner and the 8 um hole, both owned by the pad; every arc edge is in one entry."""
    side = sidecars["pad_b"]
    arcs = {(f["source"], f["facts"]["radius_um"]): f for f in features_of(side, "arc")}
    assert set(arcs) == {("pad", 5.0), ("pad", 8.0)}
    assert (len(arcs["pad", 5.0]["edges"]), len(arcs["pad", 8.0]["edges"])) == (
        2,
        8,
    )  # top and bottom outline
    assert sum(len(f["edges"]) for f in arcs.values()) == side["summary"]["arc_edges"]


def test_band_arcs_have_the_radius_of_their_offset_circle_and_the_owner_of_a_conductor_band(
    sidecars,
):
    """Bands add arcs at radii such as 8.1 and 7.9 um; a gap band between two regions has no single owner."""
    side = sidecars["pad_b_bands"]
    arcs = features_of(side, "arc")
    all_edges = [e for f in arcs for e in f["edges"]]
    assert len(all_edges) == len(set(all_edges)) == side["summary"]["arc_edges"]
    radii = {(f["source"], f["facts"]["radius_um"]) for f in arcs}
    assert ("pad", 8.1) in radii and (None, 7.9) in radii
    assert all(source in ("pad", None) for source, _ in radii)


@pytest.mark.parametrize("route", ["B", "C"])
def test_a_round_pad_has_one_arc_feature_of_its_exact_radius(route):
    """A disc of radius 5000 g: four quarter arcs on the top and four on the bottom, radius exactly 5 um."""
    side = sidecar_of(disc_model(route))
    ((arc,),) = [features_of(side, "arc")]
    assert (arc["source"], arc["facts"], len(arc["edges"])) == (
        "disc",
        {"radius_um": 5.0},
        8,
    )
    (film,) = features_of(side, "film_sidewall")
    assert film["source"] == "disc" and film["facts"] == {"thickness_um": 0.2}
    assert set(arc["edges"]) <= set(
        film["edges"]
    )  # the arcs are the outline of the curved wall


def test_a_polygonal_outline_has_no_arc_feature():
    """pad_polygon_a replaces the circle by a regular polygon and has no rounded corner."""
    assert not features_of(sidecar_of(CASES["pad_polygon_a"]()), "arc")


# --------------------------------------------------------------------------- ports


@pytest.mark.parametrize("name", ["pad_a", "pad_b", "pad_c", "pad_b_bands"])
def test_port_edge_is_the_boundary_of_the_lumped_port_faces(sidecars, name):
    """One feature per port sheet: the four edges of its face."""
    side = sidecars[name]
    ((port_id, port),) = side["port"].items()
    (feature,) = features_of(side, "port_edge")
    assert feature["source"] == port_id and feature["facts"] == {}
    assert set(feature["native_tags"]) == edge_tags_of_faces(side, port["faces"])
    assert len(feature["edges"]) == 4


def test_a_model_without_a_port_has_no_port_edge(sidecars):
    """Only the ports of the ports block."""
    assert not features_of(sidecars["bridge_b"], "port_edge")
