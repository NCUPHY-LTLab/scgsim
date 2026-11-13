"""Reading the sidecar facts the meshing module consumes, after the shared reader has checked the sidecar."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from meshing_support import feature, film, sidecar_with

from scgsim.meshing import SidecarError
from scgsim.meshing.features import load_features, read_features


def test_every_kind_of_the_interface_is_read_with_its_facts():
    """Every kind of the interface is read with its facts, in the order of kind, source and facts."""
    sidecar = sidecar_with(
        film([1, 2], 0.2),
        feature(
            "band_boundary",
            [3, 4],
            {"margins_um": [0.5, 1.0], "narrowest_band_um": 0.5},
        ),
        feature("arc", [5], {"radius_um": 10.0}, source=None),
        feature("port_edge", [7, 8]),
    )
    features = {f.kind: f for f in read_features(sidecar).features}
    assert [f.kind for f in read_features(sidecar).features] == sorted(features)
    film_wall = features["film_sidewall"]
    assert film_wall.native_tags == (1, 2)
    assert film_wall.edge_ids == ("e0", "e1")
    assert film_wall.fact("thickness_um") == 0.2
    assert features["band_boundary"].facts["margins_um"] == [0.5, 1.0]
    assert features["arc"].source is None
    assert features["arc"].fact("radius_um") == 10.0


def test_the_binding_groups_and_embeddings_are_carried_for_the_audit():
    """The binding, the groups with the members that name them, and the embeddings by native tag."""
    sidecar = sidecar_with(film([1]), embedded_faces=[("f2", "vol1")])
    result = read_features(sidecar)
    assert result.binding.id_of(1, 4) == "e3"
    assert result.binding.id_of(2, 3) == "f2"
    assert result.binding.id_of(1, 99) is None
    assert {g.name: (g.dimension, g.ids) for g in result.groups}[
        "MA::pad__sidewall"
    ] == (2, ("f0", "f1"))
    assert [(e.face, e.host_volume) for e in result.embedded_faces] == [(3, 2)]


def test_features_by_curve_lists_every_feature_that_covers_a_curve():
    """Features by curve lists every feature that covers a curve, in reading order."""
    result = read_features(
        sidecar_with(
            film([1, 2]),
            feature("arc", [2, 3], {"radius_um": 5.0}, source=None),
        )
    )
    covering = result.features_by_curve()
    assert sorted(covering) == [1, 2, 3]
    assert [f.kind for f in covering[2]] == ["arc", "film_sidewall"]


def test_the_features_are_frozen():
    """The features are frozen."""
    result = read_features(sidecar_with(film([1])))
    with pytest.raises(AttributeError):
        result.features[0].kind = "arc"  # type: ignore[misc]
    with pytest.raises(TypeError):
        result.features[0].facts["thickness_um"] = 1.0  # type: ignore[index]


def test_a_fact_the_contract_does_not_define_is_refused():
    """A new fact is a change of the contract, so a sidecar that carries one is not this version's."""
    entry = film([1])
    entry["facts"]["column_count"] = 3
    with pytest.raises(SidecarError, match="unknown field 'column_count'"):
        read_features(sidecar_with(entry))


def test_another_version_is_refused_by_name():
    """A sidecar of another version is refused by its version."""
    sidecar = sidecar_with()
    sidecar["schema"] = "scgsim.sgb.sidecar.v3"
    with pytest.raises(
        SidecarError, match="sidecar.v3.*sidecar.v4.*Rebuild the geometry"
    ):
        read_features(sidecar)


def test_an_unknown_kind_is_refused():
    """An unknown kind matches no feature record of the contract."""
    with pytest.raises(SidecarError, match="mesh_feature"):
        read_features(sidecar_with(feature("curved_edge", [1])))


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        (feature("film_sidewall", [1]), "lacks the field 'thickness_um'"),
        (film([1], 0.0), "0.0 is not greater than 0"),
        (film([1], -0.2), "-0.2 is not greater than 0"),
        (feature("arc", [1], {"radius_um": "ten"}), "'ten' is not number"),
        (
            feature(
                "band_boundary",
                [1],
                {"margins_um": [1.0, 0.5], "narrowest_band_um": 0.5},
            ),
            "ascending",
        ),
        (
            feature("band_boundary", [1], {"margins_um": [], "narrowest_band_um": 0.5}),
            "fewer than 1",
        ),
        (
            feature("band_boundary", [1], {"margins_um": [0.5]}),
            "lacks the field 'narrowest_band_um'",
        ),
    ],
)
def test_a_missing_or_invalid_fact_is_refused(entry, message):
    """A missing or invalid fact is refused."""
    with pytest.raises(SidecarError, match=message):
        read_features(sidecar_with(entry))


def test_edges_that_name_no_written_edge_are_refused():
    """A feature names written edges, each once."""
    outside = film([1])
    outside["edges"] = ["e39"]
    with pytest.raises(SidecarError, match="'e39', which edge lacks"):
        read_features(sidecar_with(outside))
    twice = film([1])
    twice["edges"] = ["e0", "e0"]
    with pytest.raises(SidecarError, match="names an edge more than once"):
        read_features(sidecar_with(twice))


def test_an_embedding_that_names_an_unknown_id_is_refused():
    """An embedding that names an unknown id is refused."""
    sidecar = sidecar_with()
    sidecar["volume"]["vol0"]["embedded_faces"].append("nope")
    with pytest.raises(SidecarError, match="'nope', which face lacks"):
        read_features(sidecar)


def test_a_sidecar_without_features_or_binding_is_refused():
    """A sidecar without features or binding is refused."""
    broken = sidecar_with()
    del broken["mesh_feature"]
    with pytest.raises(SidecarError, match="lacks the field 'mesh_feature'"):
        read_features(broken)
    broken = sidecar_with()
    del broken["binding"]
    with pytest.raises(SidecarError, match="lacks the field 'binding'"):
        read_features(broken)


def test_a_sidecar_file_loads(tmp_path: Path):
    """A sidecar file loads."""
    path = tmp_path / "case.sidecar.json"
    path.write_text(json.dumps(sidecar_with(film([1, 2]))), encoding="utf-8")
    assert load_features(path).features[0].native_tags == (1, 2)


def test_the_conductors_are_the_entities_of_conductor_volumes_emitted_or_not():
    """A Route B conductor volume is listed unemitted; a dielectric is not a conductor."""
    assert read_features(sidecar_with()).conductors == {"pad"}
    sidecar = sidecar_with(film([1]))
    sidecar["entity"]["ground"] = dict(sidecar["entity"]["pad"])
    sidecar["volume"]["vol3"] = {
        **sidecar["volume"]["vol2"],
        "entity": "ground",
    }
    sidecar["summary"]["counts"]["volumes_total"] = 4
    assert read_features(sidecar).conductors == {"pad", "ground"}
