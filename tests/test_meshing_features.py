"""Reading and validating the sidecar facts the meshing module consumes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from meshing_support import feature, film, sidecar_with

from scgsim.meshing import SidecarError
from scgsim.meshing.features import FEATURE_KINDS, load_features, read_features


def test_every_kind_of_the_interface_is_read_with_its_facts():
    """Every kind of the interface is read with its facts."""
    sidecar = sidecar_with(
        film([1, 2], 0.2),
        feature(
            "band_boundary",
            [3, 4],
            {"margins_um": [0.5, 1.0], "narrowest_band_um": 0.5},
        ),
        feature("arc", [5], {"radius_um": 10.0}, source=None),
        feature("refinement_curve", [6]),
        feature("port_edge", [7, 8]),
    )
    features = read_features(sidecar).features
    assert [f.kind for f in features] == list(FEATURE_KINDS)
    assert features[0].native_tags == (1, 2)
    assert features[0].edge_ids == ("e0", "e1")
    assert features[0].fact("thickness_um") == 0.2
    assert features[1].facts["margins_um"] == [0.5, 1.0]
    assert features[2].source is None
    assert features[2].fact("radius_um") == 10.0


def test_the_binding_groups_and_embeddings_are_carried_for_the_audit():
    """The binding groups and embeddings are carried for the audit."""
    sidecar = sidecar_with(
        film([1]),
        embedded_faces=[{"face": "f2", "host_volume": "vol1", "group": None}],
        refinement_edges=[
            {
                "edge": "e3",
                "host_face": "f1",
                "family": "r",
                "written_as_free_edge": True,
            }
        ],
    )
    result = read_features(sidecar)
    assert result.binding.id_of(1, 4) == "e3"
    assert result.binding.id_of(2, 3) == "f2"
    assert result.binding.id_of(1, 99) is None
    assert {g.name: (g.dimension, g.ids) for g in result.groups}[
        "MA::pad__sidewall"
    ] == (2, ("f0", "f1"))
    assert [(e.face, e.host_volume) for e in result.embedded_faces] == [(3, 2)]
    assert [(e.edge, e.host_face) for e in result.embedded_edges] == [(4, 2)]


def test_features_by_curve_lists_every_feature_that_covers_a_curve():
    """Features by curve lists every feature that covers a curve."""
    result = read_features(
        sidecar_with(
            film([1, 2]),
            feature("arc", [2, 3], {"radius_um": 5.0}, source=None),
        )
    )
    covering = result.features_by_curve()
    assert sorted(covering) == [1, 2, 3]
    assert [f.kind for f in covering[2]] == ["film_sidewall", "arc"]


def test_the_features_are_frozen():
    """The features are frozen."""
    result = read_features(sidecar_with(film([1])))
    with pytest.raises(AttributeError):
        result.features[0].kind = "arc"  # type: ignore[misc]
    with pytest.raises(TypeError):
        result.features[0].facts["thickness_um"] = 1.0  # type: ignore[index]


def test_a_later_additive_fact_is_accepted():
    """A later additive fact is accepted."""
    entry = film([1])
    entry["facts"]["column_count"] = 3
    assert read_features(sidecar_with(entry)).features[0].facts["column_count"] == 3


def test_the_v1_schema_is_refused_by_name():
    """The v1 schema is refused by name."""
    sidecar = sidecar_with()
    sidecar["schema"] = "scgsim.sgb.sidecar.v1"
    with pytest.raises(SidecarError, match="sidecar.v1.*sidecar.v2"):
        read_features(sidecar)


def test_an_unknown_kind_is_a_schema_error():
    """An unknown kind is a schema error."""
    with pytest.raises(SidecarError, match="kind 'curved_edge' is unknown"):
        read_features(sidecar_with(feature("curved_edge", [1])))


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        (feature("film_sidewall", [1]), "lacks the fact 'thickness_um'"),
        (film([1], 0.0), "thickness_um must be a positive number"),
        (film([1], -0.2), "thickness_um must be a positive number"),
        (
            feature("arc", [1], {"radius_um": "ten"}),
            "radius_um must be a positive number",
        ),
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
            "ascending",
        ),
        (
            feature("band_boundary", [1], {"margins_um": [0.5]}),
            "lacks the fact 'narrowest_band_um'",
        ),
    ],
)
def test_a_missing_or_invalid_fact_is_refused(entry, message):
    """A missing or invalid fact is refused."""
    with pytest.raises(SidecarError, match=message):
        read_features(sidecar_with(entry))


def test_edges_that_disagree_with_the_binding_are_refused():
    """Edges that disagree with the binding are refused."""
    wrong = film([1, 2])
    wrong["edges"] = ["e1", "e0"]
    with pytest.raises(SidecarError, match="native tag 1 is not edge 'e1'"):
        read_features(sidecar_with(wrong))
    short = film([1, 2])
    short["edges"] = ["e0"]
    with pytest.raises(SidecarError, match="1 edges but 2 native tags"):
        read_features(sidecar_with(short))
    outside = film([40])
    outside["edges"] = ["e39"]
    with pytest.raises(SidecarError, match="native tag 40"):
        read_features(sidecar_with(outside))


def test_an_embedding_that_names_an_unknown_id_is_refused():
    """An embedding that names an unknown id is refused."""
    sidecar = sidecar_with(embedded_faces=[{"face": "nope", "host_volume": "vol0"}])
    with pytest.raises(SidecarError, match="'nope'"):
        read_features(sidecar)


def test_a_sidecar_without_features_or_binding_is_refused():
    """A sidecar without features or binding is refused."""
    broken = sidecar_with()
    del broken["mesh_features"]
    with pytest.raises(SidecarError, match="mesh_features must be a list"):
        read_features(broken)
    broken = sidecar_with()
    del broken["binding"]
    with pytest.raises(SidecarError, match="binding must be an object"):
        read_features(broken)


def test_a_sidecar_file_loads(tmp_path: Path):
    """A sidecar file loads."""
    path = tmp_path / "case.sidecar.json"
    path.write_text(json.dumps(sidecar_with(film([1, 2]))), encoding="utf-8")
    assert load_features(path).features[0].native_tags == (1, 2)


def test_the_conductors_are_the_entities_of_conductor_volumes_emitted_or_not():
    """A Route B conductor volume is listed unemitted; a dielectric is not a conductor."""
    sidecar = sidecar_with(film([1]))
    sidecar["volumes"]["vol2"] = {
        "entity": "ground",
        "kind": "conductor",
        "emitted": True,
    }
    assert read_features(sidecar).conductors == {"pad", "ground"}
    assert read_features(sidecar_with()).conductors == {"pad"}
