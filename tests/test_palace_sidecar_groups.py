"""The Palace group dict from an SGB sidecar, and the attributes its surface groups lower to (file only)."""

from __future__ import annotations

import pytest
from palace_sgb_support import effective_input, sidecar_of
from test_sgb_band_labels import PORT_CASES, all_interface_insets

from scgsim.palace.mesh.attributes import lower_surface_group
from scgsim.palace.mesh.physical_groups import groups_from_sidecar, rows_of


@pytest.fixture(params=["A", "B"])
def routed(request, tmp_path):
    """(route, sidecar) of the plain build of each route."""
    route = request.param
    effective = effective_input(route, tmp_path)
    return route, sidecar_of(effective, route, tmp_path)


def test_groups_have_the_four_keys_and_no_old_record_fields(routed):
    """The dict keeps the four keys; the compiler-record fields and ``surface_groups`` are gone."""
    _, sidecar = routed
    groups = groups_from_sidecar(sidecar)
    assert set(groups) == {
        "volumes",
        "boundary_surfaces",
        "pec_surfaces",
        "port_surfaces",
    }
    forbidden = {
        "solver_use",
        "role",
        "representation",
        "equipotential_id",
        "owner_semantic_ids",
        "physical_attribute",
        "sgb_record",
        "phys_group",
    }
    for section in groups.values():
        for record in section.values():
            assert not forbidden & set(record)


def test_volumes_are_the_dim_3_groups_with_one_attribute_each(routed):
    """Volumes come from ``dim == 3`` groups, with material and kind from the sidecar."""
    _, sidecar = routed
    volumes = groups_from_sidecar(sidecar)["volumes"]
    expected = {n for n, g in sidecar["physical_groups"].items() if g["dim"] == 3}
    assert set(volumes) == expected
    assert sorted(a for v in volumes.values() for a in v["attributes"]) == list(
        range(1, len(expected) + 1)
    )
    for name, volume in volumes.items():
        group = sidecar["physical_groups"][name]
        assert volume["tags"] == group["native_tags"]
        assert (volume["kind"], volume["material"]) == (
            group["kind"],
            group["material"],
        )


def test_boundaries_ports_and_pec_follow_the_sidecar_rules(routed):
    """EXT is a domain boundary, ports sit in ``port_surfaces`` only, and PEC is MA/MS/SHEET with a Net."""
    _, sidecar = routed
    groups = groups_from_sidecar(sidecar)
    boundaries = groups["boundary_surfaces"]
    port_groups = {p["group"] for p in sidecar["ports"]}
    assert not port_groups & set(boundaries)
    for name, record in boundaries.items():
        group = sidecar["physical_groups"][name]
        assert (record["source"] == "domain_boundary") == (
            group["interface_type"] == "EXT"
        )
        assert record["surface_epr"] == bool(record["rows"])
        assert record["interface_type"] not in {"PORT"}
    pec = groups["pec_surfaces"]
    assert pec
    for name, record in pec.items():
        assert record["interface_type"] in {"MA", "MS", "SHEET"}
        assert record["net"] is not None
        assert record["entity"] == sidecar["physical_groups"][name]["entities"][0]
    for key, port in groups["port_surfaces"].items():
        (source,) = (p for p in sidecar["ports"] if f"P{p['port_index']}" == key)
        assert port["group"] == source["group"]
        assert port["host_entity_ids"] == source["host_entity_ids"]
        assert (
            port["tags"] == sidecar["physical_groups"][source["group"]]["native_tags"]
        )


def test_pec_surfaces_are_exactly_the_union_of_the_nets(routed):
    """The conductor surfaces by Net are the sidecar's ``unions.nets``: Terminal and Ground bind the same groups."""
    _, sidecar = routed
    pec = groups_from_sidecar(sidecar)["pec_surfaces"]
    by_net: dict[str, set[str]] = {}
    for name, record in pec.items():
        by_net.setdefault(record["net"], set()).add(name)
    assert by_net == {n: set(g) for n, g in sidecar["unions"]["nets"].items()}


def test_conductor_component_ids_are_the_entity_ids_of_the_component(routed):
    """Open item 7: the component id Palace binds is the S0 Entity id of the conductor."""
    route, sidecar = routed
    pec = groups_from_sidecar(sidecar)["pec_surfaces"]
    expected = {"A": {"chip/left", "chip/right"}, "B": {"ground_0000", "pad"}}[route]
    assert {record["entity"] for record in pec.values()} == expected


def test_attributes_partition_the_faces_of_each_dimension(routed):
    """Attributes are numbered from 1 per dimension and cover every face of every group exactly once."""
    _, sidecar = routed
    groups = groups_from_sidecar(sidecar)
    parts = [
        part
        for section in ("boundary_surfaces", "port_surfaces")
        for record in groups[section].values()
        for part in record["parts"]
    ]
    assert sorted(p["attribute"] for p in parts) == list(range(1, len(parts) + 1))
    tags = [tag for part in parts for tag in part["tags"]]
    assert len(tags) == len(set(tags))
    surface_tags = {
        tag
        for g in sidecar["physical_groups"].values()
        if g["dim"] == 2
        for tag in g["native_tags"]
    }
    assert set(tags) == surface_tags


def test_a_group_with_one_signature_keeps_one_attribute_under_its_name(routed):
    """Without bands every group has one signature: one attribute, named like the group."""
    _, sidecar = routed
    for name, record in groups_from_sidecar(sidecar)["boundary_surfaces"].items():
        assert len(record["parts"]) == 1
        assert record["parts"][0]["name"] == name


def _synthetic_group(sides: dict[str, tuple[str, str]]) -> tuple[dict, dict]:
    """A sheet group of faces f1..fn and its MA and MS rows; ``sides[f] = (ma side, ms side)`` ('' = no row)."""
    faces = list(sides)
    record = {"ids": faces, "native_tags": list(range(1, len(faces) + 1))}
    rows: dict[str, dict] = {}
    for position, row_type in enumerate(("MA", "MS")):
        face_sides = {f: s[position] for f, s in sides.items() if s[position]}
        if face_sides:
            rows[f"{row_type}::e__sheet"] = {
                "face_sides": face_sides,
                "shared_face_group": "e",
            }
    return record, rows


def test_a_sheet_over_vacuum_and_dielectric_gives_ma_and_ms_on_one_attribute():
    """Vacuum above and dielectric below everywhere: one signature, the MA row (above) and the MS row (below)."""
    record, rows = _synthetic_group(
        {"f1": ("above", "below"), "f2": ("above", "below")}
    )
    (part,) = lower_surface_group("e", record, rows, 5)
    assert part.attribute == 5
    assert part.rows == (("MA::e__sheet", "above"), ("MS::e__sheet", "below"))


def test_a_floating_sheet_with_vacuum_on_both_sides_has_an_ma_row_and_no_ms_row():
    """A sheet in vacuum: one attribute carrying only MA, integrated on both sides."""
    record, rows = _synthetic_group({"f1": ("both", ""), "f2": ("both", "")})
    (part,) = lower_surface_group("e", record, rows, 1)
    assert part.rows == (("MA::e__sheet", "both"),)


def test_a_sheet_over_vacuum_and_dielectric_in_different_places_splits_by_signature():
    """Where the dielectric is above instead of below the signature differs, so the faces split."""
    record, rows = _synthetic_group(
        {
            "f1": ("above", "below"),
            "f2": ("below", "above"),
            "f3": ("above", "below"),
            "f4": ("both", ""),
        }
    )
    parts = lower_surface_group("e", record, rows, 10)
    assert [p.attribute for p in parts] == [10, 11, 12]
    assert [p.name for p in parts] == ["e#1", "e#2", "e#3"]
    by_faces = {p.faces: p.rows for p in parts}
    assert by_faces[("f1", "f3")] == (
        ("MA::e__sheet", "above"),
        ("MS::e__sheet", "below"),
    )
    assert by_faces[("f2",)] == (("MA::e__sheet", "below"), ("MS::e__sheet", "above"))
    assert by_faces[("f4",)] == (("MA::e__sheet", "both"),)
    assert sorted(t for p in parts for t in p.tags) == [1, 2, 3, 4]


def test_a_group_row_that_covers_only_some_faces_leaves_the_others_unsigned():
    """A row listing some faces of its group puts the rest on an attribute with no row."""
    record, rows = _synthetic_group({"f1": ("above", ""), "f2": ("", "")})
    parts = lower_surface_group("e", record, rows, 1)
    assert sorted(p.rows for p in parts) == [(), (("MA::e__sheet", "above"),)]


def test_bands_become_separate_groups_with_family_margins(tmp_path):
    """A banded build: each band is a group with attributes, and the rows know their family's margins."""
    route = "B"
    effective = effective_input(route, tmp_path)
    _, sa_pair, *_ = PORT_CASES[route]
    insets = all_interface_insets(effective, route, sa_pair)
    sidecar = sidecar_of(effective, route, tmp_path, insets)
    groups = groups_from_sidecar(sidecar)
    rows = rows_of(groups)
    banded = {r: row for r, row in rows.items() if row["band"] is not None}
    assert banded
    for row in banded.values():
        assert row["margins_um"] == [0.5, 1.0]
        assert row["family"] is not None
        assert row["attribute_sides"]
    assert not any(
        row["shared_face_group"] for row in rows.values()
    )  # Route B has no sheet rows


def test_route_a_sheet_rows_share_the_attribute_of_their_group(tmp_path):
    """Route A: the MA and MS row of one sheet band carry the same attribute, with their own sides."""
    effective = effective_input("A", tmp_path)
    _, sa_pair, *_ = PORT_CASES["A"]
    sidecar = sidecar_of(
        effective, "A", tmp_path, all_interface_insets(effective, "A", sa_pair)
    )
    rows = rows_of(groups_from_sidecar(sidecar))
    ma, ms = rows["MA::chip/left__sheet::band=1"], rows["MS::chip/left__sheet::band=1"]
    assert set(ma["attribute_sides"]) == set(ms["attribute_sides"])
    assert set(ma["attribute_sides"].values()) == {"above"}
    assert set(ms["attribute_sides"].values()) == {"below"}
    assert ma["shared_face_group"] == ms["shared_face_group"] == "chip/left::band=1"
