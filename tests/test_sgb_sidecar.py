"""The production sidecar ``scgsim.sgb.sidecar.v1``: schema, binding, group facts, unions, lineage, determinism."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest
from sgb_emit_cases import CASES, pad_model

from scgsim.geometry.sgb.emission.plan import plan_brep
from scgsim.geometry.sgb.routes.naming import FACE_KINDS
from scgsim.geometry.sgb.sidecar.aedt import aedt_names, sanitise_aedt_name
from scgsim.geometry.sgb.sidecar.build import SidecarContext, build_sidecar
from scgsim.geometry.sgb.sidecar.files import (
    read_sidecar,
    sidecar_schema,
    write_sidecar,
)
from scgsim.geometry.sgb.slivers.report import sliver_report

TESTS = Path(__file__).resolve().parent


def sidecar_of(model, **context):
    """The sidecar of a model (not written)."""
    return build_sidecar(
        model,
        plan_brep(model.topology, model.routed),
        SidecarContext(sliver_report(model)["summary"], **context),
    )


@pytest.fixture(scope="module")
def sidecars():
    """The sidecar of every named case, built once."""
    return {name: sidecar_of(build()) for name, build in CASES.items()}


# --------------------------------------------------------------------------- schema


def test_the_schema_is_itself_a_valid_json_schema():
    """The schema is itself a valid json schema."""
    jsonschema.Draft202012Validator.check_schema(sidecar_schema())


@pytest.mark.parametrize("name", sorted(CASES))
def test_every_emitted_sidecar_validates_against_the_schema(sidecars, name):
    """Every emitted sidecar validates against the schema."""
    errors = sorted(
        jsonschema.Draft202012Validator(sidecar_schema()).iter_errors(sidecars[name]),
        key=lambda e: list(e.path),
    )
    assert not errors, [(list(e.path)[:3], e.message[:80]) for e in errors[:3]]


def test_the_schema_rejects_what_a_consumer_must_not_see(sidecars):
    """Negative controls: a missing block, an unknown top-level field, a wrong version and a bad AEDT name."""
    validator = jsonschema.Draft202012Validator(sidecar_schema())
    good = sidecars["pad_b"]
    assert not list(validator.iter_errors(good))
    for broken in (
        {k: v for k, v in good.items() if k != "mesh_hints"},
        {
            **good,
            "mesh_hints": [
                {
                    "kind": "curved_edge",
                    "source": None,
                    "edges": [],
                    "native_tags": [],
                    "size": 1.0,
                }
            ],
        },
        {
            **good,
            "surface_epr_rows": {
                "x": {
                    "group": "g",
                    "interface_type": "MA",
                    "entities": [],
                    "band": None,
                    "side": "left",
                    "shared_face_group": None,
                }
            },
        },
        {k: v for k, v in good.items() if k != "expected"},
        {**good, "stowaway": 1},
        {**good, "schema": "scgsim.sgb.sidecar.v2"},
        {**good, "route": "D"},
        {
            **good,
            "volumes": {
                k: {**v, "aedt_name": "bad name"} for k, v in good["volumes"].items()
            },
        },
    ):
        assert list(validator.iter_errors(broken))


# --------------------------------------------------------------------------- binding and counts


@pytest.mark.parametrize(
    "name", ["pad_a", "pad_b", "pad_c", "bridge_b", "pad_b_bands", "void_shell_b"]
)
def test_ids_by_tag_is_the_plan_order_and_counts_agree(sidecars, name):
    """Ids by tag is the plan order and counts agree."""
    side = sidecars[name]
    ids = side["binding"]["ids_by_tag"]
    counts = side["counts"]
    assert [len(ids[d]) for d in "0123"] == [
        counts["vertices"],
        counts["edges"],
        counts["faces"],
        counts["solids"],
    ]
    assert all(len(v) == len(set(v)) for v in ids.values())
    for group in side["physical_groups"].values():
        dim_ids = ids[str(group["dim"])]
        assert [dim_ids[t - 1] for t in group["native_tags"]] == group["ids"]


def test_primary_groups_partition_each_dimension(sidecars):
    """Primary groups partition each dimension."""
    for name, side in sidecars.items():
        for dim in (2, 3):
            members = [
                i
                for g in side["physical_groups"].values()
                if g["dim"] == dim
                for i in g["ids"]
            ]
            assert len(members) == len(set(members)), name
        tags = [g["tag"] for g in side["physical_groups"].values() if g["dim"] == 2]
        assert sorted(tags) == list(range(1, len(tags) + 1)), name


# --------------------------------------------------------------------------- what a solver adapter reads


def test_a_surface_group_names_its_entities_net_face_kind_and_adjacent_regions(
    sidecars,
):
    """A surface group names its entities, Net, one face kind and the regions on its sides."""
    groups = sidecars["pad_b"]["physical_groups"]
    ma, wall, ms = (
        groups["MA::pad::top"],
        groups["MA::pad::sidewall"],
        groups["MS::pad::bottom"],
    )
    assert ma["entities"] == wall["entities"] == ms["entities"] == ["pad"]
    assert ma["net"] == ms["net"] == "pad"
    assert (ma["face_kind"], wall["face_kind"], ms["face_kind"]) == (
        "top",
        "sidewall",
        "bottom",
    )
    assert [d["kind"] for d in ma["adjacent_domains"]] == ["vacuum"]
    assert [d["kind"] for d in ms["adjacent_domains"]] == ["dielectric"]
    sa = groups["SA::substrate|vacuum::interface"]
    assert sa["face_kind"] == "interface"
    assert sa["entities"] == ["substrate", "vacuum"] and sa["net"] is None
    assert sorted(d["kind"] for d in sa["adjacent_domains"]) == ["dielectric", "vacuum"]
    assert sa["area_um2"] > 0 and len(sa["adjacent_volumes"]) == 2


def test_a_group_never_mixes_face_kinds_and_its_name_says_which_it_is(sidecars):
    """``<kind>::<entity>::<face kind>[::band=k]``: the third part of the name is the group's one face kind."""
    for case, side in sidecars.items():
        for name, group in side["physical_groups"].items():
            if group["dim"] != 2 or group["interface_type"] == "PORT":
                continue
            assert name.split("::")[2] == group["face_kind"] in FACE_KINDS, (case, name)
            faces = [side["faces"][f] for f in group["ids"]]
            assert {f["face_kind"] for f in faces} == {group["face_kind"]}, (case, name)
    ports = [
        g
        for s in sidecars.values()
        for n, g in s["physical_groups"].items()
        if n.startswith("PORT::")
    ]
    assert ports and all(g["face_kind"] == "port" and len(g["ids"]) >= 1 for g in ports)


def test_the_example_group_names_of_the_three_routes(sidecars):
    """The names a consumer sees for the pad in ground."""
    surfaces = {
        case: sorted(
            n for n, g in sidecars[case]["physical_groups"].items() if g["dim"] == 2
        )
        for case in ("pad_a", "pad_b", "pad_c")
    }
    assert surfaces["pad_a"] == [
        "EXT::substrate::bottom", "EXT::substrate::sidewall", "EXT::vacuum::sidewall", "EXT::vacuum::top",
        "PORT::p1", "SA::substrate|vacuum::interface", "SHEET::ground::sheet", "SHEET::pad::sheet",
    ]  # fmt: skip
    assert surfaces["pad_b"] == [
        "EXT::substrate::bottom", "EXT::substrate::sidewall", "EXT::vacuum::sidewall", "EXT::vacuum::top",
        "MA::ground::sidewall", "MA::ground::top", "MA::pad::sidewall", "MA::pad::top",
        "MS::ground::bottom", "MS::pad::bottom", "PORT::p1", "SA::substrate|vacuum::interface",
    ]  # fmt: skip
    assert (
        "EXT::ground::sidewall" in surfaces["pad_c"]
        and "MM::ground|pad::interface" not in surfaces["pad_c"]
    )


def test_route_a_sheet_groups_carry_both_surface_epr_rows(sidecars):
    """Route a sheet groups carry both surface epr rows."""
    side = sidecars["pad_a_bands"]
    rows = side["surface_epr_rows"]
    assert (
        rows["MA::pad::sheet::band=2"]["group"]
        == rows["MS::pad::sheet::band=2"]["group"]
        == "SHEET::pad::sheet::band=2"
    )
    assert rows["MA::pad::sheet::band=2"]["band"] == 2
    assert rows["MA::pad::sheet::band=2"]["entities"] == ["pad"]
    assert rows["SA::substrate|vacuum::interface::band=0"]["interface_type"] == "SA"
    unbanded = sidecars["pad_a"]["surface_epr_rows"]
    assert (
        unbanded["MA::pad::sheet"]["band"] is None and "MA::ground::sheet" in unbanded
    )  # every Surface-EPR row, banded or not


def test_a_sheet_row_marks_the_side_of_the_two_sided_face_it_integrates(sidecars):
    """The MA row sees the vacuum above the sheet, the MS row the substrate below; both name the shared face group."""
    rows = sidecars["pad_a_bands"]["surface_epr_rows"]
    for k in range(4):
        ma, ms = rows[f"MA::pad::sheet::band={k}"], rows[f"MS::pad::sheet::band={k}"]
        assert (ma["side"], ms["side"]) == ("above", "below")
        assert (
            ma["shared_face_group"]
            == ms["shared_face_group"]
            == f"SHEET::pad::sheet::band={k}"
        )
    one_sided = [r for n, r in rows.items() if n.startswith("SA::")]
    assert one_sided and all(
        r["side"] is None and r["shared_face_group"] is None for r in one_sided
    )
    finite = sidecars["pad_b_bands"]["surface_epr_rows"]
    assert all(
        r["side"] is None and r["shared_face_group"] is None for r in finite.values()
    )


def test_route_b_bands_are_separate_top_ma_and_bottom_ms_rows(sidecars):
    """Route b bands are separate top ma and bottom ms rows."""
    rows = sidecars["pad_b_bands"]["surface_epr_rows"]
    assert rows["MA::pad::top::band=1"]["group"] == "MA::pad::top::band=1"
    assert rows["MS::pad::bottom::band=1"]["group"] == "MS::pad::bottom::band=1"


def test_route_c_conductor_interiors_are_volume_groups_of_kind_conductor(sidecars):
    """Route c conductor interiors are volume groups of kind conductor."""
    groups = sidecars["pad_c"]["physical_groups"]
    pad = groups["VOL::pad"]
    assert (pad["dim"], pad["kind"], pad["net"], pad["entities"]) == (
        3,
        "conductor",
        "pad",
        ["pad"],
    )
    assert (
        groups["VOL::vacuum"]["kind"] == "vacuum"
        and groups["VOL::substrate"]["kind"] == "dielectric"
    )
    assert (
        "VOL::pad" not in sidecars["pad_b"]["physical_groups"]
    )  # Route B excludes the conductor volume
    volumes = sidecars["pad_b"]["volumes"]
    assert [v["emitted"] for v in volumes.values() if v["kind"] == "conductor"] == [
        False,
        False,
    ]
    assert sidecars["pad_c"]["route_policy"]["conductor_volumes"] == "retained"


def test_unions_are_lists_of_primary_groups_and_each_group_is_listed_under_the_types_it_carries_rows_for(
    sidecars,
):
    """Every group is in its own interface union; a Route A sheet group is also under MA and MS (it carries both rows)."""
    for name, side in sidecars.items():
        by_type = side["unions"]["interface_types"]
        surface_groups = {
            n for n, g in side["physical_groups"].items() if g["dim"] == 2
        }
        listed = [g for members in by_type.values() for g in members]
        sheets = {n for n in surface_groups if n.startswith("SHEET::")}
        assert sorted(listed) == sorted([*surface_groups, *sheets, *sheets]), name
        for kind, members in by_type.items():
            assert all(
                side["physical_groups"][m]["interface_type"] == kind
                or (kind in ("MA", "MS") and m in sheets)
                for m in members
            ), name
        if sheets:
            assert set(by_type["MA"]) >= sheets and set(by_type["MS"]) >= sheets, name
        for net, members in side["unions"]["nets"].items():
            assert members and all(
                side["physical_groups"][m]["net"] == net for m in members
            ), name


def test_net_unions_list_the_exposed_conductor_groups_of_each_route(sidecars):
    """Net unions list the exposed conductor groups of each route."""
    assert sidecars["pad_a"]["unions"]["nets"]["pad"] == ["SHEET::pad::sheet"]
    finite = ["MA::pad::sidewall", "MA::pad::top", "MS::pad::bottom"]
    assert sidecars["pad_b"]["unions"]["nets"]["pad"] == finite
    assert sidecars["pad_c"]["unions"]["nets"]["pad"] == finite


def test_embedded_faces_and_ports_say_where_the_sheet_lives(sidecars):
    """Embedded faces and ports say where the sheet lives."""
    b = sidecars["pad_b"]
    (embedded,) = b["embedded_faces"]
    assert (
        embedded["group"] == "PORT::p1"
        and embedded["host_volume"] in b["binding"]["ids_by_tag"]["3"]
    )
    (port,) = b["ports"]
    assert (port["group"], port["port_index"], port["direction"]) == (
        "PORT::p1",
        1,
        [1.0, 0.0, 0.0],
    )
    assert port["host_entity_ids"] == ["pad", "ground"] and port["faces"] == [
        embedded["face"]
    ]
    assert port["host_volumes"] == [embedded["host_volume"]] and port[
        "z_um"
    ] == pytest.approx(0.1)
    a = sidecars["pad_a"]
    assert (
        a["embedded_faces"] == [] and len(a["ports"][0]["adjacent_volumes"]) == 2
    )  # Route A: a face between two volumes


def test_volume_records_carry_kind_material_aedt_name_and_planned_volume(sidecars):
    """Volume records carry kind material aedt name and planned volume."""
    for name, side in sidecars.items():
        names = [v["aedt_name"] for v in side["volumes"].values() if v["emitted"]]
        assert all(re.fullmatch(r"[A-Za-z0-9_]+", n) for n in names), name
        assert len(names) == len(set(names)), name
        assert all(
            v["aedt_name"] is None for v in side["volumes"].values() if not v["emitted"]
        ), name
    vol = {
        v["entity"]: v for v in sidecars["pad_b"]["volumes"].values() if v["emitted"]
    }
    assert vol["vacuum"]["aedt_name"] == "VOL__vacuum" and vol["substrate"][
        "volume_um3"
    ] == pytest.approx(9e6)


def test_face_records_carry_group_kind_z_and_the_two_sides(sidecars):
    """Face records carry group kind z and the two sides."""
    side = sidecars["pad_b"]
    faces = side["faces"]
    assert set(faces) == set(side["binding"]["ids_by_tag"]["2"])
    top = next(
        f
        for f in faces.values()
        if f["group"] == "MA::pad::top" and f["face_kind"] == "top"
    )
    assert top["z_um"] == [0.2, 0.2] and top["negative_volume"] is not None


# --------------------------------------------------------------------------- lineage, insets, refinements


def test_lineage_leads_every_edge_to_a_source_curve_and_every_face_to_entities(
    sidecars,
):
    """Lineage leads every edge to a source curve and every face to entities."""
    side = sidecars["pad_b"]
    lineage = side["lineage"]
    horizontal = {e for e in side["binding"]["ids_by_tag"]["1"]} - set(
        lineage["vertical_edges"]
    )
    assert set(lineage["edges"]) == horizontal
    for carriers in lineage["edges"].values():
        assert carriers and all(c in lineage["carriers"] for c in carriers)
    kinds = {c["kind"] for c in lineage["carriers"].values()}
    assert kinds == {"polygon_edge"}  # no unknown carrier is ever left unexplained
    assert lineage["carriers"]["pad/polygon0/ring0/edge2"] == {
        "kind": "polygon_edge",
        "entity": "pad",
        "polygon": 0,
        "ring": 0,
        "edge": 2,
    }
    assert all(entities for entities in lineage["vertical_edges"].values())
    for fid in side["faces"]:
        assert lineage["faces"][fid] and lineage["membership"][fid], fid
        assert all(
            0 < share <= 1.0000001 for share in lineage["membership"][fid].values()
        ), fid


def test_inset_families_record_request_margins_and_the_group_of_every_band(sidecars):
    """Inset families record request margins and the group of every band."""
    side = sidecars["pad_b_bands"]
    by_request = {i["request_id"]: i for i in side["insets"]}
    assert set(by_request) == {"band_pad_top", "band_pad_bottom", "band_gap"}
    top = by_request["band_pad_top"]
    assert top["support"] == {"entity": "pad", "side": "top"} and top["margins_um"] == [
        0.1,
        0.5,
        2.0,
    ]
    assert top["margins_grid"] == [100, 500, 2000] and top["band_count"] == 4
    assert top["bands"] == {str(k): [f"MA::pad::top::band={k}"] for k in range(4)}
    assert by_request["band_gap"]["support"]["first_entity"] == "substrate"
    for group in (
        g for g in side["physical_groups"].values() if g["dim"] == 2 and g["band"]
    ):
        assert group["band"]["family"] in by_request


def test_refinement_requests_list_their_edges_and_host_faces(sidecars):
    """Refinement requests list their edges and host faces."""
    side = sidecars["pad_a_refinement"]
    (request,) = side["refinements"]
    assert request["request_id"] == "ref" and len(request["edges"]) == 1
    (pair,) = side["refinement_edges"]
    assert pair["edge"] == request["edges"][0] and pair["family"] == "ref"
    assert side["faces"][pair["host_face"]]["group"] is not None


def test_mesh_hints_list_band_boundaries_refinement_curves_and_curved_edges_without_sizes(
    sidecars,
):
    """Hints name edges and their source request, never a size."""
    banded = sidecars["pad_b_bands"]
    hints = {(h["kind"], h["source"]): h for h in banded["mesh_hints"]}
    assert {k for k in hints if k[0] == "band_boundary"} == {
        ("band_boundary", "band_pad_top"), ("band_boundary", "band_pad_bottom"), ("band_boundary", "band_gap"),
    }  # fmt: skip
    assert ("curved_edge", None) in hints and not any(
        k[0] == "refinement_curve" for k in hints
    )
    ids = banded["binding"]["ids_by_tag"]["1"]
    for hint in banded["mesh_hints"]:
        assert set(hint) == {"kind", "source", "edges", "native_tags"}  # no sizes
        assert (
            hint["edges"] and [ids[t - 1] for t in hint["native_tags"]] == hint["edges"]
        )
    carriers = banded["lineage"]["carriers"]
    for (kind, source), hint in hints.items():
        if kind == "band_boundary":
            assert all(
                any(
                    carriers[c].get("family") == source
                    for c in banded["lineage"]["edges"][e]
                )
                for e in hint["edges"]
            )
    assert (
        len(hints[("curved_edge", None)]["edges"]) == banded["arrangement"]["arc_edges"]
    )
    refined = sidecars["pad_a_refinement"]
    (curve,) = (h for h in refined["mesh_hints"] if h["kind"] == "refinement_curve")
    assert (
        curve["source"] == "ref"
        and curve["edges"] == refined["refinements"][0]["edges"]
    )
    assert sidecars["pad_a"]["mesh_hints"] == [
        h for h in sidecars["pad_a"]["mesh_hints"] if h["kind"] == "curved_edge"
    ]


def test_arrangement_and_sliver_summaries_are_numbers_for_the_manifest(sidecars):
    """Arrangement and sliver summaries are numbers for the manifest."""
    side = sidecars["pad_b"]
    assert (
        side["arrangement"]["arc_edges"] == 10
        and side["arrangement"]["max_arc_deviation_grid"] is not None
    )


# --------------------------------------------------------------------------- the consumer mappings (documented in the package docstring)


def palace_style_groups(side):
    """The shape of today's Palace group dict, rebuilt from the sidecar alone."""
    groups = {
        "volumes": {},
        "boundary_surfaces": {},
        "pec_surfaces": {},
        "port_surfaces": {},
    }
    for name, g in side["physical_groups"].items():
        if g["dim"] == 3:
            groups["volumes"][name] = {
                "phys_group": g["tag"],
                "material_kind": g["kind"],
                "owner_semantic_ids": g["entities"],
            }
            continue
        record = {
            "phys_group": g["tag"],
            "interface_type": g["interface_type"],
            "owner_semantic_ids": g["entities"],
            "adjacent_solution_volume_ids": [d["name"] for d in g["adjacent_domains"]],
            "face_kind": g["face_kind"],
            "net_id": g["net"],
        }
        groups["boundary_surfaces"][name] = record
        if g["interface_type"] in ("MA", "MS", "SHEET") and g["net"]:
            groups["pec_surfaces"][name] = record
        if g["interface_type"] == "PORT":
            groups["port_surfaces"][name] = record
    return groups


def test_palace_can_rebuild_its_group_dict_from_the_sidecar(sidecars):
    """Palace can rebuild its group dict from the sidecar."""
    groups = palace_style_groups(sidecars["pad_b"])
    assert set(groups["volumes"]) == {"VOL::substrate", "VOL::vacuum"}
    assert {
        "MA::pad::top",
        "MS::pad::bottom",
        "MA::ground::top",
        "MS::ground::bottom",
    } <= set(groups["pec_surfaces"])
    for name, record in groups["pec_surfaces"].items():
        kinds = {d for d in record["adjacent_solution_volume_ids"]}
        assert record["owner_semantic_ids"] and record["net_id"] and kinds, name
    (port,) = groups["port_surfaces"].values()
    assert port["owner_semantic_ids"] == ["p1"]
    terminals = {g["net_id"] for g in groups["pec_surfaces"].values()}
    assert terminals == {"pad", "Ground"}  # Electrostatic terminals: one per Net


def test_aedt_can_rebuild_its_contribution_ledger_from_the_sidecar(sidecars):
    """Per face: owner, side classification MA/MS/SA/MM, orientation, the two volumes -- no geometry needed."""
    side = sidecars["pad_b"]
    ledger = []
    for fid, face in side["faces"].items():
        group = side["physical_groups"].get(face["group"]) if face["group"] else None
        if group and group["interface_type"] in ("MA", "MS", "SA", "MM"):
            ledger.append(
                (
                    fid,
                    face["group"],
                    group["interface_type"],
                    group["entities"],
                    face["face_kind"],
                )
            )
    assert {kind for _, _, kind, _, _ in ledger} == {"MA", "MS", "SA"}
    assert all(owners for _, _, _, owners, _ in ledger)
    assert {k for _, g, kind, _, k in ledger if g == "MS::pad::bottom"} == {"bottom"}


# --------------------------------------------------------------------------- AEDT names


def test_aedt_names_keep_only_the_characters_aedt_keeps():
    """Aedt names keep only the characters aedt keeps."""
    assert sanitise_aedt_name("VOL::chip/air") == "VOL__chip_air"
    assert sanitise_aedt_name("MA::pad::top::band=2") == "MA__pad__top__band_2"
    assert sanitise_aedt_name("SA::substrate|vacuum") == "SA__substrate_vacuum"
    assert sanitise_aedt_name("unié") == "uni_"
    assert sanitise_aedt_name("") == "_"


def test_colliding_aedt_names_are_told_apart_deterministically_in_tag_order():
    """Colliding aedt names are told apart deterministically in tag order."""
    names = aedt_names(
        [("v1", "VOL::a/b"), ("v2", "VOL::a_b"), ("v3", "VOL::a.b"), ("v4", "VOL::c")]
    )
    assert names == {
        "v1": "VOL__a_b",
        "v2": "VOL__a_b__2",
        "v3": "VOL__a_b__3",
        "v4": "VOL__c",
    }
    assert len(set(names.values())) == 4


# --------------------------------------------------------------------------- bytes


def test_the_sidecar_file_has_one_top_level_key_per_line_and_reads_back_equal(
    tmp_path, sidecars
):
    """The sidecar file has one top level key per line and reads back equal."""
    path = tmp_path / "s.json"
    size = write_sidecar(path, sidecars["pad_b"])
    assert path.stat().st_size == size
    lines = path.read_text().splitlines()
    assert (
        lines[0] == "{"
        and lines[-1] == "}"
        and len(lines) == len(sidecars["pad_b"]) + 2
    )
    assert read_sidecar(path) == json.loads(json.dumps(sidecars["pad_b"]))
    assert b"\r" not in path.read_bytes() and str(tmp_path) not in path.read_text()


SCRIPT = (
    f"import sys; sys.path.insert(0, {str(TESTS)!r})\n"
    "import hashlib\n"
    "from sgb_emit_cases import pad_model\n"
    "from scgsim.geometry.sgb.emission.plan import plan_brep\n"
    "from scgsim.geometry.sgb.sidecar.build import SidecarContext, build_sidecar\n"
    "from scgsim.geometry.sgb.sidecar.files import write_sidecar\n"
    "from scgsim.geometry.sgb.slivers.report import sliver_report\n"
    "import tempfile, pathlib\n"
    "digests = []\n"
    "for route in 'ABC':\n"
    "    model = pad_model(route, bands=True)\n"
    "    path = pathlib.Path(tempfile.mkdtemp()) / 's.json'\n"
    "    write_sidecar(path, build_sidecar(model, plan_brep(model.topology, model.routed), SidecarContext(sliver_report(model)['summary'])))\n"
    "    digests.append(hashlib.sha256(path.read_bytes()).hexdigest())\n"
    "print(*digests)\n"
)


def test_the_sidecar_bytes_do_not_depend_on_python_hash_randomisation():
    """Three hash seeds, one digest per route."""
    outputs = set()
    for hash_seed in ("0", "1", "424242"):
        result = subprocess.run(
            [sys.executable, "-c", SCRIPT],
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
            capture_output=True,
            text=True,
            check=True,
        )
        outputs.add(result.stdout.strip())
    assert len(outputs) == 1


def test_the_sidecar_is_the_same_for_every_tile_layout(tmp_path):
    """Tiles and workers change how the model is derived, never what it is."""
    one = pad_model("B", bands=True)
    from sgb_emit_cases import pad_requests  # noqa: PLC0415 -- used by this test only
    from sgb_scenes import pad_in_ground  # noqa: PLC0415 -- used by this test only

    from scgsim.geometry.sgb.model import build_model_from_catalogue  # noqa: PLC0415 -- used by this test only

    tiled = build_model_from_catalogue(
        pad_in_ground("B"), "B", insets=pad_requests("B"), tiles=3, workers=2
    )
    assert json.dumps(sidecar_of(one), sort_keys=True) == json.dumps(
        sidecar_of(tiled), sort_keys=True
    )
