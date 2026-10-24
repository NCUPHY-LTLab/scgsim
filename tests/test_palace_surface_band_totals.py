"""Source producer/table transport and detached exact aggregation, without native execution."""

from copy import deepcopy

import pytest

from scgsim.palace.results import (
    PalaceSurfaceBandTotals,
    SurfaceBandSelection,
    aggregate_surface_bands,
)
from scgsim.palace.results.epr import PalaceEprResult, epr_result
from test_palace_surface_bands import _resolved_producer_band_run, _surface_info
from scgsim.palace.preparation.surface_epr import build_surface_epr_postprocessing


def _case(interface="MA", *, small=False):
    resolved, _, _, summaries, _, entries, energies = _resolved_producer_band_run(
        interface_kind=interface,
        constructed_sa=interface == "SA",
        small=small,
        wall=not small,
    )
    result = epr_result(resolved)
    selections = tuple(
        SurfaceBandSelection(
            parent,
            interface,
            next(
                item["metadata"]["face_kind"]
                for item in entries
                if item["metadata"]["surface_distance_band"]["parent_surface_id"]
                == parent
            ),
        )
        for parent in dict.fromkeys(item["parent_surface_id"] for item in summaries)
    )
    return result, selections, entries, energies


def _aggregate(result, selections, **kwargs):
    options = {
        "mode": 1,
        "palace_iteration": 2,
        "cutoffs_um": (0.5, 1.0),
        "selections": selections,
    }
    options.update(kwargs)
    return aggregate_surface_bands(result, **options)


@pytest.mark.parametrize("interface", ("MA", "MS", "SA"))
def test_exact_totals_preserve_all_source_components_and_film_assumptions(interface):
    result, selections, entries, energies = _case(interface)
    totals = _aggregate(result, selections)
    for total in totals.totals:
        expected = [
            entry
            for entry in entries
            if entry["metadata"]["surface_distance_band"]["lower_um"]
            >= total["cutoff_um"]
        ]
        expected_ids = {entry["metadata"]["surface_id"] for entry in expected}
        expected_energy = sum(
            energies[name] for name in energies if name in expected_ids
        )
        assert total["energy_j"] == expected_energy
        assert total["participation"] == expected_energy / 16
        assert total["inverse_q"] == expected_energy / 16 * 0.125
        assert {member["surface_id"] for member in total["members"]} == expected_ids
        assert len(total["members"]) == len(expected_ids)
        for member in total["members"]:
            original = next(
                item
                for item in result.rows[0]["surface"]
                if item["surface_id"] == member["surface_id"]
            )
            assert member["members"] == original["members"]
            assert member["assumptions"] == original["assumptions"]
            assert member["field_side"] == original["field_side"]
            assert member["original_energy_j"] == energies[member["surface_id"]]
    assert PalaceSurfaceBandTotals.from_payload(totals.to_payload()) == totals
    assert totals.provenance["native_mask_equivalence"] is False


@pytest.mark.parametrize(
    "change,match",
    (
        ("mode", "mode"),
        ("pass", "iteration"),
        ("normalization", "normalization"),
        ("missing_band", "inventory"),
        ("missing_component", "components"),
        ("duplicate", "duplicated"),
        ("side", "side"),
        ("attribute", "attribute"),
    ),
)
def test_aggregation_rejects_incomplete_or_mismatched_source_selection(change, match):
    result, selections, _, _ = _case()
    payload = result.to_payload()
    options = {}
    if change == "mode":
        options["mode"] = 2
    elif change == "pass":
        options["palace_iteration"] = 3
    elif change == "normalization":
        options["expected_normalization_energy_j"] = 8
    elif change == "missing_band":
        payload["provenance"]["surface_distance_bands"].pop()
    elif change == "missing_component":
        payload["rows"][0]["surface"].pop()
    elif change == "duplicate":
        payload["rows"][0]["surface"].append(deepcopy(payload["rows"][0]["surface"][0]))
    elif change == "attribute":
        payload["rows"][0]["surface"][1]["attributes"] = payload["rows"][0]["surface"][
            0
        ]["attributes"]
    elif change == "side":
        selections = (
            SurfaceBandSelection(selections[0].parent_surface_id, "MA", "other"),
            *selections[1:],
        )
    with pytest.raises(ValueError, match=match):
        _aggregate(PalaceEprResult.from_payload(payload), selections, **options)


def test_declared_empty_bands_are_zero_but_not_missing_inventory():
    result, selections, _, _ = _case(small=True)
    total = _aggregate(result, selections).totals[-1]
    assert total["members"] == ()
    assert total["participation"] == 0
    assert total["bands"]


def test_new_producer_rejects_legacy_inset_request_with_migration():
    with pytest.raises(ValueError, match="source geometric.*aggregate_surface_bands"):
        build_surface_epr_postprocessing(
            {"boundary_surfaces": {"pad": _surface_info("pad", 1)}},
            {"MA": {"thickness": 0.5, "permittivity": 4, "inset_margins_um": [0.5]}},
        )


def _iteration_run(tmp_path, *, native_iteration=1):
    import csv
    import hashlib
    import json
    from dataclasses import replace
    from test_palace_surface_bands import _resolved_band_run

    original = _resolved_band_run()
    directory = tmp_path / "results/palace/iteration01"
    directory.mkdir(parents=True)
    domain_headers = [
        "m",
        "E_elec (J)",
        "E_mag (J)",
        "E_cap (J)",
        "E_ind (J)",
        "E_elec[1] (J)",
        "p_elec[1]",
        "E_mag[1] (J)",
        "p_mag[1]",
    ]
    surface_headers = ["m"]
    surface_values = [1]
    for index in range(1, 5):
        surface_headers.extend((f"p_surf[{index}]", f"Q_surf[{index}]"))
        surface_values.extend((0.125, 800))
    tables = {
        "eig": (
            ["m", "Re{f} (GHz)", "Im{f} (GHz)", "Q", "Error (Bkwd.)", "Error (Abs.)"],
            [1, 4, 0, 1, 0, 0],
        ),
        "domain-E": (domain_headers, [1, 2, 0, 2, 0, 2, 1, 0, 0]),
        "surface-Q": (surface_headers, surface_values),
        "error-indicators": (["Norm", "Minimum", "Maximum", "Mean"], [1, 0, 1, 0.5]),
    }
    for name, (headers, values) in tables.items():
        with (directory / f"{name}.csv").open("w") as handle:
            writer = csv.writer(handle)
            writer.writerows((headers, values))
    (directory / "palace.json").write_text(
        json.dumps({"Problem": {"Iteration": native_iteration}})
    )
    outputs = tuple(
        {
            "path": path.relative_to(tmp_path).as_posix(),
            "present": True,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in directory.iterdir()
    )
    index = deepcopy(original.provenance.index_map)
    index["entries"].append(
        {
            "section": "Domains.Postprocessing.Energy",
            "index": 1,
            "entry_name": "domain",
            "metadata": {"material": "Si"},
        }
    )
    final_tables = dict(original.tables)
    final_tables["domain-E"] = replace(
        final_tables["domain-E"],
        rows=({**final_tables["domain-E"].rows[0], "p_elec[1]": 1},),
    )
    return replace(
        original,
        run_dir=tmp_path,
        tables=final_tables,
        returned_receipt=replace(original.returned_receipt, output_files=outputs),
        provenance=replace(
            original.provenance,
            index_map=index,
            config={"Model": {"L0": 1e-6}, "Solver": {"Eigenmode": {"N": 1}}},
        ),
    )


def test_explicit_iteration_uses_bound_tables_and_native_iteration(tmp_path):
    resolved = _iteration_run(tmp_path)
    result = epr_result(resolved, palace_iteration=1)
    assert result.rows[0]["frequency_hz"] == 4e9
    assert result.rows[0]["normalization_energy_j"] == 4
    assert result.rows[0]["surface"][0]["participation"] == 0.125
    assert result.provenance["palace_native_iteration"] == 1
    outputs = result.provenance["source_reference"]["output_files"]
    assert outputs and all("/iteration01/" in row["path"] for row in outputs)
    assert epr_result(resolved).rows[0]["frequency_hz"] == 5e9


@pytest.mark.parametrize("defect", ("hash", "identity", "missing"))
def test_explicit_iteration_refuses_changed_missing_or_relabelled_data(
    tmp_path, defect
):
    resolved = _iteration_run(
        tmp_path, native_iteration=2 if defect == "identity" else 1
    )
    if defect == "hash":
        (tmp_path / "results/palace/iteration01/eig.csv").write_text("changed")
    elif defect == "missing":
        (tmp_path / "results/palace/iteration01/domain-E.csv").unlink()
    with pytest.raises(ValueError, match="artifact|metadata"):
        epr_result(resolved, palace_iteration=1)


def _shared_parent_case():
    """Generic table-reader fixture: two semantic rows per physical fragment.

    These are authored offline inputs, not completed native-run evidence.
    MS uses twice the raw participation of MA with the same normalization.
    """
    from dataclasses import replace

    resolved, _, _, _, _, entries, energies = _resolved_producer_band_run(wall=False)
    shared_entries = []
    surface_row = {"m": 1}
    for interface, scale in (("MA", 1), ("MS", 2)):
        for entry in entries:
            item = deepcopy(entry)
            item["index"] = len(shared_entries) + 1
            item["entry_name"] += "__" + interface
            item["metadata"]["interface_type"] = interface
            item["metadata"]["face_kind"] = "interface"
            shared_entries.append(item)
            energy = energies[item["metadata"]["surface_id"]] * scale
            number = item["index"]
            surface_row[f"p_surf[{number}]"] = energy / 16
            surface_row[f"Q_surf[{number}]"] = 1 / (energy / 16 * 0.125)
    index = deepcopy(resolved.provenance.index_map)
    index["entries"] = shared_entries
    tables = dict(resolved.tables)
    tables["surface-Q"] = replace(
        tables["surface-Q"], headers=tuple(surface_row), rows=(surface_row,)
    )
    result = epr_result(replace(
        resolved, tables=tables,
        provenance=replace(resolved.provenance, index_map=index),
    ))
    parent = result.provenance["surface_distance_bands"][0]["parent_surface_id"]
    selections = tuple(
        SurfaceBandSelection(parent, kind, "interface") for kind in ("MS", "MA")
    )
    return result, selections, entries, energies


def test_shared_parent_ms_ma_totals_include_each_fragment_once_per_interface():
    result, selections, entries, energies = _shared_parent_case()
    totals = _aggregate(result, selections)
    assert len(totals.totals) == 4
    for total in totals.totals:
        included = {
            entry["metadata"]["surface_id"]
            for entry in entries
            if entry["metadata"]["surface_distance_band"]["lower_um"]
            >= total["cutoff_um"]
        }
        # The table fixture declares MS energy=2*MA and one mode energy=16 J.
        scale = 2 if total["interface_kind"] == "MS" else 1
        energy = sum(energies[identity] for identity in included) * scale
        assert total["energy_j"] == energy
        assert total["participation"] == energy / 16
        assert total["inverse_q"] == energy / 16 * 0.125
        assert {member["surface_id"] for member in total["members"]} == included
        assert len(total["members"]) == len(included)
        assert {member["field_side"] for member in total["members"]} == {"interface"}
    assert PalaceSurfaceBandTotals.from_payload(totals.to_payload()) == totals
    # Selecting either interface from the unchanged full result is also valid.
    for selection in selections:
        single = _aggregate(result, (selection,))
        assert single.totals == tuple(
            total for total in totals.totals
            if total["interface_kind"] == selection.interface_kind
        )


@pytest.mark.parametrize(
    "defect,match",
    (
        ("selection", "selection is duplicated"),
        ("fragment", "fragment is duplicated"),
        ("attribute", "attribute is duplicated"),
        ("side", "field side"),
        ("absent_kind", "interface kind is unavailable"),
        ("component", "components"),
        ("source", "source/interval"),
        ("owner", "owner identity"),
        ("component_index", "component identities"),
        ("mode", "mode"),
        ("iteration", "iteration"),
        ("normalization", "normalization"),
    ),
)
def test_shared_parent_preserves_selection_and_member_failures(defect, match):
    result, selections, _, _ = _shared_parent_case()
    payload = result.to_payload()
    surfaces = payload["rows"][0]["surface"]
    ms = [surface for surface in surfaces if surface["interface_kind"] == "MS"]
    options = {}
    if defect == "selection":
        selections = (*selections, selections[0])
    elif defect == "fragment":
        surfaces.append(deepcopy(ms[0]))
    elif defect == "attribute":
        ms[1]["attributes"] = deepcopy(ms[0]["attributes"])
    elif defect == "side":
        selections = (
            SurfaceBandSelection(selections[0].parent_surface_id, "MS", "other"),
            selections[1],
        )
    elif defect == "absent_kind":
        payload["rows"][0]["surface"] = [
            surface for surface in surfaces if surface["interface_kind"] != "MS"
        ]
    elif defect == "component":
        surfaces.remove(ms[-1])
    elif defect == "source":
        ms[0]["surface_distance_band"]["source_polygon_id"] = "other_public_polygon"
    elif defect == "owner":
        ms[0]["owner_semantic_ids"] = ["other_public_owner"]
    elif defect == "component_index":
        band = ms[0]["surface_distance_band"]
        summary = next(
            item for item in payload["provenance"]["surface_distance_bands"]
            if item["parent_surface_id"] == band["parent_surface_id"]
            and item["band_index"] == band["band_index"]
        )
        band["component_index"] = summary["component_count"]
    elif defect == "mode":
        options["mode"] = 2
    elif defect == "iteration":
        options["palace_iteration"] = 3
    elif defect == "normalization":
        options["expected_normalization_energy_j"] = 8
    with pytest.raises(ValueError, match=match):
        _aggregate(PalaceEprResult.from_payload(payload), selections, **options)


def test_actual_ms_ma_producer_aliases_reach_shared_parent_totals():
    """Real alias/config/reader functions over generic authored transport records.

    Attribute tokens and mode tables are explicit fixture inputs, not native
    mesh/solver observations. Physical band children come from the existing
    public source-partition fixture; no geometry is split per interface.
    """
    from dataclasses import replace

    from scgsim.palace.mesh.build import _add_surface_group_records
    from scgsim.palace.preparation.surface_epr import (
        build_surface_epr_postprocessing, normalize_surface_epr_specs,
    )

    resolved, _, _, _, _, original_entries, energies = _resolved_producer_band_run(wall=False)
    groups = {"boundary_surfaces": {}, "pec_surfaces": {}, "surface_groups": {}}
    for entry in original_entries:
        metadata = entry["metadata"]
        surface_id = metadata["surface_id"]
        provenance = {
            **deepcopy(metadata["source_provenance"]),
            "conductor_source_layer_name": "generic_metal_layer",
            "surface_contribution_schema": "scgsim.surface-contributions.v1",
            "surface_contribution_ledger": [
                {"contribution_id": f"{surface_id}:MS", "classification": "MS",
                 "side": "bottom", "snapshot_reference": {"source_entity_id": "entity_1"},
                 "effective_domain_ids": ["substrate"]},
                {"contribution_id": f"{surface_id}:MA", "classification": "MA",
                 "side": "top", "snapshot_reference": {"source_entity_id": "entity_1"},
                 "effective_domain_ids": ["vacuum"]},
            ],
            "native_aggregation": {"policy": "one_row_per_interface_kind",
                "contribution_count": 2, "interface_kinds": ["MS", "MA"],
                "surface_epr_interface_kinds": ["MS", "MA"], "native_row_count": 2,
                "side_resolved_numeric_output": False},
            "mask_support": {"zero_inset_supported": True, "positive_inset_supported": True,
                             "sides": ["bottom", "top"]},
        }
        record = {"surface_id": surface_id, "interface_type": "MS_MA",
            "face_kind": "interface", "representation": "surface_sheet", "route": "A",
            "owner_semantic_ids": metadata["owner_semantic_ids"],
            "adjacent_solution_volume_ids": ["substrate", "vacuum"],
            "conductor_component_id": "generic_component", "net_id": "generic_net",
            "equipotential_id": "generic_net", "contact_kind": "interface",
            "physical_attribute": metadata["physical_attribute"],
            "source_provenance": provenance,
            "metadata": {"surface_distance_band": metadata["surface_distance_band"]}}
        attribute = entry["attributes"][0]
        _add_surface_group_records(groups=groups, name=surface_id, phys_group=attribute,
            entity_tags=(attribute,), record=record, route="A",
            solution_material_kinds={"substrate": "dielectric", "vacuum": "vacuum"})
        for kind in ("MS", "MA"):
            alias = groups["boundary_surfaces"][f"{surface_id}__{kind}"]
            assert alias["surface_id"] == surface_id
            assert alias["phys_group"] == attribute
            assert alias["face_kind"] == "interface"
            assert alias["source_provenance"] == provenance

    specs = normalize_surface_epr_specs({kind: {"thickness": .5, "permittivity": 4,
        "loss_tangent": .125} for kind in ("MS", "MA")}, materials={})
    native_rows, entries = build_surface_epr_postprocessing(groups, specs)
    assert len(native_rows) == 2 * len(original_entries)
    assert all("Mask" not in native for native in native_rows)
    surface_row = {"m": 1}
    for entry in entries:
        kind = entry["metadata"]["interface_type"]
        energy = energies[entry["metadata"]["surface_id"]] * (2 if kind == "MS" else 1)
        surface_row[f"p_surf[{entry['index']}]"] = energy / 16
        surface_row[f"Q_surf[{entry['index']}]"] = 1 / (energy / 16 * .125)
    index = deepcopy(resolved.provenance.index_map)
    index["entries"] = entries
    tables = dict(resolved.tables)
    tables["surface-Q"] = replace(tables["surface-Q"], headers=tuple(surface_row), rows=(surface_row,))
    result = epr_result(replace(resolved, tables=tables,
        provenance=replace(resolved.provenance, index_map=index)))
    parent = index["surface_distance_bands"][0]["parent_surface_id"]
    selections = tuple(SurfaceBandSelection(parent, kind, "interface") for kind in ("MS", "MA"))
    totals = _aggregate(result, selections)
    for total in totals.totals:
        included = {entry["metadata"]["surface_id"] for entry in original_entries
                    if entry["metadata"]["surface_distance_band"]["lower_um"] >= total["cutoff_um"]}
        expected_energy = sum(energy for name, energy in energies.items() if name in included)
        if total["interface_kind"] == "MS":
            expected_energy *= 2
        assert total["energy_j"] == expected_energy
        assert total["participation"] == expected_energy / 16
        assert len(total["members"]) == len(included)
        assert {member["surface_id"] for member in total["members"]} == included
        assert all(member["members"]["surface_contribution_schema"] == "scgsim.surface-contributions.v1"
                   for member in total["members"])
