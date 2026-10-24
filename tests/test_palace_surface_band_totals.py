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
