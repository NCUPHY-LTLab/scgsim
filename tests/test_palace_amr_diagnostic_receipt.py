"""File-only receipt binding for opt-in diagnostic manifests, not solver evidence."""

import json
from pathlib import Path

import pytest

from scgsim.palace.execution.returned_receipt import (
    amr_diagnostic_output_paths,
    record_for_path,
)
from scgsim.palace.results import receipt_contract


def _config(enabled=True):
    return {"Model": {"Refinement": {"SaveAdaptDiagnostics": enabled}}}


def _manifest(root: Path, files: list[str]):
    base = root / "results/palace/amr-diagnostics"
    base.mkdir(parents=True)
    (base / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "palace.amr_diagnostics",
                "record_type": "manifest",
                "schema_version": 1,
                "files": files,
            }
        )
    )
    return base


def test_opted_out_diagnostics_do_not_require_a_manifest(tmp_path):
    """Historical and explicitly disabled configurations preserve their inventory."""
    assert amr_diagnostic_output_paths(tmp_path, {}) == []
    assert amr_diagnostic_output_paths(tmp_path, _config(False)) == []


def test_missing_opted_in_manifest_is_a_required_missing_output(tmp_path):
    """An opted-in run cannot silently report success without native diagnostics."""
    paths = amr_diagnostic_output_paths(tmp_path, _config())
    assert paths == ["results/palace/amr-diagnostics/manifest.json"]
    assert record_for_path(tmp_path, paths[0])["present"] is False


def test_manifest_inventory_hashes_every_listed_file_and_keeps_missing(tmp_path):
    """Use the explicit inventory, including mesh bytes and interrupted missing files."""
    files = ["solve.json", "rank.mesh", "started-missing.json"]
    base = _manifest(tmp_path, files)
    (base / "solve.json").write_text("{}")
    (base / "rank.mesh").write_bytes(b"actual unit-fixture mesh bytes\n")
    (base / "unlisted.json").write_text("{}")
    paths = amr_diagnostic_output_paths(tmp_path, _config())
    assert paths == [
        "results/palace/amr-diagnostics/manifest.json",
        *[f"results/palace/amr-diagnostics/{name}" for name in files],
    ]
    records = [record_for_path(tmp_path, path) for path in paths]
    assert [record["present"] for record in records] == [True, True, True, False]
    first = records[1]["sha256"]
    (base / "solve.json").write_text('{"changed": true}')
    assert record_for_path(tmp_path, paths[1])["sha256"] != first


@pytest.mark.parametrize(
    "files",
    [
        ["../outside.json"],
        ["/tmp/outside.json"],
        ["manifest.json"],
        ["."],
        ["same.json", "same.json"],
    ],
)
def test_manifest_rejects_escaped_or_ambiguous_inventory(tmp_path, files):
    """Native-provided paths cannot escape their sidecar or alias its inventory."""
    _manifest(tmp_path, files)
    with pytest.raises(ValueError):
        amr_diagnostic_output_paths(tmp_path, _config())


def test_diagnostic_symlink_cannot_escape_sidecar(tmp_path):
    """Confine resolved diagnostic bytes as well as their relative path strings."""
    base = _manifest(tmp_path, ["link.json"])
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    (base / "link.json").symlink_to(outside)
    with pytest.raises(ValueError):
        amr_diagnostic_output_paths(tmp_path, _config())


def test_strict_reader_requires_all_manifest_records_in_receipt(tmp_path):
    """Omitting a listed native record from an otherwise hashed receipt is rejected."""
    base = _manifest(tmp_path, ["solve.json"])
    (base / "solve.json").write_text("{}")
    (tmp_path / "config.json").write_text(json.dumps(_config()))
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (metadata / "palace_index_map.json").write_text(
        json.dumps(
            {
                "entries": [
                    {"section": "Boundaries.Terminal", "index": 1},
                ]
            }
        )
    )
    names = [
        "terminal-C",
        "terminal-Cm",
        "terminal-Cinv",
        "terminal-V",
        "domain-E",
        "error-indicators",
    ]
    outputs = []
    for name in [*(f"{name}.csv" for name in names), "palace.json"]:
        path = tmp_path / "results/palace" / name
        path.write_text("receipt-unit bytes only\n")
        outputs.append(record_for_path(tmp_path, str(path.relative_to(tmp_path))))
    diagnostic_paths = amr_diagnostic_output_paths(tmp_path, _config())
    outputs.append(record_for_path(tmp_path, diagnostic_paths[0]))
    with pytest.raises(ValueError, match="misses required output entry.*solve.json"):
        receipt_contract._validate_returned_outputs(  # noqa: SLF001 -- isolate the receipt output-hash contract
            {}, "Electrostatic", {"output_files": outputs}, tmp_path
        )
    outputs.append(record_for_path(tmp_path, diagnostic_paths[1]))
    receipt_contract._validate_returned_outputs(  # noqa: SLF001 -- isolate the receipt output-hash contract
        {}, "Electrostatic", {"output_files": outputs}, tmp_path
    )
    (base / "solve.json").write_text('{"changed": true}')
    with pytest.raises(ValueError, match="hash mismatch"):
        receipt_contract._validate_returned_outputs(  # noqa: SLF001 -- isolate the receipt output-hash contract
            {}, "Electrostatic", {"output_files": outputs}, tmp_path
        )
