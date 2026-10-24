"""File-only Palace0.18.1 output contracts; no native mesh/solver validation."""
import hashlib
import json

import pytest

from scgsim.palace.execution.returned_receipt import _adapted_mesh_output_paths, _record_for_path
from scgsim.palace.preparation.transforms import SCHEMA_VERSION
from scgsim.palace.results.resolve import _validate_palace_identity, _validate_runtime_compatibility


def config(enabled=True):
    return {"Model": {"Mesh": "palace.msh", "Refinement": {"SaveAdaptMesh": enabled}}}


def metadata(directory, *, adapted=True):
    directory.mkdir(parents=True, exist_ok=True)
    # Only exercise native record shape/identity, not inferred mesh counts.
    payload = {"SavedAdaptedMesh": {"Dimension": 3}} if adapted else {}
    (directory / "palace.json").write_text(json.dumps(payload))


def test_new_schema_label():
    assert SCHEMA_VERSION == "1-7-0"


def test_disabled_save_has_no_fabricated_artifact_requirement(tmp_path):
    metadata(tmp_path / "results/palace")
    assert _adapted_mesh_output_paths(tmp_path, config(False)) == []


def test_no_native_adaptation_record_has_no_fabricated_output(tmp_path):
    metadata(tmp_path / "results/palace", adapted=False)
    assert _adapted_mesh_output_paths(tmp_path, config()) == []


@pytest.mark.parametrize("suffix", [".mesh", ".meshgz"])
def test_actual_variant_is_hash_bound_and_original_input_preserved(tmp_path, suffix):
    original = tmp_path / "palace.msh"
    original.write_bytes(b"original sealed mesh bytes")
    before = hashlib.sha256(original.read_bytes()).hexdigest()
    directory = tmp_path / "results/palace"
    metadata(directory)
    saved = directory / ("palace" + suffix)
    saved.write_bytes(b"export fixture bytes: file inventory only")
    paths = _adapted_mesh_output_paths(tmp_path, config())
    assert paths == ["results/palace/palace.json", "results/palace/palace" + suffix]
    record = _record_for_path(tmp_path, paths[1])
    assert record["sha256"] == hashlib.sha256(saved.read_bytes()).hexdigest()
    assert record["bytes"] == len(saved.read_bytes())
    assert record["present"] is True
    assert hashlib.sha256(original.read_bytes()).hexdigest() == before
    assert "palace.msh" not in paths


def test_native_declared_missing_output_retains_both_compression_alternatives(tmp_path):
    metadata(tmp_path / "results/palace")
    paths = _adapted_mesh_output_paths(tmp_path, config())
    assert paths == ["results/palace/palace.json", "results/palace/palace.meshgz", "results/palace/palace.mesh"]
    records = [_record_for_path(tmp_path, path) for path in paths]
    assert [record["present"] for record in records] == [True, False, False]
    assert not all(record["present"] for record in records)


def test_iteration_metadata_and_confined_top_level_links(tmp_path):
    results = tmp_path / "results/palace"
    metadata(results)
    metadata(results / "iteration01", adapted=False)
    metadata(results / "iteration02")
    saved = results / "iteration02/palace.meshgz"
    saved.write_bytes(b"per-iteration mesh fixture bytes")
    (results / "palace.meshgz").symlink_to("iteration02/palace.meshgz")
    paths = _adapted_mesh_output_paths(tmp_path, config())
    assert paths == ["results/palace/palace.json", "results/palace/palace.meshgz",
                     "results/palace/iteration02/palace.json", "results/palace/iteration02/palace.meshgz"]
    assert _record_for_path(tmp_path, paths[1])["sha256"] == _record_for_path(tmp_path, paths[3])["sha256"]


@pytest.mark.parametrize("kind", ["mesh", "metadata"])
def test_adapted_artifact_links_cannot_escape_run(tmp_path, kind):
    root = tmp_path / "run"
    results = root / "results/palace"
    metadata(results)
    outside = tmp_path / "outside"
    outside.write_text("not part of this run")
    if kind == "mesh":
        (results / "palace.meshgz").symlink_to(outside)
    else:
        (results / "palace.json").unlink()
        (results / "palace.json").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes the run"):
        _adapted_mesh_output_paths(root, config())


def test_two_compression_variants_do_not_silently_choose_one(tmp_path):
    results = tmp_path / "results/palace"
    metadata(results)
    (results / "palace.mesh").write_bytes(b"one")
    (results / "palace.meshgz").write_bytes(b"another")
    with pytest.raises(ValueError, match="ambiguous compression variants"):
        _adapted_mesh_output_paths(tmp_path, config())


@pytest.mark.parametrize("label", ["v0.16.0", "1-7-0"])
@pytest.mark.parametrize("observed_schema", [None, "1-7-0"])
def test_historical_and_new_labels_remain_bound_without_runtime_version_gate(tmp_path, label, observed_schema):
    path = tmp_path / "config.json"
    path.write_text("{}")
    identity = {"config_schema": label, "config_compatibility_probe": "native_dry_run",
                "config_compatibility_semantics": "required_before_solve",
                "runtime_identity_semantics": "observed_provenance_only"}
    _validate_palace_identity(identity)
    # Authored validation fixture, not a claimed native probe result.
    observed = {"native_git_tag": None, "version_probe_exit_code": None,
                "runtime_binary_identity": None, "runtime_binary_sha256": None}
    if observed_schema is not None:
        observed["native_schema_version"] = observed_schema
    payload = {"schema": "palace-runtime-compatibility.v1",
               "config": {"path": "config.json", "bytes": path.stat().st_size,
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "declared_schema": label},
               "compatibility": {"method": "native_dry_run", "probe_exit_code": 0, "verified": True},
               "observed_identity": observed,
               "command": {"command_style": "wrapper", "executable_identity": "palace"}}
    _validate_runtime_compatibility(payload, identity, root=tmp_path)


@pytest.mark.parametrize("kind", ["results", "iteration"])
def test_output_directories_cannot_escape_run(tmp_path, kind):
    root = tmp_path / "run"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    if kind == "results":
        (root / "results").mkdir()
        (root / "results/palace").symlink_to(outside, target_is_directory=True)
    else:
        results = root / "results/palace"
        metadata(results, adapted=False)
        (results / "iteration02").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes the run"):
        _adapted_mesh_output_paths(root, config())
