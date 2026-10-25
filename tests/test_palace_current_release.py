"""Preflight control tests using synthetic subprocess output, never native Palace."""

import json
import subprocess

import pytest

from scgsim.palace.execution import returned_receipt


OFFICIAL_COMMIT = "0dc74cdf8c36c58b69b21c4a06e816048ec0b83f"


def _preflight(monkeypatch, tmp_path, *, tag="0.18.1", version_rc=0,
               dry_run_rc=0, mutate_config=False, config_present=True):
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "config.json"
    if config_present:
        config.write_text('{"Problem":{"Type":"Eigenmode"}}\n')
    commands = []

    def synthetic_subprocess(command, **kwargs):
        commands.append(command)
        assert kwargs == {
            "check": False, "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT, "text": True,
        }
        if command == ["palace", "-serial", "--version"]:
            return subprocess.CompletedProcess(
                command, version_rc,
                f"Palace version: {tag}\nSchema version: 1-7-0\n",
            )
        assert command == ["palace", "-serial", "--dry-run", "config.json"]
        if mutate_config:
            config.write_text('{"changed":true}\n')
        return subprocess.CompletedProcess(command, dry_run_rc, "synthetic dry-run output")

    monkeypatch.setattr(returned_receipt.subprocess, "run", synthetic_subprocess)
    rc = returned_receipt._verify_config_main([
        "palace", "wrapper", "metadata/compatibility.json", "config.json",
        "--config-schema", "1-7-0",
    ])
    payload = json.loads((tmp_path / "metadata/compatibility.json").read_text())
    assert commands == [
        ["palace", "-serial", "--version"],
        ["palace", "-serial", "--dry-run", "config.json"],
    ]
    return rc, payload


@pytest.mark.parametrize("tag", ("0.18.1", "v0.18.1", OFFICIAL_COMMIT[:7], OFFICIAL_COMMIT))
def test_supported_upstream_identity_requires_and_records_native_dry_run(monkeypatch, tmp_path, tag):
    rc, payload = _preflight(monkeypatch, tmp_path, tag=tag)
    assert rc == 0
    assert payload["compatibility"] == {
        "method": "native_dry_run", "probe_exit_code": 0, "verified": True,
    }
    identity = payload["observed_identity"]
    assert identity["native_git_tag"] == tag
    assert identity["supported_upstream_release"] == "0.18.1"
    assert identity["version_probe_exit_code"] == 0
    assert payload["config"]["sha256"]


@pytest.mark.parametrize("tag", (
    "0.16.0", "0.18.1-dirty", "0dc74cd-dirty", "0dc74cd-1-gabcdef0",
    "abcdef0", OFFICIAL_COMMIT[:6], OFFICIAL_COMMIT + "0",
))
def test_explicit_native_binary_is_checked_by_exact_config_not_upstream_stamp(monkeypatch, tmp_path, tag):
    rc, payload = _preflight(monkeypatch, tmp_path, tag=tag)
    assert rc == 0
    assert payload["compatibility"]["verified"] is True
    assert payload["observed_identity"]["native_git_tag"] == tag
    assert payload["observed_identity"]["supported_upstream_release"] is None


@pytest.mark.parametrize("failure", (
    {"dry_run_rc": 1},
    {"mutate_config": True}, {"config_present": False},
))
def test_supported_label_cannot_bypass_probe_dry_run_or_config_identity(monkeypatch, tmp_path, failure):
    rc, payload = _preflight(monkeypatch, tmp_path, **failure)
    assert rc == 78
    assert payload["compatibility"]["verified"] is False


def test_new_handoff_uses_explicit_native_compatibility_and_records_identity(tmp_path):
    # Opaque files test the real file-only producer; no mesh or native result is claimed.
    from scgsim.palace.execution.handoff import prepare_handoff
    from scgsim.palace.mesh.build import MeshBuildResult
    from scgsim.palace.results.resolve import _validate_palace_identity

    run_dir = tmp_path / "prepared"
    metadata = run_dir / "metadata"
    metadata.mkdir(parents=True)
    mesh = run_dir / "palace.msh"
    mesh.write_bytes(b"opaque file-only transport fixture\n")
    config = run_dir / "config.json"
    config.write_text('{"Problem":{"Type":"Eigenmode"}}\n')
    manifest = metadata / "mesh_manifest.json"
    manifest.write_text('{"route":"B"}\n')
    provenance = metadata / "provenance.json"
    provenance.write_text('{"route":"B"}\n')
    (metadata / "palace_index_map.json").write_text('{"schema_version":1,"entries":[]}\n')
    mesh_result = MeshBuildResult(
        output_dir=run_dir, mesh_path=mesh, groups={},
        mesh_manifest_path=manifest, provenance_path=provenance,
        stack_path=metadata / "stack.json", xao_path=run_dir / "geometry.xao",
        sidecar_path=metadata / "sidecar.json",
    )
    plan = prepare_handoff(
        profile="direct-local", mesh_result=mesh_result, config_path=config,
        executable="palace", resources={"processes": 1, "threads": 1},
        problem="Eigenmode",
    )
    handoff = json.loads(plan.metadata_path.read_text())
    run = json.loads(plan.run_metadata_path.read_text())
    identity = handoff["palace_identity"]
    assert "requested_release" not in identity
    assert "release_selection_semantics" not in identity
    assert identity["runtime_identity_semantics"] == "observed_provenance_only"
    assert identity["config_compatibility_probe"] == "native_dry_run"
    assert run["palace_identity"] == identity
    assert run["status"] == "not_run"
    assert handoff["schema_version"] == run["schema_version"] == 1
    assert not plan.returned_receipt_path.exists()
    assert "verify-config" in plan.script_path.read_text()
    assert returned_receipt._compatibility_required(handoff) == identity["config_schema"]
    _validate_palace_identity(identity)
    historical = {key: value for key, value in identity.items() if key not in {
        "requested_release", "release_selection_semantics",
    }}
    _validate_palace_identity(historical)
    assert returned_receipt._compatibility_required({"palace_identity": historical}) == identity["config_schema"]


def test_version_probe_failure_stays_observed_and_dry_run_required(monkeypatch, tmp_path):
    rc, payload = _preflight(monkeypatch, tmp_path, version_rc=1)
    assert rc == 0
    assert payload["observed_identity"]["version_probe_exit_code"] == 1
    assert payload["compatibility"]["method"] == "native_dry_run"
    assert payload["compatibility"]["verified"] is True
    rc, payload = _preflight(monkeypatch, tmp_path, version_rc=1, dry_run_rc=1)
    assert rc == 78
    assert payload["compatibility"]["verified"] is False
