"""Offline control/file/lifecycle regressions; no SDK, live PID or solver proof."""

import copy
import json
from types import SimpleNamespace

import pytest

from scgsim.aedt.runtime import control
from scgsim.aedt.runtime.families import q3d
from scgsim.aedt.runtime.native.common import BoundAedtRequest, analyze_with_resources
from scgsim.aedt.runtime.transaction import _release_owned_desktop
from scgsim.aedt.results import provenance
from scgsim.aedt.specs.common import MatrixRunControl, PdkMaterial
from scgsim.aedt.specs.q3d import Q3dBodySpec, Q3dNetSpec, Q3dSpec


def _spec(version="2024.2"):
    return Q3dSpec(
        project_name="Stopped", design_name="Design", aedt_version=version,
        modeling="solid",
        materials={"vacuum": PdkMaterial("vacuum", "vacuum", False, "Vacuum"),
                   "metal": PdkMaterial("metal", "superconductor", True, None)},
        vacuum_material_id="vacuum",
        bodies=(Q3dBodySpec("pad", ((0, 0), (2, 0), (2, 2), (0, 2)), (),
                            0, 1, "metal", "signal", "Signal"),),
        nets=(Q3dNetSpec("Signal", "Signal", ("pad",)),),
        run_control=MatrixRunControl("Setup1", 1.0, 3), solve_ac_rl=False,
        region_padding_um=(1, 1, 1, 1, 1, 1),
    )


@pytest.fixture
def stored_control(tmp_path, monkeypatch):
    """Real private JSON state with a real declared source digest; no live process."""
    metadata = tmp_path / "metadata" / "aedt_handoff_metadata.json"
    metadata.parent.mkdir()
    metadata.write_text("{}")
    monkeypatch.setattr(provenance, "_observed_revision", lambda: None)
    source = provenance.runtime_source_identity()
    inputs = {"metadata_sha256": "a" * 64, "spec_sha256": "b" * 64,
              "manifest_sha256": "c" * 64}
    spec = _spec()
    monkeypatch.setattr(control, "_inputs", lambda path: (inputs, spec))
    monkeypatch.setattr(control, "runtime_source_identity", lambda: source)
    (metadata.parent / "aedt_run_receipt.json").write_text(
        json.dumps({"status": "running", "runtime_source": source})
    )
    state = {"schema_version": control._SCHEMA, "handoff": str(metadata),
             "inputs": inputs, "runtime_source": source,
             "recorder": {"pid": -1}, "phase": "closed", "intent": None,
             "desktop": {"version": "2024.2"}}
    with control._locked(metadata.parent / control._LOCK, create=True):
        control._store(metadata.parent / control._STATE, state)
    return metadata, state


@pytest.mark.parametrize("action,request_fn", [
    ("stop_and_save", control.request_q3d_stop_and_save),
    ("abort_and_save", control.request_q3d_abort_and_save),
])
def test_committed_intent_is_idempotent_without_attach(stored_control, action, request_fn):
    path, state = stored_control
    state["intent"] = {"action": action, "rpc_status": "unknown"}
    control._store(path.parent / control._STATE, state)
    assert request_fn(path) == state
    assert request_fn(path) == state


def test_conflicting_intent_is_rejected_without_attach(stored_control):
    path, state = stored_control
    state["intent"] = {"action": "stop_and_save", "rpc_status": "unknown"}
    control._store(path.parent / control._STATE, state)
    with pytest.raises(RuntimeError, match="conflicts"):
        control.request_q3d_abort_and_save(path)


def test_closed_admission_and_changed_inputs_refuse_before_native(stored_control):
    path, state = stored_control
    with pytest.raises(RuntimeError, match="admission is closed"):
        control.request_q3d_stop_and_save(path)
    state["inputs"] = {**state["inputs"], "spec_sha256": "d" * 64}
    control._store(path.parent / control._STATE, state)
    with pytest.raises(RuntimeError, match="input bytes changed"):
        control.request_q3d_stop_and_save(path)


def test_version_mismatch_refuses_before_process_or_sdk(stored_control, monkeypatch):
    path, state = stored_control
    state.update(phase="analysing", desktop={"version": "2023.2"})
    control._store(path.parent / control._STATE, state)
    monkeypatch.setattr(control, "_verify_process", lambda row: pytest.fail("too late"))
    with pytest.raises(RuntimeError, match="Desktop version differs"):
        control.request_q3d_stop_and_save(path)


@pytest.mark.parametrize("version", ["2023.2", "2024.2"])
def test_native_identity_rejects_wrong_endpoint_before_project_access(monkeypatch, version):
    monkeypatch.setattr(control, "_verify_process", lambda row: None)
    endpoint = SimpleNamespace(aedt_process_id=12, port=31, aedt_version_id=version,
                               odesktop=SimpleNamespace(GetProcessID=lambda: 12))
    state = {"desktop": {"process": {"pid": 12}, "port": 32, "version": version}}
    with pytest.raises(RuntimeError, match="did not bind the owned Desktop"):
        control._native_identity(endpoint, state)


def _prepared(tmp_path, monkeypatch, save):
    spec = _spec()
    monkeypatch.setattr(q3d, "_pyaedt_version", lambda: "1.3.0")
    project = tmp_path / "Stopped.aedt"
    project.write_bytes(b"older preparation bytes")
    app = SimpleNamespace(save_project=save,
                          desktop_class=SimpleNamespace(aedt_version_id="2024.2"))
    return q3d.PreparedQ3d(app, BoundAedtRequest.bind(tmp_path, spec), project,
                            [], {}, [], {})


def _termination(status="inactive"):
    return {"intent": {"action": "stop_and_save", "rpc_status": "returned"},
            "analysis_call": {"status": "returned", "native_return": True},
            "native_settlement": {"status": status}, "desktop": {"process": {"pid": 12}}}


def test_unconfirmed_inactivity_never_saves(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path, monkeypatch, lambda: pytest.fail("must not save"))
    result = q3d.termination_q3d_result(prepared, _termination("unavailable"), None)
    assert result["save"] == {"ok": False, "attempted": False,
                              "reason": "native_not_confirmed_inactive"}
    assert result["outputs"] == {}


def test_failed_save_does_not_seal_older_project_or_mask_analysis_error(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path, monkeypatch, lambda: False)
    def unavailable(*args, **kwargs):
        raise RuntimeError("offline export unavailable")
    monkeypatch.setattr(q3d, "_export_matrix_file", unavailable)
    monkeypatch.setattr(q3d, "read_q3d_convergence", unavailable)
    primary = RuntimeError("original Analyze failure")
    result = q3d.termination_q3d_result(prepared, _termination(), primary)
    assert result["save"]["ok"] is False
    assert result["save"]["attempted"] is True
    assert result["termination"]["primary_error"] == "RuntimeError: original Analyze failure"
    assert "Stopped.aedt" not in result["outputs"]
    assert prepared.project_path.read_bytes() == b"older preparation bytes"


@pytest.mark.parametrize("saved,inactive,expected_close", [
    (False, True, False), (True, False, False), (True, True, True),
])
def test_release_detaches_unsaved_or_active_project(saved, inactive, expected_close):
    calls = []
    def release(**kwargs):
        calls.append(kwargs)
        return True
    result = {"termination": {"control": _termination("inactive" if inactive else "active")},
              "save": {"ok": saved, "attempted": True}}
    receipt = {}
    _release_owned_desktop(SimpleNamespace(release_desktop=release), receipt, result)
    assert calls == [{"close_projects": expected_close, "close_on_exit": expected_close}]
    assert receipt["release"]["ok"] is expected_close
    if not expected_close:
        assert receipt["release"]["ownership"] == "unresolved"
        assert receipt["release"]["preserved_desktop"] == result["termination"]["control"]["desktop"]


def test_analyze_hook_wraps_one_actual_callback_and_retains_false():
    calls = []
    def analyze(**kwargs):
        calls.append(kwargs)
        return False
    def hook(call):
        calls.append("enter")
        answer = call()
        calls.append(("returned", answer))
        return answer
    evidence = {}
    assert analyze_with_resources(SimpleNamespace(analyze_setup=analyze), "Setup1", None,
                                  None, evidence, analysis_call=hook) is False
    assert calls == ["enter", {"name": "Setup1", "blocking": True}, ("returned", False)]


def test_current_inventory_enrolls_control_without_changing_v24(monkeypatch):
    monkeypatch.setattr(provenance, "_observed_revision", lambda: None)
    source = provenance.runtime_source_identity()
    provenance.validate_runtime_source(source, stage="actual")
    assert source["schema_version"] == "scgsim.aedt.runtime-source.v25"
    assert set(provenance._RUNTIME_SOURCE_V25_PATHS) - set(provenance._RUNTIME_SOURCE_V24_PATHS) == {
        "src/scgsim/aedt/runtime/control.py"}
    assert "src/scgsim/aedt/runtime/control.py" not in provenance._RUNTIME_SOURCE_V24_PATHS
    damaged = copy.deepcopy(source)
    damaged["modules"][0]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="content digest"):
        provenance.validate_runtime_source(damaged, stage="actual")


@pytest.mark.parametrize("request_fn,clean_stop", [
    (control.request_q3d_stop_and_save, True),
    (control.request_q3d_abort_and_save, False),
])
def test_public_actions_keep_distinct_clean_stop_flags(monkeypatch, request_fn, clean_stop):
    calls = []
    def record(handoff, *, clean_stop):
        calls.append((handoff, clean_stop))
        return {"action_recorded": clean_stop}
    monkeypatch.setattr(control, "_request_q3d_termination_and_save", record)
    assert request_fn("handoff") == {"action_recorded": clean_stop}
    assert calls == [("handoff", clean_stop)]


def test_detach_error_keeps_preserved_identity_and_no_close_attempt():
    calls = []
    def release(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("detach failed")
    result = {"termination": {"control": _termination()},
              "save": {"ok": False, "attempted": True}}
    receipt = {}
    with pytest.raises(RuntimeError, match="detach failed"):
        _release_owned_desktop(SimpleNamespace(release_desktop=release), receipt, result)
    assert calls == [{"close_projects": False, "close_on_exit": False}]
    assert receipt["release"]["ownership"] == "unresolved"
    assert receipt["release"]["detach"]["ok"] is False
    assert receipt["release"]["preserved_desktop"] == result["termination"]["control"]["desktop"]
