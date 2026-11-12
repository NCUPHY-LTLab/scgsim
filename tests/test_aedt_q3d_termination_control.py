"""Offline control/file/lifecycle regressions; no SDK, live PID or solver proof."""

import copy
import json
from types import SimpleNamespace

import pytest

from scgsim.aedt.results import provenance
from scgsim.aedt.results.runtime_inventory import current_runtime_source_paths
from scgsim.aedt.runtime import control
from scgsim.aedt.runtime.desktop_lifecycle import release_owned_desktop
from scgsim.aedt.runtime.families import q3d, q3d_prepared, q3d_termination
from scgsim.aedt.runtime.native.common import BoundAedtRequest
from scgsim.aedt.runtime.native.resources import analyze_with_resources
from scgsim.aedt.specs.common import MatrixRunControl, PdkMaterial
from scgsim.aedt.specs.q3d import Q3dBodySpec, Q3dNetSpec, Q3dSpec


def _spec(version="2024.2"):
    return Q3dSpec(
        project_name="Stopped",
        design_name="Design",
        aedt_version=version,
        modeling="solid",
        materials={
            "vacuum": PdkMaterial("vacuum", "vacuum", False, "Vacuum"),
            "metal": PdkMaterial("metal", "superconductor", True, None),
        },
        vacuum_material_id="vacuum",
        bodies=(
            Q3dBodySpec(
                "pad",
                ((0, 0), (2, 0), (2, 2), (0, 2)),
                (),
                0,
                1,
                "metal",
                "signal",
                "Signal",
            ),
        ),
        nets=(Q3dNetSpec("Signal", "Signal", ("pad",)),),
        run_control=MatrixRunControl("Setup1", 1.0, 3),
        solve_ac_rl=False,
        region_padding_um=(1, 1, 1, 1, 1, 1),
    )


@pytest.fixture
def stored_control(tmp_path, monkeypatch):
    """Real private JSON state with a real declared source digest; no live process."""
    metadata = tmp_path / "metadata" / "aedt_handoff_metadata.json"
    metadata.parent.mkdir()
    metadata.write_text("{}")
    monkeypatch.setattr(provenance, "_observed_revision", lambda: None)
    source = provenance.record_runtime_source("actual")
    inputs = {
        "metadata_sha256": "a" * 64,
        "spec_sha256": "b" * 64,
        "manifest_sha256": "c" * 64,
    }
    spec = _spec()
    monkeypatch.setattr(control, "_inputs", lambda path: (inputs, spec))
    monkeypatch.setattr(control, "record_runtime_source", lambda stage: source)
    (metadata.parent / "aedt_run_receipt.json").write_text(
        json.dumps({"status": "running", "runtime_source": source})
    )
    state = {
        "schema_version": control._SCHEMA,  # noqa: SLF001 -- use the recorder's authoritative wire schema for invalid-state tests.
        "handoff": str(metadata),
        "inputs": inputs,
        "runtime_source": source,
        "recorder": {"pid": -1},
        "phase": "closed",
        "intent": None,
        "desktop": {"version": "2024.2"},
    }
    with control._locked(metadata.parent / control._LOCK, create=True):  # noqa: SLF001 -- seed recorder state under its actual admission lock.
        control._store(metadata.parent / control._STATE, state)  # noqa: SLF001 -- seed actual recorder state for pre-attachment admission checks.
    return metadata, state


@pytest.mark.parametrize(
    "action,request_fn",
    [
        ("stop_and_save", control.request_q3d_stop_and_save),
        ("abort_and_save", control.request_q3d_abort_and_save),
    ],
)
def test_committed_intent_is_idempotent_without_attach(
    stored_control, action, request_fn
):
    """Return the same committed stop intent without another native attachment."""
    path, state = stored_control
    state["intent"] = {"action": action, "rpc_status": "unknown"}
    control._store(path.parent / control._STATE, state)  # noqa: SLF001 -- commit an existing intent to test the idempotent no-attach path.
    assert request_fn(path) == state
    assert request_fn(path) == state


def test_conflicting_intent_is_rejected_without_attach(stored_control):
    """Reject a conflicting stop intent before native attachment."""
    path, state = stored_control
    state["intent"] = {"action": "stop_and_save", "rpc_status": "unknown"}
    control._store(path.parent / control._STATE, state)  # noqa: SLF001 -- commit a conflicting intent before testing no-attach rejection.
    with pytest.raises(RuntimeError, match="conflicts"):
        control.request_q3d_abort_and_save(path)


def test_closed_admission_and_changed_inputs_refuse_before_native(stored_control):
    """Reject closed admission and changed prepared inputs before native access."""
    path, state = stored_control
    with pytest.raises(RuntimeError, match="admission is closed"):
        control.request_q3d_stop_and_save(path)
    state["inputs"] = {**state["inputs"], "spec_sha256": "d" * 64}
    control._store(path.parent / control._STATE, state)  # noqa: SLF001 -- corrupt the stored input digest to test admission invalidation.
    with pytest.raises(RuntimeError, match="input bytes changed"):
        control.request_q3d_stop_and_save(path)


def test_version_mismatch_refuses_before_process_or_sdk(stored_control, monkeypatch):
    """Reject a Desktop version mismatch before process or SDK access."""
    path, state = stored_control
    state.update(phase="analysing", desktop={"version": "2023.2"})
    control._store(path.parent / control._STATE, state)  # noqa: SLF001 -- seed a Desktop version mismatch without native attachment.
    monkeypatch.setattr(control, "_verify_process", lambda row: pytest.fail("too late"))
    with pytest.raises(RuntimeError, match="Desktop version differs"):
        control.request_q3d_stop_and_save(path)


@pytest.mark.parametrize("version", ["2023.2", "2024.2"])
def test_native_identity_rejects_wrong_endpoint_before_project_access(
    monkeypatch, version
):
    """Reject the wrong owned Desktop endpoint before accessing its project."""
    monkeypatch.setattr(control, "_verify_process", lambda row: None)
    endpoint = SimpleNamespace(
        aedt_process_id=12,
        port=31,
        aedt_version_id=version,
        odesktop=SimpleNamespace(GetProcessID=lambda: 12),
    )
    state = {"desktop": {"process": {"pid": 12}, "port": 32, "version": version}}
    with pytest.raises(RuntimeError, match="did not bind the owned Desktop"):
        control._native_identity(endpoint, state)  # noqa: SLF001 -- exercise the endpoint-identity guard before it can access a project.


def _prepared(tmp_path, monkeypatch, save):
    spec = _spec()
    monkeypatch.setattr(q3d_prepared, "_pyaedt_version", lambda: "1.3.0")
    project = tmp_path / "Stopped.aedt"
    project.write_bytes(b"older preparation bytes")
    app = SimpleNamespace(
        save_project=save, desktop_class=SimpleNamespace(aedt_version_id="2024.2")
    )
    return q3d.PreparedQ3d(
        app, BoundAedtRequest.bind(tmp_path, spec), project, [], {}, [], {}
    )


def _termination(status="inactive"):
    return {
        "intent": {"action": "stop_and_save", "rpc_status": "returned"},
        "analysis_call": {"status": "returned", "native_return": True},
        "native_settlement": {"status": status},
        "desktop": {"process": {"pid": 12}},
    }


def test_unconfirmed_inactivity_never_saves(tmp_path, monkeypatch):
    """Defer saving and artifact collection until native inactivity is confirmed."""
    prepared = _prepared(tmp_path, monkeypatch, lambda: pytest.fail("must not save"))
    result = q3d.termination_q3d_result(prepared, _termination("unavailable"), None)
    assert result["save"] == {
        "ok": False,
        "attempted": False,
        "reason": "native_not_confirmed_inactive",
    }
    assert result["outputs"] == {}


def test_failed_save_does_not_seal_older_project_or_mask_analysis_error(
    tmp_path, monkeypatch
):
    """Keep the primary analysis error and avoid sealing older bytes after a failed save."""
    prepared = _prepared(tmp_path, monkeypatch, lambda: False)

    def unavailable(*args, **kwargs):
        raise RuntimeError("offline export unavailable")

    monkeypatch.setattr(q3d_termination, "export_matrix_file", unavailable)
    monkeypatch.setattr(q3d_termination, "read_q3d_convergence", unavailable)
    primary = RuntimeError("original Analyze failure")
    result = q3d.termination_q3d_result(prepared, _termination(), primary)
    assert result["save"]["ok"] is False
    assert result["save"]["attempted"] is True
    assert (
        result["termination"]["primary_error"]
        == "RuntimeError: original Analyze failure"
    )
    assert "Stopped.aedt" not in result["outputs"]
    assert prepared.project_path.read_bytes() == b"older preparation bytes"


@pytest.mark.parametrize(
    "saved,inactive,expected_close",
    [
        (False, True, False),
        (True, False, False),
        (True, True, True),
    ],
)
def test_release_detaches_unsaved_or_active_project(saved, inactive, expected_close):
    """Detach retained active or unsaved projects without requesting project closure."""
    calls = []

    def release(**kwargs):
        calls.append(kwargs)
        return True

    result = {
        "termination": {"control": _termination("inactive" if inactive else "active")},
        "save": {"ok": saved, "attempted": True},
    }
    receipt = {}
    release_owned_desktop(SimpleNamespace(release_desktop=release), receipt, result)
    assert calls == [
        {"close_projects": expected_close, "close_on_exit": expected_close}
    ]
    assert receipt["release"]["ok"] is expected_close
    if not expected_close:
        assert receipt["release"]["ownership"] == "unresolved"
        assert (
            receipt["release"]["preserved_desktop"]
            == result["termination"]["control"]["desktop"]
        )


def test_analyze_hook_wraps_one_actual_callback_and_retains_false():
    """Wrap one actual analyze callback and preserve its false return value."""
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
    assert (
        analyze_with_resources(
            SimpleNamespace(analyze_setup=analyze),
            "Setup1",
            None,
            None,
            evidence,
            analysis_call=hook,
        )
        is False
    )
    assert calls == ["enter", {"name": "Setup1", "blocking": True}, ("returned", False)]


def test_current_runtime_source_enrolls_control(monkeypatch):
    """Record the stop-control module in the runtime source."""
    monkeypatch.setattr(provenance, "_observed_revision", lambda: None)
    source = provenance.record_runtime_source("actual")
    provenance.validate_runtime_source(source, stage="actual")
    assert source["schema_version"] == provenance.SOURCE_SCHEMA
    assert "src/scgsim/aedt/runtime/control.py" in current_runtime_source_paths()
    damaged = copy.deepcopy(source)
    damaged["modules"][0]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="content digest"):
        provenance.validate_runtime_source(damaged, stage="actual")


@pytest.mark.parametrize(
    "request_fn,clean_stop",
    [
        (control.request_q3d_stop_and_save, True),
        (control.request_q3d_abort_and_save, False),
    ],
)
def test_public_actions_keep_distinct_clean_stop_flags(
    monkeypatch, request_fn, clean_stop
):
    """Pass distinct clean-stop flags for the public stop and abort actions."""
    calls = []

    def record(handoff, *, clean_stop):
        calls.append((handoff, clean_stop))
        return {"action_recorded": clean_stop}

    monkeypatch.setattr(control, "_request_q3d_termination_and_save", record)
    assert request_fn("handoff") == {"action_recorded": clean_stop}
    assert calls == [("handoff", clean_stop)]


def test_detach_error_keeps_preserved_identity_and_no_close_attempt():
    """Retain Desktop identity after detach failure without attempting project closure."""
    calls = []

    def release(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("detach failed")

    result = {
        "termination": {"control": _termination()},
        "save": {"ok": False, "attempted": True},
    }
    receipt = {}
    with pytest.raises(RuntimeError, match="detach failed"):
        release_owned_desktop(SimpleNamespace(release_desktop=release), receipt, result)
    assert calls == [{"close_projects": False, "close_on_exit": False}]
    assert receipt["release"]["ownership"] == "unresolved"
    assert receipt["release"]["detach"]["ok"] is False
    assert (
        receipt["release"]["preserved_desktop"]
        == result["termination"]["control"]["desktop"]
    )
