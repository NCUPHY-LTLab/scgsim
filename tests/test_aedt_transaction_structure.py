"""Exercise handoff failure and release order with native unit-test doubles."""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from types import ModuleType

import pytest
from test_aedt_public_preparation import produce_request

from scgsim.aedt.io import file_sha256
from scgsim.aedt.runtime import transaction


def _native_double(monkeypatch, plan, events, *, prepare_fails=False, release_ok=True):
    class Desktop:
        """Record transaction release order and write the bytes observed after release."""

        aedt_version_id = "2024.2"

        def __init__(self, **kwargs):
            events.append("desktop")

        def release_desktop(self, *, close_projects, close_on_exit):
            assert close_projects is True
            assert close_on_exit is True
            events.append("release")
            (plan.run_dir / "ProtocolSource.aedt").write_bytes(b"released bytes")
            return release_ok

    @contextmanager
    def compatibility(profile, *, evidence):
        events.append("compatibility_enter")
        try:
            yield
        finally:
            events.append("compatibility_exit")

    def prepare(constructor, run_dir, spec):
        events.append("prepare")
        if prepare_fails:
            error = ValueError("primary preparation failure")
            error.add_note("native source context")
            raise error
        (run_dir / "ProtocolSource.aedt").write_bytes(b"before release")
        return {
            "outputs": {},
            "connected": {},
            "project": "ProtocolSource.aedt",
            "geometry": {},
            "setup": {},
            "solver_invoked": False,
            "workflow_status": "native_preparation_only",
        }

    def prepared_result(result):
        events.append("result")
        return result

    native = ModuleType("ansys.aedt.core")
    native.Desktop = Desktop
    native.Hfss = object
    native.Q2d = object
    native.Q3d = object
    monkeypatch.setitem(sys.modules, "ansys.aedt.core", native)
    monkeypatch.setattr(transaction, "native_compatibility", compatibility)
    monkeypatch.setattr(transaction, "pyaedt_version", lambda: "1.3.0")
    monkeypatch.setattr(
        transaction,
        "owned_application_constructor",
        lambda constructor, desktop, **kwargs: constructor,
    )
    monkeypatch.setattr(transaction, "prepare_body_eigenmode", prepare)
    monkeypatch.setattr(transaction, "prepared_body_eigenmode_result", prepared_result)


def test_preparation_seals_project_bytes_after_release(tmp_path, monkeypatch):
    """The receipt identifies released project bytes and records completed cleanup."""
    plan = produce_request(tmp_path / "request")
    monkeypatch.chdir(tmp_path)
    events = []
    _native_double(monkeypatch, plan, events)

    code = transaction.execute_handoff(
        plan.metadata_path, options=transaction.ExecutionOptions(prepare_only=True)
    )
    receipt = json.loads(plan.receipt_path.read_text())

    assert code == 0
    assert events == [
        "compatibility_enter",
        "desktop",
        "prepare",
        "result",
        "release",
        "compatibility_exit",
    ]
    assert receipt["status"] == "native_preparation_only"
    assert receipt["solver_invoked"] is False
    assert receipt["release"] == {"ok": True}
    assert receipt["save"]["identity_stage"] == "after_owned_desktop_release"
    assert receipt["outputs"]["ProtocolSource.aedt"] == file_sha256(
        plan.run_dir / "ProtocolSource.aedt"
    )


def test_primary_failure_survives_release_failure(tmp_path, monkeypatch):
    """Failed preparation still releases the Desktop and restores compatibility."""
    plan = produce_request(tmp_path / "request")
    monkeypatch.chdir(tmp_path)
    events = []
    _native_double(monkeypatch, plan, events, prepare_fails=True, release_ok=False)

    code = transaction.execute_handoff(
        plan.metadata_path, options=transaction.ExecutionOptions(prepare_only=True)
    )
    receipt = json.loads(plan.receipt_path.read_text())

    assert code == 1
    assert events == [
        "compatibility_enter",
        "desktop",
        "prepare",
        "release",
        "compatibility_exit",
    ]
    assert (
        receipt["error"]
        == "ValueError: primary preparation failure\nnative source context"
    )
    assert receipt["release"]["ok"] is False
    assert (
        receipt["release"]["error"]
        == "RuntimeError: owned AEDT Desktop release returned false"
    )


def test_tampered_handoff_fails_before_native_execution(tmp_path, monkeypatch):
    """Manifest verification rejects changed prepared bytes before receipt transition."""
    plan = produce_request(tmp_path / "request")
    monkeypatch.chdir(tmp_path)
    plan.spec_path.write_text(plan.spec_path.read_text() + "\n")
    events = []
    _native_double(monkeypatch, plan, events)

    with pytest.raises(RuntimeError, match="handoff manifest member mismatch"):
        transaction.execute_handoff(
            plan.metadata_path, options=transaction.ExecutionOptions(prepare_only=True)
        )

    assert events == []
    assert json.loads(plan.receipt_path.read_text())["status"] == "not_run"
