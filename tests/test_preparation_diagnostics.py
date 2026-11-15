"""Direct contracts for opt-in preparation diagnostics and their cleanup."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import scgsim.diagnostics as diagnostics
from scgsim.diagnostics import (  # private-import: inspect worker handoff and stage context contracts
    diagnostic_stage,
    preparation_diagnostics,
    worker_config,
    worker_diagnostics,
)


def _read_events(output_dir: Path) -> list[tuple[Path, list[dict[str, object]]]]:
    """Load each process-local JSONL file with its path for ownership checks."""
    return [
        (
            path,
            [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ],
        )
        for path in sorted(output_dir.glob("*.jsonl"))
    ]


def test_nested_worker_session_keeps_session_and_parent_span(tmp_path: Path) -> None:
    """Carry diagnostic session identity into a child and restore parent span."""
    with preparation_diagnostics(tmp_path):
        worker_payload = worker_config()
        assert worker_payload is not None
        assert set(worker_payload) == {
            "output_dir",
            "progress_interval_s",
            "profile",
            "session_id",
            "parent_span_id",
        }
        with worker_diagnostics(worker_payload):
            with diagnostic_stage("preparation.worker.task"):
                pass
        with diagnostic_stage("preparation.parent.after_worker"):
            pass

    event_files = _read_events(tmp_path)
    assert len(event_files) == 2
    session_records = [
        (path, row)
        for path, rows in event_files
        for row in rows
        if row["event"] == "start" and row["stage"] == "preparation.session"
    ]
    parent_file, parent_session = next(
        (path, row) for path, row in session_records if row["parent_span_id"] is None
    )
    child_session = next(
        row for _, row in session_records if row["parent_span_id"] is not None
    )
    parent_rows = next(rows for path, rows in event_files if path == parent_file)
    assert child_session["session_id"] == parent_session["session_id"]
    assert child_session["parent_span_id"] == parent_session["span_id"]
    parent_followup = next(
        row for row in parent_rows if row["stage"] == "preparation.parent.after_worker"
    )
    assert parent_followup["parent_span_id"] == parent_session["span_id"]


def test_stage_records_error_and_restores_context(tmp_path: Path) -> None:
    """Propagate the original stage error and keep later spans correctly nested."""
    failure = ValueError("stage failed")
    with preparation_diagnostics(tmp_path):
        with pytest.raises(ValueError) as observed:
            with diagnostic_stage("preparation.failing"):
                raise failure
        assert observed.value is failure
        with diagnostic_stage("preparation.after_failure"):
            pass

    rows = [row for _, records in _read_events(tmp_path) for row in records]
    failed_start = next(row for row in rows if row["stage"] == "preparation.failing")
    failed_error = next(
        row
        for row in rows
        if row["stage"] == "preparation.failing" and row["event"] == "error"
    )
    later_stage = next(
        row for row in rows if row["stage"] == "preparation.after_failure"
    )
    assert failed_start["event"] == "start"
    assert failed_error["error_type"] == "ValueError"
    assert later_stage["parent_span_id"] == failed_start["parent_span_id"]


def test_cleanup_failure_preserves_primary_error_and_resets_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Attach profile cleanup failure to the primary error and allow a new session."""

    class BrokenProfile:
        def enable(self) -> None:
            pass

        def disable(self) -> None:
            pass

        def dump_stats(self, path: str) -> None:
            raise OSError("profile destination failed")

    monkeypatch.setattr(diagnostics.cProfile, "Profile", BrokenProfile)
    failure = ValueError("preparation failed")
    with pytest.raises(ValueError) as observed:
        with preparation_diagnostics(tmp_path, profile=True):
            raise failure

    assert observed.value is failure
    assert any(
        "Preparation diagnostics: failed: OSError" in note for note in failure.__notes__
    )
    with preparation_diagnostics(tmp_path):
        with diagnostic_stage("preparation.after_cleanup_failure"):
            pass
    assert len(_read_events(tmp_path)) == 2


@pytest.mark.parametrize("interval", (-1.0, float("nan"), float("inf"), True))
def test_invalid_progress_interval_fails_before_opening_a_session(
    interval: float, tmp_path: Path
) -> None:
    """Reject negative, nonfinite, and boolean progress intervals at entry."""
    expected_error = TypeError if isinstance(interval, bool) else ValueError
    with pytest.raises(expected_error):
        with preparation_diagnostics(tmp_path, progress_interval_s=interval):
            pytest.fail("invalid diagnostic interval entered")
    assert not list(tmp_path.iterdir())
