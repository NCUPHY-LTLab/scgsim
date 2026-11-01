"""Sidecars retain evidence bytes while converting shared records once."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from scgsim.geometry.compiler import builder


@dataclass(frozen=True)
class _Plan:
    metadata: dict
    outputs: tuple


def _legacy_redact(value, run_root):
    if isinstance(value, _Plan):
        return _legacy_redact(asdict(value), run_root)
    if isinstance(value, dict):
        return {key: _legacy_redact(item, run_root) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_legacy_redact(item, run_root) for item in value)
    if isinstance(value, list):
        return [_legacy_redact(item, run_root) for item in value]
    if isinstance(value, str) and Path(value).is_absolute():
        return builder._redact_one_path(Path(value), run_root=run_root)  # noqa: SLF001 -- the legacy reference uses the same leaf redactor to isolate traversal changes
    return value


def test_shared_ledger_retains_exact_json_and_redaction(tmp_path, monkeypatch):
    """Keep every serialized ledger occurrence while converting shared data once."""
    ledger = [{"source_id": "parent", "path": str(tmp_path / "geometry" / "model.xao")}]
    plan = _Plan(
        {"bands": [{"ledger": ledger}, {"ledger": ledger}]},
        (str(tmp_path / "out"), "/external/fixture/source.gds"),
    )
    expected = json.dumps(_legacy_redact(plan, tmp_path), indent=2, sort_keys=True)
    calls = []
    original = builder._redact_one_path  # noqa: SLF001 -- observe the private leaf redactor without replacing its results

    def counted(path, *, run_root):
        calls.append(path)
        return original(path, run_root=run_root)

    monkeypatch.setattr(builder, "_redact_one_path", counted)
    directory = tmp_path / "metadata" / "semantic_geometry"
    size = builder._write_stage_sidecar(directory, "02_plan", plan)  # noqa: SLF001 -- verify the internal writer's exact bytes and measured size together
    content = (directory / "02_plan.json").read_bytes()
    assert content == expected.encode("utf-8")
    assert size == len(content)
    # Both JSON occurrences survive, but the common source ledger is visited once.
    assert content.count(b'"source_id": "parent"') == 2
    assert calls.count(tmp_path / "geometry" / "model.xao") == 1
    assert ledger[0]["path"] == str(tmp_path / "geometry" / "model.xao")


def test_conversion_cache_is_local_and_does_not_hold_stale_evidence(tmp_path):
    """Reuse shared data within a call and reread source changes in the next call."""
    ledger = [{"value": 1}]
    value = {"first": ledger, "second": ledger}
    initial = builder._redact_local_paths(value, run_root=tmp_path)  # noqa: SLF001 -- inspect identity reuse inside the call-local conversion cache
    assert initial["first"] is initial["second"]
    ledger[0]["value"] = 2
    updated = builder._redact_local_paths(value, run_root=tmp_path)  # noqa: SLF001 -- ensure a later private conversion does not retain the earlier cache
    assert updated["first"][0]["value"] == 2
    assert initial["first"][0]["value"] == 1


def test_timing_reports_actual_sidecar_bytes():
    """Retain measured sidecar bytes in the transaction timing snapshot."""
    transaction = builder._BuildTransaction("A", Path("metadata"), Path("geometry"))  # noqa: SLF001 -- Exercise the internal timing snapshot directly.
    transaction.sidecar_sizes["02_plan"] = 321
    payload = transaction.timing_payload()
    assert payload["sidecar_bytes"] == {"02_plan": 321}
