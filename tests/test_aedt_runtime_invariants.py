"""Runtime guards remain effective when Python optimization removes asserts."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from scgsim.aedt import batch_stages
from scgsim.aedt.presentation import runs
from scgsim.aedt.runtime import batch_protocol as protocol
from scgsim.aedt.specs.common import AedtResources
from scgsim.aedt.specs.hfss import HfssEprSpec


def test_package_manifest_requires_validated_epr_run_control(tmp_path, monkeypatch):
    """Keep package control validation active with optimized assertions disabled."""
    output = tmp_path / "package"
    output.mkdir()
    context = batch_stages.PreparationContext(
        root=tmp_path,
        receipt={},
        spec=cast(HfssEprSpec, SimpleNamespace(run_control=None)),
        project_path=tmp_path / "model.aedt",
        resource_payload={},
        solver_executable="aedtexec",
        dependencies=(),
    )
    monkeypatch.setattr(
        batch_stages, "_prepare_context", lambda *args, **kwargs: context
    )
    monkeypatch.setattr(protocol, "fresh", lambda *args: output)
    monkeypatch.setattr(batch_stages, "_write_handoff_inputs", lambda *args: None)
    monkeypatch.setattr(batch_stages, "_copy_native_files", lambda *args: "model.aedt")
    monkeypatch.setattr(batch_stages, "_stage_portable_runner", lambda *args: None)

    with pytest.raises(
        TypeError, match="native batch requires a complete body-first EPR preparation"
    ):
        batch_stages.prepare_native_batch(
            tmp_path,
            output_dir=output,
            solver_executable="aedtexec",
            resources=AedtResources(2, 8),
        )


def test_collection_rejects_missing_terminal_inventories(tmp_path, monkeypatch):
    """Keep attempt evidence admission active with optimized assertions disabled."""
    package_root = tmp_path / "package"
    attempt_root = tmp_path / "attempt"
    output = tmp_path / "collection"
    package_root.mkdir()
    attempt_root.mkdir()
    output.mkdir()
    (package_root / protocol.MANIFEST).write_text("manifest", encoding="utf-8")
    package = cast(protocol.PackageManifest, SimpleNamespace(members=()))
    attempt = protocol.BatchAttempt(
        attempt_id="attempt",
        package_sha256="package-hash",
        package_id="package",
        status="external_process_terminal",
        requested_aedt_version="2024.2",
        actual_aedt_version="2024.2",
        version_observation="recorded",
        execution_location={},
        resources_requested=protocol.Resources(2, 8),
        resource_semantics="native setting",
        started_at_unix=1,
        returncode=0,
    )
    monkeypatch.setattr(protocol, "read_package", lambda _root: package)
    monkeypatch.setattr(protocol, "read_attempt", lambda *_args: attempt)
    monkeypatch.setattr(protocol, "fresh", lambda *_args: output)
    monkeypatch.setattr(protocol, "inventory", lambda _root: [])
    monkeypatch.setattr(protocol, "copy_members", lambda *_args: None)
    monkeypatch.setattr(protocol, "sha256", lambda _path: "file-hash")

    with pytest.raises(
        RuntimeError, match="batch terminal attempt lacks returned inventories"
    ):
        batch_stages.collect_native_batch(package_root, attempt_root, output_dir=output)


def test_mode_display_requires_adaptive_result():
    """Report a missing EPR result explicitly instead of relying on assert."""
    display_type = runs._ResolvedRunDisplay  # noqa: SLF001 -- construct the presenter to test its optimized guard.
    display = display_type(
        resolved=None,
        show_details=False,
        theme="light",
        html_type=object,
        notebook_display=lambda _value: None,
        rows=[],
        benchmark={},
        result=None,
        final_pass=None,
        modes=(),
    )

    with pytest.raises(
        RuntimeError, match="adaptive EPR result is unavailable for mode display"
    ):
        display._display_mode(1)  # noqa: SLF001 -- invoke the guarded internal display stage directly.
