"""The receipt recorder a handoff copies into the run folder runs there without SCGSim.

The run script is executed for real with a stand-in ``palace`` executable and a ``python3`` that cannot import
SCGSim (``-I -S``: no site packages), so the copy must carry the run-folder contract it imports in the package.
No native Palace runs.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from scgsim.contracts import palace_run_folder as layout
from scgsim.palace.mesh.build import MeshBuildResult
from scgsim.palace.run import convergence_runner, returned_receipt
from scgsim.palace.run.handoff import prepare_handoff
from scgsim.palace.run.standalone_copy import standalone_text

_FAKE_PALACE = """#!/usr/bin/env bash
# A stand-in for Palace: it reports the supported release, accepts the dry run and writes the tables of an
# Eigenmode config with no lumped ports and no Surface-EPR rows.
if [ "$1" = "--version" ]; then
  printf 'Palace version: 0.18.1\\nSchema version: 1-7-0\\n'
  exit 0
fi
if [ "$1" = "--dry-run" ]; then
  exit 0
fi
for family in eig domain-E error-indicators; do
  echo "m,value" > "results/palace/$family.csv"
done
echo '{"GitTag": "v0.18.1"}' > results/palace/palace.json
"""


def _isolated_python(directory: Path) -> Path:
    """A ``python3`` on PATH that sees only the standard library."""
    directory.mkdir()
    wrapper = directory / "python3"
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" -I -S "$@"\n')
    wrapper.chmod(0o755)
    return directory


def _prepared_run(tmp_path: Path):
    """A direct-local handoff of opaque mesh records; only the files the handoff reads are real."""
    run_dir = tmp_path / "prepared"
    (run_dir / layout.METADATA_DIR).mkdir(parents=True)
    mesh = run_dir / layout.MESH
    mesh.write_bytes(b"opaque file-only transport fixture\n")
    config = run_dir / layout.CONFIG
    config.write_text(
        json.dumps(
            {
                "Problem": {"Type": "Eigenmode"},
                "Model": {"Mesh": layout.MESH},
                "Boundaries": {},
            }
        )
    )
    manifest = run_dir / layout.MESH_MANIFEST
    manifest.write_text('{"route":"B"}\n')
    provenance = run_dir / layout.route_provenance("B")
    provenance.write_text('{"route":"B"}\n')
    (run_dir / layout.INDEX_MAP).write_text('{"schema_version":3,"entries":[]}\n')
    executable = tmp_path / "palace"
    executable.write_text(_FAKE_PALACE)
    executable.chmod(0o755)
    mesh_result = MeshBuildResult(
        output_dir=run_dir,
        mesh_path=mesh,
        groups={},
        mesh_manifest_path=manifest,
        provenance_path=provenance,
        stack_path=run_dir / layout.SOURCE_STACK,
        xao_path=run_dir / "geometry.xao",
        sidecar_path=run_dir / "sidecar.json",
        quality_report_path=run_dir / layout.MESH_DIR / "quality.json",
    )
    return prepare_handoff(
        profile="direct-local",
        mesh_result=mesh_result,
        config_path=config,
        executable=str(executable),
        resources={"processes": 1, "threads": 1},
        problem="Eigenmode",
    )


@pytest.mark.parametrize(
    ("module", "constant"),
    [
        (returned_receipt, f'RETURNED_RECEIPT = "{layout.RETURNED_RECEIPT}"'),
        (convergence_runner, f'CONVERGENCE_DECISION = "{layout.CONVERGENCE_DECISION}"'),
    ],
)
def test_the_copy_imports_only_the_standard_library(module, constant):
    """Each copy names no SCGSim module and holds the contracts' text in place of their imports, once each."""
    text = standalone_text(module)
    imported = set()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "the copy has no package to import from"
            imported.add((node.module or "").split(".")[0])
    assert imported <= set(sys.stdlib_module_names) | {"__future__"}
    assert constant in text
    # Both copies carry the convergence contract after the run-folder contract it imports.
    run_folder = text.index("# ---- scgsim.contracts.palace_run_folder, embedded")
    convergence = text.index("# ---- scgsim.contracts.palace_convergence, embedded")
    assert run_folder < convergence
    assert text.count("def check_convergence(") == 1


def test_a_name_two_parts_of_a_copy_define_is_refused(tmp_path, monkeypatch):
    """A module defining a name a contract defines would silently replace it in the copy: refused."""
    module_path = tmp_path / "clashing.py"
    module_path.write_text(
        "from scgsim.contracts.palace_run_folder import CONFIG\n\n\ndef sha256():\n    return CONFIG\n",
        encoding="utf-8",
    )
    clashing = types.ModuleType("clashing")
    clashing.__file__ = str(module_path)
    with pytest.raises(
        RuntimeError, match="defines 'sha256' in scgsim.contracts.palace_run_folder"
    ):
        standalone_text(clashing)


@pytest.mark.skipif(
    sys.platform == "win32", reason="the run script is a POSIX shell script"
)
def test_the_run_script_and_the_copied_recorder_run_without_scgsim(tmp_path):
    """The script's preflight and receipt steps complete with the copy alone and record a completed run."""
    plan = _prepared_run(tmp_path)
    assert plan.returned_receipt_recorder_path.read_text() == standalone_text(
        returned_receipt
    )
    assert not (plan.run_dir / layout.CONVERGENCE_RUNNER).exists()
    python = _isolated_python(tmp_path / "bin")
    environment = {**os.environ, "PATH": f"{python}{os.pathsep}{os.environ['PATH']}"}
    completed = subprocess.run(
        ["bash", str(plan.script_path)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    receipt = json.loads(plan.returned_receipt_path.read_text())
    assert receipt["status"] == "completed"
    assert receipt["handoff_id"] == plan.handoff_id
    assert [record["path"] for record in receipt["output_files"]] == [
        layout.output_table("eig"),
        layout.output_table("domain-E"),
        layout.output_table("error-indicators"),
        layout.NATIVE_METADATA,
        "logs/palace-manual.log",
        layout.RUNTIME_COMPATIBILITY,
    ]
