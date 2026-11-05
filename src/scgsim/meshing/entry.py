"""The public entry point: mesh an SGB XAO to a quality threshold, or fail with the quality report."""

from __future__ import annotations

import json
import math
import shutil
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .driver import measure_bounding_box
from .errors import MeshRepairError
from .features import SidecarFeatures, load_features, read_features
from .policy import RuleSet
from .repair import STOP_CLEAN, Budgets, RepairInputs, RepairOutcome, repair_mesh
from .report import QualityReport, build_report, write_report

_WORK_DIR_PREFIX = ".meshing-work-"


@dataclass(frozen=True)
class MeshResult:
    """A mesh that met the threshold, where it and its report are, and the report itself."""

    mesh: Path  # Gmsh MSH 4.1 with the XAO's physical groups
    report_path: Path  # JSON quality report
    summary_path: Path  # text summary of the report
    report: QualityReport


def mesh_sgb(  # noqa: PLR0913 -- the three paths, the required threshold and the two policy objects
    xao: Path | str,
    sidecar: Path | str | Mapping[str, Any],
    output_dir: Path | str,
    *,
    threshold: float,
    ruleset: RuleSet = RuleSet(),  # noqa: B008 -- a frozen dataclass is a safe default
    budgets: Budgets = Budgets(),  # noqa: B008 -- a frozen dataclass is a safe default
) -> MeshResult:
    """Mesh the SGB model in ``xao`` so that no tetrahedron is below ``threshold``.

    ``sidecar`` is the ``scgsim.sgb.sidecar.v2`` file next to the XAO, or its decoded content.  The mesh
    ``<xao stem>.msh`` and the report ``<xao stem>.quality.json`` / ``.quality.txt`` are written to
    ``output_dir``.  ``threshold`` has no default: it is the element-quality bound the caller requires.

    Raises :class:`~scgsim.meshing.errors.MeshRepairError` carrying the report when the loop stops
    before the mesh is clean (rounds or tetrahedron budget spent, bad elements the rules cannot refine,
    or a mesher crash).  The report is written then too, and the last mesh is kept as
    ``<stem>.rejected.msh`` when a round made one.
    """
    if not math.isfinite(threshold):
        raise ValueError(f"threshold must be a finite number, got {threshold!r}.")
    started = time.perf_counter()
    xao = Path(xao)
    files = _Files(Path(output_dir), xao.stem)
    files.directory.mkdir(parents=True, exist_ok=True)
    # One work directory per call, so calls sharing an output directory never read or delete each
    # other's files; it sits in the output directory so the final mesh is moved, not copied.
    work_dir = Path(tempfile.mkdtemp(prefix=_WORK_DIR_PREFIX, dir=files.directory))
    try:
        sidecar_path, features = _resolve_sidecar(sidecar, work_dir)
        outcome = repair_mesh(
            RepairInputs(
                xao=xao,
                sidecar=sidecar_path,
                features=features,
                bounding_box=measure_bounding_box(xao, work_dir),
                threshold=threshold,
                ruleset=ruleset,
                budgets=budgets,
                work_dir=work_dir,
            )
        )
        report = build_report(
            outcome,
            threshold=threshold,
            ruleset=ruleset,
            budgets=budgets,
            total_seconds=time.perf_counter() - started,
        )
        write_report(report, files.report, files.summary)
        return _deliver(outcome, report, files)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


@dataclass(frozen=True)
class _Files:
    """Where a run leaves its files: all in one directory, named after the XAO."""

    directory: Path
    stem: str

    @property
    def mesh(self) -> Path:
        return self.directory / f"{self.stem}.msh"

    @property
    def rejected(self) -> Path:
        return self.directory / f"{self.stem}.rejected.msh"

    @property
    def report(self) -> Path:
        return self.directory / f"{self.stem}.quality.json"

    @property
    def summary(self) -> Path:
        return self.directory / f"{self.stem}.quality.txt"


def _resolve_sidecar(
    sidecar: Path | str | Mapping[str, Any], work_dir: Path
) -> tuple[Path, SidecarFeatures]:
    """The sidecar as a file the worker can read and as parsed features."""
    if isinstance(sidecar, Mapping):
        path = work_dir / "sidecar.json"
        path.write_text(json.dumps(sidecar), encoding="utf-8")
        return path, read_features(sidecar)
    return Path(sidecar), load_features(Path(sidecar))


def _deliver(
    outcome: RepairOutcome, report: QualityReport, files: _Files
) -> MeshResult:
    """Move the last mesh into place and return it, or raise with the report when the loop failed.

    The report has just been rewritten, so a mesh an earlier run left under the other name is removed:
    the directory never holds a mesh that the report beside it does not describe.
    """
    if outcome.stop != STOP_CLEAN:
        files.mesh.unlink(missing_ok=True)
        files.rejected.unlink(missing_ok=True)
        if outcome.last_mesh is not None:
            shutil.move(outcome.last_mesh, files.rejected)
        raise MeshRepairError(
            f"meshing stopped without a clean mesh ({outcome.stop}); see {files.report}.",
            dict(report),
        )
    assert (
        outcome.last_mesh is not None
    )  # a clean stop follows a round that made a mesh
    files.rejected.unlink(missing_ok=True)
    shutil.move(outcome.last_mesh, files.mesh)
    return MeshResult(files.mesh, files.report, files.summary, report)
