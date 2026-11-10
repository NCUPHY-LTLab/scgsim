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

from .driver import ProbeTimeLimitError, measure_bounding_box
from .errors import MeshRepairError
from .features import SidecarFeatures, load_features, read_features
from .fine_structure import check_fine_structure
from .mesher import MesherOptions
from .recipes import FeatureRecipe, SizingRecipe
from .repair import (
    STOP_CLEAN,
    STOP_TIME,
    Budgets,
    RepairInputs,
    RepairOutcome,
    RepairPolicy,
    repair_mesh,
    unmeshed_outcome,
)
from .report import (
    QualityReport,
    RunSettings,
    build_report,
    place_lines,
    write_report,
)
from .stop_places import StopPlace, stop_places

_WORK_DIR_PREFIX = ".meshing-work-"


@dataclass(frozen=True)
class MeshResult:
    """A mesh that met the threshold, where it and its report are, and the report itself."""

    mesh: Path  # Gmsh MSH 4.1 with the XAO's physical groups
    report_path: Path  # JSON quality report
    summary_path: Path  # text summary of the report
    report: QualityReport


def mesh_sgb(  # noqa: PLR0913 -- the three paths, the required threshold and the four policy objects
    xao: Path | str,
    sidecar: Path | str | Mapping[str, Any],
    output_dir: Path | str,
    *,
    threshold: float,
    recipe: SizingRecipe = FeatureRecipe(),  # noqa: B008 -- a frozen dataclass is a safe default
    repair: RepairPolicy = RepairPolicy(),  # noqa: B008 -- a frozen dataclass is a safe default
    budgets: Budgets = Budgets(),  # noqa: B008 -- a frozen dataclass is a safe default
    mesher: MesherOptions = MesherOptions(),  # noqa: B008 -- a frozen dataclass is a safe default
) -> MeshResult:
    """Mesh the SGB model in ``xao`` so that no tetrahedron is below ``threshold``.

    ``sidecar`` is the ``scgsim.sgb.sidecar.v2`` file next to the XAO, or its decoded content.  The mesh
    ``<xao stem>.msh`` and the report ``<xao stem>.quality.json`` / ``.quality.txt`` are written to
    ``output_dir``.  ``threshold`` has no default: it is the element-quality bound the caller requires.

    The pipeline is the same for every recipe: ``recipe`` chooses the initial sizes, ``repair`` how the
    loop refines around bad elements, ``budgets`` when it gives up (``Budgets(max_rounds=0)`` meshes once
    without repair; the threshold still applies), and ``mesher`` the Gmsh algorithms and the order of the
    delivered mesh.  The time budget is measured from the start of this call and scales with the model's
    footprint (its x by y extent, vacuum included): see :class:`~scgsim.meshing.repair.Budgets`.  Reading
    that extent must finish within the unscaled limit, since the scaled one is not known before.  The
    threshold and the report's counts refer to the linear mesh: with ``mesher.geometry_order`` above 1
    the accepted mesh is raised to that order afterwards, in the worker that still has the CAD, and the
    file at ``<xao stem>.msh`` is the elevated one.

    Two things are the caller's to keep, and are not checked:

    * ``xao`` and ``sidecar`` are the pair one SGB build wrote (``SgbArtifacts.xao`` and ``.sidecar``).
      The module trusts SGB's output; a sidecar of another build binds the wrong tags and still meshes.
    * One call at a time per output target (``output_dir`` and XAO stem).  Calls with different stems
      may share ``output_dir`` and run at once; two at once for the same target may leave the mesh of one
      beside the report of the other.

    Raises :class:`~scgsim.meshing.errors.MeshRepairError` carrying the report when the loop stops
    without an accepted mesh (rounds spent, a mesh over the tetrahedron budget even when clean, the time
    limit reached, a bad count that stopped falling, a clean mesh whose order could not be raised, or a
    mesher crash).  The report is written then too, and the last mesh is kept as ``<stem>.rejected.msh``
    when a round made one.  The report and the error message then say where the remaining bad elements are,
    in micrometres, and which fine edges and faces the meshing-stage check (:func:`check_fine_structure`)
    records at those places, with their owner and fix.
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
        # The scaled limit needs the footprint the probe reads, so the probe itself gets the unscaled
        # limit: the smallest any model has.
        unscaled = budgets.time_budget(None)
        try:
            bounding_box = measure_bounding_box(
                xao, work_dir, started + unscaled.limit_seconds - time.perf_counter()
            )
        except ProbeTimeLimitError:
            time_budget, outcome = unscaled, unmeshed_outcome(STOP_TIME, mesher)
        else:
            time_budget = budgets.time_budget(bounding_box.footprint_mm2)
            outcome = repair_mesh(
                RepairInputs(
                    xao=xao,
                    sidecar=sidecar_path,
                    initial=recipe.size_requests(features, bounding_box),
                    threshold=threshold,
                    policy=repair,
                    budgets=budgets,
                    mesher=mesher,
                    work_dir=work_dir,
                    deadline=started + time_budget.limit_seconds,
                )
            )
        report = build_report(
            outcome,
            RunSettings(threshold, recipe, repair, mesher, budgets, time_budget),
            time.perf_counter() - started,
            _places_of_stop(outcome, sidecar_path, threshold, repair),
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


def _places_of_stop(
    outcome: RepairOutcome, sidecar: Path, threshold: float, policy: RepairPolicy
) -> list[StopPlace]:
    """Where the bad elements of a stop are, with the fine structure the sidecar records there; none for a clean stop."""
    if outcome.stop == STOP_CLEAN or outcome.last_audit is None:
        return []
    fine = check_fine_structure(sidecar)
    return stop_places(
        outcome.last_audit.bad,
        outcome.local_sizes,
        threshold,
        policy,
        fine.findings,
    )


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
        where = "\n".join(place_lines(report))
        raise MeshRepairError(
            f"meshing stopped without a clean mesh ({outcome.stop}); see {files.report}."
            + (f"\n{where}" if where else ""),
            dict(report),
        )
    # A clean stop follows a round that made a mesh, and one that also made the elevated mesh when the
    # mesher options asked for a higher order.
    delivered = outcome.elevated_mesh or outcome.last_mesh
    assert delivered is not None
    files.rejected.unlink(missing_ok=True)
    shutil.move(delivered, files.mesh)
    return MeshResult(files.mesh, files.report, files.summary, report)
