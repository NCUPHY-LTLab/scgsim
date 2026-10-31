"""Run one prepared AEDT handoff from start to finish.

This is the only code that opens and closes an AEDT Desktop for a handoff. A run
has four steps, and ``_execute`` reads in that order:

1. Verify the handoff (``handoff_checks.py``). A failure here raises and leaves the
   receipt at ``not_run``.
2. Mark the receipt as running.
3. Open one Desktop and run the solver family. A failure here becomes a failed
   receipt, not an exception.
4. Always: release the Desktop, record the outcome and write the final receipt.

What goes in the receipt is decided in ``receipts.py``.
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scgsim.aedt._io import write_json
from scgsim.aedt.epr.geometry import precompute_inset_surfaces
from scgsim.aedt.epr.workflow import (
    analyze_saved_epr,
    prepare_epr_hfss,
    prepared_epr_result,
    solve_and_export_epr,
)
from scgsim.aedt.preparation.planar import validate_geometry_workers
from scgsim.aedt.results.epr import seal_saved_solution
from scgsim.aedt.results.provenance import runtime_source_identity
from scgsim.aedt.runtime._timing import elapsed_seconds
from scgsim.aedt.runtime.families.hfss import (
    prepare_body_eigenmode,
    prepare_hfss,
    prepared_body_eigenmode_result,
    run_hfss,
    solve_and_export_body_eigenmode,
)
from scgsim.aedt.runtime.families.q2d import prepare_q2d, prepared_q2d_result, run_q2d
from scgsim.aedt.runtime.families.q3d import prepare_q3d, prepared_q3d_result, run_q3d
from scgsim.aedt.runtime.handoff_checks import (
    VerifiedHandoff,
    load_verified_handoff,
    verify_prepared_hashes,
)
from scgsim.aedt.runtime.native.common import (
    owned_application_constructor,
    pyaedt_version,
)
from scgsim.aedt.runtime.receipts import (
    mark_running,
    read_physics_warnings,
    record_result_before_release,
    record_saved_project,
    record_solved_result,
    utc_now,
)
from scgsim.aedt.specs.common import LOCKED_PYAEDT, REQUIRED_AEDT_VERSION
from scgsim.aedt.specs.hfss_prepared import (
    HfssDrivenGeometrySpec,
    HfssEprAnalysisSpec,
    HfssEprSpec,
)
from scgsim.aedt.specs.parse import AedtSpec
from scgsim.aedt.specs.q2d import Q2dSpec
from scgsim.aedt.specs.q3d import Q3dSpec

STATUS_COMPLETED = "completed"
STATUS_EPR_ANALYSIS_COMPLETED = "epr_analysis_completed"
STATUS_FAILED = "failed"
STATUS_PREPARED_ONLY = "native_preparation_only"
_SUCCESS_STATUSES = {
    STATUS_COMPLETED,
    STATUS_EPR_ANALYSIS_COMPLETED,
    STATUS_PREPARED_ONLY,
}

_EPR_SPECS = (HfssEprAnalysisSpec, HfssEprSpec)
# Body construction attaches source attribution to its exceptions as notes.
_SPECS_WITH_FAILURE_NOTES = (Q3dSpec, HfssDrivenGeometrySpec, HfssEprSpec)


@dataclass(frozen=True)
class _Options:
    """The command-line choices that change how a handoff runs."""

    prepare_only: bool = False
    analyze_epr: bool = False
    geometry_workers: int | None = None
    cores: int | None = None
    ram_limit_percent: int | None = None


@dataclass
class _Attempt:
    """What the run has produced so far; step 4 reads it however step 3 ended."""

    desktop: Any | None = None
    status: str = STATUS_FAILED
    failure: str | None = None
    result: dict[str, Any] | None = None
    epr_solver_attempted: bool = False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one prepared SCGSim AEDT handoff")
    parser.add_argument("--handoff", required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Open local AEDT and solve the prepared handoff",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Create and verify an HFSS 3D or Q3D native model without solving it",
    )
    parser.add_argument(
        "--analyze-epr",
        action="store_true",
        help="Analyze the sealed saved-copy EPR request without solving",
    )
    parser.add_argument(
        "--geometry-workers",
        type=int,
        help="Override pure EPR inset geometry workers before AEDT launch",
    )
    parser.add_argument("--cores", type=int, help="Override local solve core count")
    parser.add_argument(
        "--ram-limit-percent",
        type=int,
        help="Override local solve RAM Limit Percentage",
    )
    args = parser.parse_args(argv)
    metadata_path = Path(args.handoff).resolve()
    if not metadata_path.is_file():
        raise FileNotFoundError(f"handoff metadata is missing: {metadata_path}")
    if not args.execute:
        raise RuntimeError(
            "prepared handoff is not executed; use run_aedt.sh or pass --execute explicitly"
        )
    return _execute(
        metadata_path,
        _Options(
            prepare_only=args.prepare_only,
            analyze_epr=args.analyze_epr,
            geometry_workers=args.geometry_workers,
            cores=args.cores,
            ram_limit_percent=args.ram_limit_percent,
        ),
    )


def _execute(metadata_path: Path, options: _Options) -> int:
    os.chdir(metadata_path.parent.parent)
    handoff = load_verified_handoff(
        metadata_path,
        prepare_only=options.prepare_only,
        analyze_epr=options.analyze_epr,
        cores=options.cores,
        ram_limit_percent=options.ram_limit_percent,
    )
    _warn_if_thin_film_epr_unvalidated(handoff.spec)

    started = time.perf_counter()
    mark_running(
        handoff,
        prepare_only=options.prepare_only,
        analyze_epr=options.analyze_epr,
        resources_overridden=(
            options.cores is not None or options.ram_limit_percent is not None
        ),
    )
    write_json(handoff.receipt_path, handoff.receipt)

    attempt = _Attempt()
    try:
        _run_on_owned_desktop(handoff, attempt, options)
    except Exception as exc:  # noqa: BLE001 -- the receipt must record any solver failure.
        attempt.failure = _describe_failure(exc, handoff.spec)
    finally:
        _close_transaction(handoff, attempt, options, started)
    return 0 if attempt.status in _SUCCESS_STATUSES else 1


def _warn_if_thin_film_epr_unvalidated(spec: AedtSpec) -> None:
    if not isinstance(spec, _EPR_SPECS):
        return
    request = spec.epr_request
    if spec.geometry.modeling != "thin_film" or request is None:
        return
    selected = request.surface_contribution_ids
    if any(
        contribution.interface_kind != "MM"
        and (selected is None or contribution.contribution_id in selected)
        for contribution in spec.geometry.contributions
    ):
        warnings.warn(
            "ThinFilm Surface-EPR uses FacePrimitive.normal as a geometric "
            "direction, not native calculator-side readback. The "
            "Surface/Adjacent mapping is empirical for pinned PyAEDT 1.3.0 / "
            "AEDT 2024.2 and may select the wrong material side; ThinFilm "
            "MA/MS participation and loss are not physically validated.",
            UserWarning,
            stacklevel=3,
        )


# Step 3: run the solver family on one owned Desktop.


def _run_on_owned_desktop(
    handoff: VerifiedHandoff, attempt: _Attempt, options: _Options
) -> None:
    spec = handoff.spec
    receipt = handoff.receipt
    receipt["mode"] = spec.mode
    verify_prepared_hashes(handoff.metadata, handoff.spec_path, spec)
    receipt["runtime_source"] = runtime_source_identity()
    if pyaedt_version() != LOCKED_PYAEDT:
        raise RuntimeError("PyAEDT lock mismatch")
    inset_plan = _precompute_epr_insets(handoff, options)

    # Imported here so that preparation and result reading work without PyAEDT.
    from ansys.aedt.core import Desktop

    desktop_started = time.perf_counter()
    # Assign before the version check, so that a wrong version still gets released.
    attempt.desktop = Desktop(
        version=REQUIRED_AEDT_VERSION,
        non_graphical=True,
        new_desktop=True,
        close_on_exit=False,
    )
    if attempt.desktop.aedt_version_id != REQUIRED_AEDT_VERSION:
        raise RuntimeError(
            f"AEDT version mismatch: {attempt.desktop.aedt_version_id!r}"
        )
    if isinstance(spec, _EPR_SPECS):
        receipt.setdefault("timings", {})["desktop_startup_seconds"] = elapsed_seconds(
            desktop_started
        )
    _run_solver_family(handoff, attempt, options.prepare_only, inset_plan)


def _precompute_epr_insets(handoff: VerifiedHandoff, options: _Options) -> Any | None:
    """Compute inset surfaces before AEDT starts; this geometry work needs no Desktop."""
    spec = handoff.spec
    setting = handoff.metadata.get("execution", {}).get("geometry_workers")
    workers = (
        options.geometry_workers if options.geometry_workers is not None else setting
    )
    validate_geometry_workers(workers)
    if not isinstance(spec, _EPR_SPECS) or spec.epr_request is None:
        return None
    started = time.perf_counter()
    inset_plan = precompute_inset_surfaces(spec.geometry, spec.epr_request, workers)
    handoff.receipt["geometry_precompute"] = {
        "seconds": elapsed_seconds(started),
        "members": len(inset_plan),
        "requested_workers": workers,
    }
    return inset_plan


def _run_solver_family(
    handoff: VerifiedHandoff,
    attempt: _Attempt,
    prepare_only: bool,
    inset_plan: Any | None,
) -> None:
    from ansys.aedt.core import Hfss, Q2d, Q3d

    spec = handoff.spec
    if isinstance(spec, HfssEprAnalysisSpec):
        attempt.result = analyze_saved_epr(
            owned_application_constructor(Hfss, attempt.desktop),
            handoff.run_dir,
            spec,
            inset_plan,
        )
        attempt.status = STATUS_EPR_ANALYSIS_COMPLETED
        return
    if isinstance(spec, HfssEprSpec):
        _run_epr_eigenmode(
            handoff,
            attempt,
            owned_application_constructor(Hfss, attempt.desktop),
            prepare_only,
            inset_plan,
        )
        return

    application: Any
    prepare: Callable[..., Any]
    prepared_result: Callable[[Any], dict[str, Any]]
    solve: Callable[..., dict[str, Any]]
    if isinstance(spec, Q3dSpec):
        application, prepare, prepared_result, solve = (
            Q3d,
            prepare_q3d,
            prepared_q3d_result,
            run_q3d,
        )
    elif isinstance(spec, Q2dSpec):
        application, prepare, prepared_result, solve = (
            Q2d,
            prepare_q2d,
            prepared_q2d_result,
            run_q2d,
        )
    else:
        application, prepare, prepared_result, solve = (
            Hfss,
            prepare_hfss,
            _prepared_hfss_result,
            run_hfss,
        )
    constructor = owned_application_constructor(application, attempt.desktop)
    if prepare_only:
        attempt.result = prepared_result(prepare(constructor, handoff.run_dir, spec))
        attempt.status = STATUS_PREPARED_ONLY
    else:
        attempt.result = solve(
            constructor,
            handoff.run_dir,
            spec,
            handoff.resources,
            handoff.receipt["resources"],
        )
        attempt.status = STATUS_COMPLETED


def _run_epr_eigenmode(
    handoff: VerifiedHandoff,
    attempt: _Attempt,
    constructor: Any,
    prepare_only: bool,
    inset_plan: Any | None,
) -> None:
    """Build and solve eigenmode; a spec without an EPR request skips the EPR steps."""
    spec = handoff.spec
    plain_eigenmode = spec.epr_request is None
    if plain_eigenmode:
        prepared = prepare_body_eigenmode(constructor, handoff.run_dir, spec)
    else:
        prepared = prepare_epr_hfss(constructor, handoff.run_dir, spec, inset_plan)
    if prepare_only:
        attempt.result = (
            prepared_body_eigenmode_result(prepared)
            if plain_eigenmode
            else prepared_epr_result(prepared)
        )
        attempt.status = STATUS_PREPARED_ONLY
        return
    # Recorded even when the solve raises, as receipt["solver_invoked"].
    attempt.epr_solver_attempted = True
    solve = solve_and_export_body_eigenmode if plain_eigenmode else solve_and_export_epr
    attempt.result = solve(prepared, handoff.resources, handoff.receipt["resources"])
    attempt.status = STATUS_COMPLETED


def _prepared_hfss_result(prepared: Any) -> dict[str, Any]:
    return {
        "outputs": {},
        "connected": {
            "aedt_version": prepared.app.desktop_class.aedt_version_id,
            "pyaedt_version": pyaedt_version(),
        },
        "project": prepared.project_path.relative_to(
            prepared.request.workspace
        ).as_posix(),
        "materials": prepared.materials,
        "region": prepared.region,
        "mesh": prepared.mesh,
        "ports": prepared.ports,
        "setup": prepared.setup,
        **({"geometry": prepared.geometry} if prepared.geometry else {}),
        "save": {"ok": True},
        "solver_invoked": False,
    }


def _describe_failure(exc: Exception, spec: AedtSpec) -> str:
    failure = f"{type(exc).__name__}: {exc}"
    notes = getattr(exc, "__notes__", ())
    if isinstance(spec, _SPECS_WITH_FAILURE_NOTES) and notes:
        failure += "\n" + "\n".join(notes)
    return failure


# Step 4: release the Desktop and write the final receipt, however step 3 ended.


def _close_transaction(
    handoff: VerifiedHandoff, attempt: _Attempt, options: _Options, started: float
) -> None:
    spec = handoff.spec
    receipt = handoff.receipt
    if attempt.result is not None:
        record_result_before_release(receipt, attempt.result, spec)
    _release_desktop(attempt, receipt)
    if attempt.failure is not None:
        attempt.status = STATUS_FAILED
        receipt["error"] = attempt.failure
    if isinstance(spec, HfssEprSpec):
        receipt["solver_invoked"] = attempt.epr_solver_attempted
    if attempt.result is not None:
        _record_after_release(handoff, attempt, options.prepare_only)
    receipt["diagnostics"] = read_physics_warnings(handoff.run_dir)
    receipt["status"] = attempt.status
    receipt["finished_at_utc"] = utc_now()
    receipt["execution_seconds"] = elapsed_seconds(started)
    write_json(handoff.receipt_path, receipt)


def _release_desktop(attempt: _Attempt, receipt: dict[str, Any]) -> None:
    if attempt.desktop is None:
        return
    started = time.perf_counter()
    try:
        released = bool(
            attempt.desktop.release_desktop(close_projects=True, close_on_exit=True)
        )
        receipt["release"] = {"ok": released}
        if not released:
            raise RuntimeError("owned AEDT Desktop release returned false")
    except Exception as exc:  # noqa: BLE001 -- the receipt records release failure.
        error = f"{type(exc).__name__}: {exc}"
        receipt["release"] = {"ok": False, "error": error}
        if attempt.failure is None:
            attempt.failure = error
    receipt.setdefault("timings", {})["release_seconds"] = elapsed_seconds(started)


def _record_after_release(
    handoff: VerifiedHandoff, attempt: _Attempt, prepare_only: bool
) -> None:
    receipt = handoff.receipt
    if prepare_only:
        # A failed release skips hashing, so "save" stays as the family reported it.
        if receipt.get("release") == {"ok": True}:
            _record_prepared_project(handoff, attempt)
    elif isinstance(handoff.spec, _EPR_SPECS):
        _record_epr_project(handoff, attempt)
    else:
        record_solved_result(receipt, attempt.result)


def _record_prepared_project(handoff: VerifiedHandoff, attempt: _Attempt) -> None:
    receipt = handoff.receipt
    try:
        if not record_saved_project(handoff.run_dir, receipt, attempt.result):
            raise RuntimeError(
                "prepared project is missing after owned Desktop release"
            )
    except Exception as exc:  # noqa: BLE001 -- keep the primary native failure.
        error = f"{type(exc).__name__}: {exc}"
        receipt["save"] = {"ok": False, "error": error}
        _fail_after_release(attempt, receipt, error)


def _record_epr_project(handoff: VerifiedHandoff, attempt: _Attempt) -> None:
    receipt = handoff.receipt
    if not record_saved_project(handoff.run_dir, receipt, attempt.result):
        _fail_after_release(
            attempt, receipt, "EPR project is missing after owned Desktop release"
        )
        return
    spec = handoff.spec
    if (
        isinstance(spec, HfssEprSpec)
        and spec.epr_request is not None
        and attempt.status == STATUS_COMPLETED
        and attempt.failure is None
        and receipt.get("release") == {"ok": True}
    ):
        _seal_saved_solution(handoff, attempt)


def _seal_saved_solution(handoff: VerifiedHandoff, attempt: _Attempt) -> None:
    """Seal saved fields for later EPR analysis; a completed EPR run requires this."""
    receipt = handoff.receipt
    spec = handoff.spec
    result = attempt.result
    try:
        sealed = seal_saved_solution(
            handoff.run_dir,
            project_name=spec.project_name,
            design_name=spec.design_name,
            setup_name=spec.run_control.setup_name,
            model_source_sha256=spec.geometry.model_sha256,
            evidence=result["saved_field_evidence"],
        )
        result["outputs"].update(
            {
                sealed["manifest"]: sealed["manifest_sha256"],
                sealed["receipt"]: sealed["receipt_sha256"],
            }
        )
        receipt["outputs"] = result["outputs"]
        receipt["saved_solution"] = sealed
    except Exception as exc:  # noqa: BLE001 -- sealing is required.
        error = f"{type(exc).__name__}: {exc}"
        receipt["saved_solution"] = {"status": "unavailable", "error": error}
        _fail_after_release(attempt, receipt, error)


def _fail_after_release(
    attempt: _Attempt, receipt: dict[str, Any], message: str
) -> None:
    """Mark the run failed, keeping any earlier failure as the reported error."""
    attempt.status = STATUS_FAILED
    if attempt.failure is None:
        attempt.failure = message
        receipt["error"] = message
