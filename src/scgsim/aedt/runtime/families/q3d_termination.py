"""Own available stopped-run native artifacts and their independent failures."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, TypedDict

from scgsim.aedt.io import file_sha256
from scgsim.aedt.results.convergence.q3d import read_q3d_convergence
from scgsim.aedt.results.matrices import parse_matrix_export
from scgsim.aedt.runtime.families.q3d_exports import (
    export_matrix_file,
    matrix_parse_options,
)
from scgsim.aedt.runtime.families.q3d_prepared import PreparedQ3d, prepared_q3d_result
from scgsim.aedt.specs.q3d import Q3dSpec


def termination_q3d_result(
    prepared: PreparedQ3d, control: dict[str, Any], analysis_error: Exception | None
) -> dict[str, Any]:
    """Save first and retain independently available native artifacts, never physics completion."""
    spec = prepared.request.parse()
    if not isinstance(spec, Q3dSpec):
        raise TypeError("bound Q3D request did not retain a Q3D spec")
    run_control = spec.run_control
    if run_control is None:
        raise ValueError("solving requires an explicit complete run_control")
    run_dir = prepared.request.workspace
    result = prepared_q3d_result(prepared)
    result["solver_invoked"] = True
    result["save"] = _SaveObservation(ok=False, attempted=True)
    workflow: dict[str, Any] = {
        "control": control,
        "mode": (control.get("intent") or {}).get("action"),
        "analysis": control["analysis_call"],
        "primary_error": (
            f"{type(analysis_error).__name__}: {analysis_error}"
            if analysis_error is not None
            else None
        ),
        "optional_artifacts": {},
    }
    if analysis_error is not None and getattr(analysis_error, "__notes__", None):
        workflow["primary_error_notes"] = list(analysis_error.__notes__)
    result["termination"] = workflow
    if control["analysis_call"].get("native_return") is False:
        workflow["primary_error"] = "Q3D Analyze returned False"
        if analysis_error is not None:
            workflow["post_analyze_error"] = (
                f"{type(analysis_error).__name__}: {analysis_error}"
            )
    if (
        control.get("native_settlement", {}).get("status") != "inactive"
        or control.get("request_readback", {}).get("status") == "unavailable"
    ):
        result["save"] = {
            "ok": False,
            "attempted": False,
            "reason": "native_not_confirmed_inactive",
        }
        workflow["optional_artifacts"] = {
            "status": "deferred",
            "reason": "native_not_confirmed_inactive",
        }
        return result
    # This save is distinct from the preparation save; failures cannot be padded
    # with a hash of the older project to make this attempt look successful.
    result["save"] = _save_stopped_project(prepared)

    def artifact(path: Path, call: Callable[[], Any]) -> dict[str, Any]:
        return _collect_artifact(path, call, run_dir, result["outputs"])

    for problem, stem in (
        (("C", "c"), ("AC RL", "ac_rl")) if spec.solve_ac_rl else (("C", "c"),)
    ):
        path = run_dir / "results/q3d" / f"{stem}_matrix.csv"
        item = artifact(
            path,
            lambda path=path, problem=problem: export_matrix_file(
                prepared.app, path, spec, problem
            ),
        )
        workflow["optional_artifacts"][stem] = item
        if item["status"] == "available":
            try:
                rows, native = parse_matrix_export(
                    path,
                    "Q3D",
                    problem,
                    run_control.frequency_ghz,
                    {
                        "C": {"Capacitance Matrix": "C", "Conductance Matrix": "G"},
                        "AC RL": {
                            "AC Inductance Matrix": "L",
                            "AC Resistance Matrix": "R",
                        },
                    }[problem],
                    **matrix_parse_options(spec, problem),
                )
                item["parsed"] = {"rows": rows, "native": native}
            except Exception as exc:  # noqa: BLE001 -- one native artifact failure must retain other independently available stopped-run artifacts.
                item["parse_error"] = f"{type(exc).__name__}: {exc}"
        convergence_path = run_dir / "results/q3d" / f"{stem}_convergence.prop"
        workflow["optional_artifacts"][stem + "_convergence"] = artifact(
            convergence_path,
            lambda problem=problem, convergence_path=convergence_path: (
                prepared.app.odesign.ExportConvergence(
                    run_control.setup_name,
                    "",
                    "CG" if problem == "C" else "AC RL",
                    str(convergence_path),
                    True,
                )
            ),
        )
    try:
        workflow["convergence"] = {
            "status": "available",
            "parsed": read_q3d_convergence(run_dir, spec),
        }
    except Exception as exc:  # noqa: BLE001 -- one native artifact failure must retain other independently available stopped-run artifacts.
        workflow["convergence"] = {
            "status": "unavailable",
            "parse_error": f"{type(exc).__name__}: {exc}",
        }
    return result


class _SaveObservation(TypedDict, total=False):
    """One stopped-project save attempt and its independent failure."""

    ok: bool
    attempted: bool
    error: str
    reason: str


def _collect_artifact(
    path: Path, call: Callable[[], Any], run_dir: Path, outputs: dict[str, str]
) -> dict[str, Any]:
    observation: dict[str, Any] = {"status": "unavailable"}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        call()
    except Exception as exc:  # noqa: BLE001 -- one native artifact failure must retain other independently available stopped-run artifacts.
        observation["error"] = f"{type(exc).__name__}: {exc}"
    # A native method can write a file and then throw. Keep that real file
    # independently of export/parse errors and without erasing either error.
    try:
        if path.is_file():
            relative = path.relative_to(run_dir).as_posix()
            digest = file_sha256(path)
            outputs[relative] = digest
            observation.update(status="available", path=relative, sha256=digest)
    except Exception as exc:  # noqa: BLE001 -- one native artifact failure must retain other independently available stopped-run artifacts.
        observation["identity_error"] = f"{type(exc).__name__}: {exc}"
    return observation


def _save_stopped_project(prepared: PreparedQ3d) -> _SaveObservation:
    """Retain this save attempt independently of the earlier preparation save."""
    observation = _SaveObservation(ok=False, attempted=True)
    try:
        if (
            prepared.app.save_project() is not True
            or not prepared.project_path.is_file()
        ):
            raise RuntimeError("Q3D stopped project save did not succeed")
        observation["ok"] = True
    except Exception as exc:  # noqa: BLE001 -- retain native save failure alongside independently available artifacts.
        observation["error"] = f"{type(exc).__name__}: {exc}"
    return observation
