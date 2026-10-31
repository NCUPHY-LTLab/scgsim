"""Offline strict AEDT result resolution.

``resolve_results`` returns a ``ResolvedRun`` only if the run receipt proves every
artifact: the preparation cohort, the native readback for the run's mode, and the
SHA-256 of every output file. It never opens AEDT.

Owns: the ``ResolvedRun`` handle, the order of checks in ``resolve_results`` and
the per-mode lists of files a run must contain. It does not own the evidence checks
themselves: ``cohort``, ``hfss_readback``, ``hfss_pec``, ``body_first_readback``,
``q2d_readback`` and ``q3d_readback`` do.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from scgsim.aedt._io import file_sha256, read_json
from scgsim.aedt.epr.models import EprResult
from scgsim.aedt.preparation.cohort import validate_geometry_source
from scgsim.aedt.results.benchmark import read_simulation_benchmark
from scgsim.aedt.results.body_first_readback import validate_lumped_readback
from scgsim.aedt.results.cohort import AEDT_MODES, validate_completion_cohort
from scgsim.aedt.results.convergence.hfss import read_hfss_convergence
from scgsim.aedt.results.epr import resolve_epr_result, resolve_saved_solution
from scgsim.aedt.results.hfss_pec import validate_hfss_pec_bindings
from scgsim.aedt.results.hfss_readback import (
    require_export_convergence_source,
    validate_driven_readback,
    validate_eigenmode_readback,
    validate_hfss_setup_and_convergence,
)
from scgsim.aedt.results.matrices import parse_matrix_export, read_q2d_rlgc_matrix
from scgsim.aedt.results.provenance import RECEIPT_V3
from scgsim.aedt.results.q2d_readback import validate_q2d_readback
from scgsim.aedt.results.q3d_readback import validate_q3d_readback
from scgsim.aedt.results.readback_common import (
    contained_in_handoff_root,
    verified_output,
)
from scgsim.aedt.runtime.benchmark import BENCHMARK_RELATIVE
from scgsim.aedt.specs.hfss import HfssDrivenSpec, HfssEigenmodeSpec
from scgsim.aedt.specs.hfss_prepared import HfssDrivenGeometrySpec, HfssEprSpec
from scgsim.aedt.specs.parse import parse_aedt_spec
from scgsim.aedt.specs.q2d import Q2dSpec
from scgsim.aedt.specs.q3d import Q3dSpec

_Q2D_SUPERSEDED_MATRICES = (
    "results/q2d/cg_matrix.csv",
    "results/q2d/rl_matrix.csv",
    "results/q2d/matrices.csv",
)
_Q3D_AC_RL_ARTIFACTS = ("results/q3d/ac_rl_matrix.csv", "results/q3d/matrices.csv")
_CAPACITANCE_AND_CONDUCTANCE = {"Capacitance Matrix": "C", "Conductance Matrix": "G"}
_EPR_SAVED_MANIFEST = "saved-solution-manifest.json"
_EPR_SAVED_RECEIPT = "metadata/saved-solution-receipt.json"


@dataclass(frozen=True)
class ResolvedRun:
    """Canonical result paths verified against the completed run receipt."""

    mode: Literal["terminal", "modal", "eigenmode", "q3d", "q2d"]
    project_path: Path
    primary_csv: Path
    touchstone_path: Path | None
    provenance_path: Path | None
    receipt_path: Path
    convergence: dict[str, Any] | None = None
    benchmark_path: Path | None = None
    _output_hashes: tuple[tuple[str, str], ...] = ()
    _spec_sha256: str | None = None
    _execution_seconds: float | None = None
    _setup_name: str | None = None
    _epr_requested: bool = False
    _epr_result: EprResult | None = None
    _lumped_boundaries: tuple[dict[str, Any], ...] = ()

    def _verified_output(self, path: Path) -> Path:
        """Re-check ``path`` against the receipt hash at the moment it is read."""
        root = self.receipt_path.parent.parent
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise RuntimeError("resolved output escapes the run directory") from exc
        return verified_output(root, relative, dict(self._output_hashes))

    def _verified_spec(self) -> Path:
        root = self.receipt_path.parent.parent
        return verified_output(
            root, "aedt_spec.json", {"spec_sha256": self._spec_sha256}, "spec_sha256"
        )

    def _parsed_spec(self) -> Any:
        root = self.receipt_path.parent.parent
        return parse_aedt_spec(read_json(self._verified_spec()), base_dir=root)

    def epr_result(self) -> EprResult | None:
        """Return the immutable, receipt-bound adaptive EPR record, if requested."""

        if not self._epr_requested:
            return None
        if self._epr_result is None:
            raise RuntimeError("requested EPR result was not resolved")
        return self._epr_result

    def show_all_results(
        self, *, show_details: bool = False, theme: str = "light"
    ) -> None:
        """Display verified primary, benchmark, and requested EPR evidence."""

        if type(show_details) is not bool:
            raise TypeError("show_details must be a bool")
        from scgsim.presentation.notebook import checked_theme

        checked_theme(theme)
        from scgsim.aedt.presentation.runs import display_resolved_run

        display_resolved_run(self, show_details=show_details, theme=theme)

    def physics_results(self) -> tuple[dict[str, str], ...]:
        """Return the verified primary result as string-valued rows."""

        self._verified_output(self.primary_csv)

        if self.mode == "q2d":
            spec = self._parsed_spec()
            if not isinstance(spec, Q2dSpec):
                raise RuntimeError("resolved Q2D result has a non-Q2D spec")
            rows, _ = read_q2d_rlgc_matrix(self.primary_csv, spec)
            return _as_text_rows(rows)

        if self.mode == "q3d":
            spec = self._parsed_spec()
            if not isinstance(spec, Q3dSpec):
                raise RuntimeError("resolved Q3D result has a non-Q3D spec")
            if not spec.solve_ac_rl:
                rows, _ = parse_matrix_export(
                    self.primary_csv,
                    "Q3D",
                    "C",
                    spec.run_control.frequency_ghz,
                    _CAPACITANCE_AND_CONDUCTANCE,
                )
                return _as_text_rows(rows)

        with self.primary_csv.open(newline="", encoding="utf-8-sig") as stream:
            return tuple(dict(row) for row in csv.DictReader(stream))

    def simulation_benchmark(self) -> dict[str, Any]:
        """Return the receipt-bound execution summary for notebook display."""

        self._verified_output(self.project_path)
        self._verified_output(self.primary_csv)
        if self._execution_seconds is None:
            raise RuntimeError("resolved execution duration is unavailable")
        return {
            "mode": self.mode,
            "execution_seconds": self._execution_seconds,
            "project_bytes": self.project_path.stat().st_size,
            "primary_csv_bytes": self.primary_csv.stat().st_size,
            "benchmark": (
                {"status": "not_recorded"}
                if self.benchmark_path is None
                else read_simulation_benchmark(
                    self._verified_output(self.benchmark_path),
                    self._setup_name,
                )
            ),
        }

    def show_simulation_benchmark(
        self, *, show_details: bool = False, theme: str = "light"
    ) -> AedtBenchmarkReport:
        """Return a notebook-displayable, offline view of native profile evidence."""
        from scgsim.presentation.notebook import checked_theme

        checked_theme(theme)
        return AedtBenchmarkReport(self.simulation_benchmark(), show_details, theme)


def _as_text_rows(rows: Any) -> tuple[dict[str, str], ...]:
    return tuple({key: str(value) for key, value in row.items()} for row in rows)


@dataclass(frozen=True)
class AedtBenchmarkReport:
    """Offline native benchmark view; ``data`` retains the complete profile tree."""

    data: dict[str, Any]
    show_details: bool = False
    theme: str = "light"

    def _ipython_display_(self) -> None:
        from scgsim.aedt.presentation.benchmark import display_benchmark

        display_benchmark(self.data, show_details=self.show_details, theme=self.theme)


# ---------------------------------------------------------------------------
# resolve_results: the order of checks decides which error a user sees first.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CompletedRun:
    """A receipt that passed every evidence check, before output files are listed."""

    root: Path
    receipt_path: Path
    receipt: dict[str, Any]
    spec: Any
    outputs: dict[str, Any]
    project_relative: str
    project: Path
    benchmark_path: Path | None

    def expected_outputs(self, *relative_paths: str) -> set[str]:
        """The exact set of files the output manifest must list for this mode."""
        expected = {self.project_relative, *relative_paths}
        if self.benchmark_path is not None:
            expected.add(BENCHMARK_RELATIVE)
        return expected

    def require_manifest(self, expected: set[str], label: str) -> None:
        if set(self.outputs) != expected:
            raise RuntimeError(f"{label} output manifest is not canonical")

    def verified(self, relative: str) -> Path:
        return verified_output(self.root, relative, self.outputs)

    def resolved(
        self,
        mode: str,
        primary_csv: Path,
        *,
        touchstone_path: Path | None = None,
        provenance_path: Path | None = None,
        epr_result: EprResult | None = None,
    ) -> ResolvedRun:
        return ResolvedRun(
            mode,
            self.project,
            primary_csv,
            touchstone_path,
            provenance_path,
            self.receipt_path,
            convergence=self.receipt["convergence"],
            benchmark_path=self.benchmark_path,
            **_captured_fields(
                self.receipt, self.outputs, self.spec, epr_result=epr_result
            ),
        )


def resolve_results(run_dir: str | Path) -> ResolvedRun:
    """Return results only when a complete exact receipt proves every artifact."""
    root = Path(run_dir).resolve()
    receipt_path, receipt = _read_completed_receipt(root)
    spec = _read_receipt_spec(root, receipt)
    validate_completion_cohort(root, receipt, spec)
    _check_source_members(root, receipt["source"], spec)
    if receipt.get("mode") not in AEDT_MODES or spec.mode != receipt.get("mode"):
        raise RuntimeError("AEDT receipt mode is invalid")
    _check_native_readback(root, receipt, spec)
    run = _check_outputs_and_project(root, receipt_path, receipt, spec)
    return _resolve_mode(run)


def _read_completed_receipt(root: Path) -> tuple[Path, dict[str, Any]]:
    """Read the receipt; require a completed run with a saved project and release."""
    receipt_path = contained_in_handoff_root(root, "metadata/aedt_run_receipt.json")
    if not receipt_path.is_file():
        raise FileNotFoundError(f"run receipt is missing: {receipt_path}")
    receipt = read_json(receipt_path)
    if not isinstance(receipt, dict):
        raise RuntimeError("handoff receipt schema is invalid")
    if receipt.get("schema_version") != RECEIPT_V3:
        raise RuntimeError(
            f"handoff receipt schema {receipt.get('schema_version')!r} is "
            f"unsupported; only {RECEIPT_V3!r} is supported"
        )
    if receipt.get("status") != "completed":
        raise RuntimeError("handoff has not completed successfully")
    save = receipt.get("save")
    if (
        not isinstance(save, dict)
        or save.get("ok") is not True
        or not isinstance(save.get("project_sha256"), str)
    ):
        raise RuntimeError("completed receipt lacks a successful exact project save")
    if receipt.get("release") != {"ok": True}:
        raise RuntimeError("completed receipt lacks a successful owned Desktop release")
    return receipt_path, receipt


def _read_receipt_spec(root: Path, receipt: dict[str, Any]) -> Any:
    source = receipt.get("source")
    if not isinstance(source, dict) or source.get("spec") != "aedt_spec.json":
        raise RuntimeError("receipt source paths are not canonical")
    spec_path = verified_output(root, "aedt_spec.json", source, "spec_sha256")
    return parse_aedt_spec(read_json(spec_path), base_dir=root)


def _check_source_members(root: Path, source: dict[str, Any], spec: Any) -> None:
    """The receipt's ``source`` lists exactly the members its spec family allows."""
    if isinstance(spec, (HfssEprSpec, HfssDrivenGeometrySpec)):
        expected_source = {
            "spec",
            "spec_sha256",
            "planar_source_sha256",
            "planar_model_sha256",
        }
        if set(source) != expected_source:
            raise RuntimeError("body-first Eigenmode source members are not canonical")
        if (
            source["planar_source_sha256"] != spec.geometry.source_sha256
            or source["planar_model_sha256"] != spec.geometry.model_sha256
        ):
            raise RuntimeError("body-first Eigenmode source identity differs")
    elif isinstance(spec, (Q2dSpec, Q3dSpec)):
        if set(source) != {"spec", "spec_sha256"}:
            raise RuntimeError(
                "Q2D receipt must not contain a GDS source"
                if isinstance(spec, Q2dSpec)
                else "Q3D body receipt must not contain an imported GDS source"
            )
        if isinstance(spec, Q3dSpec):
            validate_geometry_source(root, source, spec)
    else:
        if source.get("gds") != "geometry/design.gds":
            raise RuntimeError("receipt GDS path is not canonical")
        verified_output(root, "geometry/design.gds", source, "gds_sha256")


def _check_native_readback(root: Path, receipt: dict[str, Any], spec: Any) -> None:
    if isinstance(spec, HfssEprSpec):
        validate_lumped_readback(receipt, spec.geometry)
        if spec.lumped_rlcs:
            validate_eigenmode_readback(receipt, spec)
        if receipt.get("workflow_status") != "completed":
            raise RuntimeError("body-first Eigenmode receipt is not a completed solve")
        if not isinstance(receipt.get("geometry"), dict) or not isinstance(
            receipt.get("setup"), dict
        ):
            raise RuntimeError("body-first Eigenmode native readback is incomplete")
    else:
        _validate_readback(root, receipt, spec)
    if isinstance(spec, (HfssDrivenSpec, HfssEigenmodeSpec)):
        validate_hfss_pec_bindings(receipt, spec)


def _check_outputs_and_project(
    root: Path, receipt_path: Path, receipt: dict[str, Any], spec: Any
) -> _CompletedRun:
    """Check the output manifest header, the benchmark binding and the saved project."""
    outputs = receipt.get("outputs")
    if not isinstance(outputs, dict):
        raise TypeError("completed receipt has no output hash manifest")
    benchmark_path = _resolved_benchmark(
        root, receipt, outputs, spec.run_control.setup_name
    )
    project_relative = f"{spec.project_name}.aedt"
    if receipt.get("project") != project_relative:
        raise RuntimeError("completed receipt project path is not canonical")
    project = verified_output(root, project_relative, outputs, project_relative)
    if receipt["save"]["project_sha256"] != outputs.get(project_relative):
        raise RuntimeError("saved project hash does not match output manifest")
    return _CompletedRun(
        root,
        receipt_path,
        receipt,
        spec,
        outputs,
        project_relative,
        project,
        benchmark_path,
    )


def _resolve_mode(run: _CompletedRun) -> ResolvedRun:
    mode = run.receipt["mode"]
    if mode == "q2d":
        return _resolve_q2d(run)
    if mode == "q3d":
        return _resolve_q3d(run)
    if mode == "eigenmode":
        if isinstance(run.spec, HfssEprSpec):
            return _resolve_epr_eigenmode(run)
        return _resolve_eigenmode(run)
    return _resolve_driven(run)


def _resolve_q2d(run: _CompletedRun) -> ResolvedRun:
    if any(
        contained_in_handoff_root(run.root, relative).exists()
        for relative in _Q2D_SUPERSEDED_MATRICES
    ):
        raise RuntimeError("Q2D run contains superseded matrix artifacts")
    expected = run.expected_outputs("results/q2d/rlgc_matrix.csv")
    run.require_manifest(expected, "Q2D")
    return run.resolved("q2d", run.verified("results/q2d/rlgc_matrix.csv"))


def _resolve_q3d(run: _CompletedRun) -> ResolvedRun:
    primary = "results/q3d/c_matrix.csv"
    expected = run.expected_outputs(primary)
    if run.spec.solve_ac_rl:
        expected.update(_Q3D_AC_RL_ARTIFACTS)
        primary = "results/q3d/matrices.csv"
    elif any(
        contained_in_handoff_root(run.root, relative).exists()
        for relative in _Q3D_AC_RL_ARTIFACTS
    ):
        raise RuntimeError("Q3D capacitance-only run contains AC-RL artifacts")
    run.require_manifest(expected, "Q3D")
    return run.resolved("q3d", run.verified(primary))


def _resolve_eigenmode(run: _CompletedRun) -> ResolvedRun:
    expected = run.expected_outputs(
        "results/eigenmode/eigenmodes.csv", "results/eigenmode/eigenmodes.eig"
    )
    require_export_convergence_source(run.receipt)
    expected.add("results/eigenmode/adaptive-convergence.prop")
    run.require_manifest(expected, "Eigenmode")
    run.verified("results/eigenmode/adaptive-convergence.prop")
    return run.resolved(
        "eigenmode",
        run.verified("results/eigenmode/eigenmodes.csv"),
        provenance_path=run.verified("results/eigenmode/eigenmodes.eig"),
    )


def _resolve_epr_eigenmode(run: _CompletedRun) -> ResolvedRun:
    """Body-first Eigenmode, with an adaptive EPR result when one was requested."""
    spec = run.spec
    expected = run.expected_outputs(
        "results/epr/eigenmodes.csv",
        "results/epr/eigenmodes.eig",
        "results/epr/adaptive-convergence.prop",
        "results/epr/adaptive-mode-history.json",
    )
    if spec.epr_request is not None:
        expected.add("results/epr/adaptive-epr-result.json")
        expected.update(_saved_solution_outputs(run))
    run.require_manifest(expected, "body-first Eigenmode")
    run.verified("results/epr/adaptive-convergence.prop")
    run.verified("results/epr/adaptive-mode-history.json")
    require_export_convergence_source(run.receipt)
    if run.receipt.get("convergence") != read_hfss_convergence(run.root, spec):
        raise RuntimeError("body-first Eigenmode convergence evidence is invalid")
    adaptive_result = (
        None
        if spec.epr_request is None
        else _resolved_adaptive_epr(
            run.verified("results/epr/adaptive-epr-result.json"),
            spec,
            run.receipt["convergence"],
        )
    )
    return run.resolved(
        "eigenmode",
        run.verified("results/epr/eigenmodes.csv"),
        provenance_path=run.verified("results/epr/eigenmodes.eig"),
        epr_result=adaptive_result,
    )


def _saved_solution_outputs(run: _CompletedRun) -> set[str]:
    """Extra manifest entries when the saved field solution was sealed.

    The run may legitimately end without a saved solution, but then the receipt must
    say so explicitly, with a reason.
    """
    saved_summary = run.receipt.get("saved_solution")
    if isinstance(saved_summary, dict) and saved_summary.get("status") == "complete":
        saved = resolve_saved_solution(run.root / _EPR_SAVED_MANIFEST)
        expected_saved_summary = {
            "status": "complete",
            "manifest": _EPR_SAVED_MANIFEST,
            "manifest_sha256": file_sha256(run.root / _EPR_SAVED_MANIFEST),
            "receipt": _EPR_SAVED_RECEIPT,
            "receipt_sha256": file_sha256(run.root / _EPR_SAVED_RECEIPT),
            "content_sha256": saved.content_sha256,
            "identity": dict(saved.identity),
            "member_count": len(saved.members),
        }
        if saved_summary != expected_saved_summary:
            raise RuntimeError("completed EPR saved-solution binding is invalid")
        return {_EPR_SAVED_MANIFEST, _EPR_SAVED_RECEIPT}
    if not (
        isinstance(saved_summary, dict)
        and saved_summary.get("status") == "unavailable"
        and isinstance(saved_summary.get("error"), str)
        and saved_summary["error"]
    ):
        raise RuntimeError("completed EPR run lacks an explicit saved-solution status")
    return set()


def _resolve_driven(run: _CompletedRun) -> ResolvedRun:
    """Terminal or Modal: a result table plus a two-port Touchstone file."""
    mode = run.receipt["mode"]
    result_stem = "terminal_st" if mode == "terminal" else "modal_s"
    expected = run.expected_outputs(
        f"results/{mode}/{result_stem}.csv", f"results/{mode}/{mode}.s2p"
    )
    run.require_manifest(expected, "terminal")
    return run.resolved(
        mode,
        run.verified(f"results/{mode}/{result_stem}.csv"),
        touchstone_path=run.verified(f"results/{mode}/{mode}.s2p"),
    )


def _captured_fields(
    receipt: dict[str, Any],
    outputs: dict[str, Any],
    spec: Any,
    *,
    epr_result: EprResult | None = None,
) -> dict[str, Any]:
    """Keep the identities needed by later report reads without replaying a run."""

    return {
        "_output_hashes": tuple(sorted(outputs.items())),
        "_spec_sha256": receipt["source"]["spec_sha256"],
        "_execution_seconds": receipt.get("execution_seconds"),
        "_setup_name": spec.run_control.setup_name,
        "_epr_requested": isinstance(spec, HfssEprSpec)
        and spec.epr_request is not None,
        "_epr_result": epr_result,
        "_lumped_boundaries": tuple(receipt.get("geometry", {}).get("lumped_rlcs", ()))
        + tuple(
            item for item in receipt.get("ports", ()) if item.get("kind") == "lumped"
        ),
    }


def _resolved_adaptive_epr(
    path: Path, spec: HfssEprSpec, convergence: dict[str, Any]
) -> EprResult:
    """Load the adaptive EPR result and bind it to this run's source and passes."""
    result = resolve_epr_result(path)
    requested_modes = (
        tuple(range(1, spec.run_control.num_modes + 1))
        if spec.epr_request is None or spec.epr_request.mode_indices is None
        else spec.epr_request.mode_indices
    )
    if (
        result.result_kind != "adaptive_history"
        or result.setup_name != spec.run_control.setup_name
        or result.provenance.get("model_source_sha256") != spec.geometry.model_sha256
        or result.provenance.get("analysis_source_sha256")
        != spec.geometry.source_sha256
        or tuple(result.provenance.get("requested_modes", ())) != requested_modes
        or result.provenance.get("solver_last_completed_pass")
        != convergence["final_pass"]
    ):
        raise RuntimeError(
            "adaptive EPR result differs from the completed source and setup"
        )
    if any(
        row["mode"] not in requested_modes
        or row["native_pass"] > convergence["final_pass"]
        for row in result.rows
    ):
        raise RuntimeError(
            "adaptive EPR rows exceed the requested modes or observed passes"
        )
    return result


def _resolved_benchmark(
    root: Path, receipt: dict[str, Any], outputs: dict[str, Any], setup_name: str
) -> Path | None:
    """The benchmark file is optional, but a file and its receipt marker go together."""
    marker = receipt.get("benchmark")
    if marker is None:
        if BENCHMARK_RELATIVE in outputs:
            raise RuntimeError("benchmark output lacks a receipt binding")
        return None
    if (
        not isinstance(marker, dict)
        or set(marker) != {"status", "path"}
        or marker["path"] != BENCHMARK_RELATIVE
    ):
        raise RuntimeError("benchmark receipt binding is invalid")
    path = verified_output(root, BENCHMARK_RELATIVE, outputs)
    payload = read_simulation_benchmark(path, setup_name)
    if marker["status"] != payload["status"]:
        raise RuntimeError("benchmark receipt status differs from its artifact")
    return path


# ---------------------------------------------------------------------------
# Native readback dispatch. Kept here under this name because tests patch it.
# ---------------------------------------------------------------------------


def _validate_readback(root: Path, receipt: dict[str, Any], spec: Any) -> None:
    """Check the AEDT-read-back evidence for the spec's family, then diagnostics."""
    connected = receipt.get("connected")
    if not isinstance(connected, dict) or connected != {
        "aedt_version": spec.aedt_version,
        "pyaedt_version": spec.pyaedt_version,
    }:
        raise RuntimeError("completed receipt has invalid connected version identity")
    if isinstance(spec, Q2dSpec):
        validate_q2d_readback(root, receipt, spec)
    elif isinstance(spec, Q3dSpec):
        validate_q3d_readback(root, receipt, spec)
    elif isinstance(spec, HfssEigenmodeSpec):
        validate_eigenmode_readback(receipt, spec)
        validate_hfss_setup_and_convergence(root, receipt, spec)
    else:
        validate_driven_readback(root, receipt, spec)
    _validate_diagnostics(root, receipt.get("diagnostics"))


def _validate_diagnostics(root: Path, diagnostics: Any) -> None:
    """The batch log must be present and match the hash the receipt recorded."""
    if not isinstance(diagnostics, dict) or diagnostics.get("batch_log") != "batch.log":
        raise RuntimeError("completed receipt diagnostics are invalid")
    batch_log = contained_in_handoff_root(root, "batch.log")
    if (
        diagnostics.get("present") is not True
        or not batch_log.is_file()
        or diagnostics.get("sha256") != file_sha256(batch_log)
        or not isinstance(diagnostics.get("physics_warnings"), list)
    ):
        raise RuntimeError("completed receipt diagnostics hash is invalid")
