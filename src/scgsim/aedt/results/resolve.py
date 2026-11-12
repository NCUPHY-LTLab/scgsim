"""Offline strict AEDT result and receipt resolution."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from scgsim.aedt.epr.models import EprResult
from scgsim.aedt.io import file_sha256, read_json
from scgsim.aedt.preparation.cohort import (
    validate_geometry_source,
)
from scgsim.aedt.presentation.benchmark import display_benchmark
from scgsim.aedt.presentation.runs import display_resolved_run
from scgsim.aedt.results.benchmark import read_simulation_benchmark
from scgsim.aedt.results.convergence.hfss import read_hfss_convergence
from scgsim.aedt.results.epr import resolve_epr_result, resolve_saved_solution
from scgsim.aedt.results.hfss_readback import (
    validate_eigenmode_readback,
    validate_hfss_pec_bindings,
)
from scgsim.aedt.results.matrices import parse_matrix_export, read_q2d_rlgc_matrix
from scgsim.aedt.results.provenance import require_current_receipt
from scgsim.aedt.results.readback_dispatch import validate_result_readback
from scgsim.aedt.results.receipt_validation import (
    contained,
    validate_completion_cohort,
    verified,
)
from scgsim.aedt.results.terminal_readback import validate_lumped_readback
from scgsim.aedt.runtime.batch_protocol import CollectionContext, collection_source
from scgsim.aedt.runtime.benchmark import BENCHMARK_RELATIVE
from scgsim.aedt.specs.common import MatrixRunControl
from scgsim.aedt.specs.hfss import (
    EigenmodeRunControl,
    HfssDrivenGeometrySpec,
    HfssDrivenSpec,
    HfssEigenmodeSpec,
    HfssEprSpec,
)
from scgsim.aedt.specs.parse import parse_aedt_spec
from scgsim.aedt.specs.q2d import Q2dSpec
from scgsim.aedt.specs.q3d import Q3dSpec
from scgsim.presentation.notebook import checked_theme


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
    _saved_solution_aedt_version: str | None = None
    _requested_aedt_version: str | None = None
    _connected_aedt_version: str | None = None
    _epr_requested: bool = False
    _epr_result: EprResult | None = None
    _lumped_boundaries: tuple[dict[str, Any], ...] = ()

    @property
    def output_hashes(self) -> tuple[tuple[str, str], ...]:
        """Return the verified output hash inventory captured from the receipt."""
        return self._output_hashes

    @property
    def spec_sha256(self) -> str | None:
        """Return the verified source-spec hash captured from the receipt."""
        return self._spec_sha256

    @property
    def setup_name(self) -> str | None:
        """Return the native setup name captured for result presentation."""
        return self._setup_name

    @property
    def saved_solution_aedt_version(self) -> str | None:
        """Return the saved solution AEDT version when one was recorded."""
        return self._saved_solution_aedt_version

    @property
    def requested_aedt_version(self) -> str | None:
        """Return the requested AEDT version captured from the receipt."""
        return self._requested_aedt_version

    @property
    def connected_aedt_version(self) -> str | None:
        """Return the connected AEDT version captured from the receipt."""
        return self._connected_aedt_version

    @property
    def epr_requested(self) -> bool:
        """Report whether this run requested adaptive EPR analysis."""
        return self._epr_requested

    @property
    def lumped_boundaries(self) -> tuple[dict[str, Any], ...]:
        """Return captured lumped boundary readback records."""
        return self._lumped_boundaries

    def _verified_output(self, path: Path) -> Path:
        root = self.receipt_path.parent.parent
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise RuntimeError("resolved output escapes the run directory") from exc
        return verified(root, relative, dict(self._output_hashes))

    def _verified_spec(self) -> Path:
        root = self.receipt_path.parent.parent
        return verified(
            root, "aedt_spec.json", {"spec_sha256": self._spec_sha256}, "spec_sha256"
        )

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
        checked_theme(theme)
        display_resolved_run(self, show_details=show_details, theme=theme)

    def physics_results(self) -> tuple[dict[str, str], ...]:
        """Return the verified primary result as string-valued rows."""

        self._verified_output(self.primary_csv)

        if self.mode == "q2d":
            root = self.receipt_path.parent.parent
            spec = parse_aedt_spec(read_json(self._verified_spec()), base_dir=root)
            if not isinstance(spec, Q2dSpec):
                raise RuntimeError("resolved Q2D result has a non-Q2D spec")
            rows, _ = read_q2d_rlgc_matrix(self.primary_csv, spec)
            return tuple(
                {key: str(value) for key, value in row.items()} for row in rows
            )

        if self.mode == "q3d":
            root = self.receipt_path.parent.parent
            spec = parse_aedt_spec(read_json(self._verified_spec()), base_dir=root)
            if not isinstance(spec, Q3dSpec):
                raise RuntimeError("resolved Q3D result has a non-Q3D spec")
            run_control = cast(MatrixRunControl, spec.run_control)
            if not spec.solve_ac_rl:
                rows, _ = parse_matrix_export(
                    self.primary_csv,
                    "Q3D",
                    "C",
                    run_control.frequency_ghz,
                    {"Capacitance Matrix": "C", "Conductance Matrix": "G"},
                )
                return tuple(
                    {key: str(value) for key, value in row.items()} for row in rows
                )

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
                    cast(str, self._setup_name),
                )
            ),
        }

    def show_simulation_benchmark(
        self, *, show_details: bool = False, theme: str = "light"
    ) -> AedtBenchmarkReport:
        """Return a notebook-displayable, offline view of native profile evidence."""
        checked_theme(theme)
        return AedtBenchmarkReport(self.simulation_benchmark(), show_details, theme)


@dataclass(frozen=True)
class CompletedRun:
    """Receipt, source, and model identity validated before result resolution."""

    root: Path
    receipt: dict[str, Any]
    receipt_path: Path
    spec: Any
    mode: Literal["terminal", "modal", "eigenmode", "q3d", "q2d"]
    save: dict[str, Any]


@dataclass(frozen=True)
class OutputContext:
    """Verified shared output paths passed to one family-specific reader."""

    root: Path
    receipt: dict[str, Any]
    outputs: dict[str, Any]
    spec: Any
    mode: Literal["terminal", "modal", "eigenmode", "q3d", "q2d"]
    receipt_path: Path
    project: Path
    project_relative: str
    benchmark_path: Path | None


@dataclass(frozen=True)
class AedtBenchmarkReport:
    """Offline native benchmark view; ``data`` retains the complete profile tree."""

    data: dict[str, Any]
    show_details: bool = False
    theme: str = "light"

    def _ipython_display_(self) -> None:
        display_benchmark(self.data, show_details=self.show_details, theme=self.theme)


def _captured_fields(
    receipt: dict[str, Any],
    outputs: dict[str, Any],
    spec: Any,
    *,
    epr_result: EprResult | None = None,
    saved_solution_aedt_version: str | None = None,
) -> dict[str, Any]:
    """Keep the identities needed by later report reads without replaying a run."""

    return {
        "_output_hashes": tuple(sorted(outputs.items())),
        "_spec_sha256": receipt["source"]["spec_sha256"],
        "_execution_seconds": receipt.get("execution_seconds"),
        "_setup_name": cast(str, spec.run_control.setup_name),
        "_saved_solution_aedt_version": saved_solution_aedt_version,
        "_requested_aedt_version": spec.aedt_version,
        "_connected_aedt_version": receipt["connected"]["aedt_version"],
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
    result = resolve_epr_result(path)
    run_control = cast(EigenmodeRunControl, spec.run_control)
    requested_modes = (
        tuple(range(1, run_control.num_modes + 1))
        if spec.epr_request is None or spec.epr_request.mode_indices is None
        else spec.epr_request.mode_indices
    )
    if (
        result.result_kind != "adaptive_history"
        or result.setup_name != run_control.setup_name
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


def resolve_results(run_dir: str | Path) -> ResolvedRun:
    """Resolve results after each receipt, source, and native-readback check."""
    root = Path(run_dir).resolve()
    receipt_path, receipt = _read_completed_receipt(root)

    save = receipt.get("save")
    if (
        not isinstance(save, dict)
        or save.get("ok") is not True
        or not isinstance(save.get("project_sha256"), str)
    ):
        raise RuntimeError("completed receipt lacks a successful exact project save")
    if receipt.get("release") != {"ok": True}:
        raise RuntimeError("completed receipt lacks a successful owned Desktop release")

    source = receipt.get("source")
    mode = receipt.get("mode")
    if not isinstance(source, dict) or source.get("spec") != "aedt_spec.json":
        raise RuntimeError("receipt source paths are not canonical")
    spec_path = verified(root, "aedt_spec.json", source, "spec_sha256")
    spec = parse_aedt_spec(read_json(spec_path), base_dir=root)
    validate_completion_cohort(root, receipt, spec)
    _validate_source_binding(root, source, spec)
    mode = _validate_result_family(root, receipt, mode, spec)
    return _resolve_mode_outputs(
        CompletedRun(root, receipt, receipt_path, spec, mode, save)
    )


def _read_completed_receipt(root: Path) -> tuple[Path, dict[str, Any]]:
    """Read a completed receipt and verify any native-batch collection lineage."""
    receipt_path = contained(root, "metadata/aedt_run_receipt.json")
    if not receipt_path.is_file():
        raise FileNotFoundError(f"run receipt is missing: {receipt_path}")
    receipt = read_json(receipt_path)
    require_current_receipt(receipt)
    if receipt.get("status") != "completed":
        raise RuntimeError("handoff has not completed successfully")
    if "native_batch" in receipt:
        batch = receipt["native_batch"]
        if (
            batch.get("collection") != "collected_sealed"
            or batch.get("collector_analyze_invoked") is not False
        ):
            raise RuntimeError(
                "batch collection did not complete its native collection/seal"
            )
        package, attempt, _ = collection_source(
            root, CollectionContext.from_payload(batch["source"])
        )
        if (
            batch.get("package_id") != package.package_id
            or batch.get("attempt_id") != attempt.attempt_id
        ):
            raise RuntimeError("batch collection lineage differs")
    return receipt_path, receipt


def _validate_source_binding(root: Path, source: dict[str, Any], spec: Any) -> None:
    """Verify canonical source members for the resolved model family."""
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
        verified(root, "geometry/design.gds", source, "gds_sha256")


def _validate_result_family(
    root: Path,
    receipt: dict[str, Any],
    mode: Any,
    spec: Any,
) -> Literal["terminal", "modal", "eigenmode", "q3d", "q2d"]:
    """Check family identity, native readback, and diagnostics in receipt order."""
    if (
        mode not in {"terminal", "modal", "eigenmode", "q3d", "q2d"}
        or spec.mode != mode
    ):
        raise RuntimeError("AEDT receipt mode is invalid")
    result_mode = cast(Literal["terminal", "modal", "eigenmode", "q3d", "q2d"], mode)
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
    return result_mode


def _resolve_mode_outputs(completed: CompletedRun) -> ResolvedRun:
    """Verify the canonical output inventory and build its result view."""
    run = completed
    outputs = run.receipt.get("outputs")
    if not isinstance(outputs, dict):
        raise TypeError("completed receipt has no output hash manifest")
    benchmark_path = _resolved_benchmark(
        run.root,
        run.receipt,
        outputs,
        cast(str, run.spec.run_control.setup_name),
    )
    project_relative = f"{run.spec.project_name}.aedt"
    if run.receipt.get("project") != project_relative:
        raise RuntimeError("completed receipt project path is not canonical")
    project = verified(run.root, project_relative, outputs, project_relative)
    if run.save["project_sha256"] != outputs.get(project_relative):
        raise RuntimeError("saved project hash does not match output manifest")
    context = OutputContext(
        run.root,
        run.receipt,
        outputs,
        run.spec,
        run.mode,
        run.receipt_path,
        project,
        project_relative,
        benchmark_path,
    )
    if context.mode == "q2d":
        return _resolve_q2d_outputs(context)
    if context.mode == "q3d":
        return _resolve_q3d_outputs(context)
    if context.mode == "eigenmode":
        return _resolve_eigenmode_outputs(context)
    return _resolve_scattering_outputs(context)


def _resolve_q2d_outputs(context: OutputContext) -> ResolvedRun:
    """Verify the exact Q2D output inventory."""
    superseded = (
        "results/q2d/cg_matrix.csv",
        "results/q2d/rl_matrix.csv",
        "results/q2d/matrices.csv",
    )
    if any(contained(context.root, relative).exists() for relative in superseded):
        raise RuntimeError("Q2D run contains superseded matrix artifacts")
    expected = {
        context.project_relative,
        "results/q2d/rlgc_matrix.csv",
    }
    if context.benchmark_path is not None:
        expected.add(BENCHMARK_RELATIVE)
    if set(context.outputs) != expected:
        raise RuntimeError("Q2D output manifest is not canonical")
    return ResolvedRun(
        "q2d",
        context.project,
        verified(context.root, "results/q2d/rlgc_matrix.csv", context.outputs),
        None,
        None,
        context.receipt_path,
        convergence=context.receipt["convergence"],
        benchmark_path=context.benchmark_path,
        **_captured_fields(context.receipt, context.outputs, context.spec),
    )


def _resolve_q3d_outputs(context: OutputContext) -> ResolvedRun:
    """Verify the exact Q3D output inventory."""
    expected = {context.project_relative, "results/q3d/c_matrix.csv"}
    if context.benchmark_path is not None:
        expected.add(BENCHMARK_RELATIVE)
    primary = "results/q3d/c_matrix.csv"
    if context.spec.solve_ac_rl:
        expected.update({"results/q3d/ac_rl_matrix.csv", "results/q3d/matrices.csv"})
        primary = "results/q3d/matrices.csv"
    elif any(
        contained(context.root, relative).exists()
        for relative in ("results/q3d/ac_rl_matrix.csv", "results/q3d/matrices.csv")
    ):
        raise RuntimeError("Q3D capacitance-only run contains AC-RL artifacts")
    if set(context.outputs) != expected:
        raise RuntimeError("Q3D output manifest is not canonical")
    return ResolvedRun(
        "q3d",
        context.project,
        verified(context.root, primary, context.outputs),
        None,
        None,
        context.receipt_path,
        convergence=context.receipt["convergence"],
        benchmark_path=context.benchmark_path,
        **_captured_fields(context.receipt, context.outputs, context.spec),
    )


def _resolve_eigenmode_outputs(context: OutputContext) -> ResolvedRun:
    """Select the matching saved-solution or ordinary Eigenmode evidence path."""
    if isinstance(context.spec, HfssEprSpec):
        return _resolve_epr_outputs(context)
    return _resolve_standard_eigenmode_outputs(context)


def _resolve_epr_outputs(context: OutputContext) -> ResolvedRun:
    """Verify adaptive Eigenmode outputs and any saved-solution binding."""
    connected = context.receipt.get("connected")
    if connected != {
        "aedt_version": context.spec.aedt_version,
        "pyaedt_version": context.spec.pyaedt_version,
    }:
        raise RuntimeError("completed receipt has invalid connected version identity")
    saved_manifest = "saved-solution-manifest.json"
    saved_receipt = "metadata/saved-solution-receipt.json"
    expected = {
        context.project_relative,
        "results/epr/eigenmodes.csv",
        "results/epr/eigenmodes.eig",
        "results/epr/adaptive-convergence.prop",
        "results/epr/adaptive-mode-history.json",
    }
    if context.benchmark_path is not None:
        expected.add(BENCHMARK_RELATIVE)
    saved_solution_aedt_version = None
    if context.spec.epr_request is not None:
        expected.add("results/epr/adaptive-epr-result.json")
        saved_summary = context.receipt.get("saved_solution")
        if (
            isinstance(saved_summary, dict)
            and saved_summary.get("status") == "complete"
        ):
            saved = resolve_saved_solution(context.root / saved_manifest)
            expected_saved_summary = {
                "status": "complete",
                "manifest": saved_manifest,
                "manifest_sha256": file_sha256(context.root / saved_manifest),
                "receipt": saved_receipt,
                "receipt_sha256": file_sha256(context.root / saved_receipt),
                "content_sha256": saved.content_sha256,
                "identity": dict(saved.identity),
                "member_count": len(saved.members),
            }
            if saved.aedt_version is not None:
                expected_saved_summary["aedt_version"] = saved.aedt_version
            if saved_summary != expected_saved_summary:
                raise RuntimeError("completed EPR saved-solution binding is invalid")
            saved_solution_aedt_version = saved.aedt_version
            expected.update({saved_manifest, saved_receipt})
        elif not (
            isinstance(saved_summary, dict)
            and saved_summary.get("status") == "unavailable"
            and isinstance(saved_summary.get("error"), str)
            and saved_summary["error"]
        ):
            raise RuntimeError(
                "completed EPR run lacks an explicit saved-solution status"
            )
    if set(context.outputs) != expected:
        raise RuntimeError("body-first Eigenmode output manifest is not canonical")
    verified(context.root, "results/epr/adaptive-convergence.prop", context.outputs)
    verified(context.root, "results/epr/adaptive-mode-history.json", context.outputs)
    if context.receipt.get("convergence") != read_hfss_convergence(
        context.root, context.spec
    ):
        raise RuntimeError("body-first Eigenmode convergence evidence is invalid")
    adaptive_result = (
        None
        if context.spec.epr_request is None
        else _resolved_adaptive_epr(
            verified(
                context.root,
                "results/epr/adaptive-epr-result.json",
                context.outputs,
            ),
            context.spec,
            context.receipt["convergence"],
        )
    )
    return ResolvedRun(
        "eigenmode",
        context.project,
        verified(context.root, "results/epr/eigenmodes.csv", context.outputs),
        None,
        verified(context.root, "results/epr/eigenmodes.eig", context.outputs),
        context.receipt_path,
        convergence=context.receipt["convergence"],
        benchmark_path=context.benchmark_path,
        **_captured_fields(
            context.receipt,
            context.outputs,
            context.spec,
            epr_result=adaptive_result,
            saved_solution_aedt_version=saved_solution_aedt_version,
        ),
    )


def _resolve_standard_eigenmode_outputs(context: OutputContext) -> ResolvedRun:
    """Verify the standard Eigenmode output manifest and artifacts."""
    expected = {
        context.project_relative,
        "results/eigenmode/eigenmodes.csv",
        "results/eigenmode/eigenmodes.eig",
    }
    if context.benchmark_path is not None:
        expected.add(BENCHMARK_RELATIVE)
    expected.add("results/eigenmode/adaptive-convergence.prop")
    if set(context.outputs) != expected:
        raise RuntimeError("Eigenmode output manifest is not canonical")
    verified(
        context.root, "results/eigenmode/adaptive-convergence.prop", context.outputs
    )
    return ResolvedRun(
        "eigenmode",
        context.project,
        verified(context.root, "results/eigenmode/eigenmodes.csv", context.outputs),
        None,
        verified(context.root, "results/eigenmode/eigenmodes.eig", context.outputs),
        context.receipt_path,
        convergence=context.receipt["convergence"],
        benchmark_path=context.benchmark_path,
        **_captured_fields(context.receipt, context.outputs, context.spec),
    )


def _resolve_scattering_outputs(context: OutputContext) -> ResolvedRun:
    """Verify the terminal or modal output inventory."""

    result_stem = "terminal_st" if context.mode == "terminal" else "modal_s"
    expected = {
        context.project_relative,
        f"results/{context.mode}/{result_stem}.csv",
        f"results/{context.mode}/{context.mode}.s2p",
    }
    if context.benchmark_path is not None:
        expected.add(BENCHMARK_RELATIVE)
    if set(context.outputs) != expected:
        raise RuntimeError("terminal output manifest is not canonical")
    return ResolvedRun(
        context.mode,
        context.project,
        verified(
            context.root, f"results/{context.mode}/{result_stem}.csv", context.outputs
        ),
        verified(
            context.root, f"results/{context.mode}/{context.mode}.s2p", context.outputs
        ),
        None,
        context.receipt_path,
        convergence=context.receipt["convergence"],
        benchmark_path=context.benchmark_path,
        **_captured_fields(context.receipt, context.outputs, context.spec),
    )


def _resolved_benchmark(
    root: Path, receipt: dict[str, Any], outputs: dict[str, Any], setup_name: str
) -> Path | None:
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
    path = verified(root, BENCHMARK_RELATIVE, outputs)
    payload = read_simulation_benchmark(path, setup_name)
    if marker["status"] != payload["status"]:
        raise RuntimeError("benchmark receipt status differs from its artifact")
    return path


def _validate_readback(root: Path, receipt: dict[str, Any], spec: Any) -> None:
    """Preserve the tested resolve-module entry point for strict readback checks."""
    validate_result_readback(root, receipt, spec)
