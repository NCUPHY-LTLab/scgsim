"""The single AEDT execution transaction, receipt owner, and Desktop lifecycle."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from scgsim.aedt.epr.geometry import (
    precompute_inset_surfaces,
)
from scgsim.aedt.epr.native_workflow import (
    analyze_saved_epr,
    collect_batch_epr,
    prepare_epr_hfss,
    prepared_epr_result,
    solve_and_export_epr,
)
from scgsim.aedt.io import file_sha256, read_json, write_json
from scgsim.aedt.preparation.planar import (
    validate_geometry_workers,
)
from scgsim.aedt.results.epr import seal_saved_solution
from scgsim.aedt.results.provenance import (
    RECEIPT_V1,
    RECEIPT_V2,
    RECEIPT_V3,
    record_runtime_source,
)
from scgsim.aedt.runtime.batch_protocol import (
    CollectionContext,
    collection_source,
    object_payload,
)
from scgsim.aedt.runtime.control import (
    request_q3d_abort_and_save,
    request_q3d_stop_and_save,
)
from scgsim.aedt.runtime.desktop_lifecycle import release_owned_desktop
from scgsim.aedt.runtime.families.hfss import (
    prepare_hfss,
    run_hfss,
)
from scgsim.aedt.runtime.families.hfss_body_eigenmode import (
    prepare_body_eigenmode,
    prepared_body_eigenmode_result,
    solve_and_export_body_eigenmode,
)
from scgsim.aedt.runtime.families.q2d import prepare_q2d, prepared_q2d_result, run_q2d
from scgsim.aedt.runtime.families.q3d import (
    prepare_q3d,
    prepared_q3d_result,
    run_q3d,
)
from scgsim.aedt.runtime.handoff_validation import (
    canonical_metadata_files,
    contained,
    execution_resources,
    json_object,
    require_pristine_run,
    validate_execution_flags,
    verify_prepared_cohort,
    verify_prepared_hashes,
    warn_thin_film_surfaces,
)
from scgsim.aedt.runtime.native.common import (
    owned_application_constructor,
    pyaedt_version,
)
from scgsim.aedt.runtime.native.compatibility import native_compatibility
from scgsim.aedt.runtime.receipt import (
    read_physics_warnings,
    record_epr_project_after_release,
    record_ordinary_result_after_release,
    record_result_before_release,
)
from scgsim.aedt.specs.hfss import (
    HfssDrivenGeometrySpec,
    HfssDrivenSpec,
    HfssEigenmodeSpec,
    HfssEprAnalysisSpec,
    HfssEprSpec,
)
from scgsim.aedt.specs.parse import parse_aedt_spec
from scgsim.aedt.specs.q2d import Q2dSpec
from scgsim.aedt.specs.q3d import Q3dSpec
from scgsim.aedt.versions import resolve_aedt_version

_solve = run_hfss

_solve_q3d = run_q3d

_solve_q2d = run_q2d


def main(argv: list[str] | None = None) -> int:
    """Run the selected CLI action against a prepared handoff."""
    parser = argparse.ArgumentParser(description="Run one prepared SCGSim AEDT handoff")
    parser.add_argument("--handoff", required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Open local AEDT and solve the prepared handoff",
    )
    termination = parser.add_mutually_exclusive_group()
    termination.add_argument(
        "--stop-and-save",
        action="store_true",
        help="Request normal stop and recorder save of an owned Linux-local Q3D solve",
    )
    termination.add_argument(
        "--abort-and-save",
        action="store_true",
        help="Request immediate abort and recorder save of an owned Linux-local Q3D solve",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Create and verify an HFSS 3D, Q3D or Q2D native model without solving it",
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
    if args.stop_and_save or args.abort_and_save:
        if any((args.execute, args.prepare_only, args.analyze_epr)) or any(
            value is not None
            for value in (args.geometry_workers, args.cores, args.ram_limit_percent)
        ):
            parser.error(
                "--stop-and-save/--abort-and-save are exclusive with other actions and overrides"
            )
        request = (
            request_q3d_stop_and_save
            if args.stop_and_save
            else request_q3d_abort_and_save
        )
        state = request(args.handoff)
        print(json.dumps(state, indent=2))
        return 0 if state["intent"]["rpc_status"] == "returned" else 1
    metadata_path = Path(args.handoff).resolve()
    if not metadata_path.is_file():
        raise FileNotFoundError(f"handoff metadata is missing: {metadata_path}")
    if not args.execute:
        raise RuntimeError(
            "prepared handoff is not executed; use run_aedt.sh or pass --execute explicitly"
        )
    return execute_handoff(
        metadata_path,
        options=ExecutionOptions(
            prepare_only=args.prepare_only,
            analyze_epr=args.analyze_epr,
            geometry_workers=args.geometry_workers,
            cores=args.cores,
            ram_limit_percent=args.ram_limit_percent,
        ),
    )


@dataclass(frozen=True)
class ExecutionOptions:
    """Select a handoff action and its explicit execution overrides."""

    prepare_only: bool = False
    analyze_epr: bool = False
    geometry_workers: int | None = None
    cores: int | None = None
    ram_limit_percent: int | None = None
    batch_collection: CollectionContext | None = None


def execute_handoff(
    metadata_path: Path, *, options: ExecutionOptions | None = None
) -> int:
    """Validate, run, release, and record one prepared handoff."""
    return _ExecutionTransaction(metadata_path, options or ExecutionOptions()).run()


@dataclass
class _ExecutionTransaction:
    """Keep one handoff's mutable receipt and owned Desktop together."""

    metadata_path: Path
    options: ExecutionOptions

    def run(self) -> int:
        """Execute stages while preserving failure and Desktop cleanup order."""
        self._validate_handoff()
        self._start_receipt()
        try:
            self._run_native()
        except Exception as exc:  # noqa: BLE001 -- receipt must record any solver failure.
            self.failure = f"{type(exc).__name__}: {exc}"
            if isinstance(self.spec, (Q3dSpec, HfssDrivenGeometrySpec, HfssEprSpec)):
                # Body notes retain source attribution/native context in the existing
                # failed-receipt error, after the primary exception.
                notes = getattr(exc, "__notes__", ())
                if notes:
                    self.failure += "\n" + "\n".join(notes)
        finally:
            self._release_native()
            self._finish_receipt()
        return (
            0
            if self.status
            in {
                "completed",
                "epr_analysis_completed",
                "native_preparation_only",
                "normal_stop_saved",
                "abort_saved",
            }
            else 1
        )

    def _validate_handoff(self) -> None:
        self.run_dir = self.metadata_path.parent.parent
        os.chdir(self.run_dir)
        self.metadata = json_object(read_json(self.metadata_path), "handoff metadata")
        self.files = canonical_metadata_files(
            self.metadata, self.metadata_path, self.run_dir
        )
        self.receipt_path = self.run_dir / self.files["receipt"]
        self.receipt = json_object(read_json(self.receipt_path), "receipt")
        if self.receipt.get("schema_version") not in {
            RECEIPT_V1,
            RECEIPT_V2,
            RECEIPT_V3,
        }:
            raise RuntimeError("handoff receipt schema is invalid")
        if self.receipt.get("status") != "not_run":
            raise RuntimeError("one-shot handoff is not in not_run state")
        self.spec_path = self.run_dir / self.files["spec"]
        self.spec = parse_aedt_spec(
            json_object(read_json(self.spec_path), "spec"), base_dir=self.run_dir
        )
        self.profile = resolve_aedt_version(self.spec.aedt_version)
        if (
            not self.options.prepare_only
            and not self.options.analyze_epr
            and self.spec.run_control is None
        ):
            raise ValueError("solving requires an explicit complete run_control")
        self.resources = execution_resources(
            self.metadata, self.spec, self.options.cores, self.options.ram_limit_percent
        )
        validate_execution_flags(
            self.spec, self.run_dir, self.options.prepare_only, self.options.analyze_epr
        )
        self.batch_parent = None
        if self.options.batch_collection is None:
            require_pristine_run(self.run_dir, self.spec)
        else:
            if (
                self.options.prepare_only
                or self.options.analyze_epr
                or not isinstance(self.spec, HfssEprSpec)
                or self.spec.epr_request is None
            ):
                raise ValueError(
                    "external batch collection requires the original full EPR workflow"
                )
            self.batch_parent = collection_source(
                self.run_dir, self.options.batch_collection
            )
            package, attempt, prepared_receipt = self.batch_parent
            if (
                package.spec_sha256 != file_sha256(self.spec_path)
                or package.requested_aedt_version != self.spec.aedt_version
                or cast(dict[str, Any], prepared_receipt)["connected"]["aedt_version"]
                != self.spec.aedt_version
            ):
                raise RuntimeError(
                    "batch collection source/version differs from the prepared request"
                )
        self.cohort = verify_prepared_cohort(
            self.run_dir, self.metadata, self.receipt, self.spec
        )
        warn_thin_film_surfaces(self.spec)

    def _start_receipt(self) -> None:
        self.started = _utc_now()
        self.execution_started = time.perf_counter()
        self.receipt.update(
            {
                "schema_version": RECEIPT_V3,
                "expected_receipt_schema": RECEIPT_V3,
                "preparation_cohort": self.cohort["preparation_cohort"],
                "prepared_runtime_source": self.cohort["prepared_runtime_source"],
                "prepared_receipt_sha256": self.cohort["prepared_receipt_sha256"],
                "verified_prepared_manifest_sha256": self.cohort[
                    "verified_prepared_manifest_sha256"
                ],
                "status": "running",
                "started_at_utc": self.started,
                "save": {"ok": False},
                "release": {"ok": False},
            }
        )
        self.receipt["resources"] = {
            "requested": self.resources.to_payload() if self.resources else None,
            "request_source": (
                "cli_override"
                if self.options.cores is not None
                or self.options.ram_limit_percent is not None
                else "handoff"
                if self.resources is not None
                else "omitted"
            ),
            "status": (
                "not_used_prepare_only"
                if self.options.prepare_only
                else "not_applicable_saved_analysis"
                if self.options.analyze_epr
                else "not_started"
            ),
        }
        if self.options.prepare_only:
            self.receipt["solver_invoked"] = False
            self.receipt["model_stage"] = (
                "model_only" if self.spec.run_control is None else "model_with_setup"
            )
        if self.batch_parent is not None:
            package, attempt, prepared_receipt = self.batch_parent
            self.receipt["native_batch"] = {
                "source": cast(
                    CollectionContext, self.options.batch_collection
                ).to_payload(),
                "package_id": package.package_id,
                "attempt_id": attempt.attempt_id,
                "native_preparation": "released_verified",
                "external_solve": "process_terminal_native_results_pending_collection",
                "collection": "running",
                "collector_analyze_invoked": False,
                "external_execution_location": object_payload(
                    attempt.execution_location
                ),
                "external_requested_aedt_version": attempt.requested_aedt_version,
                "external_actual_aedt_version": attempt.actual_aedt_version,
                "resources_requested": attempt.resources_requested.to_payload(),
            }
            self.receipt["resources"] = {
                "requested": attempt.resources_requested.to_payload(),
                "status": "external_native_recipe; not applied to collector",
                "request_source": "native_batch_package",
            }
        self.receipt["requested_aedt_version"] = self.profile.requested_version
        self.receipt["connected_aedt_version"] = None
        if isinstance(self.spec, HfssEprAnalysisSpec):
            self.receipt["saved_solution_aedt_version"] = (
                self.spec.saved_solution.aedt_version
            )
        write_json(self.receipt_path, self.receipt)

        self.desktop: Any | None = None
        self.actual_aedt_version: str | None = None
        self.transport_stack = ExitStack()
        self.status = "failed"
        self.failure: str | None = None
        self.result: dict[str, Any] | None = None
        self.epr_solver_attempted = False

    def _run_native(self) -> None:
        self.receipt["mode"] = self.spec.mode
        verify_prepared_hashes(self.metadata, self.spec_path, self.spec)
        self.receipt["runtime_source"] = record_runtime_source("actual")
        if pyaedt_version() != self.profile.pyaedt_version:
            raise RuntimeError("PyAEDT lock mismatch")
        if isinstance(self.spec, Q3dSpec) and not self.options.prepare_only:
            write_json(self.receipt_path, self.receipt)
        setting = self.metadata.get("execution", {}).get("geometry_workers")
        workers = (
            self.options.geometry_workers
            if self.options.geometry_workers is not None
            else setting
        )
        validate_geometry_workers(workers)
        self.inset_plan = None
        if (
            self.batch_parent is None
            and isinstance(self.spec, (HfssEprSpec, HfssEprAnalysisSpec))
            and self.spec.epr_request is not None
        ):
            geometry_started = time.perf_counter()
            self.inset_plan = precompute_inset_surfaces(
                self.spec.geometry, self.spec.epr_request, workers
            )
            self.receipt["geometry_precompute"] = {
                "seconds": round(time.perf_counter() - geometry_started, 6),
                "members": len(self.inset_plan),
                "requested_workers": workers,
            }
        self.receipt["native_compatibility"] = {}
        self.transport_stack.enter_context(
            native_compatibility(
                self.profile, evidence=self.receipt["native_compatibility"]
            )
        )
        if (
            isinstance(self.spec, Q3dSpec)
            and not self.options.prepare_only
            and sys.platform == "linux"
            and os.getenv("PYAEDT_DOC_GENERATION", "false").lower()
            in {"true", "1", "t"}
        ):
            raise RuntimeError(
                "PyAEDT documentation mode cannot preserve stop-managed Desktop ownership"
            )
        # Load the optional solver dependency only after the receipt starts running.
        from ansys.aedt.core import Desktop, Hfss, Q2d, Q3d  # noqa: PLC0415 -- optional heavy solver dependency.

        desktop_started = time.perf_counter()
        self.desktop = Desktop(
            version=self.profile.requested_version,
            non_graphical=True,
            new_desktop=True,
            close_on_exit=False,
        )
        self.actual_aedt_version = self.desktop.aedt_version_id
        self.receipt["connected_aedt_version"] = self.actual_aedt_version
        if self.actual_aedt_version != self.profile.requested_version:
            raise RuntimeError(f"AEDT version mismatch: {self.actual_aedt_version!r}")
        if isinstance(self.spec, (HfssEprSpec, HfssEprAnalysisSpec)):
            self.receipt.setdefault("timings", {})["desktop_startup_seconds"] = round(
                time.perf_counter() - desktop_started, 6
            )
        self._dispatch_native(Hfss, Q2d, Q3d)

    def _dispatch_native(self, Hfss: Any, Q2d: Any, Q3d: Any) -> None:
        if self.batch_parent is not None:
            self.epr_solver_attempted = (
                True  # The bound external attempt ran the solver.
            )
            self.result = collect_batch_epr(
                owned_application_constructor(Hfss, self.desktop, profile=self.profile),
                self.run_dir,
                self.spec,
                self.batch_parent[2],
            )
            self.status = "completed"
        elif isinstance(self.spec, HfssEprAnalysisSpec):
            self.result = analyze_saved_epr(
                owned_application_constructor(Hfss, self.desktop, profile=self.profile),
                self.run_dir,
                self.spec,
                cast(dict[tuple[str, float], dict[str, Any]], self.inset_plan),
                connected_aedt_version=cast(str, self.actual_aedt_version),
            )
            self.status = "epr_analysis_completed"
        elif isinstance(self.spec, HfssEprSpec):
            self._run_epr(Hfss, self.spec)
        elif isinstance(self.spec, Q3dSpec):
            self._run_q3d(Q3d, self.spec)
        elif isinstance(self.spec, Q2dSpec):
            if self.options.prepare_only:
                prepared = prepare_q2d(
                    owned_application_constructor(
                        Q2d, self.desktop, profile=self.profile
                    ),
                    self.run_dir,
                    self.spec,
                )
                self.result = prepared_q2d_result(prepared)
                self.status = "native_preparation_only"
            else:
                self.result = _solve_q2d(
                    owned_application_constructor(
                        Q2d, self.desktop, profile=self.profile
                    ),
                    self.run_dir,
                    self.spec,
                    self.resources,
                    self.receipt["resources"],
                )
        elif self.options.prepare_only:
            prepared = prepare_hfss(
                owned_application_constructor(Hfss, self.desktop, profile=self.profile),
                self.run_dir,
                self.spec,
            )
            self.result = {
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
            self.status = "native_preparation_only"
        else:
            self.result = _solve(
                owned_application_constructor(Hfss, self.desktop, profile=self.profile),
                self.run_dir,
                self.spec,
                self.resources,
                self.receipt["resources"],
            )
        if (
            not isinstance(self.spec, (HfssEprAnalysisSpec, HfssEprSpec))
            and not self.options.prepare_only
            and not (self.result is not None and "termination" in self.result)
        ):
            self.status = "completed"

    def _run_epr(self, Hfss: Any, spec: HfssEprSpec) -> None:
        if spec.epr_request is None:
            prepared = prepare_body_eigenmode(
                owned_application_constructor(Hfss, self.desktop, profile=self.profile),
                self.run_dir,
                spec,
            )
        else:
            prepared = prepare_epr_hfss(
                owned_application_constructor(Hfss, self.desktop, profile=self.profile),
                self.run_dir,
                spec,
                self.inset_plan,
            )
        if self.options.prepare_only:
            self.result = (
                prepared_body_eigenmode_result(prepared)
                if spec.epr_request is None
                else prepared_epr_result(prepared)
            )
            self.status = "native_preparation_only"
        else:
            self.epr_solver_attempted = True
            self.result = (
                solve_and_export_body_eigenmode(
                    prepared, self.resources, self.receipt["resources"]
                )
                if spec.epr_request is None
                else solve_and_export_epr(
                    prepared, self.resources, self.receipt["resources"]
                )
            )
            self.status = "completed"

    def _run_q3d(self, Q3d: Any, spec: Q3dSpec) -> None:
        if self.options.prepare_only:
            prepared = prepare_q3d(
                owned_application_constructor(Q3d, self.desktop, profile=self.profile),
                self.run_dir,
                spec,
            )
            self.result = prepared_q3d_result(prepared)
            self.status = "native_preparation_only"
        else:
            self.result = _solve_q3d(
                owned_application_constructor(Q3d, self.desktop, profile=self.profile),
                self.run_dir,
                spec,
                self.resources,
                self.receipt["resources"],
                control_metadata=self.metadata_path,
                runtime_source=self.receipt["runtime_source"],
            )
            if "termination" in self.result:
                workflow = self.result["termination"]
                intent = workflow["control"].get("intent")
                self.failure = workflow["primary_error"]
                if (
                    self.failure is None
                    and workflow["analysis"].get("native_return") is not True
                ):
                    self.failure = (
                        "Q3D Analyze did not return True after normal-stop intent"
                    )
                if (
                    self.failure is None
                    and workflow["control"].get("native_settlement", {}).get("status")
                    != "inactive"
                ):
                    self.failure = "Q3D native inactivity was not confirmed after Analyze settlement"
                if self.failure is None and intent["rpc_status"] != "returned":
                    self.failure = intent.get(
                        "error", "Q3D stop RPC outcome is unknown"
                    )
                if self.failure is None and not self.result["save"]["ok"]:
                    self.failure = self.result["save"]["error"]
                self.status = (
                    (
                        "abort_saved"
                        if workflow["mode"] == "abort_and_save"
                        else "normal_stop_saved"
                    )
                    if self.failure is None
                    else "failed"
                )

    def _release_native(self) -> None:
        try:
            if self.result is not None:
                try:
                    record_result_before_release(self.receipt, self.result, self.spec)
                except Exception as exc:  # noqa: BLE001 -- record any native failure while retaining cleanup and the primary error.
                    self.receipt["result_record_error"] = f"{type(exc).__name__}: {exc}"
                    if self.failure is None:
                        self.failure = self.receipt["result_record_error"]
            if self.desktop is not None:
                release_started = time.perf_counter()
                try:
                    release_owned_desktop(self.desktop, self.receipt, self.result)
                except Exception as exc:  # noqa: BLE001 -- record any native failure while retaining cleanup and the primary error.
                    if self.failure is None:
                        self.failure = f"{type(exc).__name__}: {exc}"
                self.receipt.setdefault("timings", {})["release_seconds"] = round(
                    time.perf_counter() - release_started, 6
                )
        finally:
            try:
                self.transport_stack.close()
            except Exception as exc:  # noqa: BLE001 -- record any native failure while retaining cleanup and the primary error.
                self.receipt["native_compatibility_restore_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
                if self.failure is None:
                    self.failure = self.receipt["native_compatibility_restore_error"]

    def _finish_receipt(self) -> None:
        if self.failure is not None:
            self.status = "failed"
            self.receipt["error"] = self.failure
        if isinstance(self.spec, HfssEprSpec):
            self.receipt["solver_invoked"] = self.epr_solver_attempted
        self._seal_result()
        self.receipt["diagnostics"] = read_physics_warnings(self.run_dir)
        if self.batch_parent is not None:
            self.receipt["native_batch"]["collection"] = (
                "collected_sealed" if self.status == "completed" else "failed"
            )
            self.receipt["native_batch"]["external_solve"] = (
                "native_results_collected"
                if self.status == "completed"
                else "process_terminal_collection_failed"
            )
        self.receipt["status"] = self.status
        self.receipt["finished_at_utc"] = _utc_now()
        self.receipt["execution_seconds"] = round(
            time.perf_counter() - self.execution_started, 6
        )
        write_json(self.receipt_path, self.receipt)

    def _seal_result(self) -> None:
        if (
            self.result is not None
            and "termination" in self.result
            and self.result["termination"]["control"]
            .get("native_settlement", {})
            .get("status")
            == "inactive"
            and self.result["termination"]["control"]
            .get("request_readback", {})
            .get("status")
            != "unavailable"
            and self.result["save"]["ok"] is True
            and self.receipt.get("release", {}).get("ok") is True
        ):
            # Seal real released bytes independently, but do not overwrite the
            # stop-save failure with success from an older preparation project.
            try:
                project = contained(self.run_dir, self.result["project"])
                if not project.is_file():
                    raise RuntimeError(
                        "Q3D project is missing after normal-stop release"
                    )
                digest = file_sha256(project)
                self.result["outputs"][self.result["project"]] = digest
                self.receipt["outputs"] = self.result["outputs"]
                self.receipt["save"] = {
                    **self.result["save"],
                    "project_sha256": digest,
                    "identity_stage": "after_owned_desktop_release",
                }
            except Exception as exc:  # noqa: BLE001 -- record any native failure while retaining cleanup and the primary error.
                self.receipt["save"] = {
                    **self.result["save"],
                    "ok": False,
                    "sealing_error": f"{type(exc).__name__}: {exc}",
                }
                self.status = "failed"
                if self.failure is None:
                    self.failure = self.receipt["save"]["sealing_error"]
                    self.receipt["error"] = self.failure
        elif (
            self.result is not None
            and isinstance(
                self.spec,
                (
                    Q3dSpec,
                    Q2dSpec,
                    HfssDrivenGeometrySpec,
                    HfssDrivenSpec,
                    HfssEigenmodeSpec,
                    HfssEprSpec,
                ),
            )
            and self.options.prepare_only
        ):
            if self.receipt.get("release") == {"ok": True}:
                try:
                    if not record_epr_project_after_release(
                        self.run_dir, self.receipt, self.result
                    ):
                        raise RuntimeError(
                            "prepared project is missing after owned Desktop release"
                        )
                except Exception as exc:  # noqa: BLE001 -- retain the primary native failure.
                    self.receipt["save"] = {
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    self.status = "failed"
                    if self.failure is None:
                        self.failure = self.receipt["save"]["error"]
                        self.receipt["error"] = self.failure
        elif self.result is not None and isinstance(
            self.spec, (HfssEprAnalysisSpec, HfssEprSpec)
        ):
            self._seal_epr_result(self.result)
        elif self.result is not None and "termination" not in self.result:
            record_ordinary_result_after_release(self.receipt, self.result)

    def _seal_epr_result(self, result: dict[str, Any]) -> None:
        if not record_epr_project_after_release(self.run_dir, self.receipt, result):
            self.status = "failed"
            if self.failure is None:
                self.failure = "EPR project is missing after owned Desktop release"
                self.receipt["error"] = self.failure
        else:
            if (
                isinstance(self.spec, HfssEprSpec)
                and self.spec.epr_request is not None
                and self.status == "completed"
                and self.failure is None
                and self.receipt.get("release") == {"ok": True}
            ):
                try:
                    sealed = seal_saved_solution(
                        self.run_dir,
                        project_name=self.spec.project_name,
                        design_name=self.spec.design_name,
                        setup_name=cast(Any, self.spec.run_control).setup_name,
                        model_source_sha256=self.spec.geometry.model_sha256,
                        evidence=result["saved_field_evidence"],
                        aedt_version=cast(str, self.actual_aedt_version),
                    )
                    result["outputs"].update(
                        {
                            sealed["manifest"]: sealed["manifest_sha256"],
                            sealed["receipt"]: sealed["receipt_sha256"],
                        }
                    )
                    self.receipt["outputs"] = result["outputs"]
                    self.receipt["saved_solution"] = sealed
                except Exception as exc:  # noqa: BLE001 -- sealing is required.
                    seal_error = f"{type(exc).__name__}: {exc}"
                    self.receipt["saved_solution"] = {
                        "status": "unavailable",
                        "error": seal_error,
                    }
                    self.status = "failed"
                    if self.failure is None:
                        self.failure = seal_error
                        self.receipt["error"] = self.failure


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
