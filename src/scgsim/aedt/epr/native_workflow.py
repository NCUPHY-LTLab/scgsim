"""Author, solve, and collect EPR using the transaction-owned HFSS application."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, cast

from scgsim.aedt.data import (
    detached,
)
from scgsim.aedt.epr.adaptive import (
    adaptive_epr_result,
    adaptive_mode_history,
    trace_by_pass,
    trace_for_cache_context,
)
from scgsim.aedt.epr.analysis import combine_epr_mode
from scgsim.aedt.epr.cache import prepare_expressions_and_cache, read_cache
from scgsim.aedt.epr.evidence import PreparedEprEvidence
from scgsim.aedt.epr.expression_authoring import author_epr_expressions
from scgsim.aedt.epr.expression_codec import read_native_named_expressions
from scgsim.aedt.epr.fields import (
    install_variables,
    parse_native_scalar,
)
from scgsim.aedt.epr.integrals import (
    SURFACE_ANALYSIS_SCOPE,
    attach_saved_native_surface_membership,
    canonical_integral,
    empty_raw_integrals,
    integral_target,
    store_integral,
)
from scgsim.aedt.epr.models import (
    EprResult,
)
from scgsim.aedt.epr.native import (
    add_analysis_surfaces,
    bind_saved_analysis_surfaces,
)
from scgsim.aedt.epr.surface_references import verify_saved_native_surface_references
from scgsim.aedt.io import file_sha256, write_json
from scgsim.aedt.runtime.families.hfss_body_eigenmode import (
    PreparedBodyEigenmode,
    collect_body_eigenmode,
    prepare_body_eigenmode,
    prepared_body_eigenmode_result,
    solve_and_export_body_eigenmode,
)
from scgsim.aedt.runtime.families.hfss_exports import export_eigenmode
from scgsim.aedt.runtime.families.hfss_setup import read_body_eigenmode_setup
from scgsim.aedt.runtime.native.common import (
    BoundAedtRequest,
    detached_data,
    pyaedt_version,
)
from scgsim.aedt.runtime.native.hfss_geometry import (
    bind_saved_planar_geometry,
)
from scgsim.aedt.runtime.native.hfss_lumped import bind_saved_generic_rlcs
from scgsim.aedt.runtime.native.saved_geometry import read_saved_junction_lines
from scgsim.aedt.specs.common import AedtResources
from scgsim.aedt.specs.hfss import (
    HfssEprAnalysisSpec,
    HfssEprSpec,
)


def analyze_saved_epr(
    Hfss: Any,
    run_dir: Path,
    spec: HfssEprAnalysisSpec,
    inset_plan: dict[tuple[str, float], dict[str, Any]],
    *,
    connected_aedt_version: str,
) -> dict[str, Any]:
    """Analyze an immutable saved-field cohort through a disposable project copy."""
    analysis = _SavedFieldAnalysis(
        Hfss, run_dir, spec, inset_plan, connected_aedt_version
    )
    return analysis.run()


class _SavedFieldAnalysis:
    """Keep the workcopy, native application, and saved-field evidence together."""

    def __init__(
        self,
        Hfss: Any,
        run_dir: Path,
        spec: HfssEprAnalysisSpec,
        inset_plan: dict[tuple[str, float], dict[str, Any]],
        connected_aedt_version: str,
    ) -> None:
        self.Hfss = Hfss
        self.run_dir = run_dir
        self.spec = spec
        self.inset_plan = inset_plan
        self.connected_aedt_version = connected_aedt_version

    def run(self) -> dict[str, Any]:
        self._open_workcopy()
        self._rebind_geometry()
        self._author_expressions()
        self._evaluate_modes()
        self._record_provenance()
        self._write_and_save()
        return self._result_payload()

    def _open_workcopy(self) -> None:
        if not isinstance(self.spec, HfssEprAnalysisSpec):
            raise TypeError("saved EPR analysis requires HfssEprAnalysisSpec")
        self.timings: dict[str, Any] = {}
        started = time.perf_counter()
        self.work_root = Path(self.run_dir) / "analysis-work"
        if self.work_root.exists():
            raise FileExistsError("saved EPR analysis workcopy already exists")
        self.work_root.mkdir(parents=True)
        for member in self.spec.saved_solution.members:
            relative = Path(str(member["path"]))
            source = self.spec.saved_solution.root / relative
            if (
                source.is_symlink()
                or source.resolve() != source
                or not source.is_file()
                or source.stat().st_size != member["bytes"]
                or file_sha256(source) != member["sha256"]
            ):
                raise RuntimeError(
                    f"saved EPR member changed before workcopy: {relative.as_posix()}"
                )
            target = self.work_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        self.timings["workcopy_seconds"] = round(time.perf_counter() - started, 6)
        relative_project = self.spec.saved_solution.project_path.relative_to(
            self.spec.saved_solution.root
        )
        self.project_path = self.work_root / relative_project
        if not self.project_path.is_file():
            raise RuntimeError("saved EPR analysis project copy is missing")
        started = time.perf_counter()
        self.app = self.Hfss(
            project=str(self.project_path),
            design=self.spec.design_name,
            solution_type="Eigenmode",
            new_desktop=False,
            close_on_exit=False,
        )
        self.timings["native_reopen_seconds"] = round(time.perf_counter() - started, 6)
        if self.spec.run_control.setup_name not in self.app.setup_names:
            raise RuntimeError("saved EPR analysis setup is missing")
        observed_variation = detached_data(self.app.available_variations.nominal_values)
        observed_variation_identity = json.dumps(
            observed_variation, sort_keys=True, separators=(",", ":")
        )
        if (
            observed_variation_identity
            != self.spec.saved_solution.identity["physical_variation"]
        ):
            raise RuntimeError("saved EPR physical variation identity differs")
        expected_fields_solution = f"{self.spec.run_control.setup_name} : LastAdaptive"
        available_fields = [
            str(item)
            for item in self.app.post.available_report_solutions("Fields") or ()
        ]
        if expected_fields_solution not in available_fields:
            raise RuntimeError(
                "saved EPR field solution is unavailable on the workcopy"
            )

    def _rebind_geometry(self) -> None:
        started = time.perf_counter()
        self.geometry = bind_saved_planar_geometry(self.app, self.spec.geometry)
        bind_saved_analysis_surfaces(
            self.app,
            self.spec.geometry,
            self.geometry,
            surface_field_selectors=self.spec.surface_field_selectors,
        )
        bind_saved_generic_rlcs(self.app, self.spec.geometry, self.geometry)
        read_saved_junction_lines(
            self.app,
            self.project_path,
            self.spec.design_name,
            self.spec.geometry,
            self.geometry,
        )
        self.timings["geometry_rebind_seconds"] = round(
            time.perf_counter() - started, 6
        )

    def _author_expressions(self) -> None:
        started = time.perf_counter()
        self.expressions, self.authoring, self.phase_timings = author_epr_expressions(
            self.app,
            self.work_root,
            self.spec,
            self.geometry,
            self.inset_plan,
            namespace=f"analysis_{self.spec.saved_solution.content_sha256[:16]}",
        )
        self.timings["expression_authoring_seconds"] = round(
            time.perf_counter() - started, 6
        )
        self.timings["expression_authoring_phases"] = self.phase_timings
        self.cache = {
            "schema_version": "scgsim.aedt.epr-cache-request.v1",
            "status": "not_modified_for_saved_field_analysis",
            **self.authoring,
        }

    def _evaluate_modes(self) -> None:
        self.output_dir = Path(self.run_dir) / "results/epr"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        self.outputs, self.final_modes = export_eigenmode(
            self.app, Path(self.run_dir), self.output_dir, self.spec
        )
        self.timings["final_export_seconds"] = round(time.perf_counter() - started, 6)
        self.frequencies = self.final_modes["eigenmodes"]["frequencies_ghz"]
        self.selected_modes = (
            tuple(range(1, self.spec.run_control.num_modes + 1))
            if self.spec.epr_request.mode_indices is None
            else self.spec.epr_request.mode_indices
        )
        self.rows: list[dict[str, Any]] = []
        self.raw_evidence: list[dict[str, Any]] = []
        self.mode_timings: list[dict[str, Any]] = []
        for mode in self.selected_modes:
            mode_started = time.perf_counter()
            raw, evidence = _evaluate_mode_integrals(
                self.app,
                self.spec,
                self.expressions,
                mode=mode,
                frequency_hz=float(self.frequencies[mode - 1]) * 1e9,
            )
            self.rows.append(
                {
                    "mode": mode,
                    "status": "complete",
                    "raw_integrals": raw,
                    "raw_integral_evidence": evidence,
                    **combine_epr_mode(
                        self.spec.geometry, raw, request=self.spec.epr_request
                    ),
                }
            )
            self.raw_evidence.append({"mode": mode, "expressions": evidence})
            self.mode_timings.append(
                {
                    "mode": mode,
                    "seconds": round(time.perf_counter() - mode_started, 6),
                }
            )
        self.timings["mode_integrals"] = self.mode_timings

    def _record_provenance(self) -> None:
        self.result_provenance = {
            "model_source_sha256": self.spec.geometry.model_sha256,
            "analysis_source_sha256": self.spec.geometry.source_sha256,
            "surface_analysis_scope": SURFACE_ANALYSIS_SCOPE,
            "saved_solution_content_sha256": self.spec.saved_solution.content_sha256,
            "saved_solution_identity": detached(self.spec.saved_solution.identity),
            "saved_solution_aedt_version": self.spec.saved_solution.aedt_version,
            "requested_aedt_version": self.spec.aedt_version,
            "connected_aedt_version": self.connected_aedt_version,
            "requested_modes": list(self.selected_modes),
            "request": self.spec.epr_request.to_payload(),
            "raw_integral_evidence": self.raw_evidence,
        }
        if self.spec.surface_field_selectors:
            native_readback = self.authoring["native_expression_readback"]
            readback_path = self.work_root / native_readback["path"]
            self.outputs[readback_path.relative_to(self.run_dir).as_posix()] = (
                native_readback["sha256"]
            )
            self.result_provenance["surface_field_selectors"] = dict(
                self.spec.surface_field_selectors
            )
            self.result_provenance["surface_field_selector_bindings"] = [
                {
                    **item,
                    "native_readback": {
                        **item["native_readback"],
                        "path": (self.work_root / item["native_readback"]["path"])
                        .relative_to(self.run_dir)
                        .as_posix(),
                    },
                }
                for item in self.authoring["surface_field_selector_bindings"]
            ]
            self.result_provenance["native_expression_readback"] = {
                **native_readback,
                "path": readback_path.relative_to(self.run_dir).as_posix(),
            }

    def _write_and_save(self) -> None:
        self.result = EprResult(
            result_kind="saved_field",
            setup_name=self.spec.run_control.setup_name,
            rows=tuple(self.rows),
            provenance=self.result_provenance,
        )
        self.result_path = self.output_dir / "epr-result.json"
        started = time.perf_counter()
        write_json(self.result_path, self.result.to_payload())
        self.outputs[self.result_path.relative_to(self.run_dir).as_posix()] = (
            file_sha256(self.result_path)
        )
        self.timings["result_write_seconds"] = round(time.perf_counter() - started, 6)
        started = time.perf_counter()
        if not self.app.save_project() or not self.project_path.is_file():
            raise RuntimeError("saved EPR analysis workcopy save failed")
        self.timings["workcopy_save_seconds"] = round(time.perf_counter() - started, 6)
        self.project_relative = self.project_path.relative_to(self.run_dir).as_posix()
        self.outputs[self.project_relative] = file_sha256(self.project_path)
        native_references = self.cache.get("native_surface_references", [])
        if native_references:
            native_references = verify_saved_native_surface_references(
                self.project_path, self.spec.design_name, native_references
            )
            self.cache["native_surface_references"] = native_references
            for row in self.rows:
                attach_saved_native_surface_membership(
                    row["raw_integrals"], row, native_references
                )
            self.result = EprResult(
                result_kind="saved_field",
                setup_name=self.spec.run_control.setup_name,
                rows=tuple(self.rows),
                provenance=self.result_provenance,
            )
            started = time.perf_counter()
            write_json(self.result_path, self.result.to_payload())
            self.outputs[self.result_path.relative_to(self.run_dir).as_posix()] = (
                file_sha256(self.result_path)
            )
            self.timings["result_write_seconds"] = round(
                self.timings["result_write_seconds"] + time.perf_counter() - started, 6
            )

    def _result_payload(self) -> dict[str, Any]:
        return {
            "workflow_status": "epr_analysis_completed",
            "solver_invoked": False,
            "outputs": self.outputs,
            "connected": {
                "aedt_version": self.connected_aedt_version,
                "pyaedt_version": pyaedt_version(),
            },
            "project": self.project_relative,
            "geometry": self.geometry,
            "setup": read_body_eigenmode_setup(self.app, self.spec),
            "expressions": self.expressions,
            "cache": self.cache,
            "result": self.result.to_payload(),
            "timings": self.timings,
        }


def _saved_field_evidence(
    app: Any, spec: HfssEprSpec, convergence: dict[str, Any]
) -> dict[str, Any]:
    """Bind the live completed setup's LastAdaptive field and variation identity."""

    expected_solution = f"{cast(Any, spec.run_control).setup_name} : LastAdaptive"
    solutions = [
        str(item) for item in app.post.available_report_solutions("Fields") or ()
    ]
    if expected_solution not in solutions:
        raise RuntimeError(
            f"completed EPR solve lacks saved Fields solution {expected_solution!r}: "
            f"{solutions!r}"
        )
    variation = detached_data(app.available_variations.nominal_values)
    if not isinstance(variation, dict) or any(
        not isinstance(key, str) for key in variation
    ):
        raise RuntimeError("completed EPR physical variation readback is invalid")
    final_pass = convergence.get("final_pass")
    if type(final_pass) is not int or final_pass <= 0:
        raise RuntimeError("completed EPR convergence pass identity is invalid")
    return {
        "saved_fields": True,
        "solver_last_completed_pass": final_pass,
        "saved_fields_pass": final_pass,
        "fields_solution": expected_solution,
        "physical_variation": json.dumps(
            variation, sort_keys=True, separators=(",", ":")
        ),
    }


def _evaluate_mode_integrals(
    app: Any,
    spec: HfssEprAnalysisSpec,
    expressions: list[dict[str, Any]],
    *,
    mode: int,
    frequency_hz: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pp_values = {
        f"scgsim_epr_pp_mode_{index}": "1" if index == mode else "0"
        for index in range(1, spec.run_control.num_modes + 1)
    }
    observed = install_variables(app, pp_values)
    if observed != pp_values:
        raise RuntimeError("EPR mode postprocessing assignment readback differs")
    solution = f"{spec.run_control.setup_name} : LastAdaptive"
    raw = empty_raw_integrals()
    raw["frequency_hz"] = {"value": frequency_hz, "unit": "Hz"}
    evidence: list[dict[str, Any]] = []
    for expression in expressions:
        if expression.get("cache_only"):
            continue
        identity = expression["identity"]
        selection = identity["selection"]
        purpose = str(identity["purpose"])
        target = integral_target(purpose, selection)
        scalar = parse_native_scalar(
            app.post.fields_calculator.evaluate(
                expression["name"],
                setup=solution,
                intrinsics={"Phase": "0deg", **pp_values},
            )
        )
        value, item_evidence = canonical_integral(
            scalar, target=target, purpose=purpose
        )
        store_integral(
            raw,
            target=target,
            value=value,
            expression_identity={
                "name": expression["name"],
                "identity_sha256": identity["sha256"],
                "purpose": purpose,
            },
        )
        evidence.append(
            {
                "name": expression["name"],
                "identity_sha256": identity["sha256"],
                "purpose": purpose,
                **item_evidence,
            }
        )
    return raw, evidence


def _analysis_readback(app, spec, cache, project_path):
    cache["serialized_readback"] = read_cache(app, spec, cache["items"])
    references = cache["native_surface_references"]
    if references:
        cache["native_surface_references"] = verify_saved_native_surface_references(
            project_path, spec.design_name, references
        )


def _prepared_analysis_readback(app, spec, cache, project_path):
    references = cache.get("native_surface_references", [])
    if references:
        cache["native_surface_references"] = verify_saved_native_surface_references(
            project_path, spec.design_name, references
        )


def prepare_epr_hfss(
    Hfss: Any,
    run_dir: Path,
    spec: HfssEprSpec,
    inset_plan: dict[tuple[str, float], dict[str, Any]] | None = None,
) -> PreparedBodyEigenmode:
    """Compose EPR analysis authoring with common body-first HFSS preparation."""
    return prepare_body_eigenmode(
        Hfss,
        run_dir,
        spec,
        geometry_extension=add_analysis_surfaces,
        analysis_author=lambda app, workspace, bound, geometry: (
            prepare_expressions_and_cache(app, workspace, bound, geometry, inset_plan)
        ),
        analysis_readback=_analysis_readback,
    )


def prepared_epr_result(prepared: PreparedBodyEigenmode) -> dict[str, Any]:
    """Return preparation evidence after saved native surface readback."""
    result = prepared_body_eigenmode_result(
        prepared, analysis_readback=_prepared_analysis_readback
    )
    result["surface_analysis_scope"] = SURFACE_ANALYSIS_SCOPE
    return result


def _export_epr_analysis(  # noqa: PLR0913 -- match the existing family analysis_export callback contract.
    prepared, spec, convergence, outputs, output_dir, timings
):
    saved_field_evidence = _saved_field_evidence(prepared.app, spec, convergence)
    started = time.perf_counter()
    history = adaptive_mode_history(
        prepared.app,
        spec,
        prepared.cache["items"],
        prepared.cache["serialized_readback"]["items"],
    )
    run_dir = prepared.request.workspace
    history_path = output_dir / "adaptive-mode-history.json"
    write_json(history_path, history)
    outputs[history_path.relative_to(run_dir).as_posix()] = file_sha256(history_path)
    adaptive_result = adaptive_epr_result(
        spec,
        history,
        convergence,
        native_surface_references=prepared.cache.get("native_surface_references", []),
    )
    result_path = output_dir / "adaptive-epr-result.json"
    write_json(result_path, adaptive_result.to_payload())
    outputs[result_path.relative_to(run_dir).as_posix()] = file_sha256(result_path)
    timings["history_export_seconds"] = round(time.perf_counter() - started, 6)
    return (
        history,
        adaptive_result.to_payload(),
        {
            "surface_analysis_scope": SURFACE_ANALYSIS_SCOPE,
            "saved_field_evidence": saved_field_evidence,
        },
    )


def solve_and_export_epr(
    prepared: PreparedBodyEigenmode,
    resources: AedtResources | None = None,
    resource_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Solve the prepared model and export adaptive EPR history."""
    return solve_and_export_body_eigenmode(
        prepared, resources, resource_evidence, analysis_export=_export_epr_analysis
    )


def collect_batch_epr(Hfss, run_dir, spec, prepared_receipt):
    """Bind saved batch outputs and collect full history without native authoring."""

    if (
        not isinstance(spec, HfssEprSpec)
        or spec.epr_request is None
        or spec.run_control is None
    ):
        raise TypeError("batch collection requires the original complete EPR request")
    evidence = PreparedEprEvidence.from_payload(
        prepared_receipt,
        evidence_root=run_dir / "batch-source/package/epr-expressions",
    )
    expressions = cast(list[dict[str, Any]], evidence.expression_payloads())
    cache = cast(dict[str, Any], evidence.cache_payload())
    original_definitions = evidence.required_definitions_payload()
    project_path = run_dir / f"{spec.project_name}.aedt"
    app = Hfss(
        project=str(project_path),
        design=spec.design_name,
        solution_type="Eigenmode",
        new_desktop=False,
        close_on_exit=False,
    )
    if app.design_name != spec.design_name or app.design_type != "HFSS":
        raise RuntimeError("returned batch design identity differs")
    geometry = bind_saved_planar_geometry(app, spec.geometry)
    bind_saved_analysis_surfaces(app, spec.geometry, geometry)
    bind_saved_generic_rlcs(app, spec.geometry, geometry)
    read_saved_junction_lines(
        app, project_path, spec.design_name, spec.geometry, geometry
    )
    setup = read_body_eigenmode_setup(app, spec)
    evidence_dir = run_dir / "metadata/epr_expressions"
    evidence_dir.mkdir(parents=True, exist_ok=False)
    observed = read_native_named_expressions(
        app,
        names=list(original_definitions),
        definitions=[],
        evidence_dir=evidence_dir,
        expected_readback=original_definitions,
    )
    # Compare independent saved definitions; do not load or regenerate a CLC.
    cache["native_expression_readback"] = observed
    for expression in expressions:
        previous = expression["native_readback"]
        names = previous["expression_definition_names"]
        expression["native_readback"] = {
            **previous,
            "path": observed["path"],
            "sha256": observed["sha256"],
            "bytes": observed["bytes"],
            "status": observed["status"],
            "definitions": {name: observed["definitions"][name] for name in names},
        }
    cache["serialized_readback"] = read_cache(app, spec, cache["items"])
    references = cache.get("native_surface_references", [])
    if references:
        cache["native_surface_references"] = verify_saved_native_surface_references(
            project_path, spec.design_name, references
        )
    prepared = PreparedBodyEigenmode(
        app,
        BoundAedtRequest.bind(run_dir, spec),
        project_path,
        geometry,
        setup,
        tuple(expressions),
        cache,
        {},
    )

    def complete_analysis(  # noqa: PLR0913 -- match the existing family analysis_export callback contract.
        prepared, bound, convergence, outputs, output_dir, timings
    ):
        history, payload, evidence = _export_epr_analysis(
            prepared, bound, convergence, outputs, output_dir, timings
        )
        _require_complete_batch_history(
            bound, prepared.cache, convergence, history, payload
        )
        return history, payload, evidence

    return collect_body_eigenmode(prepared, analysis_export=complete_analysis)


def _require_complete_batch_history(spec, cache, convergence, history, payload):
    """Require the requested original cache history, never a LastAdaptive substitute."""

    final_pass = convergence["final_pass"]
    expected_passes = set(range(1, final_pass + 1))
    selected_modes = (
        tuple(range(1, spec.run_control.num_modes + 1))
        if spec.epr_request.mode_indices is None
        else spec.epr_request.mode_indices
    )
    cache_history = history["cache_integral_traces"]
    requested_titles = {item["title"] for item in cache["items"]}
    if (
        cache_history.get("status") != "complete"
        or set(cache_history.get("traces", {})) != requested_titles
    ):
        raise RuntimeError(
            "batch collection lacks the complete requested expression-cache history"
        )
    for item in cache["items"]:
        title = item["title"]
        trace, context = trace_for_cache_context(
            cache_history["traces"][title],
            item,
            cache_history["queries"].get(title, {}),
        )
        if not context["status"].startswith("verified"):
            raise RuntimeError(f"batch cache context is unverified: {title}")
        if set(trace_by_pass(trace, final_pass=final_pass)) != expected_passes:
            raise RuntimeError(f"batch cache pass history is incomplete: {title}")
    rows = payload["rows"]
    expected_rows = {
        (mode, native_pass)
        for mode in selected_modes
        for native_pass in expected_passes
    }
    if (
        {(row["mode"], row["native_pass"]) for row in rows} != expected_rows
        or len(rows) != len(expected_rows)
        or any(row["status"] != "complete" for row in rows)
    ):
        raise RuntimeError(
            "batch collection lacks complete per-mode/per-pass adaptive EPR"
        )
