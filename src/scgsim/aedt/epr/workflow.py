"""EPR handoff and native saved-field workflow orchestration."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Literal
from collections.abc import Mapping
from scgsim.aedt._io import file_sha256, write_json
from scgsim.aedt.epr.analysis import combine_epr_mode
from scgsim.aedt.epr.cache import (
    _SURFACE_ANALYSIS_SCOPE,
    _adaptive_epr_result,
    _adaptive_mode_history,
    _attach_saved_native_surface_membership,
    _author_epr_expressions,
    _canonical_integral,
    _empty_raw_integrals,
    _integral_target,
    _prepare_expressions_and_cache,
    _read_cache,
    _store_integral,
)
from scgsim.aedt.epr.fields import install_variables, parse_native_scalar
from scgsim.aedt.epr.models import (
    EprResult,
)
from scgsim.aedt._data import (
    detached,
)
from scgsim.aedt.runtime.native.hfss_geometry import (
    _read_saved_junction_lines,
    bind_saved_planar_geometry,
)
from scgsim.aedt.epr.native import (
    verify_saved_native_surface_references,
)
from scgsim.aedt.runtime.families.hfss import _export_eigenmode
from scgsim.aedt.preparation.handoff import _prepare_epr_analysis_handoff
from scgsim.aedt.specs.epr import (
    EprAnalysisRequest,
    SavedSolution,
)
from scgsim.aedt.specs.geometry import (
    PreparedPlanarGeometry,
)
from scgsim.aedt.runtime.native.common import (
    detached_data,
    pyaedt_version,
)
from scgsim.aedt.specs.common import AedtResources
from scgsim.aedt.specs.hfss import (
    EigenmodeRunControl,
    HfssEprAnalysisSpec,
    HfssEprSpec,
)
from scgsim.aedt.runtime.native.hfss_lumped import _bind_saved_generic_rlcs
from scgsim.aedt.runtime.families.hfss import (
    read_body_eigenmode_setup,
    PreparedBodyEigenmode,
    prepare_body_eigenmode,
    prepared_body_eigenmode_result,
    solve_and_export_body_eigenmode,
)
from scgsim.aedt.epr.native import add_analysis_surfaces, bind_saved_analysis_surfaces


def analyze_epr(
    *,
    saved_solution: SavedSolution,
    aedt_version: str,
    geometry: PreparedPlanarGeometry,
    project_name: str,
    design_name: str,
    run_control: EigenmodeRunControl,
    epr_request: EprAnalysisRequest,
    output_dir: str | Path,
    geometry_workers: int | None = None,
    surface_field_selectors: Mapping[str, Literal["surface", "adjacent"]] | None = None,
) -> Any:
    """Analyze one sealed saved-field cohort in an owned disposable workcopy."""

    plan = _prepare_epr_analysis_handoff(
        saved_solution=saved_solution,
        surface_field_selectors=surface_field_selectors,
        aedt_version=aedt_version,
        geometry=geometry,
        project_name=project_name,
        design_name=design_name,
        run_control=run_control,
        epr_request=epr_request,
        output_dir=output_dir,
        geometry_workers=geometry_workers,
    )
    from scgsim.aedt.results.epr import resolve_epr_result
    from scgsim.aedt.runtime.transaction import _execute

    previous = Path.cwd()
    try:
        exit_code = _execute(plan.metadata_path, analyze_epr=True)
    finally:
        import os

        os.chdir(previous)
    if exit_code != 0:
        raise RuntimeError(
            "saved EPR analysis failed; inspect the prepared run receipt for the exact native error"
        )
    return resolve_epr_result(plan.run_dir / "results/epr/epr-result.json")


def _saved_field_evidence(
    app: Any, spec: HfssEprSpec, convergence: dict[str, Any]
) -> dict[str, Any]:
    """Bind the live completed setup's LastAdaptive field and variation identity."""

    expected_solution = f"{spec.run_control.setup_name} : LastAdaptive"
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
    raw = _empty_raw_integrals()
    raw["frequency_hz"] = {"value": frequency_hz, "unit": "Hz"}
    evidence: list[dict[str, Any]] = []
    for expression in expressions:
        if expression.get("cache_only"):
            continue
        identity = expression["identity"]
        selection = identity["selection"]
        purpose = str(identity["purpose"])
        target = _integral_target(purpose, selection)
        scalar = parse_native_scalar(
            app.post.fields_calculator.evaluate(
                expression["name"],
                setup=solution,
                intrinsics={"Phase": "0deg", **pp_values},
            )
        )
        value, item_evidence = _canonical_integral(
            scalar, target=target, purpose=purpose
        )
        _store_integral(
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


def analyze_saved_epr(
    Hfss: Any,
    run_dir: Path,
    spec: HfssEprAnalysisSpec,
    inset_plan: dict[tuple[str, float], dict[str, Any]],
    *,
    connected_aedt_version: str,
) -> dict[str, Any]:
    """Analyze an immutable saved-field cohort through a disposable project copy."""

    if not isinstance(spec, HfssEprAnalysisSpec):
        raise TypeError("saved EPR analysis requires HfssEprAnalysisSpec")
    timings: dict[str, Any] = {}
    started = time.perf_counter()
    work_root = Path(run_dir) / "analysis-work"
    if work_root.exists():
        raise FileExistsError("saved EPR analysis workcopy already exists")
    work_root.mkdir(parents=True)
    for member in spec.saved_solution.members:
        relative = Path(str(member["path"]))
        source = spec.saved_solution.root / relative
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
        target = work_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    timings["workcopy_seconds"] = round(time.perf_counter() - started, 6)
    relative_project = spec.saved_solution.project_path.relative_to(
        spec.saved_solution.root
    )
    project_path = work_root / relative_project
    if not project_path.is_file():
        raise RuntimeError("saved EPR analysis project copy is missing")
    started = time.perf_counter()
    app = Hfss(
        project=str(project_path),
        design=spec.design_name,
        solution_type="Eigenmode",
        new_desktop=False,
        close_on_exit=False,
    )
    timings["native_reopen_seconds"] = round(time.perf_counter() - started, 6)
    if spec.run_control.setup_name not in app.setup_names:
        raise RuntimeError("saved EPR analysis setup is missing")
    observed_variation = detached_data(app.available_variations.nominal_values)
    observed_variation_identity = json.dumps(
        observed_variation, sort_keys=True, separators=(",", ":")
    )
    if (
        observed_variation_identity
        != spec.saved_solution.identity["physical_variation"]
    ):
        raise RuntimeError("saved EPR physical variation identity differs")
    expected_fields_solution = f"{spec.run_control.setup_name} : LastAdaptive"
    available_fields = [
        str(item) for item in app.post.available_report_solutions("Fields") or ()
    ]
    if expected_fields_solution not in available_fields:
        raise RuntimeError("saved EPR field solution is unavailable on the workcopy")
    started = time.perf_counter()
    geometry = bind_saved_planar_geometry(app, spec.geometry)
    bind_saved_analysis_surfaces(
        app, spec.geometry, geometry,
        surface_field_selectors=spec.surface_field_selectors,
    )
    _bind_saved_generic_rlcs(app, spec.geometry, geometry)
    _read_saved_junction_lines(
        app, project_path, spec.design_name, spec.geometry, geometry
    )
    timings["geometry_rebind_seconds"] = round(time.perf_counter() - started, 6)
    started = time.perf_counter()
    expressions, authoring, phase_timings = _author_epr_expressions(
        app,
        work_root,
        spec,
        geometry,
        inset_plan,
        namespace=f"analysis_{spec.saved_solution.content_sha256[:16]}",
    )
    timings["expression_authoring_seconds"] = round(time.perf_counter() - started, 6)
    timings["expression_authoring_phases"] = phase_timings
    cache = {
        "schema_version": "scgsim.aedt.epr-cache-request.v1",
        "status": "not_modified_for_saved_field_analysis",
        **authoring,
    }
    output_dir = Path(run_dir) / "results/epr"
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    outputs, final_modes = _export_eigenmode(app, Path(run_dir), output_dir, spec)
    timings["final_export_seconds"] = round(time.perf_counter() - started, 6)
    frequencies = final_modes["eigenmodes"]["frequencies_ghz"]
    selected_modes = (
        tuple(range(1, spec.run_control.num_modes + 1))
        if spec.epr_request.mode_indices is None
        else spec.epr_request.mode_indices
    )
    rows: list[dict[str, Any]] = []
    raw_evidence: list[dict[str, Any]] = []
    mode_timings: list[dict[str, Any]] = []
    for mode in selected_modes:
        mode_started = time.perf_counter()
        raw, evidence = _evaluate_mode_integrals(
            app,
            spec,
            expressions,
            mode=mode,
            frequency_hz=float(frequencies[mode - 1]) * 1e9,
        )
        rows.append(
            {
                "mode": mode,
                "status": "complete",
                "raw_integrals": raw,
                "raw_integral_evidence": evidence,
                **combine_epr_mode(spec.geometry, raw, request=spec.epr_request),
            }
        )
        raw_evidence.append({"mode": mode, "expressions": evidence})
        mode_timings.append(
            {
                "mode": mode,
                "seconds": round(time.perf_counter() - mode_started, 6),
            }
        )
    timings["mode_integrals"] = mode_timings
    result_provenance = {
        "model_source_sha256": spec.geometry.model_sha256,
        "analysis_source_sha256": spec.geometry.source_sha256,
        "surface_analysis_scope": _SURFACE_ANALYSIS_SCOPE,
        "saved_solution_content_sha256": spec.saved_solution.content_sha256,
        "saved_solution_identity": detached(spec.saved_solution.identity),
        "saved_solution_aedt_version": spec.saved_solution.aedt_version,
        "requested_aedt_version": spec.aedt_version,
        "connected_aedt_version": connected_aedt_version,
        "requested_modes": list(selected_modes),
        "request": spec.epr_request.to_payload(),
        "raw_integral_evidence": raw_evidence,
    }
    if spec.surface_field_selectors:
        native_readback = authoring["native_expression_readback"]
        readback_path = work_root / native_readback["path"]
        outputs[readback_path.relative_to(run_dir).as_posix()] = native_readback["sha256"]
        result_provenance["surface_field_selectors"] = dict(spec.surface_field_selectors)
        result_provenance["surface_field_selector_bindings"] = [
            {
                **item,
                "native_readback": {
                    **item["native_readback"],
                    "path": (work_root / item["native_readback"]["path"]).relative_to(run_dir).as_posix(),
                },
            }
            for item in authoring["surface_field_selector_bindings"]
        ]
        result_provenance["native_expression_readback"] = {
            **native_readback,
            "path": readback_path.relative_to(run_dir).as_posix(),
        }
    result = EprResult(
        result_kind="saved_field",
        setup_name=spec.run_control.setup_name,
        rows=tuple(rows),
        _legacy_payload=spec._legacy_payload,
        provenance=result_provenance,
    )
    result_path = output_dir / "epr-result.json"
    started = time.perf_counter()
    write_json(result_path, result.to_payload())
    outputs[result_path.relative_to(run_dir).as_posix()] = file_sha256(result_path)
    timings["result_write_seconds"] = round(time.perf_counter() - started, 6)
    started = time.perf_counter()
    if not app.save_project() or not project_path.is_file():
        raise RuntimeError("saved EPR analysis workcopy save failed")
    timings["workcopy_save_seconds"] = round(time.perf_counter() - started, 6)
    project_relative = project_path.relative_to(run_dir).as_posix()
    outputs[project_relative] = file_sha256(project_path)
    native_references = cache.get("native_surface_references", [])
    if native_references:
        native_references = verify_saved_native_surface_references(
            project_path, spec.design_name, native_references
        )
        cache["native_surface_references"] = native_references
        for row in rows:
            _attach_saved_native_surface_membership(
                row["raw_integrals"], row, native_references
            )
        result = EprResult(
            result_kind="saved_field",
            setup_name=spec.run_control.setup_name,
            rows=tuple(rows),
            _legacy_payload=spec._legacy_payload,
            provenance=result_provenance,
        )
        started = time.perf_counter()
        write_json(result_path, result.to_payload())
        outputs[result_path.relative_to(run_dir).as_posix()] = file_sha256(result_path)
        timings["result_write_seconds"] = round(
            timings["result_write_seconds"] + time.perf_counter() - started, 6
        )
    return {
        "workflow_status": "epr_analysis_completed",
        "solver_invoked": False,
        "outputs": outputs,
        "connected": {
            "aedt_version": connected_aedt_version,
            "pyaedt_version": pyaedt_version(),
        },
        "project": project_relative,
        "geometry": geometry,
        "setup": read_body_eigenmode_setup(app, spec),
        "expressions": expressions,
        "cache": cache,
        "result": result.to_payload(),
        "timings": timings,
    }


def _analysis_readback(app, spec, cache, project_path):
    cache["serialized_readback"] = _read_cache(app, spec, cache["items"])
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
            _prepare_expressions_and_cache(app, workspace, bound, geometry, inset_plan)
        ),
        analysis_readback=_analysis_readback,
    )


def prepared_epr_result(prepared: PreparedBodyEigenmode) -> dict[str, Any]:
    result = prepared_body_eigenmode_result(
        prepared, analysis_readback=_prepared_analysis_readback
    )
    result["surface_analysis_scope"] = _SURFACE_ANALYSIS_SCOPE
    return result


def _export_epr_analysis(prepared, spec, convergence, outputs, output_dir, timings):
    saved_field_evidence = _saved_field_evidence(prepared.app, spec, convergence)
    started = time.perf_counter()
    history = _adaptive_mode_history(
        prepared.app,
        spec,
        prepared.cache["items"],
        prepared.cache["serialized_readback"]["items"],
    )
    run_dir = prepared.request.workspace
    history_path = output_dir / "adaptive-mode-history.json"
    write_json(history_path, history)
    outputs[history_path.relative_to(run_dir).as_posix()] = file_sha256(history_path)
    adaptive_result = _adaptive_epr_result(
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
            "surface_analysis_scope": _SURFACE_ANALYSIS_SCOPE,
            "saved_field_evidence": saved_field_evidence,
        },
    )


def solve_and_export_epr(
    prepared: PreparedBodyEigenmode,
    resources: AedtResources | None = None,
    resource_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return solve_and_export_body_eigenmode(
        prepared, resources, resource_evidence, analysis_export=_export_epr_analysis
    )


def collect_batch_epr(Hfss, run_dir, spec, prepared_receipt):
    """Bind saved batch outputs and collect full history without native authoring."""
    from scgsim.aedt.runtime.families.hfss import collect_body_eigenmode
    from scgsim.aedt.runtime.native.common import BoundAedtRequest
    from scgsim.aedt.epr.fields import _read_native_named_expressions
    from scgsim.aedt.epr.evidence import PreparedEprEvidence

    if not isinstance(spec, HfssEprSpec) or spec.epr_request is None or spec.run_control is None:
        raise TypeError("batch collection requires the original complete EPR request")
    evidence = PreparedEprEvidence.from_payload(
        prepared_receipt,
        evidence_root=run_dir / "batch-source/package/epr-expressions",
    )
    expressions = evidence.expression_payloads()
    cache = evidence.cache_payload()
    original_definitions = evidence.required_definitions_payload()
    project_path = run_dir / f"{spec.project_name}.aedt"
    app = Hfss(project=str(project_path), design=spec.design_name,
               solution_type="Eigenmode", new_desktop=False, close_on_exit=False)
    if app.design_name != spec.design_name or app.design_type != "HFSS":
        raise RuntimeError("returned batch design identity differs")
    geometry = bind_saved_planar_geometry(app, spec.geometry)
    bind_saved_analysis_surfaces(app, spec.geometry, geometry)
    _bind_saved_generic_rlcs(app, spec.geometry, geometry)
    _read_saved_junction_lines(app, project_path, spec.design_name, spec.geometry, geometry)
    setup = read_body_eigenmode_setup(app, spec)
    evidence_dir = run_dir / "metadata/epr_expressions"
    evidence_dir.mkdir(parents=True, exist_ok=False)
    observed = _read_native_named_expressions(
        app, names=list(original_definitions), definitions=[],
        evidence_dir=evidence_dir, expected_readback=original_definitions,
    )
    # Compare independent saved definitions; do not load or regenerate a CLC.
    cache["native_expression_readback"] = observed
    for expression in expressions:
        previous = expression["native_readback"]
        names = previous["expression_definition_names"]
        expression["native_readback"] = {
            **previous, "path": observed["path"], "sha256": observed["sha256"],
            "bytes": observed["bytes"], "status": observed["status"],
            "definitions": {name: observed["definitions"][name] for name in names},
        }
    cache["serialized_readback"] = _read_cache(app, spec, cache["items"])
    references = cache.get("native_surface_references", [])
    if references:
        cache["native_surface_references"] = verify_saved_native_surface_references(
            project_path, spec.design_name, references
        )
    prepared = PreparedBodyEigenmode(
        app, BoundAedtRequest.bind(run_dir, spec), project_path,
        geometry, setup, tuple(expressions), cache, {},
    )

    def complete_analysis(prepared, bound, convergence, outputs, output_dir, timings):
        history, payload, evidence = _export_epr_analysis(
            prepared, bound, convergence, outputs, output_dir, timings
        )
        _require_complete_batch_history(bound, prepared.cache, convergence, history, payload)
        return history, payload, evidence

    return collect_body_eigenmode(prepared, analysis_export=complete_analysis)


def _require_complete_batch_history(spec, cache, convergence, history, payload):
    """Require the requested original cache history, never a LastAdaptive substitute."""
    from scgsim.aedt.epr.cache import _trace_for_cache_context, _trace_by_pass

    final_pass = convergence["final_pass"]
    expected_passes = set(range(1, final_pass + 1))
    selected_modes = (
        tuple(range(1, spec.run_control.num_modes + 1))
        if spec.epr_request.mode_indices is None else spec.epr_request.mode_indices
    )
    cache_history = history["cache_integral_traces"]
    requested_titles = {item["title"] for item in cache["items"]}
    if (cache_history.get("status") != "complete"
            or set(cache_history.get("traces", {})) != requested_titles):
        raise RuntimeError("batch collection lacks the complete requested expression-cache history")
    for item in cache["items"]:
        title = item["title"]
        trace, context = _trace_for_cache_context(
            cache_history["traces"][title], item, cache_history["queries"].get(title, {})
        )
        if not context["status"].startswith("verified"):
            raise RuntimeError(f"batch cache context is unverified: {title}")
        if set(_trace_by_pass(trace, final_pass=final_pass)) != expected_passes:
            raise RuntimeError(f"batch cache pass history is incomplete: {title}")
    rows = payload["rows"]
    expected_rows = {(mode, native_pass) for mode in selected_modes for native_pass in expected_passes}
    if ({(row["mode"], row["native_pass"]) for row in rows} != expected_rows
            or len(rows) != len(expected_rows)
            or any(row["status"] != "complete" for row in rows)):
        raise RuntimeError("batch collection lacks complete per-mode/per-pass adaptive EPR")
