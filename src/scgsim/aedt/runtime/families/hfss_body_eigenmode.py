"""Body-first HFSS Eigenmode: setup readback, history and the solve/export path.

Owns the eigenmode setup and solution-history readback and the prepare, solve
and export of a body-first eigenmode request. It does not own driven HFSS or
the EPR analysis extensions."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scgsim.aedt._io import file_sha256, write_json
from scgsim.aedt.results.convergence.hfss import read_hfss_convergence
from scgsim.aedt.runtime._timing import elapsed_seconds
from scgsim.aedt.runtime.benchmark import (
    export_simulation_benchmark,
)
from scgsim.aedt.runtime.families.hfss_export import export_eigenmode
from scgsim.aedt.runtime.native.common import (
    BoundAedtRequest,
    analyze_with_resources,
    detached_data,
    pyaedt_version,
    saved_setup_properties,
)
from scgsim.aedt.runtime.native.hfss_eigenmode import (
    create_eigenmode_setup,
    read_eigenmode_setup,
)
from scgsim.aedt.runtime.native.hfss_geometry import prepare_native_planar_geometry
from scgsim.aedt.runtime.native.hfss_junction_readback import read_saved_junction_lines
from scgsim.aedt.runtime.native.hfss_lumped import (
    assign_generic_rlcs,
    read_generic_rlcs,
)
from scgsim.aedt.specs.common import (
    REQUIRED_AEDT_VERSION,
    AedtResources,
)
from scgsim.aedt.specs.geometry import PreparedPlanarGeometry
from scgsim.aedt.specs.hfss_prepared import (
    HfssEprAnalysisSpec,
    HfssEprSpec,
)


def read_body_eigenmode_setup(
    app: Any, spec: HfssEprSpec | HfssEprAnalysisSpec
) -> dict[str, Any]:
    """Read the saved eigenmode setup and compare it with the requested run control."""
    setup = read_eigenmode_setup(app, spec.run_control)
    raw = saved_setup_properties(app, spec.run_control.setup_name)
    creating_setup = isinstance(spec, HfssEprSpec)
    saved_field_properties = {
        "basis": (
            "completed Eigenmode LastAdaptive Fields evidence is checked after solve"
            if creating_setup
            else "existing saved-field analysis does not re-author setup field policy"
        ),
        "serialized": {
            key: raw[key]
            for key in ("SaveAnyFields", "SaveRadFieldsOnly")
            if key in raw
        },
        "status": (
            "unexposed_pending_postsolve"
            if creating_setup
            else "not_reauthored_existing_saved_solution"
        ),
    }
    return {
        "name": setup["name"],
        "native": setup["native"],
        "saved_fields": saved_field_properties,
    }


def _frequency_solution_trace(
    data: Any, expression: str, *, category: str, solution: str
) -> dict[str, Any]:
    """Retain raw frequency and optional explicit native SI evidence."""

    trace = solution_trace(data, expression)
    if trace["unit"] != "":
        return trace
    evidence: dict[str, Any] = {
        "status": "unavailable",
        "query": {
            "category": category,
            "solution": solution,
            "expression": expression,
            "sweeps": {"Pass": "All"},
            "si_value": True,
            "real_method": "GetRealDataValues",
            "imaginary_method": "GetImagDataValues",
        },
        "unit": "Hz",
        "variations": [],
    }
    trace["si_evidence"] = evidence
    # SI evidence is optional; failure must retain the original partial trace.
    try:
        import numpy as np

        if len(data._original_data) != len(data.variations):  # noqa: SLF001 - PyAEDT has no public API for this
            raise RuntimeError("native SI frequency variation count differs")
        real_rows: list[list[Any]] = []
        imaginary_rows: list[list[Any]] = []
        for index, native in enumerate(data._original_data):  # noqa: SLF001 - PyAEDT has no public API for this
            variation = data.variations[index]
            observed: dict[str, Any] = {
                "index": index,
                "coordinates": dict(variation),
                "intrinsics": {
                    name: values.tolist()
                    for name, values in data.intrinsics_by_variation(index).items()
                },
            }
            evidence["variations"].append(observed)
            observed["real_values"] = list(native.GetRealDataValues(expression, True))
            observed["imaginary_values"] = list(
                native.GetImagDataValues(expression, True)
            )
            real_rows.extend(
                data._full_keys(  # noqa: SLF001 - PyAEDT has no public API for this
                    variation, np.asarray(observed["real_values"], dtype=float), index
                ).tolist()
            )
            imaginary_rows.extend(
                data._full_keys(  # noqa: SLF001 - PyAEDT has no public API for this
                    variation,
                    np.asarray(observed["imaginary_values"], dtype=float),
                    index,
                ).tolist()
            )
        coordinates = [row[:-1] for row in trace["real_rows"]]
        if not (
            coordinates
            == [row[:-1] for row in real_rows]
            == [row[:-1] for row in imaginary_rows]
            == [row[:-1] for row in trace["imaginary_rows"]]
        ):
            raise RuntimeError("native SI frequency coordinates differ from raw trace")
        evidence.update(
            status="available",
            columns=trace["columns"],
            real_rows=real_rows,
            imaginary_rows=imaginary_rows,
        )
    except Exception as exc:  # noqa: BLE001 -- retain optional native SI failure.
        evidence["error"] = f"{type(exc).__name__}: {exc}"
    return trace


def solution_trace(data: Any, expression: str) -> dict[str, Any]:
    """Detach one native trace while retaining its coordinate columns."""

    real, imaginary = data.full_matrix_real_imag
    variation_keys = tuple(data.variations[0]) if data.variations else ()
    intrinsic_keys = tuple(data.intrinsics_by_variation(0)) if data.variations else ()
    if any(
        tuple(variation) != variation_keys
        or tuple(data.intrinsics_by_variation(index)) != intrinsic_keys
        for index, variation in enumerate(data.variations)
    ):
        raise RuntimeError(
            "native adaptive trace coordinate columns differ by variation"
        )
    columns = [*variation_keys, *intrinsic_keys, "value"]
    real_rows = real[expression].tolist()
    imaginary_rows = imaginary[expression].tolist()
    if any(len(row) != len(columns) for row in (*real_rows, *imaginary_rows)):
        raise RuntimeError("native adaptive trace coordinate width differs")
    return {
        "columns": columns,
        "real_rows": real_rows,
        "imaginary_rows": imaginary_rows,
        "unit": data.units_data.get(expression),
    }


def read_body_eigenmode_history(app: Any, spec: HfssEprSpec) -> dict[str, Any]:
    """Read native adaptive frequency history independently of EPR cache analysis."""
    category = "Eigenmode Parameters"
    solution = f"{spec.run_control.setup_name} : AdaptivePass"
    frequency_solutions = [
        str(item) for item in app.post.available_report_solutions(category) or ()
    ]
    field_solutions = [
        str(item) for item in app.post.available_report_solutions("Fields") or ()
    ]
    if solution not in frequency_solutions:
        raise RuntimeError(
            f"native adaptive solution is unavailable for {category!r}: {frequency_solutions!r}"
        )
    quantities: list[str] = []
    for quantity_category in (
        app.post.available_quantities_categories(category, solution=solution) or ()
    ):
        for quantity in (
            app.post.available_report_quantities(
                category,
                solution=solution,
                quantities_category=quantity_category,
            )
            or ()
        ):
            if str(quantity).startswith("Mode(") and quantity not in quantities:
                quantities.append(str(quantity))
    expected = [f"Mode({index})" for index in range(1, spec.run_control.num_modes + 1)]
    if len(quantities) != len(expected) or set(quantities) != set(expected):
        raise RuntimeError(
            f"native adaptive Eigenmode quantities differ: {quantities!r}"
        )
    data = app.post.get_solution_data_per_variation(
        category, solution, [], {"Pass": "All"}, quantities
    )
    if not data:
        raise RuntimeError("native adaptive Eigenmode history is unavailable")
    data.primary_sweep = "Pass"
    rows = {
        quantity: _frequency_solution_trace(
            data, quantity, category=category, solution=solution
        )
        for quantity in expected
    }
    return {
        "schema_version": "scgsim.aedt.epr-adaptive-history.v1",
        "category": category,
        "solution": solution,
        "primary_sweep": data.primary_sweep,
        "pass_axis": data.primary_sweep_values.tolist(),
        "pass_unit": data.units_sweeps.get("Pass"),
        "frequency_traces": rows,
        "cache_integral_traces": {
            "status": "not_requested",
            "requested_titles": [],
            "items": [],
            "traces": {},
            "queries": {},
        },
        "report_inventory": {
            "eigenmode_parameter_solutions": frequency_solutions,
            "field_solutions": field_solutions,
        },
        "frequency_unit_status": "native_reported_only",
    }


@dataclass(frozen=True)
class PreparedBodyEigenmode:
    """Carrier for one mutable native application and detached preparation evidence."""

    app: Any
    request: BoundAedtRequest
    project_path: Path
    geometry: dict[str, Any]
    setup: dict[str, Any] | None
    expressions: tuple[dict[str, Any], ...]
    cache: dict[str, Any]
    timings: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.request, BoundAedtRequest):
            raise TypeError("PreparedBodyEigenmode requires a bound AEDT request")
        object.__setattr__(self, "geometry", detached_data(self.geometry))
        object.__setattr__(self, "setup", detached_data(self.setup))
        object.__setattr__(
            self, "expressions", tuple(detached_data(item) for item in self.expressions)
        )
        object.__setattr__(self, "cache", detached_data(self.cache))
        object.__setattr__(self, "timings", detached_data(self.timings))


def prepare_body_eigenmode(  # noqa: PLR0913 - three optional keyword-only hooks supplied by aedt/epr/workflow.py
    Hfss: Any,
    run_dir: Path,
    spec: HfssEprSpec,
    *,
    geometry_extension: Callable[[Any, PreparedPlanarGeometry, dict[str, Any]], None]
    | None = None,
    analysis_author: Callable[
        [Any, Path, HfssEprSpec, dict[str, Any]],
        tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]],
    ]
    | None = None,
    analysis_readback: Callable[[Any, HfssEprSpec, dict[str, Any], Path], None]
    | None = None,
) -> PreparedBodyEigenmode:
    """Create, save, and read back one body-first no-solve Eigenmode model."""

    request = BoundAedtRequest.bind(run_dir, spec)
    bound = request.parse()
    if not isinstance(bound, HfssEprSpec):
        raise TypeError("bound EPR request did not retain its schema")
    if bound.modeling not in {"solid", "thin_film"} or (
        bound.geometry.modeling != bound.modeling
    ):
        raise ValueError("new EPR HFSS preparation requires explicit modeling")
    project_path = request.workspace / f"{bound.project_name}.aedt"
    timings: dict[str, Any] = {}
    preparation_started = time.perf_counter()
    started = time.perf_counter()
    app = Hfss(
        project=str(project_path),
        design=bound.design_name,
        solution_type="Eigenmode",
        new_desktop=False,
        close_on_exit=False,
    )
    timings["native_open_seconds"] = elapsed_seconds(started)
    if app.desktop_class.aedt_version_id != REQUIRED_AEDT_VERSION:
        raise RuntimeError("HFSS EPR did not bind the owned AEDT 2024.2 desktop")
    app.modeler.model_units = "um"
    started = time.perf_counter()
    geometry = prepare_native_planar_geometry(app, bound.geometry)
    if geometry_extension is not None:
        geometry_extension(app, bound.geometry, geometry)
    assign_generic_rlcs(app, bound, geometry)
    timings["geometry_seconds"] = elapsed_seconds(started)
    timings["geometry_phases"] = geometry.pop("geometry_phases")
    started = time.perf_counter()
    if bound.run_control is not None:
        create_eigenmode_setup(app, bound.run_control)
    timings["setup_seconds"] = elapsed_seconds(started)
    started = time.perf_counter()
    if analysis_author is None or bound.run_control is None:
        expressions: list[dict[str, Any]] = []
        cache = {
            "schema_version": "scgsim.aedt.epr-cache-request.v1",
            "status": "disabled",
            "reason": (
                "no solve setup was supplied"
                if bound.run_control is None
                else "no EPR analysis request was supplied"
            ),
        }
    else:
        expressions, cache, phase_timings = analysis_author(
            app, request.workspace, bound, geometry
        )
        timings["epr_authoring_phases"] = phase_timings
    timings["epr_authoring_seconds"] = elapsed_seconds(started)
    started = time.perf_counter()
    if not app.save_project() or not project_path.is_file():
        raise RuntimeError("HFSS EPR project was not saved after preparation")
    timings["save_seconds"] = elapsed_seconds(started)
    started = time.perf_counter()
    read_saved_junction_lines(
        app, project_path, bound.design_name, bound.geometry, geometry
    )
    read_generic_rlcs(app, geometry)
    setup = (
        read_body_eigenmode_setup(app, bound) if bound.run_control is not None else None
    )
    if analysis_readback is not None and bound.run_control is not None:
        analysis_readback(app, bound, cache, project_path)
    timings["readback_seconds"] = elapsed_seconds(started)
    preparation_wall = time.perf_counter() - preparation_started
    timings["preparation_wall_seconds"] = round(preparation_wall, 6)
    timings["preparation_residual_seconds"] = round(
        max(
            0.0,
            preparation_wall
            - sum(
                timings[key]
                for key in (
                    "native_open_seconds",
                    "geometry_seconds",
                    "setup_seconds",
                    "epr_authoring_seconds",
                    "save_seconds",
                    "readback_seconds",
                )
            ),
        ),
        6,
    )
    return PreparedBodyEigenmode(
        app,
        request,
        project_path,
        geometry,
        setup,
        tuple(expressions),
        cache,
        timings,
    )


def prepared_body_eigenmode_result(
    prepared: PreparedBodyEigenmode,
    *,
    analysis_readback: Callable[[Any, HfssEprSpec, dict[str, Any], Path], None]
    | None = None,
) -> dict[str, Any]:
    """Return the canonical no-solve result without impersonating completion."""

    spec = prepared.request.parse()
    if not isinstance(spec, HfssEprSpec):
        raise TypeError("bound EPR request did not retain its schema")
    if not prepared.app.save_project() or not prepared.project_path.is_file():
        raise RuntimeError("HFSS EPR project final preparation save failed")
    if analysis_readback is not None and spec.run_control is not None:
        analysis_readback(prepared.app, spec, prepared.cache, prepared.project_path)
    relative = prepared.project_path.relative_to(prepared.request.workspace).as_posix()
    digest = file_sha256(prepared.project_path)
    result = {
        "workflow_status": "native_preparation_only",
        "solver_invoked": False,
        "outputs": {relative: digest},
        "connected": {
            "aedt_version": prepared.app.desktop_class.aedt_version_id,
            "pyaedt_version": pyaedt_version(),
        },
        "project": relative,
        "geometry": detached_data(prepared.geometry),
        "setup": detached_data(prepared.setup),
        "expressions": detached_data(prepared.expressions),
        "cache": detached_data(prepared.cache),
        "timings": detached_data(prepared.timings),
        "save": {"ok": True, "project_sha256": digest},
    }
    return result


def solve_and_export_body_eigenmode(
    prepared: PreparedBodyEigenmode,
    resources: AedtResources | None = None,
    resource_evidence: dict[str, Any] | None = None,
    *,
    analysis_export: Callable[
        [
            PreparedBodyEigenmode,
            HfssEprSpec,
            dict[str, Any],
            dict[str, str],
            Path,
            dict[str, Any],
        ],
        tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any]],
    ]
    | None = None,
) -> dict[str, Any]:
    """Run the explicit setup once and export native final/history evidence."""

    spec = prepared.request.parse()
    if not isinstance(spec, HfssEprSpec):
        raise TypeError("body-first solve/export requires HfssEprSpec")
    if spec.run_control is None:
        raise ValueError("Eigenmode solving requires an explicit complete run_control")
    timings = detached_data(prepared.timings)
    started = time.perf_counter()
    solved = analyze_with_resources(
        prepared.app,
        spec.run_control.setup_name,
        prepared.request.workspace,
        resources,
        resource_evidence if resource_evidence is not None else {},
    )
    timings["solve_seconds"] = elapsed_seconds(started)
    if not solved:
        raise RuntimeError(
            f"HFSS failed to analyze setup {spec.run_control.setup_name!r}"
        )
    started = time.perf_counter()
    if not prepared.app.save_project() or not prepared.project_path.is_file():
        raise RuntimeError("HFSS EPR solved project save failed")
    timings["post_solve_save_seconds"] = elapsed_seconds(started)
    run_dir = prepared.request.workspace
    output_dir = run_dir / "results/epr"
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    outputs, final_modes = export_eigenmode(prepared.app, run_dir, output_dir, spec)
    timings["final_export_seconds"] = elapsed_seconds(started)
    convergence_path = output_dir / "adaptive-convergence.prop"
    started = time.perf_counter()
    returned = prepared.app.export_convergence(
        spec.run_control.setup_name,
        variations="",
        output_file=str(convergence_path),
    )
    if (
        Path(returned).resolve() != convergence_path.resolve()
        or not convergence_path.is_file()
    ):
        raise RuntimeError("HFSS adaptive convergence export is missing")
    outputs[convergence_path.relative_to(run_dir).as_posix()] = file_sha256(
        convergence_path
    )
    convergence = read_hfss_convergence(run_dir, spec)
    timings["convergence_export_seconds"] = elapsed_seconds(started)
    started = time.perf_counter()
    if analysis_export is None:
        history = read_body_eigenmode_history(prepared.app, spec)
        adaptive_payload = None
        analysis_evidence: dict[str, Any] = {}
        history_path = output_dir / "adaptive-mode-history.json"
        write_json(history_path, history)
        outputs[history_path.relative_to(run_dir).as_posix()] = file_sha256(
            history_path
        )
    else:
        history, adaptive_payload, analysis_evidence = analysis_export(
            prepared, spec, convergence, outputs, output_dir, timings
        )
    if analysis_export is None:
        timings["history_export_seconds"] = elapsed_seconds(started)
    started = time.perf_counter()
    if not prepared.app.save_project() or not prepared.project_path.is_file():
        raise RuntimeError("HFSS EPR final project save failed")
    timings["final_save_seconds"] = elapsed_seconds(started)
    relative = prepared.project_path.relative_to(run_dir).as_posix()
    outputs[relative] = file_sha256(prepared.project_path)
    benchmark_relative, benchmark_digest, benchmark = export_simulation_benchmark(
        prepared.app, run_dir, spec.run_control.setup_name
    )
    outputs[benchmark_relative] = benchmark_digest
    return {
        "workflow_status": "completed",
        "solver_invoked": True,
        "outputs": outputs,
        "benchmark": benchmark,
        "connected": {
            "aedt_version": prepared.app.desktop_class.aedt_version_id,
            "pyaedt_version": pyaedt_version(),
        },
        "project": relative,
        "geometry": detached_data(prepared.geometry),
        "setup": detached_data(prepared.setup),
        "expressions": detached_data(prepared.expressions),
        "cache": detached_data(prepared.cache),
        **analysis_evidence,
        "timings": timings,
        "convergence": convergence,
        "result_readback": {
            "final_modes": final_modes,
            "adaptive_history": history,
            "adaptive_epr_result": (adaptive_payload),
        },
        "save": {"ok": True, "project_sha256": outputs[relative]},
    }
