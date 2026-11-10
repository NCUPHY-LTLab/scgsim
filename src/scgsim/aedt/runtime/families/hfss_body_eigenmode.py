"""Own source-backed eigenmode preparation, solving and collection."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scgsim.aedt.io import file_sha256, write_json
from scgsim.aedt.results.convergence.hfss import read_hfss_convergence
from scgsim.aedt.runtime.benchmark import (
    export_simulation_benchmark,
)
from scgsim.aedt.runtime.families.hfss_exports import export_eigenmode
from scgsim.aedt.runtime.families.hfss_history import read_body_eigenmode_history
from scgsim.aedt.runtime.families.hfss_setup import read_body_eigenmode_setup
from scgsim.aedt.runtime.native.common import (
    BoundAedtRequest,
    detached_data,
    pyaedt_version,
)
from scgsim.aedt.runtime.native.hfss_eigenmode import (
    create_eigenmode_setup,
)
from scgsim.aedt.runtime.native.hfss_geometry import (
    prepare_native_planar_geometry,
)
from scgsim.aedt.runtime.native.resources import analyze_with_resources
from scgsim.aedt.runtime.native.saved_geometry import read_saved_junction_lines
from scgsim.aedt.specs.common import (
    AedtResources,
)
from scgsim.aedt.specs.geometry import PreparedPlanarGeometry
from scgsim.aedt.specs.hfss import (
    HfssEprSpec,
)


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


def prepare_body_eigenmode(  # noqa: PLR0913 -- preserve the family geometry/analysis extension callback protocol.
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
    timings["native_open_seconds"] = round(time.perf_counter() - started, 6)
    app.modeler.model_units = "um"
    started = time.perf_counter()
    geometry = prepare_native_planar_geometry(app, bound.geometry)
    if geometry_extension is not None:
        geometry_extension(app, bound.geometry, geometry)
    timings["geometry_seconds"] = round(time.perf_counter() - started, 6)
    timings["geometry_phases"] = geometry.pop("geometry_phases")
    started = time.perf_counter()
    if bound.run_control is not None:
        create_eigenmode_setup(app, bound.run_control)
    timings["setup_seconds"] = round(time.perf_counter() - started, 6)
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
    timings["epr_authoring_seconds"] = round(time.perf_counter() - started, 6)
    started = time.perf_counter()
    if not app.save_project() or not project_path.is_file():
        raise RuntimeError("HFSS EPR project was not saved after preparation")
    timings["save_seconds"] = round(time.perf_counter() - started, 6)
    started = time.perf_counter()
    read_saved_junction_lines(
        app, project_path, bound.design_name, bound.geometry, geometry
    )
    setup = (
        read_body_eigenmode_setup(app, bound) if bound.run_control is not None else None
    )
    if analysis_readback is not None and bound.run_control is not None:
        analysis_readback(app, bound, cache, project_path)
    timings["readback_seconds"] = round(time.perf_counter() - started, 6)
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
    timings["solve_seconds"] = round(time.perf_counter() - started, 6)
    if not solved:
        raise RuntimeError(
            f"HFSS failed to analyze setup {spec.run_control.setup_name!r}"
        )
    return collect_body_eigenmode(
        prepared, analysis_export=analysis_export, timings=timings
    )


def collect_body_eigenmode(
    prepared: PreparedBodyEigenmode,
    *,
    analysis_export: Callable | None = None,
    timings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Export one already-solved owned project without Analyze or model creation."""
    spec = prepared.request.parse()
    if not isinstance(spec, HfssEprSpec) or spec.run_control is None:
        raise TypeError(
            "native collection requires a complete body-first Eigenmode request"
        )
    phase_timings: dict[str, Any] = (
        detached_data(prepared.timings) if timings is None else timings
    )
    started = time.perf_counter()
    if not prepared.app.save_project() or not prepared.project_path.is_file():
        raise RuntimeError("HFSS EPR solved project save failed")
    phase_timings["post_solve_save_seconds"] = round(time.perf_counter() - started, 6)
    run_dir = prepared.request.workspace
    output_dir = run_dir / "results/epr"
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    outputs, final_modes = export_eigenmode(prepared.app, run_dir, output_dir, spec)
    phase_timings["final_export_seconds"] = round(time.perf_counter() - started, 6)
    convergence_path = output_dir / "adaptive-convergence.prop"
    started = time.perf_counter()
    # Native empty VariationKeys selects the current nominal variation without
    # PyAEDT evaluating postprocessing variables before the export.
    prepared.app.odesign.ExportConvergence(
        spec.run_control.setup_name, "", str(convergence_path)
    )
    if not convergence_path.is_file():
        raise RuntimeError("HFSS adaptive convergence export is missing")
    outputs[convergence_path.relative_to(run_dir).as_posix()] = file_sha256(
        convergence_path
    )
    convergence = read_hfss_convergence(run_dir, spec)
    phase_timings["convergence_export_seconds"] = round(
        time.perf_counter() - started, 6
    )
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
            prepared, spec, convergence, outputs, output_dir, phase_timings
        )
    if analysis_export is None:
        phase_timings["history_export_seconds"] = round(
            time.perf_counter() - started, 6
        )
    started = time.perf_counter()
    if not prepared.app.save_project() or not prepared.project_path.is_file():
        raise RuntimeError("HFSS EPR final project save failed")
    phase_timings["final_save_seconds"] = round(time.perf_counter() - started, 6)
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
        "timings": phase_timings,
        "convergence": convergence,
        "result_readback": {
            "final_modes": final_modes,
            "adaptive_history": history,
            "adaptive_epr_result": (adaptive_payload),
        },
        "save": {"ok": True, "project_sha256": outputs[relative]},
    }
