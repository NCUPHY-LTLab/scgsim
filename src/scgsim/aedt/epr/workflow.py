"""Prepare and execute the public saved-field EPR analysis request."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from scgsim.aedt.preparation.handoff import prepare_epr_analysis_handoff
from scgsim.aedt.results.epr import resolve_epr_result
from scgsim.aedt.runtime.transaction import ExecutionOptions, execute_handoff
from scgsim.aedt.specs.epr import EprAnalysisRequest, SavedSolution
from scgsim.aedt.specs.geometry import PreparedPlanarGeometry
from scgsim.aedt.specs.hfss_common import EigenmodeRunControl


def analyze_epr(  # noqa: PLR0913 -- preserve the supported keyword analysis API.
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

    plan = prepare_epr_analysis_handoff(
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
    previous = Path.cwd()
    try:
        exit_code = execute_handoff(
            plan.metadata_path, options=ExecutionOptions(analyze_epr=True)
        )
    finally:
        os.chdir(previous)
    if exit_code != 0:
        raise RuntimeError(
            "saved EPR analysis failed; inspect the prepared run receipt for the exact native error"
        )
    return resolve_epr_result(plan.run_dir / "results/epr/epr-result.json")
