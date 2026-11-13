"""The public entry of the semantic geometry builder: an effective input to XAO, BREP and sidecar.

:func:`build_sgb_geometry` runs the whole pipeline for one route -- prisms, snap-rounded arrangements,
labelling, assembly, route plan, structural validation (always; a failed invariant raises) -- and then
emits: the planned topology is numbered in ``TopExp::MapShapes`` order and serialised directly as BREP
and XAO (no CAD kernel, no Boolean), next to the sidecar that binds planned ids to native tags, the
sliver report and the validation report.

The mesher takes it from there: open the XAO in Gmsh, re-apply the sidecar's embeddings, mesh.  It trusts
the XAO and sidecar of one build and does not check them again; the tests do that with ``verify_loaded``
(``scgsim.geometry.sgb.output.verification.loaded``), which also serves to diagnose a file suspected to be wrong.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scgsim.geometry.models.input import GeometryBuildInput
from scgsim.geometry.models.representation import RepresentationContext
from scgsim.geometry.sgb.bands.requests import InsetRequest
from scgsim.geometry.sgb.checks.slivers.report import sliver_report
from scgsim.geometry.sgb.exact.stopwatch import elapsed_seconds
from scgsim.geometry.sgb.model import ConformalModel, build_conformal_model
from scgsim.geometry.sgb.output.emission.plan import plan_brep
from scgsim.geometry.sgb.output.emission.writer import write_brep_file, write_xao_file
from scgsim.geometry.sgb.output.sidecar.build import SidecarContext, build_sidecar
from scgsim.geometry.sgb.output.sidecar.files import write_sidecar


@dataclass(frozen=True)
class SgbArtifacts:
    """The files of one build and what the build measured (timings and counts are not part of any file)."""

    xao: Path
    brep: Path
    sidecar: Path
    sliver_report: Path
    validation_report: Path
    route: str
    counts: dict[str, int]
    timings: dict[str, float] = field(default_factory=dict)


def build_sgb_geometry(  # noqa: PLR0913 - the effective input, the route and the build options
    build_input: GeometryBuildInput,
    representation: RepresentationContext,
    *,
    route: str,
    insets: tuple[InsetRequest, ...] = (),
    output_dir: Path,
) -> SgbArtifacts:
    """Build the conformal solid model of an effective input under ``route`` and write its files to ``output_dir``.

    ``insets`` are the typed Surface-EPR band requests of ``scgsim.geometry.sgb.bands.requests``.  Structural
    validation always runs and raises :class:`~scgsim.geometry.sgb.errors.ValidationError` before anything is
    written.
    """
    timings: dict[str, float] = {}
    started = time.perf_counter()
    model = build_conformal_model(
        build_input,
        representation,
        route,
        insets=insets,
    )
    timings["derive_and_validate"] = elapsed_seconds(started)
    return emit_model(model, output_dir, timings=timings)


def emit_model(
    model: ConformalModel,
    output_dir: Path,
    *,
    timings: dict[str, float] | None = None,
) -> SgbArtifacts:
    """Write the files of an already built and validated model (the second half of :func:`build_sgb_geometry`).

    ``timings`` collects the seconds of each emission step when given.
    """
    route = model.policy.name
    timings = {} if timings is None else timings
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"semantic_geometry_route_{route.lower()}"
    paths = {
        "xao": output_dir / f"{stem}.xao",
        "brep": output_dir / f"{stem}.brep",
        "sidecar": output_dir / f"{stem}.sidecar.json",
        "sliver_report": output_dir / f"{stem}.slivers.json",
        "validation_report": output_dir / f"{stem}.validation.json",
    }
    started = time.perf_counter()
    plan = plan_brep(model.topology, model.routed)
    timings["plan"] = elapsed_seconds(started)
    timings["write_brep"] = write_brep_file(plan, paths["brep"])
    timings["write_xao"] = write_xao_file(
        plan, paths["xao"], name=stem, brep=paths["brep"]
    )
    started = time.perf_counter()
    slivers = sliver_report(model.labelled, model.topology, model.routed)
    _write_json(paths["sliver_report"], slivers)
    # wall times are not part of any file: every file is the same for every run of the same model
    validation = {k: v for k, v in model.report.to_dict().items() if k != "timings_s"}
    _write_json(paths["validation_report"], {**validation, "counts": model.counts})
    timings["reports"] = elapsed_seconds(started)
    started = time.perf_counter()
    context = SidecarContext(
        files={key: path.name for key, path in paths.items()},
        sliver_summary=slivers["summary"],
        name=stem,
    )
    write_sidecar(paths["sidecar"], build_sidecar(model, plan, context))
    timings["sidecar"] = elapsed_seconds(started)
    return SgbArtifacts(
        paths["xao"],
        paths["brep"],
        paths["sidecar"],
        paths["sliver_report"],
        paths["validation_report"],
        route,
        {**model.counts, **plan.counts},
        timings,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_bytes(
        (json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        )
    )
