"""The public entry of the semantic geometry builder: an effective input to XAO, BREP and sidecar.

:func:`build_sgb_geometry` runs the whole pipeline for one route -- prisms, snap-rounded arrangements,
labelling, assembly, route plan, structural validation (always; a failed invariant raises) -- and then
emits: the planned topology is numbered in ``TopExp::MapShapes`` order and serialised directly as BREP
and XAO (no CAD kernel, no Boolean), next to the sidecar that binds planned ids to native tags, the
sliver report and the validation report.  With ``strict=True`` the written XAO is also read back with
OCP and put through OpenCASCADE's own checks (the optional ``occt`` dependency group).

The mesher takes it from there: open the XAO in Gmsh, re-apply the sidecar's embeddings, mesh.  It trusts
the XAO and sidecar of one build and does not check them again; the tests do that with ``verify_loaded``
(``scgsim.geometry.sgb.verification.loaded``), which also serves to diagnose a file suspected to be wrong.

Every file is byte-identical for any ``workers`` and ``tiles`` value.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scgsim.geometry.models.input import GeometryBuildInput
from scgsim.geometry.models.representation import RepresentationContext
from scgsim.geometry.sgb.cutfamilies.requests import InsetRequest, RefinementRequest
from scgsim.geometry.sgb.emission.plan import BrepPlan, plan_brep
from scgsim.geometry.sgb.emission.writer import write_brep_file, write_xao_file
from scgsim.geometry.sgb.model import ConformalModel, build_conformal_model
from scgsim.geometry.sgb.routes.policy import RoutePolicy
from scgsim.geometry.sgb.sidecar.build import SidecarContext, build_sidecar
from scgsim.geometry.sgb.sidecar.files import write_sidecar
from scgsim.geometry.sgb.slivers.report import sliver_report
from scgsim.geometry.sgb.stopwatch import elapsed_seconds
from scgsim.geometry.sgb.verification.ocp import require_ocp

if TYPE_CHECKING:
    from scgsim.geometry.sgb.verification.occt import OcctReport


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
    occt: OcctReport | None = None  # strict mode: the OCCT report of the written XAO


def build_sgb_geometry(  # noqa: PLR0913 - the effective input, the route and the build options
    build_input: GeometryBuildInput,
    representation: RepresentationContext,
    *,
    route: str | RoutePolicy,
    insets: tuple[InsetRequest, ...] = (),
    refinements: tuple[RefinementRequest, ...] = (),
    output_dir: Path,
    workers: int = 1,
    tiles: int = 1,
    strict: bool = False,
) -> SgbArtifacts:
    """Build the conformal solid model of an effective input under ``route`` and write its files to ``output_dir``.

    ``insets`` (Surface-EPR bands) and ``refinements`` (mesh-size curves) are the typed requests of
    ``scgsim.geometry.sgb.cutfamilies.requests``.  Structural validation always runs and raises
    :class:`~scgsim.geometry.sgb.errors.ValidationError` before anything is written.  ``strict=True``
    needs OCP and raises :class:`~scgsim.geometry.sgb.verification.ocp.OcctUnavailable` (naming the
    ``occt`` dependency group) before the build starts when it is missing, and
    :class:`~scgsim.geometry.sgb.verification.occt.OcctCheckError` when the written file has a finding.
    """
    if strict:
        require_ocp()
    timings: dict[str, float] = {}
    started = time.perf_counter()
    model = build_conformal_model(
        build_input,
        representation,
        route,
        insets=insets,
        refinements=refinements,
        tiles=tiles,
        workers=workers,
    )
    timings["derive_and_validate"] = elapsed_seconds(started)
    return emit_model(
        model, output_dir, workers=workers, strict=strict, timings=timings
    )


def emit_model(
    model: ConformalModel,
    output_dir: Path,
    *,
    workers: int = 1,
    strict: bool = False,
    timings: dict[str, float] | None = None,
) -> SgbArtifacts:
    """Write the files of an already built and validated model (the second half of :func:`build_sgb_geometry`).

    ``timings`` collects the seconds of each emission step when given.
    """
    if strict:
        require_ocp()
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
    timings["write_brep"] = write_brep_file(plan, paths["brep"], workers=workers)
    timings["write_xao"] = write_xao_file(
        plan, paths["xao"], name=stem, workers=workers, brep=paths["brep"]
    )
    started = time.perf_counter()
    slivers = sliver_report(model)
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
    occt = _strict_check(paths["xao"], model, plan, timings) if strict else None
    return SgbArtifacts(
        paths["xao"],
        paths["brep"],
        paths["sidecar"],
        paths["sliver_report"],
        paths["validation_report"],
        route,
        {**model.counts, **plan.counts},
        timings,
        occt,
    )


def _strict_check(
    xao: Path, model: ConformalModel, plan: BrepPlan, timings: dict[str, float]
) -> OcctReport:
    """The OCCT checks on the written XAO; the expensive self-interference pre-check stays off."""
    from scgsim.geometry.sgb.verification.occt import check_file  # noqa: PLC0415 -- optional OCP, see _require_occt

    started = time.perf_counter()
    report = check_file(
        xao,
        model.topology,
        plan,
        tolerance=plan.tolerance,
        self_interference=False,
        strict=True,
    )
    timings["occt"] = elapsed_seconds(started)
    return report


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_bytes(
        (json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        )
    )
