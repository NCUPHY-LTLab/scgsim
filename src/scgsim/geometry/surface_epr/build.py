"""Build the SGB geometry of one route with the Surface-EPR bands its specs ask for.

Normally SGB finds the faces to band from the type-level requests of the specs, in one build.  Only when a
spec customises conductors (``inset_margins_by_entity_um``) are their faces needed first, so the same input
is built once without bands, in a scratch folder that is removed, to read them from its sidecar (see
:mod:`.insets`).  Palace and AEDT build their geometry through this one step.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scgsim.geometry.models.input import GeometryBuildInput
from scgsim.geometry.models.representation import RepresentationContext
from scgsim.geometry.sgb import SgbArtifacts, build_sgb_geometry
from scgsim.geometry.sgb.cutfamilies.requests import InsetRequest
from scgsim.geometry.sgb.routes.policy import RoutePolicy

from .insets import inset_requests, needs_unbanded_build, requests_wanted

_UNBANDED_DIR = ".unbanded"


def build_banded_geometry(
    build_input: GeometryBuildInput,
    representation: RepresentationContext,
    *,
    route: str | RoutePolicy,
    surface_epr_specs: Mapping[str, Mapping[str, Any]] | None,
    output_dir: Path,
) -> SgbArtifacts:
    """Build ``route``'s geometry into ``output_dir`` with the bands of the normalized ``surface_epr_specs``."""
    specs = surface_epr_specs or {}
    insets: tuple[InsetRequest, ...] = ()
    if requests_wanted(specs):
        unbanded = (
            _unbanded_sidecar(build_input, representation, route, output_dir)
            if needs_unbanded_build(specs)
            else None
        )
        insets = inset_requests(specs, unbanded)
    return build_sgb_geometry(
        build_input,
        representation,
        route=route,
        insets=insets,
        output_dir=output_dir,
    )


def _unbanded_sidecar(
    build_input: GeometryBuildInput,
    representation: RepresentationContext,
    route: str | RoutePolicy,
    output_dir: Path,
) -> dict[str, Any]:
    """The sidecar of a build without bands; the build itself is a scratch folder that is removed."""
    scratch = output_dir / _UNBANDED_DIR
    try:
        unbanded = build_sgb_geometry(
            build_input, representation, route=route, output_dir=scratch
        )
        return json.loads(unbanded.sidecar.read_text(encoding="utf-8"))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
