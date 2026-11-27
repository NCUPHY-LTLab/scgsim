"""Build the SGB geometry of one route with the Surface-EPR bands its specs ask for.

Normally SGB finds the faces to band from the type-level requests of the specs, in one build.  Only when a
spec customises conductors (``inset_margins_by_entity_um``) are their faces needed first, so the same input
is built once without bands, in a scratch folder that is removed, to read them from its sidecar (see
:mod:`.insets`).  Palace and AEDT build their geometry through this one step, into files; a 2D model's cut
reads the same banded model in memory (:func:`build_banded_model`).
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scgsim.contracts.sgb_sidecar import Sidecar, read_sidecar
from scgsim.geometry.models.input import GeometryBuildInput
from scgsim.geometry.models.representation import RepresentationContext
from scgsim.geometry.sgb import SgbArtifacts, build_sgb_geometry
from scgsim.geometry.sgb.bands.requests import InsetRequest
from scgsim.geometry.sgb.model import ConformalModel, build_conformal_model
from scgsim.geometry.sgb.topology.routes.policy import RoutePolicy

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
    insets = _banded_inset_requests(
        build_input,
        representation,
        route,
        surface_epr_specs or {},
        output_dir / _UNBANDED_DIR,
    )
    return build_sgb_geometry(
        build_input,
        representation,
        route=route,
        insets=insets,
        output_dir=output_dir,
    )


def build_banded_model(
    build_input: GeometryBuildInput,
    representation: RepresentationContext,
    *,
    route: str | RoutePolicy,
    surface_epr_specs: Mapping[str, Mapping[str, Any]] | None,
) -> ConformalModel:
    """The validated model :func:`build_banded_geometry` would write, held in memory instead.

    Only the unbanded first build of customised conductors writes files, into a temporary folder that is
    removed: their faces are read from its sidecar.
    """
    specs = surface_epr_specs or {}
    if needs_unbanded_build(specs):
        with tempfile.TemporaryDirectory(prefix="scgsim-unbanded-") as scratch:
            insets = _banded_inset_requests(
                build_input, representation, route, specs, Path(scratch) / _UNBANDED_DIR
            )
    else:
        insets = _banded_inset_requests(build_input, representation, route, specs, None)
    return build_conformal_model(build_input, representation, route, insets=insets)


def _banded_inset_requests(
    build_input: GeometryBuildInput,
    representation: RepresentationContext,
    route: str | RoutePolicy,
    specs: Mapping[str, Mapping[str, Any]],
    scratch: Path | None,
) -> tuple[InsetRequest, ...]:
    """The inset requests of ``specs``; customised conductors first need an unbanded build in ``scratch``."""
    if not requests_wanted(specs):
        return ()
    unbanded = None
    if needs_unbanded_build(specs):
        assert scratch is not None, "customised conductors need a scratch folder"
        unbanded = _unbanded_sidecar(build_input, representation, route, scratch)
    return inset_requests(specs, unbanded)


def _unbanded_sidecar(
    build_input: GeometryBuildInput,
    representation: RepresentationContext,
    route: str | RoutePolicy,
    scratch: Path,
) -> Sidecar:
    """The sidecar of a build without bands; the build itself is the scratch folder, which is removed."""
    try:
        unbanded = build_sgb_geometry(
            build_input, representation, route=route, output_dir=scratch
        )
        return read_sidecar(unbanded.sidecar)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
