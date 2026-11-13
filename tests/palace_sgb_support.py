"""Builders for the Palace-on-SGB tests: real sidecars, and meshes made by the two stages of ``mesh()``.

A sidecar is built by SGB itself (it needs neither Gmsh nor a solver), so the file-only tests of groups,
insets and configs run on real v2 sidecars of Routes A and B.  ``mesh_run`` goes the whole way to
``palace.msh`` and needs Gmsh.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sgb_inputs import EffectiveInput, gds_pad_in_ground, thin_film_pair

from scgsim.contracts.sgb_sidecar import read_sidecar
from scgsim.geometry.sgb import build_sgb_geometry
from scgsim.geometry.sgb.bands.requests import InsetRequest
from scgsim.geometry.surface_epr.specs import normalize_surface_epr_specs
from scgsim.palace.mesh.build import (
    MeshBuildResult,
    build_route_geometry,
    mesh_route_geometry,
)
from scgsim.palace.mesh.controls import MeshControls

MATERIALS = {
    "Si": {"kind": "dielectric", "permittivity": 11.45, "loss_tangent": 0.0},
    "vacuum": {"kind": "vacuum", "permittivity": 1.0, "loss_tangent": 0.0},
    "air": {"kind": "vacuum", "permittivity": 1.0, "loss_tangent": 0.0},
}
FILM = {"film_thickness_m": 2e-9, "permittivity": 4, "loss_tangent": 0.0}


def effective_input(route: str, directory: Path) -> EffectiveInput:
    """The pad-in-ground input for Route B, the thin-film pair (two sheets and a port) for Route A."""
    return gds_pad_in_ground(directory) if route == "B" else thin_film_pair()


def sidecar_of(
    effective: EffectiveInput,
    route: str,
    directory: Path,
    insets: tuple[InsetRequest, ...] = (),
) -> dict[str, Any]:
    """The decoded sidecar of one SGB build of ``effective``."""
    artifacts = build_sgb_geometry(
        effective.build_input,
        effective.representation,
        route=route,
        insets=insets,
        output_dir=directory / f"sgb_{route}",
    )
    return read_sidecar(artifacts.sidecar)


def surface_epr_specs(**margins: list[float] | None) -> dict[str, dict[str, Any]]:
    """Normalized Surface-EPR specs for the interface types named, e.g. ``MA=[0.5, 1.0], SA=None``."""
    return normalize_surface_epr_specs(
        {
            kind: {**FILM, **({"inset_margins_um": m} if m else {})}
            for kind, m in margins.items()
        },
        materials={},
    )


def mesh_run(  # noqa: PLR0913 -- a test builder with independent switches
    effective: EffectiveInput,
    route: str,
    directory: Path,
    *,
    specs: Mapping[str, Mapping[str, Any]] | None = None,
    gds: Path | None = None,
    **controls: Any,
) -> MeshBuildResult:
    """Build the geometry of ``effective`` and mesh it into ``directory / "run"``, as ``mesh()`` does.

    The sources are written as a plan's are, from GDS bytes paired with the stack: Route B the GDS
    ``gds_pad_in_ground`` wrote (or ``gds``), Route A stand-in bytes with the thin-film provenance of a stack.
    ``controls`` are the keywords of ``set_mesh``.
    """
    if route == "B":
        source = (gds or directory / "pad_in_ground.gds").read_bytes()
        metadata: dict[str, Any] = {}
    else:
        source = b"x"
        digest = hashlib.sha256(source).hexdigest()
        metadata = {
            "route_a_thin_film": {
                "schema_version": 1,
                "variant": "substrate_face",
                "display_label": "A_PRIME",
                "source_stack": {"sha256": digest, "revision": f"sha256:{digest}"},
            },
        }
    stack = {
        "materials": {},
        "metadata": {
            "gds_sha256": hashlib.sha256(source).hexdigest(),
            "top_cell_name": "TOP",
            **metadata,
        },
    }
    geometry = build_route_geometry(
        stack=stack,
        route=route,
        output_dir=directory / "run",
        source_gds_bytes=source,
        surface_epr_specs=specs,
        source_geometry_input=effective.build_input,
        representation_context=effective.representation,
    )
    return mesh_route_geometry(geometry, MeshControls().updated(**controls))
