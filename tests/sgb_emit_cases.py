"""Named models for the emission tests: the running examples under every route, with and without cut families.

Each case is a function returning a validated :class:`ConformalModel`.  ``MESH_SIZE`` gives the coarse mesh
size (um) that suits a case when a test also meshes it.
"""

from __future__ import annotations

from collections.abc import Callable

from sgb_extreme_cases import FIXTURES, build_model
from sgb_scenes import airbridge, pad_in_ground, random_layout

from scgsim.geometry.sgb.bands.requests import (
    ConductorSurface,
    InsetRequest,
    RefinementRequest,
    SolutionInterface,
)
from scgsim.geometry.sgb.model import ConformalModel, build_model_from_catalogue

MARGINS = (0.1, 0.5, 2.0)


def pad_requests(route: str) -> tuple[InsetRequest, ...]:
    """Bands on the pad (its sheet in Route A, top and bottom face otherwise) and on the substrate-vacuum gap."""
    pad = (
        (InsetRequest("band_pad_top", ConductorSurface("pad", "sheet"), MARGINS),)
        if route == "A"
        else (
            InsetRequest("band_pad_top", ConductorSurface("pad", "top"), MARGINS),
            InsetRequest("band_pad_bottom", ConductorSurface("pad", "bottom"), MARGINS),
        )
    )
    return (
        *pad,
        InsetRequest("band_gap", SolutionInterface("substrate", "vacuum"), MARGINS),
    )


def pad_model(
    route: str, *, bands: bool = False, refine: bool = False
) -> ConformalModel:
    """The pad in ground under a route (Route C builds the finite catalogue of Route B)."""
    catalogue = pad_in_ground("A" if route == "A" else "B")
    refinements = (
        (
            RefinementRequest("ref", 0.0, ((-40.0, 55.0), (40.0, 55.0))),
        )  # on the substrate-vacuum interface
        if refine
        else ()
    )
    return build_model_from_catalogue(
        catalogue,
        route,
        insets=pad_requests(route) if bands else (),
        refinements=refinements,
    )


def random_model(route: str) -> ConformalModel:
    """Twenty random features (sheets in Route A, finite metal otherwise)."""
    catalogue = random_layout(20, seed=3, route="A" if route == "A" else "B")
    return build_model_from_catalogue(catalogue, route)


CASES: dict[str, Callable[[], ConformalModel]] = {
    "pad_a": lambda: pad_model("A"),
    "pad_b": lambda: pad_model("B"),
    "pad_c": lambda: pad_model("C"),
    "bridge_b": lambda: build_model_from_catalogue(airbridge(), "B"),
    "bridge_c": lambda: build_model_from_catalogue(airbridge(), "C"),
    "pad_a_bands": lambda: pad_model("A", bands=True),
    "pad_b_bands": lambda: pad_model("B", bands=True),
    "pad_c_bands": lambda: pad_model("C", bands=True),
    "pad_a_refinement": lambda: pad_model("A", refine=True),
    "pad_b_refinement": lambda: pad_model("B", refine=True),
    "void_shell_b": lambda: build_model(FIXTURES["floating_conductor"], "B"),
    "void_shell_c": lambda: build_model(FIXTURES["floating_conductor"], "C"),
    "floating_sheet_a": lambda: build_model(FIXTURES["floating_conductor"], "A"),
    "pad_polygon_a": lambda: build_model_from_catalogue(
        pad_in_ground("A", hole_sides=32, rounded_corner=False), "A"
    ),
    "random20_a": lambda: random_model("A"),
    "random20_b": lambda: random_model("B"),
    "random20_c": lambda: random_model("C"),
}


def support_regions(model: ConformalModel) -> dict:
    """The support region of every inset family of a model, by family id."""
    return {
        f.family_id: f.support for f in model.labelled.catalogue.families if f.support
    }


MESH_SIZE = {
    name: 40.0 if name.endswith("_a") or "_a_" in name else 60.0 for name in CASES
}
