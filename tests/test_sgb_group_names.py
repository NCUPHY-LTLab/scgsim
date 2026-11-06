"""The format of the primary surface-group names on every route: ``<prefix>::<entity>__<face kind>[::band=k]``.

The face kind attaches to the entity part with a double underscore and every other separator is ``::``.  A name
carries no height.  Entity ids may not contain ``__``, so a name is never ambiguous.
"""

from __future__ import annotations

import pytest
from sgb_scenes import airbridge, pad_in_ground

from scgsim.geometry.sgb.cutfamilies.requests import InsetRequest, InterfaceSurfaces
from scgsim.geometry.sgb.errors import InputError
from scgsim.geometry.sgb.model import build_model_from_catalogue
from scgsim.geometry.sgb.records.prisms import PrismCatalogue

BANDS = (1.0,)


def banded(catalogue: PrismCatalogue, route: str, *types: str):
    """The model with a type-level request per Surface-EPR type, margins of 1 um."""
    return build_model_from_catalogue(
        catalogue,
        route,
        insets=tuple(
            InsetRequest(t.lower(), InterfaceSurfaces(t), BANDS) for t in types
        ),
    )


def surface_groups(model) -> set[str]:
    return {n for n, g in model.routed.groups.items() if g.dim == 2}


def volume_groups(model) -> set[str]:
    return {n for n, g in model.routed.groups.items() if g.dim == 3}


# --------------------------------------------------------------------------- Route A


def test_route_a_names_without_bands():
    """A sheet is its bare entity id; the interface, the exterior and the port follow the table."""
    model = build_model_from_catalogue(pad_in_ground("A"), "A")
    assert surface_groups(model) == {
        "EXT::substrate__bottom",
        "EXT::substrate__sidewall",
        "EXT::vacuum__sidewall",
        "EXT::vacuum__top",
        "PORT::p1",
        "SA::substrate|vacuum__interface",
        "ground",
        "pad",
    }
    assert volume_groups(model) == {"substrate", "vacuum"}
    assert model.routed.surface_epr_rows == {}


def test_route_a_names_with_bands():
    """A banded sheet group keeps ``pad::band=k``; its rows and a banded interface carry ``__``."""
    model = banded(pad_in_ground("A"), "A", "MA", "SA")
    routed = model.routed
    assert {"pad::band=0", "pad::band=1", "ground::band=0"} <= set(routed.groups)
    assert {
        "SA::substrate|vacuum__interface::band=0",
        "SA::substrate|vacuum__interface::band=1",
    } <= set(routed.groups)
    assert "SA::substrate|vacuum__interface" not in routed.groups
    assert set(routed.surface_epr_rows) == {
        f"{row}::{entity}__sheet::band={k}"
        for row in ("MA", "MS")
        for entity in ("pad", "ground")
        for k in (0, 1)
    } | {f"SA::substrate|vacuum__interface::band={k}" for k in (0, 1)}
    assert routed.surface_epr_rows["MA::pad__sheet::band=1"] == "pad::band=1"
    assert routed.unions["SHEET"] == [
        f"{entity}::band={k}" for entity in ("ground", "pad") for k in (0, 1)
    ]
    assert set(routed.unions) >= {"NET::Ground", "NET::pad", "PORT"}


# --------------------------------------------------------------------------- Routes B and C


@pytest.mark.parametrize("route", ["B", "C"])
def test_routes_b_and_c_name_conductor_faces_and_exterior(route):
    """``MA::<entity>__top`` ... ``MS::<entity>__bottom``, ``__sidewall``, the interface and the exterior."""
    model = build_model_from_catalogue(airbridge(), route)
    assert {
        "MA::deck__top",
        "MA::deck__bottom",
        "MA::post_left__sidewall",
        "MA::ground_left__top",
        "MS::ground_left__bottom",
        "SA::substrate|vacuum__interface",
        "EXT::vacuum__top",
        "EXT::substrate__sidewall",
    } <= set(model.routed.groups)
    assert not any(
        name.endswith(("::top", "::bottom", "::sidewall", "::interface"))
        for name in model.routed.groups
    )
    assert set(model.routed.unions["NET::Ground"]) >= {
        "MA::deck__top",
        "MS::ground_left__bottom",
    }


def test_route_c_names_the_conductor_to_conductor_faces():
    """``MM::<a>|<b>__interface``, the two entities sorted; volumes keep their bare entity ids."""
    model = build_model_from_catalogue(airbridge(), "C")
    routed = model.routed
    assert {
        "MM::deck|post_left__interface",
        "MM::ground_left|post_left__interface",
    } <= (set(routed.groups))
    assert routed.unions["MM"] == sorted(
        n for n in routed.groups if n.startswith("MM::")
    )
    assert {
        "deck",
        "ground_left",
        "post_right",
        "substrate",
        "vacuum",
    } <= volume_groups(model)
    assert "EXT::ground_left__sidewall" in routed.groups


@pytest.mark.parametrize("route", ["B", "C"])
def test_routes_b_and_c_name_banded_faces(route):
    """A band is ``::band=k`` after the whole ``<prefix>::<entity>__<face kind>``, rows included."""
    model = banded(airbridge(), route, "MA", "MS", "SA")
    routed = model.routed
    expected_rows = {
        f"{base}::band={k}"
        for base in (
            "MA::ground_left__top",
            "MA::ground_right__top",
            "MS::ground_left__bottom",
            "MS::ground_right__bottom",
            "SA::substrate|vacuum__interface",
        )
        for k in (0, 1)
    }
    assert expected_rows <= set(routed.surface_epr_rows)
    assert expected_rows <= set(routed.groups)
    for row, group in routed.surface_epr_rows.items():
        assert row == group
        assert "z=" not in row
