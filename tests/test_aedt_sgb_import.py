"""The AEDT SGB import's geometry rules that need no AEDT session.

* the free-face loop drawn as one closed AEDT polyline, and its segment lengths;
* the junction current line read from a port face's own edges;
* which imported copy of a face each Surface-EPR row integrates, and the ``surface``/``adjacent`` choice by
  sidecar group, on the guard and sidecar of real SGB builds (``occt``).

The live import (STEP, materials, Region, boundaries) runs in AEDT and is checked there.  The module is skipped
where OCP (the ``occt`` group) is missing, since the AEDT SGB modules import the SGB exporter.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace

import pytest

pytest.importorskip(
    "OCP"
)  # the AEDT SGB modules import the SGB exporter, which needs OCP

from sgb_inputs import rectangular_junction_source  # noqa: E402

from scgsim.aedt.epr.sgb_surfaces import (  # noqa: E402
    SurfaceBindingError,
    surface_face_bindings,
    surface_field_selections,
)
from scgsim.aedt.preparation.sgb_model import prepare_sgb_model  # noqa: E402
from scgsim.aedt.runtime.families.sgb_import import (  # noqa: E402
    ImportedSgbModel,
    SgbImportError,
    current_line_um,
    polyline_of_loop,
    q3d_net_roles,
    segment_length_um,
)
from scgsim.geometry.surface_epr import normalize_surface_epr_specs  # noqa: E402

pytestmark = pytest.mark.occt


def _line(a, b):
    return {"kind": "line", "points_um": [list(a), list(b)]}


def test_a_loop_of_lines_and_an_arc_is_one_closed_polyline():
    """Each segment adds its points but the first; the polyline ends where it starts."""
    loop = [
        _line((0, 0, 0), (2, 0, 0)),
        {"kind": "arc", "points_um": [[2, 0, 0], [3, 1, 0], [2, 2, 0]]},
        _line((2, 2, 0), (0, 2, 0)),
        _line((0, 2, 0), (0, 0, 0)),
    ]
    points, kinds = polyline_of_loop(loop)
    assert kinds == ["Line", "Arc", "Line", "Line"]
    assert points[0] == points[-1] == (0.0, 0.0, 0.0)
    assert len(points) == 1 + 1 + 2 + 1 + 1


def test_a_loop_that_does_not_close_is_refused():
    """A gap between two segments is not bridged."""
    with pytest.raises(ValueError, match="not connected"):
        polyline_of_loop([_line((0, 0, 0), (1, 0, 0)), _line((2, 0, 0), (0, 0, 0))])


@pytest.mark.parametrize(
    ("mid", "length"),
    [
        ((1, 1, 0), math.pi),
        ((1, -1, 0), math.pi),
        ((1 + math.sqrt(0.5), math.sqrt(0.5), 0), math.pi / 2),
    ],
)
def test_an_arc_is_as_long_as_the_side_its_mid_point_lies_on(mid, length):
    """A half circle either way, and a quarter circle whose mid point lies on the minor arc."""
    start, end = (0, 0, 0), (2, 0, 0)
    if length == math.pi / 2:
        start, end = (2, 0, 0), (1, 1, 0)
    arc = {"kind": "arc", "points_um": [list(start), list(mid), list(end)]}
    assert segment_length_um(arc) == pytest.approx(length)


def test_the_current_line_joins_the_mid_points_of_the_port_ends():
    """A 2 x 1 um port along +x: the line runs from the left edge's middle to the right edge's."""
    a, b, c, d = (4, 0, 0), (6, 0, 0), (6, 1, 0), (4, 1, 0)
    edges = [(a, b), (b, c), (c, d), (d, a)]
    assert current_line_um(edges, (1.0, 0.0, 0.0), 1e-7) == (
        (4.0, 0.5, 0.0),
        (6.0, 0.5, 0.0),
    )


def test_a_port_end_split_into_collinear_edges_is_one_segment():
    """SGB may split a port end where another face meets it; the line still starts mid-segment."""
    a, b, c, d, m = (4, 0, 0), (6, 0, 0), (6, 1, 0), (4, 1, 0), (4, 0.25, 0)
    edges = [(a, b), (b, c), (c, d), (d, m), (m, a)]
    assert current_line_um(edges, (1.0, 0.0, 0.0), 1e-7)[0] == (4.0, 0.5, 0.0)


def test_a_port_end_with_a_gap_is_refused():
    """Two edges at one end that leave a gap between them are not one segment."""
    a, b, c, d = (4, 0, 0), (6, 0, 0), (6, 1, 0), (4, 1, 0)
    m, n = (4, 0.25, 0), (4, 0.75, 0)
    edges = [(a, b), (b, c), (c, d), (d, n), (m, a)]
    with pytest.raises(SgbImportError, match="do not form one segment"):
        current_line_um(edges, (1.0, 0.0, 0.0), 1e-7)


# ---------------------------------------------------------------- Surface-EPR rows on real SGB builds


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    """The Route C and Route A models of one junction pair with MA, MS and SA bands."""
    film = {"film_thickness_m": 3e-9, "permittivity": 10.0, "loss_tangent": 1e-3}
    specs = normalize_surface_epr_specs(
        {kind: {**film, "inset_margins_um": [0.5]} for kind in ("MA", "MS", "SA")},
        materials={},
    )
    out = {}
    for modeling in ("solid", "thin_film"):
        source, stack = rectangular_junction_source("VACUUM_REGION")
        out[modeling] = prepare_sgb_model(
            source,
            prepared_stack=stack,
            modeling=modeling,
            output_dir=tmp_path_factory.mktemp(modeling),
            surface_epr_specs=specs,
        )
    return out


def _as_imported(model):
    """The import AEDT makes of ``model``: every guarded face once on each solid that bounds it."""
    guard = json.loads(model.path("guard").read_text(encoding="utf-8"))
    sidecar = json.loads(model.path("sidecar").read_text(encoding="utf-8"))
    copies: dict = {}
    for k, solid in enumerate(guard["solids"]):
        for j, face in enumerate(solid["faces"]):
            copies.setdefault(face["face"], {})[solid["aedt_name"]] = 100 * k + j
    objects = {s["solid"]: s["aedt_name"] for s in guard["solids"]}
    return sidecar, ImportedSgbModel(objects, copies, {})


def test_route_c_rows_read_ms_and_sa_on_the_substrate_and_ma_on_the_conductor(models):
    """MS faces import on the conductor and the substrate; the row reads the substrate's copy, once."""
    sidecar, imported = _as_imported(models["solid"])
    bindings = surface_face_bindings(sidecar, imported)
    assert set(bindings) == set(sidecar["surface_epr_rows"])
    owners = {
        binding.interface_type: {name for name, _ in binding.faces}
        for binding in bindings.values()
    }
    assert owners["MS"] == {"substrate"}
    assert owners["SA"] == {"substrate"}
    assert owners["MA"] <= {"chip_left", "chip_right"}
    for binding in bindings.values():
        planned = sidecar["physical_groups"][binding.group]["ids"]
        assert len(binding.faces) == len(planned) == len(set(binding.faces))
    names = [binding.selection_name for binding in bindings.values()]
    assert len(set(names)) == len(names)
    assert all(name.replace("_", "").isalnum() for name in names)


def test_route_a_sheet_rows_read_the_substrate_face_under_the_sheet(models):
    """A Route A sheet lies on the substrate's top face, its one copy, for both its MA and MS rows."""
    sidecar, imported = _as_imported(models["thin_film"])
    bindings = surface_face_bindings(sidecar, imported)
    ma = bindings["MA::chip/left__sheet::band=0"]
    ms = bindings["MS::chip/left__sheet::band=0"]
    assert ma.group == ms.group
    assert ma.faces == ms.faces
    assert {name for name, _ in ma.faces} == {"substrate"}


def test_a_face_with_two_copies_and_no_dielectric_side_is_refused(models):
    """An MA face on two imported solids has no copy its type reads."""
    sidecar, imported = _as_imported(models["solid"])
    face = sidecar["physical_groups"]["MA::chip/left__sidewall"]["ids"][0]
    copies = {**imported.face_copies, face: {"chip_left": 1, "substrate": 2}}
    with pytest.raises(SurfaceBindingError, match="none of them"):
        surface_face_bindings(sidecar, ImportedSgbModel(imported.objects, copies, {}))


def test_selectors_are_keyed_by_group_and_default_to_the_face_itself(models):
    """``adjacent`` on one group switches only that group's rows; an unknown group is refused."""
    sidecar, imported = _as_imported(models["thin_film"])
    bindings = surface_face_bindings(sidecar, imported)
    tokens = surface_field_selections(bindings, {"chip/left::band=0": "adjacent"})
    assert tokens["MA::chip/left__sheet::band=0"] == "EnterAdjacentSurface"
    assert tokens["MS::chip/left__sheet::band=0"] == "EnterAdjacentSurface"
    assert tokens["MA::chip/left__sheet::band=1"] == "EnterSurface"
    with pytest.raises(ValueError, match="no row carries"):
        surface_field_selections(bindings, {"MA::chip/left__sheet::band=0": "surface"})
    with pytest.raises(ValueError, match="'surface' or 'adjacent'"):
        surface_field_selections(bindings, {"chip/left::band=0": "below"})


def test_q3d_net_roles_come_from_the_models_ground_declaration(tmp_path):
    """A declared ground Net is Ground, every other conductor Net Signal; no declaration is refused."""
    source, stack = rectangular_junction_source("VACUUM_REGION")
    model = prepare_sgb_model(
        source,
        prepared_stack=stack,
        modeling="solid",
        output_dir=tmp_path / "grounded",
        ground_net_ids=("B",),
    )
    sidecar = json.loads(model.path("sidecar").read_text(encoding="utf-8"))
    assert q3d_net_roles(model, sidecar) == {"A": "Signal", "B": "Ground"}
    assert type(model).from_payload(model.to_payload()) == model
    undeclared = replace(model, ground_net_ids=None)
    with pytest.raises(ValueError, match="declares no ground Nets"):
        q3d_net_roles(undeclared, sidecar)
    assert q3d_net_roles(replace(model, ground_net_ids=()), sidecar) == {
        "A": "Signal",
        "B": "Signal",
    }


def test_a_ground_net_the_model_does_not_have_is_refused(tmp_path):
    """A ground declaration naming no Net of the model fails at preparation."""
    source, stack = rectangular_junction_source("VACUUM_REGION")
    with pytest.raises(ValueError, match=r"ground Nets \['GND'\] are not Nets"):
        prepare_sgb_model(
            source,
            prepared_stack=stack,
            modeling="solid",
            output_dir=tmp_path / "bad",
            ground_net_ids=("GND",),
        )
