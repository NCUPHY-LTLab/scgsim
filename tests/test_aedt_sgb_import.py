"""The AEDT SGB import's geometry rules that need no AEDT session.

* the free-face loop drawn as one closed AEDT polyline, and its segment lengths;
* the junction current line read from a port face's own edges, and a generic RLC's native branches;
* the refusal of a port sheet with no treatment, and of lumped ports, before AEDT is touched;
* a port treatment over every face of a port sheet SGB partitions, checked before AEDT is touched;
* boundary and Net names unique across Net IDs that sanitise alike, assigned and read back under one name;
* which imported copy of a face each Surface-EPR row integrates, and the ``surface``/``adjacent`` choice by
  sidecar group, on the guard and sidecar of real SGB builds (``occt``).

The live import (STEP, materials, Region, boundaries) runs in AEDT and is checked there.  The module is skipped
where OCP (the ``occt`` group) is missing, since the AEDT SGB modules import the SGB exporter.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import pytest

pytest.importorskip(
    "OCP"
)  # the AEDT SGB modules import the SGB exporter, which needs OCP

from sgb_extreme_cases import FIXTURES, build_model  # noqa: E402
from sgb_inputs import rectangular_junction_source  # noqa: E402
from test_sgb_step_guard import export  # noqa: E402

import scgsim.aedt.runtime.families.sgb_import as sgb_import  # noqa: E402
from scgsim.aedt.epr.sgb_surfaces import (  # noqa: E402
    SurfaceBindingError,
    surface_face_bindings,
    surface_field_selections,
)
from scgsim.aedt.preparation.sgb_model import prepare_sgb_model  # noqa: E402
from scgsim.aedt.runtime.families.sgb_boundaries import (  # noqa: E402
    BoundaryPlanError,
    plan_hfss_boundaries,
    q3d_net_names,
    rlc_branches,
)
from scgsim.aedt.runtime.families.sgb_faces import import_tolerance_um  # noqa: E402
from scgsim.aedt.runtime.families.sgb_import import (  # noqa: E402
    ImportedSgbModel,
    SgbImportError,
    assign_port_rlc,
    assign_q3d_nets,
    assign_sheet_perfect_e,
    current_line_um,
    import_sgb_model,
    polyline_of_loop,
    q3d_net_roles,
    segment_length_um,
)
from scgsim.aedt.specs.common import LumpedRlc, LumpedTerminalPort  # noqa: E402
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
    """The Route C and Route A models of one junction pair with MA, MS and SA bands, and Route A with a bump."""
    film = {"film_thickness_m": 3e-9, "permittivity": 10.0, "loss_tangent": 1e-3}
    specs = normalize_surface_epr_specs(
        {kind: {**film, "inset_margins_um": [0.5]} for kind in ("MA", "MS", "SA")},
        materials={},
    )
    out = {}
    for key, modeling, bump in (
        ("solid", "solid", False),
        ("thin_film", "thin_film", False),
        ("bump", "thin_film", True),
    ):
        source, stack = rectangular_junction_source("VACUUM_REGION", bump=bump)
        out[key] = prepare_sgb_model(
            source,
            prepared_stack=stack,
            modeling=modeling,
            output_dir=tmp_path_factory.mktemp(key),
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


def test_a_generic_rlc_enables_exactly_its_authored_branches():
    """``None`` disables a branch; an authored zero stays an enabled branch of value zero."""
    branches = rlc_branches(
        LumpedRlc("sheet", "series", resistance_ohm=50.0, capacitance_f=0.0)
    )
    assert branches == {
        "RLC Type": "Serial",
        "UseResist": True,
        "Resistance": "50.0ohm",
        "UseInduct": False,
        "UseCap": True,
        "Capacitance": "0.0F",
    }


def _bare_model(tmp_path, **treatments):
    source, stack = rectangular_junction_source("VACUUM_REGION")
    return prepare_sgb_model(
        source,
        prepared_stack=stack,
        modeling="solid",
        output_dir=tmp_path / "model",
        **treatments,
    )


def test_a_port_sheet_without_a_junction_or_rlc_is_refused_before_aedt(tmp_path):
    """A port sheet with no treatment has no AEDT meaning; the import refuses it before touching the design."""
    model = _bare_model(tmp_path)
    with pytest.raises(
        ValueError, match=r"PORT::sheet.*carry no junction or LumpedRlc"
    ):
        import_sgb_model(None, model, family="hfss")


def test_lumped_ports_are_refused_until_the_driven_terminal_design_exists(tmp_path):
    """The import sets up no Driven Terminal design, so a model with lumped ports is refused."""
    port = LumpedTerminalPort(1, "p1", "sheet", ("chip_left",), ("chip_right",), 50.0)
    model = _bare_model(tmp_path, ports=(port,))
    with pytest.raises(NotImplementedError, match="Driven Terminal design"):
        import_sgb_model(None, model, family="hfss")


def test_q3d_refuses_a_lumped_rlc(tmp_path):
    """A lumped RLC belongs to HFSS Eigenmode, not Q3D."""
    model = _bare_model(
        tmp_path, lumped_rlcs=(LumpedRlc("sheet", "parallel", inductance_h=1e-8),)
    )
    with pytest.raises(ValueError, match="Q3D has no junctions or lumped RLCs"):
        import_sgb_model(None, model, family="q3d")


def test_route_a_bump_rows_read_the_body_as_route_c_reads_a_conductor(models):
    """Decision 8: a retained Route A bump is read like a Route C conductor; the sheets stay on the substrate."""
    sidecar, imported = _as_imported(models["bump"])
    bindings = surface_face_bindings(sidecar, imported)
    assert set(bindings) == set(sidecar["surface_epr_rows"])
    bodies = {
        imported.objects[solid]
        for solid, volume in sidecar["volumes"].items()
        if volume["kind"] == "conductor"
    }
    assert len(bodies) == 1
    owners = {
        name
        for binding in bindings.values()
        if binding.interface_type == "MA"
        for name, _ in binding.faces
    }
    assert owners == {
        "substrate",
        *bodies,
    }  # sheet rows on the substrate, bump rows on the bump
    for binding in bindings.values():
        planned = sidecar["physical_groups"][binding.group]["ids"]
        assert len(binding.faces) == len(planned) == len(set(binding.faces))


# ---------------------------------------------------------------- boundary assignments on a fake design


class _Native:
    """The fake design's ``oboundary``: its boundary list and Net assignments, as AEDT returns them."""

    def __init__(self, design):
        self._design = design

    def GetBoundaries(self):  # noqa: N802 -- the AEDT method name
        """Name, type, name, type ... of every boundary."""
        return [
            v
            for name, (kind, _) in self._design.boundaries.items()
            for v in (name, kind)
        ]

    def GetExcitationAssignment(self, name):  # noqa: N802 -- the AEDT method name
        """The object ids the Net ``name`` holds."""
        return [self._design.modeler[obj].id for obj in self._design.nets[name]]


class _FakeDesign:
    """The AEDT calls the boundary assignments make, on named objects.

    A name already taken gets a new one, so a second claim never replaces the first one silently; what
    each name holds at the end is what the design holds.
    """

    def __init__(self, objects):
        self.modeler = objects
        self.boundaries = {}
        self.nets = {}
        self.oboundary = _Native(self)

    @staticmethod
    def _free(name, taken):
        while name in taken:
            name += "_1"
        return name

    def _create_boundary(self, name, props, kind):
        self.boundaries[self._free(name, self.boundaries)] = (kind, props)
        return True

    def assign_perfect_e(self, faces, name):
        """Perfect E over ``faces``."""
        self.boundaries[self._free(name, self.boundaries)] = (
            "Perfect E",
            {"Faces": list(faces)},
        )
        return True

    def assign_net(self, names, net_name, role):
        """A Q3D Net over the objects ``names``."""
        self.nets[self._free(net_name, self.nets)] = list(names)
        return True


def _vertex(point):
    return SimpleNamespace(position=list(point))


def _rebuilt_sheets(guard):
    """Each guarded free face as the sheet AEDT rebuilds from its loops, and the face copies that names."""
    objects, copies = {}, {}
    for k, free in enumerate(guard["free_faces"]):
        edges = [
            SimpleNamespace(
                vertices=[_vertex(s["points_um"][0]), _vertex(s["points_um"][-1])]
            )
            for loop in free["loops"]
            for s in loop
        ]
        face = SimpleNamespace(id=1000 + k, edges=edges)
        objects[f"sheet{k}"] = SimpleNamespace(faces=[face])
        copies[free["face"]] = {f"sheet{k}": 1000 + k}
    return objects, copies


def test_a_port_sheet_split_into_several_faces_takes_one_rlc_over_all_of_them(tmp_path):
    """The round pad's wall cuts port p1 into two free faces; the RLC holds both, its line crosses their union."""
    sidecar, exported = export(
        build_model(FIXTURES["port_crossing_round_pad"], "C"), tmp_path
    )
    rlc = LumpedRlc("p1", "parallel", inductance_h=1e-8)
    plan = plan_hfss_boundaries(sidecar, exported.guard, (), (rlc,))
    (planned,) = plan.lumped_rlcs
    assert len(planned.port["faces"]) == 2
    objects, copies = _rebuilt_sheets(exported.guard)
    design = _FakeDesign(objects)
    tolerance_um = import_tolerance_um(float(sidecar["grid_um"]))
    record = assign_port_rlc(design, planned, copies, tolerance_um)
    kind, props = design.boundaries["rlc_p1"]
    assert kind == "Lumped RLC" and list(design.boundaries) == ["rlc_p1"]
    port_faces = [
        face_id for face in planned.port["faces"] for face_id in copies[face].values()
    ]
    assert sorted(props["Faces"]) == sorted(port_faces) == [1000, 1001]
    assert [f["face_id"] for f in record["faces"]] == props["Faces"]
    # x = 4 is split between the two faces at (4, 3); x = 9 lies on the larger face only.
    assert record["current_line_um"] == [[4.0, 3.0, 0.1], [9.0, 3.0, 0.1]]
    assert props["CurrentLine"] == {
        "Start": ["4.0um", "3.0um", "0.1um"],
        "End": ["9.0um", "3.0um", "0.1um"],
    }


def test_a_port_face_that_does_not_import_once_is_refused_before_aedt_is_touched(
    tmp_path, monkeypatch
):
    """A port face two solids share has two copies; the import refuses it before it touches the design."""
    model = _bare_model(
        tmp_path, lumped_rlcs=(LumpedRlc("sheet", "parallel", inductance_h=1e-8),)
    )
    sidecar = json.loads(model.path("sidecar").read_text(encoding="utf-8"))
    guard = json.loads(model.path("guard").read_text(encoding="utf-8"))
    owners = Counter(f["face"] for solid in guard["solids"] for f in solid["faces"])
    shared = min(face for face, count in owners.items() if count == 2)
    (port,) = sidecar["ports"]
    port["faces"] = [*port["faces"], shared]
    monkeypatch.setattr(sgb_import, "read_sidecar", lambda _: sidecar)
    # ``app`` is None: any use of the design raises AttributeError instead of the refusal.
    with pytest.raises(
        BoundaryPlanError, match=f"port face {shared} .* 2 copies, not one"
    ):
        import_sgb_model(None, model, family="hfss")


def _renamed_nets(sidecar, renames):
    """``sidecar`` with its Nets renamed, in its Net union and on its volumes."""
    out = json.loads(json.dumps(sidecar))
    nets = out["unions"]["nets"]
    out["unions"]["nets"] = {
        renames.get(net, net): groups for net, groups in nets.items()
    }
    for volume in out["volumes"].values():
        if volume["net"] in renames:
            volume["net"] = renames[volume["net"]]
    return out


def test_nets_whose_ids_sanitise_alike_get_a_perfect_e_each(models):
    """``chip/A`` and ``chip_A`` both sanitise to ``chip_A``; each Net gets its own Perfect E, on its own faces."""
    sidecar, imported = _as_imported(models["thin_film"])
    renamed = _renamed_nets(sidecar, {"A": "chip/A", "B": "chip_A"})
    guard = json.loads(models["thin_film"].path("guard").read_text(encoding="utf-8"))
    plan = plan_hfss_boundaries(renamed, guard, (), ())
    assert {net: name for net, (name, _) in plan.perfect_e.items()} == {
        "chip/A": "perfect_e_chip_A",
        "chip_A": "perfect_e_chip_A__2",
    }
    design = _FakeDesign({})
    records = assign_sheet_perfect_e(design, plan, imported.face_copies)
    assert len(records) == 2
    for record in records:
        kind, props = design.boundaries[record["boundary"]]
        assert kind == "Perfect E" and props["Faces"] == record["faces"]
    assert set(design.boundaries) == {r["boundary"] for r in records}


def test_q3d_nets_whose_ids_sanitise_alike_are_assigned_and_read_back_apart(models):
    """Each Net is assigned and read back under its own unique name, never one Net under the other's."""
    sidecar, imported = _as_imported(models["solid"])
    renamed = _renamed_nets(sidecar, {"A": "chip/A", "B": "chip_A"})
    roles = {"chip/A": "Signal", "chip_A": "Ground"}
    names = q3d_net_names(roles)
    assert names == {"chip/A": "chip_A", "chip_A": "chip_A__2"}
    design = _FakeDesign(
        {
            name: SimpleNamespace(id=k)
            for k, name in enumerate(imported.objects.values())
        }
    )
    records = assign_q3d_nets(design, renamed, imported.objects, roles, names)
    assert design.nets == {r["name"]: r["objects"] for r in records}
    assert {r["net"]: r["name"] for r in records} == names
