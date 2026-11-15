"""HFSS Driven Terminal on an SGB model, without AEDT: the request, its handoff and the lumped-port assignment.

The model is the two-port source of ``sgb_driven_inputs`` built by ``prepare_sgb_model(ports=...)``.  The
assignment runs on a stand-in for the AEDT design that records the boundaries it is given; the live import,
port, setup and solve are checked in AEDT.  Skipped where OCP (the ``occt`` group) is missing.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from types import SimpleNamespace

import pytest

pytest.importorskip("OCP")

from sgb_driven_inputs import DRIVEN_PORTS, driven_model  # noqa: E402
from sgb_extreme_cases import FIXTURES, build_model  # noqa: E402
from test_aedt_public_preparation import junction_model  # noqa: E402
from test_sgb_step_guard import export  # noqa: E402

from scgsim.aedt import (  # noqa: E402
    FrequencySweepSpec,
    HfssDrivenGeometrySpec,
    HfssRunControl,
    LumpedTerminalPort,
    prepare_handoff,
    prepare_hfss_driven_from_geometry,
)
from scgsim.aedt.runtime.families.sgb_boundaries import (  # noqa: E402
    plan_hfss_boundaries,
)
from scgsim.aedt.runtime.families.sgb_faces import (  # noqa: E402
    SgbImportError,
    import_tolerance_um,
)
from scgsim.aedt.runtime.families.sgb_import import lumped_port_faces  # noqa: E402
from scgsim.aedt.runtime.families.sgb_ports import assign_lumped_port  # noqa: E402
from scgsim.aedt.specs.parse import parse_aedt_spec  # noqa: E402
from scgsim.contracts.sgb_sidecar import read_sidecar  # noqa: E402
from scgsim.geometry.sgb.output.exchange.guard import read_step_guard  # noqa: E402

pytestmark = pytest.mark.occt

RUN = HfssRunControl("Setup", "Sweep", FrequencySweepSpec(1, 2, points=5))


def _spec(model=None, run_control=RUN):
    model = driven_model() if model is None else model
    return HfssDrivenGeometrySpec(
        aedt_version="2024.2",
        modeling=model.modeling,
        model=model,
        project_name="Driven",
        design_name="Design",
        run_control=run_control,
    )


def _handoff(root):
    return prepare_hfss_driven_from_geometry(
        aedt_version="2024.2",
        model=driven_model(),
        project_name="Driven",
        design_name="Design",
        run_control=RUN,
        output_dir=root,
    )


def test_the_handoff_carries_the_model_and_its_two_lumped_ports(tmp_path):
    """The handoff holds the model's four files; its request reads back with the model's ports, moved too."""
    plan = _handoff(tmp_path / "handoff")
    payload = json.loads(plan.spec_path.read_text())
    assert payload["model"]["model_dir"] == "model"
    assert "geometry" not in payload and "ports" not in payload
    moved = tmp_path / "moved"
    shutil.copytree(plan.run_dir, moved)
    parsed = parse_aedt_spec(payload, base_dir=moved)
    assert parsed.ports == DRIVEN_PORTS
    assert parsed.to_payload() == payload
    receipt = json.loads(plan.receipt_path.read_text())
    assert receipt["source"]["sgb_model_sha256"] == driven_model().identity_sha256
    metadata = json.loads(plan.metadata_path.read_text())
    assert metadata["workflow"] == "body_first_driven_terminal"
    assert metadata["execution"] == {"resources": None}
    manifest = json.loads(plan.manifest_path.read_text())
    model_files = [p for p in manifest["allowed_paths"] if p.startswith("model/")]
    assert len(model_files) == 4


def test_a_driven_request_of_the_native_builder_is_refused(tmp_path):
    """A Driven request that carries a native planar geometry is refused with the reason (decision 6)."""
    plan = _handoff(tmp_path / "handoff")
    payload = json.loads(plan.spec_path.read_text())
    payload["geometry"] = payload.pop("model")
    with pytest.raises(ValueError, match="natively built geometry"):
        parse_aedt_spec(payload, base_dir=plan.run_dir)
    with pytest.raises(ValueError, match="handoff directory that holds its model"):
        HfssDrivenGeometrySpec.from_payload(_spec().to_payload(), base_dir=None)


def test_a_port_whose_sheet_does_not_join_its_signal_and_reference_is_refused():
    """Each port sheet joins a signal Entity of its port and a reference Entity, one each."""
    wrong = replace(DRIVEN_PORTS[0], signal_entity_ids=("chip/pad_2",))
    model = replace(driven_model(), lumped_ports=(wrong, DRIVEN_PORTS[1]))
    with pytest.raises(ValueError, match="port 'p1': its sheet 'port_1' joins"):
        _spec(model)


@pytest.mark.parametrize(
    ("ports", "message"),
    [
        ((DRIVEN_PORTS[0],), "two lumped ports, 1 and 2"),
        (
            (
                DRIVEN_PORTS[0],
                replace(DRIVEN_PORTS[1], reference_entity_ids=("chip/pad_1",)),
            ),
            "must share one ordered reference",
        ),
    ],
)
def test_a_request_that_is_not_two_ports_on_one_reference_is_refused(ports, message):
    """V1 Driven Terminal is two lumped ports, 1 and 2, with one shared reference."""
    with pytest.raises(ValueError, match=message):
        _spec(replace(driven_model(), lumped_ports=ports))


def test_a_model_with_eigenmode_treatments_is_not_a_driven_model():
    """Junctions and lumped RLCs belong to HFSS Eigenmode."""
    model = replace(junction_model(), lumped_ports=DRIVEN_PORTS)
    with pytest.raises(ValueError, match="carry more than one treatment|they belong"):
        _spec(model)


def test_prepare_handoff_points_an_hfss_request_to_its_preparation(tmp_path):
    """An HFSS request is prepared by its own function, which attaches the model."""
    with pytest.raises(TypeError, match="prepare_hfss_driven_from_geometry"):
        prepare_handoff(spec=_spec(), output_dir=tmp_path / "handoff")


@pytest.mark.parametrize("modeling", ["solid", "thin_film"])
def test_each_lumped_port_takes_its_port_sheet_faces_and_direction(modeling):
    """On Route C and Route A each port takes the faces of its sheet, along the sheet's ground-to-pad direction."""
    model = driven_model(modeling)
    sidecar = read_sidecar(model.path("sidecar"))
    guard = read_step_guard(model.path("guard"))
    plan = plan_hfss_boundaries(sidecar, guard, (), (), model.lumped_ports)
    assert [item.port for item in plan.lumped_ports] == list(DRIVEN_PORTS)
    for item in plan.lumped_ports:
        assert item.sheet["group"] == f"PORT::{item.port.port_sheet_id}"
        assert item.direction == (0.0, 1.0, 0.0)
        assert item.sheet["host_entity_ids"][0] == "chip/ground"
        assert item.sheet["faces"]


class _Design:
    """The AEDT calls a lumped-port assignment makes: faces with edges on named objects, ports and Terminals."""

    def __init__(self, objects, *, terminals_per_port=1):
        self.modeler = objects
        self.ports = {}
        self.terminals = {}
        self.terminals_per_port = terminals_per_port
        self.oboundary = SimpleNamespace(
            GetExcitationsOfType=self._excitations,
            AssignLumpedPort=self._port,
            AssignTerminal=self._terminal,
        )

    def _excitations(self, kind):
        return list(self.terminals if kind == "Terminal" else self.ports)

    @staticmethod
    def _props(args):
        keys = (key.removesuffix(":=") for key in args[1::2])
        return dict(zip(keys, args[2::2], strict=True))

    def _port(self, args):
        self.ports[args[0].removeprefix("NAME:")] = self._props(args)

    def _terminal(self, args):
        name = args[0].removeprefix("NAME:")
        for k in range(self.terminals_per_port):
            self.terminals[name if k == 0 else f"{name}_{k}"] = self._props(args)


def _sheet_design(guard, **options):
    """Each guarded free face as the sheet AEDT rebuilds from its loops, its edges numbered, and its copies."""
    objects, copies = {}, {}
    edge_ids = iter(range(2000, 3000))
    for k, free in enumerate(guard["free_faces"]):
        edges = [
            SimpleNamespace(
                id=next(edge_ids),
                vertices=[
                    SimpleNamespace(position=list(s["points_um"][0])),
                    SimpleNamespace(position=list(s["points_um"][-1])),
                ],
            )
            for loop in free["loops"]
            for s in loop
        ]
        objects[f"sheet{k}"] = SimpleNamespace(
            faces=[SimpleNamespace(id=1000 + k, edges=edges)]
        )
        copies[free["face"]] = {f"sheet{k}": 1000 + k}
    return _Design(objects, **options), copies


def _sheet_names(copies):
    """Planned face -> the sheet object a free face was rebuilt as, as the import hands them to the port."""
    return {face: next(iter(names)) for face, names in copies.items()}


def _round_pad_port(tmp_path, signal_host=1):
    sidecar, exported = export(
        build_model(FIXTURES["port_crossing_round_pad"], "C"), tmp_path
    )
    # The fixture's sheet lands on one pad; the two hosts a Driven port joins stand at its ends along +x.
    hosts = ("low", "high")
    port = LumpedTerminalPort(
        1, "P1", "p1", (hosts[signal_host],), (hosts[1 - signal_host],), 50.0
    )
    (planned,) = plan_hfss_boundaries(
        sidecar, exported.guard, (), (), (port,)
    ).lumped_ports
    planned = replace(planned, sheet={**planned.sheet, "host_entity_ids": list(hosts)})
    tolerance = import_tolerance_um(float(sidecar["model"]["grid_um"]))
    return planned, exported.guard, tolerance


def _edge_ends(design, edge_ids):
    return sorted(
        tuple(v.position for v in edge.vertices)
        for obj in design.modeler.values()
        for face in obj.faces
        for edge in face.edges
        if edge.id in edge_ids
    )


@pytest.mark.parametrize(("signal_host", "x_um"), [(1, 9.0), (0, 4.0)])
def test_a_lumped_port_over_a_split_sheet_takes_all_its_faces_and_its_signal_end(
    tmp_path, signal_host, x_um
):
    """The round pad's wall cuts port p1 into two faces; the port holds both, its Terminal the signal end.

    The sheet runs along +x from its first host (x = 4) to its second (x = 9); the Terminal is on the edges
    at the signal host's end, split there into two collinear edges when the wall meets it.
    """
    planned, guard, tolerance = _round_pad_port(tmp_path, signal_host)
    design, copies = _sheet_design(guard)
    record = assign_lumped_port(design, planned, _sheet_names(copies), tolerance)
    port = design.ports["P1"]
    assert sorted(port["Faces"]) == [1000, 1001]
    assert port["LumpedPortType"] == "Terminal" and port["Impedance"] == "50.0ohm"
    assert port["RenormalizeAllTerminals"] is False
    terminal = design.terminals["P1_T1"]
    assert terminal["ParentBndID"] == "P1"
    assert terminal["Edges"] == record["terminal_edges"]
    assert {p[0] for edge in _edge_ends(design, terminal["Edges"]) for p in edge} == {
        x_um
    }
    assert record["terminal_excitation"] == "P1_T1"
    assert record["signal_host"] == planned.sheet["host_entity_ids"][signal_host]
    assert record["requested"] == planned.port.to_payload()


@pytest.mark.parametrize("terminals", [0, 2])
def test_a_lumped_port_that_does_not_create_one_terminal_is_refused(
    tmp_path, terminals
):
    """A port that AEDT gives no Terminal, or two, is refused rather than read as some port."""
    planned, guard, tolerance = _round_pad_port(tmp_path)
    design, copies = _sheet_design(guard, terminals_per_port=terminals)
    with pytest.raises(SgbImportError, match="created the Terminals .*not one"):
        assign_lumped_port(design, planned, _sheet_names(copies), tolerance)


def test_a_port_with_no_signal_host_is_refused_before_aedt(tmp_path):
    """A port whose sheet touches none of its signal Entities has no Terminal end; AEDT is not touched."""
    planned, guard, tolerance = _round_pad_port(tmp_path)
    stray = replace(planned, port=replace(planned.port, signal_entity_ids=("x",)))
    design, copies = _sheet_design(guard)
    with pytest.raises(SgbImportError, match="not one signal Entity"):
        assign_lumped_port(design, stray, _sheet_names(copies), tolerance)
    assert design.ports == {}


class _DrawingModeler(dict):
    """A modeler that draws closed flat polylines as one-face sheets with numbered edges."""

    def __init__(self):
        super().__init__()
        self._ids = iter(range(5000, 6000))

    def __getitem__(self, name):
        """An object by its current name: renaming a sheet renames it in the design."""
        return next(sheet for sheet in self.values() if sheet.name == name)

    def create_polyline(self, points, cover_surface, close_surface, name):
        """One sheet of the closed loop ``points``, its face's area the polygon's."""
        assert cover_surface and close_surface and points[0] == points[-1]
        loop = [tuple(p) for p in points[:-1]]
        pairs = list(zip(loop, [*loop[1:], loop[0]], strict=True))
        edges = [
            SimpleNamespace(
                id=next(self._ids),
                vertices=[
                    SimpleNamespace(position=list(a)),
                    SimpleNamespace(position=list(b)),
                ],
            )
            for a, b in pairs
        ]
        twice = sum(a[0] * b[1] - b[0] * a[1] for a, b in pairs)
        face = SimpleNamespace(id=next(self._ids), area=abs(twice) / 2, edges=edges)
        sheet = SimpleNamespace(name=name, faces=[face])
        self[name] = sheet
        return sheet


def _driven_plan(modeling="solid"):
    model = driven_model(modeling)
    sidecar = read_sidecar(model.path("sidecar"))
    guard = read_step_guard(model.path("guard"))
    plan = plan_hfss_boundaries(sidecar, guard, (), (), model.lumped_ports)
    return plan, guard, import_tolerance_um(float(sidecar["model"]["grid_um"]))


@pytest.mark.parametrize("modeling", ["solid", "thin_film"])
def test_a_port_face_on_the_substrate_is_drawn_again_from_its_guarded_vertices(
    modeling,
):
    """A port face on an imported solid is drawn as a sheet at the guard's vertices.

    Its Terminal is on the pad edge (y = 0), the end the ground-to-pad direction points to.
    """
    plan, guard, tolerance = _driven_plan(modeling)
    design = _Design(_DrawingModeler())
    record = assign_lumped_port(
        design, plan.lumped_ports[0], lumped_port_faces(guard, {}), tolerance
    )
    sheet = design.modeler["port_sheet_port_1"]
    assert record["faces"] == [
        {"object_name": "port_sheet_port_1", "face_id": sheet.faces[0].id}
    ]
    assert sheet.faces[0].area == pytest.approx(9.0)
    (edge,) = (e for e in sheet.faces[0].edges if e.id in record["terminal_edges"])
    assert sorted(v.position[0] for v in edge.vertices) == [1.0, 4.0]
    assert {v.position[1] for v in edge.vertices} == {0.0}


def test_a_port_face_that_is_not_its_guarded_polygon_is_refused():
    """A guard record whose vertices do not span its area (an arc, a hole) is not drawn as a polygon."""
    plan, guard, tolerance = _driven_plan()
    first = plan.lumped_ports[0]
    faces = lumped_port_faces(guard, {})
    (planned,) = first.sheet["faces"]
    faces[planned] = {**faces[planned], "area_um2": 8.0}
    with pytest.raises(SgbImportError, match="not a flat polygon of its vertices"):
        assign_lumped_port(_Design(_DrawingModeler()), first, faces, tolerance)
