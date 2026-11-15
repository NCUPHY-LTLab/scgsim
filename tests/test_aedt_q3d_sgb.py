"""The Q3D request on an SGB model, its handoff, and the terminal side read from an imported object.

* ``prepare_q3d_from_geometry`` builds the plan's SGB model and writes a handoff that carries it; the request's
  Nets are the model's, each over its conductor objects, and a terminal Entity names one object of its Net;
* a request whose Nets or objects are not the model's, and a payload of the native builder, are refused;
* a terminal side selects the planar faces of a solid, or the straight edges of a sheet, at the object's
  extreme along the side, as one patch or one segment;
* the Nets and terminals are assigned through AEDT's own calls and read back (an AEDT stand-in records them).

The live import, solve and matrix run in AEDT and are checked there.  Skipped where OCP (the ``occt`` group)
is missing, since preparation exports the SGB STEP.
"""

from __future__ import annotations

import atexit
import functools
import json
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("OCP")

from sgb_inputs import electrode_plan, rectangular_junction_source  # noqa: E402

from scgsim.aedt import (  # noqa: E402
    MatrixRunControl,
    Q3dNetSpec,
    Q3dSpec,
    Q3dTerminalPair,
    prepare_q3d_from_geometry,
    prepare_sgb_model,
)
from scgsim.aedt.preparation.sgb_handoff import attach_sgb_model  # noqa: E402
from scgsim.aedt.preparation.sgb_model import PreparedSgbModel  # noqa: E402
from scgsim.aedt.preparation.sgb_q3d import q3d_net_objects  # noqa: E402
from scgsim.aedt.runtime.families.q3d_terminals import (  # noqa: E402
    Q3dTerminalError,
    assign_q3d_nets,
    sheet_side_edges,
    solid_side_faces,
)
from scgsim.aedt.runtime.handoff_validation import verify_prepared_hashes  # noqa: E402
from scgsim.aedt.specs.parse import parse_aedt_spec  # noqa: E402
from scgsim.aedt.specs.q3d import NATIVE_PAYLOAD_REFUSAL  # noqa: E402

pytestmark = pytest.mark.occt

RUN = MatrixRunControl("Setup1", frequency_ghz=6.0, maximum_passes=2)


@functools.cache
def junction_model() -> PreparedSgbModel:
    """The Route C model of the junction pair (Nets A and B), built once into a folder that lives for the session."""
    source, stack = rectangular_junction_source("VACUUM_REGION")
    folder = Path(tempfile.mkdtemp(prefix="scgsim-test-q3d-model-"))
    atexit.register(shutil.rmtree, folder, True)
    return prepare_sgb_model(
        source, prepared_stack=stack, modeling="solid", output_dir=folder
    )


def junction_q3d_spec(version: str = "2024.2", **changes) -> Q3dSpec:
    """A capacitance-only Q3D request on the junction pair's model: both Nets Signal, no terminals."""
    model = junction_model()
    nets = tuple(
        Q3dNetSpec(name, "Signal", objects)
        for name, objects in q3d_net_objects(model).items()
    )
    spec = Q3dSpec(
        aedt_version=version,
        modeling="solid",
        project_name="Stopped",
        design_name="Design",
        model=model,
        nets=nets,
        run_control=RUN,
        solve_ac_rl=False,
    )
    return replace(spec, **changes) if changes else spec


def _prepare(monkeypatch, root: Path, **request):
    plan = electrode_plan(monkeypatch)
    plan.set_vacuum_region(padding=(0.0, 0.0, 10.0))
    arguments = {
        "aedt_version": "2024.2",
        "modeling": "solid",
        "output_dir": root,
        "project_name": "Electrodes",
        "design_name": "Q3d",
        "net_types": {"A": "Signal", "B": "Ground"},
        "physical_ground_nets": ("B",),
        "run_control": RUN,
        **request,
    }
    return prepare_q3d_from_geometry(plan.prepare(), **arguments)


# ---------------------------------------------------------------- preparation and handoff


def test_the_handoff_carries_the_model_and_the_request_names_its_nets(
    tmp_path, monkeypatch
):
    """The plan's model travels in ``model/``; Nets, objects and the terminal come from it."""
    pair = Q3dTerminalPair("chip/pad_0000", "-X", "chip/pad_0000", "+X")
    handoff = _prepare(monkeypatch, tmp_path / "run", terminal_pairs={"A": pair})
    payload = json.loads(handoff.spec_path.read_text())
    assert {"bodies", "region_padding_um", "materials", "geometry_source"}.isdisjoint(
        payload
    )
    assert payload["model"]["model_dir"] == "model"
    spec = parse_aedt_spec(payload, base_dir=handoff.run_dir)
    assert isinstance(spec, Q3dSpec)
    assert spec.to_payload() == payload
    assert spec.solve_ac_rl is False
    assert spec.model.route == "C"
    assert spec.model.ground_net_ids == ("B",)
    assert [(n.name, n.net_type, n.object_names) for n in spec.nets] == [
        ("A", "Signal", ("chip_pad_0000",)),
        ("B", "Ground", ("chip_electrode_0000",)),
    ]
    assert (spec.nets[0].source_object, spec.nets[0].source_side) == (
        "chip_pad_0000",
        "-X",
    )
    receipt = json.loads(handoff.receipt_path.read_text())
    assert receipt["source"]["sgb_model_sha256"] == spec.model.identity_sha256
    manifest = json.loads(handoff.manifest_path.read_text())
    assert [p for p in manifest["allowed_paths"] if p.startswith("model/")] == [
        f"model/{spec.model.files[role]['path']}"
        for role in ("step", "geometry", "sidecar", "guard")
    ]
    metadata = json.loads(handoff.metadata_path.read_text())
    verify_prepared_hashes(metadata, handoff.spec_path, spec)


def test_the_vacuum_comes_from_the_plan_not_a_padding(tmp_path, monkeypatch):
    """Decision 5: ``region_padding_um`` is gone; the Region is the model's ``VACUUM_REGION`` box."""
    with pytest.raises(TypeError, match="region_padding_um"):
        _prepare(monkeypatch, tmp_path / "run", region_padding_um=(0.0,) * 6)


def test_net_types_must_name_exactly_the_models_nets(tmp_path, monkeypatch):
    """A Net without a Q3D type is refused before anything is written."""
    with pytest.raises(ValueError, match="net_types names"):
        _prepare(monkeypatch, tmp_path / "run", net_types={"A": "Signal"})


def test_a_terminal_entity_must_be_one_object_of_its_net(tmp_path, monkeypatch):
    """The electrode belongs to Net B, so it cannot be Net A's source."""
    pair = Q3dTerminalPair("chip/electrode_0000", "-X", "chip/pad_0000", "+X")
    with pytest.raises(ValueError, match="0 conductor objects of Net 'A'"):
        _prepare(monkeypatch, tmp_path / "run", terminal_pairs={"A": pair})


def test_a_request_whose_nets_are_not_the_models_is_refused():
    """A missing Net and a Net over other objects than the model's are refused."""
    spec = junction_q3d_spec()
    with pytest.raises(ValueError, match="are not the model's Nets"):
        replace(spec, nets=spec.nets[:1])
    with pytest.raises(ValueError, match="names objects"):
        replace(
            spec,
            nets=(replace(spec.nets[0], object_names=("substrate",)), spec.nets[1]),
        )


@pytest.mark.parametrize("version", ("2023.2", "2024.2"))
@pytest.mark.parametrize("model_only", (False, True))
def test_the_request_round_trips_with_its_stage_and_terminals(
    tmp_path, version, model_only
):
    """The model-only choice, the solve controls and an AC/RL request's terminals survive decoding."""
    first, second = junction_q3d_spec().nets
    name = first.object_names[0]
    spec = junction_q3d_spec(
        version,
        run_control=None if model_only else RUN,
        solve_ac_rl=True,
        nets=(
            replace(
                first,
                source_object=name,
                source_side="-X",
                sink_object=name,
                sink_side="+X",
            ),
            replace(second, net_type="Ground"),
        ),
    )
    attach_sgb_model(spec.model, tmp_path)
    payload = json.loads(json.dumps(spec.to_payload()))
    restored = parse_aedt_spec(payload, base_dir=tmp_path)
    assert restored.to_payload() == payload
    assert restored.aedt_version == version
    assert restored.run_control == (None if model_only else RUN)
    assert restored.nets == spec.nets


def test_a_payload_of_the_native_builder_is_refused():
    """Decision 6: a request that declares natively built bodies is not run or read."""
    payload = junction_q3d_spec().to_payload()
    payload["bodies"] = []
    with pytest.raises(ValueError, match=NATIVE_PAYLOAD_REFUSAL):
        Q3dSpec.from_payload(payload, base_dir=Path("."))


# ---------------------------------------------------------------- terminal sides (no AEDT)


def _box_faces():
    """A 0..4 x 0..2 x 0..1 box whose +X side is split into two faces sharing edge 9."""
    return [
        (1, True, [(4, 0, 0), (4, 1, 0), (4, 1, 1), (4, 0, 1)], [1, 2, 3, 9]),
        (2, True, [(4, 1, 0), (4, 2, 0), (4, 2, 1), (4, 1, 1)], [9, 4, 5, 6]),
        (3, True, [(0, 0, 0), (0, 2, 0), (0, 2, 1), (0, 0, 1)], [7, 8, 10, 11]),
        (4, True, [(0, 0, 1), (4, 0, 1), (4, 2, 1), (0, 2, 1)], [12, 13, 14, 15]),
    ]


def test_a_solid_side_is_every_planar_face_at_the_extreme():
    """A split side takes both faces; the opposite side and the top take theirs."""
    assert solid_side_faces(_box_faces(), "+X", 1e-6) == [1, 2]
    assert solid_side_faces(_box_faces(), "-X", 1e-6) == [3]
    assert solid_side_faces(_box_faces(), "+Z", 1e-6) == [4]


def test_a_solid_side_of_two_patches_is_refused():
    """Two faces on the side that share no edge are not one terminal."""
    faces = _box_faces()
    faces[1] = (2, True, faces[1][2], [20, 4, 5, 6])  # no shared edge with face 1
    with pytest.raises(Q3dTerminalError, match="do not join into one patch"):
        solid_side_faces(faces, "+X", 1e-6)


def _sheet_edges():
    """A 0..4 x 0..2 sheet at z = 0 whose +X side is split at y = 1."""
    return [
        (1, 1.0, [(4, 0, 0), (4, 1, 0)]),
        (2, 1.0, [(4, 1, 0), (4, 2, 0)]),
        (3, 4.0, [(4, 2, 0), (0, 2, 0)]),
        (4, 2.0, [(0, 2, 0), (0, 0, 0)]),
        (5, 4.0, [(0, 0, 0), (4, 0, 0)]),
    ]


def test_a_sheet_side_is_its_straight_edges_at_the_extreme_as_one_segment():
    """A side split into collinear edges takes both of them."""
    assert sheet_side_edges(_sheet_edges(), "+X", 1e-6) == [1, 2]
    assert sheet_side_edges(_sheet_edges(), "-Y", 1e-6) == [5]


def test_a_sheet_side_with_a_gap_or_out_of_plane_is_refused():
    """A side with a gap, and a side across the sheet's plane, select no terminal."""
    edges = _sheet_edges()
    edges[1] = (2, 0.5, [(4, 1.5, 0), (4, 2, 0)])
    with pytest.raises(Q3dTerminalError, match="one unbroken segment"):
        sheet_side_edges(edges, "+X", 1e-6)
    with pytest.raises(Q3dTerminalError, match="in its plane"):
        sheet_side_edges(_sheet_edges(), "+Z", 1e-6)


# ---------------------------------------------------------------- Net and terminal assignment


class _Boundary:
    """AEDT's boundary module as far as Q3D Nets and terminals use it: what was assigned, read back."""

    def __init__(self):
        self.assignments: dict[str, list[int]] = {}
        self.calls: list[list] = []

    def GetExcitationAssignment(self, name):  # noqa: N802 -- the AEDT method name
        return self.assignments[name]

    def _assign(self, arguments):
        self.calls.append(arguments)
        self.assignments[arguments[0].removeprefix("NAME:")] = list(arguments[2])

    AssignSource = _assign  # noqa: N815 -- the AEDT method name
    AssignSink = _assign  # noqa: N815 -- the AEDT method name


def _box(object_id):
    def vertex(p):
        return SimpleNamespace(position=list(p))

    faces = [
        SimpleNamespace(
            id=face_id,
            is_planar=planar,
            vertices=[vertex(p) for p in points],
            edges=[SimpleNamespace(id=e) for e in edges],
        )
        for face_id, planar, points, edges in _box_faces()
    ]
    return SimpleNamespace(id=object_id, object_type="Solid", faces=faces)


def test_nets_and_terminals_are_assigned_with_the_requests_types_and_read_back():
    """Each Net takes its request type; a solid source and sink sit on their side faces."""
    spec = junction_q3d_spec()
    first, second = spec.nets
    first = replace(
        first,
        source_object=first.object_names[0],
        source_side="-X",
        sink_object=first.object_names[0],
        sink_side="+X",
    )
    spec = replace(spec, nets=(first, replace(second, net_type="Ground")))
    objects = {
        name: _box(10 + k)
        for k, net in enumerate(spec.nets)
        for name in net.object_names
    }
    boundary = _Boundary()
    nets: list[str] = []

    def assign_net(names, name, role):
        nets.append(name)
        boundary.assignments[name] = [objects[n].id for n in names]
        return True

    app = SimpleNamespace(
        modeler=objects,
        oboundary=boundary,
        assign_net=assign_net,
        net_names=nets,
    )
    records = assign_q3d_nets(app, spec, 1e-6)
    assert [(r["name"], r["net_type"]) for r in records] == [
        ("A", "Signal"),
        ("B", "Ground"),
    ]
    assert records[0]["source"] == {
        "name": "ASource",
        "object_name": first.object_names[0],
        "side": "-X",
        "native_face_ids": [3],
        "terminal_type": "ConstantVoltage",
    }
    assert records[0]["sink"]["native_face_ids"] == [1, 2]
    assert boundary.calls[0][:3] == ["NAME:ASource", "Faces:=", [3]]
    assert "source" not in records[1]
