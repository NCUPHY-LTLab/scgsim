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
    prepare_q3d_from_geometry,
    prepare_sgb_model,
)
from scgsim.aedt.preparation.sgb_handoff import attach_sgb_model  # noqa: E402
from scgsim.aedt.preparation.sgb_model import PreparedSgbModel  # noqa: E402
from scgsim.aedt.preparation.sgb_q3d import q3d_net_objects  # noqa: E402
from scgsim.aedt.runtime.families.q3d_nets import (  # noqa: E402
    Q3dNetError,
    assign_q3d_nets,
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
    """A Q3D request on the junction pair's model: both Nets Signal."""
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
    """The plan's model travels in ``model/``; the Nets and their objects come from it."""
    handoff = _prepare(monkeypatch, tmp_path / "run")
    payload = json.loads(handoff.spec_path.read_text())
    assert {
        "bodies",
        "region_padding_um",
        "materials",
        "geometry_source",
        "solve_ac_rl",
    }.isdisjoint(payload)
    assert payload["model"]["model_dir"] == "model"
    spec = parse_aedt_spec(payload, base_dir=handoff.run_dir)
    assert isinstance(spec, Q3dSpec)
    assert spec.to_payload() == payload
    assert spec.model.route == "C"
    assert spec.model.ground_net_ids == ("B",)
    assert [(n.name, n.net_type, n.object_names) for n in spec.nets] == [
        ("A", "Signal", ("chip_pad_0000",)),
        ("B", "Ground", ("chip_electrode_0000",)),
    ]
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


def test_the_imports_first_step_runs_before_the_handoff_is_written(
    tmp_path, monkeypatch
):
    """Issue #266: a model the Q3D import refuses is refused at preparation, before anything is written."""
    import scgsim.aedt.preparation.geometry as geometry_module  # noqa: PLC0415 -- the module whose step is replaced

    families = []

    def refused(model, *, family):
        families.append(family)
        raise ValueError("the import cannot realize this model")

    monkeypatch.setattr(geometry_module, "plan_import", refused)
    with pytest.raises(ValueError, match="cannot realize"):
        _prepare(monkeypatch, tmp_path / "run")
    assert families == ["q3d"]
    assert not (tmp_path / "run").exists()


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
def test_the_request_round_trips_with_its_stage_and_net_types(
    tmp_path, version, model_only
):
    """The model-only choice, the solve controls and the Nets' Q3D types survive decoding."""
    first, second = junction_q3d_spec().nets
    spec = junction_q3d_spec(
        version,
        run_control=None if model_only else RUN,
        nets=(first, replace(second, net_type="Ground")),
    )
    attach_sgb_model(spec.model, tmp_path)
    payload = json.loads(json.dumps(spec.to_payload()))
    restored = parse_aedt_spec(payload, base_dir=tmp_path)
    assert restored.to_payload() == payload
    assert restored.aedt_version == version
    assert restored.run_control == (None if model_only else RUN)
    assert restored.nets == spec.nets


def test_route_a_sheets_are_net_objects_with_their_stack_film_thickness(tmp_path):
    """In Route A each conductor-sheet group is one object of its Net, of the film thickness the stack declares."""
    source, stack = rectangular_junction_source("VACUUM_REGION")
    model = prepare_sgb_model(
        source, prepared_stack=stack, modeling="thin_film", output_dir=tmp_path
    )
    assert model.route == "A"
    assert dict(model.sheet_thickness_um) == {"chip/left": 0.25, "chip/right": 0.25}
    assert type(model).from_payload(model.to_payload()) == model
    assert q3d_net_objects(model) == {"A": ("chip_left",), "B": ("chip_right",)}
    assert junction_model().sheet_thickness_um == {}
    with pytest.raises(ValueError, match="finite and positive"):
        replace(model, sheet_thickness_um={"chip/left": 0.0})


def test_a_payload_of_the_native_builder_is_refused():
    """Decision 6: a request that declares natively built bodies is not run or read."""
    payload = junction_q3d_spec().to_payload()
    payload["bodies"] = []
    with pytest.raises(ValueError, match=NATIVE_PAYLOAD_REFUSAL):
        Q3dSpec.from_payload(payload, base_dir=Path("."))


# ---------------------------------------------------------------- Net assignment


def test_nets_are_assigned_with_the_requests_types_and_read_back():
    """Each Net takes its request type over its objects; a Net that reads back other objects is refused."""
    spec = junction_q3d_spec()
    first, second = spec.nets
    spec = replace(spec, nets=(first, replace(second, net_type="Ground")))
    objects = {
        name: SimpleNamespace(id=10 + k)
        for k, net in enumerate(spec.nets)
        for name in net.object_names
    }
    assignments: dict[str, list[int]] = {}
    nets: list[str] = []

    def assign_net(names, name, role):
        nets.append(name)
        assignments[name] = [objects[n].id for n in names]
        return True

    app = SimpleNamespace(
        modeler=objects,
        oboundary=SimpleNamespace(GetExcitationAssignment=assignments.__getitem__),
        assign_net=assign_net,
        net_names=nets,
    )
    records = assign_q3d_nets(app, spec)
    assert records == [
        {
            "name": "A",
            "net_type": "Signal",
            "object_names": list(first.object_names),
            "native_object_ids": [10],
        },
        {
            "name": "B",
            "net_type": "Ground",
            "object_names": list(second.object_names),
            "native_object_ids": [11],
        },
    ]

    def assign_elsewhere(names, name, role):
        nets.append(name)
        assignments[name] = [99]
        return True

    nets.clear()
    app.assign_net = assign_elsewhere
    with pytest.raises(Q3dNetError, match="native net assignment differs"):
        assign_q3d_nets(app, spec)
