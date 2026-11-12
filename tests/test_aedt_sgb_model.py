"""``prepare_sgb_model``: the guarded STEP AEDT imports, built from the same SGB model Palace meshes.

Each test builds a real SGB model of a junction pair (two face metals joined by a rectangular junction)
and checks what the AEDT runtime relies on: the realized route, the solids in the STEP, the junction's
port group, the Region box and the file binding.  Skipped where OCP (the ``occt`` group) is missing.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

pytest.importorskip("OCP")

from sgb_inputs import rectangular_junction_source  # noqa: E402

from scgsim.aedt.preparation.sgb_model import (  # noqa: E402
    PreparedSgbModel,
    prepare_sgb_model,
)
from scgsim.aedt.specs.common import (  # noqa: E402
    LumpedRlc,
    LumpedTerminalPort,
    ModalPort,
    TerminalPort,
)
from scgsim.aedt.specs.epr import PlanarJunction  # noqa: E402

pytestmark = pytest.mark.occt

JUNCTION = PlanarJunction(
    "junction", "sheet_polygon", "A", "B", (1, 0), 2.0, 10e-9, 0.0
)
BOX = (-4.0, -2.0, -2.0, 14.0, 4.0, 4.25)


RLC = LumpedRlc("sheet", "series", resistance_ohm=50.0, capacitance_f=0.0)


def _prepare(  # noqa: PLR0913 -- each test varies one treatment of the same source
    tmp_path,
    modeling,
    *,
    junctions=(JUNCTION,),
    vacuum="VACUUM_REGION",
    lumped_rlcs=(),
    ports=(),
    bump=False,
):
    source, stack = rectangular_junction_source(vacuum, bump=bump)
    return prepare_sgb_model(
        source,
        prepared_stack=stack,
        modeling=modeling,
        output_dir=tmp_path / modeling,
        junctions=junctions,
        lumped_rlcs=lumped_rlcs,
        ports=ports,
    )


def _guard(model):
    return json.loads(model.path("guard").read_text(encoding="utf-8"))


def test_solid_modeling_is_the_route_c_model_with_conductor_solids_and_no_vacuum(
    tmp_path,
):
    """``solid`` is a Route B request built as Route C: substrate and conductor solids, no vacuum."""
    model = _prepare(tmp_path, "solid")
    assert (model.requested_route, model.route) == ("B", "C")
    names = {solid["aedt_name"] for solid in _guard(model)["solids"]}
    assert names == {"substrate", "chip_left", "chip_right"}
    assert model.vacuum_bounds_um == BOX
    assert [item.port_group for item in model.junctions] == ["PORT::sheet"]


def test_thin_film_modeling_is_route_a_with_the_sheets_on_the_substrate(tmp_path):
    """``thin_film`` is Route A: only the substrate is a solid, and the sheets are its faces."""
    model = _prepare(tmp_path, "thin_film")
    assert (model.requested_route, model.route) == ("A", "A")
    guard = _guard(model)
    assert [solid["aedt_name"] for solid in guard["solids"]] == ["substrate"]
    assert (
        guard["free_faces"] == []
    )  # both sheets and the port lie on the substrate's top face
    assert model.vacuum_bounds_um[:5] == BOX[:5]


def test_the_record_round_trips_and_checks_its_files(tmp_path):
    """The payload round-trips, and a changed STEP fails the file check."""
    model = _prepare(tmp_path, "solid")
    assert PreparedSgbModel.from_payload(model.to_payload()) == model
    model.verify_files()
    model.path("step").write_bytes(model.path("step").read_bytes() + b"\n")
    with pytest.raises(ValueError, match="step file model.step is missing or changed"):
        model.verify_files()


def test_a_record_with_a_route_its_modeling_does_not_realize_is_refused(tmp_path):
    """The record refuses a route its modeling does not realize."""
    model = _prepare(tmp_path, "solid")
    with pytest.raises(ValueError, match="realized as Route C"):
        replace(model, route="A")


def test_a_model_without_the_plans_vacuum_region_is_refused(tmp_path):
    """Without the plan's VACUUM_REGION there is no box for the AEDT Region."""
    with pytest.raises(ValueError, match="VACUUM_REGION"):
        _prepare(tmp_path, "solid", vacuum="upper_air")


def test_a_junction_without_a_rectangular_junction_spec_is_refused(tmp_path):
    """A junction is not partitioned again here; it needs its plan's spec."""
    other = replace(JUNCTION, junction_id="other")
    with pytest.raises(ValueError, match="no RectangularJunctionSpec"):
        _prepare(tmp_path, "solid", junctions=(other,))


def test_a_junction_naming_other_nets_than_its_spec_is_refused(tmp_path):
    """A junction whose Nets differ from its spec's is refused."""
    swapped = replace(JUNCTION, terminal_a_net="B", terminal_b_net="A")
    with pytest.raises(ValueError, match="RectangularJunctionSpec"):
        _prepare(tmp_path, "solid", junctions=(swapped,))


def test_a_lumped_rlc_names_its_port_sheet_and_round_trips(tmp_path):
    """Decision 8: a ``LumpedRlc`` sits on the ``PORT::<port_sheet_id>`` group it names, like Palace's."""
    model = _prepare(tmp_path, "solid", junctions=(), lumped_rlcs=(RLC,))
    assert model.lumped_rlcs == (RLC,)
    assert model.to_payload()["lumped_rlcs"] == [RLC.to_payload()]
    assert PreparedSgbModel.from_payload(model.to_payload()) == model


def test_a_lumped_rlc_on_a_port_sheet_the_model_lacks_is_refused(tmp_path):
    """An RLC naming no port sheet of the model is refused after the build."""
    with pytest.raises(ValueError, match="no lumped-port group"):
        _prepare(
            tmp_path,
            "solid",
            junctions=(),
            lumped_rlcs=(replace(RLC, port_sheet_id="x"),),
        )


def test_a_port_sheet_with_a_junction_and_an_rlc_is_refused(tmp_path):
    """One port sheet carries one treatment: a junction and an RLC on it are refused before any build."""
    with pytest.raises(ValueError, match="carry more than one treatment"):
        _prepare(tmp_path, "solid", lumped_rlcs=(RLC,))
    assert not (tmp_path / "solid").exists()


def test_a_lumped_port_names_its_port_sheet(tmp_path):
    """A ``LumpedTerminalPort`` is bound to the port sheet it names."""
    port = LumpedTerminalPort(1, "p1", "sheet", ("chip_left",), ("chip_right",), 50.0)
    model = _prepare(tmp_path, "solid", junctions=(), ports=(port,))
    assert model.lumped_ports == (port,)
    assert PreparedSgbModel.from_payload(model.to_payload()) == model


@pytest.mark.parametrize(
    ("port", "kind"),
    [
        (TerminalPort(1, "wave", "-X", ("chip_right",)), "Driven wave Terminal"),
        (
            ModalPort(1, "modal", "-X", ((-4.0, 1.0, 0.0), (-4.0, 1.0, 4.0))),
            "Driven Modal",
        ),
    ],
)
def test_wave_and_modal_ports_are_refused_naming_the_follow_up(tmp_path, port, kind):
    """Decision 8: the SGB flow refuses Driven wave and modal ports, naming #135, before any build."""
    with pytest.raises(NotImplementedError, match=rf"{kind} ports.*scgsim#135"):
        _prepare(tmp_path, "solid", junctions=(), ports=(port,))
    assert not (tmp_path / "solid").exists()


def test_thin_film_keeps_a_finite_body_as_a_conductor_solid_beside_the_sheets(tmp_path):
    """Decision 8: in AEDT a Route A bump is a solid (made PEC), as Route B is built as Route C."""
    model = _prepare(tmp_path, "thin_film", bump=True)
    assert model.route == "A"
    sidecar = json.loads(model.path("sidecar").read_text(encoding="utf-8"))
    assert sidecar["route"] == "A"
    assert sidecar["route_policy"] == {
        "conductor_volumes": "retained",
        "conductor_sheets": True,
    }
    solids = {s["solid"]: s["aedt_name"] for s in _guard(model)["solids"]}
    kinds = {sidecar["volumes"][solid]["kind"] for solid in solids}
    assert kinds == {"dielectric", "conductor"}
    bodies = [s for s in solids if sidecar["volumes"][s]["kind"] == "conductor"]
    assert [sidecar["volumes"][s]["net"] for s in bodies] == ["A"]
    sheets = [
        name
        for name, group in sidecar["physical_groups"].items()
        if group.get("interface_type") == "SHEET"
    ]
    assert sheets  # the films are still sheets
