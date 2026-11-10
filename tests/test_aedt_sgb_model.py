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
from scgsim.aedt.specs.epr import PlanarJunction  # noqa: E402

pytestmark = pytest.mark.occt

JUNCTION = PlanarJunction(
    "junction", "sheet_polygon", "A", "B", (1, 0), 2.0, 10e-9, 0.0
)
BOX = (-4.0, -2.0, -2.0, 14.0, 4.0, 4.25)


def _prepare(tmp_path, modeling, *, junctions=(JUNCTION,), vacuum="VACUUM_REGION"):
    source, stack = rectangular_junction_source(vacuum)
    return prepare_sgb_model(
        source,
        prepared_stack=stack,
        modeling=modeling,
        output_dir=tmp_path / modeling,
        junctions=junctions,
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
