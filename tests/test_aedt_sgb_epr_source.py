"""What the EPR of an imported SGB model integrates, and how one mode's integrals combine (issue #67).

The models are real SGB builds of the rectangular junction pair with MA, MS and SA bands.  The integrals of
the combination tests are chosen numbers, not a solve: they check the arithmetic against values worked out
by hand.  Skipped where OCP (the ``occt`` group) is missing.
"""

from __future__ import annotations

import pytest

pytest.importorskip("OCP")

from sgb_inputs import rectangular_junction_source  # noqa: E402

from scgsim.aedt.epr.expression_codec import compile_named_expression  # noqa: E402
from scgsim.aedt.epr.integrals import (  # noqa: E402
    empty_raw_integrals,
    integral_target,
    purpose_unit,
    store_integral,
)
from scgsim.aedt.epr.sgb_energy import (  # noqa: E402
    SURFACE_GRANULARITY,
    combine_sgb_epr_mode,
)
from scgsim.aedt.epr.sgb_expressions import (  # noqa: E402
    normalized_surface_total_operations,
    sgb_expression_requests,
)
from scgsim.aedt.epr.sgb_source import sgb_epr_source  # noqa: E402
from scgsim.aedt.epr.sgb_surfaces import (  # noqa: E402
    junction_face_bindings,
    surface_face_bindings,
)
from scgsim.aedt.epr.surface import VACUUM_PERMITTIVITY_F_PER_M  # noqa: E402
from scgsim.aedt.preparation.sgb_model import prepare_sgb_model  # noqa: E402
from scgsim.aedt.runtime.families.sgb_import import planned_import  # noqa: E402
from scgsim.aedt.specs.epr import (  # noqa: E402
    EprAnalysisRequest,
    NormalizedSurfaceEprTotal,
    PlanarJunction,
)
from scgsim.contracts.sgb_sidecar import read_sidecar, views  # noqa: E402
from scgsim.geometry.sgb.output.exchange.guard import read_step_guard  # noqa: E402
from scgsim.geometry.source.surface_epr_specs import normalize_surface_epr_specs  # noqa: E402

pytestmark = pytest.mark.occt

FILM = {"film_thickness_m": 3e-9, "permittivity": 10.0, "loss_tangent": 1e-3}
JUNCTION = PlanarJunction(
    "junction", "sheet_polygon", "A", "B", (1, 0), 2.0, 10e-9, 2e-15
)


def _specs(**overrides):
    return normalize_surface_epr_specs(
        {
            kind: {**FILM, "inset_margins_um": [0.5], **overrides}
            for kind in ("MA", "MS", "SA")
        },
        materials={},
    )


def _prepare(tmp_path, modeling, *, specs=None, junction=JUNCTION, **source_options):
    source, stack = rectangular_junction_source("VACUUM_REGION", **source_options)
    return prepare_sgb_model(
        source,
        prepared_stack=stack,
        modeling=modeling,
        output_dir=tmp_path,
        junctions=(junction,),
        surface_epr_specs=_specs() if specs is None else specs,
    )


@pytest.fixture(scope="module")
def solid(tmp_path_factory):
    """The Route C model (a Route B request) of the junction pair."""
    return _prepare(tmp_path_factory.mktemp("solid"), "solid")


def test_the_domains_are_the_dielectric_solids_and_the_region(solid):
    """Conductor solids are PEC and hold no field; the vacuum is AEDT's Region."""
    source = sgb_epr_source(solid)
    assert {
        name: (d.material_id, d.relative_permittivity)
        for name, d in source.domains.items()
    } == {"substrate": ("Si", 4.0), "Region": ("air", 1.0)}


def test_entries_are_a_baseline_per_unit_then_its_inset_margin(solid):
    """Each band family is one unit: a baseline over both bands, and the 0.5 um margin over the inner band."""
    source = sgb_epr_source(solid)
    entries = {e.entry_id: e for e in source.entries}
    top = entries["MA::chip/left__top"]
    inset = entries["MA::chip/left__top__inset_0"]
    assert (top.evaluation_kind, top.margin_um, len(top.rows)) == (
        "unmasked_baseline",
        0.0,
        2,
    )
    assert (inset.evaluation_kind, inset.margin_um) == ("requested_margin", 0.5)
    assert [source.rows[row].band for row in inset.rows] == [1]
    # A sidewall has no band, so it has a baseline and no margin entry.
    assert "MA::chip/left__sidewall" in entries
    assert "MA::chip/left__sidewall__inset_0" not in entries
    baselines = [e for e in source.entries if e.evaluation_kind == "unmasked_baseline"]
    assert source.entries[: len(baselines)] == tuple(baselines)
    assert {e.entry_id: e.substrate_relative_permittivity for e in baselines} == {
        "MA::chip/left__sidewall": None,
        "MA::chip/left__top": None,
        "MA::chip/right__sidewall": None,
        "MA::chip/right__top": None,
        "MS::chip/left__bottom": 4.0,
        "MS::chip/right__bottom": 4.0,
        "SA::substrate|VACUUM_REGION__interface": 4.0,
    }


def test_the_junction_sheet_comes_from_the_sidecar_port(solid):
    """The voltage is read along the port's direction and divided by the sheet's width across it."""
    sheet = sgb_epr_source(solid).junctions["junction"]
    assert (sheet.port_group, sheet.direction, sheet.width_um) == (
        "PORT::sheet",
        (1.0, 0.0, 0.0),
        2.0,
    )


def test_a_junction_stating_another_width_is_refused(tmp_path):
    """The junction's width_um must be the width of the port sheet SGB built."""
    wrong = PlanarJunction(
        "junction", "sheet_polygon", "A", "B", (1, 0), 1.5, 10e-9, 0.0
    )
    with pytest.raises(ValueError, match=r"states width_um 1\.5.*2\.0 um wide"):
        _prepare(tmp_path, "solid", junction=wrong)


def test_surface_epr_groups_select_rows_by_sidecar_group(solid):
    """One selected group integrates its rows only, and its entries cover only those rows."""
    group = "MA::chip/left__top::band=1"
    source = sgb_epr_source(solid, surface_epr_groups=(group,))
    assert [row.group for row in source.rows.values()] == [group]
    assert [(e.entry_id, e.rows) for e in source.entries] == [
        ("MA::chip/left__top", tuple(source.rows)),
        ("MA::chip/left__top__inset_0", tuple(source.rows)),
    ]
    with pytest.raises(ValueError, match="carry no Surface-EPR row"):
        sgb_epr_source(solid, surface_epr_groups=("MA::nowhere",))


def test_a_route_a_sheet_carries_ma_and_ms_rows(tmp_path):
    """A Route A conductor sheet is one face read on its dielectric side by both its MA and MS rows."""
    source = sgb_epr_source(_prepare(tmp_path, "thin_film"))
    assert {e.entry_id for e in source.entries} >= {
        "MA::chip/left__sheet",
        "MS::chip/left__sheet",
        "MS::chip/left__sheet__inset_0",
    }


def test_vacuum_split_by_the_substrate_is_one_region(tmp_path):
    """Vacuum above and below a substrate spanning the box is two SGB volumes of one material: one Region."""
    model = _prepare(tmp_path, "solid", vacuum_z_min_um=-3)
    sidecar = read_sidecar(model.path("sidecar"))
    vacuum = [v for v in sidecar["volume"].values() if v["kind"] == "vacuum"]
    assert len(vacuum) == 2
    source = sgb_epr_source(model)
    assert {
        name: (d.material_id, d.relative_permittivity)
        for name, d in source.domains.items()
    } == {"substrate": ("Si", 4.0), "Region": ("air", 1.0)}


def test_vacuum_volumes_of_two_materials_are_refused(tmp_path):
    """The Region is one box of one material, so vacuum volumes of two materials have no Region."""
    model = _prepare(tmp_path, "solid", vacuum_z_min_um=-3)
    sidecar = read_sidecar(model.path("sidecar"))
    first = next(v for v in sidecar["volume"].values() if v["kind"] == "vacuum")
    first["material"] = "other_air"
    with pytest.raises(
        ValueError, match=r"vacuum volumes have materials \['air', 'other_air'\]"
    ):
        sgb_epr_source(model, sidecar=sidecar)


def test_a_sheet_row_selects_only_the_faces_its_face_sides_names(tmp_path):
    """The left film overhangs the substrate: its MS row leaves the overhang face out, its MA row keeps it."""
    # No inset margin: a band would need the die outline farther from the cut film than its margin.
    specs = normalize_surface_epr_specs(dict.fromkeys(("MA", "MS"), FILM), materials={})
    model = _prepare(tmp_path, "thin_film", specs=specs, substrate_x_min_um=-1)
    sidecar, imported = _as_imported(model)
    rows = sidecar["surface_epr_row"]
    members = views.group_members(sidecar)["chip/left"]
    ms_faces = set(rows["MS::chip/left__sheet"]["face_sides"])
    (overhang,) = set(members) - ms_faces
    assert rows["MA::chip/left__sheet"]["face_sides"][overhang] == "both"
    bindings = surface_face_bindings(sidecar, imported)
    # Over vacuum on both sides the overhang is a free face, imported once as its rebuilt sheet.
    ((sheet, face_id),) = imported.face_copies[overhang].items()
    assert overhang in imported.sheets
    assert (sheet, face_id) in bindings["MA::chip/left__sheet"].faces
    assert (sheet, face_id) not in bindings["MS::chip/left__sheet"].faces
    assert len(bindings["MS::chip/left__sheet"].faces) == len(ms_faces) == 1
    assert len(bindings["MA::chip/left__sheet"].faces) == len(members) == 2


def test_a_film_in_palace_native_units_is_refused(tmp_path):
    """AEDT has no Model.L0, so the film thickness must be given in metres."""
    specs = normalize_surface_epr_specs(
        {"MA": {"thickness": 3e-3, "permittivity": 10.0}}, materials={}
    )
    model = _prepare(tmp_path, "solid", specs=specs)
    with pytest.raises(ValueError, match="film_thickness_m"):
        sgb_epr_source(model)


# ---------------------------------------------------------------- combination


def _q(value, unit):
    return {"value": value, "unit": unit}


def _raw(source, *, electric=1e-12, junction=(1e-6, 0.0)):
    """One mode's integrals: every domain and row the same chosen values."""
    domains = list(source.domains)
    return {
        "frequency_hz": _q(5e9, "Hz"),
        "electric_domain_integrals_v2_m": {d: _q(electric, "V^2*m") for d in domains},
        "magnetic_domain_integrals_a2_m": {d: _q(0.0, "A^2*m") for d in domains},
        "relative_permeability": {d: _q(1.0, "1") for d in domains},
        "effective_domain_volumes_m3": {d: _q(1e-15, "m^3") for d in domains},
        "surface_integrals_v2": {
            row: {"normal": _q(2.0, "V^2"), "tangential": _q(3.0, "V^2")}
            for row in source.rows
        },
        "masked_areas_m2": {row: _q(1e-12, "m^2") for row in source.rows},
        "surface_granularity": SURFACE_GRANULARITY,
        "junction_integrals_v_m": {
            j: {"real": _q(junction[0], "V*m"), "imag": _q(junction[1], "V*m")}
            for j in source.junctions
        },
    }


def test_one_mode_combines_by_hand(solid):
    """Normalization, junction voltage and an entry's energy agree with the formulas worked by hand."""
    source = sgb_epr_source(solid)
    record = combine_sgb_epr_mode(source, _raw(source))
    eps0 = VACUUM_PERMITTIVITY_F_PER_M
    electric = eps0 * (4.0 + 1.0) * 1e-12 / 2
    voltage = 1e-6 / 2e-6  # area integral over the 2 um width
    capacitive = 2e-15 * voltage**2 / 2
    assert record["electric_energy_j"] == pytest.approx(electric)
    assert record["junctions"][0]["voltage_real_v"] == pytest.approx(voltage)
    assert record["normalization_energy_j"] == pytest.approx(electric + capacitive)
    sa = next(
        c
        for c in record["surface_contributions"]
        if c["group_id"] == "SA::substrate|VACUUM_REGION__interface"
    )
    # SA: eps0 * t * (eps_film * tangential + normal / eps_film) / 2, per row, two rows.
    per_row = eps0 * 3e-9 * (10.0 * 3.0 + 2.0 / 10.0) / 2
    assert sa["energy_j"] == pytest.approx(2 * per_row)
    assert sa["participation"] == pytest.approx(
        2 * per_row / record["normalization_energy_j"]
    )
    assert sa["inverse_q"] == pytest.approx(sa["participation"] * 1e-3)
    assert [m["band"] for m in sa["members"]] == [0, 1]


def test_the_request_selects_reported_rows_but_not_the_normalization(solid):
    """Every junction and domain enters the normalization; the request selects only what is reported."""
    source = sgb_epr_source(solid)
    full = combine_sgb_epr_mode(source, _raw(source))
    narrow = combine_sgb_epr_mode(
        source,
        _raw(source),
        request=EprAnalysisRequest(junction_ids=(), bulk_domain_ids=("Region",)),
    )
    assert narrow["normalization_energy_j"] == full["normalization_energy_j"]
    assert narrow["junctions"] == []
    assert [d["domain_id"] for d in narrow["electric_domains"]] == ["Region"]
    with pytest.raises(ValueError, match="unknown ids"):
        combine_sgb_epr_mode(
            source, _raw(source), request=EprAnalysisRequest(junction_ids=("x",))
        )


def test_integrals_of_another_model_are_refused(solid):
    """A raw mode missing a row, or of another surface granularity, is refused rather than partly combined."""
    source = sgb_epr_source(solid)
    raw = _raw(source)
    raw["surface_integrals_v2"].pop(next(iter(source.rows)))
    with pytest.raises(ValueError, match="must cover the Surface-EPR rows"):
        combine_sgb_epr_mode(source, raw)
    raw = {**_raw(source), "surface_granularity": "owner_interface_margin.v1"}
    with pytest.raises(ValueError, match="surface granularity"):
        combine_sgb_epr_mode(source, raw)


# ---------------------------------------------------------------- expressions


def _as_imported(model):
    """The sidecar of ``model`` and the import its guard plans (each face once on each solid, free faces as sheets)."""
    guard = read_step_guard(model.path("guard"))
    return read_sidecar(model.path("sidecar")), planned_import(guard)


def _requests(model, **selectors):
    sidecar, imported = _as_imported(model)
    source = sgb_epr_source(model, sidecar=sidecar)
    rows = {
        row: binding
        for row, binding in surface_face_bindings(sidecar, imported).items()
        if row in source.rows
    }
    junctions = junction_face_bindings(
        sidecar, {j: s.port_group for j, s in source.junctions.items()}, imported
    )
    requests = sgb_expression_requests(
        source,
        rows,
        junctions,
        relative_permeability=dict.fromkeys(source.domains, 1.0),
        field_selectors=selectors,
    )
    return source, junctions, requests


def test_the_junction_voltage_sheet_copies_the_substrate_port_face(solid):
    """The port face is on the substrate under the gap; its sheet is named after the junction."""
    _, junctions, _ = _requests(solid)
    (binding,) = junctions.values()
    assert binding.selection_name == "epr_junction_junction"
    assert [name for name, _ in binding.faces] == ["substrate"]


def test_each_row_is_read_on_its_sheet_with_the_sheet_normal(solid):
    """Three integrals per row with the geometry normal; an adjacent group reads off its sheet."""
    group = "SA::substrate|VACUUM_REGION__interface::band=0"
    source, _, requests = _requests(solid, **{group: "adjacent"})
    by_purpose = {r.purpose: r for r in requests}
    row = next(r for r in source.rows.values() if r.group == group)
    normal = by_purpose[f"electric_normal_{row.row}"]
    assert "Operation('Normal')" in normal.operations
    assert normal.operations[-3] == f"EnterAdjacentSurf('{row.selection_name}')"
    assert normal.adjacent_selection_name == row.selection_name
    area = by_purpose[f"masked_area_{row.row}"]
    assert area.operations[-3] == f"EnterSurface('{row.selection_name}')"
    assert area.adjacent_selection_name is None
    assert len(requests) == 3 * len(source.domains) + 3 * len(source.rows) + 2
    with pytest.raises(ValueError, match="no selected row carries"):
        _requests(solid, **{"MA::nowhere": "surface"})


def test_compiled_integrals_combine_into_one_mode(solid):
    """Each compiled expression stores its value where the combination reads it."""
    source, _, requests = _requests(solid)
    raw = empty_raw_integrals()
    raw["surface_granularity"] = SURFACE_GRANULARITY
    raw["frequency_hz"] = _q(5e9, "Hz")
    values = {"V^2*m": 1e-12, "A^2*m": 0.0, "m^3": 1e-15, "V^2": 1.0, "m^2": 1e-12}
    for request in requests:
        record, _, _ = compile_named_expression(
            purpose=request.purpose,
            operations=request.operations,
            solution="Setup1 : LastAdaptive",
            phase_degrees=0.0,
            dependencies={},
            selection=request.selection,
            adjacent_selection_name=request.adjacent_selection_name,
        )
        target = integral_target(request.purpose, record["identity"]["selection"])
        unit = purpose_unit(target.kind)
        store_integral(raw, target=target, value=_q(values.get(unit, 1e-6), unit))
    record = combine_sgb_epr_mode(source, raw)
    assert {c["group_id"] for c in record["surface_contributions"]} == {
        e.entry_id for e in source.entries
    }


def test_the_convergence_target_sums_the_rows_of_its_entries(solid):
    """An MS baseline target reads the normal integral of every MS row, each once."""
    source, _, requests = _requests(solid)
    names = {r.purpose: f"n_{k}" for k, r in enumerate(requests)}
    operations = normalized_surface_total_operations(
        source, names, NormalizedSurfaceEprTotal("MS", "unmasked_baseline", 0.0)
    )
    for row in source.rows.values():
        if row.interface_type == "MS":
            name = names[f"electric_normal_{row.row}"]
            assert operations.count(f"NameOfExpression('{name}')") == 1
    assert operations[-1] == "Operation('/')"
