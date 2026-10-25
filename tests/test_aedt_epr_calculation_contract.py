"""Independent analytical EPR arithmetic; no layout, fields, or native result.

The detached input carrier holds only material/owner/binding identities needed by
these public pure calculations. Synthetic mode/pass labels test result identity,
not native observations. No receipt, saved project, or native evidence is made.

Hand calculation, using the contract's SI vacuum permittivity below:
  thickness=1/2 m, film epsilon=2, substrate epsilon=4, N=8 V², T=2 V².
  MA: epsilon0*N/8 = epsilon0.
  MS: 2*epsilon0*N = 16*epsilon0.
  SA: epsilon0*N/8 + epsilon0*T/2 = 2*epsilon0.
  Bulk: epsilon0*4*8/2 = 16*epsilon0 J.
  Junction: C=32*epsilon0 F, |V|=1 V -> 16*epsilon0 J.
Thus the denominator is 32*epsilon0 J and p=(1/32,1/2,1/16).
Masked inputs halve N and T. All scale factors are powers of two, so exact
binary64 comparisons need no test-chosen numerical tolerance.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest

from scgsim.aedt import (
    EprResult,
    PlanarJunction,
    PreparedPlanarGeometry,
    SurfaceEprSpec,
    combine_epr_mode,
    combine_surface_epr_mode,
    reanalyze_epr,
)
from scgsim.aedt.epr.analysis import _surface_energy_coefficients


EPSILON_0 = 8.8541878128e-12  # SI constant, independently written, not imported from the formula.
ENERGIES = {"MA": EPSILON_0, "MS": 16 * EPSILON_0, "SA": 2 * EPSILON_0}
PARTICIPATIONS = {"MA": 1 / 32, "MS": 1 / 2, "SA": 1 / 16}
EVALUATIONS = (("unmasked_baseline", 0.0), ("requested_margin", 0.0), ("requested_margin", 1.0))


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _quantity(value, unit):
    return {"value": value, "unit": unit}


def _analytical_inputs():
    """Build typed detached arithmetic inputs without invoking a geometry producer."""
    source = {
        "materials": {
            "vacuum": {"kind": "vacuum", "permittivity": 1.0},
            "dielectric": {"kind": "dielectric", "permittivity": 4.0},
        },
        "solution_regions": [
            {"semantic_id": "substrate", "material_id": "dielectric", "material_kind": "dielectric"},
            {"semantic_id": "vacuum", "material_id": "vacuum", "material_kind": "vacuum"},
        ],
        "conductors": [{"semantic_id": "metal"}],
        "surface_evaluation_policy": "unmasked_plus_requested.v1",
    }
    junction = PlanarJunction(
        "junction", "junction_polygon", "terminal_a", "terminal_b", (1.0, 0.0),
        width_um=1_000_000.0, inductance_h=1.0, capacitance_f=32 * EPSILON_0,
    )
    specs = []
    bindings = []
    groups = []
    for interface in ("MA", "MS", "SA"):
        contribution_id = "analytic_" + interface
        side = "bottom" if interface == "MS" else "top"
        owners = ["substrate"] if interface == "SA" else ["metal", "substrate"]
        spec = SurfaceEprSpec(
            contribution_id, interface + "_patch_1", interface, side,
            margins_um=(0.0, 1.0), film_thickness_m=0.5,
            film_relative_permittivity=2.0, loss_tangent=0.25, source="analytical_input",
        )
        specs.append(spec)
        members = []
        for part in (1, 2):
            binding_id = f"{interface}_binding_{part}"
            geometry_ref = {"source_polygon_id": f"{interface}_patch_{part}"}
            effective_domain = "substrate" if interface == "MS" else "vacuum"
            effective_material_id = "dielectric" if interface == "MS" else "vacuum"
            bindings.append({
                "binding_id": binding_id,
                "contribution": {"contribution_id": contribution_id, "classification": interface,
                                 "side": side, "source_owner_ids": owners},
                "owner_semantic_id": "substrate" if interface == "SA" else "metal",
                "effective_domain_id": effective_domain,
                "effective_material_id": effective_material_id,
                "effective_material": source["materials"][effective_material_id],
                "substrate_domain_id": "substrate",
                "substrate_material": source["materials"]["dielectric"],
                "geometry_ref": geometry_ref,
            })
            members.append({
                "binding_id": binding_id, "contribution_id": contribution_id,
                "source_polygon_id": spec.source_polygon_id, "source_owner_ids": owners,
                "field_side": side, "effective_domain_id": effective_domain,
                "effective_material_id": effective_material_id,
                "substrate_domain_id": "substrate", "geometry_ref": geometry_ref,
            })
        for evaluation, margin in EVALUATIONS:
            identity = {
                "owner_id": "substrate" if interface == "SA" else "metal",
                "interface_kind": interface, "film_thickness_m": 0.5,
                "film_relative_permittivity": 2.0,
                "substrate_relative_permittivity": 1.0 if interface == "MA" else 4.0,
                "margin_um": margin, "evaluation_kind": evaluation,
            }
            # This supplies the existing raw-group identity codec, not an energy expectation.
            groups.append({"group_id": "surface_group_" + _digest(identity)[:24], **identity, "members": members})
    model = {"modeling": "solid", "source": source, "junctions": [junction.to_payload()]}
    complete = {**model, "contribution_catalog": [], "contributions": [item.to_payload() for item in specs],
                "surface_bindings": bindings}
    prepared = PreparedPlanarGeometry(
        modeling="solid", source=source, junctions=(junction,), contribution_catalog=(),
        contributions=tuple(specs), surface_bindings=tuple(bindings),
        model_sha256=_digest(model), source_sha256=_digest(complete),
    )
    raw = {
        "frequency_hz": _quantity(8.0, "Hz"),
        "electric_domain_integrals_v2_m": {"substrate": _quantity(8.0, "V^2*m"), "vacuum": _quantity(0.0, "V^2*m")},
        "magnetic_domain_integrals_a2_m": {key: _quantity(0.0, "A^2*m") for key in ("substrate", "vacuum")},
        "relative_permeability": {key: _quantity(1.0, "1") for key in ("substrate", "vacuum")},
        "effective_domain_volumes_m3": {key: _quantity(1.0, "m^3") for key in ("substrate", "vacuum")},
        "junction_integrals_v_m": {"junction": {"real": _quantity(1.0, "V*m"), "imag": _quantity(0.0, "V*m")}},
        "surface_granularity": "owner_interface_margin.v1",
        "surface_group_provenance": {group["group_id"]: group for group in groups},
        "surface_integrals_v2": {
            group["group_id"]: {"normal": _quantity(4.0 if group["margin_um"] else 8.0, "V^2"),
                                "tangential": _quantity(1.0 if group["margin_um"] else 2.0, "V^2")}
            for group in groups
        },
        "masked_areas_m2": {group["group_id"]: _quantity(0.5 if group["margin_um"] else 1.0, "m^2") for group in groups},
    }
    return prepared, raw


@pytest.mark.parametrize("interface,normal,tangential", (
    ("MA", EPSILON_0 / 8, 0.0),
    ("MS", 2 * EPSILON_0, 0.0),
    ("SA", EPSILON_0 / 8, EPSILON_0 / 2),
))
def test_peak_phasor_coefficients_match_independent_hand_calculation(interface, normal, tangential):
    # This coefficient-only helper is internal; public integrations are checked below.
    assert _surface_energy_coefficients(interface, 0.5, 2.0, 4.0) == (normal, tangential)


def _assert_surface_arithmetic(rows, *, thickness_scale=1):
    assert len(rows) == 9  # Three interfaces x three logical evaluations, not eighteen patches.
    identities = {(row["interface_kind"], row["evaluation_kind"], row["margin_um"]) for row in rows}
    assert identities == {(interface, kind, margin) for interface in ENERGIES for kind, margin in EVALUATIONS}
    assert len({row["group_id"] for row in rows}) == 9
    for row in rows:
        scale = 0.5 if row["margin_um"] else 1
        assert row["energy_j"] == ENERGIES[row["interface_kind"]] * scale * thickness_scale
        assert row["participation"] == PARTICIPATIONS[row["interface_kind"]] * scale * thickness_scale
        assert len(row["members"]) == 2
        assert {member["binding_id"] for member in row["members"]} == {
            row["interface_kind"] + "_binding_1", row["interface_kind"] + "_binding_2"}


def test_public_surface_and_complete_composition_share_full_denominator_and_group_once():
    prepared, raw = _analytical_inputs()
    surface = combine_surface_epr_mode(prepared, raw)
    full = combine_epr_mode(prepared, raw)
    assert full["electric_energy_j"] == 16 * EPSILON_0
    assert full["junction_capacitive_energy_j"] == 16 * EPSILON_0
    assert full["normalization_energy_j"] == surface["normalization_energy_j"] == 32 * EPSILON_0
    assert full["surface_contributions"] == surface["surface_contributions"]
    _assert_surface_arithmetic(full["surface_contributions"])


def _analytical_history(prepared, raw):
    rows = []
    for mode in (1, 2):
        for native_pass in (1, 2):
            record = deepcopy(raw)
            record["frequency_hz"] = _quantity(8.0 * mode, "Hz")
            rows.append({"mode": mode, "native_pass": native_pass, "status": "complete",
                         **combine_epr_mode(prepared, record)})
    return EprResult(
        result_kind="adaptive_history", setup_name="AnalyticalCalculationOnly", rows=tuple(rows),
        provenance={"model_source_sha256": prepared.model_sha256, "analysis_source_sha256": prepared.source_sha256,
                    "requested_modes": (1, 2), "input_kind": "analytical_contract", "native_observed": False},
    )


def test_public_reanalysis_preserves_mode_pass_margin_identity_and_power_two_scaling():
    prepared, raw = _analytical_inputs()
    result = _analytical_history(prepared, raw)
    original = result.to_payload()
    updated = reanalyze_epr(result, surface_defaults={interface: {"film_thickness_m": 1.0} for interface in ENERGIES})
    assert [(row["mode"], row["native_pass"]) for row in updated.rows] == [(1, 1), (1, 2), (2, 1), (2, 2)]
    for old, new in zip(result.rows, updated.rows, strict=True):
        assert new["frequency_hz"] == 8.0 * new["mode"]
        assert new["normalization_energy_j"] == 32 * EPSILON_0
        assert [(item["group_id"], item["evaluation_kind"], item["margin_um"]) for item in new["surface_contributions"]] == [
            (item["group_id"], item["evaluation_kind"], item["margin_um"]) for item in old["surface_contributions"]]
        _assert_surface_arithmetic(new["surface_contributions"], thickness_scale=2)
    assert result.to_payload() == original
    with pytest.raises(ValueError, match="repeat a mode/pass identity"):
        EprResult(result_kind="adaptive_history", setup_name="AnalyticalCalculationOnly",
                  rows=(result.rows[0], result.rows[0]), provenance=result.provenance)


@pytest.mark.parametrize("field", ("electric", "normal", "area", "junction"))
def test_public_composition_rejects_wrong_canonical_units(field):
    prepared, raw = _analytical_inputs()
    group_id = next(iter(raw["surface_integrals_v2"]))
    if field == "electric":
        raw["electric_domain_integrals_v2_m"]["substrate"]["unit"] = "V^2"
    elif field == "normal":
        raw["surface_integrals_v2"][group_id]["normal"]["unit"] = "V^2*m"
    elif field == "area":
        raw["masked_areas_m2"][group_id]["unit"] = "um^2"
    else:
        raw["junction_integrals_v_m"]["junction"]["real"]["unit"] = "V"
    with pytest.raises(ValueError, match="canonical unit"):
        combine_surface_epr_mode(prepared, raw)


@pytest.mark.parametrize("missing", ("electric_domain_integrals_v2_m", "surface_integrals_v2", "junction_integrals_v_m"))
def test_public_composition_requires_raw_quantity_maps(missing):
    prepared, raw = _analytical_inputs()
    del raw[missing]
    with pytest.raises(TypeError, match="must be a string-keyed mapping"):
        combine_surface_epr_mode(prepared, raw)


def test_public_composition_requires_complete_domain_and_group_membership():
    prepared, raw = _analytical_inputs()
    raw["electric_domain_integrals_v2_m"].pop("vacuum")
    with pytest.raises(ValueError, match="cover every prepared solution domain"):
        combine_surface_epr_mode(prepared, raw)
    prepared, raw = _analytical_inputs()
    raw["surface_integrals_v2"].pop(next(iter(raw["surface_integrals_v2"])))
    with pytest.raises(ValueError, match="do not cover prepared groups"):
        combine_surface_epr_mode(prepared, raw)


def test_public_composition_rejects_zero_normalization_instead_of_returning_participation():
    prepared, raw = _analytical_inputs()
    for value in raw["electric_domain_integrals_v2_m"].values():
        value["value"] = 0.0
    raw["junction_integrals_v_m"]["junction"]["real"]["value"] = 0.0
    with pytest.raises(ValueError, match="normalization energy must be finite and > 0"):
        combine_surface_epr_mode(prepared, raw)
