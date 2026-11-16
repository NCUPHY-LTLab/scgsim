"""Independent analytical EPR arithmetic of an SGB model's EPR source; no layout, fields, or native result.

The source is written by hand: two dielectric domains, one junction and, per interface type, two
Surface-EPR rows (band 0 and band 1).  Each type reports a baseline over both rows and a 1 um inset
margin over band 1 only.  Synthetic mode/pass labels test result identity, not native observations.

Hand calculation, using the contract's SI vacuum permittivity below:
  thickness=1/2 m, film epsilon=2, substrate epsilon=4; each row N=4 V², T=1 V², so a baseline sums
  N=8 V², T=2 V².
  MA: epsilon0*N/8 = epsilon0.
  MS: 2*epsilon0*N = 16*epsilon0.
  SA: epsilon0*N/8 + epsilon0*T/2 = 2*epsilon0.
  Bulk: epsilon0*4*8/2 = 16*epsilon0 J.
  Junction: C=32*epsilon0 F, |V|=1 V (1 V*m over a 1 m width) -> 16*epsilon0 J.
Thus the denominator is 32*epsilon0 J and p=(1/32,1/2,1/16).  The margin entry holds one row, so it
is half the baseline.  All scale factors are powers of two, so exact binary64 comparisons need no
test-chosen numerical tolerance.
"""

from __future__ import annotations

from copy import deepcopy

import pytest

from scgsim.aedt import EprResult, PlanarJunction, reanalyze_epr
from scgsim.aedt.epr.integrals import SURFACE_GRANULARITY
from scgsim.aedt.epr.sgb_energy import combine_sgb_epr_mode
from scgsim.aedt.epr.sgb_source import (
    EprDomain,
    EprSurfaceRow,
    JunctionSheet,
    SgbEprSource,
    SurfaceEntry,
    SurfaceFilm,
)
from scgsim.aedt.epr.surface import surface_energy_coefficients

EPSILON_0 = 8.8541878128e-12  # SI constant, independently written, not imported from the formula.
ENERGIES = {"MA": EPSILON_0, "MS": 16 * EPSILON_0, "SA": 2 * EPSILON_0}
PARTICIPATIONS = {"MA": 1 / 32, "MS": 1 / 2, "SA": 1 / 16}
EVALUATIONS = (("unmasked_baseline", 0.0), ("requested_margin", 1.0))


def _quantity(value, unit):
    return {"value": value, "unit": unit}


def _source() -> SgbEprSource:
    films = {
        kind: SurfaceFilm(kind, 0.5, 2.0, 1e-3, "analytical", None) for kind in ENERGIES
    }
    rows = {}
    entries = []
    for kind in ENERGIES:
        substrate = None if kind == "MA" else 4.0
        names = (f"{kind}_row_0", f"{kind}_row_1")
        for band, row in enumerate(names):
            rows[row] = EprSurfaceRow(
                row=row,
                group=f"{kind}::owner__band_{band}",
                interface_type=kind,
                band=band,
                family=f"{kind}::owner",
                banded_surface=f"{kind}::owner",
                entities=("owner",),
                face_kind="top",
                substrate_relative_permittivity=substrate,
                selection_name=f"epr_{row}",
            )
        entries.append(
            SurfaceEntry(
                f"{kind}::owner",
                kind,
                "unmasked_baseline",
                0.0,
                names,
                ("owner",),
                substrate,
            )
        )
        entries.append(
            SurfaceEntry(
                f"{kind}::owner__inset_0",
                kind,
                "requested_margin",
                1.0,
                names[1:],
                ("owner",),
                substrate,
            )
        )
    junction = PlanarJunction(
        "junction", "sheet_polygon", "A", "B", (1, 0), 2.0, 10e-9, 32 * EPSILON_0
    )
    return SgbEprSource(
        domains={
            "substrate": EprDomain("substrate", "Si", 4.0),
            "Region": EprDomain("Region", "air", 1.0),
        },
        films=films,
        rows=rows,
        entries=tuple(entries),
        junctions={
            "junction": JunctionSheet(
                junction, "PORT::junction", ("face",), (1.0, 0.0, 0.0), 1e6
            )
        },
    )


def _raw(source: SgbEprSource) -> dict:
    domains = tuple(source.domains)
    return {
        "frequency_hz": _quantity(8.0, "Hz"),
        "electric_domain_integrals_v2_m": {
            "substrate": _quantity(8.0, "V^2*m"),
            "Region": _quantity(0.0, "V^2*m"),
        },
        "magnetic_domain_integrals_a2_m": {
            key: _quantity(0.0, "A^2*m") for key in domains
        },
        "relative_permeability": {key: _quantity(1.0, "1") for key in domains},
        "effective_domain_volumes_m3": {key: _quantity(1.0, "m^3") for key in domains},
        "junction_integrals_v_m": {
            "junction": {"real": _quantity(1.0, "V*m"), "imag": _quantity(0.0, "V*m")}
        },
        "surface_granularity": SURFACE_GRANULARITY,
        "surface_integrals_v2": {
            row: {"normal": _quantity(4.0, "V^2"), "tangential": _quantity(1.0, "V^2")}
            for row in source.rows
        },
        "masked_areas_m2": {row: _quantity(0.5, "m^2") for row in source.rows},
    }


@pytest.mark.parametrize(
    "interface,normal,tangential",
    (
        ("MA", EPSILON_0 / 8, 0.0),
        ("MS", 2 * EPSILON_0, 0.0),
        ("SA", EPSILON_0 / 8, EPSILON_0 / 2),
    ),
)
def test_peak_phasor_coefficients_match_independent_hand_calculation(
    interface, normal, tangential
):
    """Match peak-phasor energy coefficients against independent scalar calculations."""
    assert surface_energy_coefficients(interface, 0.5, 2.0, 4.0) == (
        normal,
        tangential,
    )


def _assert_surface_arithmetic(rows, *, thickness_scale=1):
    # Three interfaces x two reported evaluations, not one row per Surface-EPR row.
    assert len(rows) == 6
    assert {
        (row["interface_kind"], row["evaluation_kind"], row["margin_um"])
        for row in rows
    } == {
        (interface, kind, margin)
        for interface in ENERGIES
        for kind, margin in EVALUATIONS
    }
    for row in rows:
        scale = 0.5 if row["margin_um"] else 1
        assert (
            row["energy_j"] == ENERGIES[row["interface_kind"]] * scale * thickness_scale
        )
        assert (
            row["participation"]
            == PARTICIPATIONS[row["interface_kind"]] * scale * thickness_scale
        )
        assert [member["band"] for member in row["members"]] == (
            [1] if row["margin_um"] else [0, 1]
        )


def test_the_composition_uses_the_full_denominator_and_counts_each_row_once():
    """Bulk plus junction capacitive energy normalize; each entry sums its own rows."""
    source = _source()
    full = combine_sgb_epr_mode(source, _raw(source))
    assert full["electric_energy_j"] == 16 * EPSILON_0
    assert full["junction_capacitive_energy_j"] == 16 * EPSILON_0
    assert full["normalization_energy_j"] == 32 * EPSILON_0
    _assert_surface_arithmetic(full["surface_contributions"])


def _analytical_history(source):
    rows = []
    for mode in (1, 2):
        for native_pass in (1, 2):
            record = deepcopy(_raw(source))
            record["frequency_hz"] = _quantity(8.0 * mode, "Hz")
            rows.append(
                {
                    "mode": mode,
                    "native_pass": native_pass,
                    "status": "complete",
                    **combine_sgb_epr_mode(source, record),
                }
            )
    return EprResult(
        result_kind="adaptive_history",
        setup_name="AnalyticalCalculationOnly",
        rows=tuple(rows),
        provenance={
            "model_source_sha256": "0" * 64,
            "analysis_source_sha256": "1" * 64,
            "requested_modes": (1, 2),
            "input_kind": "analytical_contract",
            "native_observed": False,
        },
    )


def test_reanalysis_preserves_mode_pass_margin_identity_and_power_two_scaling():
    """Preserve mode/pass identity and quadratic field scaling during reanalysis."""
    result = _analytical_history(_source())
    original = result.to_payload()
    updated = reanalyze_epr(
        result,
        surface_defaults={
            interface: {"film_thickness_m": 1.0} for interface in ENERGIES
        },
    )
    assert [(row["mode"], row["native_pass"]) for row in updated.rows] == [
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
    ]
    for old, new in zip(result.rows, updated.rows, strict=True):
        assert new["frequency_hz"] == 8.0 * new["mode"]
        assert new["normalization_energy_j"] == 32 * EPSILON_0
        assert [
            (item["group_id"], item["evaluation_kind"], item["margin_um"])
            for item in new["surface_contributions"]
        ] == [
            (item["group_id"], item["evaluation_kind"], item["margin_um"])
            for item in old["surface_contributions"]
        ]
        _assert_surface_arithmetic(new["surface_contributions"], thickness_scale=2)
    assert result.to_payload() == original
    with pytest.raises(ValueError, match="repeat a mode/pass identity"):
        EprResult(
            result_kind="adaptive_history",
            setup_name="AnalyticalCalculationOnly",
            rows=(result.rows[0], result.rows[0]),
            provenance=result.provenance,
        )


def test_epr_result_preserves_mode_error_precedence_over_raw_integral_shape():
    """Report invalid mode identity before malformed raw-integral data."""
    result = _analytical_history(_source())
    with pytest.raises(
        ValueError, match="EPR result row mode must be a positive integer"
    ):
        EprResult(
            result_kind="adaptive_history",
            setup_name="AnalyticalCalculationOnly",
            rows=({"mode": 0, "status": "complete", "raw_integrals": []},),
            provenance=result.provenance,
        )


def test_reanalysis_loss_tangent_only_preserves_stored_energy_and_participation():
    """Change loss tangent without changing stored energy or participation."""
    result = _analytical_history(_source())
    loss_tangent = 0.012
    updated = reanalyze_epr(
        result,
        surface_defaults={
            interface: {"loss_tangent": loss_tangent} for interface in ENERGIES
        },
    )
    for original_row, updated_row in zip(result.rows, updated.rows, strict=True):
        for original, recalculated in zip(
            original_row["surface_contributions"],
            updated_row["surface_contributions"],
            strict=True,
        ):
            assert recalculated["energy_j"] == original["energy_j"]
            assert recalculated["participation"] == original["participation"]
            assert recalculated["inverse_q"] == (
                original["participation"] * loss_tangent
            )


@pytest.mark.parametrize("field", ("electric", "normal", "area", "junction"))
def test_the_composition_rejects_wrong_canonical_units(field):
    """Reject integral records with noncanonical physical units."""
    source = _source()
    raw = _raw(source)
    row = next(iter(raw["surface_integrals_v2"]))
    if field == "electric":
        raw["electric_domain_integrals_v2_m"]["substrate"]["unit"] = "V^2"
    elif field == "normal":
        raw["surface_integrals_v2"][row]["normal"]["unit"] = "V^2*m"
    elif field == "area":
        raw["masked_areas_m2"][row]["unit"] = "um^2"
    else:
        raw["junction_integrals_v_m"]["junction"]["real"]["unit"] = "V"
    with pytest.raises(ValueError, match="canonical unit"):
        combine_sgb_epr_mode(source, raw)


@pytest.mark.parametrize(
    "missing",
    (
        "electric_domain_integrals_v2_m",
        "surface_integrals_v2",
        "junction_integrals_v_m",
    ),
)
def test_the_composition_requires_raw_quantity_maps(missing):
    """Require the declared raw quantity maps."""
    source = _source()
    raw = _raw(source)
    del raw[missing]
    with pytest.raises(TypeError, match="must be a string-keyed mapping"):
        combine_sgb_epr_mode(source, raw)


def test_the_composition_requires_every_domain_and_row():
    """Require every domain and every Surface-EPR row of the source."""
    source = _source()
    raw = _raw(source)
    raw["electric_domain_integrals_v2_m"].pop("Region")
    with pytest.raises(ValueError, match="must cover the domains"):
        combine_sgb_epr_mode(source, raw)
    raw = _raw(source)
    raw["surface_integrals_v2"].pop(next(iter(raw["surface_integrals_v2"])))
    with pytest.raises(ValueError, match="must cover the Surface-EPR rows"):
        combine_sgb_epr_mode(source, raw)


def test_the_composition_rejects_zero_normalization_instead_of_returning_participation():
    """Reject zero normalization before producing a participation result."""
    source = _source()
    raw = _raw(source)
    for value in raw["electric_domain_integrals_v2_m"].values():
        value["value"] = 0.0
    raw["junction_integrals_v_m"]["junction"]["real"]["value"] = 0.0
    with pytest.raises(ValueError, match="normalization energy must be finite and > 0"):
        combine_sgb_epr_mode(source, raw)


def test_the_composition_refuses_raw_of_another_surface_granularity():
    """Refuse a raw mode that is not of the SGB model's Surface-EPR rows."""
    source = _source()
    raw = _raw(source)
    raw["surface_granularity"] = "owner_interface_margin.v1"
    with pytest.raises(ValueError, match="surface granularity"):
        combine_sgb_epr_mode(source, raw)


def test_reanalysis_refuses_surface_row_without_stored_integrals():
    """Refuse to rescale a surface row whose stored field integrals are missing."""
    result = _analytical_history(_source())
    payload = result.to_payload()
    for row in payload["rows"]:
        for item in row["surface_contributions"]:
            del item["normal_integral_v2"]
    stripped = EprResult.from_payload(payload)
    with pytest.raises(ValueError, match="needed to recompute it"):
        reanalyze_epr(
            stripped,
            surface_defaults={
                interface: {"film_thickness_m": 1.0} for interface in ENERGIES
            },
        )
