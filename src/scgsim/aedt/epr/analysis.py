"""Combine raw field integrals into energies and participation ratios (EPR).

Owns: turning one mode's raw integrals (volume, surface, junction) into the
energies, normalization and participation ratios of a result row. All
quantities are peak phasors (docs/specs/aedt-runtime.qmd, "Surface estimates and
junction normalization").
Does not own: producing the integrals (`expressions.py`, `fields.py`), the thin-film
surface rows (`surface_epr.py`), the group rules (`selection.py`) or offline loss
reassessment (`reanalysis.py`).

Definitions used below, from docs/eigenmode-epr.qmd and
docs/specs/aedt-runtime.qmd:

- A participation ratio is p_i = U_i / U_norm.
- U_norm = U_E + sum of the junctions' capacitive energies U_C, where
  U_E = eps0/2 * sum over domains of eps_r * integral |E|^2 dV. Thin-film surface
  energies are numerators only; they are never added to U_norm.
- MA (metal-air), MS (metal-substrate) and SA (substrate-air) are the three
  interfaces whose thin-film energies are estimated from the field there.
- The junction convention is exp(+i*omega*t) with omega = 2*pi*Re(f).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, NamedTuple

from scgsim.aedt._data import (
    detached,
)
from scgsim.aedt.epr.selection import exact_mapping, finite, quantity
from scgsim.aedt.epr.surface_epr import (
    SurfaceIntegrals,
    native_surface_reference_rows,
    surface_contributions,
)
from scgsim.aedt.specs.epr import (
    EprAnalysisRequest,
)
from scgsim.aedt.specs.geometry import (
    PreparedPlanarGeometry,
)

_EPSILON_0_F_PER_M = 8.8541878128e-12

_MU_0_H_PER_M = 1.25663706212e-6


# --- Entry points -----------------------------------------------------------


def combine_epr_mode(
    prepared: PreparedPlanarGeometry,
    raw: Mapping[str, Any],
    *,
    request: EprAnalysisRequest | None = None,
) -> dict[str, Any]:
    """Combine one complete pass/mode raw-integral record without fallback.

    Steps: sum the domain energies, add the junction energies, form U_norm, then
    divide every numerator (junction and surface) by it.
    """

    _check_prepared_and_request(prepared, request)
    raw = exact_mapping(raw, "raw EPR mode")
    frequency_hz = quantity(raw.get("frequency_hz"), "frequency_hz", "Hz")
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be > 0")
    domain_integrals = _DomainIntegrals(
        electric=exact_mapping(
            raw.get("electric_domain_integrals_v2_m"),
            "electric_domain_integrals_v2_m",
        ),
        effective_volumes=exact_mapping(
            raw.get("effective_domain_volumes_m3"), "effective_domain_volumes_m3"
        ),
        magnetic=exact_mapping(
            raw.get("magnetic_domain_integrals_a2_m"),
            "magnetic_domain_integrals_a2_m",
        ),
        relative_permeability=exact_mapping(
            raw.get("relative_permeability"), "relative_permeability"
        ),
    )
    surface = SurfaceIntegrals(
        exact_mapping(raw.get("surface_integrals_v2"), "surface_integrals_v2"),
        exact_mapping(raw.get("masked_areas_m2"), "masked_areas_m2"),
    )
    junction = exact_mapping(
        raw.get("junction_integrals_v_m"), "junction_integrals_v_m"
    )

    _, materials, domains = epr_domain_context(prepared)
    domain_integrals.require_cover(domains)
    electric_energy_j, magnetic_energy_j, electric_rows = _domain_energies(
        materials, domains, domain_integrals
    )

    angular_frequency = 2.0 * math.pi * frequency_hz
    junction_by_id, junction_rows, capacitive_energy_j = _junction_capacitive_energies(
        prepared, junction, angular_frequency=angular_frequency
    )

    model_inductive_energy_j = sum(
        float(row["inductive_energy_j"]) for row in junction_rows
    )

    normalization_j = _epr_normalization_energy_j(
        electric_energy_j, capacitive_energy_j
    )
    selected_junctions = (
        set(junction_by_id)
        if request is None or request.junction_ids is None
        else set(request.junction_ids)
    )
    junction_rows = [
        {
            **row,
            "inductive_participation": row["inductive_energy_j"] / normalization_j,
            "capacitive_participation": row["capacitive_energy_j"] / normalization_j,
        }
        for row in junction_rows
        if row["junction_id"] in selected_junctions
    ]

    granularity, surface_rows = surface_contributions(
        prepared, raw, surface, request, normalization_j
    )
    native_surface_references = native_surface_reference_rows(prepared, raw, request)

    selected_domains = (
        set(domains)
        if request is None or request.bulk_domain_ids is None
        else set(request.bulk_domain_ids)
    )
    if not selected_domains <= set(domains):
        raise ValueError("EPR request selects an unknown bulk domain")
    result = {
        "schema_version": "scgsim.aedt.epr-mode-energy.v2",
        "surface_granularity": granularity or "per_binding.v1",
        "frequency_hz": frequency_hz,
        "electric_energy_j": electric_energy_j,
        "magnetic_energy_j": magnetic_energy_j,
        "junction_inductive_energy_j": model_inductive_energy_j,
        "magnetic_energy_balance_j": magnetic_energy_j + model_inductive_energy_j,
        "junction_capacitive_energy_j": capacitive_energy_j,
        "normalization_energy_j": normalization_j,
        "electric_domains": [
            row for row in electric_rows if row["domain_id"] in selected_domains
        ],
        "surface_contributions": surface_rows,
        "junctions": junction_rows,
    }
    if native_surface_references is not None:
        result["native_surface_references"] = native_surface_references
    return result


def combine_surface_epr_mode(
    prepared: PreparedPlanarGeometry,
    raw: Mapping[str, Any],
    *,
    request: EprAnalysisRequest | None = None,
) -> dict[str, Any]:
    """Combine surface EPR from one raw mode using the complete EPR denominator.

    Same U_norm as `combine_epr_mode`, but only the normalization and surface
    rows are returned; there is no frequency, so junction inductive energy is not
    formed.
    """

    _check_prepared_and_request(prepared, request)
    raw = exact_mapping(raw, "raw EPR mode")
    electric = exact_mapping(
        raw.get("electric_domain_integrals_v2_m"),
        "electric_domain_integrals_v2_m",
    )
    surface = SurfaceIntegrals(
        exact_mapping(raw.get("surface_integrals_v2"), "surface_integrals_v2"),
        exact_mapping(raw.get("masked_areas_m2"), "masked_areas_m2"),
    )
    junction = exact_mapping(
        raw.get("junction_integrals_v_m"), "junction_integrals_v_m"
    )

    _, materials, domains = epr_domain_context(prepared)
    if set(electric) != set(domains):
        raise ValueError(
            "electric domain integrals must cover every prepared solution domain"
        )
    electric_energy_j = sum(
        _electric_domain_energy(materials, domain_id, domain, electric)[2]
        for domain_id, domain in domains.items()
    )
    _, _, capacitive_energy_j = _junction_capacitive_energies(prepared, junction)
    normalization_j = _epr_normalization_energy_j(
        electric_energy_j, capacitive_energy_j
    )

    if (
        request is not None
        and request.bulk_domain_ids is not None
        and not set(request.bulk_domain_ids) <= set(domains)
    ):
        raise ValueError("EPR request selects an unknown bulk domain")
    _, surface_rows = surface_contributions(
        prepared, raw, surface, request, normalization_j
    )
    result = {
        "normalization_energy_j": normalization_j,
        "surface_contributions": surface_rows,
    }
    native_surface_references = native_surface_reference_rows(prepared, raw, request)
    if native_surface_references is not None:
        result["native_surface_references"] = native_surface_references
    return result


def _check_prepared_and_request(
    prepared: PreparedPlanarGeometry, request: EprAnalysisRequest | None
) -> None:
    if not isinstance(prepared, PreparedPlanarGeometry):
        raise TypeError("prepared must be PreparedPlanarGeometry")
    if request is not None and not isinstance(request, EprAnalysisRequest):
        raise TypeError("request must be EprAnalysisRequest or None")


# --- Domains (bulk electric and magnetic energy) -----------------------------


class _DomainIntegrals(NamedTuple):
    """The four per-domain raw integral records of one mode, keyed by domain id."""

    electric: Mapping[str, Any]
    effective_volumes: Mapping[str, Any]
    magnetic: Mapping[str, Any]
    relative_permeability: Mapping[str, Any]

    def require_cover(self, domains: Mapping[str, Any]) -> None:
        """Every prepared solution domain needs all four integrals."""

        if (
            set(self.electric) != set(domains)
            or set(self.magnetic) != set(domains)
            or set(self.relative_permeability) != set(domains)
        ):
            raise ValueError(
                "electric and magnetic domain integrals must cover every prepared solution domain"
            )
        if set(self.effective_volumes) != set(domains):
            raise ValueError(
                "effective volume integrals must cover every prepared solution domain"
            )


def epr_domain_context(
    prepared: PreparedPlanarGeometry,
) -> tuple[dict[str, Any], Mapping[str, Any], dict[str, Any]]:
    """Return (source, materials, domains) for the energy sums.

    When the model uses one native "Region" for the logical vacuum, the logical
    vacuum domains are replaced by that single Region so they are not counted
    twice; the mapping is checked to cover exactly the logical vacuum.
    """

    source = detached(prepared.source)
    materials = exact_mapping(source["materials"], "prepared materials")
    domains = {item["semantic_id"]: item for item in source["solution_regions"]}
    native_region = source.get("native_region")
    if native_region is not None:
        native_region = exact_mapping(native_region, "native Region mapping")
        auto_vacuum_ids = {
            domain_id
            for domain_id, domain in domains.items()
            if domain["material_kind"] == "vacuum"
            and exact_mapping(
                domain.get("metadata", {}),
                f"solution domain {domain_id!r} metadata",
            ).get("is_auto_vacuum_region")
            is True
        }
        if (
            native_region.get("method") != "single_region_absolute_offset.v1"
            or native_region.get("name") != "Region"
            or set(native_region.get("logical_vacuum_ids", ())) != auto_vacuum_ids
            or materials.get(native_region.get("material_id"), {}).get("kind")
            != "vacuum"
            or "Region" in domains
        ):
            raise ValueError("native Region mapping does not cover logical vacuum")
        domains = {
            domain_id: domain
            for domain_id, domain in domains.items()
            if domain_id not in auto_vacuum_ids
        }
        domains["Region"] = {
            "semantic_id": "Region",
            "material_id": native_region["material_id"],
            "material_kind": "vacuum",
        }
    return source, materials, domains


def _domain_energies(
    materials: Mapping[str, Any],
    domains: Mapping[str, Any],
    integrals: _DomainIntegrals,
) -> tuple[float, float, list[dict[str, Any]]]:
    """Sum electric and magnetic energy over the domains, keeping each domain's row.

    U_E = eps0 * eps_r / 2 * integral |E|^2 dV and
    U_H = mu0 * mu_r / 2 * integral |H|^2 dV, per domain.
    """

    electric_energy_j = 0.0
    magnetic_energy_j = 0.0
    electric_rows: list[dict[str, Any]] = []
    for domain_id, domain in domains.items():
        epsilon_r, integral, energy = _electric_domain_energy(
            materials, domain_id, domain, integrals.electric
        )
        magnetic_integral = quantity(
            integrals.magnetic[domain_id], f"magnetic integral {domain_id!r}", "A^2*m"
        )
        permeability_r = quantity(
            integrals.relative_permeability[domain_id],
            f"relative permeability {domain_id!r}",
            "1",
        )
        if permeability_r <= 0.0:
            raise ValueError("relative permeability must be > 0")
        effective_volume = quantity(
            integrals.effective_volumes[domain_id],
            f"effective volume {domain_id!r}",
            "m^3",
        )
        magnetic_energy = _MU_0_H_PER_M * permeability_r * magnetic_integral / 2.0
        electric_energy_j += energy
        magnetic_energy_j += magnetic_energy
        electric_rows.append(
            {
                "domain_id": domain_id,
                "material_id": domain["material_id"],
                "relative_permittivity": epsilon_r,
                "native_relative_permeability": permeability_r,
                "effective_volume_m3": effective_volume,
                "integral_v2_m": integral,
                "energy_j": energy,
                "magnetic_h_conj_dot_h_integral_a2_m": magnetic_integral,
                "magnetic_energy_j": magnetic_energy,
            }
        )
    return electric_energy_j, magnetic_energy_j, electric_rows


def _electric_domain_energy(
    materials: Mapping[str, Any],
    domain_id: str,
    domain: Mapping[str, Any],
    electric: Mapping[str, Any],
) -> tuple[float, float, float]:
    """Return (eps_r, integral |E|^2 dV, U_E) for one domain."""

    material = exact_mapping(
        materials.get(domain["material_id"]), f"material {domain['material_id']!r}"
    )
    if "permittivity" not in material:
        raise ValueError(
            f"material {domain['material_id']!r} lacks relative permittivity"
        )
    epsilon_r = finite(material["permittivity"], "relative permittivity")
    if epsilon_r <= 0.0:
        raise ValueError("relative permittivity must be > 0")
    integral = quantity(
        electric[domain_id], f"electric integral {domain_id!r}", "V^2*m"
    )
    energy = _EPSILON_0_F_PER_M * epsilon_r * integral / 2.0
    return epsilon_r, integral, energy


def electric_domain_energy_coefficient(relative_permittivity: float) -> float:
    """eps0 * eps_r / 2: multiplies integral |E|^2 dV to give a domain's U_E."""

    return _EPSILON_0_F_PER_M * relative_permittivity / 2.0


# --- Junctions and normalization ---------------------------------------------


def junction_capacitive_energy_coefficient(
    capacitance_f: float, width_um: float
) -> float:
    """C / (2 w^2): multiplies |integral E.l dA|^2 to give a junction's U_C.

    The sheet voltage is V = (1/w) * integral E.l dA, so U_C = C |V|^2 / 2.
    """

    width_m = width_um * 1e-6
    return capacitance_f / (2.0 * width_m**2)


def _junction_capacitive_energies(
    prepared: PreparedPlanarGeometry,
    junction: Mapping[str, Any],
    *,
    angular_frequency: float | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], float]:
    """Voltage, capacitive (and, with a frequency, inductive) energy per junction.

    Returns the junctions by id, one row each, and the total capacitive energy.
    With omega, I_L = V / (i omega L), so U_L = |V|^2 / (2 omega^2 L).
    """

    junction_by_id = {item.junction_id: item for item in prepared.junctions}
    if set(junction) != set(junction_by_id):
        raise ValueError("junction integrals must cover every prepared junction")
    rows: list[dict[str, Any]] = []
    capacitive_energy_j = 0.0
    for junction_id, spec in junction_by_id.items():
        value = exact_mapping(junction[junction_id], f"junction {junction_id!r}")
        real = quantity(
            value.get("real"), "junction real integral", "V*m", nonnegative=False
        )
        imag = quantity(
            value.get("imag"), "junction imaginary integral", "V*m", nonnegative=False
        )
        voltage = complex(real, imag) / (spec.width_um * 1e-6)
        voltage_squared = abs(voltage) ** 2
        row = {
            "junction_id": junction_id,
            "source_polygon_id": spec.source_polygon_id,
            "terminal_a_net": spec.terminal_a_net,
            "terminal_b_net": spec.terminal_b_net,
            "direction_xy": list(spec.direction_xy),
            "voltage_real_v": voltage.real,
            "voltage_imag_v": voltage.imag,
        }
        if angular_frequency is not None:
            inductive = voltage_squared / (
                2.0 * angular_frequency**2 * spec.inductance_h
            )
        capacitive = spec.capacitance_f * voltage_squared / 2.0
        capacitive_energy_j += capacitive
        if angular_frequency is not None:
            current = voltage / complex(0.0, angular_frequency * spec.inductance_h)
            row.update(
                {
                    "current_real_a": current.real,
                    "current_imag_a": current.imag,
                    "inductive_energy_j": inductive,
                }
            )
        row["capacitive_energy_j"] = capacitive
        rows.append(row)
    return junction_by_id, rows, capacitive_energy_j


def _epr_normalization_energy_j(
    electric_energy_j: float,
    capacitive_energy_j: float,
) -> float:
    """U_norm = U_E + sum of U_C; it must be finite and positive."""

    normalization_j = electric_energy_j + capacitive_energy_j
    if not math.isfinite(normalization_j) or normalization_j <= 0.0:
        raise ValueError("EPR normalization energy must be finite and > 0")
    return normalization_j
