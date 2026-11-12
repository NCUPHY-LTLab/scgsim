"""EPR energy composition and offline assumption reanalysis."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from scgsim.aedt.data import (
    detached,
)
from scgsim.aedt.epr.reanalysis import reanalyze_epr
from scgsim.aedt.epr.selection import exact_mapping, finite_value
from scgsim.aedt.epr.surface import (
    VACUUM_PERMITTIVITY_F_PER_M,
    SurfaceContributionInputs,
    native_surface_reference_rows,
    quantity,
    surface_contributions,
)
from scgsim.aedt.specs.epr import EprAnalysisRequest
from scgsim.aedt.specs.geometry import PreparedPlanarGeometry

__all__ = ["combine_epr_mode", "combine_surface_epr_mode", "reanalyze_epr"]

_MU_0_H_PER_M = 1.25663706212e-6


@dataclass(frozen=True)
class CompleteModeEnergyInputs:
    """Validated calculation outputs assembled into one canonical energy row."""

    prepared: PreparedPlanarGeometry
    granularity: str
    frequency_hz: float
    electric_energy_j: float
    magnetic_energy_j: float
    model_inductive_energy_j: float
    capacitive_energy_j: float
    normalization_j: float
    electric_rows: list[dict[str, Any]]
    selected_domains: set[str]
    surface_rows: list[dict[str, Any]]
    junction_rows: list[dict[str, Any]]


def epr_domain_context(
    prepared: PreparedPlanarGeometry,
) -> tuple[dict[str, Any], Mapping[str, Any], dict[str, Any]]:
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
            or materials.get(cast(str, native_region.get("material_id")), {}).get(
                "kind"
            )
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


def _electric_domain_energy(
    materials: Mapping[str, Any],
    domain_id: str,
    domain: Mapping[str, Any],
    electric: Mapping[str, Any],
) -> tuple[float, float, float]:
    material = exact_mapping(
        materials.get(domain["material_id"]), f"material {domain['material_id']!r}"
    )
    if "permittivity" not in material:
        raise ValueError(
            f"material {domain['material_id']!r} lacks relative permittivity"
        )
    epsilon_r = finite_value(material["permittivity"], "relative permittivity")
    if epsilon_r <= 0.0:
        raise ValueError("relative permittivity must be > 0")
    integral = quantity(
        electric[domain_id], f"electric integral {domain_id!r}", "V^2*m"
    )
    energy = VACUUM_PERMITTIVITY_F_PER_M * epsilon_r * integral / 2.0
    return epsilon_r, integral, energy


def electric_domain_energy_coefficient(relative_permittivity: float) -> float:
    return VACUUM_PERMITTIVITY_F_PER_M * relative_permittivity / 2.0


def junction_capacitive_energy_coefficient(
    capacitance_f: float, width_um: float
) -> float:
    width_m = width_um * 1e-6
    return capacitance_f / (2.0 * width_m**2)


def _junction_capacitive_energies(
    prepared: PreparedPlanarGeometry,
    junction: Mapping[str, Any],
    *,
    angular_frequency: float | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], float]:
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
        inductive: float | None = None
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
                    "inductive_energy_j": cast(float, inductive),
                }
            )
        row["capacitive_energy_j"] = capacitive
        rows.append(row)
    return junction_by_id, rows, capacitive_energy_j


def _epr_normalization_energy_j(
    electric_energy_j: float,
    capacitive_energy_j: float,
) -> float:
    normalization_j = electric_energy_j + capacitive_energy_j
    if not math.isfinite(normalization_j) or normalization_j <= 0.0:
        raise ValueError("EPR normalization energy must be finite and > 0")
    return normalization_j


def combine_epr_mode(
    prepared: PreparedPlanarGeometry,
    raw: Mapping[str, Any],
    *,
    request: EprAnalysisRequest | None = None,
) -> dict[str, Any]:
    """Combine one complete pass/mode raw-integral record without fallback."""

    if not isinstance(prepared, PreparedPlanarGeometry):
        raise TypeError("prepared must be PreparedPlanarGeometry")
    if request is not None and not isinstance(request, EprAnalysisRequest):
        raise TypeError("request must be EprAnalysisRequest or None")
    raw = exact_mapping(raw, "raw EPR mode")
    frequency_hz = quantity(raw.get("frequency_hz"), "frequency_hz", "Hz")
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be > 0")
    electric = exact_mapping(
        raw.get("electric_domain_integrals_v2_m"),
        "electric_domain_integrals_v2_m",
    )
    effective_volumes = exact_mapping(
        raw.get("effective_domain_volumes_m3"), "effective_domain_volumes_m3"
    )
    magnetic = exact_mapping(
        raw.get("magnetic_domain_integrals_a2_m"),
        "magnetic_domain_integrals_a2_m",
    )
    relative_permeability = exact_mapping(
        raw.get("relative_permeability"), "relative_permeability"
    )
    surface = exact_mapping(raw.get("surface_integrals_v2"), "surface_integrals_v2")
    masked_areas = exact_mapping(raw.get("masked_areas_m2"), "masked_areas_m2")
    junction = exact_mapping(
        raw.get("junction_integrals_v_m"), "junction_integrals_v_m"
    )

    _, materials, domains = epr_domain_context(prepared)
    if (
        set(electric) != set(domains)
        or set(magnetic) != set(domains)
        or set(relative_permeability) != set(domains)
    ):
        raise ValueError(
            "electric and magnetic domain integrals must cover every prepared solution domain"
        )
    if set(effective_volumes) != set(domains):
        raise ValueError(
            "effective volume integrals must cover every prepared solution domain"
        )
    electric_energy_j = 0.0
    magnetic_energy_j = 0.0
    electric_rows: list[dict[str, Any]] = []
    for domain_id, domain in domains.items():
        epsilon_r, integral, energy = _electric_domain_energy(
            materials, domain_id, domain, electric
        )
        magnetic_integral = quantity(
            magnetic[domain_id], f"magnetic integral {domain_id!r}", "A^2*m"
        )
        permeability_r = quantity(
            relative_permeability[domain_id],
            f"relative permeability {domain_id!r}",
            "1",
        )
        if permeability_r <= 0.0:
            raise ValueError("relative permeability must be > 0")
        effective_volume = quantity(
            effective_volumes[domain_id],
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
        SurfaceContributionInputs(
            prepared, raw, surface, masked_areas, request, normalization_j
        )
    )
    native_surface_references = native_surface_reference_rows(prepared, raw, request)

    selected_domains = (
        set(domains)
        if request is None or request.bulk_domain_ids is None
        else set(request.bulk_domain_ids)
    )
    if not selected_domains <= set(domains):
        raise ValueError("EPR request selects an unknown bulk domain")
    result = _complete_mode_energy_record(
        CompleteModeEnergyInputs(
            prepared,
            granularity,
            frequency_hz,
            electric_energy_j,
            magnetic_energy_j,
            model_inductive_energy_j,
            capacitive_energy_j,
            normalization_j,
            electric_rows,
            selected_domains,
            surface_rows,
            junction_rows,
        )
    )
    if native_surface_references is not None:
        result["native_surface_references"] = native_surface_references
    return result


def _complete_mode_energy_record(
    inputs: CompleteModeEnergyInputs,
) -> dict[str, Any]:
    """Assemble the canonical complete mode-energy row after all validation."""
    granularity = inputs.granularity
    frequency_hz = inputs.frequency_hz
    electric_energy_j = inputs.electric_energy_j
    magnetic_energy_j = inputs.magnetic_energy_j
    model_inductive_energy_j = inputs.model_inductive_energy_j
    capacitive_energy_j = inputs.capacitive_energy_j
    normalization_j = inputs.normalization_j
    electric_rows = inputs.electric_rows
    selected_domains = inputs.selected_domains
    surface_rows = inputs.surface_rows
    junction_rows = inputs.junction_rows
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
    return result


def combine_surface_epr_mode(
    prepared: PreparedPlanarGeometry,
    raw: Mapping[str, Any],
    *,
    request: EprAnalysisRequest | None = None,
) -> dict[str, Any]:
    """Combine surface EPR from one raw mode using the complete EPR denominator."""

    if not isinstance(prepared, PreparedPlanarGeometry):
        raise TypeError("prepared must be PreparedPlanarGeometry")
    if request is not None and not isinstance(request, EprAnalysisRequest):
        raise TypeError("request must be EprAnalysisRequest or None")
    raw = exact_mapping(raw, "raw EPR mode")
    electric = exact_mapping(
        raw.get("electric_domain_integrals_v2_m"),
        "electric_domain_integrals_v2_m",
    )
    surface = exact_mapping(raw.get("surface_integrals_v2"), "surface_integrals_v2")
    masked_areas = exact_mapping(raw.get("masked_areas_m2"), "masked_areas_m2")
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
        SurfaceContributionInputs(
            prepared, raw, surface, masked_areas, request, normalization_j
        )
    )
    result = {
        "normalization_energy_j": normalization_j,
        "surface_contributions": surface_rows,
    }
    native_surface_references = native_surface_reference_rows(prepared, raw, request)
    if native_surface_references is not None:
        result["native_surface_references"] = native_surface_references
    return result
