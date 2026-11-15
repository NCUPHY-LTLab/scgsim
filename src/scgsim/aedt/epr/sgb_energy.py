"""Combine the field integrals of one mode of an imported SGB model into its energy participation (issue #67).

The integrals of one mode (``raw``) are keyed by what :mod:`scgsim.aedt.epr.sgb_source` names: the electric
and magnetic integrals by domain (AEDT object name), the normal and tangential surface integrals and the area
by sidecar Surface-EPR row, and the junction voltage area integrals by junction.  :func:`combine_sgb_epr_mode`
then, in order:

1. sums the electric energy of every domain, ``eps0 * eps_r * integral / 2``, and the magnetic energy;
2. reads each junction's voltage (its area integral divided by the port sheet's width) and from it the
   junction's inductive and capacitive energy;
3. normalizes by the electric energy plus every junction's capacitive energy, the peak energy of the mode;
4. gives each Surface-EPR entry the sum of its rows' film energies, its participation and ``1/Q``.

Every model junction enters the normalization; ``junction_ids`` and ``bulk_domain_ids`` of the request only
select the rows reported.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from scgsim.aedt.epr.integrals import SURFACE_GRANULARITY
from scgsim.aedt.epr.sgb_source import SgbEprSource, SurfaceEntry
from scgsim.aedt.epr.surface import (
    VACUUM_PERMITTIVITY_F_PER_M,
    exact_mapping,
    quantity,
    surface_energy_j,
)
from scgsim.aedt.specs.epr import EprAnalysisRequest
from scgsim.semantics.epr import film_assumptions, inverse_quality_factor

MODE_ENERGY_SCHEMA = "scgsim.aedt.epr-mode-energy.v2"
_MU_0_H_PER_M = 1.25663706212e-6


def combine_sgb_epr_mode(
    source: SgbEprSource,
    raw: Mapping[str, Any],
    *,
    request: EprAnalysisRequest | None = None,
) -> dict[str, Any]:
    """The energy record of one complete mode (see the module)."""
    raw = exact_mapping(raw, "raw EPR mode")
    frequency_hz = quantity(raw.get("frequency_hz"), "frequency_hz", "Hz")
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be > 0")
    domains = domain_rows(source, raw)
    electric_energy_j = sum(row["energy_j"] for row in domains)
    magnetic_energy_j = sum(row["magnetic_energy_j"] for row in domains)
    junctions = junction_rows(
        source, raw, angular_frequency=2.0 * math.pi * frequency_hz
    )
    capacitive_energy_j = sum(row["capacitive_energy_j"] for row in junctions)
    inductive_energy_j = sum(row["inductive_energy_j"] for row in junctions)
    normalization_j = normalization_energy_j(electric_energy_j, capacitive_energy_j)
    selected_junctions = _selected(request, "junction_ids", source.junctions)
    selected_domains = _selected(request, "bulk_domain_ids", source.domains)
    return {
        "schema_version": MODE_ENERGY_SCHEMA,
        "surface_granularity": SURFACE_GRANULARITY,
        "frequency_hz": frequency_hz,
        "electric_energy_j": electric_energy_j,
        "magnetic_energy_j": magnetic_energy_j,
        "junction_inductive_energy_j": inductive_energy_j,
        "magnetic_energy_balance_j": magnetic_energy_j + inductive_energy_j,
        "junction_capacitive_energy_j": capacitive_energy_j,
        "normalization_energy_j": normalization_j,
        "electric_domains": [
            row for row in domains if row["domain_id"] in selected_domains
        ],
        "surface_contributions": surface_contributions(source, raw, normalization_j),
        "junctions": [
            {
                **row,
                "inductive_participation": row["inductive_energy_j"] / normalization_j,
                "capacitive_participation": row["capacitive_energy_j"]
                / normalization_j,
            }
            for row in junctions
            if row["junction_id"] in selected_junctions
        ],
    }


def domain_rows(source: SgbEprSource, raw: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The electric and magnetic energy of every domain; the integrals must cover exactly the domains."""
    tables = {
        name: exact_mapping(raw.get(name), name)
        for name in (
            "electric_domain_integrals_v2_m",
            "magnetic_domain_integrals_a2_m",
            "relative_permeability",
            "effective_domain_volumes_m3",
        )
    }
    for name, table in tables.items():
        if set(table) != set(source.domains):
            raise ValueError(f"{name} must cover the domains {sorted(source.domains)}")
    rows = []
    for name, domain in source.domains.items():
        electric = quantity(
            tables["electric_domain_integrals_v2_m"][name],
            f"electric integral {name!r}",
            "V^2*m",
        )
        magnetic = quantity(
            tables["magnetic_domain_integrals_a2_m"][name],
            f"magnetic integral {name!r}",
            "A^2*m",
        )
        permeability = quantity(
            tables["relative_permeability"][name],
            f"relative permeability {name!r}",
            "1",
        )
        if permeability <= 0.0:
            raise ValueError("relative permeability must be > 0")
        rows.append(
            {
                "domain_id": name,
                "material_id": domain.material_id,
                "relative_permittivity": domain.relative_permittivity,
                "native_relative_permeability": permeability,
                "effective_volume_m3": quantity(
                    tables["effective_domain_volumes_m3"][name],
                    f"effective volume {name!r}",
                    "m^3",
                ),
                "integral_v2_m": electric,
                "energy_j": electric_energy_coefficient(domain.relative_permittivity)
                * electric,
                "magnetic_h_conj_dot_h_integral_a2_m": magnetic,
                "magnetic_energy_j": _MU_0_H_PER_M * permeability * magnetic / 2.0,
            }
        )
    return rows


def electric_energy_coefficient(relative_permittivity: float) -> float:
    """The factor from ``integral |E|^2 dV`` to the peak electric energy of a domain."""
    return VACUUM_PERMITTIVITY_F_PER_M * relative_permittivity / 2.0


def junction_rows(
    source: SgbEprSource, raw: Mapping[str, Any], *, angular_frequency: float | None
) -> list[dict[str, Any]]:
    """Each junction's voltage and energies; ``angular_frequency`` ``None`` gives the capacitive energy only."""
    integrals = exact_mapping(
        raw.get("junction_integrals_v_m"), "junction_integrals_v_m"
    )
    if set(integrals) != set(source.junctions):
        raise ValueError(
            f"junction integrals must cover the junctions {sorted(source.junctions)}"
        )
    rows = []
    for junction_id, sheet in source.junctions.items():
        value = exact_mapping(integrals[junction_id], f"junction {junction_id!r}")
        real = quantity(
            value.get("real"), "junction real integral", "V*m", nonnegative=False
        )
        imag = quantity(
            value.get("imag"), "junction imaginary integral", "V*m", nonnegative=False
        )
        voltage = complex(real, imag) / (sheet.width_um * 1e-6)
        junction = sheet.junction
        row: dict[str, Any] = {
            "junction_id": junction_id,
            "port_group": sheet.port_group,
            "terminal_a_net": junction.terminal_a_net,
            "terminal_b_net": junction.terminal_b_net,
            "direction": list(sheet.direction),
            "width_um": sheet.width_um,
            "voltage_real_v": voltage.real,
            "voltage_imag_v": voltage.imag,
            "capacitive_energy_j": junction.capacitance_f * abs(voltage) ** 2 / 2.0,
        }
        if angular_frequency is not None:
            current = voltage / complex(0.0, angular_frequency * junction.inductance_h)
            row["current_real_a"] = current.real
            row["current_imag_a"] = current.imag
            row["inductive_energy_j"] = abs(voltage) ** 2 / (
                2.0 * angular_frequency**2 * junction.inductance_h
            )
        rows.append(row)
    return rows


def junction_capacitive_energy_coefficient(
    capacitance_f: float, width_um: float
) -> float:
    """The factor from ``|area integral|^2`` of a junction sheet to its capacitive energy."""
    return capacitance_f / (2.0 * (width_um * 1e-6) ** 2)


def normalization_energy_j(
    electric_energy_j: float, capacitive_energy_j: float
) -> float:
    """The mode's peak energy: the electric energy of the fields plus the junctions' capacitive energy."""
    total = electric_energy_j + capacitive_energy_j
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("EPR normalization energy must be finite and > 0")
    return total


def surface_contributions(
    source: SgbEprSource, raw: Mapping[str, Any], normalization_j: float
) -> list[dict[str, Any]]:
    """One contribution per Surface-EPR entry: the film energy of its rows, participation and ``1/Q``."""
    granularity = raw.get("surface_granularity")
    if granularity != SURFACE_GRANULARITY:
        raise ValueError(
            f"raw EPR mode has surface granularity {granularity!r}, not {SURFACE_GRANULARITY!r}; "
            "pass the raw mode of an AEDT run of the SGB model"
        )
    integrals = exact_mapping(raw.get("surface_integrals_v2"), "surface_integrals_v2")
    areas = exact_mapping(raw.get("masked_areas_m2"), "masked_areas_m2")
    if set(integrals) != set(source.rows) or set(areas) != set(source.rows):
        raise ValueError(
            f"surface integrals must cover the Surface-EPR rows {sorted(source.rows)}"
        )
    rows = {}
    for row_id, row in source.rows.items():
        film = source.films[row.interface_type]
        values = exact_mapping(integrals[row_id], f"surface integral {row_id!r}")
        normal = quantity(values.get("normal"), f"normal integral {row_id!r}", "V^2")
        tangential = quantity(
            values.get("tangential"),
            f"tangential integral {row_id!r}",
            "V^2",
            nonnegative=False,
        )
        rows[row_id] = {
            "normal": normal,
            "tangential": tangential,
            "area": quantity(areas[row_id], f"area {row_id!r}", "m^2"),
            "energy": surface_energy_j(
                row.interface_type,
                film.film_thickness_m,
                film.permittivity,
                # An MA film's energy does not depend on the substrate; the formula takes a number anyway.
                1.0
                if row.substrate_relative_permittivity is None
                else row.substrate_relative_permittivity,
                (normal, tangential),
            ),
        }
    return [
        _contribution(source, entry, rows, normalization_j) for entry in source.entries
    ]


def _contribution(
    source: SgbEprSource,
    entry: SurfaceEntry,
    rows: Mapping[str, Mapping[str, float]],
    normalization_j: float,
) -> dict[str, Any]:
    film = source.films[entry.interface_type]
    energy = sum(rows[row]["energy"] for row in entry.rows)
    participation = energy / normalization_j
    record: dict[str, Any] = {
        "group_id": entry.entry_id,
        "owner_id": ",".join(entry.entities),
        "interface_kind": entry.interface_type,
        "evaluation_kind": entry.evaluation_kind,
        "margin_um": entry.margin_um,
        "film_thickness_m": film.film_thickness_m,
        "film_relative_permittivity": film.permittivity,
        "members": [
            {
                "row": row,
                "group": source.rows[row].group,
                "band": source.rows[row].band,
                "selection_name": source.rows[row].selection_name,
                "energy_j": rows[row]["energy"],
            }
            for row in entry.rows
        ],
        "normal_integral_v2": sum(rows[row]["normal"] for row in entry.rows),
        "tangential_integral_v2": sum(rows[row]["tangential"] for row in entry.rows),
        "masked_area_m2": sum(rows[row]["area"] for row in entry.rows),
        "energy_j": energy,
        "participation": participation,
        "assumptions": film_assumptions(film.assumptions()),
        "inverse_q": inverse_quality_factor(participation, film.loss_tangent),
    }
    if entry.substrate_relative_permittivity is not None:
        record["substrate_relative_permittivity"] = (
            entry.substrate_relative_permittivity
        )
    return record


def _selected(
    request: EprAnalysisRequest | None, name: str, known: Mapping[str, Any]
) -> set[str]:
    """The ids a request field selects (``None``: all); an unknown id is refused."""
    value = None if request is None else getattr(request, name)
    if value is None:
        return set(known)
    unknown = sorted(set(value) - set(known))
    if unknown:
        raise ValueError(f"EPR request {name} names unknown ids {unknown}")
    return set(value)
