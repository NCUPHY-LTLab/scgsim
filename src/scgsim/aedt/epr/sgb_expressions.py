"""The field expressions the EPR of an imported SGB model needs, before any of them is loaded into AEDT (issue #67).

:func:`sgb_expression_requests` lists, from what :mod:`scgsim.aedt.epr.sgb_source` names and the sheets
:mod:`scgsim.aedt.epr.sgb_surfaces` created:

1. per domain, its volume and its electric and magnetic volume integrals;
2. per Surface-EPR row, the normal and tangential electric integrals and the area on the row's sheet, the
   field read on the sheet itself or just off it as the row's group's selector says (``surface`` or
   ``adjacent``), with the sheet's own normal (:func:`scgsim.aedt.epr.fields.surface_row_operations`);
3. per junction, the real and imaginary area integrals of the field along its port's direction on its
   voltage sheet.

:func:`normalized_surface_total_operations` is the expression-cache convergence target over the same names:
the film energy of the rows a target selects, divided by the mode's peak energy.  The requests are compiled
and loaded by the expression codec, which owns naming and readback.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from scgsim.aedt.epr.fields import (
    field_integral_operations,
    junction_voltage_operations,
    surface_row_operations,
)
from scgsim.aedt.epr.sgb_energy import (
    electric_energy_coefficient,
    junction_capacitive_energy_coefficient,
)
from scgsim.aedt.epr.sgb_source import SgbEprSource
from scgsim.aedt.epr.sgb_surfaces import (
    FieldSelector,
    JunctionSelection,
    SurfaceSelection,
)
from scgsim.aedt.epr.surface import surface_energy_coefficients
from scgsim.aedt.specs.epr import NormalizedSurfaceEprTotal

VOLUME_QUANTITIES = ("effective_volume", "electric_volume", "magnetic_energy")
SURFACE_QUANTITIES = ("electric_normal", "electric_tangential", "masked_area")
JUNCTION_QUANTITIES = ("junction_voltage_real", "junction_voltage_imag")


@dataclass(frozen=True)
class ExpressionRequest:
    """One field expression to compile: its purpose, calculator stack and the selection it integrates.

    ``adjacent_selection_name`` is the sheet an ``adjacent`` integral is read just off, else ``None``.
    """

    purpose: str
    operations: tuple[str, ...]
    selection: Mapping[str, Any]
    adjacent_selection_name: str | None = None


def sgb_expression_requests(
    source: SgbEprSource,
    rows: Mapping[str, SurfaceSelection],
    junctions: Mapping[str, JunctionSelection],
    *,
    relative_permeability: Mapping[str, float],
    field_selectors: Mapping[str, FieldSelector] | None = None,
) -> list[ExpressionRequest]:
    """Every expression of the model's EPR (see the module); ``relative_permeability`` is by domain, as AEDT read it."""
    if set(rows) != set(source.rows):
        raise ValueError("the row sheets must be exactly the selected Surface-EPR rows")
    if set(junctions) != set(source.junctions):
        raise ValueError("the voltage sheets must be exactly the model's junctions")
    selectors = dict(field_selectors or {})
    unknown = sorted(set(selectors) - {row.group for row in source.rows.values()})
    if unknown:
        raise ValueError(
            f"surface_field_selectors name groups no selected row carries: {unknown}"
        )
    return [
        *_volume_requests(source, relative_permeability),
        *_row_requests(rows, selectors),
        *_junction_requests(source, junctions),
    ]


def _volume_requests(
    source: SgbEprSource, relative_permeability: Mapping[str, float]
) -> list[ExpressionRequest]:
    out = []
    for name in source.domains:
        selection = {
            "kind": "volume",
            "semantic_id": name,
            "object_name": name,
            "relative_permeability": float(relative_permeability[name]),
        }
        for quantity in VOLUME_QUANTITIES:
            operations = field_integral_operations(
                quantity=quantity, selection_name=name
            )
            out.append(
                ExpressionRequest(f"{quantity}_{name}", tuple(operations), selection)
            )
    return out


def _row_requests(
    rows: Mapping[str, SurfaceSelection], selectors: Mapping[str, str]
) -> list[ExpressionRequest]:
    out = []
    for row_id, binding in sorted(rows.items()):
        selector = selectors.get(binding.group, "surface")
        if selector not in ("surface", "adjacent"):
            raise ValueError(
                f"the field selector of {binding.group} must be 'surface' or 'adjacent'"
            )
        selection = {
            "kind": "surface_row",
            "row": row_id,
            "group": binding.group,
            "interface_type": binding.interface_type,
            "selection_name": binding.selection_name,
            "field_selector": selector,
        }
        for quantity in SURFACE_QUANTITIES:
            # The area is that of the sheet whichever side the field is read on.
            adjacent = selector == "adjacent" and quantity != "masked_area"
            operations = surface_row_operations(
                quantity=quantity,
                selection_name=binding.selection_name,
                adjacent_side=adjacent,
            )
            out.append(
                ExpressionRequest(
                    f"{quantity}_{row_id}",
                    tuple(operations),
                    selection,
                    binding.selection_name if adjacent else None,
                )
            )
    return out


def _junction_requests(
    source: SgbEprSource, junctions: Mapping[str, JunctionSelection]
) -> list[ExpressionRequest]:
    out = []
    for junction_id, sheet in sorted(source.junctions.items()):
        binding = junctions[junction_id]
        selection = {
            "kind": "junction_sheet",
            "junction_id": junction_id,
            "object_name": binding.selection_name,
            "port_group": sheet.port_group,
            "direction": list(sheet.direction),
            "width_um": sheet.width_um,
        }
        for quantity in JUNCTION_QUANTITIES:
            operations = junction_voltage_operations(
                quantity=quantity,
                selection_name=binding.selection_name,
                direction_xy=sheet.direction[:2],
            )
            out.append(
                ExpressionRequest(
                    f"{quantity}_{junction_id}", tuple(operations), selection
                )
            )
    return out


def normalized_surface_total_operations(
    source: SgbEprSource,
    names: Mapping[str, str],
    target: NormalizedSurfaceEprTotal,
) -> list[str]:
    """The participation of the rows of every entry ``target`` selects, each row once, as one stack.

    ``names`` maps each request purpose to its compiled expression name.  The numerator is the film energy
    of those rows; the denominator the electric energy of every domain plus the capacitive energy of every
    junction, the normalization of :func:`scgsim.aedt.epr.sgb_energy.combine_sgb_epr_mode`.
    """
    selected = sorted(
        {
            row
            for entry in source.entries
            if entry.interface_type == target.interface_kind
            and entry.evaluation_kind == target.evaluation_kind
            and entry.margin_um == float(target.margin_um)
            for row in entry.rows
        }
    )
    if not selected:
        raise ValueError(
            f"the convergence target {target.to_payload()} selects no Surface-EPR row"
        )
    numerator: list[tuple[str, float]] = []
    for row_id in selected:
        row = source.rows[row_id]
        film = source.films[row.interface_type]
        normal, tangential = surface_energy_coefficients(
            row.interface_type,
            film.film_thickness_m,
            film.permittivity,
            1.0
            if row.substrate_relative_permittivity is None
            else row.substrate_relative_permittivity,
        )
        numerator += [
            (names[f"electric_normal_{row_id}"], normal),
            (names[f"electric_tangential_{row_id}"], tangential),
        ]
    denominator: list[list[str]] = [
        _weighted(
            [
                (
                    names[f"electric_volume_{name}"],
                    electric_energy_coefficient(d.relative_permittivity),
                )
            ]
        )
        for name, d in source.domains.items()
    ]
    for junction_id, sheet in sorted(source.junctions.items()):
        coefficient = junction_capacitive_energy_coefficient(
            sheet.junction.capacitance_f, sheet.width_um
        )
        if coefficient == 0.0:
            continue
        real = names[f"junction_voltage_real_{junction_id}"]
        imag = names[f"junction_voltage_imag_{junction_id}"]
        denominator.append(
            [
                f"NameOfExpression('{real}')",
                f"NameOfExpression('{real}')",
                "Operation('*')",
                f"NameOfExpression('{imag}')",
                f"NameOfExpression('{imag}')",
                "Operation('*')",
                "Operation('+')",
                f"Scalar_Constant({coefficient:.17g})",
                "Operation('*')",
            ]
        )
    return [*_weighted(numerator), *_summed(denominator), "Operation('/')"]


def _weighted(terms: Sequence[tuple[str, float]]) -> list[str]:
    """``sum(coefficient * expression)`` over the terms with a nonzero coefficient."""
    return _summed(
        [
            [
                f"NameOfExpression('{name}')",
                f"Scalar_Constant({coefficient:.17g})",
                "Operation('*')",
            ]
            for name, coefficient in terms
            if coefficient != 0.0
        ]
    )


def _summed(terms: Sequence[Sequence[str]]) -> list[str]:
    if not terms:
        raise ValueError("a normalized expression needs at least one term")
    operations: list[str] = [*terms[0]]
    for term in terms[1:]:
        operations += [*term, "Operation('+')"]
    return operations
