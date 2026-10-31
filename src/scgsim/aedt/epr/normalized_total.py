"""Native expression stack for the normalized surface EPR convergence target.

Owns: building the reverse-polish (RPN) operation list that AEDT evaluates as
EPR_surface = (weighted surface integrals) / U_norm, so the solver can watch it
converge.
Does not own: the surface integrals themselves (`expressions.py`) or the energy
coefficients (`analysis.py`, `surface_epr.py`).

U_norm is the electric energy of all domains plus the junctions' capacitive
energy (docs/eigenmode-epr.qmd#quantities-and-normalization). MA (metal-air), MS
(metal-substrate) and SA (substrate-air) are defined in docs/eigenmode-epr.qmd.
"""

from __future__ import annotations

from typing import Any

from scgsim.aedt.epr.analysis import (
    _electric_domain_energy_coefficient,
    _epr_domain_context,
    _junction_capacitive_energy_coefficient,
)
from scgsim.aedt.epr.selection import surface_integral_groups
from scgsim.aedt.epr.surface_epr import _surface_energy_coefficients
from scgsim.aedt.specs.epr import NormalizedSurfaceEprTotal
from scgsim.aedt.specs.hfss_prepared import HfssEprSpec


def _expression_name(
    expressions: list[dict[str, Any]],
    *,
    purpose: str,
    selection_kind: str,
    selection_id_key: str,
    selection_id: str,
) -> str:
    """Native name of the one authored expression with this purpose and owner."""

    matches = [
        item
        for item in expressions
        if item["identity"].get("purpose") == purpose
        and item["identity"].get("selection", {}).get("kind") == selection_kind
        and item["identity"].get("selection", {}).get(selection_id_key) == selection_id
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"compiled EPR expression binding is missing or ambiguous: {purpose!r}"
        )
    return str(matches[0]["name"])


def _normalized_surface_total_operations(
    spec: HfssEprSpec,
    expressions: list[dict[str, Any]],
    target: NormalizedSurfaceEprTotal,
) -> list[str]:
    """Reverse-polish stack for the normalized surface EPR of one interface kind.

    EPR_surface = (sum of weighted surface integrals) / U_norm, where U_norm is the
    electric energy of all domains plus the junctions' capacitive energy
    (docs/eigenmode-epr.qmd#quantities-and-normalization).
    """

    if spec.epr_request is None:
        raise RuntimeError("normalized surface convergence requires an EPR request")
    groups = [
        group
        for group in surface_integral_groups(spec.geometry, spec.epr_request)
        if group["interface_kind"] == target.interface_kind
        and group.get("evaluation_kind", "requested_margin") == target.evaluation_kind
        and float(group["margin_um"]) == target.margin_um
    ]
    if not groups:
        raise RuntimeError("normalized surface convergence target has no owner groups")
    numerator_operations = _surface_numerator_operations(groups, expressions)
    denominator_operations = _energy_denominator_operations(spec, expressions)
    return [
        *numerator_operations,
        *denominator_operations,
        "Operation('/')",
    ]


def _surface_numerator_operations(
    groups: list[dict[str, Any]], expressions: list[dict[str, Any]]
) -> list[str]:
    """Sum of the group integrals, each weighted by its thin-film energy factor."""

    numerator_terms: list[tuple[str, float]] = []
    for group in groups:
        normal_coefficient, tangential_coefficient = _surface_energy_coefficients(
            group["interface_kind"],
            float(group["film_thickness_m"]),
            float(group["film_relative_permittivity"]),
            float(group["substrate_relative_permittivity"]),
        )
        group_id = str(group["group_id"])
        if normal_coefficient != 0.0:
            numerator_terms.append(
                (
                    _expression_name(
                        expressions,
                        purpose=f"electric_normal_{group_id}",
                        selection_kind="surface_group",
                        selection_id_key="group_id",
                        selection_id=group_id,
                    ),
                    normal_coefficient,
                )
            )
        if tangential_coefficient != 0.0:
            numerator_terms.append(
                (
                    _expression_name(
                        expressions,
                        purpose=f"electric_tangential_{group_id}",
                        selection_kind="surface_group",
                        selection_id_key="group_id",
                        selection_id=group_id,
                    ),
                    tangential_coefficient,
                )
            )
    return _weighted_native_terms(numerator_terms)


def _energy_denominator_operations(
    spec: HfssEprSpec, expressions: list[dict[str, Any]]
) -> list[str]:
    """Electric energy of every domain plus the junctions' capacitive energy."""

    _, materials, domains = _epr_domain_context(spec.geometry)
    denominator_terms: list[list[str]] = []
    for domain_id, domain in domains.items():
        material = materials[domain["material_id"]]
        coefficient = _electric_domain_energy_coefficient(
            float(material["permittivity"])
        )
        name = _expression_name(
            expressions,
            purpose=f"electric_volume_{domain_id}",
            selection_kind="volume",
            selection_id_key="semantic_id",
            selection_id=domain_id,
        )
        denominator_terms.append(
            [
                f"NameOfExpression('{name}')",
                f"Scalar_Constant({coefficient:.17g})",
                "Operation('*')",
            ]
        )
    for junction in spec.geometry.junctions:
        coefficient = _junction_capacitive_energy_coefficient(
            junction.capacitance_f, junction.width_um
        )
        if coefficient == 0.0:
            continue
        real_name = _expression_name(
            expressions,
            purpose=f"junction_voltage_real_{junction.junction_id}",
            selection_kind="junction_sheet",
            selection_id_key="junction_id",
            selection_id=junction.junction_id,
        )
        imag_name = _expression_name(
            expressions,
            purpose=f"junction_voltage_imag_{junction.junction_id}",
            selection_kind="junction_sheet",
            selection_id_key="junction_id",
            selection_id=junction.junction_id,
        )
        # coefficient * (Re(V)^2 + Im(V)^2)
        denominator_terms.append(
            [
                f"NameOfExpression('{real_name}')",
                f"NameOfExpression('{real_name}')",
                "Operation('*')",
                f"NameOfExpression('{imag_name}')",
                f"NameOfExpression('{imag_name}')",
                "Operation('*')",
                "Operation('+')",
                f"Scalar_Constant({coefficient:.17g})",
                "Operation('*')",
            ]
        )
    return _native_term_sum(denominator_terms)


def _weighted_native_terms(terms: list[tuple[str, float]]) -> list[str]:
    """Reverse-polish sum of `coefficient * expression`, skipping zero weights."""

    operations: list[str] = []
    for name, coefficient in terms:
        if coefficient == 0.0:
            continue
        has_prior_term = bool(operations)
        operations.extend(
            (
                f"NameOfExpression('{name}')",
                f"Scalar_Constant({coefficient:.17g})",
                "Operation('*')",
            )
        )
        if has_prior_term:
            operations.append("Operation('+')")
    if not operations:
        raise RuntimeError("normalized expression has no weighted terms")
    return operations


def _native_term_sum(terms: list[list[str]]) -> list[str]:
    """Reverse-polish sum of already-built operation lists."""

    if not terms:
        raise RuntimeError("normalized expression has no energy terms")
    operations: list[str] = []
    for term in terms:
        has_prior_term = bool(operations)
        operations.extend(term)
        if has_prior_term:
            operations.append("Operation('+')")
    return operations
