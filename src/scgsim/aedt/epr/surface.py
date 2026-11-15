"""Surface energy composition and native-face evidence validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from scgsim.aedt.data import detached
from scgsim.aedt.epr.selection import (
    exact_mapping,
    finite_value,
    surface_integral_groups,
)
from scgsim.aedt.specs.epr import EprAnalysisRequest
from scgsim.aedt.specs.geometry import PreparedPlanarGeometry
from scgsim.semantics.epr import film_assumptions, inverse_quality_factor

# Vacuum permittivity is shared by surface and bulk electric-energy primitives.
VACUUM_PERMITTIVITY_F_PER_M = 8.8541878128e-12


def quantity(value: Any, name: str, unit: str, *, nonnegative: bool = True) -> float:
    """Validate a canonical unit-bearing quantity and return its numeric value."""
    item = exact_mapping(value, name)
    if set(item) != {"value", "unit"} or item.get("unit") != unit:
        raise ValueError(f"{name} must use canonical unit {unit!r}")
    return finite_value(item.get("value"), name, nonnegative=nonnegative)


def surface_energy_j(
    interface_kind: str,
    thickness_m: float,
    film_epsilon: float,
    substrate_epsilon: float,
    integrals: tuple[float, float],
) -> float:
    """Compute one surface energy from its normal and tangential integrals."""
    normal, tangential = integrals
    if interface_kind == "MA":
        return VACUUM_PERMITTIVITY_F_PER_M * thickness_m * normal / (2.0 * film_epsilon)
    if interface_kind == "MS":
        return (
            VACUUM_PERMITTIVITY_F_PER_M
            * thickness_m
            * substrate_epsilon**2
            * normal
            / (2.0 * film_epsilon)
        )
    if interface_kind == "SA":
        return (
            VACUUM_PERMITTIVITY_F_PER_M
            * thickness_m
            * (film_epsilon * tangential + normal / film_epsilon)
            / 2.0
        )
    raise ValueError(f"unsupported surface interface {interface_kind!r}")


def surface_energy_coefficients(
    interface_kind: str,
    thickness_m: float,
    film_epsilon: float,
    substrate_epsilon: float,
) -> tuple[float, float]:
    """Return the normal and tangential peak-phasor energy factors of one interface."""
    if interface_kind == "MA":
        return VACUUM_PERMITTIVITY_F_PER_M * thickness_m / (2.0 * film_epsilon), 0.0
    if interface_kind == "MS":
        return (
            VACUUM_PERMITTIVITY_F_PER_M
            * thickness_m
            * substrate_epsilon**2
            / (2.0 * film_epsilon),
            0.0,
        )
    if interface_kind == "SA":
        return (
            VACUUM_PERMITTIVITY_F_PER_M * thickness_m / (2.0 * film_epsilon),
            VACUUM_PERMITTIVITY_F_PER_M * thickness_m * film_epsilon / 2.0,
        )
    raise ValueError(f"unsupported surface interface {interface_kind!r}")


@dataclass(frozen=True)
class SurfaceContributionInputs:
    """Public inputs of the grouped surface calculation."""

    prepared: PreparedPlanarGeometry
    raw: Mapping[str, Any]
    surface: Mapping[str, Any]
    masked_areas: Mapping[str, Any]
    request: EprAnalysisRequest | None
    normalization_j: float


@dataclass(frozen=True)
class SurfaceContributionContext:
    """Prepared inputs of the grouped surface calculation."""

    prepared: PreparedPlanarGeometry
    raw: Mapping[str, Any]
    surface: Mapping[str, Any]
    masked_areas: Mapping[str, Any]
    request: EprAnalysisRequest | None
    normalization_j: float
    specs_by_id: dict[str, Any]


def surface_contributions(
    inputs: SurfaceContributionInputs,
) -> tuple[str, list[dict[str, Any]]]:
    """Combine the grouped surface records of one raw mode with one denominator."""
    prepared = inputs.prepared
    for item in prepared.surface_bindings:
        contribution = exact_mapping(
            item.get("contribution"), "surface binding contribution"
        )
        contribution_id = contribution.get("contribution_id")
        if not isinstance(contribution_id, str) or not contribution_id:
            raise ValueError("surface binding contribution id is invalid")
    granularity = inputs.raw.get("surface_granularity")
    if granularity is None:
        raise ValueError(
            "raw EPR mode names no surface_granularity; an AEDT run of this "
            "SCGSim writes owner_interface_margin.v1, so pass the raw mode of a "
            "current run"
        )
    if granularity != "owner_interface_margin.v1":
        raise ValueError(f"unsupported surface granularity {granularity!r}")
    context = SurfaceContributionContext(
        prepared,
        inputs.raw,
        inputs.surface,
        inputs.masked_areas,
        inputs.request,
        inputs.normalization_j,
        {item.contribution_id: item for item in prepared.contributions},
    )
    return granularity, _grouped_surface_rows(context)


def _grouped_surface_rows(
    context: SurfaceContributionContext,
) -> list[dict[str, Any]]:
    """Verify grouped provenance, then calculate its contribution rows."""
    prepared = context.prepared
    raw = context.raw
    surface = context.surface
    masked_areas = context.masked_areas
    request = context.request
    normalization_j = context.normalization_j
    specs_by_id = context.specs_by_id
    rows: list[dict[str, Any]] = []
    groups = surface_integral_groups(prepared, request)
    expected_groups = {item["group_id"]: item for item in groups}
    provenance = exact_mapping(
        raw.get("surface_group_provenance"), "surface group provenance"
    )
    if (
        set(surface) != set(expected_groups)
        or set(masked_areas) != set(expected_groups)
        or set(provenance) != set(expected_groups)
    ):
        raise ValueError("grouped surface integrals do not cover prepared groups")
    for group_id, group in expected_groups.items():
        native_group = exact_mapping(provenance[group_id], "native surface group")
        if any(native_group.get(key) != value for key, value in group.items()):
            raise ValueError(f"surface group provenance differs for {group_id!r}")
        values = exact_mapping(surface[group_id], "grouped surface integral")
        normal = quantity(values.get("normal"), "normal group integral", "V^2")
        tangential = quantity(
            values.get("tangential"),
            "tangential group integral",
            "V^2",
            nonnegative=False,
        )
        energy = surface_energy_j(
            group["interface_kind"],
            group["film_thickness_m"],
            group["film_relative_permittivity"],
            group["substrate_relative_permittivity"],
            (normal, tangential),
        )
        group_specs = {
            (
                specs_by_id[member["contribution_id"]].loss_tangent,
                specs_by_id[member["contribution_id"]].source,
                specs_by_id[member["contribution_id"]].preset,
            )
            for member in group["members"]
        }
        if len(group_specs) != 1:
            raise ValueError(
                "one physical surface group has conflicting film provenance"
            )
        loss_tangent, source_name, preset_name = group_specs.pop()
        assumptions = film_assumptions(
            {
                "film_thickness_m": group["film_thickness_m"],
                "film_relative_permittivity": group["film_relative_permittivity"],
                "loss_tangent": loss_tangent,
                "source": source_name,
                "preset": preset_name,
            }
        )
        rows.append(
            {
                **group,
                "normal_integral_v2": normal,
                "tangential_integral_v2": tangential,
                "masked_area_m2": quantity(
                    masked_areas[group_id], "masked group area", "m^2"
                ),
                "energy_j": energy,
                "participation": energy / normalization_j,
                "assumptions": assumptions,
                "inverse_q": inverse_quality_factor(
                    energy / normalization_j, loss_tangent
                ),
            }
        )
    return rows


def native_surface_reference_rows(
    prepared: PreparedPlanarGeometry,
    raw: Mapping[str, Any],
    request: EprAnalysisRequest | None,
) -> list[dict[str, Any]] | None:
    """Detach measured native-face references without treating them as film EPR."""

    selected = (
        {item.contribution_id for item in prepared.contributions}
        if request is None or request.surface_contribution_ids is None
        else set(request.surface_contribution_ids)
    )
    expected_contributions = {
        item.contribution_id
        for item in prepared.contributions
        if item.contribution_id in selected
        and item.interface_kind in {"MA", "MS", "SA"}
    }
    references_requested = prepared.modeling == "solid" and bool(expected_contributions)
    if "native_surface_references" not in raw:
        # A raw record without references has none to report; do not invent them.
        return None
    references = exact_mapping(
        raw["native_surface_references"], "native_surface_references"
    )
    if references_requested and not references:
        raise ValueError("selected Solid EPR contributions lack native face references")

    rows: list[dict[str, Any]] = []
    associated_contributions: set[str] = set()
    for reference_id, value in sorted(references.items()):
        record = exact_mapping(value, "native surface reference")
        selection = exact_mapping(
            record.get("selection"), "native surface reference selection"
        )
        if (
            selection.get("kind") != "native_surface_reference"
            or selection.get("reference_id") != reference_id
        ):
            raise ValueError("native surface reference identity differs")
        contribution_ids = selection.get("contribution_ids")
        if (
            isinstance(contribution_ids, (str, bytes))
            or not isinstance(contribution_ids, (list, tuple))
            or any(not isinstance(item, str) or not item for item in contribution_ids)
        ):
            raise TypeError("native surface reference contribution IDs are invalid")
        associated_contributions.update(contribution_ids)
        normal = quantity(record.get("normal"), "native face normal integral", "V^2")
        tangential = quantity(
            record.get("tangential"),
            "native face tangential integral",
            "V^2",
            nonnegative=False,
        )
        area = quantity(record.get("area"), "native face area integral", "m^2")
        expression_identities = exact_mapping(
            record.get("expression_identities"),
            "native surface reference expression identities",
        )
        if set(expression_identities) != {"normal", "tangential", "area"}:
            raise ValueError("native surface reference expressions are incomplete")
        row = {
            **detached(selection),
            "normal_integral_v2": normal,
            "tangential_integral_v2": tangential,
            "area_m2": area,
            "expression_identities": detached(expression_identities),
        }
        if "saved_membership" in record:
            row["saved_membership"] = detached(record["saved_membership"])
        rows.append(row)
    if references_requested and associated_contributions != expected_contributions:
        raise ValueError(
            "native face references do not cover selected EPR contributions"
        )
    return rows
