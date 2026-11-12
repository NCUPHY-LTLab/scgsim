"""Surface energy composition and native-face evidence validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

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
    """Return the existing normal and tangential peak-phasor energy factors."""
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
    """Public inputs shared by grouped and per-binding surface calculations."""

    prepared: PreparedPlanarGeometry
    raw: Mapping[str, Any]
    surface: Mapping[str, Any]
    masked_areas: Mapping[str, Any]
    request: EprAnalysisRequest | None
    normalization_j: float


@dataclass(frozen=True)
class SurfaceContributionContext:
    """Prepared inputs shared by grouped and per-binding surface calculations."""

    prepared: PreparedPlanarGeometry
    raw: Mapping[str, Any]
    surface: Mapping[str, Any]
    masked_areas: Mapping[str, Any]
    request: EprAnalysisRequest | None
    normalization_j: float
    bindings: dict[str, list[Mapping[str, Any]]]
    specs_by_id: dict[str, Any]
    selected_surfaces: set[str]
    granularity: Any


def surface_contributions(
    inputs: SurfaceContributionInputs,
) -> tuple[str, list[dict[str, Any]]]:
    """Combine grouped or per-binding surface records with one denominator."""
    prepared = inputs.prepared
    raw = inputs.raw
    surface = inputs.surface
    masked_areas = inputs.masked_areas
    request = inputs.request
    normalization_j = inputs.normalization_j
    bindings: dict[str, list[Mapping[str, Any]]] = {}
    for item in prepared.surface_bindings:
        contribution = exact_mapping(
            item.get("contribution"), "surface binding contribution"
        )
        contribution_id = contribution.get("contribution_id")
        if not isinstance(contribution_id, str) or not contribution_id:
            raise ValueError("surface binding contribution id is invalid")
        bindings.setdefault(contribution_id, []).append(item)
    specs_by_id = {item.contribution_id: item for item in prepared.contributions}
    selected_surfaces = (
        set(specs_by_id)
        if request is None or request.surface_contribution_ids is None
        else set(request.surface_contribution_ids)
    )
    context = SurfaceContributionContext(
        prepared,
        raw,
        surface,
        masked_areas,
        request,
        normalization_j,
        bindings,
        specs_by_id,
        selected_surfaces,
        raw.get("surface_granularity"),
    )
    if context.granularity == "owner_interface_margin.v1":
        rows = _grouped_surface_rows(context)
    elif context.granularity is None:
        _validate_per_binding_inputs(context)
        rows = _per_binding_surface_rows(context)
    else:
        raise ValueError(f"unsupported surface granularity {context.granularity!r}")
    return context.granularity or "per_binding.v1", rows


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


def _validate_per_binding_inputs(context: SurfaceContributionContext) -> None:
    """Check per-binding integral and mask keys before any row calculation."""
    prepared = context.prepared
    surface = context.surface
    masked_areas = context.masked_areas
    specs_by_id = context.specs_by_id
    selected_surfaces = context.selected_surfaces
    expected_surface_keys = {
        f"{binding['binding_id']}@{margin:.17g}"
        for binding in prepared.surface_bindings
        if binding["contribution"]["contribution_id"] in selected_surfaces
        for margin in specs_by_id[binding["contribution"]["contribution_id"]].margins_um
        if specs_by_id[binding["contribution"]["contribution_id"]].interface_kind
        in {"MA", "MS", "SA"}
    }
    if set(surface) != expected_surface_keys:
        raise ValueError("surface integrals do not match prepared contribution margins")
    if set(masked_areas) != expected_surface_keys:
        raise ValueError("masked areas do not match prepared contribution margins")


def _per_binding_surface_rows(
    context: SurfaceContributionContext,
) -> list[dict[str, Any]]:
    """Calculate legacy per-binding surface energies in prepared order."""
    prepared = context.prepared
    surface = context.surface
    masked_areas = context.masked_areas
    bindings = context.bindings
    selected_surfaces = context.selected_surfaces
    normalization_j = context.normalization_j
    rows: list[dict[str, Any]] = []
    for spec in prepared.contributions:
        if spec.interface_kind == "MM" or spec.contribution_id not in selected_surfaces:
            continue
        local_bindings = bindings.get(spec.contribution_id)
        if not local_bindings:
            raise ValueError(f"missing prepared binding for {spec.contribution_id!r}")
        epsilon_values: set[float] = set()
        for binding in local_bindings:
            contribution = exact_mapping(
                binding.get("contribution"), "surface binding contribution"
            )
            if (
                contribution.get("classification") != spec.interface_kind
                or contribution.get("side") != spec.field_side
            ):
                raise ValueError(
                    f"prepared binding contradicts contribution {spec.contribution_id!r}"
                )
            material_key = (
                "substrate_material"
                if spec.interface_kind == "SA"
                else "effective_material"
            )
            material = exact_mapping(
                binding.get(material_key), "surface binding material"
            )
            if "permittivity" not in material:
                raise ValueError(
                    "field-side material lacks explicit relative permittivity"
                )
            epsilon_values.add(
                finite_value(
                    material["permittivity"],
                    "field-side relative permittivity",
                )
            )
        if len(epsilon_values) != 1:
            raise ValueError(
                f"contribution {spec.contribution_id!r} spans inconsistent field-side materials"
            )
        epsilon_s = epsilon_values.pop()
        for margin in spec.margins_um:
            normal = 0.0
            tangential = 0.0
            binding_ids: list[str] = []
            for binding in local_bindings:
                binding_id = binding.get("binding_id")
                if not isinstance(binding_id, str) or not binding_id:
                    raise ValueError("surface binding id is invalid")
                key = f"{binding_id}@{margin:.17g}"
                values = exact_mapping(surface[key], f"surface integral {key!r}")
                normal += quantity(
                    values.get("normal"), "normal surface integral", "V^2"
                )
                tangential += quantity(
                    values.get("tangential"),
                    "tangential surface integral",
                    "V^2",
                    nonnegative=False,
                )
                binding_ids.append(binding_id)
            energy = surface_energy_j(
                spec.interface_kind,
                cast(float, spec.film_thickness_m),
                cast(float, spec.film_relative_permittivity),
                epsilon_s,
                (normal, tangential),
            )
            rows.append(
                {
                    "contribution_id": spec.contribution_id,
                    "interface_kind": spec.interface_kind,
                    "field_side": spec.field_side,
                    "binding_ids": binding_ids,
                    "margin_um": margin,
                    "normal_integral_v2": normal,
                    "tangential_integral_v2": tangential,
                    "masked_area_m2": quantity(
                        masked_areas[f"{binding_ids[-1]}@{margin:.17g}"]
                        if len(local_bindings) == 1
                        else {
                            "value": sum(
                                quantity(
                                    masked_areas[f"{item['binding_id']}@{margin:.17g}"],
                                    "masked area",
                                    "m^2",
                                )
                                for item in local_bindings
                            ),
                            "unit": "m^2",
                        },
                        "masked area",
                        "m^2",
                    ),
                    "energy_j": energy,
                    "participation": energy / normalization_j,
                    "assumptions": film_assumptions(
                        {
                            "film_thickness_m": spec.film_thickness_m,
                            "film_relative_permittivity": spec.film_relative_permittivity,
                            "loss_tangent": spec.loss_tangent,
                            "source": spec.source,
                            "preset": spec.preset,
                        }
                    ),
                    "inverse_q": inverse_quality_factor(
                        energy / normalization_j, spec.loss_tangent
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
        # Keep previously serialized raw records readable without fabricating
        # reference values that were not part of their original evidence.
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
