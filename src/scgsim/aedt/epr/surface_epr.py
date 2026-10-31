"""Thin-film surface energy (MA, MS, SA) and the surface rows of an EPR result.

Owns: the thin-film energy formulas for the three interfaces, grouping a mode's
surface integrals into result rows (participation and 1/Q), and the measured
native-face reference rows of Solid models.
Does not own: bulk and junction energies or U_norm (`analysis.py`), the group
rules (`selection.py`), or offline loss reassessment (`reanalysis.py`).

MA (metal-air), MS (metal-substrate) and SA (substrate-air) are the interfaces
defined in docs/eigenmode-epr.qmd; the energy formulas are in
docs/specs/aedt-runtime.qmd#surface-estimates-and-junction-normalization.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

from scgsim.aedt._data import (
    detached,
)
from scgsim.aedt.epr.selection import (
    _exact_mapping,
    _finite,
    _quantity,
    surface_integral_groups,
)
from scgsim.aedt.specs.epr import (
    EprAnalysisRequest,
)
from scgsim.aedt.specs.geometry import (
    PreparedPlanarGeometry,
)
from scgsim.semantics.epr import film_assumptions, inverse_quality_factor

_EPSILON_0_F_PER_M = 8.8541878128e-12

# Interface kinds that carry thin-film energy. MM (metal-metal) does not.
_FILM_INTERFACES = {"MA", "MS", "SA"}


class _SurfaceIntegrals(NamedTuple):
    """Raw surface integrals of one mode and the masked area of each."""

    integrals: Mapping[str, Any]
    masked_areas: Mapping[str, Any]


def _surface_energy_j(  # noqa: PLR0913 -- one argument per symbol of the formula.
    interface_kind: str,
    thickness_m: float,
    film_epsilon: float,
    substrate_epsilon: float,
    normal: float,
    tangential: float,
) -> float:
    """Thin-film energy U of one interface from its field integrals.

    With film thickness t, film permittivity eps_f and substrate permittivity
    eps_s (docs/specs/aedt-runtime.qmd#surface-estimates-and-junction-normalization):

    - MA: eps0 t / (2 eps_f) * integral |E_n|^2 dA
    - MS: eps0 t eps_s^2 / (2 eps_f) * integral |E_n|^2 dA
    - SA: eps0 t / 2 * integral (eps_f |E_t|^2 + |E_n|^2 / eps_f) dA

    `normal` and `tangential` are those integrals (V^2). The tangential integral
    only enters SA.
    """

    if interface_kind == "MA":
        return _EPSILON_0_F_PER_M * thickness_m * normal / (2.0 * film_epsilon)
    if interface_kind == "MS":
        return (
            _EPSILON_0_F_PER_M
            * thickness_m
            * substrate_epsilon**2
            * normal
            / (2.0 * film_epsilon)
        )
    if interface_kind == "SA":
        return (
            _EPSILON_0_F_PER_M
            * thickness_m
            * (film_epsilon * tangential + normal / film_epsilon)
            / 2.0
        )
    raise ValueError(f"unsupported surface interface {interface_kind!r}")


def _surface_energy_coefficients(
    interface_kind: str,
    thickness_m: float,
    film_epsilon: float,
    substrate_epsilon: float,
) -> tuple[float, float]:
    """Return the existing normal and tangential peak-phasor energy factors.

    The same formulas as `_surface_energy_j`, with the integrals factored out so
    the factors can weight native expressions in AEDT.
    """
    if interface_kind == "MA":
        return _EPSILON_0_F_PER_M * thickness_m / (2.0 * film_epsilon), 0.0
    if interface_kind == "MS":
        return (
            _EPSILON_0_F_PER_M
            * thickness_m
            * substrate_epsilon**2
            / (2.0 * film_epsilon),
            0.0,
        )
    if interface_kind == "SA":
        return (
            _EPSILON_0_F_PER_M * thickness_m / (2.0 * film_epsilon),
            _EPSILON_0_F_PER_M * thickness_m * film_epsilon / 2.0,
        )
    raise ValueError(f"unsupported surface interface {interface_kind!r}")


def _surface_contributions(
    prepared: PreparedPlanarGeometry,
    raw: Mapping[str, Any],
    surface: _SurfaceIntegrals,
    request: EprAnalysisRequest | None,
    normalization_j: float,
) -> tuple[str, list[dict[str, Any]]]:
    """Combine the prepared surface groups using the shared EPR denominator.

    Raw integrals come in one of two layouts. `owner_interface_margin.v1` has one
    integral per physical group (current). Records with no granularity are the
    older per-binding layout.
    """
    bindings = _bindings_by_contribution(prepared)
    specs_by_id = {item.contribution_id: item for item in prepared.contributions}
    selected_surfaces = (
        set(specs_by_id)
        if request is None or request.surface_contribution_ids is None
        else set(request.surface_contribution_ids)
    )
    granularity = raw.get("surface_granularity")
    if granularity == "owner_interface_margin.v1":
        surface_rows = _grouped_surface_rows(
            prepared, raw, surface, request, normalization_j, specs_by_id
        )
    elif granularity is None:
        _require_per_binding_keys(prepared, surface, specs_by_id, selected_surfaces)
        surface_rows = _per_binding_surface_rows(
            prepared, bindings, surface, normalization_j, selected_surfaces
        )
    else:
        raise ValueError(f"unsupported surface granularity {granularity!r}")
    return granularity or "per_binding.v1", surface_rows


def _bindings_by_contribution(
    prepared: PreparedPlanarGeometry,
) -> dict[str, list[Mapping[str, Any]]]:
    """Group the prepared surface bindings by the contribution they belong to."""

    bindings: dict[str, list[Mapping[str, Any]]] = {}
    for item in prepared.surface_bindings:
        contribution = _exact_mapping(
            item.get("contribution"), "surface binding contribution"
        )
        contribution_id = contribution.get("contribution_id")
        if not isinstance(contribution_id, str) or not contribution_id:
            raise ValueError("surface binding contribution id is invalid")
        bindings.setdefault(contribution_id, []).append(item)
    return bindings


def _grouped_surface_rows(  # noqa: PLR0913 -- one argument per input of the step.
    prepared: PreparedPlanarGeometry,
    raw: Mapping[str, Any],
    surface: _SurfaceIntegrals,
    request: EprAnalysisRequest | None,
    normalization_j: float,
    specs_by_id: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """One row per physical surface group: energy, participation and 1/Q."""

    surface_rows: list[dict[str, Any]] = []
    groups = surface_integral_groups(prepared, request)
    expected_groups = {item["group_id"]: item for item in groups}
    provenance = _exact_mapping(
        raw.get("surface_group_provenance"), "surface group provenance"
    )
    if (
        set(surface.integrals) != set(expected_groups)
        or set(surface.masked_areas) != set(expected_groups)
        or set(provenance) != set(expected_groups)
    ):
        raise ValueError("grouped surface integrals do not cover prepared groups")
    for group_id, group in expected_groups.items():
        native_group = _exact_mapping(provenance[group_id], "native surface group")
        if any(native_group.get(key) != value for key, value in group.items()):
            raise ValueError(f"surface group provenance differs for {group_id!r}")
        values = _exact_mapping(surface.integrals[group_id], "grouped surface integral")
        normal = _quantity(values.get("normal"), "normal group integral", "V^2")
        tangential = _quantity(
            values.get("tangential"),
            "tangential group integral",
            "V^2",
            nonnegative=False,
        )
        energy = _surface_energy_j(
            group["interface_kind"],
            group["film_thickness_m"],
            group["film_relative_permittivity"],
            group["substrate_relative_permittivity"],
            normal,
            tangential,
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
        surface_rows.append(
            {
                **group,
                "normal_integral_v2": normal,
                "tangential_integral_v2": tangential,
                "masked_area_m2": _quantity(
                    surface.masked_areas[group_id], "masked group area", "m^2"
                ),
                "energy_j": energy,
                "participation": energy / normalization_j,
                "assumptions": assumptions,
                # 1/Q contribution = p * tan(delta) (docs/eigenmode-epr.qmd).
                "inverse_q": inverse_quality_factor(
                    energy / normalization_j, loss_tangent
                ),
            }
        )
    return surface_rows


def _require_per_binding_keys(
    prepared: PreparedPlanarGeometry,
    surface: _SurfaceIntegrals,
    specs_by_id: Mapping[str, Any],
    selected_surfaces: set[str],
) -> None:
    """Older layout: integrals are keyed `<binding_id>@<margin>` for every margin."""

    expected_surface_keys = {
        f"{binding['binding_id']}@{margin:.17g}"
        for binding in prepared.surface_bindings
        if binding["contribution"]["contribution_id"] in selected_surfaces
        for margin in specs_by_id[binding["contribution"]["contribution_id"]].margins_um
        if specs_by_id[binding["contribution"]["contribution_id"]].interface_kind
        in _FILM_INTERFACES
    }
    if set(surface.integrals) != expected_surface_keys:
        raise ValueError("surface integrals do not match prepared contribution margins")
    if set(surface.masked_areas) != expected_surface_keys:
        raise ValueError("masked areas do not match prepared contribution margins")


def _per_binding_surface_rows(
    prepared: PreparedPlanarGeometry,
    bindings: Mapping[str, list[Mapping[str, Any]]],
    surface: _SurfaceIntegrals,
    normalization_j: float,
    selected_surfaces: set[str],
) -> list[dict[str, Any]]:
    """One row per (contribution, margin), summing that contribution's bindings."""

    surface_rows: list[dict[str, Any]] = []
    for spec in prepared.contributions:
        if spec.interface_kind == "MM" or spec.contribution_id not in selected_surfaces:
            continue
        local_bindings = bindings.get(spec.contribution_id)
        if not local_bindings:
            raise ValueError(f"missing prepared binding for {spec.contribution_id!r}")
        epsilon_s = _field_side_permittivity(spec, local_bindings)
        for margin in spec.margins_um:
            normal = 0.0
            tangential = 0.0
            binding_ids: list[str] = []
            for binding in local_bindings:
                binding_id = binding.get("binding_id")
                if not isinstance(binding_id, str) or not binding_id:
                    raise ValueError("surface binding id is invalid")
                key = f"{binding_id}@{margin:.17g}"
                values = _exact_mapping(
                    surface.integrals[key], f"surface integral {key!r}"
                )
                normal += _quantity(
                    values.get("normal"), "normal surface integral", "V^2"
                )
                tangential += _quantity(
                    values.get("tangential"),
                    "tangential surface integral",
                    "V^2",
                    nonnegative=False,
                )
                binding_ids.append(binding_id)
            energy = _surface_energy_j(
                spec.interface_kind,
                spec.film_thickness_m,
                spec.film_relative_permittivity,
                epsilon_s,
                normal,
                tangential,
            )
            surface_rows.append(
                {
                    "contribution_id": spec.contribution_id,
                    "interface_kind": spec.interface_kind,
                    "field_side": spec.field_side,
                    "binding_ids": binding_ids,
                    "margin_um": margin,
                    "normal_integral_v2": normal,
                    "tangential_integral_v2": tangential,
                    "masked_area_m2": _quantity(
                        surface.masked_areas[f"{binding_ids[-1]}@{margin:.17g}"]
                        if len(local_bindings) == 1
                        else {
                            "value": sum(
                                _quantity(
                                    surface.masked_areas[
                                        f"{item['binding_id']}@{margin:.17g}"
                                    ],
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
    return surface_rows


def _field_side_permittivity(
    spec: Any, local_bindings: list[Mapping[str, Any]]
) -> float:
    """The one relative permittivity on the field side of a contribution's bindings.

    SA reads the substrate material; MA and MS read the effective material.
    """

    epsilon_values: set[float] = set()
    for binding in local_bindings:
        contribution = _exact_mapping(
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
        material = _exact_mapping(binding.get(material_key), "surface binding material")
        if "permittivity" not in material:
            raise ValueError("field-side material lacks explicit relative permittivity")
        epsilon_values.add(
            _finite(
                material["permittivity"],
                "field-side relative permittivity",
            )
        )
    if len(epsilon_values) != 1:
        raise ValueError(
            f"contribution {spec.contribution_id!r} spans inconsistent field-side materials"
        )
    return epsilon_values.pop()


def _native_surface_reference_rows(
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
        if item.contribution_id in selected and item.interface_kind in _FILM_INTERFACES
    }
    references_requested = prepared.modeling == "solid" and bool(expected_contributions)
    if "native_surface_references" not in raw:
        # Keep previously serialized raw records readable without fabricating
        # reference values that were not part of their original evidence.
        return None
    references = _exact_mapping(
        raw["native_surface_references"], "native_surface_references"
    )
    if references_requested and not references:
        raise ValueError("selected Solid EPR contributions lack native face references")

    rows: list[dict[str, Any]] = []
    associated_contributions: set[str] = set()
    for reference_id, value in sorted(references.items()):
        record = _exact_mapping(value, "native surface reference")
        selection = _exact_mapping(
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
        normal = _quantity(record.get("normal"), "native face normal integral", "V^2")
        tangential = _quantity(
            record.get("tangential"),
            "native face tangential integral",
            "V^2",
            nonnegative=False,
        )
        area = _quantity(record.get("area"), "native face area integral", "m^2")
        expression_identities = _exact_mapping(
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
