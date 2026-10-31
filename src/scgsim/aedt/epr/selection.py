"""Which surface bindings an EPR analysis integrates over, and how they group.

Owns: the pure rules that turn prepared surface bindings into owner groups. These
rules are shared by the request contract and the analysis, so both agree on
group ids.
Does not own: authoring or evaluating the integrals (`cache.py`).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

from scgsim.aedt._data import (
    detached,
)
from scgsim.aedt.specs.epr import (
    EprAnalysisRequest,
    surface_evaluations,
)
from scgsim.aedt.specs.geometry import (
    PreparedPlanarGeometry,
)


def _finite(value: Any, name: str, *, nonnegative: bool = True) -> float:
    """`value` as a finite float (non-negative by default); reject bool and text."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        qualifier = "finite non-negative" if nonnegative else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _quantity(value: Any, name: str, unit: str, *, nonnegative: bool = True) -> float:
    """The number in a `{"value", "unit"}` record, which must use exactly `unit`."""

    item = _exact_mapping(value, name)
    if set(item) != {"value", "unit"} or item.get("unit") != unit:
        raise ValueError(f"{name} must use canonical unit {unit!r}")
    return _finite(item.get("value"), name, nonnegative=nonnegative)


def _exact_mapping(value: Any, name: str) -> Mapping[str, Any]:
    """`value` itself if it is a mapping with only string keys."""

    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed mapping")
    return value


def surface_integral_groups(
    prepared: PreparedPlanarGeometry,
    request: EprAnalysisRequest | None,
) -> tuple[dict[str, Any], ...]:
    """Partition physical bindings only where one exact film coefficient applies.

    Surface bindings that share an owner, interface kind (MA, MS or SA), film
    thickness, film permittivity, substrate permittivity and margin get the same
    energy coefficient, so their field integrals can be summed into one group and
    weighted once (formulas: docs/specs/aedt-runtime.qmd). MM (metal-metal)
    bindings carry no thin-film energy and are skipped.
    """

    selected = (
        {item.contribution_id for item in prepared.contributions}
        if request is None or request.surface_contribution_ids is None
        else set(request.surface_contribution_ids)
    )
    specs = {item.contribution_id: item for item in prepared.contributions}
    evaluation_policy = prepared.source.get("surface_evaluation_policy")
    groups = _members_by_signature(prepared, selected, specs, evaluation_policy)
    result: list[dict[str, Any]] = []
    # Sorted by repr so group order does not depend on binding order.
    for signature, members in sorted(groups.items(), key=lambda item: repr(item[0])):
        result.append(_build_group(signature, members, specs, evaluation_policy))
    if len({item["group_id"] for item in result}) != len(result):
        raise ValueError("surface group identities are not unique")
    return tuple(result)


def _members_by_signature(
    prepared: PreparedPlanarGeometry,
    selected: set[str],
    specs: Mapping[str, Any],
    evaluation_policy: Any,
) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    """Group the selected bindings by the signature that fixes their coefficient."""

    domains = {
        item["semantic_id"]: item for item in prepared.source["solution_regions"]
    }
    conductor_ids = {item["semantic_id"] for item in prepared.source["conductors"]}
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for binding in prepared.surface_bindings:
        record = _exact_mapping(binding["contribution"], "surface provenance")
        contribution_id = record["contribution_id"]
        if contribution_id not in selected:
            continue
        spec = specs[contribution_id]
        if spec.interface_kind == "MM":
            continue
        source_owners = tuple(record["source_owner_ids"])
        owner_id = _physical_owner(
            prepared,
            binding,
            spec.interface_kind,
            source_owners,
            domains,
            conductor_ids,
        )
        # The substrate side of an MS or SA interface is the film's second medium.
        material = (
            binding["substrate_material"]
            if spec.interface_kind in {"MS", "SA"}
            else binding["effective_material"]
        )
        epsilon_s = _finite(material["permittivity"], "group substrate permittivity")
        if epsilon_s <= 0.0:
            raise ValueError("group substrate permittivity must be positive")
        signature = (
            owner_id,
            spec.interface_kind,
            spec.film_thickness_m,
            spec.film_relative_permittivity,
            epsilon_s,
        )
        member = {
            "binding_id": binding["binding_id"],
            "contribution_id": contribution_id,
            "source_polygon_id": spec.source_polygon_id,
            "source_owner_ids": list(source_owners),
            "field_side": spec.field_side,
            "effective_domain_id": binding["effective_domain_id"],
            "effective_material_id": binding["effective_material_id"],
            "substrate_domain_id": binding["substrate_domain_id"],
            "geometry_ref": detached(binding["geometry_ref"]),
        }
        for evaluation_kind, margin in surface_evaluations(
            spec.margins_um, policy=evaluation_policy
        ):
            evaluation_key = (
                (evaluation_kind, float(margin))
                if evaluation_policy is not None
                else (float(margin),)
            )
            groups.setdefault((*signature, *evaluation_key), []).append(member)
    return groups


def _physical_owner(  # noqa: PLR0913 -- the owner lookups are the inputs.
    prepared: PreparedPlanarGeometry,
    binding: Mapping[str, Any],
    interface_kind: str,
    source_owners: tuple[Any, ...],
    domains: Mapping[str, Any],
    conductor_ids: set[str],
) -> str:
    """The dielectric (SA) or conductor (MA, MS) that physically owns the binding."""

    if interface_kind == "SA":
        owner_id = binding["substrate_domain_id"]
        if (
            owner_id not in source_owners
            or owner_id not in domains
            or prepared.source["materials"][domains[owner_id]["material_id"]]["kind"]
            != "dielectric"
        ):
            raise ValueError("SA group lacks its physical dielectric owner")
        return owner_id
    owner_id = binding["owner_semantic_id"]
    if owner_id not in source_owners or owner_id not in conductor_ids:
        raise ValueError("metal group lacks its physical conductor owner")
    return owner_id


def _build_group(
    signature: tuple[Any, ...],
    members: list[dict[str, Any]],
    specs: Mapping[str, Any],
    evaluation_policy: Any,
) -> dict[str, Any]:
    """Turn one signature and its members into a group with a stable hashed id."""

    member_assumptions = {
        (
            specs[member["contribution_id"]].loss_tangent,
            specs[member["contribution_id"]].source,
            specs[member["contribution_id"]].preset,
        )
        for member in members
    }
    if len(member_assumptions) != 1:
        raise ValueError("one physical surface group has conflicting film provenance")
    owner_id, interface_kind, thickness, film_epsilon, substrate_epsilon = signature[:5]
    if evaluation_policy is None:
        evaluation_kind = None
        margin = signature[5]
    else:
        evaluation_kind, margin = signature[5:]
    members.sort(key=lambda item: (item["binding_id"], item["contribution_id"]))
    if len({item["binding_id"] for item in members}) != len(members):
        raise ValueError("surface group repeats a physical binding")
    group_identity = {
        "owner_id": owner_id,
        "interface_kind": interface_kind,
        "film_thickness_m": thickness,
        "film_relative_permittivity": film_epsilon,
        "substrate_relative_permittivity": substrate_epsilon,
        "margin_um": margin,
    }
    if evaluation_kind is not None:
        group_identity["evaluation_kind"] = evaluation_kind
    group_id = (
        "surface_group_"
        + hashlib.sha256(
            json.dumps(group_identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:24]
    )
    return {"group_id": group_id, **group_identity, "members": members}
