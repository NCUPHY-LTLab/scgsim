"""Validate structured compiler record identities and historical contribution provenance."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

from .provenance import (
    STRUCTURED_FACE_KINDS,
    STRUCTURED_INTERFACE_TYPES,
    STRUCTURED_SURFACE_FIELDS,
)


def sgb_records_route_context(
    records: Sequence[Any], expected_route: str | None = None
) -> Literal["A", "B"]:
    """Require one explicit structured SGB Route A/B context."""
    routes: set[str] = set()
    if expected_route is not None:
        expected_route = expected_route.upper()
        if expected_route not in {"A", "B"}:
            raise ValueError("expected_route must be 'A' or 'B'.")
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("SGB physical group records must be JSON objects.")
        route = optional_string(record.get("route"))
        if route not in {"A", "B"}:
            raise ValueError(
                "SGB physical-group records require explicit route A or B."
            )
        routes.add(route)
    if expected_route is not None and routes != {expected_route}:
        raise ValueError(
            f"SGB physical-group records must all use route {expected_route}, got {sorted(routes)}."
        )
    if len(routes) != 1:
        raise ValueError("SGB physical-group records must all use a single route.")
    return cast(Literal["A", "B"], routes.pop())


def is_solver_active_volume(record: Mapping[str, Any]) -> bool:
    """Recognize only explicitly active solution material volumes."""
    return (
        int(record.get("dimension", 0) or 0) == 3
        and record.get("role") == "material_volume"
        and record.get("interface_type") == "material_volume"
        and record.get("solver_use", "solver_active") == "solver_active"
    )


def has_structured_record_marker(record: Mapping[str, Any]) -> bool:
    """Recognize structured semantic fields without inferring a surface role."""
    return any(field in record for field in STRUCTURED_SURFACE_FIELDS)


def optional_string(value: Any) -> str | None:
    """Return trimmed text when present, else None."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def required_string(value: Any, field: str) -> str:
    """Require one nonempty string using the established string normalization."""
    text = optional_string(value)
    if text is None:
        raise ValueError(f"{field} must be a non-empty string.")
    return text


def required_distinct_strings(value: Any, field: str, *, count: int) -> tuple[str, ...]:
    """Require the declared number of distinct identity strings."""
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise ValueError(f"{field} must contain exactly {count} strings.")
    result = tuple(required_string(item, field) for item in value)
    if len(set(result)) != count:
        raise ValueError(f"{field} values must be distinct.")
    return result


def normalized_direction(value: Any, name: str) -> tuple[float, float, float]:
    """Validate the authored normalized XY port direction."""
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, (list, tuple))
        or len(value) != 3
    ):
        raise ValueError(f"SGB lumped-port sheet {name!r} requires a 3D direction.")
    direction = tuple(float(item) for item in value)
    length = math.hypot(direction[0], direction[1])
    if (
        not all(math.isfinite(item) for item in direction)
        or not math.isclose(direction[2], 0.0, abs_tol=1e-12)
        or not math.isclose(length, 1.0, rel_tol=1e-12, abs_tol=1e-12)
    ):
        raise ValueError(
            f"SGB lumped-port sheet {name!r} direction must be normalized finite XY."
        )
    return (direction[0], direction[1], 0.0)


def is_nonempty_string_sequence(value: Any) -> bool:
    """Recognize an explicit nonempty sequence of identity strings."""
    return (
        isinstance(value, (list, tuple))
        and bool(value)
        and all(isinstance(item, str) and item.strip() for item in value)
    )


def validate_structured_domain_boundary_record(
    name: str, record: Mapping[str, Any]
) -> None:
    """Require the source identity fields carried by a domain boundary."""
    required = (
        "surface_id",
        "representation",
        "interface_type",
        "contact_kind",
        "face_kind",
        "owner_semantic_ids",
        "adjacent_solution_volume_ids",
        "conductor_component_id",
        "net_id",
        "equipotential_id",
        "source_provenance",
        "physical_attribute",
    )
    missing = [field for field in required if field not in record]
    if missing:
        raise ValueError(
            f"SGB domain boundary {name!r} misses fields: {', '.join(missing)}."
        )
    if not optional_string(record.get("route")):
        raise ValueError(f"SGB domain boundary {name!r} has empty route.")
    if optional_string(record.get("representation")) is None:
        raise ValueError(f"SGB domain boundary {name!r} has empty representation.")
    if not is_nonempty_string_sequence(record.get("owner_semantic_ids")):
        raise ValueError(f"SGB domain boundary {name!r} needs owner_semantic_ids.")
    if not is_nonempty_string_sequence(record.get("adjacent_solution_volume_ids")):
        raise ValueError(
            f"SGB domain boundary {name!r} needs adjacent_solution_volume_ids."
        )
    for field in ("source_provenance", "physical_attribute"):
        if not isinstance(record.get(field), Mapping):
            raise TypeError(f"SGB domain boundary {name!r} needs mapping {field}.")


def validate_structured_surface_record(name: str, record: Mapping[str, Any]) -> None:
    """Require complete structured surface identity and representation."""
    required_strings = ("surface_id", "interface_type", "face_kind", "representation")
    missing = [
        field
        for field in required_strings
        if optional_string(record.get(field)) is None
    ]
    missing.extend(
        field
        for field in (
            "contact_kind",
            "conductor_component_id",
            "net_id",
            "equipotential_id",
        )
        if field not in record
    )
    if missing:
        raise ValueError(
            f"SGB surface {name!r} misses structured fields: {', '.join(missing)}."
        )
    representation = optional_string(record.get("representation"))
    route = optional_string(record.get("route"))
    if representation is None:
        raise ValueError(f"SGB surface {name!r} has empty representation.")
    if route not in {"A", "B"}:
        raise ValueError(f"SGB surface {name!r} has invalid route {route!r}.")
    interface_type = str(record.get("interface_type")).upper()
    if interface_type not in STRUCTURED_INTERFACE_TYPES:
        raise ValueError(
            f"SGB surface {name!r} has unsupported interface_type {interface_type!r}."
        )
    if not is_nonempty_string_sequence(record.get("owner_semantic_ids")):
        raise ValueError(f"SGB surface {name!r} needs exact owner_semantic_ids.")
    if not is_nonempty_string_sequence(record.get("adjacent_solution_volume_ids")):
        raise ValueError(
            f"SGB surface {name!r} needs exact adjacent_solution_volume_ids."
        )
    for field in ("source_provenance", "physical_attribute"):
        if not isinstance(record.get(field), Mapping):
            raise TypeError(f"SGB surface {name!r} needs mapping {field}.")
    face_kind = str(record.get("face_kind")).upper()
    if face_kind not in STRUCTURED_FACE_KINDS:
        raise ValueError(
            f"SGB surface {name!r} has invalid face_kind {record.get('face_kind')!r}."
        )
    if interface_type == "SA":
        component_id = optional_string(record.get("conductor_component_id"))
        if component_id is not None:
            raise ValueError(
                f"SGB SA surface {name!r} must not carry conductor_component_id."
            )


def validate_surface_contribution_ledger(
    name: str,
    record: Mapping[str, Any],
    interface_type: str,
) -> None:
    """Validate additive side provenance without reconstructing historical rows."""
    provenance = record.get("source_provenance")
    if not isinstance(provenance, Mapping):
        return
    sources = _surface_contribution_sources(provenance)
    if not sources:
        return
    contribution_identities: dict[str, str] = {}
    classifications: set[str] = set()
    for source in sources:
        ledger = source.get("surface_contribution_ledger")
        aggregation = source.get("native_aggregation")
        mask_support = source.get("mask_support")
        if (
            not isinstance(ledger, (list, tuple))
            or not ledger
            or source.get("surface_contribution_schema")
            != "scgsim.surface-contributions.v1"
            or not isinstance(aggregation, Mapping)
            or not isinstance(mask_support, Mapping)
        ):
            raise ValueError(
                f"SGB surface {name!r} has incomplete contribution provenance."
            )
        local_contribution_ids: set[str] = set()
        local_classifications: list[str] = []
        for contribution in ledger:
            if not isinstance(contribution, Mapping):
                raise TypeError(
                    f"SGB surface {name!r} contribution entries must be mappings."
                )
            contribution_id = optional_string(contribution.get("contribution_id"))
            classification = optional_string(contribution.get("classification"))
            side = optional_string(contribution.get("side"))
            if (
                contribution_id is None
                or contribution_id in local_contribution_ids
                or classification not in {"MA", "MS", "SA", "MM", "SS", "AA"}
                or side is None
                or not isinstance(contribution.get("snapshot_reference"), Mapping)
            ):
                raise ValueError(
                    f"SGB surface {name!r} has invalid or duplicate contribution identity."
                )
            local_contribution_ids.add(contribution_id)
            identity = _serialized_contribution_identity(contribution)
            previous = contribution_identities.setdefault(contribution_id, identity)
            if previous != identity:
                raise ValueError(
                    f"SGB surface {name!r} has conflicting repeated contribution "
                    f"identity {contribution_id!r}."
                )
            classifications.add(classification)
            local_classifications.append(classification)
        unique_local = tuple(dict.fromkeys(local_classifications))
        surface_epr_local = tuple(
            classification
            for classification in local_classifications
            if classification in {"MA", "MS", "SA"}
        )
        unique_surface_epr = tuple(dict.fromkeys(surface_epr_local))
        aggregate_once = len(surface_epr_local) > 1 and len(unique_surface_epr) == 1
        local_sides = tuple(
            contribution.get("side")
            for contribution in ledger
            if isinstance(contribution, Mapping)
        )
        expected_policy = (
            "semantic_only_no_native_row"
            if not unique_surface_epr
            else "shared_native_surface_once"
            if aggregate_once
            else "one_row_per_interface_kind"
        )
        expected_positive_mask_support = bool(unique_surface_epr) and not aggregate_once
        if (
            aggregation.get("policy") != expected_policy
            or aggregation.get("contribution_count") != len(ledger)
            or tuple(aggregation.get("interface_kinds", ())) != unique_local
            or tuple(aggregation.get("surface_epr_interface_kinds", ()))
            != unique_surface_epr
            or aggregation.get("native_row_count")
            != (1 if aggregate_once else len(unique_surface_epr))
            or aggregation.get("side_resolved_numeric_output") is not False
            or mask_support.get("zero_inset_supported") is not True
            or mask_support.get("positive_inset_supported")
            is not expected_positive_mask_support
            or tuple(mask_support.get("sides", ())) != local_sides
            or (
                aggregate_once
                and (len(ledger) != 2 or set(local_sides) != {"bottom", "top"})
            )
        ):
            raise ValueError(
                f"SGB surface {name!r} contribution aggregation is inconsistent."
            )
    expected = set(interface_type.split("_"))
    if classifications != expected:
        raise ValueError(
            f"SGB surface {name!r} contribution kinds {sorted(classifications)!r} "
            f"do not match interface_type {interface_type!r}."
        )


def _surface_contribution_sources(
    provenance: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    if "surface_contribution_ledger" in provenance:
        return (provenance,)
    sources = provenance.get("sources", ())
    if not isinstance(sources, (list, tuple)):
        return ()
    return tuple(
        contribution_source
        for source in sources
        if isinstance(source, Mapping)
        for contribution_source in _surface_contribution_sources(source)
    )


def validate_structured_volume_record(name: str, record: Mapping[str, Any]) -> None:
    """Require explicit volume representation and source mappings."""
    for field in ("representation", "source_provenance", "physical_attribute"):
        if field not in record:
            raise ValueError(f"SGB volume {name!r} misses {field}.")
    if not optional_string(record.get("representation")):
        raise ValueError(f"SGB volume {name!r} has empty representation.")
    if not isinstance(record.get("source_provenance"), Mapping):
        raise TypeError(f"SGB volume {name!r} needs mapping source_provenance.")
    if not isinstance(record.get("physical_attribute"), Mapping):
        raise TypeError(f"SGB volume {name!r} needs mapping physical_attribute.")


def structured_conductor_source_layer(name: str, record: Mapping[str, Any]) -> str:
    """Resolve one exact conductor source layer from direct or grouped provenance."""
    provenance = record.get("source_provenance")
    if not isinstance(provenance, Mapping):
        raise TypeError(f"SGB surface {name!r} needs mapping source_provenance.")
    direct = optional_string(provenance.get("conductor_source_layer_name"))
    if direct is not None:
        return direct
    sources = provenance.get("sources")
    if not isinstance(sources, (list, tuple)) or not sources:
        raise ValueError(
            f"SGB surface {name!r} needs exact conductor source-layer provenance."
        )

    source_layers: set[str] = set()
    for source in sources:
        if not isinstance(source, Mapping):
            raise TypeError(f"SGB surface {name!r} has invalid provenance source.")
        source_layer = optional_string(source.get("conductor_source_layer_name"))
        if source_layer is None:
            raise ValueError(
                f"SGB surface {name!r} lacks conductor_source_layer_name in source provenance."
            )
        source_layers.add(source_layer)

    if len(source_layers) != 1:
        raise ValueError(
            f"SGB surface {name!r} has ambiguous conductor source-layer provenance."
        )
    return source_layers.pop()


def _serialized_contribution_identity(contribution: Mapping[str, Any]) -> str:
    """Canonicalize one complete serialized EvidenceResult transport record."""
    return json.dumps(
        dict(contribution),
        sort_keys=True,
        separators=(",", ":"),
    )


def validate_route_representation(
    *, name: str, record: Mapping[str, Any], route: str
) -> None:
    """Validate the declared surface representation for its geometry route."""
    representation = optional_string(record.get("representation"))
    interface_type = str(record.get("interface_type", "")).upper()
    if interface_type in {"SA", "AA", "SS"}:
        if representation != "solution_surface":
            raise ValueError(
                f"SGB solution interface {name!r} must retain solution_surface representation."
            )
        return
    allowed = (
        {"surface_sheet", "cutout_boundary_shell"}
        if route == "A"
        else {"cutout_boundary_shell"}
    )
    if representation not in allowed:
        raise ValueError(
            f"SGB Route-{route} surface {name!r} representation {representation!r} is incompatible."
        )
