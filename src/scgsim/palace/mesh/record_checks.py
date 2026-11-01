"""Validators for SGB physical-group records before they become mesh groups.

SGB is the semantic geometry builder (``scgsim.geometry.compiler``). It writes one
JSON record per physical group to ``04_export_physical_groups.json``. This module
owns the checks that a record is complete and self-consistent. It does not call
Gmsh and does not decide which group a record joins; ``groups.py`` does that.

Which error a user sees first depends on the order of checks, so callers keep the
call order and the checks here keep their internal order.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from .._values import optional_stripped_string

# Interface kinds written by SGB. MA = metal-vacuum (air), MS = metal-substrate,
# SA = substrate-vacuum (air), MS_MA = one surface carrying both the MS and MA
# aliases, AA = vacuum-vacuum, MM = metal-metal, SS = substrate-substrate.
STRUCTURED_INTERFACE_TYPES = {"MA", "MS", "SA", "MS_MA", "AA", "MM", "SS"}
STRUCTURED_FACE_KINDS = {
    "TOP": "top",
    "BOTTOM": "bottom",
    "SIDEWALL": "sidewall",
    "INTERFACE": "interface",
    "SHEET_CONTACT_CAP": "sheet_contact_cap",
}
# A surface record is "structured" when it carries any of these SGB fields;
# otherwise it is a plain named surface group with no solver meaning.
_STRUCTURED_SURFACE_FIELDS = frozenset(
    (
        "representation",
        "surface_id",
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
)


def required_string(value: Any, field: str) -> str:
    """Return the stripped string, or raise naming ``field``."""
    text = optional_stripped_string(value)
    if text is None:
        raise ValueError(f"{field} must be a non-empty string.")
    return text


def required_distinct_strings(value: Any, field: str, *, count: int) -> tuple[str, ...]:
    """Return exactly ``count`` distinct non-empty strings, or raise."""
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise ValueError(f"{field} must contain exactly {count} strings.")
    result = tuple(required_string(item, field) for item in value)
    if len(set(result)) != count:
        raise ValueError(f"{field} values must be distinct.")
    return result


def normalized_direction(value: Any, name: str) -> tuple[float, float, float]:
    """Return a lumped-port direction as a unit XY vector; z must be zero."""
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
    """Report whether ``value`` is a non-empty list/tuple of non-blank strings."""
    return (
        isinstance(value, (list, tuple))
        and bool(value)
        and all(isinstance(item, str) and item.strip() for item in value)
    )


def validate_structured_domain_boundary_record(
    name: str, record: Mapping[str, Any]
) -> None:
    """Require every SGB field a domain-boundary surface must carry."""
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
    if not optional_stripped_string(record.get("route")):
        raise ValueError(f"SGB domain boundary {name!r} has empty route.")
    if optional_stripped_string(record.get("representation")) is None:
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
    """Require the identity, representation and provenance fields of a conductor or interface surface."""
    required_strings = ("surface_id", "interface_type", "face_kind", "representation")
    missing = [
        field
        for field in required_strings
        if optional_stripped_string(record.get(field)) is None
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
    representation = optional_stripped_string(record.get("representation"))
    route = optional_stripped_string(record.get("route"))
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
        component_id = optional_stripped_string(record.get("conductor_component_id"))
        if component_id is not None:
            raise ValueError(
                f"SGB SA surface {name!r} must not carry conductor_component_id."
            )


def validate_surface_contribution_ledger(
    name: str,
    record: Mapping[str, Any],
    interface_type: str,
) -> None:
    """Validate additive side provenance without reconstructing historical rows.

    A surface may merge several source contributions (for example the metal-vacuum
    and metal-substrate sides of one sheet). Each source lists them in a ledger.
    The ledger kinds must add up to ``interface_type`` and the native aggregation
    block must describe how those kinds become surface-EPR (energy participation
    ratio) rows.
    """
    provenance = record.get("source_provenance")
    if not isinstance(provenance, Mapping):
        return
    sources = _surface_contribution_sources(provenance)
    if not sources:
        return
    contribution_identities: dict[str, str] = {}
    classifications: set[str] = set()
    for source in sources:
        _validate_ledger_source(name, source, contribution_identities, classifications)
    expected = set(interface_type.split("_"))
    if classifications != expected:
        raise ValueError(
            f"SGB surface {name!r} contribution kinds {sorted(classifications)!r} "
            f"do not match interface_type {interface_type!r}."
        )


def _validate_ledger_source(
    name: str,
    source: Mapping[str, Any],
    contribution_identities: dict[str, str],
    classifications: set[str],
) -> None:
    """Check one source's ledger entries, then its native aggregation block.

    ``contribution_identities`` and ``classifications`` are shared across sources
    and updated here, so a contribution repeated by two sources must be identical.
    """
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
    local_classifications = _ledger_classifications(
        name, ledger, contribution_identities, classifications
    )
    unique_local = tuple(dict.fromkeys(local_classifications))
    surface_epr_local = tuple(
        classification
        for classification in local_classifications
        if classification in {"MA", "MS", "SA"}
    )
    unique_surface_epr = tuple(dict.fromkeys(surface_epr_local))
    # Several entries of one surface-EPR kind share a single native row.
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


def _ledger_classifications(
    name: str,
    ledger: Sequence[Any],
    contribution_identities: dict[str, str],
    classifications: set[str],
) -> list[str]:
    """Validate each ledger entry and return its classifications in ledger order."""
    local_contribution_ids: set[str] = set()
    local_classifications: list[str] = []
    for contribution in ledger:
        if not isinstance(contribution, Mapping):
            raise TypeError(
                f"SGB surface {name!r} contribution entries must be mappings."
            )
        contribution_id = optional_stripped_string(contribution.get("contribution_id"))
        classification = optional_stripped_string(contribution.get("classification"))
        side = optional_stripped_string(contribution.get("side"))
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
    return local_classifications


def _surface_contribution_sources(
    provenance: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    """Find the provenance nodes that own a contribution ledger, descending through ``sources``."""
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
    """Require the provenance fields of a solver-active solution volume."""
    for field in ("representation", "source_provenance", "physical_attribute"):
        if field not in record:
            raise ValueError(f"SGB volume {name!r} misses {field}.")
    if not optional_stripped_string(record.get("representation")):
        raise ValueError(f"SGB volume {name!r} has empty representation.")
    if not isinstance(record.get("source_provenance"), Mapping):
        raise TypeError(f"SGB volume {name!r} needs mapping source_provenance.")
    if not isinstance(record.get("physical_attribute"), Mapping):
        raise TypeError(f"SGB volume {name!r} needs mapping physical_attribute.")


def structured_conductor_source_layer(name: str, record: Mapping[str, Any]) -> str:
    """Return the single conductor source layer a surface came from.

    Read directly from the provenance, or from its ``sources`` when they all agree."""
    provenance = record.get("source_provenance")
    if not isinstance(provenance, Mapping):
        raise TypeError(f"SGB surface {name!r} needs mapping source_provenance.")
    direct = optional_stripped_string(provenance.get("conductor_source_layer_name"))
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
        source_layer = optional_stripped_string(
            source.get("conductor_source_layer_name")
        )
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
    """Check the surface representation against what the geometry route allows.

    Route A may use sheets or cutout shells; Route B only cutout boundary shells.
    Solution-only interfaces (SA, AA, SS) always keep ``solution_surface``."""
    representation = optional_stripped_string(record.get("representation"))
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


def validate_surface_adjacency(
    name: str,
    record: Mapping[str, Any],
    interface_type: str,
    material_kinds: Mapping[str, str],
) -> None:
    """Check the material kinds on the sides of a surface against its interface type.

    The pairing is read from explicit stack materials, never from geometric
    orientation or material names."""
    adjacent_groups = _surface_local_adjacencies(record)
    if not adjacent_groups:
        raise ValueError(
            f"SGB surface {name!r} has no explicit adjacent solution volumes."
        )
    for adjacent in adjacent_groups:
        kinds: list[str] = []
        for identifier in adjacent:
            if not isinstance(identifier, str) or identifier not in material_kinds:
                raise ValueError(
                    f"SGB surface {name!r} adjacent solution id {identifier!r} "
                    "cannot resolve explicit material kind."
                )
            kinds.append(material_kinds[identifier])
        if interface_type == "MS_MA":
            if str(record.get("face_kind", "")).lower() != "interface" or sorted(
                kinds
            ) != ["dielectric", "vacuum"]:
                raise ValueError(
                    f"SGB MS_MA surface {name!r} requires face_kind=interface "
                    "and exactly dielectric/vacuum adjacency in every local source."
                )
        elif interface_type in {"MA", "MS", "SA"}:
            # The required material pairing is an explicit semantic compatibility
            # check, never a geometric orientation or material-name inference.
            expected = {"MA": "vacuum", "MS": "dielectric", "SA": "vacuum"}[
                interface_type
            ]
            if interface_type == "SA":
                if sorted(kinds) != ["dielectric", "vacuum"]:
                    raise ValueError(
                        f"SGB SA surface {name!r} requires exactly "
                        "dielectric/vacuum adjacency in every local source."
                    )
            elif expected not in kinds:
                raise ValueError(
                    f"SGB {interface_type} surface {name!r} lacks explicit adjacent "
                    f"{expected} solution material in a local source."
                )


def _surface_local_adjacencies(
    record: Mapping[str, Any],
) -> tuple[tuple[str, ...], ...]:
    """Read exact source-local adjacency without treating a group union as a pair."""
    provenance = record.get("source_provenance")
    members = (
        _surface_provenance_members(provenance)
        if isinstance(provenance, Mapping)
        else ()
    )
    sources = tuple(
        source
        for source in members
        if isinstance(source.get("surface_contribution_ledger"), (list, tuple))
    )
    if sources and len(sources) != len(members):
        raise ValueError(
            "SGB surface contribution provenance mixes ledgered and legacy sources."
        )
    local: list[tuple[str, ...]] = []
    for source in sources:
        ledger = source.get("surface_contribution_ledger")
        if not isinstance(ledger, (list, tuple)):
            continue
        identifiers: list[str] = []
        for contribution in ledger:
            if not isinstance(contribution, Mapping):
                continue
            effective = contribution.get("effective_domain_ids")
            if not isinstance(effective, (list, tuple)):
                continue
            for identifier in effective:
                if isinstance(identifier, str) and identifier not in identifiers:
                    identifiers.append(identifier)
        if identifiers:
            local.append(tuple(identifiers))
    if local:
        return tuple(local)
    adjacent = record.get("adjacent_solution_volume_ids")
    if not isinstance(adjacent, (list, tuple)) or not adjacent:
        return ()
    return (tuple(adjacent),)


def _surface_provenance_members(
    provenance: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    """Flatten nested provenance ``sources`` into the leaf provenance mappings."""
    sources = provenance.get("sources")
    if not isinstance(sources, (list, tuple)):
        return (provenance,)
    return tuple(
        member
        for source in sources
        if isinstance(source, Mapping)
        for member in _surface_provenance_members(source)
    )


def solution_material_kinds(
    records: Sequence[Any], stack: Mapping[str, Any]
) -> dict[str, str]:
    """Map SGB solution-volume stable IDs to explicit stack material kinds.

    The kind (vacuum or dielectric) is read from the stack, never inferred from
    geometry or names. Surface checks later use it to decide which side of an
    interface is substrate and which is vacuum.
    """
    materials = stack.get("materials")
    if not isinstance(materials, Mapping):
        raise TypeError("stack must carry explicit materials while meshing.")
    result: dict[str, str] = {}
    for record in records:
        if not isinstance(record, Mapping) or not is_solver_active_volume(record):
            continue
        for material_id, material_kind, volume_id, owner_id in _volume_material_rows(
            record
        ):
            _check_material_row(
                materials, material_id, material_kind, volume_id, owner_id
            )
            for stable_id in (volume_id, owner_id):
                previous = result.setdefault(stable_id, material_kind)
                if previous != material_kind:
                    raise ValueError(
                        f"Structured solution id {stable_id!r} has conflicting material kinds."
                    )
    if not result:
        raise ValueError(
            "SGB sidecar has no explicit solution-volume material records."
        )
    return result


def _volume_material_rows(
    record: Mapping[str, Any],
) -> list[tuple[Any, Any, Any, Any]]:
    """Pair material ids/kinds with volume and owner ids of one volume record.

    A record may list one material for many volumes (broadcast) or one material
    per volume; any other cardinality is rejected.
    """
    physical = record.get("physical_attribute")
    provenance = record.get("source_provenance")
    ids = physical.get("material_ids") if isinstance(physical, Mapping) else None
    kinds = physical.get("material_kinds") if isinstance(physical, Mapping) else None
    volume_ids = (
        provenance.get("volume_ids") if isinstance(provenance, Mapping) else None
    )
    owner_ids = record.get("owner_semantic_ids")
    if not all(
        isinstance(value, (list, tuple)) and value
        for value in (ids, kinds, volume_ids, owner_ids)
    ):
        raise ValueError(
            "SGB solution volume needs explicit material_ids/material_kinds, provenance.volume_ids, and owner_semantic_ids."
        )
    if not (
        (len(ids) == len(kinds) == len(volume_ids) == len(owner_ids))
        or (len(ids) == 1 and len(kinds) == 1 and len(volume_ids) == len(owner_ids))
    ):
        raise ValueError(
            "SGB solution volume material kind and stable-id cardinalities must agree."
        )
    is_broadcast = len(ids) == 1
    return list(
        zip(
            ids if not is_broadcast else [ids[0]] * len(volume_ids),
            kinds if not is_broadcast else [kinds[0]] * len(volume_ids),
            volume_ids,
            owner_ids,
            strict=not is_broadcast,
        )
    )


def _check_material_row(
    materials: Mapping[str, Any],
    material_id: Any,
    material_kind: Any,
    volume_id: Any,
    owner_id: Any,
) -> None:
    """Require non-empty ids and a record kind that matches the stack material."""
    if not all(
        isinstance(value, str) and value
        for value in (material_id, material_kind, volume_id, owner_id)
    ):
        raise ValueError(
            "SGB solution volume stable ids, material ids, and material kinds must be non-empty strings."
        )
    material = materials.get(material_id)
    stack_kind = material.get("kind") if isinstance(material, Mapping) else None
    if stack_kind not in {"vacuum", "dielectric"}:
        raise ValueError(
            "SGB solution volume material must resolve vacuum or dielectric kind."
        )
    if material_kind not in {"vacuum", "dielectric"}:
        raise ValueError(
            "SGB solution volume material_kinds must be vacuum or dielectric."
        )
    if material_kind != stack_kind:
        raise ValueError(
            "SGB solution volume material_kind disagrees with explicit stack material kind."
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
        route = optional_stripped_string(record.get("route"))
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
    return routes.pop()


def is_solver_active_volume(record: Mapping[str, Any]) -> bool:
    """Report whether a record is an explicit solver-active material volume."""
    return (
        int(record.get("dimension", 0) or 0) == 3
        and record.get("role") == "material_volume"
        and record.get("interface_type") == "material_volume"
        and record.get("solver_use", "solver_active") == "solver_active"
    )


def has_structured_record_marker(record: Mapping[str, Any]) -> bool:
    """Report whether a surface record carries any SGB structured field."""
    return any(field in record for field in _STRUCTURED_SURFACE_FIELDS)
