"""Portable Palace EPR records, retained historical provenance and immutable serialization."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from scgsim.semantics.epr import film_assumptions, inverse_quality_factor


def checked_number(value: Any, name: str, *, positive: bool = False) -> float:
    """Require a finite nonnegative EPR value, optionally excluding zero."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or (positive and value == 0)
    ):
        raise ValueError(
            f"{name} must be finite and {'positive' if positive else 'non-negative'}"
        )
    return float(value)


def surface_distance_band(value: Any) -> dict[str, Any]:
    """Validate retained geometric-band provenance in micrometres without rebuilding geometry."""
    if not isinstance(value, Mapping):
        raise TypeError("Palace geometric-band provenance must be a mapping")
    required = {
        "parent_surface_id",
        "source_entity_id",
        "source_polygon_id",
        "lower_um",
        "upper_um",
        "band_index",
        "component_index",
        "precision_um",
    }
    missing = required.difference(value)
    if missing:
        raise ValueError(
            f"Palace geometric-band provenance is missing fields: {sorted(missing)}"
        )
    canonical_fields = {"distances_um", "boundary_segments"}
    if canonical_fields.issubset(value):
        if not isinstance(value["distances_um"], (list, tuple)):
            raise TypeError("Palace geometric-band distances_um must be a sequence")
        if not isinstance(value["boundary_segments"], (list, tuple)):
            raise TypeError(
                "Palace geometric-band boundary_segments must be a sequence"
            )
    elif not canonical_fields.isdisjoint(value) or "origin_segments" not in value:
        raise ValueError("Palace geometric-band source-boundary schema is invalid")
    elif not isinstance(value["origin_segments"], (list, tuple)):
        raise TypeError("Palace geometric-band origin_segments must be a sequence")
    _validate_band_scalars(value)
    return dict(value)


def _plain(value: Any) -> Any:
    """Convert immutable EPR mappings and sequences into JSON-compatible containers."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"unsupported EPR value {type(value).__name__}")


def _freeze(value: Any, *, verified: Any = None) -> Any:
    """Detach immutable EPR containers, reusing already verified nested values."""
    if verified is not None and value is verified:
        return verified
    if isinstance(value, Mapping):
        prior = verified if isinstance(verified, Mapping) else {}
        return MappingProxyType(
            {
                str(key): _freeze(item, verified=prior.get(str(key)))
                for key, item in value.items()
            }
        )
    if isinstance(value, (tuple, list)):
        prior = verified if isinstance(verified, (tuple, list)) else ()
        return tuple(
            _freeze(item, verified=prior[index] if index < len(prior) else None)
            for index, item in enumerate(value)
        )
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"unsupported EPR value {type(value).__name__}")


@dataclass(frozen=True)
class PalaceEprResult:
    """Portable exact-mode snapshot without a native run or mutable Sim handle."""

    rows: tuple[Mapping[str, Any], ...]
    provenance: Mapping[str, Any]
    _verified_source: PalaceEprResult | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        rows = tuple(self.rows)
        if not rows or any(not isinstance(row, Mapping) for row in rows):
            raise ValueError("Palace EPR requires non-empty row mappings")
        identities: set[int] = set()
        group_sources: dict[str, tuple[Any, ...]] = {}
        for row in rows:
            mode = row.get("mode")
            if type(mode) is not int or mode <= 0 or mode in identities:
                raise ValueError("Palace EPR modes must be unique positive integers")
            identities.add(mode)
            checked_number(row.get("frequency_hz"), "frequency_hz", positive=True)
            checked_number(
                row.get("normalization_energy_j"),
                "normalization_energy_j",
                positive=True,
            )
            for name in ("surface", "bulk", "junction"):
                if not isinstance(row.get(name), (list, tuple)) or any(
                    not isinstance(item, Mapping) for item in row[name]
                ):
                    raise TypeError(f"Palace EPR {name} must contain mappings")
            for surface in row["surface"]:
                _validate_surface_result(surface, group_sources)
            for bulk in row["bulk"]:
                if not isinstance(bulk.get("domain_id"), str) or not bulk["domain_id"]:
                    raise ValueError("Palace EPR bulk domain_id is invalid")
                checked_number(bulk.get("participation"), "bulk participation")
            _validate_ports(row["junction"])
        if not isinstance(self.provenance, Mapping) or not isinstance(
            self.provenance.get("handoff_id"), str
        ):
            raise ValueError("Palace EPR provenance lacks handoff identity")
        if self._verified_source is not None and not isinstance(
            self._verified_source, PalaceEprResult
        ):
            raise TypeError("verified source must be a PalaceEprResult")
        prior_rows = (
            self._verified_source.rows if self._verified_source is not None else ()
        )
        object.__setattr__(
            self,
            "rows",
            tuple(
                _freeze(
                    row, verified=prior_rows[index] if index < len(prior_rows) else None
                )
                for index, row in enumerate(rows)
            ),
        )
        object.__setattr__(
            self,
            "provenance",
            _freeze(
                self.provenance,
                verified=self._verified_source.provenance
                if self._verified_source is not None
                else None,
            ),
        )

    def to_payload(self) -> dict[str, Any]:
        """Serialize mode rows and provenance under the detached EPR schema."""
        return {
            "schema_version": "scgsim.palace.epr-result.v1",
            "rows": _plain(self.rows),
            "provenance": _plain(self.provenance),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> PalaceEprResult:
        """Validate the schema before reconstructing an immutable detached EPR result."""
        if (
            not isinstance(payload, Mapping)
            or set(payload) != {"schema_version", "rows", "provenance"}
            or payload.get("schema_version") != "scgsim.palace.epr-result.v1"
        ):
            raise ValueError("Palace EPR payload is not canonical")
        return cls(tuple(payload["rows"]), payload["provenance"])


def _validate_surface_result(
    surface: Any, group_sources: dict[str, tuple[Any, ...]]
) -> None:
    """Validate film values and preserve original-group identity across modes."""
    for name in (
        "group_id",
        "surface_id",
        "interface_kind",
        "field_side",
        "evaluation_kind",
    ):
        if not isinstance(surface.get(name), str) or not surface[name]:
            raise ValueError(f"Palace EPR surface {name} is missing")
    if surface["interface_kind"] not in {"MA", "MS", "SA"} or surface[
        "evaluation_kind"
    ] not in {
        "unmasked_baseline",
        "requested_margin",
        "geometric_band",
    }:
        raise ValueError("Palace EPR surface identity is invalid")
    is_band = surface["evaluation_kind"] == "geometric_band"
    if is_band:
        surface_distance_band(surface.get("surface_distance_band"))
        if surface.get("margin_um") is not None:
            raise ValueError("Palace geometric-band result must not claim a margin")
    else:
        checked_number(surface.get("margin_um"), "surface margin")
    if not isinstance(surface.get("members"), Mapping):
        raise TypeError("Palace EPR source provenance must be a mapping")
    owners = surface.get("owner_semantic_ids")
    if (
        not isinstance(owners, (list, tuple))
        or not owners
        or any(not isinstance(owner, str) or not owner for owner in owners)
    ):
        raise ValueError("Palace EPR surface owners are invalid")
    checked_number(surface.get("original_participation"), "original participation")
    checked_number(surface.get("participation"), "participation")
    assumption = film_assumptions(surface.get("assumptions"), allow_unknown=True)
    original = film_assumptions(surface.get("original_assumptions"), allow_unknown=True)
    expected_q = inverse_quality_factor(
        surface["participation"], assumption["loss_tangent"]
    )
    actual_q = surface.get("inverse_q")
    if (expected_q is None) != (actual_q is None) or (
        expected_q is not None
        and not math.isclose(
            checked_number(actual_q, "inverse_q"),
            expected_q,
            rel_tol=1e-12,
            abs_tol=0.0,
        )
    ):
        raise ValueError("Palace EPR inverse_q disagrees with participation and loss")
    source = (
        surface["surface_id"],
        surface["interface_kind"],
        tuple(owners),
        tuple(sorted(original.items())),
    )
    prior = group_sources.setdefault(surface["group_id"], source)
    if source != prior:
        raise ValueError(
            "Palace EPR original group identity has conflicting provenance"
        )


def _validate_band_scalars(value: Any) -> None:
    """Validate retained historical geometric-band scalar identities and units."""
    for name in ("parent_surface_id", "source_entity_id"):
        if not isinstance(value[name], str) or not value[name]:
            raise ValueError(f"Palace geometric-band {name} must be non-empty text")
    source_polygon_id = value["source_polygon_id"]
    if source_polygon_id is not None and not isinstance(source_polygon_id, str):
        raise TypeError("Palace geometric-band source_polygon_id must be text or None")
    if source_polygon_id == "":
        raise ValueError("Palace geometric-band source_polygon_id must not be empty")
    for name in ("band_index", "component_index"):
        if type(value[name]) is not int or value[name] < 0:
            raise ValueError(f"Palace geometric-band {name} must be non-negative")
    checked_number(value["lower_um"], "geometric-band lower_um")
    if value["upper_um"] is not None:
        checked_number(value["upper_um"], "geometric-band upper_um")
    checked_number(value["precision_um"], "geometric-band precision_um", positive=True)


def _validate_ports(ports: Any) -> None:
    """Retain signed finite Palace port participation and exact port identity."""
    for port in ports:
        if not isinstance(port.get("port_name"), str) or not port["port_name"]:
            raise ValueError("Palace EPR port_name is invalid")
        value = port.get("participation")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError("Palace EPR signed port participation is invalid")
