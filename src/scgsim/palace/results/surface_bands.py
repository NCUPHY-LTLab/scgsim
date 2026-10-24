"""Offline totals over complete, source-declared geometric surface bands."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from scgsim.geometry.models.surface_bands import SurfaceDistanceBandSpec

from .epr import PalaceEprResult, _freeze, _number, _plain


@dataclass(frozen=True)
class SurfaceBandSelection:
    """One exact parent, interface and recorded field side to include."""

    parent_surface_id: str
    interface_kind: str
    field_side: str

    def __post_init__(self) -> None:
        if not isinstance(self.parent_surface_id, str) or not self.parent_surface_id:
            raise ValueError("parent_surface_id must be explicit nonempty text")
        if self.interface_kind not in {"MA", "MS", "SA"}:
            raise ValueError("interface_kind must be MA, MS, or SA")
        if not isinstance(self.field_side, str) or not self.field_side:
            raise ValueError("field_side must be explicit nonempty text")

    def to_payload(self) -> dict[str, str]:
        return {
            "parent_surface_id": self.parent_surface_id,
            "interface_kind": self.interface_kind,
            "field_side": self.field_side,
        }

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> SurfaceBandSelection:
        if set(value) != {"parent_surface_id", "interface_kind", "field_side"}:
            raise ValueError("surface-band selection fields are not canonical")
        return cls(**value)


@dataclass(frozen=True)
class PalaceSurfaceBandTotals:
    """Detached totals; member records retain source and film assumptions."""

    mode: int
    palace_iteration: int
    normalization_energy_j: float
    cutoffs_um: tuple[float, ...]
    selections: tuple[SurfaceBandSelection, ...]
    totals: tuple[Mapping[str, Any], ...]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        if type(self.mode) is not int or self.mode <= 0:
            raise ValueError("mode must be a positive integer")
        if type(self.palace_iteration) is not int or self.palace_iteration <= 0:
            raise ValueError("palace_iteration must be a positive integer")
        _number(self.normalization_energy_j, "normalization_energy_j", positive=True)
        object.__setattr__(self, "cutoffs_um", tuple(self.cutoffs_um))
        object.__setattr__(self, "selections", tuple(self.selections))
        object.__setattr__(self, "totals", tuple(_freeze(item) for item in self.totals))
        object.__setattr__(self, "provenance", _freeze(self.provenance))

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "scgsim.palace.surface-band-totals.v1",
            "mode": self.mode,
            "palace_iteration": self.palace_iteration,
            "normalization_energy_j": self.normalization_energy_j,
            "cutoffs_um": list(self.cutoffs_um),
            "selections": [item.to_payload() for item in self.selections],
            "totals": _plain(self.totals),
            "provenance": _plain(self.provenance),
        }

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> PalaceSurfaceBandTotals:
        if (
            set(value)
            != {
                "schema_version",
                "mode",
                "palace_iteration",
                "normalization_energy_j",
                "cutoffs_um",
                "selections",
                "totals",
                "provenance",
            }
            or value.get("schema_version") != "scgsim.palace.surface-band-totals.v1"
        ):
            raise ValueError("surface-band totals payload is not canonical")
        return cls(
            value["mode"],
            value["palace_iteration"],
            value["normalization_energy_j"],
            tuple(value["cutoffs_um"]),
            tuple(
                SurfaceBandSelection.from_payload(item) for item in value["selections"]
            ),
            tuple(value["totals"]),
            value["provenance"],
        )


def _parent_inventory(summaries: Sequence[Mapping[str, Any]], parent: str):
    rows = [item for item in summaries if item.get("parent_surface_id") == parent]
    if not rows:
        raise ValueError(f"required source bands missing for parent {parent!r}")
    geometry = SurfaceDistanceBandSpec.from_payload(_plain(rows[0]))
    expected = set(range(len(geometry.distances_um) + 1))
    indices = [item.get("band_index") for item in rows]
    if len(indices) != len(set(indices)) or set(indices) != expected:
        raise ValueError(
            f"required band inventory incomplete or duplicated for {parent!r}"
        )
    by_index = {}
    for item in rows:
        if SurfaceDistanceBandSpec.from_payload(_plain(item)) != geometry:
            raise ValueError("parent bands disagree on source geometry identity")
        index = item["band_index"]
        lower = 0.0 if index == 0 else geometry.distances_um[index - 1]
        upper = (
            geometry.distances_um[index] if index < len(geometry.distances_um) else None
        )
        if item.get("lower_um") != lower or item.get("upper_um") != upper:
            raise ValueError("source band interval differs from declared cuts")
        children = item.get("child_surface_ids")
        count = item.get("component_count")
        if (
            not isinstance(children, (list, tuple))
            or type(count) is not int
            or count < 0
            or len(children) != count
            or len(set(children)) != count
            or any(not isinstance(child, str) or not child for child in children)
        ):
            raise ValueError("source band child/component inventory is incomplete")
        by_index[index] = item
    return geometry, by_index


def aggregate_surface_bands(
    result: PalaceEprResult,
    *,
    mode: int,
    palace_iteration: int,
    selections: Sequence[SurfaceBandSelection],
    cutoffs_um: Sequence[float],
    expected_normalization_energy_j: float | None = None,
) -> PalaceSurfaceBandTotals:
    """Sum complete bands at distance >= each exact caller-selected cutoff.

    A cutoff must coincide with a source band boundary for every selected parent.
    No geometric clipping, native Mask, Q2D correction or field re-evaluation occurs.
    """
    if not isinstance(result, PalaceEprResult):
        raise TypeError("result must be PalaceEprResult")
    if type(mode) is not int or mode <= 0:
        raise ValueError("mode must be a positive integer")
    if type(palace_iteration) is not int or palace_iteration <= 0:
        raise ValueError("palace_iteration must be a positive integer")
    if result.provenance.get("palace_native_iteration") != palace_iteration:
        raise ValueError(
            "requested Palace iteration differs from actual result provenance"
        )
    rows = [row for row in result.rows if row["mode"] == mode]
    if len(rows) != 1:
        raise ValueError("requested Palace mode is unavailable")
    row = rows[0]
    normalization = row["normalization_energy_j"]
    if (
        expected_normalization_energy_j is not None
        and expected_normalization_energy_j != normalization
    ):
        raise ValueError("requested normalization differs from the selected mode")
    selected = tuple(selections)
    if not selected or any(
        not isinstance(item, SurfaceBandSelection) for item in selected
    ):
        raise TypeError("explicit source-parent selections are required")
    parent_ids = [item.parent_surface_id for item in selected]
    if len(parent_ids) != len(set(parent_ids)):
        raise ValueError("a parent must be selected once with one interface/field side")
    cutoffs = tuple(_number(value, "cutoff_um") for value in cutoffs_um)
    if not cutoffs or len(cutoffs) != len(set(cutoffs)):
        raise ValueError("cutoffs must be explicit and unique")
    summaries = result.provenance.get("surface_distance_bands")
    if not isinstance(summaries, (tuple, list)) or any(
        not isinstance(item, Mapping) for item in summaries
    ):
        raise ValueError("required source-band summaries are unavailable")
    verified = []
    fragment_ids = set()
    physical_attributes = set()
    for selection in selected:
        geometry, inventory = _parent_inventory(summaries, selection.parent_surface_id)
        if not set(cutoffs).issubset({0.0, *geometry.distances_um}):
            raise ValueError("cutoff must match a declared source-band boundary")
        components = {}
        for surface in row["surface"]:
            band = surface.get("surface_distance_band")
            if (
                not isinstance(band, Mapping)
                or band.get("parent_surface_id") != selection.parent_surface_id
            ):
                continue
            if (
                surface["interface_kind"] != selection.interface_kind
                or surface["field_side"] != selection.field_side
            ):
                raise ValueError(
                    "selected parent interface/field side differs from result"
                )
            if surface["evaluation_kind"] != "geometric_band":
                raise ValueError("selected fragments must be source geometric bands")
            index = band.get("band_index")
            if index not in inventory:
                raise ValueError("component has no matching source band")
            summary = inventory[index]
            if (
                SurfaceDistanceBandSpec.from_payload(_plain(band)) != geometry
                or band.get("lower_um") != summary["lower_um"]
                or band.get("upper_um") != summary["upper_um"]
            ):
                raise ValueError("component source/interval differs from source band")
            surface_id = surface["surface_id"]
            if (selection.interface_kind, surface_id) in fragment_ids:
                raise ValueError("physical fragment is duplicated in selected members")
            fragment_ids.add((selection.interface_kind, surface_id))
            for attribute in surface.get("attributes", ()):
                if (selection.interface_kind, attribute) in physical_attributes:
                    raise ValueError(
                        "physical attribute is duplicated in selected members"
                    )
                physical_attributes.add((selection.interface_kind, attribute))
            components.setdefault(index, []).append(surface)
        for index, summary in sorted(inventory.items()):
            members = components.get(index, [])
            if {member["surface_id"] for member in members} != set(
                summary["child_surface_ids"]
            ):
                raise ValueError(
                    "required source band components are missing or unmatched"
                )
            component_indices = [
                member["surface_distance_band"]["component_index"] for member in members
            ]
            if set(component_indices) != set(range(summary["component_count"])) or len(
                component_indices
            ) != len(set(component_indices)):
                raise ValueError(
                    "source band component identities are incomplete or duplicated"
                )
            if "parent_owner_semantic_ids" in summary and any(
                tuple(member["owner_semantic_ids"])
                != tuple(summary["parent_owner_semantic_ids"])
                for member in members
            ):
                raise ValueError("component owner identity differs from source parent")
            verified.append((selection, summary, members))
    totals = []
    for interface in dict.fromkeys(item.interface_kind for item in selected):
        for cutoff in cutoffs:
            bands = [
                (summary, members)
                for selection, summary, members in verified
                if selection.interface_kind == interface
                and summary["lower_um"] >= cutoff
            ]
            members = [member for _, values in bands for member in values]
            # Preserve the exact raw table-derived inputs and effective film data.
            # Energies here are derived from stored p and the one mode normalization.
            records = [
                {
                    **_plain(member),
                    "original_energy_j": member["original_participation"]
                    * normalization,
                    "energy_j": member["participation"] * normalization,
                    "energy_basis": "stored participation times selected mode normalization",
                }
                for member in members
            ]
            inverse_q = [member["inverse_q"] for member in members]
            totals.append(
                {
                    "interface_kind": interface,
                    "cutoff_um": cutoff,
                    "distance_selection": "greater_than_or_equal",
                    "participation": sum(member["participation"] for member in members),
                    "original_participation": sum(
                        member["original_participation"] for member in members
                    ),
                    "energy_j": sum(member["energy_j"] for member in records),
                    "inverse_q": None
                    if any(value is None for value in inverse_q)
                    else sum(inverse_q),
                    "members": records,
                    "bands": [_plain(summary) for summary, _ in bands],
                }
            )
    return PalaceSurfaceBandTotals(
        mode,
        palace_iteration,
        normalization,
        cutoffs,
        selected,
        tuple(totals),
        {
            "selection_kind": "source_geometric_bands",
            "native_mask_equivalence": False,
            "frequency_hz": row["frequency_hz"],
            "source_result_provenance": _plain(result.provenance),
        },
    )
