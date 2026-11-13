"""Validate Surface-EPR index entries and match them to the config's ``Dielectric`` rows.

MA, MS and SA denote metal-air, metal-substrate and substrate-air interfaces.  An index entry of the index map
(``INDEX_MAP_SCHEMA_VERSION``) is one Palace ``Dielectric`` row.  It lists the sidecar rows it integrates (``metadata.rows``, each
with its group, band, side, entities, Net and attributes) and the film it applies (``epr_spec``).  A
whole-surface entry has ``baseline_index == index``.  A margin entry carries ``inset`` and is the union of
the bands of a whole-surface entry from the margin's band inwards; its margins are those of the spec, or of
the customised conductor the surface belongs to (``inset_margins_by_entity_um``).  Native Mask/Inset rows do not exist.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from scgsim.contracts.palace_index_map import DIELECTRIC_SECTION
from scgsim.contracts.palace_run_folder import SURFACE_FILM_SCHEMA
from scgsim.geometry.surface_epr.specs import margins_of
from scgsim.palace.results.artifacts import (
    checked_number,
    expect_list,
    expect_mapping,
    expect_scalar,
    optional_dict,
    refusal,
)

SURFACE_INTERFACE_TYPES = ("MA", "MS", "SA")

_SIDES = {"above", "below", "both"}
_ROW_SIDES = _SIDES | {"mixed"}


def surface_epr_entries(index_map: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """The validated Surface-EPR index entries by index: whole surfaces first, then margins.

    Reads only the index map; ``validate_surface_epr_config`` matches the entries to a config.
    """
    entries = expect_list(index_map.get("entries"), "palace_index_map.entries")
    dielectric = sorted(
        (
            expect_mapping(entry, "palace dielectric index entry")
            for entry in entries
            if expect_mapping(entry, "palace index entry").get("section")
            == DIELECTRIC_SECTION
        ),
        key=lambda entry: expect_scalar(entry, "index", int),
    )
    for entry in dielectric:
        _validate_entry(entry)
    by_index = {entry["index"]: entry for entry in dielectric}
    if len(by_index) != len(dielectric):
        raise ValueError(
            refusal(
                "surface EPR index entries must have unique indices",
                found=[entry["index"] for entry in dielectric],
            )
        )
    wholes = [entry for entry in dielectric if "inset" not in entry]
    margins = [entry for entry in dielectric if "inset" in entry]
    if [entry["index"] for entry in wholes] != list(range(1, len(wholes) + 1)):
        raise ValueError(
            refusal(
                "surface EPR whole-surface entries must retain the initial contiguous indices",
                found=[entry["index"] for entry in wholes],
                expected=list(range(1, len(wholes) + 1)),
            )
        )
    if [entry["index"] for entry in margins] != list(
        range(len(wholes) + 1, len(dielectric) + 1)
    ):
        raise ValueError(
            refusal(
                "surface EPR margin entries must append after all surfaces",
                found=[entry["index"] for entry in margins],
                expected=list(range(len(wholes) + 1, len(dielectric) + 1)),
            )
        )
    for entry in wholes:
        baseline_index = expect_scalar(entry, "baseline_index", int)
        if baseline_index != entry["index"]:
            raise ValueError(
                refusal(
                    f"whole-surface entry {entry['index']}'s baseline_index must equal its index",
                    found=baseline_index,
                    expected=entry["index"],
                )
            )
    _validate_surface_inventory(wholes)
    _validate_margins(by_index, wholes, margins)
    return tuple(dielectric)


def entry_rows(entry: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The sidecar rows of one validated Surface-EPR entry."""
    return entry["metadata"]["rows"]


def validate_surface_epr_config(
    config: dict[str, Any], index_map: dict[str, Any]
) -> None:
    """Match the config's ``Dielectric`` rows, film values and cracking switch to the index evidence."""
    entries = surface_epr_entries(index_map)
    boundaries = optional_dict(config.get("Boundaries")) or {}
    postprocessing = optional_dict(boundaries.get("Postprocessing")) or {}
    rows = [
        expect_mapping(row, "config dielectric row")
        for row in expect_list(
            postprocessing.get("Dielectric", []),
            "config.Boundaries.Postprocessing.Dielectric",
            allow_empty=True,
        )
    ]
    if any("Mask" in row for row in rows):
        raise ValueError(
            refusal(
                "native Mask surface EPR rows are not accepted",
                found=[row.get("Index") for row in rows if "Mask" in row],
            )
        )
    configured = {expect_scalar(row, "Index", int): row for row in rows}
    if len(configured) != len(rows) or set(configured) != {
        entry["index"] for entry in entries
    }:
        raise ValueError(
            refusal(
                "config and index evidence disagree on dielectric rows",
                found=[row.get("Index") for row in rows],
                expected=sorted(entry["index"] for entry in entries),
            )
        )
    model = expect_mapping(config.get("Model"), "config.Model")
    model_l0_m = checked_number(model.get("L0"), "config Model.L0", positive=True)
    for entry in entries:
        _match_configured_row(configured[entry["index"]], entry, model_l0_m)
    _validate_internal_boundary_cracking(model, entries)


def _validate_entry(entry: dict[str, Any]) -> None:
    """One entry: its rows, the film it applies, and that its fields agree with its rows."""
    role = expect_scalar(entry, "role", str)
    if role != "surface_epr":
        raise ValueError(
            refusal(
                f"surface EPR index entry {entry.get('index')} role must be surface_epr",
                found=role,
            )
        )
    metadata = expect_mapping(entry.get("metadata"), "surface EPR metadata")
    rows = expect_list(metadata.get("rows"), "surface EPR rows")
    for row in rows:
        _validate_row(expect_mapping(row, "surface EPR row"))
    row_ids = [row["row"] for row in rows]
    if len(set(row_ids)) != len(row_ids):
        raise ValueError(
            refusal(
                f"surface EPR rows repeat a row id within entry {entry.get('index')}",
                found=row_ids,
            )
        )
    if len({row["interface_type"] for row in rows}) != 1:
        raise ValueError(
            refusal(
                f"surface EPR rows of entry {entry.get('index')} must share one interface type",
                found=sorted({row["interface_type"] for row in rows}),
            )
        )
    attributes = _positive_ints(entry.get("attributes"), "surface EPR attributes")
    row_attributes = sorted({a for row in rows for a in row["attributes"]})
    if attributes != row_attributes:
        raise ValueError(
            refusal(
                f"surface EPR entry {entry.get('index')} attributes disagree with their row evidence",
                found=attributes,
                expected=row_attributes,
            )
        )
    row_groups = sorted({row["group"] for row in rows})
    if entry.get("physical_names") != row_groups:
        raise ValueError(
            refusal(
                f"surface EPR entry {entry.get('index')} physical names disagree with their row evidence",
                found=entry.get("physical_names"),
                expected=row_groups,
            )
        )
    _validate_film(expect_mapping(entry.get("epr_spec"), "surface EPR spec"))


def _validate_row(row: dict[str, Any]) -> None:
    """The identity a reader relies on in one sidecar row: group, band, side, entities, Net, attributes."""
    for field in ("row", "group", "face_kind"):
        _non_empty_text(row.get(field), f"surface EPR row {field}")
    if row.get("interface_type") not in SURFACE_INTERFACE_TYPES:
        raise ValueError(
            refusal(
                f"surface EPR row {row.get('row')!r} interface_type must be MA, MS, or SA",
                found=row.get("interface_type"),
            )
        )
    band = row.get("band")
    if band is not None and (type(band) is not int or band < 0):
        raise ValueError(
            refusal(
                f"surface EPR row {row.get('row')!r} band must be null or a non-negative integer",
                found=band,
            )
        )
    if row.get("side") is not None and row["side"] not in _ROW_SIDES:
        raise ValueError(
            refusal(
                f"surface EPR row {row.get('row')!r} side is invalid",
                found=row["side"],
                expected=sorted(_ROW_SIDES),
            )
        )
    face_sides = row.get("face_sides")
    if face_sides is not None and (
        not isinstance(face_sides, dict)
        or not all(
            isinstance(face, str) and side in _SIDES
            for face, side in face_sides.items()
        )
    ):
        raise ValueError(
            refusal(
                f"surface EPR row {row.get('row')!r} face_sides must map faces to {sorted(_SIDES)}",
                found=face_sides,
            )
        )
    shared = row.get("shared_face_group")
    if shared is not None:
        _non_empty_text(shared, "surface EPR row shared_face_group")
    entities = expect_list(row.get("entities"), "surface EPR row entities")
    for entity in entities:
        _non_empty_text(entity, "surface EPR row entity")
    if row.get("net") is not None:
        _non_empty_text(row["net"], "surface EPR row net")
    _positive_ints(row.get("attributes"), "surface EPR row attributes")
    lineage = expect_mapping(row.get("lineage"), "surface EPR row lineage")
    expect_mapping(lineage.get("membership"), "surface EPR lineage membership")


def _validate_film(spec: dict[str, Any]) -> None:
    """The film values, the Model.L0 they were converted with, and their SI thickness."""
    if spec.get("schema_version") != SURFACE_FILM_SCHEMA:
        raise ValueError(f"surface EPR spec schema must be {SURFACE_FILM_SCHEMA!r}.")
    thickness = checked_number(
        spec.get("thickness"), "surface EPR thickness", positive=True
    )
    checked_number(spec.get("permittivity"), "surface EPR permittivity", positive=True)
    loss = spec.get("loss_tangent")
    if loss is not None and (_finite(loss) is None or loss < 0):
        raise ValueError(
            refusal(
                "surface EPR loss_tangent must be finite and non-negative or null",
                found=loss,
            )
        )
    scale = checked_number(
        spec.get("model_l0_m"), "surface EPR model_l0_m", positive=True
    )
    physical = checked_number(
        spec.get("film_thickness_m"), "surface EPR SI thickness", positive=True
    )
    if not math.isclose(physical, thickness * scale, rel_tol=1e-12):
        raise ValueError(
            refusal(
                "surface EPR SI thickness conflicts with recorded Model.L0",
                found=physical,
                expected=thickness * scale,
            )
        )
    margins = spec.get("inset_margins_um")
    if margins is not None:
        _validate_margin_sequence(margins)
    by_entity = spec.get("inset_margins_by_entity_um")
    if by_entity is not None:
        for entity, entity_margins in expect_mapping(
            by_entity, "surface EPR inset margins by entity"
        ).items():
            _non_empty_text(entity, "surface EPR customised entity")
            _validate_margin_sequence(entity_margins)


def _validate_margin_sequence(margins: Any) -> None:
    values = [
        checked_number(margin, "surface EPR inset margin", positive=True)
        for margin in expect_list(margins, "surface EPR inset margins")
    ]
    if any(b <= a for a, b in zip(values, values[1:], strict=False)):
        raise ValueError(
            refusal(
                "surface EPR inset margins must be strictly increasing", found=values
            )
        )


def _validate_surface_inventory(wholes: list[dict[str, Any]]) -> None:
    """Reject a sidecar row or a physical attribute used twice within an interface type."""
    row_ids: set[str] = set()
    members: set[tuple[str, int]] = set()
    for entry in wholes:
        for row in entry_rows(entry):
            if row["row"] in row_ids:
                raise ValueError(
                    refusal("surface EPR row is duplicated", found=row["row"])
                )
            row_ids.add(row["row"])
        interface_type = entry_rows(entry)[0]["interface_type"]
        for attribute in entry["attributes"]:
            if (interface_type, attribute) in members:
                raise ValueError(
                    refusal(
                        f"surface EPR physical attribute is duplicated for {interface_type}",
                        found=attribute,
                    )
                )
            members.add((interface_type, attribute))


def _validate_margins(
    by_index: dict[int, dict[str, Any]],
    wholes: list[dict[str, Any]],
    margins: list[dict[str, Any]],
) -> None:
    """Bind each margin to its whole surface, its bands and its place in the spec's margin sequence."""
    seen: dict[int, list[int]] = {entry["index"]: [] for entry in wholes}
    for entry in margins:
        inset = expect_mapping(entry.get("inset"), "surface EPR inset")
        baseline = by_index.get(expect_scalar(entry, "baseline_index", int))
        if baseline is None or baseline not in wholes:
            raise ValueError(
                refusal(
                    f"margin entry {entry.get('index')} must bind one whole-surface entry",
                    found=entry.get("baseline_index"),
                    expected=[whole["index"] for whole in wholes],
                )
            )
        margin_index = expect_scalar(inset, "margin_index", int)
        margin_um = checked_number(
            inset.get("margin_um"), "surface EPR margin_um", positive=True
        )
        first_band = expect_scalar(inset, "first_band", int)
        bands = expect_list(inset.get("bands"), "surface EPR inset bands")
        rows = entry_rows(entry)
        if (
            margin_index < 0
            or first_band < 1
            or bands != sorted(row["band"] for row in rows if row["band"] is not None)
            or len(bands) != len(rows)
            or min(bands) < first_band
        ):
            raise ValueError(
                refusal(
                    f"surface EPR inset of entry {entry.get('index')} disagrees with its band rows",
                    found={
                        "margin_index": margin_index,
                        "first_band": first_band,
                        "bands": bands,
                    },
                    expected=sorted(
                        row["band"] for row in rows if row["band"] is not None
                    ),
                )
            )
        configured = margins_of(
            baseline["epr_spec"],
            {entity for row in entry_rows(baseline) for entity in row["entities"]},
        )
        if margin_index >= len(configured) or configured[margin_index] != margin_um:
            raise ValueError(
                refusal(
                    f"surface EPR margin of entry {entry.get('index')} disagrees with the configured "
                    f"margin sequence {configured}",
                    found={"margin_index": margin_index, "margin_um": margin_um},
                )
            )
        if margin_index in seen[baseline["index"]]:
            raise ValueError(
                refusal(
                    f"surface EPR margin ordinal is duplicated for whole surface {baseline['index']}",
                    found=margin_index,
                )
            )
        seen[baseline["index"]].append(margin_index)
        if (
            entry["epr_spec"] != baseline["epr_spec"]
            or entry_rows(entry)[0]["interface_type"]
            != entry_rows(baseline)[0]["interface_type"]
            or not set(entry["attributes"]) <= set(baseline["attributes"])
            or not {row["row"] for row in rows}
            <= {row["row"] for row in entry_rows(baseline)}
        ):
            raise ValueError(
                refusal(
                    f"margin entry {entry.get('index')} must be part of whole surface "
                    f"{baseline['index']} and share its film",
                    found={
                        "attributes": entry["attributes"],
                        "epr_spec": entry["epr_spec"],
                    },
                    expected={
                        "attributes": baseline["attributes"],
                        "epr_spec": baseline["epr_spec"],
                    },
                )
            )


def _match_configured_row(
    row: dict[str, Any], entry: dict[str, Any], model_l0_m: float
) -> None:
    """One config row against its index entry: attributes, type, film values, Model.L0."""
    configured = expect_list(row.get("Attributes"), "config dielectric Attributes")
    if configured != entry["attributes"]:
        raise ValueError(
            refusal(
                f"config surface EPR row {entry['index']} attributes do not match index evidence",
                found=configured,
                expected=entry["attributes"],
            )
        )
    spec = entry["epr_spec"]
    if float(spec["model_l0_m"]) != float(model_l0_m):
        raise ValueError(
            refusal(
                f"surface film Model.L0 of entry {entry['index']} disagrees with config",
                found=spec["model_l0_m"],
                expected=model_l0_m,
            )
        )
    expected = {
        "Type": entry_rows(entry)[0]["interface_type"],
        "Thickness": spec["thickness"],
        "Permittivity": spec["permittivity"],
        # Palace needs a numeric loss tangent; the index keeps an unknown one as null.
        "LossTan": 0.0 if spec["loss_tangent"] is None else spec["loss_tangent"],
    }
    if any(key not in row or row[key] != value for key, value in expected.items()):
        raise ValueError(
            refusal(
                f"config surface EPR row {entry['index']} fields do not match source index evidence",
                found={key: row.get(key) for key in expected},
                expected=expected,
            )
        )


def _validate_internal_boundary_cracking(
    model: dict[str, Any], entries: tuple[dict[str, Any], ...]
) -> None:
    """``CrackInternalBoundaryElements`` is set exactly when a row integrates a two-sided sheet face."""
    needed = any(
        row["shared_face_group"] is not None
        for entry in entries
        for row in entry_rows(entry)
    )
    if (model.get("CrackInternalBoundaryElements") is True) != needed:
        raise ValueError(
            refusal(
                "config Model.CrackInternalBoundaryElements disagrees with the shared-face rows",
                found=model.get("CrackInternalBoundaryElements"),
                expected=needed,
            )
        )


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _positive_ints(value: Any, label: str) -> list[int]:
    items = expect_list(value, label)
    if not all(type(item) is int and item > 0 for item in items) or items != sorted(
        set(items)
    ):
        raise ValueError(f"{label} must be sorted unique positive integers.")
    return items


def _non_empty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be non-empty text.")
    return value
