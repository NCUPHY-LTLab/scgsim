"""Validate native dielectric-film and Inset index/config bindings.

MA, MS and SA denote metal-air, metal-substrate and substrate-air interfaces.
Historical masks retain their recorded material-side and length conventions."""

from __future__ import annotations

import math
from typing import Any

from scgsim.palace.results.artifacts import (
    expect_list,
    expect_mapping,
    expect_scalar,
    optional_dict,
)


def surface_mask_entries(index_map: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    """Return validated native Inset rows without inferring identity from names."""
    entries = expect_list(index_map.get("entries"), "palace_index_map.entries")
    dielectric = tuple(
        expect_mapping(entry, "palace dielectric index entry")
        for entry in entries
        if expect_scalar(expect_mapping(entry, "palace index entry"), "section", str)
        == "Boundaries.Postprocessing.Dielectric"
    )
    masks = tuple(entry for entry in dielectric if "mask" in entry)
    if not masks:
        return ()
    by_index = {expect_scalar(entry, "index", int): entry for entry in dielectric}
    if len(by_index) != len(dielectric):
        raise ValueError("surface EPR index entries must have unique indices.")
    baselines = tuple(entry for entry in dielectric if "mask" not in entry)
    baseline_indexes = sorted(expect_scalar(entry, "index", int) for entry in baselines)
    mask_indexes = sorted(expect_scalar(entry, "index", int) for entry in masks)
    if baseline_indexes != list(range(1, len(baselines) + 1)):
        raise ValueError(
            "surface EPR baselines must retain the initial contiguous indices."
        )
    if mask_indexes != list(range(len(baselines) + 1, len(dielectric) + 1)):
        raise ValueError("surface EPR masks must append after all baseline rows.")
    _validate_baseline_films(baselines)
    _validate_baseline_indices(by_index, dielectric)
    _validate_mask_bindings(by_index, masks)
    _validate_margin_sequences(baseline_indexes, by_index, masks)
    return tuple(sorted(masks, key=lambda entry: expect_scalar(entry, "index", int)))


def validate_surface_mask_config(
    config: dict[str, Any], index_map: dict[str, Any]
) -> None:
    """Match dielectric rows, film values and optional Inset margins to index evidence."""
    masks = surface_mask_entries(index_map)
    boundaries = optional_dict(config.get("Boundaries")) or {}
    postprocessing = optional_dict(boundaries.get("Postprocessing")) or {}
    declared_rows = postprocessing.get("Dielectric")
    config_declares_masks = isinstance(declared_rows, list) and any(
        isinstance(row, dict) and "Mask" in row for row in declared_rows
    )
    current_entries = [
        entry
        for entry in index_map["entries"]
        if entry.get("section") == "Boundaries.Postprocessing.Dielectric"
        and isinstance(entry.get("epr_spec"), dict)
        and entry["epr_spec"].get("schema_version") == "scgsim.palace.surface-film.v2"
    ]
    if not masks:
        if config_declares_masks:
            raise ValueError("config contains unindexed surface EPR masks.")
        if not current_entries:
            return
    # Current unmasked integrations use the same config/index binding as historical Masks.
    _validate_fragment_inventory(current_entries)
    rows = expect_list(
        postprocessing.get("Dielectric"),
        "config.Boundaries.Postprocessing.Dielectric",
        allow_empty=True,
    )
    configured = {
        expect_scalar(
            expect_mapping(row, "config dielectric row"), "Index", int
        ): expect_mapping(row, "config dielectric row")
        for row in rows
    }
    model = expect_mapping(config.get("Model"), "config.Model")
    model_l0_m = model.get("L0")
    if (
        isinstance(model_l0_m, bool)
        or not isinstance(model_l0_m, (int, float))
        or not math.isfinite(float(model_l0_m))
        or model_l0_m <= 0
    ):
        raise ValueError("config Model.L0 must be a finite positive length.")
    dielectric_entries = [
        expect_mapping(entry, "palace dielectric index entry")
        for entry in expect_list(index_map.get("entries"), "palace_index_map.entries")
        if expect_mapping(entry, "palace index entry").get("section")
        == "Boundaries.Postprocessing.Dielectric"
    ]
    indexed_dielectric_indexes = {
        expect_scalar(entry, "index", int) for entry in dielectric_entries
    }
    if (
        len(configured) != len(rows)
        or set(configured) != indexed_dielectric_indexes
        or len(rows) != len(dielectric_entries)
    ):
        raise ValueError("config and index evidence disagree on dielectric rows.")
    indexed_mask_indexes = {expect_scalar(entry, "index", int) for entry in masks}
    configured_mask_indexes = {
        index for index, row in configured.items() if "Mask" in row
    }
    if configured_mask_indexes != indexed_mask_indexes:
        raise ValueError("config and index evidence disagree on masked surface rows.")
    _validate_configured_films(
        configured, dielectric_entries, indexed_mask_indexes, model_l0_m
    )


def _validate_margin_sequences(
    baseline_indexes: list[int],
    by_index: dict[int, dict[str, Any]],
    masks: tuple[dict[str, Any], ...],
) -> None:
    """Require each baseline mask sequence to retain its ordered micrometre margins."""
    for baseline_index in baseline_indexes:
        baseline = by_index[baseline_index]
        epr_spec = expect_mapping(baseline.get("epr_spec"), "surface EPR spec")
        if "inset_margins_um" in epr_spec:
            expected_margins = expect_list(
                epr_spec.get("inset_margins_um"), "surface EPR inset margins"
            )
            if not expected_margins:
                raise ValueError(
                    "surface EPR inset margins must be non-empty when present."
                )
            for margin in expected_margins:
                if (
                    isinstance(margin, bool)
                    or not isinstance(margin, (int, float))
                    or not math.isfinite(float(margin))
                    or margin < 0
                ):
                    raise ValueError(
                        "surface EPR inset margins must be finite non-negative reals."
                    )
        else:
            expected_margins = []
        masked_for_baseline = sorted(
            (
                entry
                for entry in masks
                if expect_scalar(entry, "baseline_index", int) == baseline_index
            ),
            key=lambda entry: expect_scalar(
                expect_mapping(entry["mask"], "surface EPR mask"), "margin_index", int
            ),
        )
        ordinals = [
            expect_scalar(
                expect_mapping(entry["mask"], "surface EPR mask"), "margin_index", int
            )
            for entry in masked_for_baseline
        ]
        if ordinals != list(range(len(expected_margins))):
            raise ValueError(
                "surface EPR mask ordinals must match the configured margin sequence."
            )
        margins = [
            expect_mapping(entry["mask"], "surface EPR mask").get("margin_um")
            for entry in masked_for_baseline
        ]
        if margins != expected_margins:
            raise ValueError(
                "surface EPR mask margins must match the configured margin sequence."
            )


def _validate_mask_bindings(
    by_index: dict[int, dict[str, Any]], masks: tuple[dict[str, Any], ...]
) -> None:
    """Bind each Inset to its baseline and preserve the Model.L0 length conversion."""
    for entry in masks:
        index = expect_scalar(entry, "index", int)
        baseline_index = expect_scalar(entry, "baseline_index", int)
        if index == baseline_index or "mask" in by_index[baseline_index]:
            raise ValueError("masked surface EPR row must bind one unmasked baseline.")
        mask = expect_mapping(entry.get("mask"), "surface EPR mask")
        if mask.get("type") != "Inset":
            raise ValueError("surface EPR mask type must be Inset.")
        margin_index = expect_scalar(mask, "margin_index", int)
        if margin_index < 0:
            raise ValueError("surface EPR mask margin_index must be non-negative.")
        for field in ("margin_um", "native_margin", "model_l0_m"):
            value = mask.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise ValueError(
                    f"surface EPR mask {field} must be finite non-negative."
                )
        if float(mask["model_l0_m"]) <= 0:
            raise ValueError("surface EPR mask model_l0_m must be positive.")
        expected_native_margin = (
            float(mask["margin_um"]) * 1e-6 / float(mask["model_l0_m"])
        )
        if float(mask["native_margin"]) != expected_native_margin:
            raise ValueError("surface EPR mask native margin disagrees with Model.L0.")
        baseline = by_index[baseline_index]
        for field in ("attributes", "physical_names", "metadata", "epr_spec"):
            if entry.get(field) != baseline.get(field):
                raise ValueError(
                    "masked surface EPR row must retain baseline provenance."
                )


def _validate_baseline_indices(
    by_index: dict[int, dict[str, Any]], dielectric: tuple[dict[str, Any], ...]
) -> None:
    """Require baseline links to identify existing unmasked dielectric rows."""
    for entry in dielectric:
        index = expect_scalar(entry, "index", int)
        baseline_index = expect_scalar(entry, "baseline_index", int)
        if baseline_index < 1 or baseline_index not in by_index:
            raise ValueError(
                "surface EPR baseline_index must identify a dielectric row."
            )
        if "mask" not in entry and baseline_index != index:
            raise ValueError(
                "unmasked surface EPR row baseline_index must equal index."
            )


def _validate_baseline_films(baselines: tuple[dict[str, Any], ...]) -> None:
    """Validate recorded material, source-surface and SI/native film-thickness identities."""
    for baseline in baselines:
        metadata = expect_mapping(baseline.get("metadata"), "surface EPR metadata")
        if metadata.get("interface_type") not in {"MA", "MS", "SA"}:
            raise ValueError("surface EPR interface_type must be MA, MS, or SA.")
        for field in ("surface_id", "face_kind"):
            if not isinstance(metadata.get(field), str) or not metadata[field]:
                raise ValueError(f"surface EPR {field} must be non-empty text.")
        owners = expect_list(
            metadata.get("owner_semantic_ids"), "surface EPR owner semantic ids"
        )
        if not owners or not all(isinstance(owner, str) and owner for owner in owners):
            raise ValueError("surface EPR owners must be non-empty text.")
        expect_mapping(metadata.get("source_provenance"), "surface EPR provenance")
        epr_spec = expect_mapping(baseline.get("epr_spec"), "surface EPR spec")
        film_v2 = epr_spec.get("schema_version") == "scgsim.palace.surface-film.v2"
        for field, positive in (
            ("thickness", True),
            ("permittivity", True),
            ("loss_tangent", False),
        ):
            value = epr_spec.get(field)
            if field == "loss_tangent" and film_v2 and value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or (value <= 0 if positive else value < 0)
            ):
                raise ValueError(f"surface EPR {field} has an invalid value.")
        if film_v2:
            scale = epr_spec.get("model_l0_m")
            physical = epr_spec.get("film_thickness_m")
            if (
                isinstance(scale, bool)
                or not isinstance(scale, (int, float))
                or not math.isfinite(scale)
                or scale <= 0
                or isinstance(physical, bool)
                or not isinstance(physical, (int, float))
                or not math.isfinite(physical)
                or physical <= 0
                or not math.isclose(
                    physical, epr_spec["thickness"] * scale, rel_tol=1e-12
                )
            ):
                raise ValueError(
                    "surface EPR SI thickness conflicts with recorded Model.L0"
                )


def _validate_configured_films(
    configured: dict[int, dict[str, Any]],
    dielectric_entries: list[dict[str, Any]],
    indexed_mask_indexes: set[int],
    model_l0_m: float,
) -> None:
    """Match configured film fields and native margins to each exact dielectric index."""
    for entry in dielectric_entries:
        index = expect_scalar(entry, "index", int)
        row = configured.get(index)
        if row is None:
            raise ValueError("surface EPR index is absent from config.")
        if expect_list(
            row.get("Attributes"), "config dielectric Attributes"
        ) != expect_list(entry.get("attributes"), "surface EPR index attributes"):
            raise ValueError(
                "config surface EPR attributes do not match index evidence."
            )
        metadata = expect_mapping(entry.get("metadata"), "surface EPR metadata")
        epr_spec = expect_mapping(entry.get("epr_spec"), "surface EPR spec")
        if epr_spec.get("schema_version") == "scgsim.palace.surface-film.v2" and float(
            epr_spec["model_l0_m"]
        ) != float(model_l0_m):
            raise ValueError("surface film Model.L0 disagrees with config")
        expected_fields = {
            "Type": metadata.get("interface_type"),
            "Thickness": epr_spec.get("thickness"),
            "Permittivity": epr_spec.get("permittivity"),
            "LossTan": 0.0
            if epr_spec.get("loss_tangent") is None
            else epr_spec["loss_tangent"],
        }
        if any(
            key not in row or row[key] != value
            for key, value in expected_fields.items()
        ):
            raise ValueError(
                "config surface EPR fields do not match source index evidence."
            )
        if index not in indexed_mask_indexes:
            continue
        mask = expect_mapping(row.get("Mask"), "config dielectric Mask")
        evidence = expect_mapping(entry.get("mask"), "surface EPR mask")
        if mask != {"Type": "Inset", "Margin": evidence["native_margin"]}:
            raise ValueError("config surface EPR mask does not match index evidence.")
        if float(evidence["model_l0_m"]) != float(model_l0_m):
            raise ValueError("surface EPR mask Model.L0 disagrees with config.")


def _validate_fragment_inventory(current_entries: list[dict[str, Any]]) -> None:
    """Reject repeated surface identities and physical attributes within an interface type."""
    identities: set[tuple[str, str]] = set()
    physical_members: set[tuple[str, int]] = set()
    for entry in current_entries:
        if "mask" in entry:
            continue
        metadata = expect_mapping(entry.get("metadata"), "surface EPR metadata")
        interface = expect_scalar(metadata, "interface_type", str)
        surface_id = expect_scalar(metadata, "surface_id", str)
        if interface not in {"MA", "MS", "SA"} or not surface_id:
            raise ValueError("surface EPR integration identity is invalid.")
        identity = (interface, surface_id)
        if identity in identities:
            raise ValueError("surface EPR surface identity is duplicated.")
        identities.add(identity)
        attributes = expect_list(entry.get("attributes"), "surface EPR attributes")
        for attribute in attributes:
            if type(attribute) is not int or attribute <= 0:
                raise ValueError("surface EPR attributes must be positive integers.")
            member = (interface, attribute)
            if member in physical_members:
                raise ValueError("surface EPR physical attribute is duplicated.")
            physical_members.add(member)
