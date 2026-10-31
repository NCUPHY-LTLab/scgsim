"""Check ``metadata/palace_index_map.json`` on its own.

The index map records, for every Terminal, lumped port, energy domain and dielectric
surface in ``config.json``, the 1-based Palace index it was given and what it
stands for. Result tables are columns per index, so this map is what lets a column be
read as a physical quantity.

This module checks the map alone. Agreement between the map and ``config.json`` is
checked in ``_config_checks.py``.

Surface EPR rows come in two kinds. A baseline row integrates the whole interface. A
masked row repeats a baseline with an ``Inset`` mask that excludes a margin at the
patch edge. Every requested margin is its own row that points back to its baseline.
"""

from __future__ import annotations

import math
from typing import Any, TypedDict

from ._expect import _expect_list, _expect_mapping, _expect_scalar
from ._run_documents import IndexMap

# Film spec as ``preparation/surface_epr.py`` writes it. Version 2 added the SI
# fields ``model_l0_m`` and ``film_thickness_m``.
FILM_SPEC_SCHEMA = "scgsim.palace.surface-film.v2"


class IndexCounts(TypedDict):
    """How many rows of each kind the index map declares."""

    terminal: int
    lumped_port: int
    domain: int
    surface: int


class MaskEvidence(TypedDict):
    """The ``mask`` block of a masked surface-EPR index entry."""

    type: str  # always "Inset"
    margin_index: int  # position in the requested margin sequence, from 0
    margin_um: float
    native_margin: float  # margin in Model.L0 units, as written to config.json
    model_l0_m: float


class SurfaceMaskEntry(TypedDict):
    """A masked ``Boundaries.Postprocessing.Dielectric`` index entry.

    The entry carries more keys (``section``, ``role``, ``entry_name`` ...). Only the
    ones the resolver reads are named here.
    """

    index: int
    baseline_index: int
    attributes: list[int]
    physical_names: list[str]
    metadata: dict[str, Any]
    epr_spec: dict[str, Any]
    mask: MaskEvidence


def _index_counts(index_map: IndexMap) -> IndexCounts:
    entries = _expect_list(index_map.get("entries"), "palace_index_map.entries")
    counts = {
        "terminal": 0,
        "lumped_port": 0,
        "domain": 0,
        "surface": 0,
    }
    for entry in entries:
        data = _expect_mapping(entry, "palace_index_map.entry")
        section = _expect_scalar(data, "section", str, fallback="")
        if section.startswith("Boundaries.Terminal"):
            counts["terminal"] += 1
        elif section.startswith("Boundaries.LumpedPort"):
            counts["lumped_port"] += 1
        elif section.startswith("Domains.Postprocessing.Energy"):
            counts["domain"] += 1
        elif section == "Boundaries.Postprocessing.Dielectric":
            counts["surface"] += 1
    return counts


def _validate_index_entries(index_map: IndexMap) -> None:
    entries = index_map.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("palace_index_map.json must include non-empty entries list.")
    expected_roles = {
        "Boundaries.Terminal": "terminal",
        "Boundaries.LumpedPort": "port_surface",
        "Domains.Postprocessing.Energy": "domain_energy",
        "Boundaries.Postprocessing.Dielectric": "surface_epr",
    }
    indices: dict[str, list[int]] = {section: [] for section in expected_roles}
    names: set[tuple[str, str]] = set()
    for entry in entries:
        if not isinstance(entry, dict) or not entry:
            raise ValueError("index map entries must be non-empty mappings.")
        section = _expect_scalar(entry, "section", str)
        if section not in expected_roles:
            raise ValueError(f"index map has unsupported section {section!r}.")
        if _expect_scalar(entry, "role", str) != expected_roles[section]:
            raise ValueError("index map section and role are inconsistent.")
        index = _expect_scalar(entry, "index", int)
        if index < 1:
            raise ValueError("index map indices must be positive.")
        indices[section].append(index)
        entry_name = _expect_scalar(entry, "entry_name", str)
        if not entry_name or (section, entry_name) in names:
            raise ValueError(
                "index entry names must be non-empty and unique per section."
            )
        names.add((section, entry_name))
        physical_names = _expect_list(
            entry.get("physical_names"), "index physical_names"
        )
        if not all(isinstance(name, str) and name for name in physical_names):
            raise ValueError("index physical_names must be non-empty strings.")
        entry_attributes = _expect_list(entry.get("attributes"), "index attributes")
        for attribute in entry_attributes:
            if (
                not isinstance(attribute, int)
                or isinstance(attribute, bool)
                or attribute < 1
            ):
                raise ValueError("index attributes must be positive integers.")
        metadata = _expect_mapping(entry.get("metadata"), "index metadata")
        if section == "Boundaries.Terminal":
            _expect_scalar(entry, "terminal_name", str)
            _expect_scalar(entry, "net_id", str)
            _expect_list(
                entry.get("conductor_component_ids"), "terminal conductor components"
            )
        elif section == "Boundaries.LumpedPort":
            _expect_scalar(entry, "port_name", str)
            owners = _expect_list(
                metadata.get("owner_semantic_ids"), "port owner semantic ids"
            )
            if len(owners) != 2 or not all(
                isinstance(owner, str) and owner for owner in owners
            ):
                raise ValueError(
                    "lumped port index entry must have exactly two owners."
                )
        elif section == "Domains.Postprocessing.Energy":
            _expect_mapping(metadata.get("material"), "domain index material")
        elif section == "Boundaries.Postprocessing.Dielectric":
            _expect_mapping(
                metadata.get("physical_attribute"), "surface index attributes"
            )
    for section, values in indices.items():
        if values and sorted(values) != list(range(1, len(values) + 1)):
            raise ValueError(f"index map {section} entries must be ordered 1..N.")
    _surface_mask_entries(index_map)


def _surface_mask_entries(index_map: IndexMap) -> tuple[SurfaceMaskEntry, ...]:
    """Return validated native Inset rows without inferring identity from names."""
    entries = _expect_list(index_map.get("entries"), "palace_index_map.entries")
    dielectric = tuple(
        _expect_mapping(entry, "palace dielectric index entry")
        for entry in entries
        if _expect_scalar(_expect_mapping(entry, "palace index entry"), "section", str)
        == "Boundaries.Postprocessing.Dielectric"
    )
    masks = tuple(entry for entry in dielectric if "mask" in entry)
    if not masks:
        return ()
    by_index = {_expect_scalar(entry, "index", int): entry for entry in dielectric}
    if len(by_index) != len(dielectric):
        raise ValueError("surface EPR index entries must have unique indices.")
    baselines = tuple(entry for entry in dielectric if "mask" not in entry)
    baseline_indexes = sorted(
        _expect_scalar(entry, "index", int) for entry in baselines
    )
    mask_indexes = sorted(_expect_scalar(entry, "index", int) for entry in masks)
    if baseline_indexes != list(range(1, len(baselines) + 1)):
        raise ValueError(
            "surface EPR baselines must retain the initial contiguous indices."
        )
    # TODO(human): why masked rows must come after every baseline row (rather than
    # sitting next to their baseline). The code only enforces the layout.
    if mask_indexes != list(range(len(baselines) + 1, len(dielectric) + 1)):
        raise ValueError("surface EPR masks must append after all baseline rows.")
    _check_baseline_rows(baselines)
    _check_baseline_links(dielectric, by_index)
    _check_mask_rows(masks, by_index)
    _check_margin_sequences(baseline_indexes, by_index, masks)
    return tuple(sorted(masks, key=lambda entry: _expect_scalar(entry, "index", int)))


def _check_baseline_rows(baselines: tuple[dict[str, Any], ...]) -> None:
    """Each unmasked row names a known interface type and a valid film."""
    for baseline in baselines:
        metadata = _expect_mapping(baseline.get("metadata"), "surface EPR metadata")
        # MA, MS, SA: metal-air, metal-substrate, substrate-air interface (docs/eigenmode-epr.qmd).
        if metadata.get("interface_type") not in {"MA", "MS", "SA"}:
            raise ValueError("surface EPR interface_type must be MA, MS, or SA.")
        for field in ("surface_id", "face_kind"):
            if not isinstance(metadata.get(field), str) or not metadata[field]:
                raise ValueError(f"surface EPR {field} must be non-empty text.")
        owners = _expect_list(
            metadata.get("owner_semantic_ids"), "surface EPR owner semantic ids"
        )
        if not owners or not all(isinstance(owner, str) and owner for owner in owners):
            raise ValueError("surface EPR owners must be non-empty text.")
        _expect_mapping(metadata.get("source_provenance"), "surface EPR provenance")
        _check_film_spec(_expect_mapping(baseline.get("epr_spec"), "surface EPR spec"))


def _check_film_spec(epr_spec: dict[str, Any]) -> None:
    """The dielectric film of one surface: native and SI thickness must agree."""
    found = epr_spec.get("schema_version")
    if found != FILM_SPEC_SCHEMA:
        raise ValueError(
            f"surface EPR film spec schema_version is {found!r}; "
            f"supported: {FILM_SPEC_SCHEMA!r}."
        )
    # A missing loss_tangent is allowed: config.json then gets LossTan 0.0.
    for field, positive in (
        ("thickness", True),
        ("permittivity", True),
        ("loss_tangent", False),
    ):
        value = epr_spec.get(field)
        if field == "loss_tangent" and value is None:
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or (value <= 0 if positive else value < 0)
        ):
            raise ValueError(f"surface EPR {field} has an invalid value.")
    # Native thickness is in Model.L0 units; the SI thickness must equal it times L0.
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
        or not math.isclose(physical, epr_spec["thickness"] * scale, rel_tol=1e-12)
    ):
        raise ValueError("surface EPR SI thickness conflicts with recorded Model.L0")


def _check_baseline_links(
    dielectric: tuple[dict[str, Any], ...], by_index: dict[int, dict[str, Any]]
) -> None:
    """Every row points at a dielectric row; unmasked rows point at themselves."""
    for entry in dielectric:
        index = _expect_scalar(entry, "index", int)
        baseline_index = _expect_scalar(entry, "baseline_index", int)
        if baseline_index < 1 or baseline_index not in by_index:
            raise ValueError(
                "surface EPR baseline_index must identify a dielectric row."
            )
        if "mask" not in entry and baseline_index != index:
            raise ValueError(
                "unmasked surface EPR row baseline_index must equal index."
            )


def _check_mask_rows(
    masks: tuple[dict[str, Any], ...], by_index: dict[int, dict[str, Any]]
) -> None:
    """Each masked row is a valid Inset that repeats its baseline unchanged."""
    # Palace reads Margin in Model.L0 units (docs/specs/palace-electrostatic-sgb.qmd),
    # so native_margin = margin_um * 1e-6 / model_l0_m. The test below is exact
    # float equality on purpose: it must match the writer's expression order.
    for entry in masks:
        index = _expect_scalar(entry, "index", int)
        baseline_index = _expect_scalar(entry, "baseline_index", int)
        if index == baseline_index or "mask" in by_index[baseline_index]:
            raise ValueError("masked surface EPR row must bind one unmasked baseline.")
        mask = _expect_mapping(entry.get("mask"), "surface EPR mask")
        if mask.get("type") != "Inset":
            raise ValueError("surface EPR mask type must be Inset.")
        margin_index = _expect_scalar(mask, "margin_index", int)
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


def _check_margin_sequences(
    baseline_indexes: list[int],
    by_index: dict[int, dict[str, Any]],
    masks: tuple[dict[str, Any], ...],
) -> None:
    """Masked rows must reproduce the baseline's `inset_margins_um` in order."""
    for baseline_index in baseline_indexes:
        baseline = by_index[baseline_index]
        epr_spec = _expect_mapping(baseline.get("epr_spec"), "surface EPR spec")
        if "inset_margins_um" in epr_spec:
            expected_margins = _expect_list(
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
                if _expect_scalar(entry, "baseline_index", int) == baseline_index
            ),
            key=lambda entry: _expect_scalar(
                _expect_mapping(entry["mask"], "surface EPR mask"), "margin_index", int
            ),
        )
        ordinals = [
            _expect_scalar(
                _expect_mapping(entry["mask"], "surface EPR mask"), "margin_index", int
            )
            for entry in masked_for_baseline
        ]
        if ordinals != list(range(len(expected_margins))):
            raise ValueError(
                "surface EPR mask ordinals must match the configured margin sequence."
            )
        margins = [
            _expect_mapping(entry["mask"], "surface EPR mask").get("margin_um")
            for entry in masked_for_baseline
        ]
        if margins != expected_margins:
            raise ValueError(
                "surface EPR mask margins must match the configured margin sequence."
            )
