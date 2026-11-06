"""Surface-EPR lowering: Palace ``Dielectric`` rows over the attributes of SGB's rows and bands.

``inset_margins_um`` and ``inset_margins_by_entity_um`` are geometry: SGB cuts them into the model as bands
(``mesh()`` requests them from these specs), and a margin ``m_k`` is the union of the attributes of bands
``k`` to ``K`` of every band family that serves the spec, listed as plain attributes.  The bands of a family
come from the sidecar (``insets[*].bands``, or ``expansion.banded[*].bands`` of a type-level request), never
from group names.  Stock Palace integrates plain attributes; native Mask/Inset is neither written nor
accepted.  Omit both margin keys for baseline rows only.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from numbers import Real
from typing import Any, cast

from scgsim.geometry.sgb.records.grid import Grid
from scgsim.semantics.epr import film_assumptions

from ..mesh.insets import check_bandable_face_kinds
from ..mesh.margins import margins_of
from ..mesh.physical_groups import Groups, rows_of


def normalize_surface_epr_specs(
    specs: Mapping[str, Mapping[str, Any]],
    *,
    materials: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Normalize configured MA/MS/SA layers from numeric or explicit material data."""
    if not isinstance(specs, Mapping):
        raise TypeError("surface_epr specs must be a mapping.")
    unknown_types = set(specs).difference({"MA", "MS", "SA"})
    if unknown_types:
        raise ValueError(
            f"surface_epr has unsupported interface types: {sorted(unknown_types)}."
        )
    result: dict[str, dict[str, Any]] = {}
    for interface_type, raw in sorted(specs.items()):
        result[interface_type] = _normalize_interface_spec(
            interface_type, raw, materials
        )
    return result


def _normalize_interface_spec(
    interface_type: str,
    raw: Mapping[str, Any],
    materials: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Normalize one interface without changing validation or material precedence."""
    if not isinstance(raw, Mapping):
        raise TypeError(f"Surface EPR {interface_type} spec must be a mapping.")
    unknown = set(raw).difference(
        {
            "thickness",
            "film_thickness_m",
            "permittivity",
            "material_id",
            "loss_tangent",
            "source",
            "preset",
            "face_kinds",
            "required",
            "inset_margins_um",
            "inset_margins_by_entity_um",
        }
    )
    if unknown:
        raise ValueError(
            f"Surface EPR {interface_type} has unknown fields: {sorted(unknown)}."
        )
    thickness = raw.get("thickness")
    thickness_m = raw.get("film_thickness_m")
    if (thickness is None) == (thickness_m is None):
        raise ValueError(
            f"Surface EPR {interface_type} requires exactly one native thickness or film_thickness_m."
        )
    if thickness is not None and not _positive(thickness):
        raise ValueError(f"Surface EPR {interface_type} requires positive thickness.")
    if thickness_m is not None:
        film_assumptions({"film_thickness_m": thickness_m}, partial=True)
    numeric = raw.get("permittivity")
    material_id = raw.get("material_id")
    if (numeric is None) == (material_id is None):
        raise ValueError(
            f"Surface EPR {interface_type} requires exactly one of permittivity or material_id."
        )
    numeric, default_loss, resolved = _resolve_film_material(
        interface_type, material_id, numeric, materials
    )
    if not _positive(numeric):
        raise ValueError(
            f"Surface EPR {interface_type} requires positive resolved permittivity."
        )
    loss_tangent = raw.get("loss_tangent", default_loss)
    film_assumptions({"loss_tangent": loss_tangent}, partial=True)
    source = raw.get("source", "material" if material_id is not None else "explicit")
    preset = raw.get("preset")
    film_assumptions({"source": source, "preset": preset}, partial=True)
    face_kinds = raw.get("face_kinds")
    if face_kinds is not None:
        if not isinstance(face_kinds, (list, tuple)) or not face_kinds:
            raise ValueError(
                f"Surface EPR {interface_type} face_kinds must be a non-empty sequence."
            )
        normalized_faces = sorted({str(value).strip().lower() for value in face_kinds})
        if not all(value for value in normalized_faces):
            raise ValueError(
                f"Surface EPR {interface_type} face_kinds must be non-empty."
            )
        resolved["face_kinds"] = normalized_faces
    required = raw.get("required", True)
    if not isinstance(required, bool):
        raise TypeError(f"Surface EPR {interface_type} required must be a bool.")
    _set_inset_margins(interface_type, raw, resolved)
    if "face_kinds" in resolved and (
        "inset_margins_um" in resolved or "inset_margins_by_entity_um" in resolved
    ):
        check_bandable_face_kinds(interface_type, resolved["face_kinds"])
    return {
        **(
            {"thickness": float(thickness)}
            if thickness is not None
            else {"film_thickness_m": float(cast(Any, thickness_m))}
        ),
        "permittivity": float(cast(Any, numeric)),
        "loss_tangent": None if loss_tangent is None else float(loss_tangent),
        "source": source,
        "preset": preset,
        "required": required,
        **resolved,
    }


def _resolve_film_material(
    interface_type: str,
    material_id: Any,
    numeric: Any,
    materials: Mapping[str, Mapping[str, Any]],
) -> tuple[Any, Any, dict[str, Any]]:
    """Resolve explicit dielectric identity and its optional default loss."""
    if material_id is not None:
        if not isinstance(material_id, str) or not material_id.strip():
            raise ValueError(
                f"Surface EPR {interface_type} material_id must be non-empty."
            )
        material = materials.get(material_id)
        if not isinstance(material, Mapping) or material.get("kind") != "dielectric":
            raise ValueError(
                f"Surface EPR {interface_type} material_id must resolve an explicit dielectric material."
            )
        numeric = material.get("permittivity")
        default_loss = material.get("loss_tangent")
        resolved = {"material_id": material_id}
    else:
        default_loss = None
        resolved = {}
    return numeric, default_loss, resolved


def _set_inset_margins(
    interface_type: str, raw: Mapping[str, Any], resolved: dict[str, Any]
) -> None:
    """Retain the margins SGB cuts into the model as bands: the spec's, and those of customised conductors."""
    if "inset_margins_um" in raw:
        resolved["inset_margins_um"] = _validated_margins(
            raw["inset_margins_um"], f"Surface EPR {interface_type} inset_margins_um"
        )
    if "inset_margins_by_entity_um" in raw:
        resolved["inset_margins_by_entity_um"] = _by_entity_margins(
            interface_type, raw["inset_margins_by_entity_um"]
        )


def _by_entity_margins(interface_type: str, raw: Any) -> dict[str, list[float]]:
    """Margins per customised conductor, checked like ``inset_margins_um``."""
    label = f"Surface EPR {interface_type} inset_margins_by_entity_um"
    if interface_type == "SA":
        raise ValueError(
            f"{label} customises conductors, and an SA interface lies between solution regions; "
            "use inset_margins_um for SA."
        )
    if not isinstance(raw, Mapping) or not raw:
        raise TypeError(f"{label} must be a non-empty mapping of entity id to margins.")
    result: dict[str, list[float]] = {}
    for entity, margins in sorted(raw.items(), key=lambda item: str(item[0])):
        if not isinstance(entity, str) or not entity:
            raise ValueError(f"{label} keys must be non-empty entity ids.")
        result[entity] = _validated_margins(margins, f"{label}[{entity!r}]")
    return result


def _validated_margins(margins: Any, label: str) -> list[float]:
    """The ordered, strictly increasing positive margins of one sequence."""
    if isinstance(margins, (str, bytes)) or not isinstance(margins, (list, tuple)):
        raise TypeError(f"{label} must be an ordered non-empty sequence.")
    if not margins:
        raise ValueError(f"{label} must be non-empty.")
    normalized = []
    for margin in margins:
        if not _positive(margin):
            raise ValueError(
                f"{label} values must be finite positive reals "
                "(the whole surface is the baseline row)."
            )
        normalized.append(float(margin))
    if any(b <= a for a, b in zip(normalized, normalized[1:], strict=False)):
        raise ValueError(f"{label} must be strictly increasing.")
    return normalized


def build_surface_epr_postprocessing(
    groups: Groups,
    specs: Mapping[str, Mapping[str, Any]],
    *,
    model_l0_m: float = 1e-6,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Emit Palace ``Dielectric`` rows and their index entries (index map schema 3) from the Surface-EPR rows of ``groups``.

    For each spec (interface type) and each surface unit of that type -- an unbanded row, or all bands of
    one band family -- one baseline row covers every attribute of the unit.  For each inset margin of the
    unit (the spec's, or its conductor's own), a further row covers the attributes of the unit's bands from
    the margin's band to the last.
    """
    if not _positive(model_l0_m):
        raise ValueError("Model.L0 must be a finite positive length in metres.")
    rows: list[dict[str, Any]] = []
    index_map: list[dict[str, Any]] = []
    margin_requests: list[_MarginRequest] = []
    all_rows = rows_of(groups)
    members: set[tuple[str, int]] = set()
    for interface_type, preset in sorted(specs.items()):
        units = _units(all_rows, interface_type, preset)
        if not units and preset["required"]:
            raise ValueError(
                f"Required Surface EPR {interface_type} has no matching structured group."
            )
        snapshot = _spec_snapshot(preset, model_l0_m)
        for unit in units:
            attributes = _unit_attributes(unit.rows)
            for attribute in attributes:
                if (interface_type, attribute) in members:
                    raise ValueError("Surface EPR physical attribute is duplicated.")
                members.add((interface_type, attribute))
            index = len(rows) + 1
            rows.append(_dielectric_row(index, interface_type, attributes, snapshot))
            entry = _index_entry(index, unit.name, unit.rows, attributes, snapshot)
            entry["baseline_index"] = index
            index_map.append(entry)
            margin_requests.extend(
                _MarginRequest(index, unit, margin_index, margin)
                for margin_index, margin in enumerate(unit.margins_um)
            )

    for request in margin_requests:
        selected = request.selected_rows()
        if not selected:
            continue  # nothing of this unit lies that far from a gap edge (unbanded, or a sidewall)
        attributes = _unit_attributes(selected)
        index = len(rows) + 1
        spec = _spec_snapshot(specs[request.unit.interface_type], model_l0_m)
        rows.append(
            _dielectric_row(index, request.unit.interface_type, attributes, spec)
        )
        entry = _index_entry(
            index,
            f"{request.unit.name}__inset_{request.margin_index}",
            selected,
            attributes,
            spec,
        )
        entry["baseline_index"] = request.baseline_index
        entry["inset"] = request.description(selected)
        index_map.append(entry)
    return rows, index_map


@dataclass(frozen=True)
class _Unit:
    """The rows of one interface type that one baseline row covers: an unbanded row, or all bands of a family.

    ``family`` is the sidecar record of the band family (``margins_grid`` in model-grid units with the
    ``grid_um`` they are measured in, and ``bands``, band index to the groups that carry it) and ``margins_um``
    the margins this unit is asked for; an unbanded row has neither.
    """

    name: str
    interface_type: str
    rows: tuple[Mapping[str, Any], ...]
    family: Mapping[str, Any] | None = None
    margins_um: tuple[float, ...] = ()


@dataclass(frozen=True)
class _MarginRequest:
    """One inset margin of a spec on one unit."""

    baseline_index: int
    unit: _Unit
    margin_index: int
    margin_um: float

    def first_band(self) -> int | None:
        """Index of the first band at least ``margin_um`` from the gap edge, or None for an unbanded unit."""
        if self.unit.family is None:
            return None
        family = self.unit.family
        # A margin is the same length as a band's when both are the same multiple of the model grid.
        wanted = Grid(family["grid_um"]).lattice_units(self.margin_um)
        if wanted is not None and wanted in family["margins_grid"]:
            return list(family["margins_grid"]).index(wanted) + 1
        raise ValueError(
            f"Surface EPR {self.unit.interface_type} margin {self.margin_um} um is not among "
            f"the margins {family['margins_um']} of the bands of {self.unit.name!r}."
        )

    def selected_rows(self) -> tuple[Mapping[str, Any], ...]:
        """The rows on the groups of the family's bands from the margin's band to the last."""
        first = self.first_band()
        if first is None or self.unit.family is None:
            return ()
        groups = {
            group
            for band, band_groups in self.unit.family["bands"].items()
            if int(band) >= first
            for group in band_groups
        }
        return tuple(row for row in self.unit.rows if row["group"] in groups)

    def description(self, selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """How the row was selected, for the index map."""
        return {
            "margin_um": self.margin_um,
            "margin_index": self.margin_index,
            "first_band": self.first_band(),
            "bands": sorted(row["band"] for row in selected),
        }


def _units(
    rows: Mapping[str, Mapping[str, Any]],
    interface_type: str,
    preset: Mapping[str, Any],
) -> list[_Unit]:
    """The surface units of one interface type, in sorted row order.

    A row with no band is a unit of its own (a sidewall, or a face SGB did not band).  The banded rows of one
    band family are one unit: SGB gives each face one family, whose bands are listed in the sidecar.
    """
    face_filter = preset.get("face_kinds")
    singles: list[_Unit] = []
    by_family: dict[str, list[Mapping[str, Any]]] = {}
    for row_id, row in sorted(rows.items()):
        if row["interface_type"] != interface_type:
            continue
        if face_filter is not None and row["face_kind"] not in face_filter:
            continue
        if row["band"] is None:
            singles.append(_Unit(row_id, interface_type, ({**row, "row": row_id},)))
        else:
            by_family.setdefault(row["family"], []).append({**row, "row": row_id})
    banded = [
        _family_unit(interface_type, preset, sorted(members, key=_band_order))
        for members in by_family.values()
    ]
    return sorted(
        [*_unbanded_names_apart(singles, banded), *banded], key=lambda unit: unit.name
    )


def _unbanded_names_apart(singles: list[_Unit], banded: list[_Unit]) -> list[_Unit]:
    """Rename an unbanded unit that shares its label with a band family.

    The solution interface of one dielectric-vacuum pair lies at several heights, and only some of them
    can be banded (the substrate underside has no gap edge).  Entry names must tell them apart.
    """
    taken = {unit.name for unit in banded}
    return [
        replace(unit, name=f"{unit.name}__unbanded") if unit.name in taken else unit
        for unit in singles
    ]


def _family_unit(
    interface_type: str, preset: Mapping[str, Any], members: Sequence[Mapping[str, Any]]
) -> _Unit:
    first = members[0]
    return _Unit(
        _unit_label(first),
        interface_type,
        tuple(members),
        family={
            "margins_um": first["margins_um"],
            "margins_grid": first["margins_grid"],
            "grid_um": first["grid_um"],
            "bands": first["family_bands"],
        },
        margins_um=tuple(margins_of(preset, first["entities"])),
    )


def _unit_label(first_band_row: Mapping[str, Any]) -> str:
    """A label for the index: the row id of a family's first band without its band suffix.

    Only a label.  Which rows a margin selects comes from the sidecar's bands, never from names.
    """
    return first_band_row["row"].removesuffix(f"::band={first_band_row['band']}")


def _band_order(row: Mapping[str, Any]) -> tuple[int, str]:
    return (row["band"], row["row"])


def _unit_attributes(rows: Sequence[Mapping[str, Any]]) -> list[int]:
    return sorted({attribute for row in rows for attribute in row["attribute_sides"]})


def _spec_snapshot(preset: Mapping[str, Any], model_l0_m: float) -> dict[str, Any]:
    """The spec with its thickness in native units, as recorded in the index map."""
    native_thickness = (
        preset["film_thickness_m"] / float(model_l0_m)
        if "film_thickness_m" in preset
        else preset["thickness"]
    )
    return {
        **preset,
        "thickness": native_thickness,
        "film_thickness_m": native_thickness * float(model_l0_m),
        "model_l0_m": float(model_l0_m),
        "schema_version": "scgsim.palace.surface-film.v2",
    }


def _dielectric_row(
    index: int,
    interface_type: str,
    attributes: list[int],
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "Index": index,
        "Attributes": attributes,
        "Type": interface_type,
        "Thickness": snapshot["thickness"],
        "Permittivity": snapshot["permittivity"],
        # Palace needs a numeric value; the index snapshot keeps None distinct.
        "LossTan": 0.0
        if snapshot["loss_tangent"] is None
        else snapshot["loss_tangent"],
    }


def _index_entry(
    index: int,
    name: str,
    rows: Sequence[Mapping[str, Any]],
    attributes: list[int],
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "section": "Boundaries.Postprocessing.Dielectric",
        "index": index,
        "entry_name": name,
        "role": "surface_epr",
        "attributes": attributes,
        "physical_names": sorted({row["group"] for row in rows}),
        "metadata": {"rows": [_row_metadata(row) for row in rows]},
        "epr_spec": dict(snapshot),
    }


def _row_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    """What a reader needs to know about one sidecar row: identity, sides, owners, Net and lineage."""
    return {
        "row": row["row"],
        "group": row["group"],
        "interface_type": row["interface_type"],
        "face_kind": row["face_kind"],
        "band": row["band"],
        "side": row["side"],
        "face_sides": row["face_sides"],
        "shared_face_group": row["shared_face_group"],
        "entities": list(row["entities"]),
        "net": row["net"],
        "attributes": sorted(row["attribute_sides"]),
        # Lineage membership is attribution only: it weights nothing.
        "lineage": {"membership": row["membership"]},
    }


def _positive(value: Any) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )
