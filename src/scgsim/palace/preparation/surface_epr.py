"""Surface-EPR lowering: Palace ``Dielectric`` rows over the attributes of SGB's rows and bands.

``inset_margins_um`` are geometry: SGB cuts them into the model as bands (``mesh()`` requests them from
these specs), and a margin ``m_k`` is the union of the attributes of bands ``k`` to ``K`` of its support,
listed as plain attributes.  Stock Palace integrates plain attributes; native Mask/Inset is neither written
nor accepted.  Omit ``inset_margins_um`` for baseline rows only.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any, cast

from scgsim.semantics.epr import film_assumptions

from ..mesh.physical_groups import Groups, rows_of, same_length


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
    """Retain the ordered, strictly increasing positive margins SGB cuts into the model as bands."""
    if "inset_margins_um" in raw:
        inset_margins_um = raw["inset_margins_um"]
        if isinstance(inset_margins_um, (str, bytes)) or not isinstance(
            inset_margins_um, (list, tuple)
        ):
            raise TypeError(
                f"Surface EPR {interface_type} inset_margins_um must be an ordered non-empty sequence."
            )
        if not inset_margins_um:
            raise ValueError(
                f"Surface EPR {interface_type} inset_margins_um must be non-empty."
            )
        normalized_margins = []
        for margin in inset_margins_um:
            if not _positive(margin):
                raise ValueError(
                    f"Surface EPR {interface_type} inset_margins_um values must be finite positive reals "
                    "(the whole surface is the baseline row)."
                )
            normalized_margins.append(float(margin))
        if any(
            b <= a
            for a, b in zip(normalized_margins, normalized_margins[1:], strict=False)
        ):
            raise ValueError(
                f"Surface EPR {interface_type} inset_margins_um must be strictly increasing."
            )
        resolved["inset_margins_um"] = normalized_margins


def build_surface_epr_postprocessing(
    groups: Groups,
    specs: Mapping[str, Mapping[str, Any]],
    *,
    model_l0_m: float = 1e-6,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Emit Palace ``Dielectric`` rows and their index entries (index map schema 3) from the Surface-EPR rows of ``groups``.

    For each spec (interface type) and each surface unit of that type -- an unbanded row, or all bands of
    one support -- one baseline row covers every attribute of the unit.  For each inset margin of the spec,
    a further row covers the attributes of the unit's bands from the margin's band to the last.
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
                for margin_index, margin in enumerate(
                    preset.get("inset_margins_um", ())
                )
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
    """The rows of one interface type that one baseline row covers: an unbanded row, or all bands of a support."""

    name: str
    interface_type: str
    rows: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class _MarginRequest:
    """One inset margin of a spec on one unit."""

    baseline_index: int
    unit: _Unit
    margin_index: int
    margin_um: float

    def first_band(self) -> int | None:
        """Index of the first band at least ``margin_um`` from the gap edge, or None for an unbanded unit."""
        margins = next(
            (row["margins_um"] for row in self.unit.rows if row["band"] is not None),
            None,
        )
        if margins is None:
            return None
        for position, margin in enumerate(margins):
            if same_length(margin, self.margin_um):
                return position + 1
        raise ValueError(
            f"Surface EPR {self.unit.interface_type} margin {self.margin_um} um is not among "
            f"the margins {margins} of the bands of {self.unit.name!r}."
        )

    def selected_rows(self) -> tuple[Mapping[str, Any], ...]:
        """The rows of the bands from the margin's band to the last."""
        first = self.first_band()
        if first is None:
            return ()
        return tuple(
            row
            for row in self.unit.rows
            if row["band"] is not None and row["band"] >= first
        )

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
    """The surface units of one interface type, in the established sorted order."""
    face_filter = preset.get("face_kinds")
    members: dict[str, list[Mapping[str, Any]]] = {}
    for row_id, row in sorted(rows.items()):
        if row["interface_type"] != interface_type:
            continue
        if face_filter is not None and row["face_kind"] not in face_filter:
            continue
        members.setdefault(_unit_name(row_id, row), []).append({**row, "row": row_id})
    return [
        _Unit(name, interface_type, tuple(sorted(items, key=_band_order)))
        for name, items in members.items()
    ]


def _unit_name(row_id: str, row: Mapping[str, Any]) -> str:
    """The row id without its band suffix: all bands of one support share a unit."""
    return row_id if row["band"] is None else row_id.rsplit("::band=", 1)[0]


def _band_order(row: Mapping[str, Any]) -> tuple[int, str]:
    return (-1 if row["band"] is None else row["band"], row["row"])


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
