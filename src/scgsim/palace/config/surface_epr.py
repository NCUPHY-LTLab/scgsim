"""Surface-EPR lowering: Palace ``Dielectric`` rows over the attributes of SGB's rows and bands.

``inset_margins_um`` and ``inset_margins_by_entity_um`` are geometry: SGB cuts them into the model as bands
(``mesh()`` requests them from these specs), and a margin ``m_k`` is the union of the attributes of bands
``k`` to ``K`` of every band family that serves the spec, listed as plain attributes.  The bands of a family
come from the sidecar's tables (``band_family[*].bands``, the family a ``surface_epr_row``'s group is a band
of), never from group names.  Stock Palace integrates plain attributes; native Mask/Inset is neither written
nor accepted.  Omit both margin keys for baseline rows only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from scgsim.geometry.sgb.exact.grid import Grid
from scgsim.geometry.surface_epr.specs import margins_of, positive_real

from ..mesh.physical_groups import Groups, rows_of


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
    if not positive_real(model_l0_m):
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
