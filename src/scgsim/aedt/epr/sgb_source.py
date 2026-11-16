"""What the EPR of an imported SGB model integrates, read from its sidecar and its prepared record (issue #67).

:func:`sgb_epr_source` reads, without AEDT:

1. the **domains**: each imported dielectric solid, by its AEDT object name, and the ``Region``, the vacuum
   AEDT creates from the plan's ``VACUUM_REGION`` box: one domain however many vacuum volumes the box holds,
   since they share one material.  A conductor solid is PEC, so it holds no field;
2. the **films**: one per interface type the prepared Surface-EPR specs declare, ``MA`` (metal-air), ``MS``
   (metal-substrate) or ``SA`` (substrate-air), with its thickness in metres, permittivity and loss tangent;
3. the **rows**: every sidecar Surface-EPR row of a declared type on a selected group (``surface_epr_groups``,
   ``None`` for every group), each with the permittivity of the dielectric on its side and the name of the
   non-model sheet AEDT integrates it on (:mod:`scgsim.aedt.epr.sgb_surfaces`);
4. the **entries**: the contributions the result reports, the way Palace's ``Dielectric`` rows group the
   same rows (``scgsim.palace.config.surface_epr``): per interface type and surface unit (an unbanded row, or
   every band of one band family) a baseline over all its rows, and per inset margin of the unit an entry
   over its bands from the margin's band to the last.  The entry names are Palace's, so the two backends'
   contributions are told apart by name only;
5. the **junction sheets**: each junction's port faces, the sidecar's direction from its first host conductor
   to its second, and the sheet's width across that direction, by which the area integral of the field along
   the direction is divided to give the junction voltage.

Each row is integrated once; an entry sums its rows' energies.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from scgsim.aedt.preparation.sgb_model import PreparedSgbModel, port_width_um
from scgsim.aedt.specs.epr import PlanarJunction
from scgsim.contracts.sgb_sidecar import Sidecar, read_sidecar
from scgsim.contracts.sgb_sidecar.records import SurfaceGroup
from scgsim.geometry.exact.grid import Grid
from scgsim.geometry.sgb.output.sidecar.aedt import aedt_names
from scgsim.geometry.surface_epr.specs import margins_of

REGION = "Region"
SURFACE_TYPES = ("MA", "MS", "SA")
BASELINE = "unmasked_baseline"
MARGIN = "requested_margin"


@dataclass(frozen=True)
class EprDomain:
    """One region the field energy is integrated over: an imported dielectric solid, or the Region, by AEDT name."""

    object_name: str
    material_id: str
    relative_permittivity: float


@dataclass(frozen=True)
class SurfaceFilm:
    """The lossy film one interface type assumes: thickness (m), relative permittivity, loss tangent and source."""

    interface_type: str
    film_thickness_m: float
    permittivity: float
    loss_tangent: float | None
    source: str
    preset: str | None

    def assumptions(self) -> dict[str, Any]:
        """The film as the result records its assumptions."""
        return {
            "film_thickness_m": self.film_thickness_m,
            "film_relative_permittivity": self.permittivity,
            "loss_tangent": self.loss_tangent,
            "source": self.source,
            "preset": self.preset,
        }


@dataclass(frozen=True)
class EprSurfaceRow:
    """One sidecar Surface-EPR row to integrate and the AEDT sheet that carries its faces.

    ``substrate_relative_permittivity`` is that of the dielectric on the row's side (``MS``, ``SA``); an ``MA``
    row's energy does not depend on it, so it is ``None`` there.
    """

    row: str
    group: str
    interface_type: str
    band: int | None
    family: str | None
    banded_surface: str | None
    entities: tuple[str, ...]
    face_kind: str
    substrate_relative_permittivity: float | None
    selection_name: str


@dataclass(frozen=True)
class SurfaceEntry:
    """One reported Surface-EPR contribution: a unit's baseline (every row) or one of its inset margins."""

    entry_id: str
    interface_type: str
    evaluation_kind: str
    margin_um: float
    rows: tuple[str, ...]
    entities: tuple[str, ...]
    substrate_relative_permittivity: float | None


@dataclass(frozen=True)
class JunctionSheet:
    """One junction, the faces of the port sheet it sits on, and the line the voltage is read along."""

    junction: PlanarJunction
    port_group: str
    faces: tuple[str, ...]
    direction: tuple[float, float, float]
    width_um: float


@dataclass(frozen=True)
class SgbEprSource:
    """Everything the EPR of one SGB model integrates and how its integrals combine (see the module)."""

    domains: Mapping[str, EprDomain]
    films: Mapping[str, SurfaceFilm]
    rows: Mapping[str, EprSurfaceRow]
    entries: tuple[SurfaceEntry, ...]
    junctions: Mapping[str, JunctionSheet]


def sgb_epr_source(
    model: PreparedSgbModel,
    *,
    surface_epr_groups: Sequence[str] | None = None,
    sidecar: Sidecar | None = None,
) -> SgbEprSource:
    """Read what the EPR of ``model`` integrates; ``surface_epr_groups`` selects sidecar groups (``None``: all)."""
    sidecar = read_sidecar(model.path("sidecar")) if sidecar is None else sidecar
    films = surface_films(model.surface_epr_specs)
    rows = surface_rows(model, sidecar, films, surface_epr_groups)
    return SgbEprSource(
        domains=epr_domains(model, sidecar),
        films=films,
        rows=rows,
        entries=surface_entries(sidecar, model.surface_epr_specs or {}, rows),
        junctions=junction_sheets(model, sidecar),
    )


# ---------------------------------------------------------------- domains


def epr_domains(model: PreparedSgbModel, sidecar: Sidecar) -> dict[str, EprDomain]:
    """The imported dielectric solids by AEDT object name, then the Region of the vacuum's material."""
    out: dict[str, EprDomain] = {}
    vacuum: set[str] = set()
    for volume_id, volume in sorted(sidecar["volume"].items()):
        if volume["kind"] == "vacuum":
            vacuum.add(volume["material"])
        elif volume["kind"] == "dielectric" and volume["emitted"]:
            name = volume["aedt_name"]
            if name is None:
                raise ValueError(f"the emitted volume {volume_id!r} has no AEDT name")
            out[name] = _domain(model, name, volume["material"])
    # The vacuum may be several volumes (a substrate spanning the box splits it above and below); the Region
    # is one box, so they must share one material, as the import requires.
    if len(vacuum) != 1:
        raise ValueError(
            f"the SGB model's vacuum volumes have materials {sorted(vacuum)}; the Region is one box "
            "of one material"
        )
    if REGION in out:
        raise ValueError(f"the SGB model has a dielectric solid named {REGION!r}")
    out[REGION] = _domain(model, REGION, vacuum.pop())
    return out


def _domain(model: PreparedSgbModel, object_name: str, material_id: str) -> EprDomain:
    permittivity = model.materials[material_id].get("permittivity")
    if (
        isinstance(permittivity, bool)
        or not isinstance(permittivity, (int, float))
        or not permittivity > 0
    ):
        raise ValueError(
            f"material {material_id!r} of {object_name!r} has no positive permittivity"
        )
    return EprDomain(object_name, material_id, float(permittivity))


# ---------------------------------------------------------------- films and rows


def surface_films(
    specs: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, SurfaceFilm]:
    """The film of each declared interface type.

    AEDT has no Palace ``Model.L0``, so a spec must give its thickness as ``film_thickness_m``; a spec in
    Palace's native ``thickness`` is refused rather than given a length unit here.
    """
    out = {}
    for interface_type, spec in sorted((specs or {}).items()):
        if spec.get("film_thickness_m") is None:
            raise ValueError(
                f"Surface EPR {interface_type}: AEDT needs the film thickness as film_thickness_m "
                "(metres); the native thickness is in Palace's Model.L0"
            )
        out[interface_type] = SurfaceFilm(
            interface_type,
            float(spec["film_thickness_m"]),
            float(spec["permittivity"]),
            None if spec.get("loss_tangent") is None else float(spec["loss_tangent"]),
            str(spec["source"]),
            spec.get("preset"),
        )
    return out


def row_selection_names(sidecar: Sidecar) -> dict[str, str]:
    """Row id -> the AEDT name of the non-model sheet that carries the row's faces (``epr_<row>``)."""
    return aedt_names(
        [(row, f"epr_{row}") for row in sorted(sidecar["surface_epr_row"])]
    )


def surface_rows(
    model: PreparedSgbModel,
    sidecar: Sidecar,
    films: Mapping[str, SurfaceFilm],
    surface_epr_groups: Sequence[str] | None,
) -> dict[str, EprSurfaceRow]:
    """The rows of the declared types on the selected groups, filtered by each spec's ``face_kinds``.

    A group named in ``surface_epr_groups`` that carries no row of a declared type is refused, and so is a
    required spec (Palace's ``required``) with no row in the model, whatever the selection.
    """
    names = row_selection_names(sidecar)
    specs = model.surface_epr_specs or {}
    available = {
        row["group"]
        for row in sidecar["surface_epr_row"].values()
        if row["interface_type"] in films
    }
    selected = set(available) if surface_epr_groups is None else set(surface_epr_groups)
    unknown = sorted(selected - available)
    if unknown:
        raise ValueError(
            f"surface_epr_groups {unknown} carry no Surface-EPR row of a declared type; "
            f"the model's groups are {sorted(available)}"
        )
    out = {}
    present: set[str] = set()
    for row_id, row in sorted(sidecar["surface_epr_row"].items()):
        interface_type = row["interface_type"]
        group = surface_group(sidecar, row["group"])
        face_kinds = specs.get(interface_type, {}).get("face_kinds")
        if interface_type not in films or (
            face_kinds is not None and group["face_kind"] not in face_kinds
        ):
            continue
        present.add(interface_type)
        if row["group"] not in selected:
            continue
        out[row_id] = EprSurfaceRow(
            row=row_id,
            group=row["group"],
            interface_type=interface_type,
            band=row["band"],
            family=None if group["band"] is None else group["band"]["family"],
            banded_surface=row["banded_surface"],
            entities=tuple(group["entities"]),
            face_kind=group["face_kind"],
            substrate_relative_permittivity=_substrate_permittivity(
                model, row_id, group, interface_type
            ),
            selection_name=names[row_id],
        )
    for interface_type in films:
        if (
            specs[interface_type].get("required", True)
            and interface_type not in present
        ):
            raise ValueError(
                f"Required Surface EPR {interface_type} has no row in the SGB model; pass required=False "
                f"in its spec, or plan.set_surface_epr() without {interface_type}"
            )
    return out


def surface_group(sidecar: Sidecar, name: str) -> SurfaceGroup:
    """The surface group ``name``; a Surface-EPR row and a port always name one (dimension 2)."""
    group = sidecar["group"][name]
    if group["dim"] != 2:
        raise ValueError(f"group {name!r} is not a surface group")
    return cast(SurfaceGroup, group)


def _substrate_permittivity(
    model: PreparedSgbModel, row_id: str, group: SurfaceGroup, interface_type: str
) -> float | None:
    """The permittivity of the one dielectric on the row's side; ``MA`` has none it depends on."""
    if interface_type == "MA":
        return None
    materials = {
        d["material"] for d in group["adjacent_domains"] if d["kind"] == "dielectric"
    }
    if len(materials) != 1:
        raise ValueError(
            f"Surface-EPR row {row_id} ({interface_type}) borders dielectrics {sorted(materials)}, not one"
        )
    return float(model.materials[materials.pop()]["permittivity"])


# ---------------------------------------------------------------- entries


def surface_entries(
    sidecar: Sidecar,
    specs: Mapping[str, Mapping[str, Any]],
    rows: Mapping[str, EprSurfaceRow],
) -> tuple[SurfaceEntry, ...]:
    """Every baseline entry, then every margin entry, in Palace's order (see the module)."""
    baselines: list[SurfaceEntry] = []
    margins: list[SurfaceEntry] = []
    for interface_type in sorted({row.interface_type for row in rows.values()}):
        for unit in _units(rows, interface_type):
            baselines.append(
                _entry(unit.name, interface_type, BASELINE, 0.0, unit.rows)
            )
            if unit.family is None:
                continue
            margins_um = margins_of(specs[interface_type], unit.rows[0].entities)
            for index, margin in enumerate(margins_um):
                first = _first_band(sidecar, unit.family, margin, unit.name)
                selected = [row for row in unit.rows if (row.band or 0) >= first]
                if selected:
                    name = f"{unit.name}__inset_{index}"
                    margins.append(
                        _entry(name, interface_type, MARGIN, margin, selected)
                    )
    return (*baselines, *margins)


@dataclass(frozen=True)
class _Unit:
    """The rows of one type one baseline covers: an unbanded row, or every band of one band family."""

    name: str
    rows: list[EprSurfaceRow]
    family: str | None


def _units(rows: Mapping[str, EprSurfaceRow], interface_type: str) -> list[_Unit]:
    """The units of one type, by name.

    An unbanded unit is named by its row id, a banded one by the surface its rows are bands of; an unbanded
    unit whose name a family also has gets ``__unbanded``, as in Palace.
    """
    singles: list[_Unit] = []
    by_family: dict[str, list[EprSurfaceRow]] = {}
    for row in sorted(rows.values(), key=lambda item: item.row):
        if row.interface_type != interface_type:
            continue
        if row.family is None or row.banded_surface is None:
            singles.append(_Unit(row.row, [row], None))
        else:
            by_family.setdefault(row.family, []).append(row)
    banded = [
        _Unit(
            str(members[0].banded_surface),
            sorted(members, key=lambda item: (item.band or 0, item.row)),
            family,
        )
        for family, members in by_family.items()
    ]
    taken = {unit.name for unit in banded}
    singles = [
        _Unit(f"{unit.name}__unbanded", unit.rows, None) if unit.name in taken else unit
        for unit in singles
    ]
    return sorted([*singles, *banded], key=lambda unit: unit.name)


def _first_band(
    sidecar: Sidecar, family_id: str, margin_um: float, unit_name: str
) -> int:
    """The first band at least ``margin_um`` from the gap edge: the band whose margin is the same grid multiple."""
    family = sidecar["band_family"][family_id]
    wanted = Grid(sidecar["model"]["grid_um"]).lattice_units(margin_um)
    if wanted is None or wanted not in family["margins_grid"]:
        raise ValueError(
            f"Surface EPR margin {margin_um} um is not among the margins {family['margins_um']} "
            f"of the bands of {unit_name!r}"
        )
    return list(family["margins_grid"]).index(wanted) + 1


def _entry(
    entry_id: str,
    interface_type: str,
    evaluation_kind: str,
    margin_um: float,
    rows: Iterable[EprSurfaceRow],
) -> SurfaceEntry:
    rows = list(rows)
    permittivities = {row.substrate_relative_permittivity for row in rows}
    if len(permittivities) != 1:
        raise ValueError(
            f"Surface-EPR entry {entry_id} spans dielectrics of permittivities {permittivities}"
        )
    return SurfaceEntry(
        entry_id=entry_id,
        interface_type=interface_type,
        evaluation_kind=evaluation_kind,
        margin_um=float(margin_um),
        rows=tuple(row.row for row in rows),
        entities=tuple(sorted({entity for row in rows for entity in row.entities})),
        substrate_relative_permittivity=permittivities.pop(),
    )


# ---------------------------------------------------------------- junctions


def junction_sheets(
    model: PreparedSgbModel, sidecar: Sidecar
) -> dict[str, JunctionSheet]:
    """Each junction's port faces, direction and width across it, from the sidecar's port record."""
    ports = {port["group"]: port for port in sidecar["port"].values()}
    out = {}
    for item in model.junctions:
        port = ports[item.port_group]
        # The voltage is read along the port's in-plane direction, the same one the Lumped RLC's current line
        # follows (scgsim.aedt.runtime.families.sgb_import).
        dx, dy = float(port["direction"][0]), float(port["direction"][1])
        norm = math.hypot(dx, dy)
        direction = (dx / norm, dy / norm, 0.0)
        out[item.junction.junction_id] = JunctionSheet(
            junction=item.junction,
            port_group=item.port_group,
            faces=tuple(port["faces"]),
            direction=direction,
            width_um=port_width_um(sidecar, port["faces"], direction),
        )
    return out
