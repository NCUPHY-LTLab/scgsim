"""The entries of ``palace_index_map.json``, and the EPR request checked against them.

An entry says which physical object one row of ``config.json`` is: its section and index name the row, and its
fields say what the row covers.  There is one record per section: ``PortEntry`` (``Boundaries.LumpedPort``),
``TerminalEntry`` (``Boundaries.Terminal``), ``EnergyEntry`` (``Domains.Postprocessing.Energy``) and
``SurfaceEprEntry`` (``Boundaries.Postprocessing.Dielectric``).  ``to_payload()`` writes an entry as the index
map holds it, key for key.

``EprRequest`` is the Eigenmode selection of reported EPR quantities (``set_epr_request``); ``check_epr_request``
refuses one that selects something the entries of the config being written do not list.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

SURFACE_INTERFACES = ("MA", "MS", "SA")
EPR_SELECTION_SCHEMA = "scgsim.palace.epr-selection.v1"


@dataclass(frozen=True)
class PortEntry:
    """The ``LumpedPort`` row of one requested port: Palace port ``P<index>`` on its authored sheet.

    ``sheet`` is the sheet's identity in the mesh: its id, the conductors it lands on and the volumes around it.
    """

    section: ClassVar[str] = "Boundaries.LumpedPort"

    index: int
    entry_name: str
    attributes: list[int]
    physical_names: list[str]
    port_name: str
    sheet: dict[str, Any]

    def to_payload(self) -> dict[str, Any]:
        """The entry as the index map holds it."""
        return {
            "section": self.section,
            "index": self.index,
            "entry_name": self.entry_name,
            "role": "port_surface",
            "attributes": self.attributes,
            "physical_names": self.physical_names,
            "port_name": self.port_name,
            "metadata": self.sheet,
        }


@dataclass(frozen=True)
class TerminalEntry:
    """The ``Terminal`` row of one Electrostatic terminal: the conductor surfaces of one exact Net."""

    section: ClassVar[str] = "Boundaries.Terminal"

    index: int
    name: str
    net_id: str
    conductor_component_ids: list[str]
    attributes: list[int]
    physical_names: list[str]

    def to_payload(self) -> dict[str, Any]:
        """The entry as the index map holds it; the Net and conductors are written twice, as readers expect."""
        return {
            "section": self.section,
            "index": self.index,
            "entry_name": self.name,
            "role": "terminal",
            "attributes": self.attributes,
            "physical_names": self.physical_names,
            "metadata": {
                "net_id": self.net_id,
                "conductor_component_ids": self.conductor_component_ids,
            },
            "terminal_name": self.name,
            "net_id": self.net_id,
            "conductor_component_ids": self.conductor_component_ids,
            "terminal_attributes": self.attributes,
        }


@dataclass(frozen=True)
class EnergyEntry:
    """The ``Energy`` row of one solution volume: a bulk domain named by its group, with its material."""

    section: ClassVar[str] = "Domains.Postprocessing.Energy"

    index: int
    name: str
    attributes: list[int]
    material: dict[str, str]
    entities: list[str]

    def to_payload(self) -> dict[str, Any]:
        """The entry as the index map holds it."""
        return {
            "section": self.section,
            "index": self.index,
            "entry_name": self.name,
            "role": "domain_energy",
            "physical_name": self.name,
            "physical_names": [self.name],
            "attributes": self.attributes,
            "metadata": {"material": self.material, "entities": self.entities},
        }


@dataclass(frozen=True)
class SurfaceRow:
    """One sidecar Surface-EPR row a ``Dielectric`` row integrates: its identity, sides, owners, Net and lineage.

    ``shared_face_group`` is set exactly for a Route A conductor sheet, whose MA and MS rows integrate one face
    from its two sides.  Lineage ``membership`` is attribution only: it weights nothing.
    """

    row: str
    group: str
    interface_type: str
    face_kind: str
    band: int | None
    side: Any
    face_sides: Any
    shared_face_group: str | None
    entities: list[str]
    net: str | None
    attributes: list[int]
    membership: Any

    def to_payload(self) -> dict[str, Any]:
        """The row as the index map holds it."""
        return {
            "row": self.row,
            "group": self.group,
            "interface_type": self.interface_type,
            "face_kind": self.face_kind,
            "band": self.band,
            "side": self.side,
            "face_sides": self.face_sides,
            "shared_face_group": self.shared_face_group,
            "entities": self.entities,
            "net": self.net,
            "attributes": self.attributes,
            "lineage": {"membership": self.membership},
        }


@dataclass(frozen=True)
class InsetSelection:
    """How an inset row was chosen: the margin (µm) asked for, and the bands from that margin's band on."""

    margin_um: float
    margin_index: int
    first_band: int | None
    bands: list[int]

    def to_payload(self) -> dict[str, Any]:
        """The selection as the index map holds it."""
        return {
            "margin_um": self.margin_um,
            "margin_index": self.margin_index,
            "first_band": self.first_band,
            "bands": self.bands,
        }


@dataclass(frozen=True)
class SurfaceEprEntry:
    """The ``Dielectric`` row of one surface unit, whole (the baseline) or from one inset margin inward.

    ``epr_spec`` is the spec the row was written from, with its thickness in native units.  A baseline row is its
    own ``baseline_index``; an inset row names its baseline and carries its ``inset`` selection.
    """

    section: ClassVar[str] = "Boundaries.Postprocessing.Dielectric"

    index: int
    entry_name: str
    attributes: list[int]
    physical_names: list[str]
    rows: tuple[SurfaceRow, ...]
    epr_spec: dict[str, Any]
    baseline_index: int
    inset: InsetSelection | None = None

    def to_payload(self) -> dict[str, Any]:
        """The entry as the index map holds it."""
        payload: dict[str, Any] = {
            "section": self.section,
            "index": self.index,
            "entry_name": self.entry_name,
            "role": "surface_epr",
            "attributes": self.attributes,
            "physical_names": self.physical_names,
            "metadata": {"rows": [row.to_payload() for row in self.rows]},
            "epr_spec": self.epr_spec,
            "baseline_index": self.baseline_index,
        }
        if self.inset is not None:
            payload["inset"] = self.inset.to_payload()
        return payload


IndexEntry = PortEntry | TerminalEntry | EnergyEntry | SurfaceEprEntry


@dataclass(frozen=True)
class EprRequest:
    """The EPR quantities an Eigenmode run reports; None selects every configured one, ``()`` none.

    ``surface_interfaces`` holds ``MA``, ``MS`` or ``SA``; ``bulk_domain_ids`` the ``entry_name`` of
    ``Energy`` entries; ``port_names`` the names of requested ports.  The selection changes no physical solve.
    """

    surface_interfaces: tuple[str, ...] | None = None
    bulk_domain_ids: tuple[str, ...] | None = None
    port_names: tuple[str, ...] | None = None

    @classmethod
    def selected(
        cls,
        *,
        surface_interfaces: Sequence[str] | None,
        bulk_domain_ids: Sequence[str] | None,
        port_names: Sequence[str] | None,
    ) -> EprRequest:
        """Check the three selections of ``set_epr_request`` and hold them as tuples of distinct IDs.

        Raises:
            TypeError: a selection is not a sequence of IDs.
            ValueError: a selection holds an empty or repeated ID, or an interface other than MA, MS, SA.
        """
        surfaces = _selection(surface_interfaces, "surface_interfaces")
        if surfaces is not None and set(surfaces) - set(SURFACE_INTERFACES):
            raise ValueError(
                "surface_interfaces supports only MA, MS, and SA: call "
                "set_epr_request(surface_interfaces=...) with those."
            )
        return cls(
            surfaces,
            _selection(bulk_domain_ids, "bulk_domain_ids"),
            _selection(port_names, "port_names"),
        )

    def to_payload(self) -> dict[str, Any]:
        """The request as the index map records it."""
        return {
            "schema_version": EPR_SELECTION_SCHEMA,
            "surface_interfaces": _listed(self.surface_interfaces),
            "bulk_domain_ids": _listed(self.bulk_domain_ids),
            "port_names": _listed(self.port_names),
        }


def _selection(values: Sequence[str] | None, name: str) -> tuple[str, ...] | None:
    """One selection as a tuple of distinct non-empty IDs, or None for all."""
    if values is None:
        return None
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(
            f"{name} must be a sequence of IDs or None: call set_epr_request({name}=[...])."
        )
    normalized = tuple(values)
    if any(not isinstance(value, str) or not value for value in normalized):
        raise ValueError(
            f"{name} must contain non-empty text IDs: call set_epr_request({name}=[...])."
        )
    if len(normalized) != len(set(normalized)):
        raise ValueError(
            f"{name} must not repeat IDs: call set_epr_request({name}=[...]) with each once."
        )
    return normalized


def _listed(values: tuple[str, ...] | None) -> list[str] | None:
    return None if values is None else list(values)


def check_epr_request(
    request: EprRequest | None, entries: Sequence[IndexEntry]
) -> None:
    """Refuse an EPR request that selects a binding the entries of the config being written do not list.

    Raises:
        ValueError: a selected interface, bulk domain or port has no entry.
    """
    if request is None:
        return
    offered = {
        "surface_interfaces": (
            request.surface_interfaces,
            {
                row.interface_type
                for entry in entries
                if isinstance(entry, SurfaceEprEntry)
                for row in entry.rows
            },
        ),
        "bulk_domain_ids": (
            request.bulk_domain_ids,
            {entry.name for entry in entries if isinstance(entry, EnergyEntry)},
        ),
        "port_names": (
            request.port_names,
            {entry.port_name for entry in entries if isinstance(entry, PortEntry)},
        ),
    }
    for name, (chosen, configured) in offered.items():
        unknown = sorted(set(chosen or ()) - configured)
        if unknown:
            raise ValueError(
                f"EPR request selects an unknown {name} binding {unknown}; the config has "
                f"{sorted(configured)}: call set_epr_request() with those, then write_config()."
            )


__all__ = [
    "EPR_SELECTION_SCHEMA",
    "SURFACE_INTERFACES",
    "EnergyEntry",
    "EprRequest",
    "IndexEntry",
    "InsetSelection",
    "PortEntry",
    "SurfaceEprEntry",
    "SurfaceRow",
    "TerminalEntry",
    "check_epr_request",
]
