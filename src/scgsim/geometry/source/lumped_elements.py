"""Lumped elements on the plan: the declarations a study gives and the records ``prepare()`` resolves them to.

A lumped element is a fact of the device, the same for every solver: the authored port sheet it sits on, the
two conductors that sheet lands on, and its circuit.  ``LumpedElement`` is a resistance, an inductance and a
capacitance in parallel, any of them absent (a junction as its inductance included); ``LumpedPort`` is an
excitation port, with the Net of its reference conductor and its reference impedance.
``GeometryPlan.set_lumped_elements`` takes them (``checked_lumped_elements``); ``prepare()`` resolves each one on
the port sheets the source stage found (``resolved_lumped_elements``), refusing what cannot be one with the
sheet's layer and place, and the snapshot carries the ``LumpedElementRecord``s every simulation reads.

The records are numbered once, here: the ports first, in the order given (port *i* is row and column *i* of
the S-matrix), then the other elements, in the order given.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from scgsim.geometry.models.grid import format_box, geometry_grid_context
from scgsim.geometry.models.input import GeometryBuildInput, SemanticEntitySpec
from scgsim.geometry.models.junctions import (
    JunctionTerminalBinding,
    junction_net_id,
    junction_occurrence_path,
)
from scgsim.geometry.models.regions import PortSheetRegionRecord
from scgsim.refusal import Declared, shown

_SET_LUMPED_ELEMENTS = "the call to GeometryPlan.set_lumped_elements"
_FIX_THE_LAYOUT = "the layout"
# Each circuit component's field, its symbol and its unit, in the order R, L, C.
_CIRCUIT_UNITS = {
    "resistance_ohm": ("R", "ohm"),
    "inductance_h": ("L", "H"),
    "capacitance_f": ("C", "F"),
}


@dataclass(frozen=True)
class LumpedElement:
    """A resistance, an inductance and a capacitance in parallel on one authored port sheet; ``None`` is absent.

    ``sheet`` is the qualified name of the port the component authors (``"chip/o_junction_lumped"``).  The
    parallel circuit is the one Palace's ``LumpedPort`` and HFSS's parallel Lumped RLC both are.  A junction is
    an element with an inductance and optionally a capacitance.

    Raises:
        ValueError: ``sheet`` is empty, a given value is not a finite number above zero, or none is given.
    """

    sheet: str
    resistance_ohm: float | None = None
    inductance_h: float | None = None
    capacitance_f: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "sheet", _sheet_name(self.sheet, "LumpedElement"))
        given = False
        for field in _CIRCUIT_UNITS:
            value = getattr(self, field)
            if value is None:
                continue
            object.__setattr__(
                self, field, _positive(value, field, f"LumpedElement({self.sheet!r})")
            )
            given = True
        if not given:
            raise ValueError(
                _declared(f"LumpedElement({self.sheet!r})").refusal(
                    "give at least one of resistance_ohm, inductance_h and capacitance_f"
                )
            )

    @property
    def circuit(self) -> dict[str, float]:
        """The given components by field name, in the order R, L, C."""
        return {
            field: value
            for field in _CIRCUIT_UNITS
            if (value := getattr(self, field)) is not None
        }

    def values_text(self) -> str:
        """The given components as a refusal or a report prints them (``L 1e-08 H``)."""
        return ", ".join(
            f"{_CIRCUIT_UNITS[field][0]} {value!r} {_CIRCUIT_UNITS[field][1]}"
            for field, value in self.circuit.items()
        )


@dataclass(frozen=True)
class LumpedPort:
    """An excitation port on one authored port sheet: its reference conductor's Net and its reference impedance.

    The conductor of the sheet in ``reference_net`` is the port's reference; the other is its signal.

    Raises:
        ValueError: ``sheet`` or ``reference_net`` is empty, or ``impedance_ohm`` is not a finite number above
            zero.
    """

    sheet: str
    reference_net: str
    impedance_ohm: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "sheet", _sheet_name(self.sheet, "LumpedPort"))
        if not isinstance(self.reference_net, str) or not self.reference_net:
            raise ValueError(
                _declared(f"LumpedPort({self.sheet!r})").expected(
                    "reference_net", "the nonempty ID of a Net", self.reference_net
                )
            )
        object.__setattr__(
            self,
            "impedance_ohm",
            _positive(
                self.impedance_ohm, "impedance_ohm", f"LumpedPort({self.sheet!r})"
            ),
        )

    def values_text(self) -> str:
        """The port's facts as a refusal or a report prints them."""
        return f"reference Net {self.reference_net!r}, impedance {self.impedance_ohm!r} ohm"


Declaration = LumpedElement | LumpedPort


@dataclass(frozen=True)
class LumpedElementRecord:
    """One declared lumped element as ``prepare()`` resolved it on the layout.

    ``number`` counts from 1, the ports first (see the module).  ``port_sheet_id``, ``source_layer``
    (``layer/datatype``), ``target_layer`` and ``direction`` are the authored sheet's; ``start`` and ``end`` are
    the conductor at the start of the direction and the one it points to, each with its occurrence and Net;
    ``box_um`` is the sheet's bounding box ``(x0, y0, x1, y1)``, where a refusal says the sheet is, printed on the
    model grid ``grid_um`` (None for an input without one).
    """

    number: int
    declaration: Declaration
    port_sheet_id: str
    source_polygon_id: str
    source_layer: str
    target_layer: str
    direction: tuple[float, ...]
    start: JunctionTerminalBinding
    end: JunctionTerminalBinding
    box_um: tuple[float, float, float, float]
    grid_um: float | None

    @property
    def name(self) -> str:
        """The authored sheet name, the element's name on every solver."""
        return self.declaration.sheet

    @property
    def is_port(self) -> bool:
        """Whether the element is an excitation port."""
        return isinstance(self.declaration, LumpedPort)

    @property
    def reference(self) -> JunctionTerminalBinding:
        """A port's reference conductor, the one in its ``reference_net``."""
        port = self._port()
        return self.start if self.start.net_id == port.reference_net else self.end

    @property
    def signal(self) -> JunctionTerminalBinding:
        """A port's signal conductor, the one not in its ``reference_net``."""
        return self.end if self.reference is self.start else self.start

    def place(self) -> str:
        """Where the sheet is drawn: its layer and its box in micrometres."""
        return (
            f"on layer {self.source_layer} at {format_box(self.box_um, self.grid_um)}"
        )

    def describe(self) -> str:
        """The element as a refusal or a report names it: its name, its kind and its values."""
        kind = "lumped port" if self.is_port else "lumped element"
        return f"{kind} {self.name!r} ({self.declaration.values_text()})"

    def to_payload(self) -> dict[str, Any]:
        """The record as ``input_summary()`` shows it."""
        payload: dict[str, Any] = {
            "number": self.number,
            "name": self.name,
            "kind": "port" if self.is_port else "element",
            "port_sheet_id": self.port_sheet_id,
            "layer": self.source_layer,
            "target_layer": self.target_layer,
            "start": self.start.to_payload(),
            "end": self.end.to_payload(),
        }
        if isinstance(self.declaration, LumpedPort):
            payload |= {
                "reference_net": self.declaration.reference_net,
                "impedance_ohm": self.declaration.impedance_ohm,
                "signal": self.signal.source_entity_id,
                "reference": self.reference.source_entity_id,
            }
        else:
            payload |= self.declaration.circuit
        return payload

    def _port(self) -> LumpedPort:
        if not isinstance(self.declaration, LumpedPort):
            raise TypeError(f"{self.describe()} is not an excitation port")
        return self.declaration


def checked_lumped_elements(elements: object) -> tuple[Declaration, ...]:
    """``elements`` as the plan keeps them: a sequence of declarations, each naming its own sheet.

    Raises:
        TypeError: ``elements`` is not a sequence, or an entry is neither a ``LumpedElement`` nor a
            ``LumpedPort``.
        ValueError: two entries name one sheet.
    """
    where = Declared("lumped elements", _SET_LUMPED_ELEMENTS)
    if isinstance(elements, (str, bytes, Mapping)) or not isinstance(
        elements, Sequence
    ):
        raise TypeError(
            where.expected(
                "elements", "a sequence of LumpedElement and LumpedPort", elements
            )
        )
    declarations = tuple(elements)
    for entry in declarations:
        if not isinstance(entry, (LumpedElement, LumpedPort)):
            raise TypeError(
                where.expected("every entry", "a LumpedElement or a LumpedPort", entry)
            )
    sheets = [entry.sheet for entry in declarations]
    repeated = sorted({sheet for sheet in sheets if sheets.count(sheet) > 1})
    if repeated:
        raise ValueError(
            where.refusal(
                f"sheets {repeated} are declared more than once; a sheet carries one element"
            )
        )
    return declarations


def resolved_lumped_elements(
    build_input: GeometryBuildInput, declarations: Sequence[Declaration]
) -> tuple[LumpedElementRecord, ...]:
    """Resolve each declaration on the port sheets of ``build_input``, the ports numbered first.

    Raises:
        ValueError: a declaration names no authored sheet or several, its sheet does not land on exactly two
            conductors, both conductors are in one Net, or a port's ``reference_net`` is neither one's Net.
    """
    if not declarations:
        return ()
    ordered = [
        *(item for item in declarations if isinstance(item, LumpedPort)),
        *(item for item in declarations if isinstance(item, LumpedElement)),
    ]
    entities = {entity.semantic_id: entity for entity in build_input.entities}
    grid_um = geometry_grid_context(build_input).grid_size_um
    return tuple(
        _resolved(number, declaration, build_input, entities, grid_um)
        for number, declaration in enumerate(ordered, start=1)
    )


def _resolved(
    number: int,
    declaration: Declaration,
    build_input: GeometryBuildInput,
    entities: Mapping[str, SemanticEntitySpec],
    grid_um: float | None,
) -> LumpedElementRecord:
    """One declaration on its authored sheet, with the sheet's two conductors in direction order."""
    region = _authored_sheet(declaration, build_input, grid_um)
    box = _box(region)
    where = _refusal_subject(declaration, region, box, grid_um)
    hosts = tuple(dict.fromkeys(item.host_semantic_id for item in region.overlaps))
    if len(hosts) != 2:
        raise ValueError(
            Declared(where, _FIX_THE_LAYOUT).refusal(
                f"it lands on {len(hosts)} conductor{'' if len(hosts) == 1 else 's'}, {list(hosts)}; "
                "draw the sheet so that it lands on "
                f"exactly two conductors of its target layer {region.metadata.get('target_layer')!r}, "
                "one at each end"
            )
        )
    start, end = sorted(
        (_terminal(entities[host]) for host in hosts),
        key=lambda item: _along_direction(region, item.source_entity_id),
    )
    if start.net_id == end.net_id:
        raise ValueError(
            Declared(where, "the layout or the call to GeometryPlan.set_nets").refusal(
                f"both conductors it lands on, {list(hosts)}, are in Net {start.net_id!r}; "
                "a lumped element joins two Nets"
            )
        )
    if isinstance(declaration, LumpedPort) and declaration.reference_net not in {
        start.net_id,
        end.net_id,
    }:
        raise ValueError(
            Declared(where, _SET_LUMPED_ELEMENTS).refusal(
                f"reference_net {declaration.reference_net!r} is not the Net of either conductor it lands on "
                f"({start.source_entity_id} in {start.net_id!r}, {end.source_entity_id} in {end.net_id!r})"
            )
        )
    return LumpedElementRecord(
        number=number,
        declaration=declaration,
        port_sheet_id=region.port_sheet_id,
        source_polygon_id=region.source_polygon_id,
        source_layer=region.source_layer,
        target_layer=str(region.metadata["target_layer"]),
        direction=tuple(float(value) for value in region.metadata["direction"]),
        start=start,
        end=end,
        box_um=box,
        grid_um=grid_um,
    )


def _authored_sheet(
    declaration: Declaration, build_input: GeometryBuildInput, grid_um: float | None
) -> PortSheetRegionRecord:
    """The one port sheet the layout authors under the declaration's name."""
    regions = build_input.port_sheet_regions
    matches = [
        item
        for item in regions
        if item.metadata.get("source_name") == declaration.sheet
    ]
    kind = "lumped port" if isinstance(declaration, LumpedPort) else "lumped element"
    if not matches:
        authored = sorted({str(item.metadata.get("source_name")) for item in regions})
        raise ValueError(
            Declared(f"{kind} {declaration.sheet!r}", _SET_LUMPED_ELEMENTS).refusal(
                f"the layout authors no port sheet of that name; its port sheets are {shown(authored)}"
            )
        )
    if len(matches) > 1:
        places = ", ".join(format_box(_box(item), grid_um) for item in matches)
        raise ValueError(
            Declared(f"{kind} {declaration.sheet!r}", _FIX_THE_LAYOUT).refusal(
                f"its name selects {len(matches)} sheets on layer {matches[0].source_layer}, at {places}; "
                "an element is one sheet"
            )
        )
    return matches[0]


def _refusal_subject(
    declaration: Declaration,
    region: PortSheetRegionRecord,
    box: tuple[float, float, float, float],
    grid_um: float | None,
) -> str:
    kind = "lumped port" if isinstance(declaration, LumpedPort) else "lumped element"
    return (
        f"{kind} {declaration.sheet!r}, its sheet on layer {region.source_layer} at "
        f"{format_box(box, grid_um)}"
    )


def _terminal(entity: SemanticEntitySpec) -> JunctionTerminalBinding:
    """The binding of one conductor a sheet lands on; a conductor without a Net is a Net of its own."""
    return JunctionTerminalBinding(
        source_entity_id=entity.semantic_id,
        source_occurrence_path=junction_occurrence_path(entity.metadata),
        net_id=junction_net_id(entity),
    )


def _along_direction(region: PortSheetRegionRecord, host: str) -> float:
    """The mean position of the sheet's landing on ``host``, measured along the authored direction.

    The rule Palace's partition orders a sheet's two terminals by (``scgsim.palace.source.ports``), so both
    solvers cut the same rectangle; issue #282 makes it one rule in the geometry package.
    """
    dx, dy = region.metadata["direction"][:2]
    points = [
        point
        for overlap in region.overlaps
        if overlap.host_semantic_id == host
        for point in overlap.overlap_loop
    ]
    return sum(x * dx + y * dy for x, y in points) / len(points)


def _box(region: PortSheetRegionRecord) -> tuple[float, float, float, float]:
    xs = [x for x, _ in region.exterior]
    ys = [y for _, y in region.exterior]
    return (min(xs), min(ys), max(xs), max(ys))


def _sheet_name(value: object, record: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(
            _declared(record).expected(
                "sheet", "the nonempty qualified name of an authored port sheet", value
            )
        )
    return value


def _positive(value: object, field: str, record: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(_declared(record).expected(field, "a number", value))
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(
            _declared(record).expected(field, "a finite number above zero", value)
        )
    return number


def _declared(record: str) -> Declared:
    return Declared(record, "the record given to GeometryPlan.set_lumped_elements")


__all__ = [
    "Declaration",
    "LumpedElement",
    "LumpedElementRecord",
    "LumpedPort",
    "checked_lumped_elements",
    "resolved_lumped_elements",
]
