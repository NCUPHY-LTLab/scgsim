"""Rectangular-junction requests, their terminal bindings, and the geometry of a derived junction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


def _text(value: object, name: str) -> None:
    """Reject missing identity text at a record boundary."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be nonempty text")


@dataclass(frozen=True)
class JunctionTerminalBinding:
    """Bind one junction terminal to its source Entity, occurrence, and net."""

    source_entity_id: str
    source_occurrence_path: str
    net_id: str

    def __post_init__(self) -> None:
        for name in ("source_entity_id", "source_occurrence_path", "net_id"):
            _text(getattr(self, name), name)

    def to_payload(self) -> dict[str, str]:
        """Serialize the qualified source terminal identity."""
        return {
            "source_entity_id": self.source_entity_id,
            "source_occurrence_path": self.source_occurrence_path,
            "net_id": self.net_id,
        }

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> JunctionTerminalBinding:
        """Reconstruct a terminal binding and apply its identity checks."""
        return cls(**dict(value))


@dataclass(frozen=True)
class RectangularJunctionSpec:
    """Request a rectangular junction from an explicit port sheet and terminals."""

    junction_id: str
    port_sheet_id: str
    source_polygon_id: str
    source_occurrence_path: str
    terminal_a: JunctionTerminalBinding
    terminal_b: JunctionTerminalBinding

    def __post_init__(self) -> None:
        for name in (
            "junction_id",
            "port_sheet_id",
            "source_polygon_id",
            "source_occurrence_path",
        ):
            _text(getattr(self, name), name)
        if not isinstance(self.terminal_a, JunctionTerminalBinding) or not isinstance(
            self.terminal_b, JunctionTerminalBinding
        ):
            raise TypeError("terminal bindings must be JunctionTerminalBinding")
        if self.terminal_a.source_entity_id == self.terminal_b.source_entity_id:
            raise ValueError(
                "junction terminals must identify distinct source Entities"
            )
        if self.terminal_a.net_id == self.terminal_b.net_id:
            raise ValueError("junction terminal Nets must be distinct")

    def to_payload(self) -> dict[str, Any]:
        """Serialize the request, including both source-bound terminals."""
        return {
            "junction_id": self.junction_id,
            "port_sheet_id": self.port_sheet_id,
            "source_polygon_id": self.source_polygon_id,
            "source_occurrence_path": self.source_occurrence_path,
            "terminal_a": self.terminal_a.to_payload(),
            "terminal_b": self.terminal_b.to_payload(),
        }

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> RectangularJunctionSpec:
        """Rebuild terminal records before validating the junction request."""
        fields = dict(value)
        fields["terminal_a"] = JunctionTerminalBinding.from_payload(
            fields["terminal_a"]
        )
        fields["terminal_b"] = JunctionTerminalBinding.from_payload(
            fields["terminal_b"]
        )
        return cls(**fields)


# The metadata ``part`` of a polygon the junction partition derives as a terminal's end piece.
JUNCTION_END_PIECE_PART = "junction_end_piece"


def is_junction_end_piece(polygon: Any) -> bool:
    """Whether a polygon is a junction end piece the partition derived, not an authored polygon.

    A landing end piece overlaps its terminal, so a selector point that names the terminal's
    authored polygon can lie inside the end piece too.  Selection by point binds authored
    polygons only, as it did before the partition.
    """
    return polygon.metadata.get("part") == JUNCTION_END_PIECE_PART


def junction_net_id(entity: Any) -> str:
    """The Net a junction terminal conductor belongs to, as SGB keys conductors.

    A conductor without a declared Net is a Net of its own, named by its Entity.
    """
    return entity.net_id or entity.semantic_id


# A standalone layout, bound by ``set_geometry``, records no source occurrences; its port
# sheets and conductors all share this one occurrence identity.
STANDALONE_OCCURRENCE_PATH = "<standalone>"


def junction_occurrence_path(metadata: Mapping[str, Any]) -> str:
    """The source occurrence a port sheet or conductor records, or the standalone layout's."""
    path = metadata.get("source_occurrence_path")
    return STANDALONE_OCCURRENCE_PATH if path is None else str(path)


@dataclass(frozen=True)
class JunctionGeometry:
    """Geometry algorithm input without solver inductance or capacitance settings."""

    junction_id: str
    source_polygon_id: str
    terminal_a_net: str
    terminal_b_net: str
    direction_xy: tuple[float, float]
    width_um: float
    metadata: Mapping[str, Any] = field(default_factory=dict)
