"""Explicit source ownership for solver-neutral rectangular junction geometry."""

from collections.abc import Mapping
from dataclasses import dataclass, field


def _text(value, name):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be nonempty text")


@dataclass(frozen=True)
class JunctionTerminalBinding:
    source_entity_id: str
    source_occurrence_path: str
    net_id: str

    def __post_init__(self):
        for name in ("source_entity_id", "source_occurrence_path", "net_id"):
            _text(getattr(self, name), name)

    def to_payload(self):
        return {
            "source_entity_id": self.source_entity_id,
            "source_occurrence_path": self.source_occurrence_path,
            "net_id": self.net_id,
        }

    @classmethod
    def from_payload(cls, value):
        return cls(**dict(value))


@dataclass(frozen=True)
class RectangularJunctionSpec:
    junction_id: str
    port_sheet_id: str
    source_polygon_id: str
    source_occurrence_path: str
    terminal_a: JunctionTerminalBinding
    terminal_b: JunctionTerminalBinding

    def __post_init__(self):
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

    def to_payload(self):
        return {
            "junction_id": self.junction_id,
            "port_sheet_id": self.port_sheet_id,
            "source_polygon_id": self.source_polygon_id,
            "source_occurrence_path": self.source_occurrence_path,
            "terminal_a": self.terminal_a.to_payload(),
            "terminal_b": self.terminal_b.to_payload(),
        }

    @classmethod
    def from_payload(cls, value):
        fields = dict(value)
        fields["terminal_a"] = JunctionTerminalBinding.from_payload(
            fields["terminal_a"]
        )
        fields["terminal_b"] = JunctionTerminalBinding.from_payload(
            fields["terminal_b"]
        )
        return cls(**fields)


@dataclass(frozen=True)
class JunctionGeometry:
    """Algorithm input without solver inductance/capacitance settings."""

    junction_id: str
    source_polygon_id: str
    terminal_a_net: str
    terminal_b_net: str
    direction_xy: tuple[float, float]
    width_um: float
    metadata: Mapping = field(default_factory=dict)
