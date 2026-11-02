"""Reject malformed nested terminal native payloads with strict errors."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from scgsim.aedt.results.terminal_readback import validate_terminal_native_evidence
from scgsim.aedt.specs.common import TerminalPort
from scgsim.aedt.specs.hfss import HfssDrivenSpec


def _terminal_fixture() -> tuple[list[dict[str, object]], SimpleNamespace]:
    ports = (
        TerminalPort(1, "P1", "-X", ("gnd",)),
        TerminalPort(2, "P2", "+X", ("gnd",)),
    )
    spec = SimpleNamespace(ports=ports)
    records: list[dict[str, object]] = []
    for port in ports:
        terminal_name = f"{port.name}:T1"
        terminal_impedance = "50ohm"
        records.append(
            {
                "index": port.index,
                "boundary": port.name,
                "terminal_excitation": terminal_name,
                "face_id": port.index,
                "face_center_um": [0, 0, 0],
                "requested": {
                    "reference_objects": list(port.reference_objects),
                    "renormalize": False,
                    "deembed_um": port.deembed_um,
                },
                "native": {
                    "excitation_names": [item.name for item in ports[: port.index]],
                    "terminal_names": [terminal_name],
                    "reference_conductors": list(port.reference_objects),
                    "reference_conductor_ids": [port.index],
                    "boundary_properties": {
                        "Deembed": True,
                        "Deembed Dist": f"{port.deembed_um:g}um",
                        "Name": port.name,
                        "Num Terminals": "1",
                        "Renorm All Terminals": False,
                        "Type": "Wave Port",
                        "Wave Port Type": "Terminal",
                    },
                    "terminal_properties": {
                        "Name": terminal_name,
                        "Port Name": port.name,
                        "Terminal Renormalizing Impedance": terminal_impedance,
                        "Type": "Terminal",
                    },
                    "inactive_renormalizing_impedance": terminal_impedance,
                },
            }
        )
    return records, spec


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        (
            "boundary_properties",
            None,
            "completed receipt has invalid terminal boundary properties",
        ),
        (
            "terminal_properties",
            [],
            "completed receipt has invalid terminal excitation properties",
        ),
    ),
)
def test_current_terminal_readback_rejects_malformed_nested_properties(
    field: str, value: object, message: str
) -> None:
    """Reject malformed nested property records with TypeError, not AttributeError."""
    records, spec = _terminal_fixture()
    native = records[0]["native"]
    if not isinstance(native, dict):
        raise RuntimeError("terminal fixture native payload is invalid")
    native[field] = value

    with pytest.raises(TypeError, match=message):
        validate_terminal_native_evidence(
            records, cast(HfssDrivenSpec, spec), current=True
        )


def test_current_terminal_readback_accepts_complete_nested_properties() -> None:
    """Keep the complete current terminal evidence shape accepted."""
    records, spec = _terminal_fixture()

    validate_terminal_native_evidence(records, cast(HfssDrivenSpec, spec), current=True)
