"""Own native HFSS boundary, terminal and property reads."""

from __future__ import annotations

import math
import re
from typing import Any

from scgsim.aedt.specs.common import (
    TerminalPort,
)


def native_value_matches(observed: Any, expected: Any) -> bool:
    """Accept AEDT native scalar spelling without converting its recorded value."""
    if isinstance(expected, bool):
        return observed is expected or observed == str(expected).lower()
    if isinstance(expected, int):
        return observed == expected or observed == str(expected)
    return observed == expected


def native_terminal_readback(
    hfss: Any, boundary_name: str, terminal_name: str, port: TerminalPort
) -> dict[str, Any]:
    """Read port and terminal state from AEDT's Excitations object tree."""
    excitation_names = hfss.get_oo_name(hfss.odesign, "Excitations")
    if boundary_name not in excitation_names:
        raise RuntimeError(f"AEDT terminal boundary is missing: {boundary_name!r}")
    terminal_names = hfss.get_oo_name(hfss.odesign, f"Excitations\\{boundary_name}")
    if terminal_names != [terminal_name]:
        raise RuntimeError(
            f"AEDT terminal child mismatch for {boundary_name!r}: {terminal_names!r}"
        )
    boundary_properties = native_oo_properties(hfss, f"Excitations\\{boundary_name}")
    terminal_properties = native_oo_properties(
        hfss, f"Excitations\\{boundary_name}\\{terminal_name}"
    )
    require_native_properties(
        boundary_properties,
        {
            "Name": boundary_name,
            "Type": "Wave Port",
            "Wave Port Type": "Terminal",
            "Num Terminals": 1,
            "Deembed Dist": f"{port.deembed_um:g}um",
            "Renorm All Terminals": False,
        },
        f"terminal boundary {boundary_name!r}",
    )
    require_native_properties(
        terminal_properties,
        {"Name": terminal_name, "Port Name": boundary_name, "Type": "Terminal"},
        f"terminal child {terminal_name!r}",
    )
    if boundary_properties.get("Deembed") not in (True, False, "true", "false"):
        raise RuntimeError("native Wave deembed flag is unavailable")
    return {
        "excitation_names": excitation_names,
        "terminal_names": terminal_names,
        "boundary_properties": boundary_properties,
        "terminal_properties": terminal_properties,
        "inactive_renormalizing_impedance": terminal_properties.get(
            "Terminal Renormalizing Impedance"
        ),
    }


def native_modal_oo_readback(hfss: Any, boundary_name: str) -> dict[str, Any]:
    """Read the direct AEDT object-tree state of one modal wave port."""
    excitation_names = hfss.get_oo_name(hfss.odesign, "Excitations")
    if boundary_name not in excitation_names:
        raise RuntimeError(f"AEDT modal boundary is missing: {boundary_name!r}")
    properties = native_oo_properties(hfss, f"Excitations\\{boundary_name}")
    require_native_properties(
        properties,
        {
            "Name": boundary_name,
            "Type": "Wave Port",
            "Num Modes": 1,
            "Deembed": False,
            "Renorm All Modes": False,
        },
        f"modal boundary {boundary_name!r}",
    )
    return {
        "excitation_names": excitation_names,
        "boundary_properties": properties,
    }


def native_position_um(value: Any) -> float:
    """Read one saved AEDT modal coordinate as a finite micrometer value."""
    if isinstance(value, bool):
        raise ValueError("AEDT modal coordinate must be a finite length")
    if isinstance(value, str):
        match = re.fullmatch(
            r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([A-Za-z]+)",
            value.strip(),
        )
        if match:
            from ansys.aedt.core.generic.constants import AEDT_UNITS, unit_converter  # noqa: PLC0415 -- optional native unit/SI evidence loads its SDK or NumPy dependency only when queried.

            units = match.group(2)
            if units not in AEDT_UNITS["Length"]:
                raise ValueError("AEDT modal coordinate units must be a length")
            value = unit_converter(
                float(match.group(1)),
                unit_system="Length",
                input_units=units,
                output_units="um",
            )
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("AEDT modal coordinate must be a finite length") from exc
    if not math.isfinite(result):
        raise ValueError("AEDT modal coordinate must be a finite length")
    return result


def native_terminal_reference_ids(hfss: Any) -> list[int]:
    """Read terminal reference conductors from HFSS native design properties."""
    try:
        values = hfss.design_properties["BoundarySetup"]["ProductSpecificData"][
            "TerminalReferenceConductors"
        ]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            "AEDT native terminal reference conductors are unavailable"
        ) from exc
    if not isinstance(values, list) or not values:
        raise RuntimeError("AEDT native terminal reference conductors are invalid")
    try:
        return [int(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise RuntimeError("AEDT native terminal reference IDs are invalid") from exc


def native_oo_properties(hfss: Any, path: str) -> dict[str, Any]:
    """Return properties directly from an AEDT object-oriented child object."""
    child = hfss.get_oo_object(hfss.odesign, path)
    if not child:
        raise RuntimeError(f"AEDT native child object is unavailable: {path!r}")
    names = child.GetPropNames()
    if not names:
        raise RuntimeError(f"AEDT native child has no properties: {path!r}")
    return {name: child.GetPropValue(name) for name in names}


def require_native_properties(
    observed: dict[str, Any], expected: dict[str, Any], context: str
) -> None:
    """Require semantic settings from their direct AEDT native representation."""
    mismatches = {
        key: {"expected": value, "observed": observed.get(key)}
        for key, value in expected.items()
        if key not in observed or not native_value_matches(observed[key], value)
    }
    if mismatches:
        raise RuntimeError(f"{context} native property mismatch: {mismatches!r}")


def text_value(value: Any, field: str) -> str:
    """Require the nonempty native identity text used by a port record."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be non-empty text")
    return value
