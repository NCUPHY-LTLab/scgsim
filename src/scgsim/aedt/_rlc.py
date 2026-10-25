"""Exact generic RLC scalar and topology interpretation without native imports."""

from decimal import Decimal
import re
from typing import Any


def rlc_native_topology(topology: str) -> str:
    """Translate the authored generic topology to its native spelling."""
    return {"series": "Serial", "parallel": "Parallel"}[topology]


def rlc_scalar_si(value: Any, quantity: str) -> Decimal:
    """Shared exact live/offline RLC unit decoding; raw evidence is retained."""
    units = {
        "resistance_ohm": {
            "uOhm": "1e-6",
            "mOhm": "1e-3",
            "ohm": "1",
            "kOhm": "1e3",
            "megohm": "1e6",
            "GOhm": "1e9",
        },
        "inductance_h": {
            "fH": "1e-15",
            "pH": "1e-12",
            "nH": "1e-9",
            "uH": "1e-6",
            "mH": "1e-3",
            "H": "1",
        },
        "capacitance_f": {
            "fF": "1e-15",
            "pF": "1e-12",
            "nF": "1e-9",
            "uF": "1e-6",
            "mF": "1e-3",
            "F": "1",
        },
    }
    match = re.fullmatch(
        r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)\s*([A-Za-z]+)\s*",
        str(value),
    )
    if match is None or match[2] not in units[quantity]:
        raise RuntimeError(f"native generic RLC scalar is unavailable: {value!r}")
    return Decimal(match[1]) * Decimal(units[quantity][match[2]])
