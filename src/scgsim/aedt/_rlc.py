"""Read AEDT lumped RLC values and topology names, with no AEDT imports.

The live runtime and the offline result reader both decode RLC values here, so
they cannot disagree. Values are exact `Decimal`s, never floats.
"""

import re
from decimal import Decimal
from typing import Any


def rlc_native_topology(topology: str) -> str:
    """Translate the authored generic topology to its native spelling."""
    return {"series": "Serial", "parallel": "Parallel"}[topology]


def rlc_scalar_si(value: Any, quantity: str) -> Decimal:
    """Decode an AEDT value such as "1.5nH" into exact SI units (ohm, henry or farad).

    `quantity` selects the accepted unit spellings. The caller keeps the raw
    string as evidence. Anything that is not a number followed by a known unit
    raises RuntimeError.
    """
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
