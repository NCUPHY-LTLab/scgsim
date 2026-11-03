"""AEDT object names of the emitted solids (design brief decision D4, experiment E4).

AEDT keeps only ``[A-Za-z0-9_]`` of an imported object name and drops a name with any non-ASCII
character altogether, so SGB writes the sanitised name itself and the adapter asserts, after the import,
that the objects AEDT created are exactly these names.  Two solids whose sanitised names coincide are
told apart by a numeric suffix in tag order, so the result is deterministic and never relies on AEDT's own
suffixing.
"""

from __future__ import annotations

import re

AEDT_ALLOWED = re.compile(r"[^A-Za-z0-9_]")


def sanitise_aedt_name(name: str) -> str:
    """``name`` with every character AEDT would drop replaced by ``_`` (``VOL::chip/air`` -> ``VOL__chip_air``)."""
    cleaned = AEDT_ALLOWED.sub("_", name)
    return cleaned or "_"


def aedt_names(solid_groups: list[tuple[str, str]]) -> dict[str, str]:
    """Solid id -> unique AEDT name, from ``(solid id, group name)`` pairs in tag order.

    A collision gets ``__2``, ``__3`` ... appended to the later solid(s); the suffixed name is checked
    against every name already taken, so the final names are pairwise distinct.
    """
    taken: set[str] = set()
    out: dict[str, str] = {}
    for solid_id, group_name in solid_groups:
        base = sanitise_aedt_name(group_name)
        candidate, count = base, 1
        while candidate in taken:
            count += 1
            candidate = f"{base}__{count}"
        taken.add(candidate)
        out[solid_id] = candidate
    return out
