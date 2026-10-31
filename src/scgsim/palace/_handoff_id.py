"""The Palace handoff ID, computed identically when a handoff is written and checked.

Owns the canonical payload and hash that define a handoff ID. It does not own the
inputs: the writer (``execution/handoff.py``) gathers them, and the checker
(``results/_run_identity.py``) recomputes the ID from the returned run's metadata
rather than trusting the copied value.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any


def compute_handoff_id(
    *,
    profile: str,
    route: str,
    problem: str,
    source_revisions: Mapping[str, Any],
    palace_identity: Mapping[str, Any],
    hashes: Sequence[Any],
    execution_identity: Mapping[str, Any],
    route_a_thin_film: Mapping[str, Any] | None = None,
) -> str:
    """SHA-256 of the canonical JSON of the identity inputs.

    ``route_a_thin_film`` is left out of the payload entirely when ``None``, so
    handoffs without it keep the IDs they had before the field existed.
    """
    payload = {
        "route": route,
        "problem": problem,
        "profile": profile,
        "source_revisions": source_revisions,
        "palace_identity": palace_identity,
        "hashes": list(hashes),
        "execution_identity": execution_identity,
        **(
            {"route_a_thin_film": route_a_thin_film}
            if route_a_thin_film is not None
            else {}
        ),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
