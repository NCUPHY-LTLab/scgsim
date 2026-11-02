"""Own detached Q3D model-stage records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scgsim.aedt.runtime.native.common import (
    BoundAedtRequest,
    detached_data,
)
from scgsim.aedt.runtime.native.common import (
    pyaedt_version as _pyaedt_version,
)


@dataclass(frozen=True)
class PreparedQ3d:
    """Carrier for one prepared Q3D app and its native readback records."""

    app: Any
    request: BoundAedtRequest
    project_path: Path
    materials: list[dict[str, Any]]
    region: dict[str, Any]
    nets: list[dict[str, Any]]
    setup: dict[str, Any] | None

    def __post_init__(self) -> None:
        if not isinstance(self.request, BoundAedtRequest):
            raise TypeError("PreparedQ3d requires a bound AEDT request")
        for name in ("materials", "region", "nets", "setup"):
            object.__setattr__(self, name, detached_data(getattr(self, name)))


def prepared_q3d_result(prepared: PreparedQ3d) -> dict[str, Any]:
    """Detach native preparation facts; the transaction seals the released project."""
    return {
        "outputs": {},
        "connected": {
            "aedt_version": prepared.app.desktop_class.aedt_version_id,
            "pyaedt_version": _pyaedt_version(),
        },
        "project": prepared.project_path.relative_to(
            prepared.request.workspace
        ).as_posix(),
        "materials": detached_data(prepared.materials),
        "region": detached_data(prepared.region),
        "nets": detached_data(prepared.nets),
        "setup": detached_data(prepared.setup),
        "save": {"ok": True},
        "solver_invoked": False,
    }
