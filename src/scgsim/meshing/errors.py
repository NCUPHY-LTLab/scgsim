"""Exceptions of the meshing module."""

from __future__ import annotations

from typing import Any


class MeshingError(Exception):
    """Base class of every error the meshing module raises on purpose."""


class SidecarError(MeshingError, ValueError):
    """The sidecar is not a readable ``scgsim.sgb.sidecar.v3`` document."""


class MeshRepairError(MeshingError):
    """The repair loop stopped before the mesh met the quality threshold.

    ``report`` is the quality report as a JSON-serialisable mapping: the recipe, the repair policy, the
    mesher options, the threshold, the stop reason, the remaining bad elements with their groups and
    feature kinds, and ``stop_places``: where those elements are and the fine structure recorded there.
    """

    def __init__(self, message: str, report: dict[str, Any]) -> None:
        super().__init__(message)
        self.report = report
