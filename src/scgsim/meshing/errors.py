"""Exceptions of the meshing module."""

from __future__ import annotations

from typing import Any


class MeshingError(Exception):
    """Base class of every error the meshing module raises on purpose."""


class SidecarError(MeshingError, ValueError):
    """The sidecar is not a readable ``scgsim.sgb.sidecar.v2`` document."""


class MeshRepairError(MeshingError):
    """The repair loop stopped before the mesh met the quality threshold.

    ``report`` is the quality report as a JSON-serialisable mapping: the rule-set version, the threshold,
    the stop reason and the remaining bad elements with their groups and feature kinds.
    """

    def __init__(self, message: str, report: dict[str, Any]) -> None:
        super().__init__(message)
        self.report = report
