"""Describe the real Gmsh namespaces used by common native construction.

These type-only contracts keep Gmsh optional until native execution begins.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypedDict

if TYPE_CHECKING:
    import gmsh


class GmshModelApi(Protocol):
    """Native topology, OCC and mesh namespace used by incidence operations."""

    @property
    def model(self) -> type[gmsh.model]:
        """Expose native topology, OCC construction and mesh namespaces."""
        ...


class GmshKernelApi(GmshModelApi, Protocol):
    """Native model plus the option namespace required by shell correction."""

    @property
    def option(self) -> type[gmsh.option]:
        """Expose native option reads and writes."""
        ...


class GmshApi(GmshKernelApi, Protocol):
    """Native model transport used by complete plan lowering."""

    def merge(self, fileName: str) -> None:
        """Merge a detached native boundary transport into the active model."""
        ...

    def write(self, fileName: str) -> None:
        """Write the active native model to its transport path."""
        ...


class NativeTimingRecord(TypedDict):
    """The existing measured stage label, elapsed seconds and completion status."""

    stage: str
    seconds: float
    status: str
