"""One meshing round as data: what the driver sends to the Gmsh worker.

Owns :class:`MeshJob` and its JSON form.  It imports no Gmsh, so both sides of the process boundary read it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .mesher import MesherOptions
from .sizing import LocalSize, SizeField, SizeRequests


@dataclass(frozen=True)
class Elevation:
    """Where the worker writes the elevated mesh, and which linear mesh it elevates: the one the loop delivers."""

    msh: Path
    max_tetrahedra: int
    # False for a report-only strategy, whose mesh is delivered, and so elevated, below the threshold too.
    enforce_threshold: bool = True


@dataclass(frozen=True)
class MeshJob:
    """Mesh one XAO once, with the given sizes, and audit the result.

    The audit always measures the linear mesh, which is written to ``msh``.  With an ``elevation``, a linear
    mesh that the loop would accept (no tetrahedron below the threshold, within the tetrahedron budget) is
    then raised to ``mesher.geometry_order`` in the same session and written to ``elevation.msh``.
    ``attribute=False`` counts the bad elements without attributing them, for meshes nobody repairs.
    """

    xao: Path
    sidecar: Path
    msh: Path
    sizes: SizeRequests
    threshold: float
    mesher: MesherOptions = MesherOptions()
    attribute: bool = True
    elevation: Elevation | None = None

    def to_json(self) -> dict[str, Any]:
        """The JSON form of the job; curve sizes are keyed by tag in ascending order."""
        return {
            "xao": str(self.xao),
            "sidecar": str(self.sidecar),
            "msh": str(self.msh),
            "max_size": self.sizes.max_size,
            "gradation": self.sizes.gradation,
            "curve_sizes": {
                str(t): s for t, s in sorted(self.sizes.curve_sizes.items())
            },
            "fields": [
                {"curves": list(f.curves), "size": f.size, "distance": f.distance}
                for f in self.sizes.fields
            ],
            "local_sizes": [
                {
                    "centre": list(local.centre),
                    "size": local.size,
                    "radius": local.radius,
                }
                for local in self.sizes.local_sizes
            ],
            "threshold": self.threshold,
            "mesher": asdict(self.mesher),
            "attribute": self.attribute,
            "elevation": None
            if self.elevation is None
            else {
                "msh": str(self.elevation.msh),
                "max_tetrahedra": self.elevation.max_tetrahedra,
                "enforce_threshold": self.elevation.enforce_threshold,
            },
        }

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> MeshJob:
        """Rebuild a job from its JSON form."""
        sizes = SizeRequests(
            max_size=float(raw["max_size"]),
            gradation=float(raw["gradation"]),
            curve_sizes=MappingProxyType(
                {int(t): float(s) for t, s in raw["curve_sizes"].items()}
            ),
            fields=tuple(
                SizeField(
                    tuple(int(c) for c in f["curves"]),
                    float(f["size"]),
                    float(f["distance"]),
                )
                for f in raw["fields"]
            ),
            local_sizes=tuple(
                LocalSize(
                    _triple(local["centre"]),
                    float(local["size"]),
                    None if local["radius"] is None else float(local["radius"]),
                )
                for local in raw["local_sizes"]
            ),
        )
        return cls(
            xao=Path(raw["xao"]),
            sidecar=Path(raw["sidecar"]),
            msh=Path(raw["msh"]),
            sizes=sizes,
            threshold=float(raw["threshold"]),
            mesher=MesherOptions(**raw["mesher"]),
            attribute=bool(raw["attribute"]),
            elevation=None
            if raw["elevation"] is None
            else Elevation(
                Path(raw["elevation"]["msh"]),
                int(raw["elevation"]["max_tetrahedra"]),
                bool(raw["elevation"]["enforce_threshold"]),
            ),
        )


def _triple(values: list[float]) -> tuple[float, float, float]:
    return (float(values[0]), float(values[1]), float(values[2]))
