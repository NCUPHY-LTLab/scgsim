"""One meshing round as data: what the driver sends to the Gmsh worker.

Owns :class:`MeshJob` and its JSON form.  It imports no Gmsh, so both sides of the process boundary read it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .policy import DELAUNAY, SizeField, SizeRequests


@dataclass(frozen=True)
class MeshJob:
    """Mesh one XAO once, with the given sizes, and audit the result.

    ``attribute=False`` counts the bad elements without attributing them, for meshes nobody repairs.
    ``algorithm_2d`` is Gmsh's surface algorithm, left at its default when ``None``.
    """

    xao: Path
    sidecar: Path
    msh: Path
    sizes: SizeRequests
    threshold: float
    algorithm_3d: int = DELAUNAY
    algorithm_2d: int | None = None
    attribute: bool = True

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
            "threshold": self.threshold,
            "algorithm_3d": self.algorithm_3d,
            "algorithm_2d": self.algorithm_2d,
            "attribute": self.attribute,
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
        )
        return cls(
            xao=Path(raw["xao"]),
            sidecar=Path(raw["sidecar"]),
            msh=Path(raw["msh"]),
            sizes=sizes,
            threshold=float(raw["threshold"]),
            algorithm_3d=int(raw["algorithm_3d"]),
            algorithm_2d=None
            if raw["algorithm_2d"] is None
            else int(raw["algorithm_2d"]),
            attribute=bool(raw["attribute"]),
        )
