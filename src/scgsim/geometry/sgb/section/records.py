"""The 2D model SGB holds in memory: the plane section of its validated 3D model along a cut line.

Owns the records of the section: vertices, edges and faces in the section's frame (``u`` along the cut in grid
units, ``z`` the stack height in micrometres), each tied to what it is the trace of in the 3D model, and the
groups and Surface-EPR rows the cut crosses.

* An **edge** is the trace of exactly one 3D face: a horizontal face gives a horizontal edge at its level over
  the stretch of the cut it covers, a wall a vertical edge where the cut crosses it.  Its group is that face's
  primary group, so the interface type, face kind, Entity, Net and band carry over unchanged.
* A **face** is the trace of one emitted 3D volume: one connected part of the volume's cross-section, so a
  volume the cut crosses twice gives two faces.  Its group is the volume's, the owning Entity.

Groups have ``dim`` 2 (faces) and 1 (edges), one dimension below the 3D model's.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

from scgsim.geometry.models.section import CutLine
from scgsim.geometry.sgb.bands.requests import InsetRequest
from scgsim.geometry.sgb.checks.validation.report import ValidationReport
from scgsim.geometry.sgb.exact import identity
from scgsim.geometry.sgb.exact.labels import Domain, SheetOwner
from scgsim.geometry.sgb.model import ConformalModel
from scgsim.geometry.sgb.section.frame import CutFrame
from scgsim.geometry.sgb.topology.assembly.records import Loop
from scgsim.geometry.sgb.topology.routes.groups import PhysicalGroup


@dataclass(frozen=True)
class SectionVertex:
    """A point of the section: ``u`` grid units along the cut from its start, at height ``z`` um."""

    id: str
    u: int
    z: Fraction


@dataclass(frozen=True)
class SectionEdge:
    """An edge of the section, the trace of one 3D face, from ``v0`` to ``v1``.

    A horizontal edge runs towards larger ``u``, a vertical edge upwards.  ``left`` and ``right`` are the faces
    of the section on either side of ``v0 -> v1`` (above and below a horizontal edge; at smaller and larger
    ``u`` beside a vertical one); ``None`` is a side with no face: outside the model, or an excluded conductor.
    The domains are those of the 3D face on the same sides.  ``curves`` are the carriers (source curves or band
    curves) the cut crosses where a vertical edge stands; a horizontal edge has none.
    """

    id: str
    kind: Literal["horizontal", "vertical"]
    v0: str
    v1: str
    trace: str  # the 3D face this edge is the trace of
    group: str
    left: str | None
    right: str | None
    left_domain: Domain | None
    right_domain: Domain | None
    owner: SheetOwner | None = None
    bands: tuple[tuple[str, int], ...] = ()
    curves: tuple[str, ...] = ()


@dataclass(frozen=True)
class SectionFace:
    """A face of the section: one connected part of the trace of an emitted 3D volume.

    ``loops`` hold ``(edge id, +1 / -1)`` entries with the face on their left: the outer loop first
    (counter-clockwise in the ``(u, z)`` plane), then the holes.  ``embedded`` are edges with this face on both
    sides.  ``twice_area`` is in grid units times micrometres.
    """

    id: str
    volume: str
    domain: Domain
    loops: tuple[Loop, ...]
    embedded: tuple[str, ...]
    twice_area: Fraction


@dataclass
class SectionModel:
    """The plane section of a validated 3D model along a cut line, validated in 2D.

    ``groups`` are the primary groups (``dim`` 2 for faces, 1 for edges) and ``unions`` the 3D unions restricted
    to them; ``surface_epr_rows`` maps each Surface-EPR row the cut crosses to its group, and
    ``band_family_requests`` are the concrete requests of the band families whose bands the cut crosses (one per
    family, its id the family's), with their margins.  ``source`` is the
    3D model, which keeps the sources and lineage the section refers to.
    """

    source: ConformalModel
    cut: CutLine
    frame: CutFrame
    vertices: dict[str, SectionVertex]
    edges: dict[str, SectionEdge]
    faces: dict[str, SectionFace]
    groups: dict[str, PhysicalGroup]
    unions: dict[str, list[str]]
    surface_epr_rows: dict[str, str]
    band_family_requests: tuple[InsetRequest, ...]
    report: ValidationReport

    @property
    def counts(self) -> dict[str, int]:
        """Entity counts of the section by kind."""
        return {
            "vertices": len(self.vertices),
            "edges": len(self.edges),
            "faces": len(self.faces),
            "groups": len(self.groups),
        }

    def u_um(self, u: int) -> float:
        """A ``u`` in grid units, in micrometres."""
        return self.frame.grid.to_um(u)

    def identity_digest(self) -> str:
        """One hash over every record of the section and its groups: equal for two sections iff they are equal."""
        return identity.digest(
            "section",
            {
                "cut": [list(self.cut.start_um), list(self.cut.end_um)],
                "grid": Fraction(repr(self.frame.grid.size_um)),
                "vertices": [self.vertices[k] for k in sorted(self.vertices)],
                "edges": [self.edges[k] for k in sorted(self.edges)],
                "faces": [self.faces[k] for k in sorted(self.faces)],
                "groups": {k: [g.dim, sorted(g.ids)] for k, g in self.groups.items()},
                "unions": self.unions,
                "surface_epr_rows": self.surface_epr_rows,
                "band_families": [r.request_id for r in self.band_family_requests],
            },
        )
