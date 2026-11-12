"""The load check the tests run on the opened XAO (design brief section 14, "Validation layers").

The mesher trusts the XAO and sidecar of one build and does not run it; it also serves to diagnose a
file suspected to be wrong.

Owns :func:`verify_loaded`: given the sidecar and a Gmsh model that has just opened the XAO (and called
``gmsh.model.occ.synchronize()``), prove that what was loaded is what was planned -- never by
inspecting the model, always against numbers the sidecar carries:

* entity counts per dimension equal the planned counts;
* every vertex sits where the plan put it, which proves the tag numbering that the rest relies on;
* every edge, face and solid is bounded by exactly the planned edges, faces and solids;
* every face area and solid volume equals the plan (an inside-out shell is a wrong or negative volume);
* the physical groups equal the planned groups (names, dimensions, tags, members).

:func:`apply_sidecar_embeds` re-applies the embeddings the XAO cannot carry (a port sheet inside its host
volume, a refinement curve inside its host face).  :func:`verify_xao` is the convenience that opens the
file in a fresh Gmsh session first.  Failures are collected, not raised one by one; call
:meth:`LoadReport.raise_if_failed` to stop.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import gmsh

from scgsim.contracts.sgb_sidecar import Sidecar, views
from scgsim.contracts.sgb_sidecar.records import IdsByTag
from scgsim.geometry.sgb.errors import SgbError
from scgsim.geometry.sgb.stopwatch import elapsed_seconds
from scgsim.geometry.sgb.verification.tolerances import (
    RELATIVE_TOLERANCE,
    VERTEX_TOLERANCE_UM,
    relative_error,
)

DIMENSIONS = ("0", "1", "2", "3")


class LoadCheckError(SgbError):
    """The loaded model differs from the plan."""


@dataclass
class LoadReport:
    """The outcome of :func:`verify_loaded`; ``problems`` is empty exactly when everything matched."""

    problems: list[str] = field(default_factory=list)
    counts: dict[str, dict[str, int]] = field(
        default_factory=dict
    )  # "native" / "planned" -> dim -> n
    max_vertex_offset_um: float = 0.0
    boundaries_mismatching: dict[str, int] = field(default_factory=dict)
    max_relative_face_area_error: float = 0.0
    max_relative_volume_error: float = 0.0
    groups: dict[str, int] = field(default_factory=dict)
    timings_s: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Whether every check passed."""
        return not self.problems

    def raise_if_failed(self) -> None:
        """Fail closed: any difference between the loaded model and the plan stops the build."""
        if self.problems:
            raise LoadCheckError(
                f"{len(self.problems)} difference(s) between the loaded model and the plan: "
                + "; ".join(self.problems[:6])
            )


def verify_xao(xao_path: Path, sidecar: Sidecar) -> LoadReport:
    """Open ``xao_path`` in a fresh Gmsh session, check it against the sidecar and close the session."""
    gmsh.initialize(["-noenv"])
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber(
            "Geometry.OCCAutoFix", 0
        )  # a needed fix would be a planning defect
        gmsh.open(str(xao_path))
        gmsh.model.occ.synchronize()
        return verify_loaded(sidecar)
    finally:
        gmsh.finalize()


def verify_loaded(sidecar: Sidecar, *, groups: bool = True) -> LoadReport:
    """Check the model currently loaded in Gmsh against the sidecar; see the module docstring.

    ``groups=False`` skips the physical-group comparison, for a model loaded from the BREP file, which
    carries no groups (only the XAO does).
    """
    report = LoadReport()
    expected = views.expected(sidecar)
    ids = sidecar["binding"]["ids_by_tag"]
    clock = _Clock(report)
    _check_counts(report, ids)
    _check_vertices(report, expected["vertices_um"])
    clock.lap("counts_and_vertex_placement")
    boundaries = {
        "edges": _count_wrong_boundaries(1, 0, expected["edge_vertices"]),
        "faces": _count_wrong_boundaries(2, 1, expected["face_edges"]),
        "solids": _count_wrong_boundaries(3, 2, expected["solid_faces"]),
    }
    report.boundaries_mismatching = boundaries
    if any(boundaries.values()):
        report.problems.append(f"boundary mismatch: {boundaries}")
    clock.lap("boundaries")
    if groups:
        _check_groups(report, _planned_groups(sidecar))
    clock.lap("physical_groups")
    report.max_relative_volume_error = _worst_relative_error(
        3, expected["solid_volume_um3"], "volume", report
    )
    clock.lap("volumes")
    report.max_relative_face_area_error = _worst_relative_error(
        2, expected["face_area_um2"], "face area", report
    )
    clock.lap("face_areas")
    return report


def apply_sidecar_embeds(sidecar: Sidecar) -> dict[str, int]:
    """Embed the sidecar's sheets in their host volumes and refinement curves in their host faces.

    Neither the BREP nor the XAO can carry mesh embedding, so the mesher calls this once after loading
    and before meshing.  Returns how many faces and edges were embedded.
    """
    index = views.TagIndex.of(sidecar)
    sheets = views.embedded_faces(sidecar)
    curves = views.embedded_edges(sidecar)
    for face, host in sheets:
        gmsh.model.mesh.embed(2, [index.tag(2, face)], 3, index.tag(3, host))
    by_host: dict[str, list[int]] = {}
    for edge, host in curves:
        by_host.setdefault(host, []).append(index.tag(1, edge))
    for host_face, edges in sorted(by_host.items()):
        gmsh.model.mesh.embed(1, edges, 2, index.tag(2, host_face))
    return {"faces": len(sheets), "edges": len(curves)}


class _Clock:
    def __init__(self, report: LoadReport) -> None:
        self.report = report
        self.mark = time.perf_counter()

    def lap(self, name: str) -> None:
        self.report.timings_s[name] = elapsed_seconds(self.mark)
        self.mark = time.perf_counter()


def _check_counts(report: LoadReport, ids: IdsByTag) -> None:
    native = {d: len(gmsh.model.getEntities(int(d))) for d in DIMENSIONS}
    planned = {
        "0": len(ids["0"]),
        "1": len(ids["1"]),
        "2": len(ids["2"]),
        "3": len(ids["3"]),
    }
    report.counts = {"native": native, "planned": planned}
    if native != planned:
        report.problems.append(f"entity counts {native} != planned {planned}")


def _check_vertices(report: LoadReport, positions: list[list[float]]) -> None:
    worst = 0.0
    for tag, (px, py, pz) in enumerate(positions, start=1):
        x, y, z, *_ = gmsh.model.getBoundingBox(0, tag)
        worst = max(worst, abs(x - px), abs(y - py), abs(z - pz))
    report.max_vertex_offset_um = worst
    if worst > VERTEX_TOLERANCE_UM:
        report.problems.append(
            f"vertex tags are not where the plan put them (max offset {worst:.3g} um)"
        )


def _count_wrong_boundaries(
    dim: int, boundary_dim: int, wanted: list[list[int]]
) -> int:
    """Entities of ``dim`` whose boundary is not exactly the planned set of entities of ``boundary_dim``."""
    wrong = 0
    for tag, want in enumerate(wanted, start=1):
        got = {
            t
            for d, t in gmsh.model.getBoundary(
                [(dim, tag)], combined=False, oriented=False
            )
            if d == boundary_dim
        }
        wrong += got != set(want)
    return wrong


def _planned_groups(sidecar: Sidecar) -> dict[str, tuple[int, int, list[int]]]:
    """Each planned group's dimension, tag and member tags, sorted."""
    index = views.TagIndex.of(sidecar)
    members = views.group_members(sidecar)
    return {
        name: (g["dim"], g["tag"], sorted(index.tags_of(g["dim"], members[name])))
        for name, g in sidecar["group"].items()
    }


def _check_groups(
    report: LoadReport, expect: Mapping[str, tuple[int, int, list[int]]]
) -> None:
    # ``getEntitiesForPhysicalGroup`` scans every entity of the dimension per group (groups x entities:
    # quadratic); asking each entity for its groups touches every entity once.
    members: dict[tuple[int, int], list[int]] = {}
    for dim in (2, 3):
        for _, entity in gmsh.model.getEntities(dim):
            for group_tag in gmsh.model.getPhysicalGroupsForEntity(dim, entity):
                members.setdefault((dim, int(group_tag)), []).append(int(entity))
    loaded = {
        gmsh.model.getPhysicalName(d, t): (d, t, sorted(members.get((d, t), [])))
        for d, t in gmsh.model.getPhysicalGroups()
    }
    report.groups = {"planned": len(expect), "loaded": len(loaded)}
    if expect != loaded:
        missing = sorted(set(expect) - set(loaded))
        extra = sorted(set(loaded) - set(expect))
        different = sorted(
            n for n in set(expect) & set(loaded) if expect[n] != loaded[n]
        )
        report.problems.append(
            f"physical groups differ from the plan (missing {missing[:3]}, extra {extra[:3]}, different {different[:3]})"
        )


def _worst_relative_error(
    dim: int, planned: list[float], what: str, report: LoadReport
) -> float:
    worst, negative = 0.0, 0
    for tag, want in enumerate(planned, start=1):
        mass = gmsh.model.occ.getMass(dim, tag)
        negative += mass <= 0
        worst = max(worst, relative_error(mass, want))
    if negative:
        report.problems.append(
            f"{negative} {what}(s) with non-positive measure (shell orientation)"
        )
    if worst > RELATIVE_TOLERANCE:
        report.problems.append(f"a {what} differs from the plan (relative {worst:.3g})")
    return worst
