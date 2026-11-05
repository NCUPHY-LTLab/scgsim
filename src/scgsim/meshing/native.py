"""Gmsh meshing in the current process: load the XAO, apply the sizes, mesh, audit and write.

Owns every Gmsh option the module sets, each with the measurement behind it.  It runs only inside the
worker process (``worker``): a fresh process per round is a requirement, because results in a long
multi-session process differed from a fresh one and HXT can crash the process.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from pathlib import Path

import gmsh

from .audit import audit_current_mesh
from .features import SidecarFeatures
from .findings import MeshAudit
from .job import MeshJob
from .policy import BoundingBox, SizeField, SizeRequests

# Distance fields sample each curve; one sample per half element keeps the distance accurate to the
# element size.  The bounds keep a 0.2 um sidewall edge and a 4 mm outline both affordable.
_MIN_SAMPLES = 20
_MAX_SAMPLES = 8000
_MSH_VERSION = 4.1


def mesh_in_current_process(job: MeshJob, features: SidecarFeatures) -> MeshAudit:
    """Run ``job`` in a fresh Gmsh session of this process and return the audit of the mesh it wrote."""
    gmsh.initialize(["-noenv"])
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(job.xao))
        _embed(features)
        _set_options(job)
        _add_size_fields(job.sizes)
        started = time.perf_counter()
        gmsh.model.mesh.generate(3)
        mesh_seconds = time.perf_counter() - started
        audit = audit_current_mesh(
            features, job.threshold, attribute=job.attribute, mesh_seconds=mesh_seconds
        )
        _write(job.msh)
        return audit
    finally:
        gmsh.finalize()


def bounding_box_of(xao: Path) -> BoundingBox:
    """Bounding box of the model in ``xao`` (the rules size the coarse mesh from its diagonal)."""
    gmsh.initialize(["-noenv"])
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(xao))
        x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(-1, -1)
        return BoundingBox((x0, y0, z0), (x1, y1, z1))
    finally:
        gmsh.finalize()


def _embed(features: SidecarFeatures) -> None:
    """Re-apply the embeddings the XAO cannot carry: sheet faces inside volumes, curves inside faces."""
    for sheet in features.embedded_faces:
        gmsh.model.mesh.embed(2, [sheet.face], 3, sheet.host_volume)
    curves_by_host: dict[int, list[int]] = defaultdict(list)
    for curve in features.embedded_edges:
        curves_by_host[curve.host_face].append(curve.edge)
    for host_face, curves in sorted(curves_by_host.items()):
        gmsh.model.mesh.embed(1, curves, 2, host_face)


def _set_options(job: MeshJob) -> None:
    option = gmsh.option
    # HXT (10) with only a global maximum size meshed every 100 um case cleanly; it is also the algorithm
    # whose boundary recovery handles the thin films.  The driver retries with Delaunay (1) if HXT dies.
    option.setNumber("Mesh.Algorithm3D", job.algorithm_3d)
    option.setNumber("Mesh.MeshSizeMax", job.sizes.max_size)
    if job.algorithm_2d is not None:
        option.setNumber("Mesh.Algorithm", job.algorithm_2d)
    option.setNumber("Mesh.MshFileVersion", _MSH_VERSION)
    # Save only what the XAO's physical groups name, so the file carries the groups and nothing extra.
    option.setNumber("Mesh.SaveAll", 0)


def _add_size_fields(sizes: SizeRequests) -> None:
    """Background size field: each requested size near its curves, growing to ``max_size`` with distance."""
    wanted = [*_graded_fields(sizes), *sizes.fields]
    if not wanted:
        return
    fields = [
        _size_field(list(w.curves), w.size, sizes.max_size, w.distance) for w in wanted
    ]
    background = fields[0] if len(fields) == 1 else _minimum_of(fields)
    gmsh.model.mesh.field.setAsBackgroundMesh(background)
    option = gmsh.option
    # With a field active, every other size source would override or extend it.  Extension from the
    # boundary (1) exploded the element count on the 1 to 4 mm chips; sizes from points and from
    # curvature are the sources the field replaces.
    option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    option.setNumber("Mesh.MeshSizeFromPoints", 0)
    option.setNumber("Mesh.MeshSizeFromCurvature", 0)


def _graded_fields(sizes: SizeRequests) -> list[SizeField]:
    """One field per distinct size of ``curve_sizes``, grown by the gradation.

    The gradation is 1 um of size per um of distance: 1 or less stayed clean, 1.5 or more created new
    near-flat tetrahedra where the refined region ends.
    """
    curves_by_size: dict[float, list[int]] = defaultdict(list)
    for curve, size in sorted(sizes.curve_sizes.items()):
        curves_by_size[size].append(curve)
    return [
        SizeField(tuple(curves), size, (sizes.max_size - size) / sizes.gradation)
        for size, curves in sorted(curves_by_size.items())
    ]


def _size_field(
    curves: list[int], size: float, max_size: float, distance: float
) -> int:
    """Threshold of the distance to ``curves``: ``size`` on them, ``max_size`` from ``distance`` away."""
    field = gmsh.model.mesh.field
    distance_field = field.add("Distance")
    field.setNumbers(distance_field, "CurvesList", curves)
    field.setNumber(distance_field, "Sampling", _samples(curves, size))
    threshold = field.add("Threshold")
    field.setNumber(threshold, "InField", distance_field)
    field.setNumber(threshold, "SizeMin", size)
    field.setNumber(threshold, "SizeMax", max_size)
    field.setNumber(threshold, "DistMin", 0.0)
    field.setNumber(threshold, "DistMax", distance)
    return threshold


def _minimum_of(fields: list[int]) -> int:
    field = gmsh.model.mesh.field
    combined = field.add("Min")
    field.setNumbers(combined, "FieldsList", fields)
    return combined


def _samples(curves: list[int], size: float) -> int:
    """Samples per curve of a Distance field: two per element along the longest of the curves."""
    longest = max(gmsh.model.occ.getMass(1, curve) for curve in curves)
    return int(min(max(math.ceil(2 * longest / size), _MIN_SAMPLES), _MAX_SAMPLES))


def _write(msh: Path) -> None:
    msh.parent.mkdir(parents=True, exist_ok=True)
    gmsh.write(str(msh))
