"""Gmsh meshing in the current process: load the XAO, apply the sizes, mesh, audit and write.

Owns every Gmsh option the module sets, each with the measurement behind it.  It runs only inside the
worker process (``worker``): a fresh process per round is a requirement, because results in a long
multi-session process differed from a fresh one and HXT can crash the process.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import gmsh

from .audit import audit_current_mesh
from .features import SidecarFeatures
from .findings import MeshAudit
from .job import MeshJob
from .sizing import BoundingBox, LocalSize, SizeField, SizeRequests

# Distance fields sample each curve; one sample per half element keeps the distance accurate to the
# element size.  The bounds keep a 0.2 um sidewall edge and a 4 mm outline both affordable.
_MIN_SAMPLES = 20
_MAX_SAMPLES = 8000
_MSH_VERSION = 4.1


@dataclass(frozen=True)
class NativeRound:
    """What a worker round produced: the audit of the linear mesh, and whether it was also elevated."""

    audit: MeshAudit
    elevated: bool
    elevation_error: str | None  # why an acceptable mesh was not elevated


def mesh_in_current_process(job: MeshJob, features: SidecarFeatures) -> NativeRound:
    """Run ``job`` in a fresh Gmsh session of this process and return what it wrote.

    The linear mesh is audited and written first.  Raising the order happens last, in this session, because
    placing the extra nodes of a curved element on its CAD face needs the CAD that only this session still
    has: a mesh read back from a file has lost it.
    """
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
        return _elevate_if_accepted(job, audit)
    finally:
        gmsh.finalize()


def _elevate_if_accepted(job: MeshJob, audit: MeshAudit) -> NativeRound:
    """Raise the order of the mesh and write it, when the job asks and the loop would accept this mesh.

    A mesh the loop rejects is never delivered, so elevating it would only cost time.  The audit above
    stays the one of the linear mesh: it is not repeated after the order is raised.
    """
    elevation = job.elevation
    if elevation is None or not audit.accepted(
        elevation.max_tetrahedra, enforce_threshold=elevation.enforce_threshold
    ):
        return NativeRound(audit, elevated=False, elevation_error=None)
    try:
        gmsh.model.mesh.setOrder(job.mesher.geometry_order)
        if job.mesher.high_order_optimize:
            gmsh.model.mesh.optimize("HighOrder")
        _write(elevation.msh)
    except Exception as error:  # noqa: BLE001 -- Gmsh raises a bare Exception; the loop decides what a failure means
        return NativeRound(audit, elevated=False, elevation_error=str(error))
    return NativeRound(audit, elevated=True, elevation_error=None)


def bounding_box_of(xao: Path) -> BoundingBox:
    """Bounding box of the model in ``xao`` (a recipe may size the coarse mesh from its diagonal)."""
    gmsh.initialize(["-noenv"])
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(xao))
        x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(-1, -1)
        return BoundingBox((x0, y0, z0), (x1, y1, z1))
    finally:
        gmsh.finalize()


def _embed(features: SidecarFeatures) -> None:
    """Re-apply the embedding the XAO cannot carry: sheet faces inside volumes."""
    for sheet in features.embedded_faces:
        gmsh.model.mesh.embed(2, [sheet.face], 3, sheet.host_volume)


def _set_options(job: MeshJob) -> None:
    option = gmsh.option
    # Delaunay (1) is the default (see MesherOptions.algorithm_3d); HXT (10) is the other choice.  The
    # driver retries a failed round with the other one, since HXT can abort the whole process.
    option.setNumber("Mesh.Algorithm3D", job.mesher.algorithm_3d)
    # Generate and audit a linear mesh without Gmsh's automatic optimisation; the order is raised
    # afterwards, once the audit has passed (see _elevate_if_accepted).
    option.setNumber("Mesh.ElementOrder", 1)
    option.setNumber("Mesh.HighOrderOptimize", 0)
    # One thread.  HXT with 4 or 16 threads was faster (25 s to 11 s and 6-8 s on the 1 mm chip) but not
    # reproducible: two identical runs gave 33,758 and 33,761 tetrahedra.  Gmsh's default is 1; setting
    # it states that determinism depends on it.
    option.setNumber("General.NumThreads", 1)
    option.setNumber("Mesh.MeshSizeMax", job.sizes.max_size)
    if job.mesher.algorithm_2d is not None:
        option.setNumber("Mesh.Algorithm", job.mesher.algorithm_2d)
    option.setNumber("Mesh.MshFileVersion", _MSH_VERSION)
    # Save only what the XAO's physical groups name, so the file carries the groups and nothing extra.
    option.setNumber("Mesh.SaveAll", 0)


def _add_size_fields(sizes: SizeRequests) -> None:
    """Background size field: each requested size near its curves or around its centre, growing to ``max_size``."""
    wanted = [*_graded_fields(sizes), *sizes.fields]
    if not wanted and not sizes.local_sizes:
        return
    fields = [
        _size_field(list(w.curves), w.size, sizes.max_size, w.distance) for w in wanted
    ]
    fields += [_ball_field(local, sizes) for local in sizes.local_sizes]
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


def _ball_field(local: LocalSize, sizes: SizeRequests) -> int:
    """``local.size`` within its core radius, growing linearly by the gradation to ``max_size`` beyond.

    Gmsh's Ball field is a closed-form function of the point: unlike a Distance field it samples no curve.
    """
    field = gmsh.model.mesh.field
    ball = field.add("Ball")
    x, y, z = local.centre
    field.setNumber(ball, "XCenter", x)
    field.setNumber(ball, "YCenter", y)
    field.setNumber(ball, "ZCenter", z)
    field.setNumber(ball, "Radius", local.core_radius)
    field.setNumber(ball, "VIn", local.size)
    field.setNumber(ball, "VOut", sizes.max_size)
    field.setNumber(ball, "Thickness", (sizes.max_size - local.size) / sizes.gradation)
    return ball


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
