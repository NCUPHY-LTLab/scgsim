"""The repair loop: mesh, audit, refine the curves that bad elements touch, repeat within a budget.

Owns the deterministic decisions between rounds, whatever recipe chose the initial sizes.  Every round is
a fresh worker process, so the loop's only state is the per-curve size table and the history; the same
sidecar, XAO, initial sizes, policy and threshold give the same sequence of meshes.  The loop refines
curves only: a bad element that touches no curve and no point (it lies in a face or the interior) is
reported and counted, not refined, because the loop has no remedy for it.

The audit and the threshold apply to the linear mesh.  When the mesher options ask for a higher order, the
worker of the round that passes the audit raises the order and writes that mesh as well; the loop keeps
both files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from .driver import RoundOutcome, run_round
from .findings import BadElement, MeshAudit
from .job import Elevation, MeshJob
from .mesher import MesherOptions
from .sizing import SizeRequests, ladder_size

# Why the loop stopped.  Only "clean" is success.
STOP_CLEAN = "clean"
STOP_ROUNDS = "rounds"  # the refinement-round budget is spent
STOP_ELEMENTS = "elements"  # the mesh exceeds the tetrahedron budget
STOP_CRASH = "mesh_crash"  # no algorithm produced a mesh
STOP_UNATTRIBUTED = (
    "unattributed"  # bad elements remain but none touches a curve the loop may refine
)
# The linear mesh passed but its order could not be raised.
STOP_ELEVATION = "elevation_failed"


@dataclass(frozen=True)
class Budgets:
    """Limits of the loop: refinement rounds after the first mesh, and tetrahedra in any accepted mesh.

    ``max_tetrahedra`` is a hard cap: a mesh above it is rejected even when its quality is clean.
    """

    max_rounds: int = 8
    max_tetrahedra: int = 2_000_000


@dataclass(frozen=True)
class RepairPolicy:
    """How the loop refines the curves around bad elements.  It does not depend on the sizing recipe."""

    # A bad element's longest edge, times this, is the first size tried for the curves around it.
    edge_fraction: float = 0.5
    # Only curves whose local edge is at least this fraction of the longest local edge of the element's
    # candidates are refined; the others cannot shorten the element's long edge.
    culprit_fraction: float = 0.5
    # Shorter curves (the 0.2 um vertical edges of a film wall) are ignored, since refining a curve
    # cannot shorten an element edge below its length.
    min_curve_length_um: float = 1.0


@dataclass(frozen=True)
class RepairInputs:
    """Everything the loop reads: the model files, the recipe's initial sizes, and the policies."""

    xao: Path
    sidecar: Path
    # The recipe's sizes; the loop never makes a curve it sized coarser than that.
    initial: SizeRequests
    threshold: float
    policy: RepairPolicy
    budgets: Budgets
    mesher: MesherOptions
    work_dir: Path


@dataclass(frozen=True)
class RoundRecord:
    """One round as the report shows it (timing is kept apart so records compare equal across runs)."""

    index: int
    refined_curves: int  # curves with a size request in this round
    smallest_size: float | None
    algorithm_3d: int
    fallback: str | None
    failure: str | None
    tetrahedra: int | None
    min_quality: float | None
    bad_count: int | None
    elevation_error: str | None


@dataclass(frozen=True)
class RepairOutcome:
    """How the loop ended and the last mesh it produced."""

    stop: str
    rounds: int  # refinement rounds done after the first mesh
    history: tuple[RoundRecord, ...]
    round_seconds: tuple[float, ...]
    last_audit: MeshAudit | None  # audit of the last round that made a mesh
    last_mesh: Path | None  # that round's linear mesh
    # The same mesh at the requested order, when the loop ended clean and an order above 1 was asked.
    elevated_mesh: Path | None
    # Order of the mesh to deliver: the requested one when elevated, else 1.
    geometry_order: int
    curve_sizes: dict[int, float]  # the size table of the last round that was meshed
    algorithm_3d: int  # algorithm of the last mesh
    fallback: str | None  # why HXT was abandoned in the last round that made a mesh


@dataclass(frozen=True)
class _MadeMesh:
    """The last round that produced a mesh: its outcome, audit, file and the size table it used."""

    outcome: RoundOutcome
    audit: MeshAudit
    mesh: Path
    elevated_mesh: Path | None
    curve_sizes: dict[int, float]


def repair_mesh(inputs: RepairInputs) -> RepairOutcome:
    """Run the loop until the mesh is clean or a budget or a crash ends it."""
    initial = inputs.initial
    rule_sizes = dict(initial.curve_sizes)
    curve_sizes = dict(initial.curve_sizes)
    history: list[RoundRecord] = []
    seconds: list[float] = []
    last: _MadeMesh | None = None
    stop = STOP_ROUNDS
    for index in range(inputs.budgets.max_rounds + 1):
        job = _job(inputs, curve_sizes, index)
        outcome = run_round(job, inputs.work_dir)
        history.append(_record(index, curve_sizes, outcome))
        seconds.append(outcome.wall_seconds)
        if outcome.audit is None:
            stop = STOP_CRASH
            break
        last = _MadeMesh(
            outcome,
            outcome.audit,
            job.msh,
            job.elevation.msh if job.elevation and outcome.elevated else None,
            dict(curve_sizes),
        )
        stopped = _stop_reason(outcome.audit, outcome.elevation_error, index, inputs)
        if stopped is not None:
            stop = stopped
            break
        refined = refine_sizes(
            curve_sizes, outcome.audit.bad, rule_sizes, initial.max_size, inputs.policy
        )
        if refined is None:
            stop = STOP_UNATTRIBUTED
            break
        curve_sizes = refined
    return _outcome(stop, history, seconds, last, inputs.mesher)


def refine_sizes(
    sizes: dict[int, float],
    bad: tuple[BadElement, ...],
    rule_sizes: dict[int, float],
    max_size: float,
    policy: RepairPolicy,
) -> dict[int, float] | None:
    """The next size table, or ``None`` when no bad element offers a curve to refine.

    A curve's first size is the smaller of half the bad elements' long edge there and the size the recipe
    asked for it (``rule_sizes``), rounded down the ladder; a curve that is still touched after it had a
    size goes one ladder step finer than its current size.
    """
    touched = _touched_curves(bad, policy)
    if not touched:
        return None
    refined = dict(sizes)
    for tag, limit in sorted(touched.items()):
        target = policy.edge_fraction * limit
        if tag in rule_sizes:
            target = min(target, rule_sizes[tag])
        size = ladder_size(target, max_size)
        refined[tag] = min(size, sizes[tag] / 2) if tag in sizes else size
    return refined


def _touched_curves(
    bad: tuple[BadElement, ...], policy: RepairPolicy
) -> dict[int, float]:
    """Curve tag -> the longest edge next to it that a bad element has there, for the curves to refine.

    Of an element's candidate curves (its own curves and those meeting at its points), only the culprits
    count: curves long enough to matter, whose local edge is a substantial part of the longest one.
    """
    touched: dict[int, float] = {}
    for element in bad:
        local = {
            c.tag: c.local_edge
            for c in element.candidates
            if c.length >= policy.min_curve_length_um and c.local_edge > 0
        }
        if not local:
            continue
        longest = max(local.values())
        for tag, edge in local.items():
            if edge >= policy.culprit_fraction * longest:
                touched[tag] = max(touched.get(tag, 0.0), min(element.edge_max, edge))
    return touched


def _job(inputs: RepairInputs, curve_sizes: dict[int, float], index: int) -> MeshJob:
    initial = inputs.initial
    msh = inputs.work_dir / f"round_{index:02d}.msh"
    return MeshJob(
        xao=inputs.xao,
        sidecar=inputs.sidecar,
        msh=msh,
        sizes=SizeRequests(
            initial.max_size,
            initial.gradation,
            MappingProxyType(dict(curve_sizes)),
            initial.fields,
        ),
        threshold=inputs.threshold,
        mesher=inputs.mesher,
        elevation=_elevation(inputs, msh),
    )


def _elevation(inputs: RepairInputs, linear_msh: Path) -> Elevation | None:
    """Ask for the elevated mesh beside the linear one, only when the requested order is above 1."""
    if inputs.mesher.geometry_order <= 1:
        return None
    return Elevation(
        linear_msh.with_suffix(".elevated.msh"), inputs.budgets.max_tetrahedra
    )


def _stop_reason(
    audit: MeshAudit, elevation_error: str | None, index: int, inputs: RepairInputs
) -> str | None:
    """The reason to stop after a round that made a mesh; ``None`` to refine and go on.

    The tetrahedron budget caps every mesh the loop accepts, a clean one included.  A clean mesh whose
    order could not be raised is not delivered: the caller asked for a mesh of that order.
    """
    if audit.accepted(inputs.budgets.max_tetrahedra):
        return STOP_ELEVATION if elevation_error else STOP_CLEAN
    if audit.tetrahedra > inputs.budgets.max_tetrahedra:
        return STOP_ELEMENTS
    if index >= inputs.budgets.max_rounds:
        return STOP_ROUNDS
    return None


def _record(
    index: int, curve_sizes: dict[int, float], outcome: RoundOutcome
) -> RoundRecord:
    audit = outcome.audit
    return RoundRecord(
        index=index,
        refined_curves=len(curve_sizes),
        smallest_size=min(curve_sizes.values()) if curve_sizes else None,
        algorithm_3d=outcome.algorithm_3d,
        fallback=outcome.fallback,
        failure=outcome.failure,
        tetrahedra=audit.tetrahedra if audit else None,
        min_quality=audit.min_quality if audit else None,
        bad_count=audit.bad_count if audit else None,
        elevation_error=outcome.elevation_error,
    )


def _outcome(
    stop: str,
    history: list[RoundRecord],
    seconds: list[float],
    last: _MadeMesh | None,
    mesher: MesherOptions,
) -> RepairOutcome:
    elevated_mesh = last.elevated_mesh if last and stop == STOP_CLEAN else None
    return RepairOutcome(
        stop=stop,
        rounds=len(history) - 1,
        history=tuple(history),
        round_seconds=tuple(seconds),
        last_audit=last.audit if last else None,
        last_mesh=last.mesh if last else None,
        elevated_mesh=elevated_mesh,
        geometry_order=mesher.geometry_order if elevated_mesh else 1,
        curve_sizes=last.curve_sizes if last else {},
        algorithm_3d=last.outcome.algorithm_3d if last else history[-1].algorithm_3d,
        fallback=last.outcome.fallback if last else history[-1].fallback,
    )
