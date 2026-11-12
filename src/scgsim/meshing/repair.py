"""The repair loop: mesh, audit, refine around the bad elements, repeat within a budget.

Owns the deterministic decisions between rounds, whatever recipe chose the initial sizes.  Every round is
a fresh worker process, so the loop's only state is the list of local sizes and the history; the same
sidecar, XAO, initial sizes, policy and threshold give the same sequence of meshes.  The recipe's sizes
stay as they are.  Each bad element gets a local size, a ball around it (see :func:`refine_locally`),
wherever it is: at a point, on a curve, in a face or in the interior.

The loop stops when the mesh is clean, or when a budget ends it: the rounds, the tetrahedra, the time
(:class:`Budgets`), or a bad count that no longer falls (``STALL_ROUNDS``).  Only a clean stop delivers a
mesh, except for a report-only strategy (``RepairInputs.enforce_threshold`` false): it stops after its
first mesh and delivers it within the tetrahedron budget whatever its quality.

The audit and the threshold apply to the linear mesh.  When the mesher options ask for a higher order, the
worker of the round that passes the audit raises the order and writes that mesh as well; the loop keeps
both files.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType

from .driver import RoundOutcome, run_round
from .findings import BadElement, MeshAudit
from .job import Elevation, MeshJob
from .mesher import MesherOptions
from .sizing import LocalSize, SizeRequests, ladder_size

# Why the loop stopped.  Only "clean" is success; a report-only strategy also delivers on "report_only".
STOP_CLEAN = "clean"
# A report-only strategy's mesh, delivered with tetrahedra below the threshold it does not enforce.
STOP_REPORT_ONLY = "report_only"
STOP_ROUNDS = "rounds"  # the refinement-round budget is spent
STOP_ELEMENTS = "elements"  # the mesh exceeds the tetrahedron budget
STOP_TIME = "time_limit"  # the time budget ran out
STOP_STALLED = "stalled"  # the bad count stopped falling (see STALL_ROUNDS)
STOP_CRASH = "mesh_crash"  # no algorithm produced a mesh
# The linear mesh passed but its order could not be raised.
STOP_ELEVATION = "elevation_failed"
# The stops after which the loop delivers its last mesh.
DELIVERED_STOPS = (STOP_CLEAN, STOP_REPORT_ONLY)

# The loop stops when this many rounds in a row leave more bad elements than the fewest any earlier round
# had (Human, 2026-11-10, issue 124): refining further has stopped paying, which is what geometry the
# mesher cannot repair (an edge of one grid step, a spike) looks like.
STALL_ROUNDS = 2


@dataclass(frozen=True)
class TimeBudget:
    """The time budget of one model: its footprint and the limit and target that footprint gives."""

    footprint_mm2: float | None  # None when the extent could not be read in time
    limit_seconds: float
    target_seconds: float


@dataclass(frozen=True)
class Budgets:
    """Limits of the loop: refinement rounds, tetrahedra in any accepted mesh, and wall-clock time.

    ``max_tetrahedra`` is a hard cap: a mesh above it is rejected even when its quality is clean.

    The time budget is the Human's rule of 2026-11-10 (issue 124): a model whose footprint (its x by y
    extent) is at most ``time_reference_area_mm2`` must mesh within ``time_limit_seconds``, and a larger
    one gets proportionally more.  The clock starts with the meshing call.  The limit decides whether a
    round may start and how long its worker may run; writing the round's request and reading its result
    are not counted, so a result can arrive later than the limit by that much.  A round still running at
    the limit is stopped and the call fails with its report; it never returns a mesh below the threshold.
    ``time_target_seconds`` is the good-case target on the same scale; the report says whether it was met,
    but it stops nothing.
    """

    max_rounds: int = 8
    max_tetrahedra: int = 2_000_000
    time_limit_seconds: float = 180.0
    time_target_seconds: float = 60.0
    time_reference_area_mm2: float = 4.0

    def __post_init__(self) -> None:
        """Reject a time budget that is not a positive number."""
        for name in (
            "time_limit_seconds",
            "time_target_seconds",
            "time_reference_area_mm2",
        ):
            value = getattr(self, name)
            if not (
                isinstance(value, (int, float)) and math.isfinite(value) and value > 0
            ):
                raise ValueError(f"{name} must be a positive number, got {value!r}.")

    def time_budget(self, footprint_mm2: float | None) -> TimeBudget:
        """The limit and target for a model of this footprint: the base values times max(1, A / A_ref).

        ``None`` (the footprint is not known) gives the base values, the smallest any model has.
        """
        scale = (
            1.0
            if footprint_mm2 is None
            else max(1.0, footprint_mm2 / self.time_reference_area_mm2)
        )
        return TimeBudget(
            footprint_mm2,
            self.time_limit_seconds * scale,
            self.time_target_seconds * scale,
        )


@dataclass(frozen=True)
class RepairPolicy:
    """How the loop refines around bad elements.  It does not depend on the sizing recipe."""

    # A bad element's longest edge, times this, is the size first asked for around it.
    edge_fraction: float = 0.5


@dataclass(frozen=True)
class RepairInputs:
    """Everything the loop reads: the model files, the recipe's initial sizes, and the policies."""

    xao: Path
    sidecar: Path
    # The recipe's sizes; every round keeps them and adds the loop's local sizes.
    initial: SizeRequests
    threshold: float
    policy: RepairPolicy
    budgets: Budgets
    mesher: MesherOptions
    work_dir: Path
    # time.perf_counter() reading at which the time budget runs out; infinite for no time limit.
    deadline: float = math.inf
    # False for a report-only strategy: one mesh, no repair, delivered below the threshold too.
    enforce_threshold: bool = True


@dataclass(frozen=True)
class RoundRecord:
    """One round as the report shows it (timing is kept apart so records compare equal across runs)."""

    index: int
    local_sizes: int  # local sizes the round was meshed with
    smallest_local_size: float | None
    algorithm_3d: int
    fallback: str | None
    failure: str | None
    tetrahedra: int | None
    min_quality: float | None
    bad_count: int | None
    elevation_error: str | None
    timed_out: bool


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
    curve_sizes: dict[int, float]  # the recipe's curve sizes, which every round used
    local_sizes: tuple[
        LocalSize, ...
    ]  # the local sizes of the last round that was meshed
    algorithm_3d: int  # algorithm of the last mesh
    fallback: str | None  # why HXT was abandoned in the last round that made a mesh


@dataclass(frozen=True)
class _MadeMesh:
    """The last round that produced a mesh: its outcome, audit, file and the local sizes it used."""

    outcome: RoundOutcome
    audit: MeshAudit
    mesh: Path
    elevated_mesh: Path | None
    local_sizes: tuple[LocalSize, ...]


def repair_mesh(inputs: RepairInputs) -> RepairOutcome:
    """Run the loop until the mesh is clean or a budget, a stall or a crash ends it."""
    local: tuple[LocalSize, ...] = ()
    history: list[RoundRecord] = []
    seconds: list[float] = []
    last: _MadeMesh | None = None
    stop = STOP_ROUNDS
    for index in range(inputs.budgets.max_rounds + 1):
        remaining = inputs.deadline - time.perf_counter()
        if remaining <= 0:
            stop = STOP_TIME
            break
        job = _job(inputs, local, index)
        outcome = run_round(
            job, inputs.work_dir, remaining if math.isfinite(remaining) else None
        )
        history.append(_record(index, local, outcome))
        seconds.append(outcome.wall_seconds)
        if outcome.audit is None:
            stop = STOP_TIME if outcome.timed_out else STOP_CRASH
            break
        last = _MadeMesh(
            outcome,
            outcome.audit,
            job.msh,
            job.elevation.msh if job.elevation and outcome.elevated else None,
            local,
        )
        stopped = _stop_reason(outcome, history, inputs)
        if stopped is not None:
            stop = stopped
            break
        local = refine_locally(
            local,
            outcome.audit.bad,
            inputs.initial.max_size,
            inputs.threshold,
            inputs.policy,
        )
    return _outcome(stop, history, seconds, last, inputs)


def unmeshed_outcome(stop: str, mesher: MesherOptions) -> RepairOutcome:
    """The outcome of a call that stopped before its first round: no history and no mesh."""
    return RepairOutcome(
        stop=stop,
        rounds=0,
        history=(),
        round_seconds=(),
        last_audit=None,
        last_mesh=None,
        elevated_mesh=None,
        geometry_order=1,
        curve_sizes={},
        local_sizes=(),
        algorithm_3d=mesher.algorithm_3d,
        fallback=None,
    )


def refine_locally(
    local: tuple[LocalSize, ...],
    bad: tuple[BadElement, ...],
    max_size: float,
    threshold: float,
    policy: RepairPolicy,
) -> tuple[LocalSize, ...]:
    """The local sizes of the next round: each bad element refines the local size it lies in, or gets its own.

    Every bad element asks for a size (see :func:`wanted_size`), and where it is refined depends on its
    shape:

    * A *needle* has an edge shorter than that size: a model edge the mesher cannot lengthen (one grid
      step at a spike or a band join).  It is refined where it is thinnest, at the midpoint of that edge,
      with a fine core one element wide.  A core as wide as the element would cost (width / size)**3
      elements, millions for a 5 um element refined to 0.04 um.
    * A *sliver* has no such edge: it is flat because of where the mesher put its four points.  It is
      refined over the region it covers: a fine core of half its longest edge around its centroid, which
      costs at most about (longest / shortest edge)**3 elements.

    An element lies in a local size when the size's fine core covers what the element needs: its point,
    and for a sliver its whole core.  The core did not fix it, so it must get finer.  An element no core
    covers gets a local size of its own, so a sliver beside a needle is not squeezed into the needle's
    one-element core.  Claiming only by the core lets the refinement follow a thin feature along its
    length.  On the flip-chip Disk Qubit (issue 124), whose transmon pads have spikes one nanometre wide
    and over a micrometre long, claiming every element within the first element's longest edge left 4
    elements there after 7 rounds; claiming by the core cleaned them in 2.  Refining slivers at a point
    only, without the wide core, left 1 to 14 slivers along the band edges with bands on; the wide core
    cleaned the default recipe's mesh in 2.

    A local size takes the finest its elements ask for, on the ladder, and one that existed already goes
    at least one ladder step finer.  A needle core shrinks with its size, so an element it claimed at the
    old width can be left uncovered by the final core: such an element leaves it for a local size of its
    own, and the size it left is sized again from the elements still covered.  Elements are taken in tag
    order and a local size is found by its position in the list, so the same bad elements always give the
    same sizes.
    """
    refined = list(local)
    members: dict[int, list[_Claim]] = {}
    for element in sorted(bad, key=lambda e: e.element):
        claim = _claim(element, threshold, policy)
        holder = _holding(refined, claim)
        if holder is None:
            refined.append(claim.own_size(max_size))
            holder = len(refined) - 1
        members.setdefault(holder, []).append(claim)
    moved: list[_Claim] = []
    for index, claims in sorted(members.items()):
        previous = local[index] if index < len(local) else None
        refined[index], left = _settle(refined[index], previous, claims, max_size)
        moved.extend(left)
    refined.extend(claim.own_size(max_size) for claim in moved)
    return tuple(refined)


@dataclass(frozen=True)
class _Claim:
    """What one bad element asks of the local sizes: where, how fine, and how wide a core."""

    spot: tuple[float, float, float]
    size: float
    radius: float | None  # None for a needle, whose core is as wide as its size

    @property
    def reach(self) -> float:
        """How far from ``spot`` a core must reach to cover it: a sliver's radius, a needle's point only.

        A needle's core is one element of whatever size it is given, so a core holding its point covers it.
        """
        return 0.0 if self.radius is None else self.radius

    def own_size(self, max_size: float) -> LocalSize:
        """A local size for this element alone."""
        return LocalSize(self.spot, ladder_size(self.size, max_size), self.radius)


def _claim(element: BadElement, threshold: float, policy: RepairPolicy) -> _Claim:
    size = wanted_size(element, threshold, policy)
    if element.edge_min < size:
        return _Claim(element.shortest_edge_midpoint, size, None)
    return _Claim(element.centroid, size, element.edge_max / 2)


def _settle(
    ball: LocalSize,
    previous: LocalSize | None,
    claims: list[_Claim],
    max_size: float,
) -> tuple[LocalSize, list[_Claim]]:
    """The final size of one local size and the claims its final core no longer covers.

    Removing a claim can only make the size coarser and a needle core wider, so the claims kept stay
    covered; the loop ends when every kept claim is covered.
    """
    kept = list(claims)
    while True:
        size = ladder_size(min(c.size for c in kept), max_size)
        if previous is not None:
            size = min(size, previous.size / 2)
        settled = replace(ball, size=size)
        outside = [c for c in kept if not _covers(settled, c)]
        if not outside:
            return settled, [c for c in claims if c not in kept]
        kept = [c for c in kept if c not in outside]
        if not kept:
            # Only an existing size can lose every claim (a new one keeps its creator, at the centre).
            # It still goes one step finer, as every size that held a bad element does.
            assert previous is not None
            return replace(ball, size=previous.size / 2), list(claims)


def wanted_size(element: BadElement, threshold: float, policy: RepairPolicy) -> float:
    """The size one bad element asks for: ``edge_fraction`` of its longest edge, scaled by its quality deficit.

    An element whose quality is limited by one short edge it cannot lose (a model edge of a nanometre) has a
    quality roughly proportional to its shortest over its longest edge, so its long edges must shrink by
    quality / threshold to reach the threshold.  The scale never makes the size coarser
    (``edge_fraction`` times the longest edge) nor finer than ``edge_fraction`` times the shortest edge,
    which also covers an inverted element (quality at or below zero).
    """
    deficit = element.quality / threshold
    scale = min(1.0, max(deficit, element.edge_min / element.edge_max))
    return policy.edge_fraction * element.edge_max * scale


def refinement_reach(
    element: BadElement,
    local: tuple[LocalSize, ...],
    threshold: float,
    policy: RepairPolicy,
) -> tuple[int | None, tuple[float, float, float], float]:
    """Where the loop refines a bad element and how far that refinement reaches.

    Returns the index of the local size whose fine core holds the element, with that core's centre and radius;
    or, when none holds it, ``None`` with the place and reach the element would ask for: its point and the size
    it wants for a needle (one element of that size), its centroid and half its longest edge for a sliver.
    """
    claim = _claim(element, threshold, policy)
    holder = _holding(list(local), claim)
    if holder is not None:
        ball = local[holder]
        return holder, ball.centre, ball.core_radius
    return None, claim.spot, claim.size if claim.radius is None else claim.radius


def _holding(local: list[LocalSize], claim: _Claim) -> int | None:
    """Index of the first local size whose fine core covers ``claim``; ``None`` when none does."""
    for index, ball in enumerate(local):
        if _covers(ball, claim):
            return index
    return None


def _covers(ball: LocalSize, claim: _Claim) -> bool:
    return math.dist(ball.centre, claim.spot) + claim.reach <= ball.core_radius


def _job(inputs: RepairInputs, local: tuple[LocalSize, ...], index: int) -> MeshJob:
    initial = inputs.initial
    msh = inputs.work_dir / f"round_{index:02d}.msh"
    return MeshJob(
        xao=inputs.xao,
        sidecar=inputs.sidecar,
        msh=msh,
        sizes=SizeRequests(
            initial.max_size,
            initial.gradation,
            MappingProxyType(dict(initial.curve_sizes)),
            initial.fields,
            # A recipe's own local sizes come first and stay as they are; the loop refines only its own.
            (*initial.local_sizes, *local),
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
        linear_msh.with_suffix(".elevated.msh"),
        inputs.budgets.max_tetrahedra,
        inputs.enforce_threshold,
    )


def _stop_reason(
    outcome: RoundOutcome, history: list[RoundRecord], inputs: RepairInputs
) -> str | None:
    """The reason to stop after a round that made a mesh; ``None`` to refine and go on.

    The tetrahedron budget caps every mesh the loop accepts, a clean one included.  A clean mesh whose
    order could not be raised is not delivered: the caller asked for a mesh of that order.  A report-only
    strategy stops after its first mesh, delivered under the same two conditions.
    """
    audit = outcome.audit
    assert audit is not None
    if audit.accepted(inputs.budgets.max_tetrahedra):
        return STOP_ELEVATION if outcome.elevation_error else STOP_CLEAN
    if audit.tetrahedra > inputs.budgets.max_tetrahedra:
        return STOP_ELEMENTS
    if not inputs.enforce_threshold:
        return STOP_ELEVATION if outcome.elevation_error else STOP_REPORT_ONLY
    if stalled([r.bad_count for r in history if r.bad_count is not None]):
        return STOP_STALLED
    if history[-1].index >= inputs.budgets.max_rounds:
        return STOP_ROUNDS
    return None


def stalled(bad_counts: list[int]) -> bool:
    """Whether the last ``STALL_ROUNDS`` counts all stayed at or above the fewest of the counts before them."""
    if len(bad_counts) <= STALL_ROUNDS:
        return False
    return min(bad_counts[-STALL_ROUNDS:]) >= min(bad_counts[:-STALL_ROUNDS])


def _record(
    index: int, local: tuple[LocalSize, ...], outcome: RoundOutcome
) -> RoundRecord:
    audit = outcome.audit
    return RoundRecord(
        index=index,
        local_sizes=len(local),
        smallest_local_size=min((ball.size for ball in local), default=None),
        algorithm_3d=outcome.algorithm_3d,
        fallback=outcome.fallback,
        failure=outcome.failure,
        tetrahedra=audit.tetrahedra if audit else None,
        min_quality=audit.min_quality if audit else None,
        bad_count=audit.bad_count if audit else None,
        elevation_error=outcome.elevation_error,
        timed_out=outcome.timed_out,
    )


def _outcome(
    stop: str,
    history: list[RoundRecord],
    seconds: list[float],
    last: _MadeMesh | None,
    inputs: RepairInputs,
) -> RepairOutcome:
    elevated_mesh = last.elevated_mesh if last and stop in DELIVERED_STOPS else None
    final = last.outcome if last else None
    return RepairOutcome(
        stop=stop,
        rounds=max(len(history) - 1, 0),
        history=tuple(history),
        round_seconds=tuple(seconds),
        last_audit=last.audit if last else None,
        last_mesh=last.mesh if last else None,
        elevated_mesh=elevated_mesh,
        geometry_order=inputs.mesher.geometry_order if elevated_mesh else 1,
        curve_sizes=dict(inputs.initial.curve_sizes),
        local_sizes=last.local_sizes if last else (),
        algorithm_3d=final.algorithm_3d
        if final
        else _last_algorithm(history, inputs.mesher),
        fallback=final.fallback if final else _last_fallback(history),
    )


def _last_algorithm(history: list[RoundRecord], mesher: MesherOptions) -> int:
    """The algorithm of the last round tried; the requested one when the time ran out before any round."""
    return history[-1].algorithm_3d if history else mesher.algorithm_3d


def _last_fallback(history: list[RoundRecord]) -> str | None:
    return history[-1].fallback if history else None
