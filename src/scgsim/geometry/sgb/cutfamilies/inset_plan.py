"""From inset requests to the band families of a model: type-level expansion, one set of margins per face and skipped faces.

Owns the step between the requests a caller writes and the concrete requests ``cutfamilies.supports`` resolves
(one band family per face).  A concrete request (:class:`ConductorSurface` or :class:`SolutionInterface`) is one
face and stays exactly as written.  A type-level request (:class:`InterfaceSurfaces`) is expanded here into the faces
SGB would give a Surface-EPR row of that type (``supports.faces_of_type``), so a consumer never builds the model once
to read the faces from the sidecar.

* **One family per face, one set of margins.**  Requests that land on one face are one family, so they must ask
  for the same margins (equal lattice lengths, so ``0.3`` and ``0.1 + 0.2`` agree): a Route A sheet serves an
  ``MA`` and an ``MS`` request with one family, and two requests that differ on one face are refused.  A concrete
  request keeps its own id; a face asked for only by type is a family with the id ``<id of the first request in id order>::<entity>::<side>``
  (``<first>|<second>::z=<z>`` for an interface), so ids, lineage and band group names (up to the collision qualifier) depend on the requests and
  the faces only.  Two concrete requests on one face are not merged: they
  are refused here, by the face they resolve to, because both families would put their bands into the same
  physical groups.  Refusing them before any partition also covers a face a port later covers entirely, where
  no band group is left to collide.
* **No gap edge.**  A concrete request fails closed on a face with no gap edge.  A face only type-level requests
  ask for is skipped and recorded (:class:`SkippedFace`), because "every face of a type" includes faces bands cannot be
  measured on (the substrate underside).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from scgsim.geometry.sgb.cutfamilies.requests import (
    ConductorSurface,
    InsetRequest,
    InterfaceSurfaces,
    SolutionInterface,
)
from scgsim.geometry.sgb.cutfamilies.supports import (
    FaceId,
    ResolvedSupport,
    faces_of_type,
    lattice_margins,
    locate_face,
    physical_labelling,
    resolve_support,
)
from scgsim.geometry.sgb.errors import InputError, NoGapEdgeError
from scgsim.geometry.sgb.records.grid import Grid
from scgsim.geometry.sgb.records.prisms import PrismCatalogue

NO_GAP_EDGE = "no_gap_edge"
"""Why a face is skipped: no conductor edge with exposed dielectric next to it to measure the bands from."""


@dataclass(frozen=True)
class SkippedFace:
    """A face a type-level request asked for and SGB left unbanded."""

    support: ConductorSurface | SolutionInterface
    reason: str = NO_GAP_EDGE


@dataclass(frozen=True)
class TypeExpansion:
    """What one type-level request became: the families that carry its bands and the faces it skipped."""

    request: InsetRequest
    banded: tuple[str, ...]  # family ids, in face order
    skipped: tuple[SkippedFace, ...]


@dataclass(frozen=True)
class InsetPlan:
    """The band families of a model (one resolved support each) and the expansion of each type-level request."""

    supports: tuple[ResolvedSupport, ...] = ()
    expansions: tuple[TypeExpansion, ...] = ()

    @property
    def requests(self) -> tuple[InsetRequest, ...]:
        """The concrete request behind every band family, in family order."""
        return tuple(support.request for support in self.supports)


@dataclass
class _FamilyJob:
    """One face to band: the family request, and the type-level requests that ask for the face too."""

    request: InsetRequest
    face: FaceId
    typed: list[InsetRequest] = field(default_factory=list)
    named: bool = False  # a concrete request names the face, so it cannot be skipped


@dataclass(frozen=True)
class _NamedFace:
    """A concrete request and the face it names."""

    request: InsetRequest
    face: FaceId


def plan_insets(
    catalogue: PrismCatalogue, requests: tuple[InsetRequest, ...]
) -> InsetPlan:
    """Expand the type-level requests, merge what lands on one face and resolve every family's support."""
    if not requests:
        return InsetPlan()
    labelled = physical_labelling(catalogue)
    named: list[_NamedFace] = []
    typed: dict[FaceId, list[InsetRequest]] = {}
    type_requests: list[InsetRequest] = []
    for request in requests:
        lattice_margins(request, catalogue.grid)
        if isinstance(request.support, InterfaceSurfaces):
            _check_type_request(request)
            type_requests.append(request)
            for face in faces_of_type(
                request.request_id, request.support, catalogue, labelled
            ):
                typed.setdefault(face, []).append(request)
        else:
            named.append(_NamedFace(request, locate_face(request, catalogue, labelled)))
    banded: dict[str, list[str]] = {r.request_id: [] for r in type_requests}
    skipped: dict[str, list[SkippedFace]] = {r.request_id: [] for r in type_requests}
    supports: list[ResolvedSupport] = []
    for job in _jobs(named, typed, catalogue.grid):
        try:
            supports.append(resolve_support(job.request, catalogue, labelled))
        except NoGapEdgeError:
            if job.named:
                raise
            for ask in job.typed:
                skipped[ask.request_id].append(SkippedFace(job.face.support()))
            continue
        for ask in job.typed:
            banded[ask.request_id].append(job.request.request_id)
    expansions = tuple(
        TypeExpansion(r, tuple(banded[r.request_id]), tuple(skipped[r.request_id]))
        for r in type_requests
    )
    return InsetPlan(tuple(supports), expansions)


def _check_type_request(request: InsetRequest) -> None:
    if request.interface is not None:
        raise InputError(
            f"inset request {request.request_id!r}: a type-level request carries its type in the support; "
            "leave interface unset"
        )


def _jobs(
    named: list[_NamedFace], typed: dict[FaceId, list[InsetRequest]], grid: Grid
) -> list[_FamilyJob]:
    """The families to build: named faces in request order, then the faces only a type asked for in face order."""
    named_faces = _faces_named_once(named)
    jobs = [_named_job(item, typed.get(item.face, []), grid) for item in named]
    for face in sorted(set(typed) - named_faces):
        asks = sorted(typed[face], key=lambda r: r.request_id)
        family_id = f"{asks[0].request_id}::{face.label()}"
        margins = _shared_margins(face, asks, grid)
        request = InsetRequest(family_id, face.support(), margins)
        jobs.append(_FamilyJob(request, face, asks))
    return jobs


def _faces_named_once(named: list[_NamedFace]) -> set[FaceId]:
    """The faces the concrete requests name; two concrete requests naming one face are an error."""
    by_face: dict[FaceId, list[str]] = {}
    for item in named:
        by_face.setdefault(item.face, []).append(item.request.request_id)
    for face, ids in sorted(by_face.items()):
        if len(ids) > 1:
            names = " and ".join(repr(i) for i in sorted(ids))
            raise InputError(
                f"inset requests {names} name the same face ({face.label()}); a face has one band "
                "family, so name it once with all its margins in one request"
            )
    return set(by_face)


def _named_job(item: _NamedFace, asks: list[InsetRequest], grid: Grid) -> _FamilyJob:
    """The family of a concrete request, as written; the type-level requests on its face must ask the same margins."""
    if not asks:
        return _FamilyJob(item.request, item.face, named=True)
    ordered = sorted(asks, key=lambda r: r.request_id)
    _shared_margins(item.face, [item.request, *ordered], grid)
    return _FamilyJob(item.request, item.face, ordered, named=True)


def _shared_margins(
    face: FaceId, asks: list[InsetRequest], grid: Grid
) -> tuple[float, ...]:
    """The margins every request on ``face`` asks for; requests that differ on the lattice are an error."""
    first = asks[0]
    expected = lattice_margins(first, grid)
    for ask in asks[1:]:
        if lattice_margins(ask, grid) != expected:
            raise InputError(
                f"inset requests {first.request_id!r} and {ask.request_id!r} ask different margins "
                f"({list(first.margins_um)} and {list(ask.margins_um)} um) for the same face ({face.label()}); "
                "a face has one band family, so every request on it must ask the same margins"
            )
    return first.margins_um
