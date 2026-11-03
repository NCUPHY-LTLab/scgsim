"""Stages S2 to S9 as one call: the validated conformal model of a route.

Given an effective ``GeometryBuildInput`` (or a prism catalogue) and a route, this returns
the labelled arrangements, the assembled topology, the route's group plan and the
validation report.  Validation always runs and a failed invariant raises
:class:`ValidationError`: no model that failed an invariant is ever returned.

Emission (S10), the sidecar and the cut-family generators build on this result in later
sub-stages.  ``tiles`` and ``workers`` configure the per-tile runtime; the model is
identical for every value (property-tested).
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field

from scgsim.geometry.models.input import GeometryBuildInput
from scgsim.geometry.models.representation import RepresentationContext
from scgsim.geometry.sgb import identity
from scgsim.geometry.sgb.assembly.builder import assemble_topology
from scgsim.geometry.sgb.assembly.records import Topology
from scgsim.geometry.sgb.cutfamilies.generate import with_cut_families
from scgsim.geometry.sgb.cutfamilies.requests import InsetRequest, RefinementRequest
from scgsim.geometry.sgb.derivation.labelled import (
    LabelledArrangements,
    derive_with_runtime,
    model_bbox,
)
from scgsim.geometry.sgb.parallel import Runtime
from scgsim.geometry.sgb.prisms.catalogue import build_prism_catalogue
from scgsim.geometry.sgb.records.prisms import PrismCatalogue
from scgsim.geometry.sgb.routes.groups import RoutedModel, apply_route
from scgsim.geometry.sgb.routes.policy import RoutePolicy, check_route, route_policy
from scgsim.geometry.sgb.validation.report import ValidationReport
from scgsim.geometry.sgb.validation.validate import validate_model


@dataclass
class ConformalModel:
    """A validated conformal model: arrangements, topology, route plan and the checks that passed."""

    labelled: LabelledArrangements
    topology: Topology
    routed: RoutedModel
    report: ValidationReport
    timings: dict[str, float] = field(default_factory=dict)  # stage -> seconds
    stage_kind: dict[str, str] = field(
        default_factory=dict
    )  # stage -> "serial" | "parallel"
    amdahl: dict[str, float] = field(
        default_factory=dict
    )  # serial/parallel wall, worker work, Amdahl limit
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def policy(self) -> RoutePolicy:
        """The route policy the model was built for."""
        return self.routed.policy

    def identity_sets(self) -> dict[str, frozenset[str]]:
        """Every entity id by kind: equal for two builds iff their identity is equal."""
        topo = self.topology
        sets = self.labelled.identity_sets()
        sets.update(
            {
                "vertices": frozenset(topo.vertices),
                "edges": frozenset(topo.edges),
                "faces": frozenset(topo.faces),
                "volumes": frozenset(topo.volumes),
            }
        )
        sets.update(
            {f"G:{name}": frozenset(g.ids) for name, g in self.routed.groups.items()}
        )
        return sets

    def identity_digest(self) -> str:
        """One hash over every entity id."""
        sets = self.identity_sets()
        return identity.digest("model", {k: sorted(v) for k, v in sorted(sets.items())})


def build_conformal_model(  # noqa: PLR0913 - the effective input, the route and its options
    build_input: GeometryBuildInput,
    representation: RepresentationContext,
    route: str,
    *,
    insets: tuple[InsetRequest, ...] = (),
    refinements: tuple[RefinementRequest, ...] = (),
    tiles: int = 1,
    workers: int = 1,
) -> ConformalModel:
    """The validated conformal model of an effective input under a route, with the requested cut families."""
    catalogue = build_prism_catalogue(build_input, representation)
    return build_model_from_catalogue(
        catalogue,
        route,
        insets=insets,
        refinements=refinements,
        tiles=tiles,
        workers=workers,
    )


def build_model_from_catalogue(  # noqa: PLR0913 - the catalogue, the route and its options
    catalogue: PrismCatalogue,
    route: str,
    *,
    insets: tuple[InsetRequest, ...] = (),
    refinements: tuple[RefinementRequest, ...] = (),
    tiles: int = 1,
    workers: int = 1,
    keep_raw: bool = False,
) -> ConformalModel:
    """The validated conformal model of a prism catalogue under a route, with the requested cut families.

    ``insets`` (Surface-EPR bands) and ``refinements`` (mesh-size curves) are the typed requests of
    ``cutfamilies.requests``; they become cut families before the arrangements are built.
    """
    policy = route_policy(route)
    check_route(policy, catalogue.prisms)
    catalogue = with_cut_families(catalogue, insets, refinements)
    # The pipeline allocates millions of small acyclic objects; the cyclic collector would only
    # re-scan them, so it is paused for the build.
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        with Runtime(tiles, workers, model_bbox(catalogue)) as rt:
            rt.keep_raw = keep_raw
            return _build(catalogue, policy, rt)
    finally:
        if was_enabled:
            gc.enable()


def _build(
    catalogue: PrismCatalogue, policy: RoutePolicy, rt: Runtime
) -> ConformalModel:
    labelled = derive_with_runtime(catalogue, rt)
    topology = assemble_topology(labelled, rt)
    with rt.serial("routes"):
        routed = apply_route(topology, policy)
    report = validate_model(labelled, topology, routed, rt)
    report.raise_if_failed()
    timings = {**rt.watch.serial, **rt.watch.parallel}
    kinds = {
        **{name: "serial" for name in rt.watch.serial},
        **{name: "parallel" for name in rt.watch.parallel},
    }
    counts = {**labelled.counts, **topology.counts(), "groups": len(routed.groups)}
    return ConformalModel(
        labelled,
        topology,
        routed,
        report,
        dict(timings),
        kinds,
        rt.watch.totals(),
        counts,
    )
