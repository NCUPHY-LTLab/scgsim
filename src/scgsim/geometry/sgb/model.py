"""Stages S2 to S9 as one call: the validated conformal model of a route.

Given an effective ``GeometryBuildInput`` (or a prism catalogue) and a route, this returns
the labelled arrangements, the assembled topology, the route's group plan and the
validation report.  Validation always runs and a failed invariant raises
:class:`ValidationError`: no model that failed an invariant is ever returned.

The steps, in order: the prism catalogue (S2); the route and entity-name checks; the cut
families of the inset requests (S3, measured on the physical pre-pass, S4 and S5 of the
physical prisms alone); the labelled arrangements with those families (S4 and S5); the
topology (S6); the route plan (S7 and S8); the validation (S9).  ``build`` writes the files
of the result (S10).
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from fractions import Fraction

from scgsim.geometry.models.input import GeometryBuildInput
from scgsim.geometry.models.representation import RepresentationContext
from scgsim.geometry.sgb.arrangement.derivation.labelled import (
    LabelledArrangements,
    derive_arrangements,
    derive_physical_arrangements,
)
from scgsim.geometry.sgb.bands.generate import (
    CutFamilies,
    build_cut_families,
    check_request_ids,
)
from scgsim.geometry.sgb.bands.inset_plan import InsetPlan, plan_insets
from scgsim.geometry.sgb.bands.requests import InsetRequest
from scgsim.geometry.sgb.checks.validation.report import ValidationReport
from scgsim.geometry.sgb.checks.validation.validate import validate_model
from scgsim.geometry.sgb.prisms.catalogue import build_prism_catalogue
from scgsim.geometry.sgb.records import identity
from scgsim.geometry.sgb.records.prisms import PrismCatalogue
from scgsim.geometry.sgb.stopwatch import Stopwatch
from scgsim.geometry.sgb.topology.assembly.builder import assemble_topology
from scgsim.geometry.sgb.topology.assembly.records import Topology
from scgsim.geometry.sgb.topology.routes.groups import RoutedModel, apply_route
from scgsim.geometry.sgb.topology.routes.naming import check_entity_names
from scgsim.geometry.sgb.topology.routes.policy import (
    RoutePolicy,
    check_route,
    route_policy,
)

LENGTH_UNIT = "um"
"""The unit of the model grid and of every length a hand-off carries."""


@dataclass
class ConformalModel:
    """A validated conformal model: arrangements, topology, route plan and the checks that passed."""

    labelled: LabelledArrangements
    topology: Topology
    routed: RoutedModel
    report: ValidationReport
    insets: tuple[
        InsetRequest, ...
    ]  # the inset requests the model was built with, as given
    inset_plan: InsetPlan  # the band families the inset requests became
    timings: dict[str, float] = field(default_factory=dict)  # stage -> seconds
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
        """One hash over every entity id and the physical grid they sit on.

        Entity ids hash lattice coordinates, so the same lattice at two grid sizes has the same ids but
        different areas and volumes; the exact grid size (the decimal text of the float, as
        ``Grid.exact_units`` reads it) and the length unit therefore enter the hash.
        """
        sets = self.identity_sets()
        return identity.digest(
            "model",
            {
                "grid": Fraction(repr(self.topology.grid_um)),
                "length_unit": LENGTH_UNIT,
                "entities": {k: sorted(v) for k, v in sorted(sets.items())},
            },
        )


def build_conformal_model(
    build_input: GeometryBuildInput,
    representation: RepresentationContext,
    route: str,
    *,
    insets: tuple[InsetRequest, ...] = (),
) -> ConformalModel:
    """The validated conformal model of an effective input under a route, with the requested inset bands."""
    catalogue = build_prism_catalogue(build_input, representation)
    return build_model_from_catalogue(catalogue, route, insets=insets)


def build_model_from_catalogue(
    catalogue: PrismCatalogue,
    route: str,
    *,
    insets: tuple[InsetRequest, ...] = (),
    keep_raw: bool = False,
) -> ConformalModel:
    """The validated conformal model of a prism catalogue under a route, with the requested inset bands.

    ``insets`` (Surface-EPR bands) are the typed requests of ``bands.requests``; they become cut families
    before the arrangements are built.  Pass bands only as
    ``insets``: the sidecar reads each band family's support and margins from its request, so a catalogue that
    already carries inset families builds but fails when its sidecar is written.
    """
    policy = route_policy(route)
    check_route(policy, catalogue.prisms)
    check_entity_names(
        p.entity_id for p in catalogue.prisms if p.material_kind != "port"
    )
    cut_families = _cut_families(catalogue, insets)
    # The pipeline allocates millions of small acyclic objects; the cyclic collector would only
    # re-scan them, so it is paused for the build.
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        return _build(cut_families, insets, policy, keep_raw=keep_raw)
    finally:
        if was_enabled:
            gc.enable()


def _cut_families(
    catalogue: PrismCatalogue, insets: tuple[InsetRequest, ...]
) -> CutFamilies:
    """S3 on its pre-pass: the arrangements of the physical prisms alone, the faces the requests land on, the families.

    Without requests there is no pre-pass and no band family.
    """
    check_request_ids(insets, catalogue)
    plan = InsetPlan()
    if insets:
        physical = derive_physical_arrangements(catalogue)
        plan = plan_insets(catalogue, insets, physical)
    return build_cut_families(catalogue, plan)


def _build(
    cut_families: CutFamilies,
    insets: tuple[InsetRequest, ...],
    policy: RoutePolicy,
    *,
    keep_raw: bool,
) -> ConformalModel:
    """S4 to S9 on the catalogue with its cut families."""
    watch = Stopwatch()
    labelled = derive_arrangements(
        cut_families.catalogue, keep_raw=keep_raw, watch=watch
    )
    topology = assemble_topology(labelled, watch)
    with watch.stage("routes"):
        routed = apply_route(topology, policy)
    report = validate_model(labelled, topology, routed, watch)
    report.raise_if_failed()
    counts = {**labelled.counts, **topology.counts(), "groups": len(routed.groups)}
    return ConformalModel(
        labelled,
        topology,
        routed,
        report,
        insets,
        cut_families.plan,
        dict(watch.seconds),
        counts,
    )
