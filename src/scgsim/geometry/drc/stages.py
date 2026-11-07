"""Rules 3 to 5: run the production stages and turn each refusal into a finding.

Owns the order and the fail-closed handling of the stages a simulation passes before meshing:
``GeometryPlan`` construction and ``prepare()`` plus the physical interpretation (rule 3), SGB's
structural validation and sliver report (rule 4) and the Surface-EPR bands (rule 5). The stages are
the production functions, called the way ``EigenmodeSim`` calls them on Route B; this module adds no
geometry of its own. A stage that refuses stops the stages after it, which are reported as skipped.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from scgsim.geometry.drc.report import (
    PHYSICAL_INTERPRETATION,
    SGB_STRUCTURE,
    SURFACE_EPR_BANDS,
    Layer,
    LayoutFinding,
    SkippedRule,
)
from scgsim.geometry.models.physical import ApproximationProfile
from scgsim.geometry.models.representation import RepresentationContext
from scgsim.geometry.planning.physical import interpret_physical_geometry
from scgsim.geometry.sgb.cutfamilies.requests import (
    BAND_TYPES,
    InsetRequest,
    InterfaceSurfaces,
)
from scgsim.geometry.sgb.errors import (
    ContactError,
    NonManifoldError,
    OverlapError,
    SgbError,
    ValidationError,
)
from scgsim.geometry.sgb.model import ConformalModel, build_model_from_catalogue
from scgsim.geometry.sgb.prisms.catalogue import build_prism_catalogue
from scgsim.geometry.sgb.records.prisms import PrismCatalogue
from scgsim.geometry.sgb.slivers.report import sliver_report
from scgsim.geometry.source.occurrence import SourceRecipeError
from scgsim.geometry.source.plan import GeometryPlan

ROUTE = "B"
"""The route ``EigenmodeSim`` uses unless the caller selects Route A."""
PROFILE = ApproximationProfile("route_b", "v1", {})
"""The profile ``EigenmodeSim`` interprets Route B with when none is set."""


@dataclass(frozen=True)
class StageInputs:
    """What the production stages need, as the caller gave it to ``simulation_drc``."""

    component: Any
    root_id: str
    layer_stack: Any
    material_records: Mapping[str, Mapping[str, Any]]
    instances: Mapping[str, Any]
    nets: Mapping[str, Sequence[str]] | None
    coupon_padding_um: float
    vacuum_padding_um: (
        Any  # the forms GeometryPlan.set_vacuum_region takes, or None for no region
    )
    inset_margins_um: Sequence[float] | None
    min_feature_um: float | None


@dataclass
class StageOutcome:
    """The findings of the stages and the stages that did not run."""

    findings: list[LayoutFinding] = field(default_factory=list)
    skipped: list[SkippedRule] = field(default_factory=list)


def run_production_stages(inputs: StageInputs) -> StageOutcome:
    """Run rules 3 to 5 in order, stopping at the first stage that refuses."""
    outcome = StageOutcome()
    stages = _Stages(inputs, outcome)
    interpreted = stages.interpret()
    if interpreted is None:
        _skip(
            outcome,
            (SGB_STRUCTURE, SURFACE_EPR_BANDS),
            "physical interpretation refused the layout",
        )
        return outcome
    catalogue = stages.check_structure(*interpreted)
    if catalogue is None:
        _skip(
            outcome,
            (SURFACE_EPR_BANDS,),
            "SGB structural validation refused the layout",
        )
        return outcome
    stages.check_bands(catalogue)
    return outcome


def _skip(outcome: StageOutcome, rules: tuple[str, ...], reason: str) -> None:
    """Report ``rules`` as skipped because an earlier stage refused for ``reason``."""
    outcome.skipped.extend(SkippedRule(rule, reason) for rule in rules)


class _Stages:
    """The stages of one check; each returns what the next one needs, or ``None`` after a refusal."""

    def __init__(self, inputs: StageInputs, outcome: StageOutcome) -> None:
        self._inputs = inputs
        self._outcome = outcome
        self._plan: GeometryPlan | None = None

    def interpret(self) -> tuple[Any, RepresentationContext] | None:
        """Rule 3: the plan, ``prepare()`` and the physical interpretation of Route B."""
        try:
            plan = self._build_plan()
            snapshot = plan.prepare()
            effective, _, physical = interpret_physical_geometry(
                snapshot.geometry_input, snapshot.stack, PROFILE
            )
        except Exception as error:  # noqa: BLE001 - a production refusal is a finding, whatever its type
            self._outcome.findings.append(self._refusal(PHYSICAL_INTERPRETATION, error))
            return None
        return effective, physical.context

    def check_structure(
        self, effective: Any, context: RepresentationContext
    ) -> PrismCatalogue | None:
        """Rule 4: SGB's structural validation, then the sliver report against the caller's limit."""
        try:
            catalogue = build_prism_catalogue(effective, context)
            model = build_model_from_catalogue(catalogue, ROUTE)
        except Exception as error:  # noqa: BLE001 - a production refusal is a finding, whatever its type
            self._outcome.findings.append(self._refusal(SGB_STRUCTURE, error))
            return None
        self._outcome.findings.extend(self._sliver_findings(model))
        return catalogue

    def check_bands(self, catalogue: PrismCatalogue) -> None:
        """Rule 5: build the model again with the Surface-EPR bands of the caller's margins."""
        margins = self._inputs.inset_margins_um
        if not margins:
            self._outcome.skipped.append(
                SkippedRule(SURFACE_EPR_BANDS, "inset_margins_um was not given")
            )
            return
        try:
            build_model_from_catalogue(catalogue, ROUTE, insets=_band_requests(margins))
        except Exception as error:  # noqa: BLE001 - a production refusal is a finding, whatever its type
            self._outcome.findings.append(self._refusal(SURFACE_EPR_BANDS, error))

    def _build_plan(self) -> GeometryPlan:
        inputs = self._inputs
        plan = GeometryPlan(
            inputs.component,
            root_id=inputs.root_id,
            layer_stack=inputs.layer_stack,
            material_records=inputs.material_records,
            coupon_padding_um=inputs.coupon_padding_um,
        )
        self._plan = plan
        # A child can only be registered under a parent that is already registered.
        for path in sorted(inputs.instances, key=lambda item: (item.count("/"), item)):
            plan.register_instance(path, inputs.instances[path])
        if inputs.nets is not None:
            plan.set_nets(inputs.nets)
        if inputs.vacuum_padding_um is not None:
            plan.set_vacuum_region(inputs.vacuum_padding_um)
        return plan

    def _refusal(self, rule: str, error: Exception) -> LayoutFinding:
        """The finding for a stage that refused with ``error``, with the location and layer the error carries."""
        entity = error.subject if isinstance(error, SourceRecipeError) else None
        return LayoutFinding(
            rule,
            self._entity_layer(entity),
            entity,
            getattr(error, "location_um", None),
            _message(error),
            _hint(rule, error),
        )

    def _entity_layer(self, entity_id: str | None) -> Layer | None:
        """The GDS layer an Entity is declared on, read from the plan that refused it."""
        if entity_id is None or self._plan is None:
            return None
        try:
            geometry = next(
                item.geometry
                for item in self._plan.entities
                if item.semantic_id == entity_id
            )
            return int(geometry["gds_layer"]), int(geometry["gds_datatype"])
        except (StopIteration, KeyError, TypeError, ValueError):
            return None  # a contribution or port has no Entity of its own

    def _sliver_findings(self, model: ConformalModel) -> list[LayoutFinding]:
        """Horizontal edges and faces narrower than the limit the caller set; the report itself sets no limit.

        Vertical edges and walls measure the thickness the layer stack declares, which is no layout defect.
        """
        limit = self._inputs.min_feature_um
        if limit is None:
            return []
        topology = model.topology
        everything = len(topology.edges) + len(topology.faces)
        report = sliver_report(model, top=everything, drift=False)
        return [
            _sliver_finding(entry, what, limit)
            for what, key in (("edge", "shortest_edges"), ("face", "narrowest_faces"))
            for entry in report[key]
            if entry["kind"] == "horizontal" and entry["value_um"] < limit
        ]


def _sliver_finding(entry: Mapping[str, Any], what: str, limit: float) -> LayoutFinding:
    x, y = entry["location_um"][:2]
    return LayoutFinding(
        SGB_STRUCTURE,
        None,
        ", ".join(entry["sources"]) or None,
        (x, y),
        f"model {what} {entry['id']} measures {entry['value_um']} um, under the {limit} um limit",
        "widen the feature or move the geometry that creates it",
    )


def _band_requests(margins_um: Sequence[float]) -> tuple[InsetRequest, ...]:
    """One type-level request per Surface-EPR type, as ``set_surface_epr`` makes them for these margins."""
    return tuple(
        InsetRequest(
            f"epr_{interface_type}",
            InterfaceSurfaces(interface_type),
            tuple(float(margin) for margin in margins_um),
        )
        for interface_type in BAND_TYPES
    )


def _message(error: Exception) -> str:
    """The production message; an error of another kind keeps its type name, which the message may lack."""
    if isinstance(error, (ValueError, SgbError)):
        return str(error)
    return f"{type(error).__name__}: {error}"


def _hint(rule: str, error: Exception) -> str:
    if rule == SURFACE_EPR_BANDS and isinstance(error, ValidationError):
        return "SGB's own consistency check refused the bands at this location; report it to SCGSim with this finding"
    if rule == SURFACE_EPR_BANDS and not isinstance(error, OverlapError):
        return "give inset_margins_um strictly increasing positive multiples of the model grid, or fix the geometry the message names"
    if isinstance(error, ContactError):
        return "separate the conductors of different Nets by more than the grid, or give them one Net"
    if isinstance(error, OverlapError):
        return "remove the overlap, or declare distinct priorities for the overlapping Entities"
    if isinstance(error, NonManifoldError):
        return "give the two cells a shared edge, or separate them"
    if isinstance(error, SgbError):
        return "the message names the Entities and the stage that refused; change that geometry"
    if isinstance(error, SourceRecipeError):
        return "fix that Entity's source polygons or its selector; the message names the cause"
    return "the production stage refused this input; the message names the cause"
