"""The line cross-section of a plan at a cut: the plan's 3D model in memory, its plane section and the line on it.

This is an advanced entry, below the supported API.  A study does not draw cut lines: in a later step SCGSim's
line finder places each cut itself and calls :func:`cross_section_at` to check what lies there, and the 2D
simulations take the :class:`~scgsim.geometry.models.section.LineCrossSection` it returns.  Until the finder
lands, a cut can be passed here by hand.

The steps, in order: the plan's validated 3D model on the route, in memory (:func:`build_plan_model`); its plane
section along the cut (``sgb.section.build_section_model``, which refuses a cut that breaks a cut rule); and the
classification of that section (``sgb.section.line_cross_section``, which refuses a section that is not one
line).  No step depends on a solver and nothing is written.

The 3D model is the one the solvers build from the same snapshot when it has no lumped ports: the physical
interpretation for the route (``interpretation.interpret_physical_geometry``), the plan's vacuum, and the
Surface-EPR bands of the plan's specs (``surface_epr.build_banded_model``).  Route B (finite conductors, which
Palace excludes from its mesh) and Route C (conductor solids, as AEDT builds them) give equal records.
"""

from __future__ import annotations

from typing import Literal

from scgsim.geometry.interpretation.physical import interpret_physical_geometry
from scgsim.geometry.models.physical import ApproximationProfile, ProfileName
from scgsim.geometry.models.section import CutLine, LineCrossSection
from scgsim.geometry.sgb.model import ConformalModel
from scgsim.geometry.sgb.section import build_section_model, line_cross_section
from scgsim.geometry.source.plan import GeometryPlanSnapshot
from scgsim.geometry.source.validation import has_vacuum_solution_region
from scgsim.geometry.surface_epr.build import build_banded_model
from scgsim.refusal import Declared

SectionRoute = Literal["B", "C"]

ROUTE_PROFILES: dict[str, ProfileName] = {"B": "route_b", "C": "solid"}
"""The approximation profile each route is interpreted with, as the solvers interpret it: Palace builds Route B
with ``route_b`` and AEDT builds Route C (its ``solid`` modeling) with ``solid``; both keep conductors finite."""

_CROSS_SECTION_AT = Declared("cross_section_at", "the call to cross_section_at")
_BUILD_PLAN_MODEL = Declared("build_plan_model", "the call to build_plan_model")


def cross_section_at(
    snapshot: GeometryPlanSnapshot, cut: CutLine, *, route: SectionRoute = "B"
) -> LineCrossSection:
    """The classified line cross-section of ``snapshot``'s layout along ``cut`` (see the module docstring).

    Raises:
        TypeError: ``snapshot`` is no ``GeometryPlanSnapshot`` or ``cut`` no ``CutLine``.
        ValueError: ``route`` is not ``"B"`` or ``"C"``, or the plan has no vacuum region.
        InputError: the cut breaks a cut rule, or the section is not one line
            (:class:`~scgsim.geometry.exact.errors.InputError`, with the place in micrometres).
    """
    if not isinstance(cut, CutLine):
        raise TypeError(_CROSS_SECTION_AT.expected("cut", "a CutLine", cut))
    model = _plan_model(snapshot, route, _CROSS_SECTION_AT)
    return line_cross_section(build_section_model(model, cut))


def build_plan_model(
    snapshot: GeometryPlanSnapshot, *, route: SectionRoute = "B"
) -> ConformalModel:
    """The plan's validated 3D model on ``route``, in memory, with the bands of the plan's Surface-EPR specs.

    Raises the ``TypeError`` and ``ValueError`` of :func:`cross_section_at`, naming this call.
    """
    return _plan_model(snapshot, route, _BUILD_PLAN_MODEL)


def _plan_model(snapshot: object, route: object, call: Declared) -> ConformalModel:
    """The model of :func:`build_plan_model`; a refusal of an argument names ``call``, where it was given."""
    if not isinstance(snapshot, GeometryPlanSnapshot):
        raise TypeError(
            call.expected(
                "snapshot",
                "a GeometryPlanSnapshot from GeometryPlan.prepare()",
                snapshot,
            )
        )
    profile = ApproximationProfile(_profile_name(route, call), "v1", {})
    effective_input, _stack, plan = interpret_physical_geometry(
        snapshot.geometry_input, snapshot.stack, profile
    )
    if not has_vacuum_solution_region(effective_input):
        raise ValueError(
            Declared("the model", "the GeometryPlan").refusal(
                "has no vacuum region: the plan declares none; call plan.set_vacuum_region(...) before prepare()"
            )
        )
    return build_banded_model(
        effective_input,
        plan.context,
        route=str(route),
        surface_epr_specs=snapshot.surface_epr_specs,
    )


def _profile_name(route: object, call: Declared) -> ProfileName:
    """The profile of a route a section is taken on; Route A and unknown routes are refused."""
    if isinstance(route, str) and route in ROUTE_PROFILES:
        return ROUTE_PROFILES[route]
    if route == "A":
        raise ValueError(
            call.refusal(
                "takes route 'B' or 'C': Route A models conductors as sheets, which have no cross-section "
                "this version classifies"
            )
        )
    raise ValueError(call.expected("route", "'B' or 'C'", route))
