"""The geometry half of an AEDT simulation, run on a layout without AEDT: the check ``simulation_drc`` makes.

An AEDT simulation builds the plan's SGB model on the route its ``modeling`` realizes (``thin_film`` Route
A, ``solid`` Route C, issue #67 decision 7), exports its STEP with the import guard, and imports that into
AEDT.  The check runs every step of that pipeline that needs no AEDT, in the order the pipeline runs them:

1. the plan the simulation would be given, from the layout and its PDK data (:func:`layout_snapshot`);
2. :func:`~scgsim.aedt.preparation.sgb_model.prepare_sgb_model`: the physical interpretation, the
   port-sheet binding, the SGB build with its Surface-EPR bands, the STEP export and its guard, the Region
   box and the AEDT names of Nets and RLCs;
3. the import's own first step (:func:`~scgsim.aedt.runtime.families.sgb_import.plan_import`): the files
   again, every boundary planned on the guarded faces, and every port sheet treated;
4. for HFSS Eigenmode, the selection of every Surface-EPR row on the import the guard plans.

A step that refuses raises its own error, unchanged, so the check refuses a layout with the error the
simulation's preparation or import would raise.  AEDT meshes the model itself, so there is no meshing
stage: a :class:`SimulationDrcReport` carries the geometry stage only.  The model is built into a temporary
folder that is removed.

The check is Palace's (:mod:`scgsim.palace`'s ``simulation_drc``) in shape, with the same fixed settings
where they mean the same thing: zero coupon padding, the vacuum below, and every authored port sheet with a
fixed inductance.  It is a separate implementation, as each simulation class checks the geometry half of
its own pipeline (issue #247).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

from scgsim.aedt.epr.sgb_surfaces import surface_face_bindings
from scgsim.aedt.preparation.sgb_model import PreparedSgbModel
from scgsim.aedt.runtime.families.sgb_import import plan_import, planned_import
from scgsim.aedt.specs.common import LumpedRlc
from scgsim.geometry.source.plan import GeometryPlan, GeometryPlanSnapshot

_ROOT_ID = "chip"

# A layout with no vacuum of its own is checked in this (x, y, z) padding, as ``GeometryPlan.set_vacuum_region``
# takes it: the value Palace's check uses (issue #182), so both backends check one model.  None in the
# plane, so the Region ends at the layout's planar extent, and 1000 um below and above.
DRC_VACUUM_PADDING_UM = (0.0, 0.0, 1000.0)

# The geometry never reads a port sheet's inductance, but a port sheet in HFSS needs a treatment.
_PORT_INDUCTANCE_H = 1.0e-9


@dataclass(frozen=True)
class StageResult:
    """Whether one stage of an AEDT ``simulation_drc`` passed."""

    stage: Literal["geometry"]
    passed: bool


@dataclass(frozen=True)
class SimulationDrcReport:
    """The geometry stage of one AEDT ``simulation_drc`` call; AEDT meshes the model itself, so it has no meshing stage."""

    geometry: StageResult

    @property
    def passed(self) -> bool:
        """Whether the geometry stage passed."""
        return self.geometry.passed

    def render_text(self) -> str:
        """The stage's verdict."""
        return f"geometry stage: {'passed' if self.geometry.passed else 'refused'}\n"


def layout_snapshot(
    component: Any,
    *,
    layer_stack: Any,
    material_records: Mapping[str, Mapping[str, Any]],
    instances: Mapping[str, Any] | None,
    nets: Mapping[str, Sequence[str]] | None,
) -> GeometryPlanSnapshot:
    """The prepared snapshot a user would build from ``component`` for an AEDT simulation.

    The plan's root is ``"chip"``, so ``instances`` keys and ``nets`` members read ``"chip/<name>"``;
    ``nets`` is given after the instances are registered, as a simulation's plan is built.  Coupon padding
    is zero, since the check is about the drawn layout.  A layout whose stack has no ``VACUUM_REGION`` gets
    the plan's vacuum region padded by :data:`DRC_VACUUM_PADDING_UM`, since AEDT's Region is that box.
    """
    plan = GeometryPlan(
        component,
        root_id=_ROOT_ID,
        layer_stack=layer_stack,
        material_records=material_records,
        coupon_padding_um=0.0,
    )
    for path, reference in (instances or {}).items():
        plan.register_instance(path, reference)
    if nets is not None:
        plan.set_nets(nets)
    snapshot = plan.prepare()
    if "VACUUM_REGION" in snapshot.stack["solution_regions"]:
        return snapshot
    plan.set_vacuum_region(DRC_VACUUM_PADDING_UM)
    return plan.prepare()


def fixed_port_rlcs(snapshot: GeometryPlanSnapshot) -> tuple[LumpedRlc, ...]:
    """Every port sheet of ``snapshot``'s source with a parallel inductance the geometry never reads.

    Each names its sheet as a ``LumpedRlc`` does, by the ``PORT::<port_sheet_id>`` group of the model.
    """
    return tuple(
        LumpedRlc(region.port_sheet_id, "parallel", inductance_h=_PORT_INDUCTANCE_H)
        for region in snapshot.geometry_input.port_sheet_regions
    )


def run_geometry_stage(
    build_model: Callable[[Path], PreparedSgbModel],
    check_model: Callable[[PreparedSgbModel], None],
) -> SimulationDrcReport:
    """Build the model into a temporary folder with ``build_model``, check it with ``check_model``, remove the folder.

    Either one's error is the geometry stage's refusal and propagates unchanged.
    """
    with TemporaryDirectory(prefix="scgsim-aedt-simulation-drc-") as folder:
        check_model(build_model(Path(folder)))
    return SimulationDrcReport(StageResult("geometry", passed=True))


def check_eigenmode_model(model: PreparedSgbModel) -> None:
    """The HFSS import's first step on ``model``, then the selection of every Surface-EPR row it carries."""
    guard, sidecar, _ = plan_import(model, family="hfss")
    surface_face_bindings(sidecar, planned_import(guard))
