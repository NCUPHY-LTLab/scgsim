"""The geometry half of an AEDT simulation, run on its plan without AEDT: the check ``simulation_drc`` makes.

An AEDT simulation builds the plan's SGB model on the route its ``modeling`` realizes (``thin_film`` Route
A, ``solid`` Route C, issue #67 decision 7), exports its STEP with the import guard, and imports that into
AEDT.  The check runs every step of that pipeline that needs no AEDT, in the order the pipeline runs them:

1. the snapshot of the plan the simulation is given, as ``set_plan(plan.prepare())`` binds it;
2. :func:`~scgsim.aedt.preparation.sgb_model.prepare_sgb_model`: the physical interpretation, the
   port-sheet binding, the SGB build with its Surface-EPR bands, the STEP export and its guard, the Region
   box and the AEDT names of Nets and RLCs;
3. the import's own first step (:func:`~scgsim.aedt.runtime.families.sgb_import.plan_import`): the files
   again, every boundary planned on the guarded faces, and every port sheet treated;
4. for HFSS Eigenmode, the selection of every Surface-EPR row on the import the guard plans.

A step that refuses raises its own error, unchanged, so the check refuses a plan with the error the
simulation's preparation or import would raise.  AEDT meshes the model itself, so there is no meshing
stage: a :class:`SimulationDrcReport` carries the geometry stage only.  The model is built into a temporary
folder that is removed.

The check is Palace's (:mod:`scgsim.palace`'s ``simulation_drc``) in shape: it takes the plan the
simulation is given and adds nothing to it (issue #266), and it gives every port sheet, a rectangular
junction's included, a fixed inductance.  It is a separate implementation, as each simulation class checks
the geometry half of its own pipeline (issue #247).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

from scgsim.aedt.epr.sgb_surfaces import surface_face_bindings
from scgsim.aedt.preparation.sgb_model import PreparedSgbModel
from scgsim.aedt.runtime.families.sgb_import import plan_import, planned_import
from scgsim.aedt.specs.common import LumpedRlc
from scgsim.geometry.source.plan import GeometryPlanSnapshot

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


def fixed_port_rlcs(snapshot: GeometryPlanSnapshot) -> tuple[LumpedRlc, ...]:
    """Every port sheet of ``snapshot``'s source with a parallel inductance the geometry never reads.

    A rectangular junction's sheet is one of them: the plan binds its terminals to an authored sheet.  Each
    names its sheet as a ``LumpedRlc`` does, by the ``PORT::<port_sheet_id>`` group of the model.
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
