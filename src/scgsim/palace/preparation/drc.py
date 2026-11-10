"""The two stages of ``simulation_drc``: the geometry build, then the meshing-stage check of the built model.

Owns the order of the stages and the report that keeps them apart.  The geometry stage is the preparation
and SGB build that ``mesh()`` runs; a layout it refuses raises that stage's own error, unchanged.  Only a
model the geometry stage built goes to the meshing stage (``scgsim.meshing.check_fine_structure``), which
finds the fine edges and faces the mesher cannot use and says where they are and who owns them.  A
meshing-stage refusal raises :class:`SimulationDrcError` carrying both stages; a pass returns the
:class:`SimulationDrcReport`, warnings included.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

from scgsim.meshing import FineFeature, check_fine_structure

from ..mesh.build import RouteGeometry

Stage = Literal["geometry", "meshing"]


@dataclass(frozen=True)
class StageResult:
    """Whether one stage of ``simulation_drc`` passed, and what it found."""

    stage: Stage
    passed: bool
    findings: tuple[FineFeature, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class SimulationDrcReport:
    """Both stages of one ``simulation_drc`` call: the geometry stage first, then the meshing stage."""

    geometry: StageResult
    meshing: StageResult

    @property
    def passed(self) -> bool:
        """Whether both stages passed (a meshing-stage warning does not fail it)."""
        return self.geometry.passed and self.meshing.passed

    def render_text(self) -> str:
        """Each stage's verdict, and every meshing-stage finding as one paragraph."""
        lines = [f"geometry stage: {'passed' if self.geometry.passed else 'refused'}"]
        refusing = sum(1 for f in self.meshing.findings if f.action == "refuse")
        warning = len(self.meshing.findings) - refusing
        verdict = "passed" if self.meshing.passed else "refused"
        lines.append(
            f"meshing stage: {verdict} ({refusing} refusing, {warning} warning)"
        )
        lines.extend(f"  {finding.describe()}" for finding in self.meshing.findings)
        return "\n".join(lines) + "\n"


class SimulationDrcError(Exception):
    """The meshing stage refused a layout whose geometry stage passed.

    ``report`` is the :class:`SimulationDrcReport` of both stages; its meshing findings say where each fine
    edge or face is, who owns it and how to fix it.
    """

    def __init__(self, report: SimulationDrcReport) -> None:
        refusals = [f for f in report.meshing.findings if f.action == "refuse"]
        text = "\n".join(f"  {finding.describe()}" for finding in refusals)
        super().__init__(
            "simulation_drc: the geometry stage passed; the meshing stage refused "
            f"{len(refusals)} fine edge(s) or face(s) the mesher cannot use:\n{text}"
        )
        self.report = report


def run_stages(build_geometry: Callable[[Path], RouteGeometry]) -> SimulationDrcReport:
    """Build the geometry into a temporary folder, check the built model for meshing, and remove the folder.

    ``build_geometry`` is the simulation's own geometry build; its error is the geometry stage's refusal and
    propagates unchanged, so the meshing stage runs only on a model that built.
    """
    with TemporaryDirectory(prefix="scgsim-simulation-drc-") as folder:
        # The geometry build requires an empty folder; the temporary one is, and goes away with the check.
        geometry = build_geometry(Path(folder))
        meshing = check_fine_structure(geometry.artifacts.sidecar)
    report = SimulationDrcReport(
        geometry=StageResult("geometry", passed=True),
        meshing=StageResult("meshing", meshing.passed, meshing.findings),
    )
    if not report.passed:
        raise SimulationDrcError(report)
    return report
