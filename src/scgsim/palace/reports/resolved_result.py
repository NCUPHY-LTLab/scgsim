"""The result a user holds after resolving a returned run: the checked run record with its notebook views.

``resolve_palace_result`` reads and checks the run folder with ``results.resolve`` and returns that record as a
``ResolvedPalaceResult``, whose methods are the views of this stage.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import cast

from scgsim.palace.reports.plotting import ReportTheme
from scgsim.palace.reports.reports import (
    show_all_results,
    show_physics_quantities,
    show_run_trustworthiness,
    show_simulation_benchmark,
)
from scgsim.palace.reports.trust import PalaceTrustReport
from scgsim.palace.reports.views import (
    PhysicsQuantitiesReport,
    SimulationBenchmarkReport,
)
from scgsim.palace.results.epr import epr_result
from scgsim.palace.results.epr_records import PalaceEprResult
from scgsim.palace.results.resolve import read_resolved_run
from scgsim.palace.results.resolved_records import ResolvedRunData


def resolve_palace_result(
    run_dir: str | Path, *, expected_handoff_id: str
) -> ResolvedPalaceResult:
    """Validate and resolve a completed Palace manual-handoff run folder."""
    run = read_resolved_run(run_dir, expected_handoff_id=expected_handoff_id)
    return ResolvedPalaceResult(
        **{field.name: getattr(run, field.name) for field in fields(run)}
    )


@dataclass(frozen=True)
class ResolvedPalaceResult(ResolvedRunData):
    """One checked Palace run folder, with the EPR it yields and the views a notebook shows of it."""

    def epr_result(self) -> PalaceEprResult:
        """Detach the verified final Eigenmode EPR quantities for offline work."""

        return epr_result(self)

    def show_run_trustworthiness(
        self, *, theme: str = "light", show_details: bool = False
    ) -> PalaceTrustReport:
        """Build the identity and AMR view using the original artifact-validation order."""
        return show_run_trustworthiness(
            self,
            theme=cast(ReportTheme, theme),
            show_details=show_details,
        )

    def show_all_results(
        self,
        *,
        theme: str = "light",
        ranking_limit: int | None = 20,
        show_details: bool = False,
    ) -> None:
        """Display identity, benchmark and physics views in the order a notebook reads them."""
        show_all_results(
            self,
            theme=cast(ReportTheme, theme),
            ranking_limit=ranking_limit,
            show_details=show_details,
        )

    def show_physics_quantities(
        self, *, theme: str = "light", ranking_limit: int | None = 20
    ) -> PhysicsQuantitiesReport:
        """Build the selected physics view from this sealed run without recomputing solver data."""
        return show_physics_quantities(
            self, theme=cast(ReportTheme, theme), ranking_limit=ranking_limit
        )

    def show_simulation_benchmark(
        self,
        *,
        show_details: bool = False,
    ) -> SimulationBenchmarkReport:
        """Project recorded run cost and timing while retaining complete benchmark metadata."""
        return show_simulation_benchmark(self, show_details=show_details)
