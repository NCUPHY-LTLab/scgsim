"""The notebook views of a Palace Driven run: whether it converged, its S-parameters and Surface EPR, its cost.

``DrivenTrustReport`` answers, in the order a reader asks: which run this is (the problem, route, profile, AMR
passes and parallel resources), whether its numbers converged (the sweep, the FEM order, the mesh, a declared
convergence or "not declared", and the samples of the adaptive sweep for each port), and two facts of the
S-matrix it records without judging them: how far it is from reciprocal and from passive.
``DrivenPhysicsReport`` draws the S-parameters and the Surface EPR (``driven_figures``).  The benchmark is the
shared ``SimulationBenchmarkReport`` with this report's cards: the full solves, their time and memory.

``reports`` builds the trust report of a resolved run, with its ``driven_result()``, or of a returned run folder
read as it came back, complete or partial: that report has the run's identity, failure, sweep and cost, and
no S-parameters.
"""

from __future__ import annotations

import copy
import html
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np

from scgsim.contracts.palace_convergence import ConvergenceCheck
from scgsim.contracts.palace_run_folder import DRIVEN
from scgsim.palace.config.driven_sweep import DrivenSweep, FrequencyRange, Log, Points
from scgsim.palace.mesh.summary import MeshSummary
from scgsim.palace.reports.convergence import verdict_card
from scgsim.palace.reports.driven_figures import (
    s_parameter_figures,
    surface_epr_figures,
)
from scgsim.palace.reports.observations import CostCards
from scgsim.palace.reports.plotting import (
    ReportTheme,
    card_html,
    cards_html,
    checked_show_details,
    checked_theme,
    elapsed_time_figure,
    fmt,
    fmt_mb,
    fmt_seconds,
    show_figure,
    table_html,
)
from scgsim.palace.reports.trust import failure_notice_html, provenance_html
from scgsim.palace.reports.trust_records import ReportIdentity, ReportProvenance
from scgsim.palace.reports.views import DrivenBenchmarkData, SimulationBenchmarkReport
from scgsim.palace.results.driven_records import PalaceDrivenResult
from scgsim.palace.results.inspection import failure_payload
from scgsim.palace.results.inspection_records import PalaceFailureDiagnosis
from scgsim.presentation.notebook import Card
from scgsim.semantics.s_parameters import SParameters


@dataclass(frozen=True)
class DrivenTrustReport:
    """Which Driven run this is, whether its numbers converged, and how reciprocal and passive its S-matrix is.

    ``result`` is the run's ``driven_result()``, None for a returned run folder read as it came back, or when
    ``driven_result()`` refused the run: ``readback_refusal`` then says why, an adaptive sweep that reached its
    maximum number of samples for a port, say.  ``sweep`` is the one the sealed config requests, None when the
    folder has no config.  ``full_solves`` is Palace's count of linear solves (``LinearSolver.TotalSolves``):
    one per excited port and frequency the field was solved at.
    """

    run_dir: Path
    route: str
    profile: str
    completeness: Literal["complete", "partial"]
    failure: PalaceFailureDiagnosis | None
    identity: ReportIdentity
    passes_recorded: int
    sweep: DrivenSweep | None
    finite_element_order: int | None
    full_solves: int | None
    cost: CostCards
    durations: dict[str, float]
    provenance: ReportProvenance
    convergence: ConvergenceCheck
    result: PalaceDrivenResult | None
    readback_refusal: str | None = None
    theme: ReportTheme = "light"
    show_details: bool = False

    @property
    def problem(self) -> str:
        """The run's problem, ``Driven``."""
        return DRIVEN.name

    @property
    def reciprocity_error(self) -> float | None:
        """The largest |S_ij - S_ji| of the result (``reciprocity_error``); None without a result or with one port."""
        return None if self.result is None else reciprocity_error(self.result)

    @property
    def largest_singular_value(self) -> float | None:
        """The largest singular value of the result's S (``largest_singular_value``); None without a result."""
        return None if self.result is None else largest_singular_value(self.result)

    def mesh_summary(self) -> MeshSummary:
        """The run's mesh summary, as the resolved result gives it."""
        return copy.deepcopy(self.provenance["mesh_summary"])

    def with_theme(self, theme: ReportTheme) -> DrivenTrustReport:
        """This report with validated display colors and unchanged source data."""
        checked = checked_theme(theme)
        return self if self.theme == checked else replace(self, theme=checked)

    def show_run_trustworthiness(
        self, *, theme: ReportTheme = "light", show_details: bool = False
    ) -> DrivenTrustReport:
        """This report with the selected theme, and the run's full record when ``show_details``."""
        return replace(
            self.with_theme(theme), show_details=checked_show_details(show_details)
        )

    def show_physics_quantities(
        self, *, theme: ReportTheme = "light"
    ) -> DrivenPhysicsReport:
        """The S-parameters and Surface EPR of the run against frequency."""
        return DrivenPhysicsReport(self.with_theme(theme))

    def show_simulation_benchmark(
        self, *, theme: ReportTheme = "light", show_details: bool = False
    ) -> SimulationBenchmarkReport:
        """The full solves, their time and memory, the mesh and Palace's timers, with every machine field."""
        return SimulationBenchmarkReport(
            trust=self.with_theme(theme),
            data=self.benchmark_data(),
            show_details=checked_show_details(show_details),
        )

    def show_all_results(
        self, *, theme: ReportTheme = "light", show_details: bool = False
    ) -> None:
        """Display the trust, physics and benchmark views, in the order a reader asks about a run."""
        from IPython.display import display  # noqa: PLC0415 -- IPython is optional; detached result reading needs no notebook runtime.

        trust = self.show_run_trustworthiness(theme=theme, show_details=show_details)
        display(trust)
        display(trust.show_physics_quantities(theme=theme))
        display(trust.show_simulation_benchmark(theme=theme, show_details=show_details))

    def _ipython_display_(self) -> None:
        from IPython.display import HTML, display  # noqa: PLC0415 -- IPython is optional; detached result reading needs no notebook runtime.

        display(HTML(self.identity_html()))
        display(HTML(self.convergence_html()))
        network = self.reciprocity_and_passivity_html()
        if network:
            display(HTML(network))
        if self.show_details:
            display(HTML(provenance_html(self.provenance)))

    def identity_html(self) -> str:
        """The run's identity cards, then the notice of a failed run."""
        cards: list[Card] = [
            ("Problem", self.problem),
            ("Route", self.route),
            ("Profile", self.profile),
            ("AMR passes recorded", str(self.passes_recorded)),
            ("MPI × OpenMP", self.identity.get("resources")),
        ]
        if self.show_details:
            cards += [
                ("Completeness", self.completeness),
                ("Receipt", self.identity.get("receipt")),
                ("Palace git", self.identity.get("git_tag")),
                ("Handoff id", self.identity.get("handoff_id_short")),
            ]
        elif self.completeness != "complete":
            # A run that is not complete changes how its numbers are read, so it is said up front.
            cards.append(("Completeness", self.completeness))
        failure = "" if self.failure is None else failure_notice_html(self.failure)
        return f"<section><h3>Run Identity</h3>{cards_html(cards)}{failure}</section>"

    def convergence_html(self) -> str:
        """The sweep, the FEM order, the mesh and the declared convergence, then the adaptive sweep's samples."""
        cards: list[Card] = [
            ("Sweep", _sweep_text(self.sweep)),
            ("Sampling", _sampling_text(self.sweep)),
            ("FEM order", fmt(self.finite_element_order)),
            ("Mesh elements", fmt(self.cost.get("mesh_elements"))),
            ("Convergence", verdict_card(self.convergence)),
        ]
        return (
            f"<section><h3>Convergence</h3>{cards_html(cards)}"
            f"{self._refusal_html()}{self._sampling_table_html()}</section>"
        )

    def _refusal_html(self) -> str:
        """Why ``driven_result()`` refused the run, when it did: the run's numbers are not read."""
        if self.readback_refusal is None:
            return ""
        return (
            "<div style='border-left:4px solid #bf8700;background:rgba(191,135,0,0.10);"
            "padding:0.65rem 0.8rem;margin:0.7rem 0'><strong>Not read</strong><br>"
            f"{html.escape(self.readback_refusal)}</div>"
        )

    def _sampling_table_html(self) -> str:
        """One row per port of an adaptive sweep: its full solves, the model's error and where it solved."""
        if self.result is None or not self.result.adaptive_sampling:
            return ""
        names = {
            number: port.name for number, port in enumerate(self.result.ports, start=1)
        }
        tolerance = self.sweep.adaptive_tolerance if self.sweep is not None else None
        error = (
            "Estimated error"
            if tolerance is None
            else f"Estimated error ≤ {tolerance:g}"
        )
        rows = [
            [
                f"{item.excited_port}: {names[item.excited_port]}",
                str(item.samples),
                f"{item.error:.3g}",
                ", ".join(
                    f"{frequency / 1e9:.4g}"
                    for frequency in item.sampled_frequencies_hz
                ),
            ]
            for item in self.result.adaptive_sampling
        ]
        return table_html(("Port", "Samples", error, "Sampled at (GHz)"), rows)

    def reciprocity_and_passivity_html(self) -> str:
        """The S-matrix's distance from reciprocal and from passive, recorded and not judged."""
        passivity = self.largest_singular_value
        if passivity is None:
            return ""
        reciprocity = self.reciprocity_error
        cards: list[Card] = [("Largest singular value", f"{passivity:.6g}")]
        if reciprocity is not None:
            cards.insert(0, ("Reciprocity error", f"{reciprocity:.2g}"))
        heading = "Passivity" if reciprocity is None else "Reciprocity and passivity"
        return f"<section><h3>{heading}</h3>{cards_html(cards)}</section>"

    def card_html(self, label: str, value: Any) -> str:
        """Escape and render one card, as the benchmark's mesh summary draws them."""
        return card_html(label, value)

    def benchmark_cards_html(self) -> str:
        """The full solves, the wall time in all and per solve, the peak memory and the solver's size."""
        total = self.durations.get("Total")
        per_solve = (
            total / self.full_solves if total is not None and self.full_solves else None
        )
        cards: list[Card] = [
            ("Full solves", fmt(self.full_solves)),
            ("Total wall", fmt_seconds(total)),
            ("Wall time per solve", fmt_seconds(per_solve)),
            ("Peak node memory", fmt_mb(self.cost.get("peak_node_memory_megabytes"))),
            ("Peak rank memory", fmt_mb(self.cost.get("peak_memory_megabytes"))),
            ("DOFs", fmt(self.cost.get("problem_degrees_of_freedom"))),
            ("Mesh elements", fmt(self.cost.get("mesh_elements"))),
            ("MPI × OpenMP", self.identity.get("resources")),
        ]
        return f"<section><h3>Simulation Benchmark</h3>{cards_html(cards)}</section>"

    def benchmark_time_figure(self) -> Any:
        """Palace's elapsed time by timer."""
        return elapsed_time_figure(self.durations, self.theme)

    def benchmark_pass_table_html(self) -> str:
        """Nothing: the readback reads a Driven run's final solve, no AMR pass of its own."""
        return ""

    def benchmark_pass_figures(self) -> list[Any]:
        """Nothing, as ``benchmark_pass_table_html``."""
        return []

    def benchmark_data(self) -> DrivenBenchmarkData:
        """The benchmark's machine data: full solves, cost, timers, mesh, resources and the run's state."""
        return {
            "full_solves": self.full_solves,
            "cost": CostCards(**self.cost),
            "durations": dict(self.durations),
            "mesh_summary": self.mesh_summary(),
            "resources": {
                "requested_resources": self.provenance.get("requested_resources", {}),
                "resolved_resources": self.provenance.get("resolved_resources", {}),
            },
            "run_state": {
                "route": self.route,
                "problem": self.problem,
                "status": self.identity.get("receipt"),
                "completeness": self.completeness,
                "failure": failure_payload(self.failure),
            },
        }


@dataclass(frozen=True)
class DrivenPhysicsReport:
    """The S-parameters of a Driven run against frequency, with the plan's port names, and its Surface EPR."""

    trust: DrivenTrustReport

    def _ipython_display_(self) -> None:
        from IPython.display import HTML, display  # noqa: PLC0415 -- IPython is optional; detached result reading needs no notebook runtime.

        result = self.trust.result
        if result is None:
            display(HTML(self._unread_html()))
            return
        display(
            HTML(f"<section><h3>Physics Quantities</h3>{ports_html(result)}</section>")
        )
        for figure in s_parameter_figures(result, self.trust.theme):
            show_figure(figure)
        surface = surface_epr_figures(result, self.trust.theme)
        if surface:
            display(HTML("<section><h3>Surface EPR</h3></section>"))
            for figure in surface:
                show_figure(figure)

    def _unread_html(self) -> str:
        """Why the view has no S-parameters to draw."""
        reason = (
            self.trust.readback_refusal
            or "A run folder read as it came back has none; resolve_palace_result() reads them."
        )
        return (
            "<section><h3>Physics Quantities</h3>"
            f"<p style='opacity:0.75'>No S-parameters were read. {html.escape(reason)}</p></section>"
        )


def ports_html(result: SParameters) -> str:
    """The S-matrix's ports in order, each with its name in the plan and its impedance."""
    return table_html(
        ("Port", "Name", "Impedance (Ω)"),
        [
            [str(number), port.name, f"{port.impedance_ohm:g}"]
            for number, port in enumerate(result.ports, start=1)
        ],
    )


def reciprocity_error(result: SParameters) -> float | None:
    """The largest |S_ij - S_ji| over every pair of ports and every frequency; None for one port.

    A network without magnetized or active parts is reciprocal, S_ij = S_ji, so this is how far the solved
    S-matrix is from it: what the discretization and the solver's tolerances leave.  It is recorded, not
    judged against a threshold.
    """
    count = len(result.ports)
    if count < 2:
        return None
    return max(
        abs(forward - backward)
        for output in range(1, count + 1)
        for excited in range(output + 1, count + 1)
        for forward, backward in zip(
            result.s(output, excited), result.s(excited, output), strict=True
        )
    )


def largest_singular_value(result: SParameters) -> float:
    """The largest singular value of S over the frequencies: at most 1 for a passive network.

    It is the largest ratio of outgoing to incoming wave amplitude over every way of driving the ports, so
    above 1 the solved network would return more power than it is fed.  It is recorded, not judged.
    """
    count = len(result.ports)
    # matrices[k] is S at frequency k, rows the output port and columns the excited one.
    matrices = np.array(
        [
            [result.s(output, excited) for excited in range(1, count + 1)]
            for output in range(1, count + 1)
        ]
    ).transpose(2, 0, 1)
    return float(np.linalg.norm(matrices, ord=2, axis=(1, 2)).max())


def _sweep_text(sweep: DrivenSweep | None) -> str | None:
    """``2–10 GHz, 81 points``: each range, then how many frequencies they merge into when that is not plain.

    Two ranges read ``4–9 GHz, 501 points; 7.3–7.34 GHz, 4,001 points; 4,497 frequencies``.
    """
    if sweep is None:
        return None
    texts = [_range_text(item) for item in sweep.ranges]
    # One range says how many frequencies it has, unless Palace merged some of them.
    if len(sweep.ranges) > 1 or _range_count(sweep.ranges[0]) != sweep.frequency_count:
        count = sweep.frequency_count
        texts.append(f"{count:,} frequency" if count == 1 else f"{count:,} frequencies")
    return "; ".join(texts)


# A Points range is listed in full up to this many frequencies, and as its count and span beyond.
_LISTED_POINTS = 3


def _range_text(item: FrequencyRange) -> str:
    """One range of a sweep: ``4–9 GHz, 501 points``, ``0.1–10 GHz, 201 points, log-spaced`` or ``7.3, 7.32 GHz``.

    A range of full solves ends in ``, full solve``: ``7.35665 GHz, full solve``.
    """
    return _range_frequencies_text(item) + (", full solve" if item.full_solve else "")


def _range_frequencies_text(item: FrequencyRange) -> str:
    """The frequencies of one range: ``4–9 GHz, 501 points``, ``..., log-spaced`` or ``7.3, 7.32 GHz``."""
    if isinstance(item, Points):
        values = [f / 1e9 for f in item.frequencies_hz]
        if len(values) <= _LISTED_POINTS:
            return ", ".join(f"{value:g}" for value in values) + " GHz"
        return f"{len(values):,} points, {min(values):g}–{max(values):g} GHz"
    text = f"{item.start_hz / 1e9:g}–{item.stop_hz / 1e9:g} GHz, {item.points:,} points"
    return text + ", log-spaced" if isinstance(item, Log) else text


def _range_count(item: FrequencyRange) -> int:
    """How many frequencies one range lists, before Palace merges them."""
    return len(item.frequencies_hz) if isinstance(item, Points) else item.points


def _sampling_text(sweep: DrivenSweep | None) -> str | None:
    """``uniform`` (every frequency solved) or ``adaptive, tolerance 0.001`` with its cap when one is set."""
    if sweep is None:
        return None
    if sweep.adaptive_tolerance is None:
        return "uniform"
    cap = (
        ""
        if sweep.adaptive_max_samples is None
        else f", at most {sweep.adaptive_max_samples} samples"
    )
    return f"adaptive, tolerance {sweep.adaptive_tolerance:g}{cap}"
