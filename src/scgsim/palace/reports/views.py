"""Benchmark and physical-quantity display projections of a trust report."""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from typing import Any

from scgsim.palace.reports.plotting import (
    mesh_summary_html,
    show_figure,
)
from scgsim.palace.reports.surface_figures import (
    SURFACE_TYPES,
    latest_margin_snapshot,
    series_label,
    surface_loss_html,
    surface_margin_html,
    surface_margin_participation_figure,
    surface_ranking_figure,
)
from scgsim.palace.results.observations import read_optional_json
from scgsim.palace.results.report_models import (
    SurfaceEprSeriesSnapshot,
    SurfaceMarginEprSeriesSnapshot,
)
from scgsim.palace.results.snapshots import (
    surface_margin_snapshots,
    surface_snapshots,
)
from scgsim.palace.results.surface_readers import read_surface_margin_bindings


@dataclass(frozen=True)
class SimulationBenchmarkReport:
    """Human-readable simulation cost with complete machine data in ``data``."""

    trust: Any
    data: dict[str, Any]
    show_details: bool = False

    def _ipython_display_(self) -> None:
        from IPython.display import HTML, display  # noqa: PLC0415 -- IPython is optional; detached result reading needs no notebook runtime.

        display(HTML(self.trust.benchmark_cards_html()))
        display(
            HTML(
                mesh_summary_html(
                    self.data["mesh_summary"], card_html=self.trust.card_html
                )
            )
        )
        timing = self.trust.benchmark_time_figure()
        if isinstance(timing, str):
            display(HTML(timing))
        else:
            show_figure(timing)
        display(HTML(self.trust.benchmark_pass_table_html()))
        for figure in self.trust.benchmark_pass_figures():
            show_figure(figure)
        if self.show_details:
            payload = html.escape(json.dumps(self.data, indent=2, sort_keys=True))
            display(
                HTML(
                    "<section><h4>Benchmark metadata</h4>"
                    "<p style='opacity:0.75;margin-top:0'>Complete machine-readable "
                    "cost, timing, resource, and run-state fields.</p>"
                    f"<pre style='white-space:pre-wrap'>{payload}</pre></section>"
                )
            )


@dataclass(frozen=True)
class PhysicsQuantitiesReport:
    """Latest readable physical quantities built from structured solver data."""

    trust: Any
    ranking_limit: int | None = 20

    @property
    def snapshots(self) -> tuple[SurfaceEprSeriesSnapshot, ...]:
        """Expose detached whole-surface series in their native pass and mode order."""
        return surface_snapshots(self.trust.passes)

    @property
    def margin_snapshots(self) -> tuple[SurfaceMarginEprSeriesSnapshot, ...]:
        """Expose margin-row series separately from the whole-surface rows."""
        return surface_margin_snapshots(self.trust.passes)

    @property
    def margins_configured(self) -> bool:
        """Check the run's index map for margin rows."""
        index_map = read_optional_json(
            self.trust.run_dir / "metadata" / "palace_index_map.json"
        )
        return bool(read_surface_margin_bindings(index_map))

    def _ipython_display_(self) -> None:
        from IPython.display import HTML, display  # noqa: PLC0415 -- IPython is optional; detached result reading needs no notebook runtime.

        display(HTML(self._heading_html()))
        snapshots = self.snapshots
        if not snapshots:
            display(
                HTML(
                    "<p style='opacity:0.75'>Surface-EPR physics is unavailable: "
                    "no readable surface-Q snapshot could be bound to structured "
                    "index-map semantics.</p>"
                )
            )
            return
        for series_index in sorted({item.series_index for item in snapshots}):
            series = tuple(
                item for item in snapshots if item.series_index == series_index
            )
            latest = series[-1]
            label = series_label(latest)
            display(HTML(f"<h4>{html.escape(label)}</h4>"))
            ranking = surface_ranking_figure(
                latest, self.ranking_limit, self.trust.theme
            )
            if ranking is not None:
                show_figure(ranking)
            display(HTML(surface_loss_html(latest, self.trust.problem)))
            margin_latest = latest_margin_snapshot(
                self.margin_snapshots,
                pass_index=latest.pass_index,
                source=latest.source,
                series_index=latest.series_index,
            )
            if margin_latest is not None:
                for interface_type in SURFACE_TYPES:
                    figure = surface_margin_participation_figure(
                        margin_latest,
                        latest,
                        interface_type,
                        self.trust.theme,
                    )
                    if figure is not None:
                        show_figure(figure)
                display(HTML(surface_margin_html(margin_latest)))
            elif self.margins_configured:
                display(
                    HTML(
                        "<p style='opacity:0.75'>Margin-row Surface EPR is "
                        "unavailable for the selected source/pass; whole-surface "
                        "Surface-EPR remains separately readable.</p>"
                    )
                )

    def _heading_html(self) -> str:
        state = (
            "complete returned run"
            if self.trust.completeness == "complete"
            else "partial / convergence not established"
        )
        limit = (
            "all surfaces" if self.ranking_limit is None else str(self.ranking_limit)
        )
        latest_source = self.trust.selection.selected_source or "unavailable"
        return (
            "<section><h3>Physics Quantities</h3>"
            "<p style='opacity:0.75;margin-top:0'>Latest-readable individual-surface "
            "participation and loss interpretation are shown only after run identity "
            "and numerical evidence. Modes and excitations remain separate. "
            f"Run state: {html.escape(state)}; latest readable snapshot: "
            f"{html.escape(latest_source)}; ranking limit: "
            f"{html.escape(limit)}; snapshot integrity: "
            f"{html.escape(self.trust.selection.integrity)}. Complete bound records "
            "remain available through "
            "<code>snapshots</code>.</p></section>"
        )
