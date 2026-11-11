"""Analyze previously resolved or inspected Palace runs without importing views.

This entry accepts source objects, never a run-directory string in place of
strict result resolution. Historical outputs retain their raw matrices and
native ZZ norms; missing per-terminal solver validity remains unavailable.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from scgsim.palace.results.amr_computation import (
    AmrAnalysisContext,
    analyze_snapshot_series,
)
from scgsim.palace.results.amr_diagnostic_io import NativeDiagnosticSource
from scgsim.palace.results.amr_diagnostics import (
    bind_native_attempts,
    read_native_amr_diagnostics,
)
from scgsim.palace.results.amr_models import (
    AmrDiagnosticObservations,
    CapacitanceReference,
    PalaceAmrAnalysis,
)
from scgsim.palace.results.models import ResolvedRunData
from scgsim.palace.results.observations import read_optional_json
from scgsim.palace.results.report_models import (
    AmrPassSnapshot,
    PalaceFailureDiagnosis,
    PalaceResultSelection,
)
from scgsim.palace.results.snapshots import collect_amr_passes
from scgsim.palace.results.surface_readers import (
    read_surface_bindings,
    read_surface_margin_bindings,
)
from scgsim.palace.results.tables import configured_mode_count


@runtime_checkable
class PartialAmrReport(Protocol):
    """The detached inspection fields consumed from a partial or completed report."""

    @property
    def run_dir(self) -> Path:
        """Locate the already inspected run's configuration and structured index."""
        ...

    @property
    def problem(self) -> str:
        """Identify the native problem without deriving it from filenames."""
        ...

    @property
    def passes(self) -> tuple[AmrPassSnapshot, ...]:
        """Expose the inspection owner's readable native snapshots in source order."""
        ...

    @property
    def selection(self) -> PalaceResultSelection:
        """Retain the inspector's source-selection and integrity observations."""
        ...

    @property
    def failure(self) -> PalaceFailureDiagnosis | None:
        """Keep a failed native attempt independent of readable earlier snapshots."""
        ...

    @property
    def provenance(self) -> Mapping[str, Any]:
        """Expose any returned receipt already captured by the inspection owner."""
        ...

    @property
    def completeness(self) -> str:
        """Retain complete or partial run state without implying convergence."""
        ...


@dataclass(frozen=True)
class _AnalysisSource:
    """Already inspected pass records plus their authoritative conductor-net index."""

    context: AmrAnalysisContext
    passes: tuple[AmrPassSnapshot, ...]
    index_map: Mapping[str, Any] | None
    config: Mapping[str, Any] | None
    receipt: Mapping[str, Any] | None


def analyze_amr(
    result: ResolvedRunData | PartialAmrReport,
    reference: CapacitanceReference | None = None,
) -> PalaceAmrAnalysis:
    """Detach AMR observations without synthesizing references or solver validity.

    ``reference.terminal_ids`` uses the exact conductor ``net_id`` values from
    indexed ``Boundaries.Terminal`` rows. Display labels and matrix positions
    alone never establish the identity of a reference terminal.
    """
    source = _analysis_source(result)
    terminal_ids = indexed_terminal_ids(source.index_map)
    snapshots = tuple(
        replace(snapshot, terminal_ids=terminal_ids)
        if snapshot.terminal_ids is None and terminal_ids is not None
        else snapshot
        for snapshot in source.passes
    )
    diagnostics = _native_diagnostics(source)
    snapshots = bind_native_attempts(snapshots, diagnostics, source.context.run_dir)
    return analyze_snapshot_series(snapshots, source.context, diagnostics, reference)


def indexed_terminal_ids(index_map: Mapping[str, Any] | None) -> tuple[str, ...] | None:
    """Read exact conductor net IDs in native terminal-index order, or leave unknown."""
    if index_map is None or not isinstance(index_map.get("entries"), list):
        return None
    entries = index_map["entries"]
    terminals = tuple(
        entry
        for entry in entries
        if isinstance(entry, Mapping) and entry.get("section") == "Boundaries.Terminal"
    )
    if not terminals:
        return None
    if any(type(entry.get("index")) is not int for entry in terminals):
        return None
    ordered = sorted(terminals, key=lambda entry: entry["index"])
    if [entry["index"] for entry in ordered] != list(range(1, len(ordered) + 1)):
        return None
    ids = tuple(entry.get("net_id") for entry in ordered)
    if any(not isinstance(name, str) or not name for name in ids) or len(
        set(ids)
    ) != len(ids):
        return None
    return tuple(str(name) for name in ids)


def _analysis_source(result: ResolvedRunData | PartialAmrReport) -> _AnalysisSource:
    """Preserve strict result identity or the existing inspector's partial integrity."""
    if isinstance(result, ResolvedRunData):
        if (
            not result.has_returned_outputs
            or not result.returned_receipt.identity_verified
        ):
            raise ValueError(
                "AMR analysis requires an identity-verified resolved result"
            )
        index_map = result.provenance.index_map
        collected = collect_amr_passes(
            result.run_dir,
            result.problem,
            read_surface_bindings(index_map),
            read_surface_margin_bindings(index_map),
            failed_attempt=result.status != "completed",
            mode_count=configured_mode_count(result.provenance.config),
        )
        context = AmrAnalysisContext(
            result.run_dir, result.problem, result.status, "receipt_bound"
        )
        return _AnalysisSource(
            context,
            collected.passes,
            index_map,
            result.provenance.config,
            result.provenance.returned_receipt,
        )
    if not isinstance(result, PartialAmrReport):
        raise TypeError(
            "AMR analysis requires a resolved result or an inspected report"
        )
    receipt = result.provenance.get("returned_receipt")
    status = receipt.get("status") if isinstance(receipt, Mapping) else None
    if not isinstance(status, str):
        status = "failed" if result.failure is not None else result.completeness
    context = AmrAnalysisContext(
        result.run_dir, result.problem, status, result.selection.integrity
    )
    index_map = read_optional_json(result.run_dir / "metadata/palace_index_map.json")
    config = read_optional_json(result.run_dir / "config.json")
    return _AnalysisSource(
        context,
        result.passes,
        index_map,
        config,
        receipt if isinstance(receipt, Mapping) else None,
    )


def _native_diagnostics(source: _AnalysisSource) -> AmrDiagnosticObservations:
    """Read opt-in sidecars under the configured output scope, never by guessed records."""
    if source.config is None:
        return AmrDiagnosticObservations()
    enabled = (
        source.config.get("Model", {})
        .get("Refinement", {})
        .get("SaveAdaptDiagnostics", False)
    )
    if not isinstance(enabled, bool):
        raise TypeError("native SaveAdaptDiagnostics must be a boolean")
    if not enabled:
        return AmrDiagnosticObservations(reason="native_diagnostics_not_requested")
    output = source.config.get("Problem", {}).get("Output")
    if not isinstance(output, str) or not output:
        raise ValueError("native AMR analysis requires the configured Problem.Output")
    terminal_ids = {}
    if source.index_map is not None:
        for entry in source.index_map.get("entries", ()):
            if (
                isinstance(entry, Mapping)
                and entry.get("section") == "Boundaries.Terminal"
            ):
                index, net_id = entry.get("index"), entry.get("net_id")
                if type(index) is int and isinstance(net_id, str) and net_id:
                    if index in terminal_ids or net_id in terminal_ids.values():
                        raise ValueError(
                            "native terminal-to-net identity is duplicated"
                        )
                    terminal_ids[index] = net_id
    return read_native_amr_diagnostics(
        NativeDiagnosticSource(
            run_dir=source.context.run_dir,
            output_dir=source.context.run_dir / output,
            terminal_net_ids=terminal_ids,
            receipt=source.receipt,
            require_receipt=source.context.integrity == "receipt_bound",
        )
    )
