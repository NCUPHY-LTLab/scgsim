"""Adapt authoritative native AMR version-1 sidecars into detached observations.

Manifest-listed attempts remain visible before any C table exists. Capacitance
belongs to an attempt only through its explicit written/iteration/order fields;
a stale parent CSV is never assigned to a started next attempt.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from scgsim.palace.results.amr_diagnostic_io import (
    NativeDiagnosticReader,
    NativeDiagnosticSource,
)
from scgsim.palace.results.amr_models import (
    AmrDiagnosticObservations,
    AmrSolveObservation,
)
from scgsim.palace.results.amr_refinement_diagnostics import read_refinement_observation
from scgsim.palace.results.amr_solve_diagnostics import read_solve_observation
from scgsim.palace.results.report_models import AmrPassSnapshot


def read_native_amr_diagnostics(
    source: NativeDiagnosticSource,
) -> AmrDiagnosticObservations:
    """Read only exact manifest references, with strict or explicitly unsealed proof."""
    reader = NativeDiagnosticReader(source)
    manifest = reader.read_manifest()
    integrity = "receipt_bound" if source.require_receipt else "observed_unsealed"
    if manifest is None:
        return AmrDiagnosticObservations(
            reason="native_diagnostics_manifest_unavailable",
            enabled=True,
            integrity=integrity,
            missing_files=("manifest.json",),
        )
    solves = tuple(
        solve
        for path in reader.referenced_paths(manifest, "solves")
        if (solve := read_solve_observation(reader, path)) is not None
    )
    refinements = tuple(
        transition
        for path in reader.referenced_paths(manifest, "transitions")
        if (transition := read_refinement_observation(reader, path)) is not None
    )
    if len({solve.native_iteration for solve in solves}) != len(solves):
        raise ValueError("native AMR manifest repeats a solve iteration identity")
    pairs = tuple((item.before_iteration, item.after_iteration) for item in refinements)
    if len(set(pairs)) != len(pairs):
        raise ValueError("native AMR manifest repeats a refinement transition identity")
    by_iteration = {solve.native_iteration: solve for solve in solves}
    for transition in refinements:
        before = by_iteration.get(transition.before_iteration)
        after = by_iteration.get(transition.after_iteration)
        if before is not None and before.mesh_id != transition.premesh_id:
            raise ValueError("native AMR transition pre-mesh disagrees with its solve")
        if after is not None and after.mesh_id != transition.next_mesh_id:
            raise ValueError("native AMR transition next-mesh disagrees with its solve")
    return AmrDiagnosticObservations(
        solves=tuple(sorted(solves, key=lambda item: item.native_iteration)),
        refinements=refinements,
        reason="native_diagnostic_records_incomplete"
        if reader.missing_records
        else None,
        enabled=True,
        source=(reader.base / "manifest.json")
        .relative_to(source.run_dir.resolve())
        .as_posix(),
        integrity=integrity,
        missing_files=tuple(dict.fromkeys(reader.missing_records)),
    )


def bind_native_attempts(
    snapshots: Sequence[AmrPassSnapshot],
    diagnostics: AmrDiagnosticObservations,
    run_dir: Path,
) -> tuple[AmrPassSnapshot, ...]:
    """Retain every emitted native attempt and strip C not written for that iteration."""
    if not diagnostics.enabled or not diagnostics.solves:
        return tuple(snapshots)
    bound = []
    for solve in diagnostics.solves:
        candidates = tuple(
            snapshot
            for snapshot in snapshots
            if snapshot.native_iteration == solve.native_iteration
        )
        snapshot = candidates[-1] if candidates else _attempt_snapshot(solve, run_dir)
        ambiguous = (
            len({candidate.capacitance_matrix_f for candidate in candidates}) > 1
        )
        attributable = (
            solve.capacitance_written is True
            and solve.capacitance_iteration == solve.native_iteration
        )
        matrix = (
            snapshot.capacitance_matrix_f if attributable and not ambiguous else None
        )
        bound.append(
            replace(
                snapshot,
                capacitance_matrix_f=matrix,
                native_iteration=solve.native_iteration,
                terminal_ids=solve.terminal_ids
                if solve.terminal_identity_reason is None and solve.terminal_ids
                else None,
                degrees_of_freedom=solve.true_dofs,
                mesh_elements=solve.owned_cells,
                error_norm=solve.indicator_norm,
            )
        )
    known_iterations = {solve.native_iteration for solve in diagnostics.solves}
    bound.extend(
        snapshot
        for snapshot in snapshots
        if snapshot.native_iteration not in known_iterations
    )
    return tuple(
        sorted(
            bound,
            key=lambda snapshot: (
                snapshot.native_iteration is None,
                snapshot.native_iteration or 0,
                snapshot.pass_index,
            ),
        )
    )


def _attempt_snapshot(solve: AmrSolveObservation, run_dir: Path) -> AmrPassSnapshot:
    """Represent an actual manifest attempt with unavailable, rather than invented, output."""
    return AmrPassSnapshot(
        pass_index=solve.native_iteration - 1,
        source=f"native-attempt:{solve.source}",
        path=run_dir / solve.source,
        frequencies_ghz=None,
        eig_columns=None,
        port_epr=None,
        capacitance_matrix_f=None,
        surface_epr=None,
        error_indicators=None,
        error_norm=solve.indicator_norm,
        degrees_of_freedom=solve.true_dofs,
        mesh_elements=solve.owned_cells,
        elapsed_total_s=None,
        peak_node_memory_mb=None,
        native_iteration=solve.native_iteration,
        terminal_ids=solve.terminal_ids if solve.terminal_ids else None,
    )
