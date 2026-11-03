"""Read exact owned marked/refined sets and native pre/post refinement identities.

Rank-local coarse IDs are scoped by pre-mesh and rank. Fine-to-coarse mappings
establish actual parent splitting before rebalancing, independently of marking.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from scgsim.palace.results.amr_diagnostic_io import (
    NativeDiagnosticReader,
    native_float,
    native_indices,
    native_int,
    native_text,
)
from scgsim.palace.results.amr_models import AmrCellIdentity, AmrRefinementObservation


@dataclass(frozen=True)
class RankRefinement:
    """One rank's complete owned-cell refinement map in its original coarse scope."""

    rank: int
    premesh_id: str
    cells_before: int
    cells_after: int
    marked: frozenset[AmrCellIdentity]
    refined: frozenset[AmrCellIdentity]


def read_refinement_observation(
    reader: NativeDiagnosticReader, relative: str
) -> AmrRefinementObservation | None:
    """Read one inventoried native transition without assuming marked-set containment."""
    record = reader.read_record(relative, "transition")
    if record is None:
        return None
    before = native_int(record, "pre_iteration", minimum=1)
    after = native_int(record, "next_iteration", minimum=1)
    if after != before + 1:
        raise ValueError(
            "native AMR transition does not identify consecutive iterations"
        )
    state = native_text(record, "state")
    if state not in {"started", "refined", "completed"}:
        raise ValueError("invalid native AMR transition state")
    premesh_id = native_text(record, "pre_mesh_id")
    refined_mesh_id = native_text(record, "refined_mesh_id")
    next_mesh_id = native_text(record, "next_mesh_id")
    before_cells = native_int(record, "owned_global_cells_pre")
    if state == "started":
        marked = refined = None
        counts_validation = "transition_started"
        after_refinement = after_rebalance = None
    else:
        reader.referenced_paths(record, "refined_mesh_paths")
        ranks = tuple(
            _read_rank(reader, path, premesh_id, refined_mesh_id)
            for path in reader.referenced_paths(record, "rank_records")
        )
        complete_ranks = tuple(rank for rank in ranks if rank is not None)
        after_refinement = native_int(record, "owned_global_cells_post_refinement")
        after_rebalance = (
            native_int(record, "owned_global_cells_post_rebalance")
            if state == "completed"
            else None
        )
        if len(complete_ranks) != len(ranks):
            marked = refined = None
            counts_validation = "rank_records_incomplete"
        else:
            marked, refined = _combine_rank_sets(complete_ranks, reader.mpi_size)
            _validate_global_counts(record, complete_ranks, marked, refined)
            counts_validation = "verified_exact_rank_sets"
    conformity = native_text(record, "mesh_conformity")
    if conformity not in {"C", "NC"}:
        raise ValueError("unsupported native AMR mesh conformity")
    return AmrRefinementObservation(
        before_iteration=before,
        after_iteration=after,
        premesh_id=premesh_id,
        cells_before=before_cells,
        cells_after=after_rebalance
        if after_rebalance is not None
        else after_refinement,
        marked_parents=marked,
        refined_parents=refined,
        marked_squared_indicator_sum=native_float(record, "eta_squared_marked"),
        total_squared_indicator_sum=native_float(record, "eta_squared_total"),
        source=reader.path(relative)
        .relative_to(reader.source.run_dir.resolve())
        .as_posix(),
        state=state,
        refined_mesh_id=refined_mesh_id,
        next_mesh_id=next_mesh_id,
        cells_after_refinement=after_refinement,
        cells_after_rebalance=after_rebalance,
        theta_requested=native_float(record, "theta_requested"),
        threshold=native_float(record, "threshold"),
        mesh_conformity=conformity,
        max_nc_levels=native_int(record, "max_nc_levels"),
        declared_marked_count=None
        if state == "started"
        else native_int(record, "marked_count"),
        declared_refined_parent_count=None
        if state == "started"
        else native_int(record, "refined_parent_count"),
        declared_extra_refined_parent_count=None
        if state == "started"
        else native_int(record, "extra_refined_parent_count"),
        counts_validation=counts_validation,
    )


def _read_rank(
    reader: NativeDiagnosticReader, relative: str, premesh_id: str, refined_mesh_id: str
) -> RankRefinement | None:
    """Validate an owned fine-to-coarse map and preserve mesh/rank identity."""
    record = reader.read_record(relative, "rank")
    if record is None:
        return None
    if (
        record.get("pre_mesh_id") != premesh_id
        or record.get("refined_mesh_id") != refined_mesh_id
    ):
        raise ValueError("native AMR rank record mesh identity mismatch")
    rank = native_int(record, "rank")
    before = native_int(record, "owned_cells_pre")
    after = native_int(record, "owned_cells_post_refinement")
    marked = frozenset(native_indices(record, "marked"))
    refined = frozenset(native_indices(record, "refined_parents"))
    if any(index >= before for index in marked | refined):
        raise ValueError("native AMR set contains a non-owned coarse-cell index")
    _validate_fine_map(record, before, after, refined)
    if (
        native_int(record, "marked_count") != len(marked)
        or native_int(record, "refined_parent_count") != len(refined)
        or native_int(record, "extra_refined_parent_count") != len(refined - marked)
    ):
        raise ValueError("native AMR rank counts disagree with the exact parent sets")
    return RankRefinement(
        rank=rank,
        premesh_id=premesh_id,
        cells_before=before,
        cells_after=after,
        marked=frozenset(AmrCellIdentity(premesh_id, rank, index) for index in marked),
        refined=frozenset(
            AmrCellIdentity(premesh_id, rank, index) for index in refined
        ),
    )


def _validate_fine_map(
    record: dict[str, Any], before: int, after: int, refined: frozenset[int]
) -> None:
    """Require each owned fine leaf once and recover distinct actual split parents."""
    mapping = record.get("fine_to_coarse")
    if not isinstance(mapping, list) or any(
        not isinstance(item, dict) for item in mapping
    ):
        raise TypeError("native AMR fine_to_coarse must contain mapping records")
    fine = tuple(native_int(item, "fine") for item in mapping)
    coarse = tuple(native_int(item, "coarse") for item in mapping)
    if (
        len(fine) != after
        or set(fine) != set(range(after))
        or len(set(fine)) != len(fine)
    ):
        raise ValueError("native AMR map must cover owned fine cells exactly once")
    if any(index >= before for index in coarse):
        raise ValueError("native AMR map references a non-owned coarse parent")
    if set(coarse) != set(range(before)):
        raise ValueError("native AMR map must retain every owned coarse parent")
    actual = frozenset(
        parent for parent, leaves in Counter(coarse).items() if leaves > 1
    )
    if actual != refined:
        raise ValueError("native AMR refined parents disagree with the owned fine map")


def _combine_rank_sets(
    ranks: tuple[RankRefinement, ...], mpi_size: int | None
) -> tuple[frozenset[AmrCellIdentity], frozenset[AmrCellIdentity]]:
    """Combine owned sets only when every manifest rank is represented exactly once."""
    identifiers = tuple(rank.rank for rank in ranks)
    if (
        mpi_size is None
        or len(set(identifiers)) != len(identifiers)
        or set(identifiers) != set(range(mpi_size))
    ):
        raise ValueError(
            "native AMR transition rank coverage is incomplete or duplicated"
        )
    return (
        frozenset(cell for rank in ranks for cell in rank.marked),
        frozenset(cell for rank in ranks for cell in rank.refined),
    )


def _validate_global_counts(
    record: dict[str, Any],
    ranks: tuple[RankRefinement, ...],
    marked: frozenset[AmrCellIdentity],
    refined: frozenset[AmrCellIdentity],
) -> None:
    """Check the producer's integer aggregates against its complete owned rank records."""
    expected = {
        "marked_count": len(marked),
        "refined_parent_count": len(refined),
        "extra_refined_parent_count": len(refined - marked),
        "owned_global_cells_pre": sum(rank.cells_before for rank in ranks),
        "owned_global_cells_post_refinement": sum(rank.cells_after for rank in ranks),
    }
    if any(native_int(record, key) != count for key, count in expected.items()):
        raise ValueError(
            "native AMR global counts disagree with the owned rank records"
        )
