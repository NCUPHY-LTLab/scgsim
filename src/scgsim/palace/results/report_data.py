"""Collect one snapshot per Palace solver pass for the run report.

A run keeps its final tables in ``results/palace`` and, with adaptive mesh refinement
(AMR), one folder per pass (``iteration1``, ``iteration2`` ...). The trust report wants
all of them, including those of a run that failed, so this module is lenient where
``resolve.py`` is strict: it reads what exists and records what it could not read.

Here: finding the snapshots, reading their eigenfrequencies, capacitance matrices and
error indicators, and deciding whether two passes show the same physics. Not here: the
value records (``_report_types.py``), the Surface EPR readers
(``_surface_epr_reader.py``), and the receipt-based failure diagnosis
(``_inspection.py``).
"""

from __future__ import annotations

import math
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, NamedTuple

from ._artifact_files import read_json
from ._report_types import (
    AmrPassSnapshot,
    CollectedSnapshots,
    PalaceFailureDiagnosis,
    PalaceResultSelection,
    PassCostRecord,
    SurfaceEprRecord,
    SurfaceEprSeriesSnapshot,
    SurfaceMaskEprRecord,
    SurfaceMaskEprSeriesSnapshot,
)
from ._result_tables import read_csv_table
from ._surface_epr_reader import read_surface_epr, read_surface_mask_epr

_ITERATION_DIR = re.compile(r"^iteration(\d+)$")
FREQ_HEADER = "Re{f} (GHz)"
_CAP_HEADER_PREFIX = "C[i]["
_INDEX_COLUMNS = {"m", "i"}
SKIP_EIG_COLUMNS = {"Error (Bkwd.)", "Error (Abs.)"}
ERROR_INDICATOR_TRACES = ("Norm", "Maximum", "Mean")


def _parent_has_physics(results: Path, problem: str) -> bool:
    if problem == "Eigenmode":
        return (results / "eig.csv").is_file()
    return (results / "terminal-C.csv").is_file()


def _same_primary_physics(left: AmrPassSnapshot, right: AmrPassSnapshot) -> bool:
    if left.frequencies_ghz is not None and right.frequencies_ghz is not None:
        return left.frequencies_ghz == right.frequencies_ghz
    if left.capacitance_matrix_f is not None and right.capacitance_matrix_f is not None:
        return left.capacitance_matrix_f == right.capacitance_matrix_f
    return False


def _same_masked_physics(
    left: tuple[SurfaceMaskEprSeriesSnapshot, ...] | None,
    right: tuple[SurfaceMaskEprSeriesSnapshot, ...] | None,
) -> bool:
    if left is None or right is None:
        return left is right
    return tuple(
        (snapshot.series_index, snapshot.series_kind, snapshot.records)
        for snapshot in left
    ) == tuple(
        (snapshot.series_index, snapshot.series_kind, snapshot.records)
        for snapshot in right
    )


def _problem_size_conflicts(left: AmrPassSnapshot, right: AmrPassSnapshot) -> bool:
    return any(
        left_value is not None and right_value is not None and left_value != right_value
        for left_value, right_value in (
            (left.degrees_of_freedom, right.degrees_of_freedom),
            (left.mesh_elements, right.mesh_elements),
        )
    )


def mapping_max(payload: Any) -> float | None:
    if isinstance(payload, dict):
        value = payload.get("Max")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return None
    if isinstance(payload, (int, float)) and not isinstance(payload, bool):
        return float(payload)
    return None


def read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return read_json(path)


def as_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(float(value)):
        return None
    return int(value)


def declares_surface_masks(
    config: dict[str, Any] | None, index_map: dict[str, Any] | None
) -> bool:
    boundaries = config.get("Boundaries") if isinstance(config, dict) else None
    postprocessing = (
        boundaries.get("Postprocessing") if isinstance(boundaries, dict) else None
    )
    dielectric = (
        postprocessing.get("Dielectric") if isinstance(postprocessing, dict) else None
    )
    if isinstance(dielectric, list) and any(
        isinstance(row, dict) and "Mask" in row for row in dielectric
    ):
        return True
    entries = index_map.get("entries") if isinstance(index_map, dict) else None
    return isinstance(entries, list) and any(
        isinstance(entry, dict) and "mask" in entry for entry in entries
    )


def collect_amr_passes(
    root: Path,
    problem: str,
    surface_bindings: tuple[dict[str, Any], ...] | None,
    surface_mask_bindings: tuple[dict[str, Any], ...] | None,
    *,
    failed_attempt: bool,
) -> CollectedSnapshots:
    results = root / "results" / "palace"
    if not results.is_dir():
        return CollectedSnapshots((), "missing")
    iteration_dirs = []
    for child in results.iterdir():
        match = _ITERATION_DIR.match(child.name)
        if match and child.is_dir():
            iteration_dirs.append((int(match.group(1)), child))
    iteration_dirs.sort()
    snapshots: list[AmrPassSnapshot] = []
    for index, path in iteration_dirs:
        snapshot = _load_optional_snapshot(
            pass_index=index - 1,
            source=path.name,
            path=path,
            problem=problem,
            surface_bindings=surface_bindings,
            surface_mask_bindings=surface_mask_bindings,
        )
        if snapshot is not None:
            snapshots.append(snapshot)
    if not _parent_has_physics(results, problem):
        return CollectedSnapshots(tuple(snapshots), "missing")
    parent = _load_optional_snapshot(
        pass_index=max((index for index, _path in iteration_dirs), default=0),
        source="final",
        path=results,
        problem=problem,
        surface_bindings=surface_bindings,
        surface_mask_bindings=surface_mask_bindings,
    )
    if parent is None:
        return CollectedSnapshots(tuple(snapshots), "unreadable")
    if snapshots and _same_primary_physics(snapshots[-1], parent):
        if failed_attempt and _problem_size_conflicts(snapshots[-1], parent):
            return CollectedSnapshots(tuple(snapshots), "unreadable")
        parent = replace(parent, pass_index=snapshots[-1].pass_index)
        if _same_masked_physics(
            snapshots[-1].surface_mask_epr, parent.surface_mask_epr
        ):
            snapshots[-1] = replace(parent, pass_index=snapshots[-1].pass_index)
            return CollectedSnapshots(tuple(snapshots), "readable")
    return CollectedSnapshots((*snapshots, parent), "readable")


def _load_optional_snapshot(  # noqa: PLR0913 - keyword-only; mirrors _load_snapshot
    *,
    pass_index: int,
    source: str,
    path: Path,
    problem: str,
    surface_bindings: tuple[dict[str, Any], ...] | None,
    surface_mask_bindings: tuple[dict[str, Any], ...] | None,
) -> AmrPassSnapshot | None:
    try:
        return _load_snapshot(
            pass_index=pass_index,
            source=source,
            path=path,
            problem=problem,
            surface_bindings=surface_bindings,
            surface_mask_bindings=surface_mask_bindings,
        )
    except (OSError, TypeError, ValueError):
        return None


def _load_snapshot(  # noqa: PLR0913 - keyword-only; each argument is a separate fact about the pass
    *,
    pass_index: int,
    source: str,
    path: Path,
    problem: str,
    surface_bindings: tuple[dict[str, Any], ...] | None,
    surface_mask_bindings: tuple[dict[str, Any], ...] | None,
) -> AmrPassSnapshot:
    eig_columns = (
        _read_numeric_table_columns(path / "eig.csv")
        if problem == "Eigenmode"
        else None
    )
    port_epr = (
        _read_numeric_table_columns(path / "port-EPR.csv")
        if problem == "Eigenmode"
        else None
    )
    capacitance = (
        _read_capacitance(path / "terminal-C.csv")
        if problem == "Electrostatic"
        else None
    )
    error_indicators = _read_error_indicators(path / "error-indicators.csv")
    frequencies = None if eig_columns is None else eig_columns.get(FREQ_HEADER)
    expected_rows = (
        len(frequencies)
        if frequencies is not None
        else len(capacitance)
        if capacitance is not None
        else 0
    )
    surface_epr = read_surface_epr(
        path / "surface-Q.csv",
        problem=problem,
        pass_index=pass_index,
        source=source,
        frequencies_ghz=frequencies,
        expected_rows=expected_rows,
        bindings=surface_bindings,
        surface_count=len(surface_bindings or ()) + len(surface_mask_bindings or ()),
    )
    surface_mask_epr = read_surface_mask_epr(
        path / "surface-mask-Q.csv",
        path / "surface-mask-energy.csv",
        problem=problem,
        pass_index=pass_index,
        source=source,
        expected_rows=expected_rows,
        bindings=surface_mask_bindings,
    )
    error_norm = None if error_indicators is None else error_indicators.get("Norm")
    palace_cost = _read_palace_cost(path / "palace.json")
    return AmrPassSnapshot(
        pass_index=pass_index,
        source=source,
        path=path,
        frequencies_ghz=frequencies,
        eig_columns=eig_columns,
        port_epr=port_epr,
        capacitance_matrix_f=capacitance,
        surface_epr=surface_epr,
        error_indicators=error_indicators,
        error_norm=error_norm,
        degrees_of_freedom=palace_cost.degrees_of_freedom,
        mesh_elements=palace_cost.mesh_elements,
        elapsed_total_s=palace_cost.elapsed_total_s,
        peak_node_memory_mb=palace_cost.peak_node_memory_mb,
        surface_mask_epr=surface_mask_epr,
    )


class _PalaceCost(NamedTuple):
    """Size, time and memory that Palace recorded in one pass's ``palace.json``."""

    degrees_of_freedom: int | None
    mesh_elements: int | None
    elapsed_total_s: float | None
    peak_node_memory_mb: float | None


def _read_palace_cost(path: Path) -> _PalaceCost:
    """Every field is ``None`` when the file or the field is absent."""
    payload = read_optional_json(path)
    problem_block = payload.get("Problem") if payload else None
    if not isinstance(problem_block, dict):
        problem_block = {}
    elapsed = None
    if payload is not None:
        durations = payload.get("ElapsedTime", {})
        if isinstance(durations, dict):
            duration_map = durations.get("Durations")
            if isinstance(duration_map, dict) and isinstance(
                duration_map.get("Total"), (int, float)
            ):
                elapsed = float(duration_map["Total"])
    return _PalaceCost(
        degrees_of_freedom=as_int(problem_block.get("DegreesOfFreedom")),
        mesh_elements=as_int(problem_block.get("MeshElements")),
        elapsed_total_s=elapsed,
        peak_node_memory_mb=mapping_max(
            payload.get("PeakNodeMemoryMegabytes") if payload else None
        ),
    )


def _read_numeric_table_columns(path: Path) -> dict[str, tuple[float, ...]] | None:
    if not path.is_file():
        return None
    table = read_csv_table(path)
    columns: dict[str, tuple[float, ...]] = {}
    for header in table.headers:
        if header in _INDEX_COLUMNS:
            continue
        values: list[float] = []
        usable = True
        for row in table.rows:
            value = row.get(header)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                usable = False
                break
            values.append(float(value))
        if usable and values:
            columns[header] = tuple(values)
    return columns or None


def _read_capacitance(path: Path) -> tuple[tuple[float, ...], ...] | None:
    if not path.is_file():
        return None
    table = read_csv_table(path)
    matrix: list[tuple[float, ...]] = []
    value_headers = [
        header for header in table.headers if header.startswith(_CAP_HEADER_PREFIX)
    ]
    if not value_headers:
        return None
    for row in table.rows:
        values: list[float] = []
        for header in value_headers:
            value = row.get(header)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return None
            values.append(float(value))
        matrix.append(tuple(values))
    return tuple(matrix) if matrix else None


def _read_error_indicators(path: Path) -> dict[str, float] | None:
    if not path.is_file():
        return None
    table = read_csv_table(path)
    if not table.rows:
        return None
    payload: dict[str, float] = {}
    for header in table.headers:
        value = table.rows[0].get(header)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            payload[header] = float(value)
    return payload or None


__all__ = [
    "AmrPassSnapshot",
    "PalaceFailureDiagnosis",
    "PalaceResultSelection",
    "PassCostRecord",
    "SurfaceEprRecord",
    "SurfaceEprSeriesSnapshot",
    "SurfaceMaskEprRecord",
    "SurfaceMaskEprSeriesSnapshot",
]
