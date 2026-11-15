"""Bind native surface-Q CSV columns to the Surface-EPR rows recorded in the index map."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, cast

from scgsim.contracts.palace_run_folder import palace_problem
from scgsim.palace.results.index_contract import validate_index_entries
from scgsim.palace.results.inspection_records import (
    SurfaceEprRecord,
    SurfaceEprSeriesSnapshot,
    SurfaceMarginEprRecord,
    SurfaceMarginEprSeriesSnapshot,
)
from scgsim.palace.results.resolved_records import ParsedTable
from scgsim.palace.results.surface_contract import entry_rows, surface_epr_entries
from scgsim.palace.results.tables import (
    read_csv_table,
    requested_modes,
    validate_surface_table,
)


@dataclass(frozen=True)
class SurfaceBinding:
    """What one whole-surface Dielectric column integrates: its sidecar rows, Entities, Nets and film loss."""

    index: int
    interface_type: str
    name: str
    rows: tuple[str, ...]
    face_kinds: tuple[str, ...]
    entities: tuple[str, ...]
    nets: tuple[str, ...]
    attributes: tuple[int, ...]
    loss_tangent: float | None
    lineage: dict[str, dict[str, float]]

    def record_fields(self) -> dict[str, Any]:
        """The binding's fields by name, as a result record of its column takes them."""
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass(frozen=True)
class SurfaceMarginBinding:
    """What one margin Dielectric column integrates: its rows, its whole surface and the bands it unions."""

    index: int
    interface_type: str
    name: str
    rows: tuple[str, ...]
    face_kinds: tuple[str, ...]
    entities: tuple[str, ...]
    nets: tuple[str, ...]
    attributes: tuple[int, ...]
    loss_tangent: float | None
    baseline_index: int
    margin_index: int
    margin_um: float
    bands: tuple[int, ...]

    def record_fields(self) -> dict[str, Any]:
        """The binding's fields by name, as a result record of its column takes them."""
        return {field.name: getattr(self, field.name) for field in fields(self)}


def _surface_inverse_loss(records: Sequence[SurfaceEprRecord]) -> float | None:
    """Sum participation-weighted loss only when every surface loss tangent is known."""
    if any(record.loss_tangent is None for record in records):
        return None
    inverse_loss = sum(
        record.participation * float(cast(float, record.loss_tangent))
        for record in records
    )
    return inverse_loss if math.isfinite(inverse_loss) and inverse_loss >= 0 else None


def read_surface_bindings(
    index_map: dict[str, Any] | None,
) -> tuple[SurfaceBinding, ...] | None:
    """Bind each whole-surface Dielectric index to the sidecar rows, Entities and Nets it integrates."""
    entries = _entries(index_map)
    bindings = tuple(
        SurfaceBinding(**_identity(entry), lineage=_lineage(entry))
        for entry in entries
        if "inset" not in entry
    )
    return bindings or None


def read_surface_margin_bindings(
    index_map: dict[str, Any] | None,
) -> tuple[SurfaceMarginBinding, ...] | None:
    """Bind each margin Dielectric index to its whole surface and the bands it unions."""
    entries = _entries(index_map)
    bindings = tuple(
        SurfaceMarginBinding(
            **_identity(entry),
            baseline_index=entry["baseline_index"],
            margin_index=entry["inset"]["margin_index"],
            margin_um=float(entry["inset"]["margin_um"]),
            bands=tuple(entry["inset"]["bands"]),
        )
        for entry in entries
        if "inset" in entry
    )
    return bindings or None


def _entries(index_map: dict[str, Any] | None) -> tuple[dict[str, Any], ...]:
    """The validated Surface-EPR entries, or none when the index map is absent or unreadable."""
    if index_map is None:
        return ()
    try:
        validate_index_entries(index_map)
        return surface_epr_entries(index_map)
    except (KeyError, TypeError, ValueError):
        return ()


def _identity(entry: dict[str, Any]) -> dict[str, Any]:
    """What a result record says about the surface its column integrates."""
    rows = entry_rows(entry)
    return {
        "index": entry["index"],
        "interface_type": rows[0]["interface_type"],
        "name": entry["entry_name"],
        "rows": tuple(row["row"] for row in rows),
        "face_kinds": tuple(sorted({row["face_kind"] for row in rows})),
        "entities": tuple(sorted({e for row in rows for e in row["entities"]})),
        "nets": tuple(sorted({row["net"] for row in rows if row["net"] is not None})),
        "attributes": tuple(entry["attributes"]),
        "loss_tangent": entry["epr_spec"]["loss_tangent"],
    }


def _lineage(entry: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Which Entities share each face of the rows, as attribution only."""
    return {
        face: dict(owners)
        for row in entry_rows(entry)
        for face, owners in row["lineage"]["membership"].items()
    }


def _read_surface_table(path: Path, mode_count: int | None) -> ParsedTable:
    """Read a surface-Q table, keeping the requested modes when their count is known."""
    table = read_csv_table(path)
    return table if mode_count is None else requested_modes(table, mode_count)


@dataclass(frozen=True)
class SurfaceEprReader:
    """A native surface-Q table and its pass-local source bindings for detached EPR."""

    path: Path
    problem: str
    pass_index: int
    source: str
    frequencies_ghz: tuple[float, ...] | None
    expected_rows: int
    bindings: tuple[SurfaceBinding, ...] | None
    surface_count: int
    mode_count: int | None = None

    def read(self) -> tuple[SurfaceEprSeriesSnapshot, ...] | None:
        """Read whole-surface participation and loss series in native mode or excitation order."""
        if not self.path.is_file() or not self.bindings or self.expected_rows <= 0:
            return None
        try:
            table = _read_surface_table(self.path, self.mode_count)
            validate_surface_table(table, self.surface_count, self.expected_rows)
            snapshots: list[SurfaceEprSeriesSnapshot] = []
            problem = palace_problem(self.problem)
            for row in table.rows:
                series_index = int(row[problem.series_index])
                records = tuple(
                    SurfaceEprRecord(
                        **binding.record_fields(),
                        participation=float(row[f"p_surf[{binding.index}]"]),
                        quality_factor=float(row[f"Q_surf[{binding.index}]"]),
                    )
                    for binding in self.bindings
                )
                inverse_loss = _surface_inverse_loss(records)
                quality_factor = (
                    1.0 / inverse_loss
                    if inverse_loss is not None and inverse_loss > 0
                    else None
                )
                frequency_hz = None
                if self.frequencies_ghz is not None and 0 < series_index <= len(
                    self.frequencies_ghz
                ):
                    frequency_hz = self.frequencies_ghz[series_index - 1] * 1e9
                t1_seconds = (
                    quality_factor / (2.0 * math.pi * frequency_hz)
                    if quality_factor is not None
                    and frequency_hz is not None
                    and frequency_hz > 0
                    else None
                )
                snapshots.append(
                    SurfaceEprSeriesSnapshot(
                        pass_index=self.pass_index,
                        source=self.source,
                        series_index=series_index,
                        series_kind=problem.series,
                        records=records,
                        quality_factor_total=quality_factor,
                        t1_seconds=t1_seconds,
                        loss_status=(
                            "unavailable_missing"
                            if inverse_loss is None
                            else "unavailable_nonfinite"
                            if quality_factor is None
                            else "available"
                        ),
                    )
                )
            return tuple(snapshots)
        except (KeyError, OSError, TypeError, ValueError):
            return None


@dataclass(frozen=True)
class SurfaceMarginEprReader:
    """The margin rows of the same surface-Q table, read as an all-or-nothing source-local snapshot."""

    path: Path
    problem: str
    pass_index: int
    source: str
    expected_rows: int
    bindings: tuple[SurfaceMarginBinding, ...] | None
    surface_count: int
    mode_count: int | None = None

    def read(self) -> tuple[SurfaceMarginEprSeriesSnapshot, ...] | None:
        """Read margin participation and Q for every mode or excitation."""
        if not self.path.is_file() or not self.bindings or self.expected_rows <= 0:
            return None
        try:
            table = _read_surface_table(self.path, self.mode_count)
            validate_surface_table(table, self.surface_count, self.expected_rows)
            problem = palace_problem(self.problem)
            return tuple(
                SurfaceMarginEprSeriesSnapshot(
                    pass_index=self.pass_index,
                    source=self.source,
                    series_index=int(row[problem.series_index]),
                    series_kind=problem.series,
                    records=tuple(
                        SurfaceMarginEprRecord(
                            **binding.record_fields(),
                            participation=float(row[f"p_surf[{binding.index}]"]),
                            quality_factor=float(row[f"Q_surf[{binding.index}]"]),
                        )
                        for binding in self.bindings
                    ),
                )
                for row in table.rows
            )
        except (KeyError, OSError, TypeError, ValueError):
            return None
