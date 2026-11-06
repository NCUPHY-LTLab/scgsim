"""Bind native surface-Q CSV columns to the Surface-EPR rows recorded in the index map."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from scgsim.palace.results.index_contract import validate_index_entries
from scgsim.palace.results.report_models import (
    SurfaceEprRecord,
    SurfaceEprSeriesSnapshot,
    SurfaceMarginEprRecord,
    SurfaceMarginEprSeriesSnapshot,
)
from scgsim.palace.results.surface_contract import entry_rows, surface_epr_entries
from scgsim.palace.results.tables import read_csv_table, validate_surface_table


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
) -> tuple[dict[str, Any], ...] | None:
    """Bind each whole-surface Dielectric index to the sidecar rows, Entities and Nets it integrates."""
    entries = _entries(index_map)
    bindings = tuple(
        _identity(entry) | {"lineage": _lineage(entry)}
        for entry in entries
        if "inset" not in entry
    )
    return bindings or None


def read_surface_margin_bindings(
    index_map: dict[str, Any] | None,
) -> tuple[dict[str, Any], ...] | None:
    """Bind each margin Dielectric index to its whole surface and the bands it unions."""
    entries = _entries(index_map)
    bindings = tuple(
        _identity(entry)
        | {
            "baseline_index": entry["baseline_index"],
            "margin_index": entry["inset"]["margin_index"],
            "margin_um": float(entry["inset"]["margin_um"]),
            "bands": tuple(entry["inset"]["bands"]),
        }
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


@dataclass(frozen=True)
class SurfaceEprReader:
    """A native surface-Q table and its pass-local source bindings for detached EPR."""

    path: Path
    problem: str
    pass_index: int
    source: str
    frequencies_ghz: tuple[float, ...] | None
    expected_rows: int
    bindings: tuple[dict[str, Any], ...] | None
    surface_count: int

    def read(self) -> tuple[SurfaceEprSeriesSnapshot, ...] | None:
        """Read whole-surface participation and loss series in native mode or excitation order."""
        if not self.path.is_file() or not self.bindings or self.expected_rows <= 0:
            return None
        try:
            table = read_csv_table(self.path)
            validate_surface_table(table, self.surface_count, self.expected_rows)
            snapshots: list[SurfaceEprSeriesSnapshot] = []
            index_name = "m" if self.problem == "Eigenmode" else "i"
            for row in table.rows:
                series_index = int(row[index_name])
                records = tuple(
                    SurfaceEprRecord(
                        **binding,
                        participation=float(row[f"p_surf[{binding['index']}]"]),
                        quality_factor=float(row[f"Q_surf[{binding['index']}]"]),
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
                        series_kind="mode"
                        if self.problem == "Eigenmode"
                        else "excitation",
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
    bindings: tuple[dict[str, Any], ...] | None
    surface_count: int

    def read(self) -> tuple[SurfaceMarginEprSeriesSnapshot, ...] | None:
        """Read margin participation and Q for every mode or excitation."""
        if not self.path.is_file() or not self.bindings or self.expected_rows <= 0:
            return None
        try:
            table = read_csv_table(self.path)
            validate_surface_table(table, self.surface_count, self.expected_rows)
            index_name = "m" if self.problem == "Eigenmode" else "i"
            return tuple(
                SurfaceMarginEprSeriesSnapshot(
                    pass_index=self.pass_index,
                    source=self.source,
                    series_index=int(row[index_name]),
                    series_kind="mode" if self.problem == "Eigenmode" else "excitation",
                    records=tuple(
                        SurfaceMarginEprRecord(
                            **binding,
                            participation=float(row[f"p_surf[{binding['index']}]"]),
                            quality_factor=float(row[f"Q_surf[{binding['index']}]"]),
                        )
                        for binding in self.bindings
                    ),
                )
                for row in table.rows
            )
        except (KeyError, OSError, TypeError, ValueError):
            return None
