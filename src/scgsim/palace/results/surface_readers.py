"""Bind native surface and mask CSV rows to recorded semantic surface identities."""

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
    SurfaceMaskEprRecord,
    SurfaceMaskEprSeriesSnapshot,
)
from scgsim.palace.results.tables import (
    read_csv_table,
    validate_surface_mask_table,
    validate_surface_table,
)

_SURFACE_TYPES = ("MA", "MS", "SA")


def _surface_inverse_loss(records: Sequence[SurfaceEprRecord]) -> float | None:
    """Sum participation-weighted loss only when every surface loss tangent is known."""
    if any(record.loss_tangent is None for record in records):
        return None
    inverse_loss = sum(
        record.participation * float(cast(float, record.loss_tangent))
        for record in records
    )
    return inverse_loss if math.isfinite(inverse_loss) and inverse_loss >= 0 else None


def _optional_string(value: Any) -> str | None:
    """Read an optional semantic identifier while rejecting empty or nontext identities."""
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("optional semantic identity must be a non-empty string.")
    return value


def read_surface_bindings(
    index_map: dict[str, Any] | None,
) -> tuple[dict[str, Any], ...] | None:
    """Bind baseline dielectric indices to exact recorded surface and conductor provenance."""
    if index_map is None:
        return None
    try:
        validate_index_entries(index_map)
        entries = [
            entry
            for entry in index_map["entries"]
            if entry["section"] == "Boundaries.Postprocessing.Dielectric"
        ]
        entries.sort(key=lambda entry: entry["index"])
        bindings: list[dict[str, Any]] = []
        for entry in entries:
            if "mask" in entry:
                continue
            metadata = entry["metadata"]
            interface_type = metadata.get("interface_type")
            surface_id = metadata.get("surface_id")
            face_kind = metadata.get("face_kind")
            owners = metadata.get("owner_semantic_ids")
            provenance = metadata.get("source_provenance")
            if interface_type not in _SURFACE_TYPES:
                raise ValueError("surface interface_type must be MA, MS, or SA.")
            if not isinstance(surface_id, str) or not surface_id:
                raise ValueError("surface_id must be a non-empty string.")
            if not isinstance(face_kind, str) or not face_kind:
                raise ValueError("face_kind must be a non-empty string.")
            if (
                not isinstance(owners, list)
                or not owners
                or not all(isinstance(owner, str) and owner for owner in owners)
            ):
                raise ValueError("owner_semantic_ids must be non-empty strings.")
            if not isinstance(provenance, dict):
                raise TypeError("source_provenance must be a mapping.")
            epr_spec = entry.get("epr_spec")
            loss_tangent = (
                epr_spec.get("loss_tangent") if isinstance(epr_spec, dict) else None
            )
            if (
                isinstance(epr_spec, dict)
                and epr_spec.get("schema_version") != "scgsim.palace.surface-film.v2"
                and loss_tangent == 0
            ):
                loss_tangent = None
            if loss_tangent is not None and (
                isinstance(loss_tangent, bool)
                or not isinstance(loss_tangent, (int, float))
                or not math.isfinite(float(loss_tangent))
                or loss_tangent < 0
            ):
                raise ValueError(
                    "surface loss_tangent must be finite and non-negative."
                )
            bindings.append(
                {
                    "index": entry["index"],
                    "interface_type": interface_type,
                    "surface_id": surface_id,
                    "face_kind": face_kind,
                    "owner_semantic_ids": tuple(owners),
                    "net_id": _optional_string(metadata.get("net_id")),
                    "equipotential_id": _optional_string(
                        metadata.get("equipotential_id")
                    ),
                    "source_provenance": provenance,
                    "loss_tangent": (
                        None if loss_tangent is None else float(loss_tangent)
                    ),
                }
            )
        return tuple(bindings) or None
    except (KeyError, TypeError, ValueError):
        return None


def read_surface_mask_bindings(
    index_map: dict[str, Any] | None,
) -> tuple[dict[str, Any], ...] | None:
    """Bind native Inset indices to their recorded baseline and ordered margin metadata."""
    if index_map is None:
        return None
    try:
        validate_index_entries(index_map)
        bindings: list[dict[str, Any]] = []
        for entry in sorted(index_map["entries"], key=lambda item: item["index"]):
            if (
                entry.get("section") != "Boundaries.Postprocessing.Dielectric"
                or "mask" not in entry
            ):
                continue
            metadata = entry["metadata"]
            mask = entry["mask"]
            owners = metadata.get("owner_semantic_ids")
            provenance = metadata.get("source_provenance")
            if (
                metadata.get("interface_type") not in _SURFACE_TYPES
                or not isinstance(metadata.get("surface_id"), str)
                or not metadata["surface_id"]
                or not isinstance(metadata.get("face_kind"), str)
                or not metadata["face_kind"]
                or not isinstance(owners, list)
                or not owners
                or not all(isinstance(owner, str) and owner for owner in owners)
                or not isinstance(provenance, dict)
            ):
                raise ValueError("surface mask provenance is incomplete.")
            epr_spec = entry.get("epr_spec")
            loss_tangent = (
                epr_spec.get("loss_tangent") if isinstance(epr_spec, dict) else None
            )
            if (
                isinstance(epr_spec, dict)
                and epr_spec.get("schema_version") != "scgsim.palace.surface-film.v2"
                and loss_tangent == 0
            ):
                loss_tangent = None
            if loss_tangent is not None and (
                isinstance(loss_tangent, bool)
                or not isinstance(loss_tangent, (int, float))
                or not math.isfinite(float(loss_tangent))
                or loss_tangent < 0
            ):
                raise ValueError(
                    "surface mask loss_tangent must be finite and non-negative."
                )
            bindings.append(
                {
                    "index": int(entry["index"]),
                    "baseline_index": int(entry["baseline_index"]),
                    "margin_index": int(mask["margin_index"]),
                    "margin_um": float(mask["margin_um"]),
                    "native_margin": float(mask["native_margin"]),
                    "model_l0_m": float(mask["model_l0_m"]),
                    "interface_type": metadata["interface_type"],
                    "surface_id": metadata["surface_id"],
                    "face_kind": metadata["face_kind"],
                    "owner_semantic_ids": tuple(owners),
                    "net_id": _optional_string(metadata.get("net_id")),
                    "equipotential_id": _optional_string(
                        metadata.get("equipotential_id")
                    ),
                    "source_provenance": provenance,
                    "loss_tangent": (
                        None if loss_tangent is None else float(loss_tangent)
                    ),
                }
            )
        return tuple(bindings) or None
    except (KeyError, TypeError, ValueError):
        return None


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
    surface_count: int | None = None

    def read(self) -> tuple[SurfaceEprSeriesSnapshot, ...] | None:
        """Read baseline participation and loss series in native mode or excitation order."""
        if not self.path.is_file() or not self.bindings or self.expected_rows <= 0:
            return None
        try:
            table = read_csv_table(self.path)
            validate_surface_table(
                table,
                len(self.bindings)
                if self.surface_count is None
                else self.surface_count,
                self.expected_rows,
            )
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
class SurfaceMaskEprReader:
    """Load both native mask tables as one source-local, all-or-nothing snapshot."""

    q_path: Path
    energy_path: Path
    problem: str
    pass_index: int
    source: str
    expected_rows: int
    bindings: tuple[dict[str, Any], ...] | None

    def read(self) -> tuple[SurfaceMaskEprSeriesSnapshot, ...] | None:
        """Read mask participation and energy together as one source-local snapshot."""
        if (
            not self.q_path.is_file()
            or not self.energy_path.is_file()
            or not self.bindings
            or self.expected_rows <= 0
        ):
            return None
        try:
            q_table = read_csv_table(self.q_path)
            energy_table = read_csv_table(self.energy_path)
            validate_surface_mask_table(
                q_table,
                problem=self.problem,
                expected_rows=self.expected_rows,
                entries=self.bindings,
            )
            validate_surface_mask_table(
                energy_table,
                problem=self.problem,
                expected_rows=self.expected_rows,
                entries=self.bindings,
            )
            index_name = "m" if self.problem == "Eigenmode" else "i"
            snapshots: list[SurfaceMaskEprSeriesSnapshot] = []
            for q_row, energy_row in zip(q_table.rows, energy_table.rows, strict=True):
                series_index = int(q_row[index_name])
                if energy_row[index_name] != q_row[index_name]:
                    raise ValueError("native mask Q and energy rows do not align.")
                records = tuple(
                    SurfaceMaskEprRecord(
                        **binding,
                        participation=float(q_row[f"p_surf_mask[{binding['index']}]"]),
                        quality_factor=float(q_row[f"Q_surf_mask[{binding['index']}]"]),
                        energy_j=float(
                            energy_row[f"E_surf_mask[{binding['index']}] (J)"]
                        ),
                        contribution_status=(
                            "zero_contribution"
                            if q_row[f"p_surf_mask[{binding['index']}]"] == 0
                            or energy_row[f"E_surf_mask[{binding['index']}] (J)"] == 0
                            else "available"
                        ),
                    )
                    for binding in self.bindings
                )
                snapshots.append(
                    SurfaceMaskEprSeriesSnapshot(
                        pass_index=self.pass_index,
                        source=self.source,
                        series_index=series_index,
                        series_kind="mode"
                        if self.problem == "Eigenmode"
                        else "excitation",
                        records=records,
                    )
                )
            return tuple(snapshots)
        except (KeyError, OSError, TypeError, ValueError):
            return None
