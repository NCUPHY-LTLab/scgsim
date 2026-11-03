"""Read the version-1 native AMR inventory with confinement and receipt binding.

The manifest is the sole discovery authority. Unlisted files are never read;
strict consumers verify each consumed file's sealed size and SHA-256 first.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class NativeDiagnosticSource:
    """Explicit native output scope and optional existing SCGSim receipt proof."""

    run_dir: Path
    output_dir: Path
    terminal_net_ids: Mapping[int, str]
    receipt: Mapping[str, Any] | None = None
    require_receipt: bool = False


@dataclass
class NativeDiagnosticReader:
    """One manifest inventory and the files it authorizes this consumer to read."""

    source: NativeDiagnosticSource
    inventory: frozenset[str] = field(default_factory=frozenset, init=False)
    mpi_size: int | None = field(default=None, init=False)
    missing_records: list[str] = field(default_factory=list, init=False)

    @property
    def base(self) -> Path:
        """Confine the diagnostic directory to the explicitly supplied run scope."""
        root = self.source.run_dir.resolve()
        output = self.source.output_dir.resolve()
        if not output.is_relative_to(root):
            raise ValueError("native AMR output directory escapes its run scope")
        base = (output / "amr-diagnostics").resolve()
        if not base.is_relative_to(root):
            raise ValueError("native AMR diagnostic directory escapes its run scope")
        return base

    def read_manifest(self) -> dict[str, Any] | None:
        """Read the exact manifest, then establish its inventory and rank count."""
        record = self._read_path(self.path("manifest.json"))
        if record is None:
            return None
        validate_record(record, "manifest")
        if record.get("coordinate_units") != "nondimensional":
            raise ValueError("unsupported native AMR coordinate units")
        self.mpi_size = native_int(record, "mpi_size", minimum=1)
        files = record_paths(record, "files")
        paths = tuple(self.path(relative) for relative in files)
        if self.path("manifest.json") in paths or len(set(paths)) != len(paths):
            raise ValueError(
                "native AMR inventory repeats a file or includes its manifest"
            )
        self.inventory = frozenset(files)
        for path in paths:
            self._verify_inventory_path(path)
        self.referenced_paths(record, "solves")
        self.referenced_paths(record, "transitions")
        return record

    def read_record(self, relative: str, kind: str) -> dict[str, Any] | None:
        """Read one explicitly inventoried JSON record and verify its record identity."""
        if relative not in self.inventory:
            raise ValueError(f"native AMR record is absent from inventory: {relative}")
        record = self._read_path(self.path(relative))
        if record is not None:
            validate_record(record, kind)
        return record

    def referenced_paths(
        self, record: Mapping[str, Any], field_name: str
    ) -> tuple[str, ...]:
        """Require every record or mesh reference to belong to the manifest inventory."""
        paths = record_paths(record, field_name)
        for relative in paths:
            self.path(relative)
            if relative not in self.inventory:
                raise ValueError(
                    f"native AMR {field_name} path is unlisted: {relative}"
                )
        return paths

    def path(self, relative: str) -> Path:
        """Apply the producer's relative-path domain and reject symlink escape."""
        if (
            not isinstance(relative, str)
            or not relative
            or ".." in relative
            or re.fullmatch(r"[A-Za-z0-9_./-]+", relative) is None
            or Path(relative).is_absolute()
        ):
            raise ValueError("native AMR paths must be confined relative file paths")
        candidate = (self.base / relative).resolve()
        if candidate == self.base.resolve() or not candidate.is_relative_to(
            self.base.resolve()
        ):
            raise ValueError("native AMR path escapes the diagnostic directory")
        return candidate

    def _read_path(self, path: Path) -> dict[str, Any] | None:
        """Verify strict receipt identity before decoding any consumed native JSON."""
        if not path.is_file():
            if self.source.require_receipt:
                raise FileNotFoundError(path)
            self.missing_records.append(path.relative_to(self.base).as_posix())
            return None
        data = path.read_bytes()
        if self.source.require_receipt:
            self._verify_receipt_path(path, data)
        record = json.loads(data)
        if not isinstance(record, dict):
            raise TypeError("native AMR records must contain JSON objects")
        return record

    def _verify_receipt_path(self, path: Path, data: bytes) -> None:
        """Bind consumed bytes to the exact run-relative returned-output entry."""
        expected = self._receipt_record(path)
        if (len(data), hashlib.sha256(data).hexdigest()) != expected:
            raise ValueError(f"native AMR receipt hash mismatch: {path.name}")

    def _verify_inventory_path(self, path: Path) -> None:
        """Verify every inventoried file, streaming meshes rather than loading them in memory."""
        if not path.is_file():
            if self.source.require_receipt:
                raise FileNotFoundError(path)
            self.missing_records.append(path.relative_to(self.base).as_posix())
            return
        if not self.source.require_receipt:
            return
        expected = self._receipt_record(path)
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                size += len(chunk)
                digest.update(chunk)
        if (size, digest.hexdigest()) != expected:
            raise ValueError(f"native AMR receipt hash mismatch: {path.name}")

    def _receipt_record(self, path: Path) -> tuple[Any, Any]:
        """Find the unique present returned-file record for the exact run-relative path."""
        if self.source.receipt is None:
            raise ValueError(
                "strict native AMR analysis requires returned receipt proof"
            )
        relative = path.relative_to(self.source.run_dir.resolve()).as_posix()
        records = self.source.receipt.get("output_files")
        if not isinstance(records, (list, tuple)):
            raise ValueError("native AMR receipt has no output-file records")
        matches = tuple(
            record
            for record in records
            if isinstance(record, Mapping) and record.get("path") == relative
        )
        if len(matches) != 1 or matches[0].get("present") is not True:
            raise ValueError(
                f"native AMR file has no unique present receipt record: {relative}"
            )
        return matches[0].get("bytes"), matches[0].get("sha256")


def validate_record(record: Mapping[str, Any], kind: str) -> None:
    """Require the authoritative version-1 schema and expected native record type."""
    if (
        record.get("schema") != "palace.amr_diagnostics"
        or type(record.get("schema_version")) is not int
        or record["schema_version"] != 1
        or record.get("record_type") != kind
    ):
        raise ValueError(f"unsupported native AMR {kind} record identity")


def record_paths(record: Mapping[str, Any], field_name: str) -> tuple[str, ...]:
    """Read a producer-defined exact path list without searching the output tree."""
    values = record.get(field_name)
    if not isinstance(values, list) or any(
        not isinstance(path, str) for path in values
    ):
        raise TypeError(f"native AMR {field_name} must be an explicit path list")
    if len(set(values)) != len(values):
        raise ValueError(f"native AMR {field_name} repeats a path")
    return tuple(values)


def native_int(record: Mapping[str, Any], name: str, *, minimum: int | None = 0) -> int:
    """Read an integer in the native schema's explicitly defined count/index domain."""
    value = record.get(name)
    if type(value) is not int or (minimum is not None and value < minimum):
        raise ValueError(f"native AMR {name} must be an integer at least {minimum}")
    return value


def native_float(record: Mapping[str, Any], name: str) -> float:
    """Read a finite JSON number as required by the native diagnostics schema."""
    value = record.get(name)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"native AMR {name} must be a finite JSON number")
    return float(value)


def native_text(record: Mapping[str, Any], name: str) -> str:
    """Read an explicitly recorded nonempty native mesh or record identifier."""
    value = record.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"native AMR {name} must be nonempty text")
    return value


def native_indices(record: Mapping[str, Any], name: str) -> tuple[int, ...]:
    """Read unique nonnegative native terminal or coarse-cell indices."""
    values = record.get(name)
    if not isinstance(values, list) or any(
        type(value) is not int or value < 0 for value in values
    ):
        raise ValueError(f"native AMR {name} must contain nonnegative integer indices")
    if len(set(values)) != len(values):
        raise ValueError(f"native AMR {name} repeats an index")
    return tuple(values)
