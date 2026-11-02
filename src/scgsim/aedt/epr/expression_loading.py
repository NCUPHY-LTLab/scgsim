"""Load a batch CLC library and attach verified native definitions to its records."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scgsim.aedt.epr.expression_codec import read_native_named_expressions


def load_compiled_expressions(
    app: Any,
    compiled: Sequence[tuple[dict[str, Any], list[tuple[str, str]], dict[str, bytes]]],
    evidence_dir: Path,
) -> dict[str, Any]:
    """Load dependency-ordered local definitions once, then verify every name."""
    return _ExpressionLoad(app, compiled, evidence_dir).load()


@dataclass
class _ExpressionLoad:
    """Keep one calculator, batch library, and its native readback together."""

    app: Any
    compiled: Sequence[tuple[dict[str, Any], list[tuple[str, str]], dict[str, bytes]]]
    evidence_dir: Path

    def load(self) -> dict[str, Any]:
        self._check_collisions()
        self._write_library()
        self._load_library()
        self._verify_loaded_names()
        self._attach_saved_definitions()
        return self._timings()

    def _check_collisions(self) -> None:
        self.definitions = [
            definition for _, blocks, _ in self.compiled for definition in blocks
        ]
        self.names = [name for name, _ in self.definitions]
        if not self.names or len(self.names) != len(set(self.names)):
            raise RuntimeError(
                "EPR batch definitions are empty or repeat a native name"
            )
        self.calculator = self.app.post.fields_calculator
        started = time.perf_counter()
        existing = [
            name for name in self.names if self.calculator.is_expression_defined(name)
        ]
        if existing:
            raise RuntimeError(f"HFSS expressions already exist: {existing!r}")
        self.collision_check_seconds = time.perf_counter() - started

    def _write_library(self) -> None:
        started = time.perf_counter()
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.files = {
            filename: data
            for _, _, evidence in self.compiled
            for filename, data in evidence.items()
        }
        if len(self.files) != sum(len(evidence) for _, _, evidence in self.compiled):
            raise RuntimeError("EPR evidence filenames repeat")
        for filename, data in self.files.items():
            (self.evidence_dir / filename).write_bytes(data)
        self.library = self.evidence_dir / "scgsim_epr_batch.clc"
        self.library_bytes = "".join(block for _, block in self.definitions).encode(
            "utf-8"
        )
        self.library.write_bytes(self.library_bytes)
        self.evidence_write_seconds = time.perf_counter() - started

    def _load_library(self) -> None:
        started = time.perf_counter()
        try:
            self.result = self.calculator.ofieldsreporter.LoadNamedExpressions(
                str(self.library), "Fields", self.names
            )
        except Exception as exc:
            try:
                present = [
                    name
                    for name in self.names
                    if self.calculator.is_expression_defined(name)
                ]
                progress = (
                    f"{len(present)}/{len(self.names)} definitions became present"
                )
            except Exception as readback_exc:  # noqa: BLE001 -- retain the original native load error when partial readback also fails.
                progress = f"{len(self.names)} requested definitions; partial readback unavailable: {readback_exc!r}"
            raise RuntimeError(
                f"HFSS batch named-expression load raised after {progress}"
            ) from exc
        self.batch_import_seconds = time.perf_counter() - started

    def _verify_loaded_names(self) -> None:
        started = time.perf_counter()
        present = [
            name for name in self.names if self.calculator.is_expression_defined(name)
        ]
        if self.result is False or len(present) != len(self.names):
            present_set = set(present)
            missing = [name for name in self.names if name not in present_set]
            raise RuntimeError(
                f"HFSS batch named-expression load was partial or failed: "
                f"{len(present)}/{len(self.names)} present; missing={missing!r}; returned={self.result!r}"
            )
        self.postload_readback_seconds = round(time.perf_counter() - started, 6)

    def _attach_saved_definitions(self) -> None:
        readback_started = time.perf_counter()
        self.native_expression_readback = read_native_named_expressions(
            self.app,
            names=self.names,
            definitions=self.definitions,
            evidence_dir=self.evidence_dir,
        )
        self.native_definition_readback_seconds = time.perf_counter() - readback_started
        for expression, blocks, _ in self.compiled:
            expression["definition_status"] = "native_readback_verified"
            expression["native_readback"] = {
                "status": self.native_expression_readback["status"],
                "api": self.native_expression_readback["api"],
                "path": self.native_expression_readback["path"],
                "sha256": self.native_expression_readback["sha256"],
                "bytes": self.native_expression_readback["bytes"],
                "expression_definition_names": [name for name, _ in blocks],
                "definitions": {
                    name: self.native_expression_readback["definitions"][name]
                    for name, _ in blocks
                },
            }

    def _timings(self) -> dict[str, Any]:
        return {
            "library_sha256": hashlib.sha256(self.library_bytes).hexdigest(),
            "native_definition_count": len(self.names),
            "main_expression_count": len(self.compiled),
            "collision_check_seconds": round(self.collision_check_seconds, 6),
            "evidence_write_seconds": round(self.evidence_write_seconds, 6),
            "batch_import_seconds": round(self.batch_import_seconds, 6),
            "postload_readback_seconds": self.postload_readback_seconds,
            "native_definition_readback_seconds": round(
                self.native_definition_readback_seconds, 6
            ),
            "native_expression_readback": self.native_expression_readback,
        }
