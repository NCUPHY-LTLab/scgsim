"""Compile immutable CLC definitions and verify saved native expression readback."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def clc_block(
    name: str, operations: Sequence[str], expression: str | None = None
) -> str:
    """Encode one immutable native Named_Expression block."""
    lines = ["$begin 'Named_Expression'", f"\tName('{name}')"]
    if expression is not None:
        lines.append(f"\tExpression('{expression}')")
    lines.extend(f"\t{operation}" for operation in operations)
    lines.append("$end 'Named_Expression'")
    return "\n".join(lines) + "\n"


def _field_selection_operations(block: str) -> list[dict[str, str]]:
    pattern = re.compile(r"(?m)^\s*(Enter[A-Za-z]+)\(([^\r\n]*)\)\s*$")
    selections = []
    for command, argument in pattern.findall(block):
        argument = argument.strip()
        if len(argument) >= 2 and argument[0] == argument[-1] and argument[0] in "'\"":
            argument = argument[1:-1]
        if command == "EnterAdjacentSurf":
            command = "EnterAdjacentSurface"
        selections.append({"command": command, "argument": argument})
    return selections


def parse_native_expression_artifact(
    payload: bytes,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Decode definition evidence and reference-bearing bodies in one pass."""
    text = payload.decode("utf-8")
    block_pattern = re.compile(
        r"(?ms)^\$begin 'Named_Expression'\s*(.*?)^\$end 'Named_Expression'\s*$"
    )
    name_pattern = re.compile(r"(?m)^\s*Name\(['\"]([^'\"]+)['\"]\)\s*$")
    native_definitions: dict[str, dict[str, Any]] = {}
    blocks: dict[str, str] = {}
    for match in block_pattern.finditer(text):
        block = match.group(1)
        name_match = name_pattern.search(block)
        if name_match is None:
            raise RuntimeError(
                "HFSS native expression readback lacks a definition name"
            )
        name = name_match.group(1)
        if name in native_definitions:
            if blocks[name] != block:
                raise RuntimeError(
                    f"HFSS native expression readback conflicts for {name!r}"
                )
            continue
        blocks[name] = block
        native_definitions[name] = {
            "sha256": hashlib.sha256(block.encode("utf-8")).hexdigest(),
            "field_selection_operations": _field_selection_operations(block),
        }
    return native_definitions, blocks


def parse_native_expression_readback(payload: bytes) -> dict[str, dict[str, Any]]:
    """Parse saved native CLC definitions without changing their evidence schema."""
    return parse_native_expression_artifact(payload)[0]


def _required_native_definitions(
    definitions: Mapping[str, dict[str, Any]],
    blocks: Mapping[str, str],
    roots: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Follow required named helpers, leaving unrelated shared definitions alone."""
    reference_pattern = re.compile(
        r"\s*NameOfExpression\s*\(\s*(['\"])([^'\"\r\n]+)\1\s*\)\s*"
    )
    # field_integral_operations emits this native magnetic vector, not a helper.
    intrinsic_names = {"<Hx,Hy,Hz>"}
    required: dict[str, dict[str, Any]] = {}
    active: set[str] = set()
    visited: set[str] = set()
    pending = [(name, False) for name in reversed(roots)]
    while pending:
        name, leaving = pending.pop()
        if leaving:
            active.remove(name)
            visited.add(name)
            continue
        if name in visited:
            continue
        if name in active:
            raise RuntimeError(
                f"required native CLC named references cycle at {name!r}"
            )
        if name not in definitions or name not in blocks:
            raise RuntimeError(
                f"required native CLC named reference is unresolved: {name!r}"
            )
        active.add(name)
        required[name] = definitions[name]
        pending.append((name, True))
        references = []
        for line in blocks[name].splitlines():
            if not re.match(r"\s*NameOfExpression\b", line):
                continue
            match = reference_pattern.fullmatch(line)
            if match is None:
                raise RuntimeError(
                    "required native CLC named reference grammar is unsupported"
                )
            reference = match.group(2)
            if reference not in intrinsic_names:
                references.append(reference)
        pending.extend((reference, False) for reference in reversed(references))
    return {name: required[name] for name in sorted(required)}


def validated_expression_readback(
    expressions: Sequence[Mapping[str, Any]],
    cache: Mapping[str, Any],
    *,
    evidence_root: Path,
) -> dict[str, dict[str, Any]]:
    """Bind retained per-expression evidence to actual saved CLC bytes."""
    return _ExpressionReadback(expressions, cache, evidence_root).validate()


@dataclass
class _ExpressionReadback:
    """Keep parsed artifact identities and required expression roots together."""

    expressions: Sequence[Mapping[str, Any]]
    cache: Mapping[str, Any]
    evidence_root: Path

    def validate(self) -> dict[str, dict[str, Any]]:
        self.parsed_files: dict[str, dict[str, dict[str, Any]]] = {}
        self.parsed_blocks: dict[str, dict[str, str]] = {}
        self.file_identities: dict[str, tuple[Any, Any]] = {}
        self.definitions: dict[str, dict[str, Any]] = {}

        if not self.expressions:
            raise RuntimeError("native preparation lacks saved CLC expression evidence")
        for expression in self.expressions:
            self._validate_expression(expression)
        # Aggregate definitions include unrelated library entries, while roots stay per expression.
        self.roots = list(self.definitions)
        if "native_expression_readback" in self.cache:
            self._validate_aggregate()
        return self._required_definitions()

    def _read_record(self, record: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        if (
            record.get("status") != "native_clc_saved_and_field_selection_read_back"
            or record.get("api") != "FieldsReporter.SaveNamedExpressions"
        ):
            raise RuntimeError("prepared CLC readback operation is unavailable")
        relative = record.get("path")
        if not isinstance(relative, str):
            raise RuntimeError("prepared CLC readback path is unavailable")
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError("prepared CLC readback path is not contained")
        try:
            suffix = path.relative_to("metadata/epr_expressions")
            actual = (self.evidence_root / suffix).resolve()
            actual.relative_to(self.evidence_root.resolve())
        except ValueError as exc:
            raise RuntimeError("prepared CLC readback path is not contained") from exc
        identity = (record.get("sha256"), record.get("bytes"))
        if (
            relative in self.file_identities
            and self.file_identities[relative] != identity
        ):
            raise RuntimeError("prepared CLC artifact identities conflict")
        if relative not in self.parsed_files:
            payload = actual.read_bytes()
            if (
                not payload
                or len(payload) != record.get("bytes")
                or hashlib.sha256(payload).hexdigest() != record.get("sha256")
            ):
                raise RuntimeError(
                    "prepared CLC readback artifact is missing or changed"
                )
            self.parsed_files[relative], self.parsed_blocks[relative] = (
                parse_native_expression_artifact(payload)
            )
            self.file_identities[relative] = identity
        return self.parsed_files[relative]

    def _validate_expression(self, expression: Mapping[str, Any]) -> None:
        if expression.get("definition_status") != "native_readback_verified":
            raise RuntimeError("native preparation lacks saved CLC definition readback")
        record = expression.get("native_readback")
        if not isinstance(record, Mapping):
            raise RuntimeError("prepared expression CLC readback is unavailable")
        actual = self._read_record(record)
        names = record.get("expression_definition_names")
        recorded = record.get("definitions")
        if (
            not isinstance(names, list)
            or not names
            or any(not isinstance(name, str) for name in names)
            or len(set(names)) != len(names)
            or not isinstance(recorded, Mapping)
            or set(recorded) != set(names)
        ):
            raise RuntimeError("prepared expression definition membership differs")
        parent_name = expression.get("name")
        if not isinstance(parent_name, str) or parent_name not in recorded:
            raise RuntimeError(
                "prepared parent expression lacks its own native definition"
            )
        for name in names:
            if name not in actual or recorded[name] != actual[name]:
                raise RuntimeError(
                    "prepared expression differs from native CLC readback"
                )
            if name in self.definitions and self.definitions[name] != actual[name]:
                raise RuntimeError("prepared expression definitions conflict")
            self.definitions[name] = actual[name]

    def _validate_aggregate(self) -> None:
        aggregate = self.cache["native_expression_readback"]
        if not isinstance(aggregate, Mapping):
            raise RuntimeError("prepared aggregate CLC readback is invalid")
        actual = self._read_record(aggregate)
        names = aggregate.get("definition_names")
        recorded = aggregate.get("definitions")
        if (
            not isinstance(names, list)
            or not names
            or any(not isinstance(name, str) for name in names)
            or len(set(names)) != len(names)
            or not isinstance(recorded, Mapping)
            or set(names) != set(recorded)
            or not set(self.definitions).issubset(recorded)
        ):
            raise RuntimeError("prepared aggregate CLC readback membership differs")
        for name in names:
            if name not in actual or recorded[name] != actual[name]:
                raise RuntimeError(
                    "prepared aggregate differs from native CLC readback"
                )
            if name in self.definitions and self.definitions[name] != recorded[name]:
                raise RuntimeError(
                    "prepared aggregate CLC readback conflicts with expressions"
                )

    def _required_definitions(self) -> dict[str, dict[str, Any]]:
        actual_union: dict[str, dict[str, Any]] = {}
        block_union: dict[str, str] = {}
        for relative, actual in self.parsed_files.items():
            for name, value in actual.items():
                if name in actual_union and actual_union[name] != value:
                    raise RuntimeError("prepared native CLC artifacts conflict")
                actual_union[name] = value
                block_union[name] = self.parsed_blocks[relative][name]
        return _required_native_definitions(actual_union, block_union, self.roots)


def read_native_named_expressions(
    app: Any,
    *,
    names: Sequence[str],
    definitions: Sequence[tuple[str, str]],
    evidence_dir: Path,
    expected_readback: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Save and bind AEDT's actual CLC definitions for the loaded expressions."""

    path = evidence_dir / "scgsim_epr_native_readback.clc"
    if path.exists():
        raise FileExistsError("native EPR expression readback already exists")
    reporter = app.post.fields_calculator.ofieldsreporter
    saved = reporter.SaveNamedExpressions(str(path), list(names), True)
    if saved is False:
        raise RuntimeError("HFSS native named-expression readback save returned false")
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError("HFSS native named-expression readback is missing or empty")
    payload = path.read_bytes()
    native_definitions = parse_native_expression_readback(payload)
    if set(native_definitions) != set(names):
        raise RuntimeError(
            "HFSS native expression readback names differ from the loaded definitions"
        )
    if expected_readback is not None and native_definitions != expected_readback:
        raise RuntimeError(
            "returned native CLC definitions differ from prepared readback"
        )
    for name, expected_block in definitions:
        expected_selectors = _field_selection_operations(expected_block)
        actual_selectors = native_definitions[name]["field_selection_operations"]
        if expected_selectors != actual_selectors:
            raise RuntimeError(
                f"HFSS native field selection readback differs for {name!r}"
            )
    return {
        "status": "native_clc_saved_and_field_selection_read_back",
        "api": "FieldsReporter.SaveNamedExpressions",
        "path": (Path("metadata/epr_expressions") / path.name).as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "definition_names": list(names),
        "definitions": native_definitions,
    }


def compile_named_expression(  # noqa: PLR0913 -- preserve the established field-expression keyword API.
    *,
    purpose: str,
    operations: Sequence[str],
    solution: str,
    phase_degrees: float,
    dependencies: Mapping[str, str],
    selection: Mapping[str, Any],
    namespace: str = "prepared",
    adjacent_selection_name: str | None = None,
) -> tuple[dict[str, Any], list[tuple[str, str]], dict[str, bytes]]:
    """Compile one immutable field definition without an AEDT call."""

    if solution != str(solution).strip() or not solution:
        raise ValueError("solution must be non-empty canonical text")
    if not isinstance(purpose, str) or not purpose:
        raise ValueError("purpose must be non-empty text")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", namespace):
        raise ValueError("expression namespace must be a safe native identifier")
    if isinstance(operations, (str, bytes)) or any(
        not isinstance(item, str) or not item or "\n" in item or "\0" in item
        for item in operations
    ):
        raise ValueError("expression operations must be canonical text entries")
    for key in ("object_name", "selection_name"):
        value = selection.get(key)
        if value is not None and (
            not isinstance(value, str)
            or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", value)
        ):
            raise ValueError(f"expression selection {key} is not a safe native name")
    identity = {
        "schema_version": "scgsim.aedt.epr-expression.v1",
        "namespace": namespace,
        "purpose": purpose,
        "operations": list(operations),
        "solution": solution,
        "phase_degrees": float(phase_degrees),
        "dependencies": dict(sorted(dependencies.items())),
        "selection": dict(selection),
    }
    identity["sha256"] = _digest(identity)
    # AEDT resolves one-file CLC dependencies by native-name order, not file order.
    ranks: dict[object, str] = {
        "surface_group": "20",
        "normalized_surface_total": "30",
        "custom_convergence": "30",
    }
    rank = ranks.get(selection.get("kind"), "10")
    name = f"scgsim_epr_{rank}_{identity['sha256'][:24]}"
    definitions: list[tuple[str, str]] = []
    evidence: dict[str, bytes] = {}
    if adjacent_selection_name is None:
        block = clc_block(name, operations)
        definitions.append((name, block))
        evidence[f"{name}.clc"] = block.encode("utf-8")
    else:
        if selection.get("selection_name") != adjacent_selection_name or list(
            operations[-3:]
        ) != [
            f"EnterAdjacentSurf('{adjacent_selection_name}')",
            "Operation('SurfaceValue')",
            "Operation('Integrate')",
        ]:
            raise ValueError("adjacent surface expression selection is inconsistent")
        integrand_name = f"scgsim_epr_00_{identity['sha256'][:24]}_integrand"
        integrand_block = clc_block(integrand_name, operations[:-3])
        final_block = clc_block(
            name,
            [
                f"NameOfExpression('{integrand_name}')",
                f"EnterAdjacentSurface('{adjacent_selection_name}')",
                "Operation('SurfaceValue')",
                "Operation('Integrate')",
            ],
            expression=(
                f"Integrate(AdjacentSurface({adjacent_selection_name}), "
                f"{integrand_name})"
            ),
        )
        definitions.extend(((integrand_name, integrand_block), (name, final_block)))
        evidence[f"{integrand_name}.clc"] = integrand_block.encode("utf-8")
        evidence[f"{name}.clc"] = (integrand_block + final_block).encode("utf-8")
    record = {
        "name": name,
        "identity": identity,
        "definition_sha256": hashlib.sha256(evidence[f"{name}.clc"]).hexdigest(),
        "definition_status": "authored_and_loaded",
        "solution": solution,
        "phase_degrees": float(phase_degrees),
    }
    return record, definitions, evidence
