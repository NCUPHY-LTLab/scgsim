#!/usr/bin/env python3
"""Run the selected HFSS documentation blocks through file-only preparation."""

from __future__ import annotations

import ast
import contextlib
import io
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CodeBlock:
    language: str
    source: str
    line: int


@dataclass(frozen=True)
class Selection:
    path: str
    heading: str | None
    ordinal: int
    marker: str


SELECTIONS = (
    Selection(
        "docs/course/aedt-inputs.qmd",
        "Make the shared source and mapping",
        1,
        "coupon.write_gds(source_gds, with_metadata=False)",
    ),
    Selection(
        "docs/course/aedt-eigenmode.qmd",
        None,
        1,
        "handoff = prepare_handoff(",
    ),
)


def _frontmatter_end(lines: list[str], path: Path) -> int:
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"{path}: expected YAML front matter at line 1")
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            return index + 1
    raise ValueError(f"{path}: unterminated YAML front matter")


def _fenced_blocks(lines: list[str], path: Path, start: int, end: int) -> list[CodeBlock]:
    blocks: list[CodeBlock] = []
    index = start
    while index < end:
        opening = re.fullmatch(r"```([A-Za-z0-9_-]*)\s*", lines[index])
        if opening is None:
            index += 1
            continue
        language = opening.group(1)
        block_line = index + 1
        index += 1
        body: list[str] = []
        while index < end and lines[index].strip() != "```":
            body.append(lines[index])
            index += 1
        if index == end:
            raise ValueError(f"{path}:{block_line}: unterminated code fence")
        blocks.append(CodeBlock(language, "\n".join(body), block_line))
        index += 1
    return blocks


def _select_blocks(selection: Selection, root: Path) -> list[CodeBlock]:
    path = root / selection.path
    lines = path.read_text(encoding="utf-8").splitlines()
    body_start = _frontmatter_end(lines, path)

    if selection.heading is None:
        start, end = body_start, len(lines)
    else:
        target = f"## {selection.heading}"
        heading_index = next(
            (index for index in range(body_start, len(lines)) if lines[index] == target),
            None,
        )
        if heading_index is None:
            raise ValueError(f"{path}: required section heading is missing: {target}")
        start = heading_index + 1
        end = len(lines)
        for index in range(start, len(lines)):
            heading = re.match(r"^(#{1,6})\s+", lines[index])
            if heading is not None and len(heading.group(1)) <= 2:
                end = index
                break

    blocks = _fenced_blocks(lines, path, start, end)
    if len(blocks) != selection.ordinal:
        raise ValueError(
            f"{path}: expected {selection.ordinal} fenced block(s) in the selected scope, "
            f"found {len(blocks)}; review the teaching structure before running"
        )
    block = blocks[selection.ordinal - 1]
    if block.language != "python":
        raise ValueError(f"{path}:{block.line}: selected block is not fenced as python")
    if selection.marker not in block.source:
        raise ValueError(f"{path}:{block.line}: expected cutoff marker is missing")
    _validate_syntax(block, path)
    return [block]


def _call_name(function: ast.expr) -> str | None:
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        parent = _call_name(function.value)
        return f"{parent}.{function.attr}" if parent else None
    return None


def _validate_syntax(block: CodeBlock, path: Path) -> None:
    try:
        compile(block.source, f"{path}:{block.line}", "exec")
    except SyntaxError as error:
        raise ValueError(f"{path}:{block.line}: selected Python block has invalid syntax") from error


def _validate_prepare_cutoff(block: CodeBlock, path: Path) -> None:
    tree = ast.parse(block.source, filename=f"{path}:{block.line}")
    prepare_assignments = [
        index
        for index, statement in enumerate(tree.body)
        if isinstance(statement, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "handoff" for target in statement.targets)
        and isinstance(statement.value, ast.Call)
        and _call_name(statement.value.func) == "prepare_handoff"
    ]
    if len(prepare_assignments) != 1:
        raise ValueError(f"{path}:{block.line}: expected one direct prepare_handoff assignment")
    prepare_index = prepare_assignments[0]
    if prepare_index != len(tree.body) - 2:
        raise ValueError(
            f"{path}:{block.line}: statements after prepare_handoff changed; review before running"
        )
    final_statement = tree.body[-1]
    print_call = (
        final_statement.value
        if isinstance(final_statement, ast.Expr)
        and isinstance(final_statement.value, ast.Call)
        and _call_name(final_statement.value.func) == "print"
        else None
    )
    if print_call is None:
        raise ValueError(f"{path}:{block.line}: expected the documented handoff-path print after preparation")


def _run(root: Path) -> dict[str, object]:
    selected: list[tuple[Selection, CodeBlock]] = []
    for selection in SELECTIONS:
        blocks = _select_blocks(selection, root)
        selected.append((selection, blocks[0]))

    _validate_prepare_cutoff(selected[-1][1], root / selected[-1][0].path)

    sys.dont_write_bytecode = True
    namespace: dict[str, object] = {"__name__": "__main__"}
    captured_output = io.StringIO()
    original_cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="scgsim-doc-example-") as temporary:
        work_dir = Path(temporary).resolve()
        try:
            os.chdir(work_dir)
            with contextlib.redirect_stdout(captured_output):
                for selection, block in selected:
                    filename = f"{selection.path}:{block.line}"
                    exec(compile(block.source, filename, "exec"), namespace, namespace)

            handoff = namespace.get("handoff")
            if handoff is None:
                raise RuntimeError("selected QMD blocks did not produce a handoff")
            output_files: dict[str, dict[str, int | str]] = {}
            for attribute in ("spec_path", "archive_path"):
                value = getattr(handoff, attribute)
                path = Path(value).resolve()
                if not path.is_relative_to(work_dir):
                    raise RuntimeError(f"prepared {attribute} escaped the temporary working directory")
                if not path.is_file():
                    raise RuntimeError(f"prepared {attribute} is not a file: {path}")
                output_files[attribute] = {"name": path.name, "bytes": path.stat().st_size}

            return {
                "status": "PREPARE_HANDOFF_COMPLETE",
                "selected_blocks": [
                    {"path": selection.path, "heading": selection.heading, "ordinal": selection.ordinal}
                    for selection, _block in selected
                ],
                "temporary_work_directory": work_dir.name,
                "temporary_outputs_removed_after_check": True,
                "prepared_files": output_files,
                "document_stdout": captured_output.getvalue().splitlines(),
                "native_execution": "not requested",
            }
        finally:
            os.chdir(original_cwd)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    try:
        result = _run(root)
    except ModuleNotFoundError as error:
        raise SystemExit(f"required documentation-example dependency is missing: {error.name}") from error
    import json

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
