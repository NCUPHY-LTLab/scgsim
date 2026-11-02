"""Check repository-wide readable Python rules that other tools miss.

Owns private imports across files and source length limits. It does not check
formatting, documentation coverage, call signatures, or package layers.
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import re
import sys
import tokenize
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

PRIVATE_IMPORT_MARK = "private-import:"
LONG_FILE_MARK = "long-file:"
LONG_FUNCTION_MARK = "long-function:"
LAZY_IMPORT_RULE = "PLC0415"


@dataclass(frozen=True)
class Finding:
    """One style violation with a stable location and explanation."""

    path: str
    line: int
    rule: str
    message: str


def main(argv: list[str] | None = None) -> int:
    """Parse inputs, inspect Python files, and print the requested report."""
    args = _parse_args(argv)
    paths = _resolve_python_files(args.paths)
    findings = _inspect(paths, args.max_file_lines, args.max_function_lines)
    _print_findings(findings, args.format)
    return 1 if findings else 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--max-file-lines", type=_positive_integer, default=800)
    parser.add_argument("--max-function-lines", type=_positive_integer, default=150)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    return parser.parse_args(argv)


def _positive_integer(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _resolve_python_files(inputs: list[Path]) -> list[Path]:
    files: set[Path] = set()
    for path in inputs:
        if not path.exists():
            raise ValueError(f"path does not exist: {path}")
        if path.is_dir():
            files.update(
                candidate.resolve()
                for candidate in path.rglob("*.py")
                if candidate.is_file()
            )
        elif path.is_file() and path.suffix == ".py":
            files.add(path.resolve())
        else:
            raise ValueError(f"expected a Python file or directory: {path}")
    if not files:
        raise ValueError("no Python files found")
    return sorted(files)


def _inspect(
    paths: Iterable[Path], file_limit: int, function_limit: int
) -> list[Finding]:
    findings: list[Finding] = []
    for path in paths:
        findings.extend(_inspect_file(path, file_limit, function_limit))
    return findings


def _inspect_file(path: Path, file_limit: int, function_limit: int) -> list[Finding]:
    source = path.read_text(encoding="utf-8")
    line_count = _physical_line_count(source)
    tree = ast.parse(source, filename=str(path))
    comments = _comment_lines(source)
    findings = _lazy_import_reason_findings(path, tree, comments)
    findings.extend(_private_import_findings(path, tree, comments))
    findings.extend(_long_file_findings(path, line_count, comments, file_limit))
    findings.extend(_long_function_findings(path, tree, comments, function_limit))
    return findings


def _lazy_import_reason_findings(
    path: Path, tree: ast.Module, comments: dict[int, tuple[str, int]]
) -> list[Finding]:
    """Require a reason when an import suppresses Ruff's PLC0415 rule."""
    findings: list[Finding] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        comment = comments.get(node.lineno)
        if comment is None or not _suppresses_rule(comment[0], LAZY_IMPORT_RULE):
            continue
        if _lazy_import_has_reason(comment[0]):
            continue
        previous_comment = comments.get(node.lineno - 1)
        if (
            previous_comment
            and previous_comment[1] == node.col_offset
            and previous_comment[0].lstrip("#").strip()
            and not _is_noqa_only_comment(previous_comment[0])
        ):
            continue
        findings.append(
            Finding(
                str(path),
                node.lineno,
                "lazy-import-reason",
                "PLC0415 suppression needs a same-line reason or preceding explanation comment",
            )
        )
    return findings


def _is_noqa_only_comment(comment: str) -> bool:
    """Reject a preceding suppression directive as an explanation."""
    content = comment.lstrip("#").strip()
    return bool(
        re.fullmatch(
            r"noqa(?:\s*:\s*[A-Z]+\d{4}(?:\s*,\s*[A-Z]+\d{4})*)?",
            content,
            flags=re.IGNORECASE,
        )
    )


def _suppresses_rule(comment: str, rule: str) -> bool:
    directive = _noqa_directive(comment)
    if directive is None:
        return False
    if not directive.startswith(":"):
        return True
    codes = directive[1:].split("#", maxsplit=1)[0]
    codes = re.split(r"\s+-{1,2}\s+", codes, maxsplit=1)[0]
    return any(code.strip().upper() == rule for code in codes.split(","))


def _lazy_import_has_reason(comment: str) -> bool:
    directive = _noqa_directive(comment)
    if directive is None or not directive.startswith(":"):
        return False
    content = directive[1:].strip()
    reason_separator = re.search(r"\s+(?:--?|—|–)\s+|\s+#\s*", content)
    if reason_separator is None:
        return False
    codes = content[: reason_separator.start()]
    reason = content[reason_separator.end() :].strip()
    has_lazy_import_code = any(
        code.strip().upper() == LAZY_IMPORT_RULE for code in codes.split(",")
    )
    return has_lazy_import_code and bool(reason)


def _noqa_directive(comment: str) -> str | None:
    match = re.search(r"\bnoqa\b", comment, flags=re.IGNORECASE)
    return None if match is None else comment[match.end() :]


def _physical_line_count(source: str) -> int:
    newline_count = source.count("\n") + source.count("\r") - source.count("\r\n")
    ends_with_newline = source.endswith(("\n", "\r"))
    return newline_count + bool(source and not ends_with_newline)


def _comment_lines(source: str) -> dict[int, tuple[str, int]]:
    comments: dict[int, tuple[str, int]] = {}
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            comments[token.start[0]] = (token.string, token.start[1])
    return comments


def _has_reason(comment: str | None, marker: str) -> bool:
    if comment is None:
        return False
    _, separator, reason = comment.partition(marker)
    return bool(separator and reason.strip())


def _private_import_findings(
    path: Path, tree: ast.Module, comments: dict[int, tuple[str, int]]
) -> list[Finding]:
    findings: list[Finding] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module_names = [node.module or ""]
            imported_names = [alias.name for alias in node.names]
        elif isinstance(node, ast.Import):
            module_names = [alias.name for alias in node.names]
            imported_names = []
        else:
            continue
        comment = comments.get(node.lineno)
        if comment and _has_reason(comment[0], PRIVATE_IMPORT_MARK):
            continue
        names = [*module_names, *imported_names]
        for name in names:
            components = name.split(".")
            private_name = next(
                (part for part in components if _is_private(part)), None
            )
            if private_name is None:
                continue
            findings.append(
                Finding(
                    str(path),
                    node.lineno,
                    "private-import",
                    f"imports private name {private_name!r}; shared names must not start with an underscore",
                )
            )
    return findings


def _is_private(name: str) -> bool:
    is_dunder = name.startswith("__") and name.endswith("__") and len(name) > 4
    return name.startswith("_") and not is_dunder


def _long_file_findings(
    path: Path, line_count: int, comments: dict[int, tuple[str, int]], limit: int
) -> list[Finding]:
    if line_count <= limit:
        return []
    if any(
        column == 0 and _has_reason(comment, LONG_FILE_MARK)
        for line_number, (comment, column) in comments.items()
        if line_number <= 10
    ):
        return []
    return [
        Finding(
            str(path), 1, "long-file", f"file has {line_count} lines; limit is {limit}"
        )
    ]


def _long_function_findings(
    path: Path, tree: ast.Module, comments: dict[int, tuple[str, int]], limit: int
) -> list[Finding]:
    findings: list[Finding] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        length = (node.end_lineno or node.lineno) - node.lineno + 1
        comment = comments.get(node.lineno)
        has_function_reason = comment and _has_reason(comment[0], LONG_FUNCTION_MARK)
        if length <= limit or has_function_reason:
            continue
        findings.append(
            Finding(
                str(path),
                node.lineno,
                "long-function",
                f"{node.name} has {length} lines; limit is {limit}",
            )
        )
    return findings


def _print_findings(findings: list[Finding], output_format: str) -> None:
    if output_format == "json":
        print(json.dumps([asdict(finding) for finding in findings], sort_keys=True))
    elif findings:
        for finding in findings:
            print(f"{finding.path}:{finding.line}: {finding.rule}: {finding.message}")
    else:
        print("readable style: no findings")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, SyntaxError, ValueError, tokenize.TokenError) as error:
        print(
            f"readable style: error: {type(error).__name__}: {error}", file=sys.stderr
        )
        sys.exit(2)
