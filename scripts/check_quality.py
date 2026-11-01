"""Run the repository's Python quality checks over tracked source files.

This command owns inventory selection and check reporting. It does not run
tests, solver workflows, or provide substitute checks when a tool is absent.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Check:
    """One quality command and the executable needed to start it."""

    name: str
    command: tuple[str, ...]
    executable: Path | None = None


def main(argv: list[str] | None = None) -> int:
    """Validate the repository, build the scope, and report every check."""
    arguments = _parse_arguments(argv)
    repository = _resolve_repository(arguments.root)
    if repository is None:
        return 2

    python_files = _tracked_python_files(repository)
    if python_files is None:
        return 2
    if not python_files:
        print("quality inventory: no tracked Python files", file=sys.stderr)
        return 1

    checks = _quality_checks(python_files)
    if arguments.list_only:
        _print_plan(repository, checks)
        return 0
    return _run_checks(repository, checks)


def _parse_arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        help="repository root (defaults to this script's repository)",
    )
    parser.add_argument(
        "--list",
        dest="list_only",
        action="store_true",
        help="show commands without running them",
    )
    return parser.parse_args(argv)


def _resolve_repository(root: Path | None) -> Path | None:
    expected_root = (root or Path(__file__).resolve().parents[1]).resolve()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=expected_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as error:
        print(f"repository check could not start: {error}", file=sys.stderr)
        return None
    if result.returncode:
        print(f"repository check failed: {result.stderr.strip()}", file=sys.stderr)
        return None
    actual_root = Path(result.stdout.strip()).resolve()
    if actual_root != expected_root:
        print(
            f"repository root mismatch: expected {expected_root}, found {actual_root}",
            file=sys.stderr,
        )
        return None
    return expected_root


def _tracked_python_files(repository: Path) -> list[Path] | None:
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z", "--", "*.py"],
            cwd=repository,
            capture_output=True,
            check=False,
        )
    except OSError as error:
        print(f"Python inventory could not start: {error}", file=sys.stderr)
        return None
    if result.returncode:
        message = result.stderr.decode(errors="replace").strip()
        print(f"Python inventory failed: {message}", file=sys.stderr)
        return None
    return [Path(os.fsdecode(item)) for item in result.stdout.split(b"\0") if item]


def _quality_checks(python_files: list[Path]) -> list[Check]:
    python = sys.executable
    files = tuple(str(path) for path in python_files)
    lint_imports = Path(sys.prefix) / "bin" / "lint-imports"
    return [
        Check("Ruff format", (python, "-m", "ruff", "format", "--check", "--", *files)),
        Check("Ruff check", (python, "-m", "ruff", "check", "--", *files)),
        Check(
            "Readable style", (python, "scripts/check_readable_style.py", "--", *files)
        ),
        Check("Pyright", (python, "-m", "pyright", "--pythonpath", python, "src")),
        Check("Import layers", (str(lint_imports),), lint_imports),
    ]


def _print_plan(repository: Path, checks: list[Check]) -> None:
    print(f"Repository: {repository}")
    print("Checks listed only; none were executed.")
    for check in checks:
        print(f"{check.name}: {_display_command(check.command)}")


def _display_command(command: tuple[str, ...]) -> str:
    return shlex.join(command)


def _run_checks(repository: Path, checks: list[Check]) -> int:
    failures = 0
    for check in checks:
        if check.executable is not None and not check.executable.is_file():
            print(f"{check.name}: FAILED (missing tool: {check.executable})")
            failures += 1
            continue
        try:
            result = subprocess.run(check.command, cwd=repository, check=False)
        except OSError as error:
            print(f"{check.name}: FAILED (could not start: {error})")
            failures += 1
            continue
        status = (
            "passed" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
        )
        print(f"{check.name}: {status}")
        failures += result.returncode != 0
    print(f"Quality checks: {len(checks) - failures}/{len(checks)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
