"""Behavior checks for the standalone readable-style command."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "check_readable_style.py"


def _write(tmp_path: Path, name: str, source: str) -> Path:
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return path


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def test_clean_file_and_json_output(tmp_path: Path) -> None:
    """Accept a clean module and emit an empty JSON findings list."""
    path = _write(tmp_path, "clean.py", "import os\n\ndef run():\n    return os.name\n")
    result = _run(str(path), "--format", "json")
    assert result.returncode == 0
    assert json.loads(result.stdout) == []


def test_empty_python_file_is_valid(tmp_path: Path) -> None:
    """Treat an empty Python source file as a valid clean input."""
    path = _write(tmp_path, "empty.py", "")
    result = _run(str(path))
    assert result.returncode == 0
    assert result.stdout == "readable style: no findings\n"


@pytest.mark.parametrize(
    "statement",
    [
        "from package import _private",
        "from .package import _private",
        "from _private_package import public_name",
        "import _private_package",
        "import package._private_module",
    ],
)
def test_private_import_forms_are_reported(tmp_path: Path, statement: str) -> None:
    """Report private names across absolute and relative import forms."""
    path = _write(tmp_path, "module.py", statement + "\n")
    result = _run(str(path))
    assert result.returncode == 1
    assert "private-import" in result.stdout


def test_dunder_names_are_allowed(tmp_path: Path) -> None:
    """Allow standard dunder metadata names in imports."""
    path = _write(tmp_path, "dunder.py", "from package import __version__\n")
    assert _run(str(path)).returncode == 0


def test_double_underscore_non_dunder_name_is_private(tmp_path: Path) -> None:
    """Report double-underscore names that are not dunder names."""
    path = _write(tmp_path, "internal.py", "from package import __internal\n")
    result = _run(str(path))
    assert result.returncode == 1
    assert "private-import" in result.stdout


def test_size_limits_allow_exact_boundary_and_report_over_boundary(
    tmp_path: Path,
) -> None:
    """Accept the configured file limit and report the first line over it."""
    exact = _write(tmp_path, "exact.py", "\n".join("pass" for _ in range(4)) + "\n")
    over = _write(tmp_path, "over.py", "\n".join("pass" for _ in range(5)) + "\n")
    assert _run(str(exact), "--max-file-lines", "4").returncode == 0
    result = _run(str(over), "--max-file-lines", "4")
    assert result.returncode == 1
    assert "long-file" in result.stdout


def test_function_and_async_function_limits(tmp_path: Path) -> None:
    """Apply function line limits to synchronous and asynchronous functions."""
    body = "def regular():\n" + "    pass\n" * 3
    body += "async def asynchronous():\n" + "    pass\n" * 3
    path = _write(tmp_path, "functions.py", body)
    assert _run(str(path), "--max-function-lines", "4").returncode == 0
    result = _run(str(path), "--max-function-lines", "3", "--format", "json")
    findings = json.loads(result.stdout)
    assert result.returncode == 1
    assert [finding["rule"] for finding in findings] == [
        "long-function",
        "long-function",
    ]


def test_exception_requires_nonempty_comment_reason(tmp_path: Path) -> None:
    """Accept an exception comment only when it includes a reason."""
    allowed = _write(
        tmp_path, "allowed.py", "from x import _y  # private-import: upstream API\n"
    )
    blank = _write(tmp_path, "blank.py", "from x import _y  # private-import:\n")
    assert _run(str(allowed)).returncode == 0
    assert _run(str(blank)).returncode == 1


def test_lazy_import_suppression_requires_same_line_reason(tmp_path: Path) -> None:
    """Require a readable reason for explicitly suppressing PLC0415."""
    allowed = _write(
        tmp_path,
        "allowed.py",
        "import optional_package  # noqa: F401, PLC0415 - optional feature dependency\n",
    )
    blank = _write(tmp_path, "blank.py", "import optional_package  # noqa: PLC0415\n")
    empty_adjacent = _write(
        tmp_path,
        "empty_adjacent.py",
        "# \nimport optional_package  # noqa: PLC0415\n",
    )
    noqa_adjacent = _write(
        tmp_path,
        "noqa_adjacent.py",
        "# noqa: PLC0415\nimport optional_package  # noqa: PLC0415\n",
    )
    assert _run(str(allowed)).returncode == 0
    result = _run(str(blank), "--format", "json")
    assert result.returncode == 1
    assert json.loads(result.stdout)[0]["rule"] == "lazy-import-reason"
    assert _run(str(empty_adjacent)).returncode == 1
    assert _run(str(noqa_adjacent)).returncode == 1


def test_lazy_import_suppression_accepts_adjacent_reason_and_multiline_import(
    tmp_path: Path,
) -> None:
    """Attach a preceding reason to the import start line, including multiline."""
    source = (
        "# Optional feature dependency is loaded only by this operation.\n"
        "from optional_package import (  # noqa: PLC0415\n"
        "    one,\n"
        "    two,\n"
        ")\n"
    )
    path = _write(tmp_path, "multiline.py", source)
    assert _run(str(path)).returncode == 0


def test_uppercase_noqa_is_reported_when_ruff_accepts_it(tmp_path: Path) -> None:
    """Detect a missing reason even when Ruff accepts uppercase NOQA."""
    path = _write(
        tmp_path,
        "uppercase_noqa.py",
        "def render():\n    import optional_package  # NOQA: PLC0415\n",
    )
    ruff = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--select", "PLC0415", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert ruff.returncode == 0, ruff.stdout + ruff.stderr
    checker = _run(str(path), "--format", "json")
    assert checker.returncode == 1
    assert json.loads(checker.stdout)[0]["rule"] == "lazy-import-reason"


def test_indented_adjacent_explanation_and_exclusions(tmp_path: Path) -> None:
    """Accept an explanation at import indentation, not empty or noqa comments."""
    allowed = _write(
        tmp_path,
        "indented_allowed.py",
        "def render():\n    # Optional import used only for interactive display.\n"
        "    import optional_package  # noqa: PLC0415\n",
    )
    empty = _write(
        tmp_path,
        "indented_empty.py",
        "def render():\n    # \n    import optional_package  # noqa: PLC0415\n",
    )
    noqa_only = _write(
        tmp_path,
        "indented_noqa.py",
        "def render():\n    # noqa: PLC0415\n"
        "    import optional_package  # noqa: PLC0415\n",
    )
    assert _run(str(allowed)).returncode == 0
    assert _run(str(empty)).returncode == 1
    assert _run(str(noqa_only)).returncode == 1


def test_string_cannot_masquerade_as_exception_comment(tmp_path: Path) -> None:
    """Ignore directive text inside a string literal."""
    source = '"# private-import: reason"\nfrom x import _y\n'
    path = _write(tmp_path, "string.py", source)
    assert _run(str(path)).returncode == 1


def test_long_file_and_function_exceptions_need_comments(tmp_path: Path) -> None:
    """Honor reasoned comments attached to the relevant file and function."""
    file_exception = _write(
        tmp_path, "file.py", "# long-file: generated table\npass\npass\n"
    )
    func_exception = _write(
        tmp_path,
        "function.py",
        "def work():  # long-function: ordered kernel\n    pass\n    pass\n",
    )
    assert _run(str(file_exception), "--max-file-lines", "2").returncode == 0
    assert _run(str(func_exception), "--max-function-lines", "2").returncode == 0


@pytest.mark.parametrize(
    "source",
    [
        "def small():  # long-file: misleading local exception\n    pass\n    pass\n",
        "def outer():\n    # long-file: misleading nested exception\n    pass\n    pass\n",
    ],
)
def test_long_file_exception_must_be_a_top_level_comment(
    tmp_path: Path, source: str
) -> None:
    """Reject file exceptions placed inline or inside a function."""
    path = _write(tmp_path, "nested.py", source)
    result = _run(str(path), "--max-file-lines", "2", "--format", "json")
    assert result.returncode == 1
    assert json.loads(result.stdout)[0]["rule"] == "long-file"


def test_unicode_line_separators_inside_string_do_not_inflate_file_size(
    tmp_path: Path,
) -> None:
    """Count physical source lines instead of Unicode string separators."""
    path = _write(tmp_path, "unicode.py", 'message = "a\u2028b\u2028c"\n')
    assert _run(str(path), "--max-file-lines", "2").returncode == 0


def test_duplicate_inputs_are_reported_once(tmp_path: Path) -> None:
    """Deduplicate repeated identical file arguments."""
    path = _write(tmp_path, "bad.py", "from x import _y\n")
    result = _run(str(path), str(path), "--format", "json")
    assert result.returncode == 1
    assert len(json.loads(result.stdout)) == 1


def test_canonical_paths_are_reported_once(tmp_path: Path) -> None:
    """Deduplicate path aliases that resolve to the same source file."""
    path = _write(tmp_path, "bad.py", "from x import _y\n")
    (path.parent / "nested").mkdir()
    aliased_path = path.parent / "nested" / ".." / path.name
    result = _run(str(path), str(aliased_path), "--format", "json")
    assert result.returncode == 1
    findings = json.loads(result.stdout)
    assert len(findings) == 1
    assert findings[0]["path"] == str(path.resolve())


def test_missing_path_and_empty_directory_fail_explicitly(tmp_path: Path) -> None:
    """Fail with clear errors for missing paths and empty directories."""
    missing = _run(str(tmp_path / "missing.py"))
    empty = _run(str(tmp_path))
    assert missing.returncode == 2
    assert "does not exist" in missing.stderr
    assert empty.returncode == 2
    assert "no Python files" in empty.stderr


def test_syntax_error_fails_explicitly(tmp_path: Path) -> None:
    """Return an explicit error status for invalid Python syntax."""
    path = _write(tmp_path, "broken.py", "def broken(:\n")
    result = _run(str(path), "--format", "json")
    assert result.returncode == 2
    assert "SyntaxError" in result.stderr


def test_text_clean_output_and_invalid_limit(tmp_path: Path) -> None:
    """Print stable clean text and reject nonpositive line limits."""
    path = _write(tmp_path, "clean.py", "answer = 42\n")
    result = _run(str(path))
    invalid = _run(str(path), "--max-file-lines", "0")
    assert result.stdout == "readable style: no findings\n"
    assert result.returncode == 0
    assert invalid.returncode == 2
