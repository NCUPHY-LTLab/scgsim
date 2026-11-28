"""Behavior checks for the unified quality command."""

from __future__ import annotations

import importlib.util
import shlex
import subprocess
import sys
import venv
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "check_quality.py"
SPEC = importlib.util.spec_from_file_location("check_quality", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
QUALITY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = QUALITY
SPEC.loader.exec_module(QUALITY)


def _tracked(root: Path, output: bytes, monkeypatch: pytest.MonkeyPatch) -> None:
    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        if command[:2] == ["git", "rev-parse"]:
            return subprocess.CompletedProcess(command, 0, str(root).encode(), b"")
        if command[:2] == ["git", "ls-files"]:
            return subprocess.CompletedProcess(command, 0, output, b"")
        raise AssertionError(command)

    monkeypatch.setattr(QUALITY.subprocess, "run", run)


def test_repository_root_mismatch_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject a checkout whose Git root differs from the requested root."""
    expected = tmp_path / "child"
    expected.mkdir()
    parent = tmp_path.resolve()
    monkeypatch.setattr(
        QUALITY.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, f"{parent}\n", ""),
    )
    assert QUALITY._resolve_repository(expected) is None  # noqa: SLF001 -- test root validation directly


def test_empty_tracked_inventory_is_explicit_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail clearly when Git returns no tracked Python paths."""
    _tracked(tmp_path, b"", monkeypatch)
    monkeypatch.setattr(QUALITY, "_resolve_repository", lambda root: tmp_path)
    assert QUALITY.main(["--root", str(tmp_path)]) == 1


def test_inventory_preserves_whitespace_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Decode NUL-separated Git inventory without splitting path spaces."""
    _tracked(tmp_path, b"src/file with spaces.py\0tests/added.py\0", monkeypatch)
    files = QUALITY._tracked_python_files(tmp_path)  # noqa: SLF001 -- test inventory decoding directly
    assert files == [Path("src/file with spaces.py"), Path("tests/added.py")]


def test_commands_use_current_python_complete_inventory_and_src(tmp_path: Path) -> None:
    """Build each check with the active interpreter and its intended scope."""
    files = [Path("src/a file.py"), Path("tests/test_a.py")]
    checks = QUALITY._quality_checks(files)  # noqa: SLF001 -- inspect generated argv precisely
    python = QUALITY.sys.executable
    assert checks[0].command == (
        python,
        "-m",
        "ruff",
        "format",
        "--check",
        "--",
        "src/a file.py",
        "tests/test_a.py",
    )
    assert checks[1].command[:6] == (
        python,
        "-m",
        "ruff",
        "check",
        "--",
        "src/a file.py",
    )
    assert checks[2].command[:4] == (
        python,
        "scripts/check_readable_style.py",
        "--",
        "src/a file.py",
    )
    assert checks[3].command == (
        python,
        "-m",
        "pyright",
        "--pythonpath",
        python,
        "src",
    )
    assert checks[4].command == (
        str(QUALITY._script_executable("lint-imports")),  # noqa: SLF001 -- inspect the generated tool path directly
    )


@pytest.mark.parametrize(
    ("platform", "filename"),
    [("linux", "lint-imports"), ("win32", "lint-imports.exe")],
)
def test_import_linter_uses_active_environment_scripts_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, platform: str, filename: str
) -> None:
    """Resolve the pinned import-linter executable in platform scripts layout."""
    monkeypatch.setattr(QUALITY.sysconfig, "get_path", lambda key: str(tmp_path))
    assert QUALITY._script_executable(  # noqa: SLF001 -- exercise platform path selection directly
        "lint-imports", platform=platform
    ) == (tmp_path / filename)


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="the fake console launcher in this integration test uses a POSIX shell",
)
def test_real_virtual_environment_keeps_its_interpreter_and_accepts_option_like_paths(
    tmp_path: Path,
) -> None:
    """Keep venv identity and safely pass a tracked path beginning with a dash."""
    environment = tmp_path / "venv with spaces"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
    python = environment / "bin" / "python"
    site_packages = Path(
        subprocess.run(
            [str(python), "-c", "import site; print(site.getsitepackages()[0])"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    for module_name in ("ruff", "pyright"):
        module = site_packages / module_name
        module.mkdir()
        (module / "__main__.py").write_text(
            "import sys\nprint(sys.prefix, file=sys.stderr)\n", encoding="utf-8"
        )

    repository = tmp_path / "repo with spaces"
    (repository / "src").mkdir(parents=True)
    (repository / "scripts").mkdir()
    (repository / "--module.py").write_text("value = 1\n", encoding="utf-8")
    (repository / "scripts" / "check_readable_style.py").write_bytes(
        (Path(__file__).parents[1] / "scripts" / "check_readable_style.py").read_bytes()
    )
    tool = environment / "bin" / "lint-imports"
    tool.write_text(
        "#!/bin/sh\n"
        f"exec {shlex.quote(str(python))} -c 'import sys; print(sys.prefix)'\n",
        encoding="utf-8",
    )
    tool.chmod(0o755)
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(["git", "-C", str(repository), "add", "--all"], check=True)

    result = subprocess.run(
        [str(python), str(SCRIPT), "--root", str(repository)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count(str(environment)) == 1
    assert result.stderr.count(str(environment)) == 3


def test_checks_continue_and_report_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Run later checks after a failure and return a failing summary."""
    commands: list[tuple[str, ...]] = []

    def run(command: tuple[str, ...], **kwargs: object) -> SimpleNamespace:
        """Record invocations and fail only the second check."""
        commands.append(command)
        return SimpleNamespace(returncode=2 if len(commands) == 2 else 0)

    checks = [QUALITY.Check(str(index), (f"tool-{index}",)) for index in range(3)]
    monkeypatch.setattr(QUALITY.subprocess, "run", run)
    assert QUALITY._run_checks(tmp_path, checks) == 1  # noqa: SLF001 -- isolate continuation behavior
    assert len(commands) == 3
    assert "2/3 passed" in capsys.readouterr().out


def test_missing_tool_fails_but_later_checks_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Report a missing executable while continuing with available checks."""
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        QUALITY.subprocess,
        "run",
        lambda command, **kwargs: (
            calls.append(command) or SimpleNamespace(returncode=0)
        ),
    )
    checks = [
        QUALITY.Check("missing", ("not-installed",), tmp_path / "missing"),
        QUALITY.Check("present", ("present",)),
    ]
    assert QUALITY._run_checks(tmp_path, checks) == 1  # noqa: SLF001 -- isolate missing-tool behavior
    assert calls == [("present",)]
    assert "missing tool" in capsys.readouterr().out


def test_list_only_prints_plan_without_running_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Show the command plan without launching any quality check."""
    _tracked(tmp_path, b"src/one.py\0", monkeypatch)
    monkeypatch.setattr(QUALITY, "_resolve_repository", lambda root: tmp_path)
    monkeypatch.setattr(
        QUALITY, "_tracked_python_files", lambda root: [Path("src/one.py")]
    )
    monkeypatch.setattr(
        QUALITY.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("list mode executed a command"),
    )
    assert QUALITY.main(["--root", str(tmp_path), "--list"]) == 0
    output = capsys.readouterr().out
    assert "none were executed" in output
    assert "pyright --pythonpath" in output
    assert " src" in output


@pytest.mark.parametrize(("codes", "expected"), [([0, 0, 0], 0), ([0, 1, 0], 1)])
def test_cli_exit_code_reflects_check_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, codes: list[int], expected: int
) -> None:
    """Return success only when every configured quality check succeeds."""
    monkeypatch.setattr(QUALITY, "_resolve_repository", lambda root: tmp_path)
    monkeypatch.setattr(QUALITY, "_tracked_python_files", lambda root: [Path("one.py")])
    monkeypatch.setattr(
        QUALITY,
        "_quality_checks",
        lambda files: [QUALITY.Check(str(index), (str(index),)) for index in codes],
    )
    results = iter(codes)
    monkeypatch.setattr(
        QUALITY.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=next(results)),
    )
    assert QUALITY.main(["--root", str(tmp_path)]) == expected
