"""Explicit opt-in for owner-supplied, read-only private golden fixtures."""

from __future__ import annotations

import re
from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register explicit external-manifest selection for private data checks."""
    group = parser.getgroup("private golden")
    group.addoption(
        "--private-golden-manifest",
        metavar="PATH",
        help="Owner-authorized external fixture manifest",
    )
    group.addoption(
        "--run-private-golden",
        action="store_true",
        help="Require private golden execution with an explicit manifest",
    )
    parser.addoption(
        "--fail-on-skip",
        action="store_true",
        help="Fail the run when any test or module is skipped",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Reject requested private execution without an explicit manifest."""
    if config.getoption("run_private_golden") and not config.getoption(
        "private_golden_manifest"
    ):
        raise pytest.UsageError(
            "private_golden NOT_RUN: --run-private-golden requires --private-golden-manifest PATH"
        )


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Keep private checks skipped unless the caller supplies their manifest."""
    private = [item for item in items if item.get_closest_marker("private_golden")]
    manifest = config.getoption("private_golden_manifest")
    explicitly_selected = re.search(r"\bprivate_golden\b", config.option.markexpr or "")
    if private and explicitly_selected and not manifest:
        raise pytest.UsageError(
            "private_golden NOT_RUN: selecting private_golden requires --private-golden-manifest PATH"
        )
    if not manifest:
        for item in private:
            item.add_marker(
                pytest.mark.skip(
                    reason="NOT_RUN: no owner-authorized private fixture manifest supplied"
                )
            )


def pytest_sessionfinish(session: pytest.Session) -> None:
    """Turn any skip into a failure when the caller asked for a complete run.

    CI installs every dependency a selected test needs, so a skip there means a
    missing or broken dependency, which would otherwise pass unnoticed.
    """
    if not session.config.getoption("fail_on_skip"):
        return
    if _skipped_reports(session.config) and session.exitstatus == pytest.ExitCode.OK:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_terminal_summary(
    terminalreporter: pytest.TerminalReporter, config: pytest.Config
) -> None:
    """Say why a run with skips failed under --fail-on-skip."""
    skipped = _skipped_reports(config)
    if config.getoption("fail_on_skip") and skipped:
        terminalreporter.write_sep(
            "=",
            f"--fail-on-skip: {len(skipped)} skipped, so this run fails (-rs lists them)",
            red=True,
        )


def _skipped_reports(config: pytest.Config) -> list[object]:
    reporter = config.pluginmanager.get_plugin("terminalreporter")
    return list(reporter.stats.get("skipped", [])) if reporter is not None else []


@pytest.fixture
def private_golden_manifest(request: pytest.FixtureRequest) -> Path:
    """Resolve the caller's manifest or skip the unselected private check."""
    value = request.config.getoption("private_golden_manifest")
    if not value:
        pytest.skip("NOT_RUN: no owner-authorized private fixture manifest supplied")
    return Path(value).expanduser().resolve()
