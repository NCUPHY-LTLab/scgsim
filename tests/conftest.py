"""Explicit opt-in for owner-supplied, read-only private golden fixtures."""

from __future__ import annotations

import re
from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("private golden")
    group.addoption("--private-golden-manifest", metavar="PATH", help="Owner-authorized external fixture manifest")
    group.addoption("--run-private-golden", action="store_true", help="Require private golden execution with an explicit manifest")


def pytest_configure(config: pytest.Config) -> None:
    if config.getoption("run_private_golden") and not config.getoption("private_golden_manifest"):
        raise pytest.UsageError("private_golden NOT_RUN: --run-private-golden requires --private-golden-manifest PATH")


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    private = [item for item in items if item.get_closest_marker("private_golden")]
    manifest = config.getoption("private_golden_manifest")
    explicitly_selected = re.search(r"\bprivate_golden\b", config.option.markexpr or "")
    if private and explicitly_selected and not manifest:
        raise pytest.UsageError("private_golden NOT_RUN: selecting private_golden requires --private-golden-manifest PATH")
    if not manifest:
        for item in private:
            item.add_marker(pytest.mark.skip(reason="NOT_RUN: no owner-authorized private fixture manifest supplied"))


@pytest.fixture
def private_golden_manifest(request: pytest.FixtureRequest) -> Path:
    value = request.config.getoption("private_golden_manifest")
    if not value:
        pytest.skip("NOT_RUN: no owner-authorized private fixture manifest supplied")
    return Path(value).expanduser().resolve()
