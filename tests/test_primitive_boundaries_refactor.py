"""Regression tests for optional primitive dependency boundaries."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_tagged_grid_loads_optional_shapely_before_empty_input_return():
    """Keep the dependency failure ahead of tagged-grid input handling."""
    repository = Path(__file__).parents[1]
    source_root = repository / "src"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source_root)
    program = """
import sys
from importlib.abc import MetaPathFinder

class MissingShapely(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "shapely" or fullname.startswith("shapely."):
                raise ModuleNotFoundError(
                    f"No module named '{fullname}'", name=fullname
                )
        return None

sys.meta_path.insert(0, MissingShapely())
from scgsim.geometry.primitives.tagged_grid import build_tagged_subdivision

try:
    build_tagged_subdivision((), {}, 1.0)
except ModuleNotFoundError as error:
    assert error.name == "shapely"
else:
    raise AssertionError("tagged-grid construction returned without Shapely")
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
