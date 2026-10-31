"""Q2D convergence result reader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scgsim.aedt.results.convergence.common import (
    ConvergenceProblem,
    read_matrix_convergence,
)
from scgsim.aedt.specs.q2d import Q2dSpec


def read_q2d_convergence(run_dir: Path, spec: Q2dSpec) -> dict[str, Any]:
    """Parse and bind CG/RL convergence from AEDT 2024.2 result files."""
    return read_matrix_convergence(
        run_dir,
        spec,
        solver="Q2D",
        problems=(
            ConvergenceProblem("cg", "CGConv", "de", "ee"),
            ConvergenceProblem("rl", "RLConv", "de", "ee"),
        ),
    )
