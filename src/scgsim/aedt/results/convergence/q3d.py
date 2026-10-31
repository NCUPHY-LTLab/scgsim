"""Q3D convergence result reader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scgsim.aedt.results.convergence.common import (
    ConvergenceProblem,
    read_matrix_convergence,
)
from scgsim.aedt.specs.q3d import Q3dSpec


def read_q3d_convergence(run_dir: Path, spec: Q3dSpec) -> dict[str, Any]:
    """Parse and bind capacitance/AC-RL convergence from AEDT 2024.2 files."""
    problems = [ConvergenceProblem("capacitance", "CapConv", "delta", None)]
    if spec.solve_ac_rl:
        problems.append(ConvergenceProblem("ac_rl", "ACRLConv", "delta", None))
    return read_matrix_convergence(
        run_dir,
        spec,
        solver="Q3D",
        problems=tuple(problems),
        forbidden_block_names=() if spec.solve_ac_rl else ("ACRLConv",),
    )
