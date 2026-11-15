"""Q3D convergence result reader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scgsim.aedt.results.convergence.common import ConvergenceProblem, read_convergence
from scgsim.aedt.specs.q3d import Q3dSpec


def read_q3d_convergence(run_dir: Path, spec: Q3dSpec) -> dict[str, Any]:
    """Parse and bind the capacitance convergence from AEDT 2024.2 files; an AC-RL block is refused."""
    return read_convergence(
        run_dir,
        spec,
        solver="Q3D",
        problems=(ConvergenceProblem("capacitance", "CapConv", "delta"),),
        forbidden_block_names=("ACRLConv",),
    )
