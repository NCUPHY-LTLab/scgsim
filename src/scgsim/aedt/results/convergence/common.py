"""Shared strict convergence-report parsing used by Q2D and Q3D."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from scgsim.aedt.results.convergence.files import (
    contained_evidence_path,
    evidence_source,
    read_evidence_text,
)
from scgsim.aedt.specs.common import MatrixRunControl
from scgsim.aedt.specs.q2d import Q2dSpec
from scgsim.aedt.specs.q3d import Q3dSpec


def read_convergence(
    run_dir: Path,
    spec: Q2dSpec | Q3dSpec,
    *,
    solver: str,
    problems: tuple[ConvergenceProblem, ...],
    forbidden_block_names: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Read exact matrix convergence blocks and bind their native profile status."""
    root = run_dir.resolve()
    results_dir = contained_evidence_path(
        root, f"{spec.project_name}.aedtresults", solver
    )
    asol_path = contained_evidence_path(results_dir, f"{spec.design_name}.asol", solver)
    asol = read_evidence_text(asol_path, solver)
    profile_names = set(re.findall(r"\bPRF\([^\n]*\bFile='([^']+\.profile)'", asol))
    if len(profile_names) != 1:
        raise RuntimeError(
            f"{solver} native solution does not identify exactly one profile"
        )
    profile_path = contained_evidence_path(
        results_dir / f"{spec.design_name}.results", profile_names.pop(), solver
    )
    profile = read_evidence_text(profile_path, solver)

    block_names = {problem.block_name for problem in problems}
    blocks: dict[str, _SolutionBlock] = {}
    found_names: set[str] = set()
    for match in re.finditer(
        r"(?ms)^\s*\$begin '(?P<id>\d+)'\s*$"
        r"(?P<body>.*?)^\s*\$end '(?P=id)'\s*$",
        asol,
    ):
        body = match.group("body")
        name = _optional(body, r"(?m)^\s*ConvSetupName='([^']+)'\s*$")
        if name is not None:
            found_names.add(name)
        if name in block_names:
            if name in blocks:
                raise RuntimeError(f"{solver} native solution repeats {name}")
            blocks[name] = _SolutionBlock(match.group("id"), body)
    if set(blocks) != block_names:
        raise RuntimeError(
            f"{solver} native solution lacks exact matrix convergence blocks"
        )
    if set(forbidden_block_names) & found_names:
        raise RuntimeError(
            f"{solver} native solution contains a disabled convergence block"
        )

    control = cast(MatrixRunControl, spec.run_control)
    evidence = {
        problem.result_key: _problem_evidence(
            solver, blocks[problem.block_name], profile, problem, control
        )
        for problem in problems
    }
    return {
        "sources": {
            "asol": evidence_source(asol_path, root),
            "profile": evidence_source(profile_path, root),
        },
        **evidence,
    }


def _problem_evidence(
    solver: str,
    block: _SolutionBlock,
    profile: str,
    problem: ConvergenceProblem,
    control: MatrixRunControl,
) -> dict[str, Any]:
    target = _float(_required(block.body, r"(?m)^\s*ConvTarget='([^']+)'\s*$"))
    maximum_passes = _int(_required(block.body, r"(?m)^\s*MaxPasses='([^']+)'\s*$"))
    if not math.isclose(
        target, control.convergence_percent, rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError(
            f"{solver} native convergence target does not match the spec"
        )
    if maximum_passes != control.maximum_passes:
        raise RuntimeError(
            f"{solver} native convergence maximum passes do not match the spec"
        )

    pass_rows = re.findall(r"(?m)^\s*c\((.*)\)\s*$", block.body)
    if not pass_rows:
        raise RuntimeError(f"{solver} native convergence block has no adaptive passes")
    final = pass_rows[-1]
    final_pass = _int(_field(final, "p"))
    final_triangles = _int(_field(final, "tri"))
    final_delta = _float(_field(final, problem.delta_field))
    final_error = (
        _float(_field(final, problem.error_field)) if problem.error_field else None
    )
    if (
        final_pass <= 0
        or final_pass > maximum_passes
        or final_triangles <= 0
        or final_delta < 0.0
        or (final_error is not None and final_error < 0.0)
    ):
        raise RuntimeError(f"{solver} native final convergence values are invalid")

    profile_block = _block(profile, block.block_id, solver)
    converged_text = "Adaptive Passes converged"
    not_converged_text = "Adaptive Passes did not converge"
    converged_count = profile_block.count(converged_text)
    not_converged_count = profile_block.count(not_converged_text)
    if (converged_count, not_converged_count) == (1, 0):
        converged = True
        stop_reason = converged_text
    elif (converged_count, not_converged_count) == (0, 1):
        converged = False
        stop_reason = not_converged_text
    else:
        raise RuntimeError(f"{solver} native profile has ambiguous convergence status")

    result = {
        "target_percent": target,
        "converged": converged,
        "stop_reason": stop_reason,
        "final_pass": final_pass,
        "final_matrix_delta_percent": final_delta,
        "final_triangle_count": final_triangles,
    }
    if final_error is not None:
        result["final_error_percent"] = final_error
    return result


def _block(text: str, name: str, solver: str) -> str:
    begin = f"$begin '{name}'"
    end = f"$end '{name}'"
    if text.count(begin) != 1 or text.count(end) != 1:
        raise RuntimeError(f"{solver} native profile lacks one exact {name} block")
    return text.split(begin, 1)[1].split(end, 1)[0]


def _required(text: str, pattern: str) -> str:
    values = re.findall(pattern, text)
    if len(values) != 1:
        raise RuntimeError(
            "AEDT native matrix convergence field is missing or repeated"
        )
    return values[0]


def _optional(text: str, pattern: str) -> str | None:
    values = re.findall(pattern, text)
    if len(values) > 1:
        raise RuntimeError("AEDT native matrix convergence field is repeated")
    return values[0] if values else None


def _field(row: str, name: str) -> str:
    return _required(row, rf"(?:^|,\s*){re.escape(name)}=([^,\s)]+)")


def _float(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise RuntimeError(
            "AEDT native matrix convergence value is not numeric"
        ) from exc
    if not math.isfinite(result):
        raise RuntimeError("AEDT native matrix convergence value is not finite")
    return result


def _int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise RuntimeError(
            "AEDT native matrix convergence value is not an integer"
        ) from exc
    return result


@dataclass(frozen=True)
class ConvergenceProblem:
    """One solver quantity and its exact native convergence-block field names."""

    result_key: str
    block_name: str
    delta_field: str
    error_field: str | None = None


@dataclass(frozen=True)
class _SolutionBlock:
    """A native convergence block with its profile-binding identity."""

    block_id: str
    body: str
