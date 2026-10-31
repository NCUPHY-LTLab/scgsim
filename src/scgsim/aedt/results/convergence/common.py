"""Read Q2D and Q3D matrix convergence from AEDT's native solution files.

The ``.asol`` file holds one block per convergence problem (for example Q3D
capacitance), and the ``.profile`` file says whether the adaptive passes converged.
Owns the parsing and the match against the spec's run control. It does not own
HFSS, whose files differ (``convergence/hfss.py``).
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, NamedTuple

from scgsim.aedt.results.convergence._evidence_files import (
    contained_evidence_path,
    read_evidence_text,
    source_record,
)
from scgsim.aedt.specs.q2d import Q2dSpec
from scgsim.aedt.specs.q3d import Q3dSpec


class ConvergenceProblem(NamedTuple):
    """One adaptive problem to read, such as Q3D capacitance.

    ``block_name`` is the ``ConvSetupName`` of its block in the ``.asol`` file.
    ``delta_field`` and ``error_field`` name the pass-row fields holding the matrix
    change and the error; ``error_field`` is ``None`` when the solver has none.
    """

    key: str
    block_name: str
    delta_field: str
    error_field: str | None


class _NativeBlock(NamedTuple):
    block_id: str
    body: str


def read_matrix_convergence(
    run_dir: Path,
    spec: Q2dSpec | Q3dSpec,
    *,
    solver: str,
    problems: tuple[ConvergenceProblem, ...],
    forbidden_block_names: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Return the sources and one evidence record per problem, keyed by ``key``.

    ``forbidden_block_names`` are blocks that must not exist, for a problem the
    spec did not enable.
    """
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

    blocks = _convergence_blocks(asol, problems, forbidden_block_names, solver)
    evidence = {
        problem.key: _problem_evidence(
            solver, blocks[problem.block_name], profile, problem, spec.run_control
        )
        for problem in problems
    }
    return {
        "sources": {
            "asol": source_record(asol_path, root),
            "profile": source_record(profile_path, root),
        },
        **evidence,
    }


def _convergence_blocks(
    asol: str,
    problems: tuple[ConvergenceProblem, ...],
    forbidden_block_names: tuple[str, ...],
    solver: str,
) -> dict[str, _NativeBlock]:
    """Find the one block for each wanted problem; refuse repeats and forbidden ones."""
    block_names = {problem.block_name for problem in problems}
    blocks: dict[str, _NativeBlock] = {}
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
            blocks[name] = _NativeBlock(match.group("id"), body)
    if set(blocks) != block_names:
        raise RuntimeError(
            f"{solver} native solution lacks exact matrix convergence blocks"
        )
    if set(forbidden_block_names) & found_names:
        raise RuntimeError(
            f"{solver} native solution contains a disabled convergence block"
        )
    return blocks


def _problem_evidence(
    solver: str,
    block: _NativeBlock,
    profile: str,
    problem: ConvergenceProblem,
    run_control: Any,
) -> dict[str, Any]:
    """Check one problem's block against the run control and summarise its last pass."""
    body = block.body
    target = _matrix_convergence_float(
        _required(body, r"(?m)^\s*ConvTarget='([^']+)'\s*$")
    )
    maximum_passes = _int(_required(body, r"(?m)^\s*MaxPasses='([^']+)'\s*$"))
    if not math.isclose(
        target, run_control.convergence_percent, rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError(
            f"{solver} native convergence target does not match the spec"
        )
    if maximum_passes != run_control.maximum_passes:
        raise RuntimeError(
            f"{solver} native convergence maximum passes do not match the spec"
        )

    pass_rows = re.findall(r"(?m)^\s*c\((.*)\)\s*$", body)
    if not pass_rows:
        raise RuntimeError(f"{solver} native convergence block has no adaptive passes")
    final = pass_rows[-1]
    final_pass = _int(_field(final, "p"))
    final_triangles = _int(_field(final, "tri"))
    final_delta = _matrix_convergence_float(_field(final, problem.delta_field))
    final_error = (
        _matrix_convergence_float(_field(final, problem.error_field))
        if problem.error_field
        else None
    )
    if (
        final_pass <= 0
        or final_pass > maximum_passes
        or final_triangles <= 0
        or final_delta < 0.0
        or (final_error is not None and final_error < 0.0)
    ):
        raise RuntimeError(f"{solver} native final convergence values are invalid")

    converged, stop_reason = _converged_status(
        _block(profile, block.block_id, solver), solver
    )
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


def _converged_status(profile_block: str, solver: str) -> tuple[bool, str]:
    """The profile states convergence exactly once, in one of two fixed phrases."""
    converged_text = "Adaptive Passes converged"
    not_converged_text = "Adaptive Passes did not converge"
    counts = (
        profile_block.count(converged_text),
        profile_block.count(not_converged_text),
    )
    if counts == (1, 0):
        return True, converged_text
    if counts == (0, 1):
        return False, not_converged_text
    raise RuntimeError(f"{solver} native profile has ambiguous convergence status")


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


def _matrix_convergence_float(value: str) -> float:
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
